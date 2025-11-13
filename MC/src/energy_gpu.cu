#include "energy.hpp"
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>

static inline void cuda_check(cudaError_t err, const char* file, int line){
    if (err != cudaSuccess){ throw std::runtime_error(std::string("CUDA Error: ")+cudaGetErrorString(err)+" at "+file+":"+std::to_string(line)); }
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

__device__ inline real d_min_image(real dx, real L){ dx -= L * nearbyint(dx / L); return dx; }

__global__ void k_lj_energy(const real* __restrict__ x,
                            const real* __restrict__ y,
                            const real* __restrict__ z,
                            const int*    __restrict__ type, // 0-based
                            const int*    __restrict__ head,
                            const int*    __restrict__ list,
                            int N, int T,
                            real Lx,real Ly,real Lz,
                            const real* __restrict__ lj_eps,
                            const real* __restrict__ lj_sig,
                            const real* __restrict__ lj_shift_tbl,
                            int use_shift,
                            real* __restrict__ E_out){
    extern __shared__ real ssum[];
    int tid = threadIdx.x; real e_local = 0.0;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += gridDim.x*blockDim.x){
        real xi=x[i], yi=y[i], zi=z[i]; int ti=type[i]; int beg=head[i], end=head[i+1];
        for (int k=beg;k<end;++k){ int j=list[k];
            real dx = d_min_image(xi - x[j], Lx);
            real dy = d_min_image(yi - y[j], Ly);
            real dz = d_min_image(zi - z[j], Lz);
            real r2 = dx*dx + dy*dy + dz*dz; real r = sqrt(r2);
            int tj=type[j]; real eps = lj_eps[ti*T + tj]; real sig = lj_sig[ti*T + tj]; if (eps<=0.0 || sig<=0.0) continue;
            real sr = sig / r; real sr2=sr*sr; real sr6=sr2*sr2*sr2; real sr12=sr6*sr6;
            real e = real(4.0)*eps*(sr12 - sr6);
            if (use_shift){ e -= lj_shift_tbl[ti*T + tj]; }
            e_local += e;
        }
    }
    ssum[tid] = e_local; __syncthreads();
    for (int s=blockDim.x/2; s>0; s>>=1){ if (tid < s){ ssum[tid] += ssum[tid+s]; } __syncthreads(); }
    if (tid==0) atomicAdd(E_out, ssum[0]);
}

real compute_lj_sr_energy_gpu(const DeviceAtoms& atoms,
                              const DeviceParams& dparams,
                              const DeviceCSR& neigh,
                              const Box& box){
    const int N = atoms.natoms;
    const int T = dparams.ntypes;
    if (N==0 || neigh.nnz==0) return 0.0;
    thrust::device_vector<real> dE(1,0.0);

    int threads = 256; int blocks = std::min((N+threads-1)/threads, 8192);
    size_t shmem = threads * sizeof(real);
    k_lj_energy<<<blocks,threads,shmem>>>(
        thrust::raw_pointer_cast(atoms.x.data()),
        thrust::raw_pointer_cast(atoms.y.data()),
        thrust::raw_pointer_cast(atoms.z.data()),
        thrust::raw_pointer_cast(atoms.type.data()),
        thrust::raw_pointer_cast(neigh.head.data()),
        thrust::raw_pointer_cast(neigh.list.data()),
        N, T, box.Lx, box.Ly, box.Lz,
        thrust::raw_pointer_cast(dparams.lj_eps.data()),
        thrust::raw_pointer_cast(dparams.lj_sig.data()),
        dparams.lj_shift ? thrust::raw_pointer_cast(dparams.lj_shift_tbl.data()) : nullptr,
        dparams.lj_shift ? 1 : 0,
        thrust::raw_pointer_cast(dE.data()));
    CUDA_CHECK(cudaDeviceSynchronize());
    real E = dE[0];
    return E;
}

// Coulomb short-range (cutoff, no shift) GPU implementation
static constexpr real KE_KJ_PER_MOL_NM_E2 = real(138.935458); // GROMACS ke

__global__ void k_coul_energy(const real* __restrict__ x,
                              const real* __restrict__ y,
                              const real* __restrict__ z,
                              const real* __restrict__ q,
                              const int*    __restrict__ head,
                              const int*    __restrict__ list,
                              int N,
                              real Lx,real Ly,real Lz,
                              real ke_over_epsr,
                              real* __restrict__ E_out){
    extern __shared__ real ssum[];
    int tid = threadIdx.x; real e_local = 0.0;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += gridDim.x*blockDim.x){
        real xi=x[i], yi=y[i], zi=z[i]; real qi=q[i]; int beg=head[i], end=head[i+1];
        for (int k=beg;k<end;++k){ int j=list[k];
            real dx = d_min_image(xi - x[j], Lx);
            real dy = d_min_image(yi - y[j], Ly);
            real dz = d_min_image(zi - z[j], Lz);
            real r2 = dx*dx + dy*dy + dz*dz; real r = sqrt(r2);
            real qq = qi * q[j]; if (qq==0.0) continue;
            // pure cutoff (no shift): E = ke/epsr * q_i q_j / r, for r<=rc (enforced by neighbor list)
            real e = ke_over_epsr * qq * (real(1.0)/r);
            e_local += e;
        }
    }
    ssum[tid] = e_local; __syncthreads();
    for (int s=blockDim.x/2; s>0; s>>=1){ if (tid < s){ ssum[tid] += ssum[tid+s]; } __syncthreads(); }
    if (tid==0) atomicAdd(E_out, ssum[0]);
}

real compute_coul_sr_energy_gpu(const DeviceAtoms& atoms,
                                const DeviceParams& dparams,
                                const DeviceCSR& neigh,
                                const Box& box){
    const int N = atoms.natoms;
    if (N==0 || neigh.nnz==0) return 0.0;

    thrust::device_vector<real> dE(1,0.0);
    real ke_over_epsr = KE_KJ_PER_MOL_NM_E2 / dparams.epsilon_r;

    int threads = 256; int blocks = std::min((N+threads-1)/threads, 8192);
    size_t shmem = threads * sizeof(real);
    k_coul_energy<<<blocks,threads,shmem>>>(
        thrust::raw_pointer_cast(atoms.x.data()),
        thrust::raw_pointer_cast(atoms.y.data()),
        thrust::raw_pointer_cast(atoms.z.data()),
        thrust::raw_pointer_cast(atoms.q.data()),
        thrust::raw_pointer_cast(neigh.head.data()),
        thrust::raw_pointer_cast(neigh.list.data()),
        N, box.Lx, box.Ly, box.Lz,
        ke_over_epsr,
        thrust::raw_pointer_cast(dE.data()));
    CUDA_CHECK(cudaDeviceSynchronize());
    real E = dE[0];
    return E;
}
