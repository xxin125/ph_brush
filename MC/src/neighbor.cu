#include "neighbor.hpp"
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

static inline void cuda_check(cudaError_t err, const char* file, int line){
    if (err != cudaSuccess){ throw std::runtime_error(std::string("CUDA Error: ")+cudaGetErrorString(err)+" at "+file+":"+std::to_string(line)); }
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

__device__ inline real d_min_image(real dx, real L){ dx -= L * nearbyint(dx / L); return dx; }

__device__ inline bool d_is_bonded12(int i, int j, const int* __restrict__ head, const int* __restrict__ list){
    int lo=head[i], hi=head[i+1]-1; while(lo<=hi){ int mid=(lo+hi)>>1; int v=list[mid]; if(v==j) return true; if(v<j) lo=mid+1; else hi=mid-1; } return false;
}

__global__ void k_count_half(const real* __restrict__ x,
                             const real* __restrict__ y,
                             const real* __restrict__ z,
                             int N, real Lx,real Ly,real Lz, real rc2,
                             const int* __restrict__ bhead,
                             const int* __restrict__ blist,
                             int* __restrict__ counts){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=N) return;
    real xi=x[i], yi=y[i], zi=z[i];
    int cnt=0;
    for (int j=i+1;j<N;++j){
        if (d_is_bonded12(i,j,bhead,blist)) continue;
        real dx = d_min_image(xi - x[j], Lx);
        real dy = d_min_image(yi - y[j], Ly);
        real dz = d_min_image(zi - z[j], Lz);
        real r2 = dx*dx + dy*dy + dz*dz;
        if (r2 <= rc2) ++cnt;
    }
    counts[i]=cnt;
}

__global__ void k_fill_half(const real* __restrict__ x,
                            const real* __restrict__ y,
                            const real* __restrict__ z,
                            int N, real Lx,real Ly,real Lz, real rc2,
                            const int* __restrict__ bhead,
                            const int* __restrict__ blist,
                            const int* __restrict__ head,
                            int* __restrict__ list){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=N) return;
    real xi=x[i], yi=y[i], zi=z[i];
    int w=0; int off=head[i];
    for (int j=i+1;j<N;++j){
        if (d_is_bonded12(i,j,bhead,blist)) continue;
        real dx = d_min_image(xi - x[j], Lx);
        real dy = d_min_image(yi - y[j], Ly);
        real dz = d_min_image(zi - z[j], Lz);
        real r2 = dx*dx + dy*dy + dz*dz;
        if (r2 <= rc2){ list[off + w] = j; ++w; }
    }
}

DeviceCSR copy_csr_to_device(const CSR& host){
    DeviceCSR dev;
    dev.rows = host.head.empty() ? 0 : (int)host.head.size() - 1;
    dev.nnz = (int)host.list.size();
    dev.head.assign(host.head.begin(), host.head.end());
    dev.list.assign(host.list.begin(), host.list.end());
    return dev;
}

DeviceCSR build_half_neighbors_gpu(const DeviceAtoms& atoms,
                                   const Box& box,
                                   const DeviceCSR& bonds12_dev,
                                   real rc){
    DeviceCSR out;
    const int N = atoms.natoms;
    out.rows = N;
    if (N<=0){
        out.nnz = 0;
        out.head.assign(1, 0);
        out.list.clear();
        return out;
    }
    const real Lx=box.Lx, Ly=box.Ly, Lz=box.Lz;
    thrust::device_vector<int> dcounts(N, 0);

    auto dx = thrust::raw_pointer_cast(atoms.x.data());
    auto dy = thrust::raw_pointer_cast(atoms.y.data());
    auto dz = thrust::raw_pointer_cast(atoms.z.data());
    auto dbhead = thrust::raw_pointer_cast(bonds12_dev.head.data());
    auto dblist = thrust::raw_pointer_cast(bonds12_dev.list.data());

    int threads=256; int blocks=(N+threads-1)/threads;
    real rc2 = rc*rc;
    k_count_half<<<blocks,threads>>>(
        dx, dy, dz,
        N, Lx,Ly,Lz, rc2,
        dbhead,
        dblist,
        thrust::raw_pointer_cast(dcounts.data()));
    CUDA_CHECK(cudaDeviceSynchronize());

    out.head.resize(N+1);
    thrust::exclusive_scan(dcounts.begin(), dcounts.end(), out.head.begin());
    int last_head = out.head[N-1];
    int last_cnt = dcounts[N-1];
    int total = last_head + last_cnt;
    out.head[N] = total;
    out.nnz = total;
    out.list.resize(total);

    k_fill_half<<<blocks,threads>>>(
        dx, dy, dz,
        N, Lx,Ly,Lz, rc2,
        dbhead,
        dblist,
        thrust::raw_pointer_cast(out.head.data()),
        thrust::raw_pointer_cast(out.list.data()));
    CUDA_CHECK(cudaDeviceSynchronize());

    return out;
}
