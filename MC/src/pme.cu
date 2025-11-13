#include "pme.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

static inline void CUDA_CHECK(cudaError_t e, const char* f, int l){
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA: ")+cudaGetErrorString(e)+" at "+f+":"+std::to_string(l));
}
#define CUDA_OK(x) CUDA_CHECK(x, __FILE__, __LINE__)
static inline void CUFFT_OK(cufftResult r){
    if (r != CUFFT_SUCCESS) throw std::runtime_error("cuFFT error code "+std::to_string(r));
}

#if defined(PME_USE_FLOAT_PRECISION)
static inline void cufft_exec_forward(cufftHandle plan, real* realGrid, PMECufftComplex* kGrid){
    CUFFT_OK(cufftExecR2C(plan,
                          reinterpret_cast<cufftReal*>(realGrid),
                          reinterpret_cast<cufftComplex*>(kGrid)));
}
static constexpr cufftType kPlanType = CUFFT_R2C;
#else
static inline void cufft_exec_forward(cufftHandle plan, real* realGrid, PMECufftComplex* kGrid){
    CUFFT_OK(cufftExecD2Z(plan,
                          reinterpret_cast<cufftDoubleReal*>(realGrid),
                          reinterpret_cast<cufftDoubleComplex*>(kGrid)));
}
static constexpr cufftType kPlanType = CUFFT_D2Z;
#endif

// ---------------- math/helpers ----------------
__device__ inline real d_min_image(real dx, real L){
    dx -= L * nearbyint(dx / L);
    return dx;
}

// choose >=n FFT-friendly (2/3/5-smooth)
static int good_fft_dim(int n){
    const int NMAX = std::max(16, n*2 + 32);
    auto is_good = [](int x){ for (int p: {2,3,5}) while (x%p==0) x/=p; return x==1; };
    for (int m = std::max(8,n); m <= NMAX; ++m) if (is_good(m)) return m;
    return n;
}

// find alpha from rtol at rc: erfc(alpha*rc) <= rtol
static double calc_alpha_from_rtol(double rc, double rtol){
    double beta = 5.0; int i=0;
    while (std::erfc(beta*rc) > rtol){ beta *= 2.0; if (++i>60) break; }
    double low=0.0, high=beta;
    for (int it=0; it<60; ++it){
        beta = 0.5*(low+high);
        if (std::erfc(beta*rc) > rtol) low = beta; else high = beta;
    }
    return beta;
}

// reciprocal box H = inv(box)^T (no 2*pi) and volume (orthorhombic)
static void compute_recip_and_volume(const Box& b, real H[3][3], real* V){
    real vol = b.Lx * b.Ly * b.Lz;
    *V = vol;
    H[0][0]=real(1.0)/b.Lx; H[0][1]=0;       H[0][2]=0;
    H[1][0]=0;       H[1][1]=real(1.0)/b.Ly; H[1][2]=0;
    H[2][0]=0;       H[2][1]=0;       H[2][2]=real(1.0)/b.Lz;
}

// ---------------- SPME 1D modulus |DFT(B-spline)|^2 ----------------
// GMX convention: pme_order counts stencil points, splineOrder = pme_order - 1.
// Build De Boor coefficients data[0..splineOrder-1] and take the DFT magnitude.
static std::vector<real> make_spme_bspline_moduli_1d(int n, int pme_order){
    const int splineOrder = pme_order - 1;
    if (splineOrder < 1) throw std::runtime_error("pme_order must be >= 2");

    // De Boor recursion for cardinal B-spline coefficients
    std::vector<double> data(splineOrder, 0.0);
    data[0] = 1.0;
    for (int k = 2; k <= splineOrder; ++k){
        const double invk = 1.0 / k;
        for (int m = k - 1; m > 0; --m){
            data[m] = invk * ((k - m) * data[m - 1] + (m + 1) * data[m]);
        }
        data[0] = invk * data[0];
    }

    // DFT magnitude squared on length-n grid
    std::vector<real> mod(n, 0.0);
    for (int i = 0; i < n; ++i){
        double sc = 0.0, ss = 0.0;
        for (int j = 0; j < splineOrder; ++j){
            const double arg = (2.0 * M_PI * i * (j + 1)) / double(n);
            sc += data[j] * std::cos(arg);
            ss += data[j] * std::sin(arg);
        }
        double v = sc*sc + ss*ss;
        // clamp to avoid exact zeros
        if (!(v > 0.0)) v = 1e-30;
        mod[i] = static_cast<real>(v);
    }

    // Nyquist fix: if n is even, patch the potential zero to avoid division issues.
    if (n % 2 == 0){
        const int h = n/2;
        if (!(mod[h] > real(0.0))) mod[h] = mod[h-1];
    }
    return mod;
}

struct SquareOp { __host__ __device__ real operator()(real v) const { return v*v; } };

struct ChargeZOp {
    __host__ __device__ real operator()(const thrust::tuple<real,real>& t) const {
        return thrust::get<0>(t) * thrust::get<1>(t);
    }
};

struct ChargeZ2Op {
    __host__ __device__ real operator()(const thrust::tuple<real,real>& t) const {
        real qi = thrust::get<0>(t);
        real zi = thrust::get<1>(t);
        return qi * zi * zi;
    }
};

// ---------------- spread charges (cubic B-spline; x fastest) ----------------
__device__ inline void bspline_cubic(real t, real w[4]){
    real t2 = t*t, t3 = t2*t;
    w[0] = (real(1) - 3*t + 3*t2 - t3) / real(6.0);
    w[1] = (real(4) - 6*t2 + 3*t3) / real(6.0);
    w[2] = (real(1) + 3*t + 3*t2 - 3*t3) / real(6.0);
    w[3] = t3 / real(6.0);
}

__global__ void k_spread_cubic(const real* __restrict__ x,
                               const real* __restrict__ y,
                               const real* __restrict__ z,
                               const real* __restrict__ q,
                               int N, int nx,int ny,int nz,
                               real Lx,real Ly,real Lz,
                               real* __restrict__ grid)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=N) return;

    real gx = (x[i]/Lx)*nx; int ix = floor(gx); real tx = gx - ix;
    real gy = (y[i]/Ly)*ny; int iy = floor(gy); real ty = gy - iy;
    real gz = (z[i]/Lz)*nz; int iz = floor(gz); real tz = gz - iz;

    real wx[4], wy[4], wz[4];
    bspline_cubic(tx,wx); bspline_cubic(ty,wy); bspline_cubic(tz,wz);
    const real qi = q[i];

    // realGrid layout: ((z*ny)+y)*nx + x (x fastest)
    for (int a=0;a<4;++a){
        int ixa = (ix + a - 1 + nx) % nx;
        for (int b=0;b<4;++b){
            int iyb = (iy + b - 1 + ny) % ny;
            real wxy = qi * wx[a]*wy[b];
            for (int c=0;c<4;++c){
                int izc = (iz + c - 1 + nz) % nz;
                size_t off = (size_t(izc)*ny + iyb)*nx + ixa;
                atomicAdd(grid + off, wxy*wz[c]);
            }
        }
    }
}

// ---------------- reciprocal kernel (half-spectrum in x; wrap y/z; GMX/SPME formula) ----------------
__global__ void k_recip_energy_and_apply(real elFactor, real ewaldFactor,
                                         int nx,int ny,int nz,
                                         real Hxx,real Hxy,real Hxz,
                                         real Hyx,real Hyy,real Hyz,
                                         real Hzx,real Hzy,real Hzz,
                                         real boxVol,
                                         const real* __restrict__ modX,
                                         const real* __restrict__ modY,
                                         const real* __restrict__ modZ,
                                         PMECufftComplex* __restrict__ kgrid,
                                         real* __restrict__ E_out)
{
    extern __shared__ real ssum[];
    int tid = threadIdx.x;
    real e_local = 0.0;

    const int nKx = nx/2 + 1;  // half-spectrum in x
    const int nKy = ny;        // full
    const int nKz = nz;        // full
    const size_t total = size_t(nKx) * nKy * nKz;

    for (size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
         idx < total; idx += gridDim.x*blockDim.x)
    {
        // linear -> (kz, ky, kx)
        int kz = idx / (size_t(nKy) * nKx);
        size_t rem = idx - size_t(kz) * size_t(nKy) * size_t(nKx);
        int ky = rem / nKx;
        int kx = rem % nKx;

        if (kx==0 && ky==0 && kz==0) continue; // skip origin

        // wrap-around for full-spectrum dims (y,z)
        int mx = kx;                          // 0..nx/2
        int my = (ky <= nKy/2) ? ky : ky - nKy;
        int mz = (kz <= nKz/2) ? kz : kz - nKz;

        // k = H^T m  (H has no 2*pi)
        real kxv = mx*Hxx + my*Hyx + mz*Hzx;
        real kyv = mx*Hxy + my*Hyy + mz*Hzy;
        real kzv = mx*Hxz + my*Hyz + mz*Hzz;
        real m2k = kxv*kxv + kyv*kyv + kzv*kzv;
        if (m2k == real(0.0)) continue;

        // denom = m2k * pi * V * Bx * By * Bz   (SPME/GMX units absorb 4*pi into ke)
        real denom = m2k * real(M_PI) * boxVol * modX[kx] * modY[ky] * modZ[kz];
        if (!(denom > real(0.0))) continue; // safety

        real eterm = elFactor * exp(-ewaldFactor * m2k) / denom;

        // half-complex layout: ((z*ny)+y)*(nx/2+1) + kx
        size_t off = (size_t(kz)*nKy + ky)*nKx + kx;
        PMECufftComplex oldv = kgrid[off];
        PMECufftComplex newv;
        newv.x = oldv.x * eterm;
        newv.y = oldv.y * eterm;
        kgrid[off] = newv;

        // corner only on half-spectrum dim x at 0 or Nx/2
        real corner = (kx==0 || kx==nx/2) ? real(0.5) : real(1.0);

        // accumulate energy: corner * 2 * Re(new * conj(old))
        real tmp1k = real(2.0) * (newv.x*oldv.x + newv.y*oldv.y);
        e_local += corner * tmp1k;
    }

    // block reduction
    ssum[tid] = e_local;
    __syncthreads();
    for (int s=blockDim.x/2; s>0; s>>=1){
        if (tid < s) ssum[tid] += ssum[tid+s];
        __syncthreads();
    }
    if (tid==0) atomicAdd(E_out, ssum[0]);
}

// ---------------- real-space Ewald energy (i<j half-neighbor list) ----------------
__global__ void k_real_ewald_energy(const real* __restrict__ x,
                                    const real* __restrict__ y,
                                    const real* __restrict__ z,
                                    const real* __restrict__ q,
                                    const int*    __restrict__ head,
                                    const int*    __restrict__ list,
                                    int N, real Lx,real Ly,real Lz,
                                    real ke_over_epsr, real alpha,
                                    real* __restrict__ E_out)
{
    extern __shared__ real ssum[];
    int tid = threadIdx.x; real e_local = 0.0;

    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += gridDim.x*blockDim.x){
        real xi=x[i], yi=y[i], zi=z[i]; real qi=q[i];
        int beg=head[i], end=head[i+1];
        for (int p=beg; p<end; ++p){
            int j = list[p];
            real dx = d_min_image(xi - x[j], Lx);
            real dy = d_min_image(yi - y[j], Ly);
            real dz = d_min_image(zi - z[j], Lz);
            real r2 = dx*dx + dy*dy + dz*dz;
            real r  = sqrt(r2);
            real qq = qi * q[j];
            if (qq==real(0.0)) continue;
            real val = erfc(alpha*r) / r;
            e_local += ke_over_epsr * qq * val;
        }
    }
    ssum[tid] = e_local;
    __syncthreads();
    for (int s=blockDim.x/2; s>0; s>>=1){
        if (tid < s) ssum[tid] += ssum[tid+s];
        __syncthreads();
    }
    if (tid==0) atomicAdd(E_out, ssum[0]);
}

// ---------------- API ----------------
void pmeCreate(PMEPlan* plan, const DeviceParams& params, const Box& box, real rc){
    if (!plan) throw std::runtime_error("pmeCreate: null plan");

    // GMX convention reminder for consistency checks
    plan->order = params.pme_order;
    plan->use3dc = params.pme_3dc_enabled;

    // alpha & ewaldFactor
    plan->alpha = (params.ewald_alpha>0 ? params.ewald_alpha
                                        : static_cast<real>(calc_alpha_from_rtol(static_cast<double>(rc),
                                                                                static_cast<double>(params.ewald_rtol))));
    plan->ewaldFactor = (real(M_PI)/plan->alpha) * (real(M_PI)/plan->alpha);

    // ke/epsilon_r
    const real ke = real(138.935458);   // kJ mol^-1 nm e^-2
    plan->elFactor = ke / params.epsilon_r;

    // box vol & reciprocal
    compute_recip_and_volume(box, plan->recipBox, &plan->vol);

    // grid dims from spacing (>= order) and FFT-friendly
    int nx = good_fft_dim((int)std::ceil(box.Lx / params.pme_spacing));
    int ny = good_fft_dim((int)std::ceil(box.Ly / params.pme_spacing));
    int nz = good_fft_dim((int)std::ceil(box.Lz / params.pme_spacing));
    nx = std::max(nx, plan->order);
    ny = std::max(ny, plan->order);
    nz = std::max(nz, plan->order);
    plan->nx = nx; plan->ny = ny; plan->nz = nz;

    // 1D SPME moduli (host -> device, full dimension length)
    auto mx = make_spme_bspline_moduli_1d(nx, plan->order);
    auto my = make_spme_bspline_moduli_1d(ny, plan->order);
    auto mz = make_spme_bspline_moduli_1d(nz, plan->order);
    plan->modX = thrust::device_vector<real>(mx.begin(), mx.end());
    plan->modY = thrust::device_vector<real>(my.begin(), my.end());
    plan->modZ = thrust::device_vector<real>(mz.begin(), mz.end());

    // grids & plan (R2C: half-spectrum on last dim x)
    plan->ngridReal = nx*ny*nz;
    plan->ngridK    = (nx/2 + 1)*ny*nz;
    plan->realGrid.resize(plan->ngridReal);
    plan->kGrid.resize(plan->ngridK);

    CUFFT_OK(cufftPlan3d(&plan->planR2C, nz, ny, nx, kPlanType)); // dims order (z,y,x); x is fastest
}

void pmeDestroy(PMEPlan* plan){
    if (!plan) return;
    if (plan->planR2C) { cufftDestroy(plan->planR2C); plan->planR2C = 0; }
    plan->realGrid.clear(); plan->kGrid.clear();
    plan->modX.clear(); plan->modY.clear(); plan->modZ.clear();
}

real pmeEnergy(PMEPlan& plan,
               const DeviceAtoms& atoms,
               const DeviceCSR& neigh,
               const Box& box,
               PMEEnergyComponents* components)
{
    const int N = atoms.natoms;
    if (N==0) {
        if (components){
            components->real_space = 0.0;
            components->recip_space = 0.0;
            components->self_term = 0.0;
            components->qcorr_term = 0.0;
        }
        return 0.0;
    }

    // clear real grid & energy buffer
    CUDA_OK(cudaMemset((void*)thrust::raw_pointer_cast(plan.realGrid.data()), 0, sizeof(real)*plan.ngridReal));
    thrust::device_vector<real> dE(1); dE[0]=0.0;

    // 1) real-space Ewald energy
    {
        int threads = 256, blocks = (N + threads - 1)/threads;
        size_t shmem = threads*sizeof(real);
        k_real_ewald_energy<<<blocks, threads, shmem>>>(
            thrust::raw_pointer_cast(atoms.x.data()),
            thrust::raw_pointer_cast(atoms.y.data()),
            thrust::raw_pointer_cast(atoms.z.data()),
            thrust::raw_pointer_cast(atoms.q.data()),
            thrust::raw_pointer_cast(neigh.head.data()),
            thrust::raw_pointer_cast(neigh.list.data()),
            N, box.Lx, box.Ly, box.Lz,
            plan.elFactor, plan.alpha,
            thrust::raw_pointer_cast(dE.data()));
        CUDA_OK(cudaDeviceSynchronize());
    }
    real E_real = dE[0]; dE[0]=0.0;

    // 2) spread charges to real grid (cubic, x-fastest layout)
    {
        int threads = 256, blocks = (N + threads - 1)/threads;
        k_spread_cubic<<<blocks, threads>>>(
            thrust::raw_pointer_cast(atoms.x.data()),
            thrust::raw_pointer_cast(atoms.y.data()),
            thrust::raw_pointer_cast(atoms.z.data()),
            thrust::raw_pointer_cast(atoms.q.data()),
            N, plan.nx, plan.ny, plan.nz,
            box.Lx, box.Ly, box.Lz,
            thrust::raw_pointer_cast(plan.realGrid.data()));
        CUDA_OK(cudaDeviceSynchronize());
    }

    // 3) forward FFT: real -> k
    cufft_exec_forward(plan.planR2C,
        thrust::raw_pointer_cast(plan.realGrid.data()),
        thrust::raw_pointer_cast(plan.kGrid.data()));

    // 4) reciprocal space apply & accumulate energy (SPME / GMX formula)
    {
        const size_t total = size_t(plan.nx/2 + 1) * plan.ny * plan.nz;
        int threads = 256;
        int blocks  = (int)std::min<size_t>((total + threads - 1)/threads, 32768);
        size_t shmem = threads*sizeof(real);
        k_recip_energy_and_apply<<<blocks, threads, shmem>>>(
            plan.elFactor, plan.ewaldFactor,
            plan.nx, plan.ny, plan.nz,
            plan.recipBox[0][0], plan.recipBox[0][1], plan.recipBox[0][2],
            plan.recipBox[1][0], plan.recipBox[1][1], plan.recipBox[1][2],
            plan.recipBox[2][0], plan.recipBox[2][1], plan.recipBox[2][2],
            plan.vol,
            thrust::raw_pointer_cast(plan.modX.data()),
            thrust::raw_pointer_cast(plan.modY.data()),
            thrust::raw_pointer_cast(plan.modZ.data()),
            thrust::raw_pointer_cast(plan.kGrid.data()),
            thrust::raw_pointer_cast(dE.data()));
        CUDA_OK(cudaDeviceSynchronize());
    }
    real E_recip = real(0.5) * dE[0]; // kernel used 2*Re(...); multiply 0.5 here
    dE[0]=0.0;

    // 5) self & net-charge corrections (host)
    const real ke_over_eps = plan.elFactor;
    real Q  = thrust::reduce(atoms.q.begin(), atoms.q.end(), real(0.0), thrust::plus<real>());
    real Q2 = thrust::transform_reduce(atoms.q.begin(), atoms.q.end(), SquareOp{}, real(0.0), thrust::plus<real>());

    real E_self  = - ke_over_eps * plan.alpha / std::sqrt(M_PI) * Q2;
    real E_qcorr = - (real(M_PI) * ke_over_eps) * (Q*Q) / (real(2.0) * plan.vol * plan.alpha * plan.alpha);
    real E_slab  = 0.0;

    if (plan.use3dc){
        const real dipole_coeff = (real(2.0) * real(M_PI) / plan.vol) * plan.elFactor;
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(atoms.q.begin(), atoms.z.begin()));
        auto end   = thrust::make_zip_iterator(thrust::make_tuple(atoms.q.end(),   atoms.z.end()));
        real Mz = thrust::transform_reduce(begin, end, ChargeZOp{}, real(0.0), thrust::plus<real>());
        E_slab = dipole_coeff * Mz * Mz;

        const real charge_tol = real(1e-4);
        if (std::abs(Q) > charge_tol){
            real sumQZ2 = thrust::transform_reduce(begin, end, ChargeZ2Op{}, real(0.0), thrust::plus<real>());
            real Lz = box.Lz;
            E_slab -= dipole_coeff * Q * (sumQZ2 + Q * (Lz * Lz) / real(12.0));
        }
    }

    if (components){
        components->real_space  = E_real;
        components->recip_space = E_recip;
        components->self_term   = E_self;
        components->qcorr_term  = E_qcorr + E_slab;
    }

    return E_real + E_recip + E_self + E_qcorr + E_slab;
}
