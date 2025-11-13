#pragma once
#include <cufft.h>
#include <thrust/device_vector.h>
#include "types.hpp"
#include "device_data.hpp"

struct PMEEnergyComponents {
    real real_space{0.0};
    real recip_space{0.0};
    real self_term{0.0};
    real qcorr_term{0.0};
};

#if defined(PME_USE_FLOAT_PRECISION)
using PMECufftReal = cufftReal;
using PMECufftComplex = cufftComplex;
#else
using PMECufftReal = cufftDoubleReal;
using PMECufftComplex = cufftDoubleComplex;
#endif

// Lightweight plan that owns grid buffers, spline moduli, and cuFFT handles.
struct PMEPlan {
    int nx{0}, ny{0}, nz{0};
    int order{4};
    real alpha{0.0};
    real ewaldFactor{0.0};
    real elFactor{0.0};
    real vol{0.0};
    real recipBox[3][3]{};
    bool use3dc{false};

    thrust::device_vector<real> modX, modY, modZ;
    thrust::device_vector<real> realGrid;
    thrust::device_vector<PMECufftComplex> kGrid;

    cufftHandle planR2C{0};
    int ngridReal{0};
    int ngridK{0};
};

void pmeCreate(PMEPlan* plan, const DeviceParams& params, const Box& box, real rc);
void pmeDestroy(PMEPlan* plan);

real pmeEnergy(PMEPlan& plan,
               const DeviceAtoms& atoms,
               const DeviceCSR& neigh,
               const Box& box,
               PMEEnergyComponents* components = nullptr);
