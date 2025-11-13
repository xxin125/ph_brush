#pragma once
#include <thrust/device_vector.h>
#include "types.hpp"

struct DeviceAtoms {
    int natoms{0};
    thrust::device_vector<real> x;
    thrust::device_vector<real> y;
    thrust::device_vector<real> z;
    thrust::device_vector<real> q;
    thrust::device_vector<int> type;
};

struct DeviceCSR {
    int rows{0};
    int nnz{0};
    thrust::device_vector<int> head; // size rows+1
    thrust::device_vector<int> list; // size nnz
};

struct DeviceParams {
    int ntypes{0};
    real rc{0.0};
    real epsilon_r{1.0};
    bool lj_shift{true};
    std::string coulombtype;
    real ewald_rtol{real(1e-5)};
    real ewald_alpha{0.0};
    real pme_spacing{0.12};
    int pme_order{4};
    bool pme_3dc_enabled{false};
    real pme_3dc_zfac{3.0};
    thrust::device_vector<real> lj_eps;
    thrust::device_vector<real> lj_sig;
    thrust::device_vector<real> lj_shift_tbl;
};
