#pragma once
#include "types.hpp"
#include "device_data.hpp"

// GPU version: short-range LJ potential energy using half neighbor list (i<j).
// If Params.lj_shift is true, subtract precomputed LJ(rc) per pair.
real compute_lj_sr_energy_gpu(const DeviceAtoms& atoms,
                              const DeviceParams& dparams,
                              const DeviceCSR& neigh,
                              const Box& box);

// GPU version: short-range Coulomb with cutoff (no shift).
// Supports only coulombtype=="coul_cut". Energy per pair:
// E_ij = (ke/epsilon_r) * q_i q_j * (1/r), r <= rc; 0 otherwise.
real compute_coul_sr_energy_gpu(const DeviceAtoms& atoms,
                                const DeviceParams& dparams,
                                const DeviceCSR& neigh,
                                const Box& box);
