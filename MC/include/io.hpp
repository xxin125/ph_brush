#pragma once
#include "types.hpp"
#include <string>

// Read LAMMPS data file (full style). Parses box, Atoms # full, and Bonds.
// Builds 1-2 adjacency CSR (per-atom bonded neighbors), sorted.
System read_lammps_full(const std::string& path);
CSR build_bond_adjacency(const System& sys);

// Read params.in: cutoff and LJ table (explicit pairs).
Params read_params(const std::string& path, int ntypes);

// Wrap all atom coordinates into [0,L) in each dimension.
void wrap_positions(System& sys);

// Write a LAMMPS data file (atoms + bonds) using the provided box lengths.
void write_lammps_full(const std::string& path, const System& sys, const Box& boxForOutput);
