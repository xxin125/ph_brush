#pragma once
#include <vector>
#include <string>
#include "precision.hpp"

struct Box {
    real Lx{0}, Ly{0}, Lz{0};
};

struct Atom {
    int id{0};
    int mol{0};
    int type{0};
    real q{0.0};
    real x{0}, y{0}, z{0};
};

struct Bond {
    int id{0};
    int type{0};
    int i{0}; // 0-based atom index
    int j{0}; // 0-based atom index
};

struct System {
    Box box;
    std::vector<Atom> atoms;
    std::vector<Bond> bonds;
    int natoms{0};
    int nbonds{0};
    int ntypes{0};
};

struct Params {
    real rc{0.0};
    int ntypes{0};
    std::vector<real> lj_eps; // size ntypes*ntypes
    std::vector<real> lj_sig; // size ntypes*ntypes
    bool lj_shift{true};        // potential-shift to zero at rc
    std::vector<real> lj_shift_tbl; // precomputed LJ(rc) per pair, size ntypes*ntypes
    // Coulomb
    std::string coulombtype{"coul_cut"}; // "coul_cut" or "pme"
    real epsilon_r{1.0};      // relative dielectric constant
    real ewald_rtol{real(1e-5)};
    real ewald_alpha{0.0};
    real pme_spacing{0.12};
    int pme_order{4};
    bool pme_3dc_enabled{false};
    real pme_3dc_zfac{3.0};

    int NH_N_type{4};
    int NH_P_type{5};
    int W_type{7};
    int Cl_type{8};
    real target_protonation_pct{10.0};
    int rng_seed{2025};
    int energy_interval{100};
    int patience{3000};
    int max_attempts{1000000};
    real beta{real(1.0 / (0.008314462 * 300.0))};
    real w_z_extra_nm{1.0};
    int direct_pct_threshold{90};
    real mu_eff{0.0};
    real mu_eta{0.3};
    int gc_block_fixed{2000};
    int sgmc_epoch_attempt_max{100000};
    real sgmc_nacc_frac_round{real(0.10)};
    bool phase2_wc{true};
};

struct CSR {
    std::vector<int> head; // size N+1
    std::vector<int> list; // size head[N]
};
