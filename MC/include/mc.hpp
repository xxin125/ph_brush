#ifndef MC_HPP
#define MC_HPP

#include <vector>
#include <random>
#include <string>
#include "types.hpp"
#include "device_data.hpp"
#include "pme.hpp"

struct MCPlan {
    std::vector<int> idx_nh_n;
    std::vector<int> idx_nh_p;
    std::vector<int> idx_w;
    int to_flip{0};
    real z_cut{0.0};
};

struct Phase1Result {
    int attempted{0};
    int accepted{0};
    real E{0.0};
    std::vector<int> nh_n_remaining;
    std::vector<int> w_remaining;
    std::vector<int> nh_p_mc;
    std::vector<int> cl_mc;
};

struct Phase2Result {
    int attempted{0};
    int accepted{0};
    real E{0.0};
    real best_E{0.0};
};

struct PhaseGCResult {
    int attempted{0};
    int accepted{0};
    int accepted_forward{0};
    int accepted_backward{0};
    int blocks_done{0};
    int Nh{0};
    int Nprot{0};
    int target_count{0};
    bool reached_target{false};
    real E{0.0};
    real mu_eff{0.0};
};

MCPlan build_plan(const System& sys, const Params& p);

Phase1Result apply_direct_protonation(System& sys,
                                      DeviceAtoms& d_atoms,
                                      const DeviceParams& d_params,
                                      const DeviceCSR& d_neigh,
                                      const Box& box,
                                      PMEPlan* plan,
                                      const Params& p,
                                      const MCPlan& mcplan);

Phase2Result phase2_swap_jitter(System& sys,
                                DeviceAtoms& d_atoms,
                                const DeviceParams& d_params,
                                const DeviceCSR& d_neigh,
                                const Box& box,
                                PMEPlan* plan,
                                const Params& p,
                                const MCPlan& mcplan,
                                const Phase1Result& ph1,
                                std::mt19937& rng);

PhaseGCResult phase_gc_mu_control(System& sys,
                                  DeviceAtoms& d_atoms,
                                  const DeviceParams& d_params,
                                  const DeviceCSR& d_neigh,
                                  const Box& box,
                                  PMEPlan* plan,
                                  Params& p,
                                  const MCPlan& mcplan,
                                  std::mt19937& rng,
                                  Phase1Result* seed_out=nullptr);

#endif // MC_HPP
