#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "mc.hpp"
#include "energy.hpp"
#include "report.hpp"

static inline real clamp_real(real x, real lo, real hi){
    return (x < lo) ? lo : (x > hi ? hi : x);
}

template <typename T>
static void write_device_value(thrust::device_vector<T>& vec, int idx, T value){
    if (idx < 0 || idx >= static_cast<int>(vec.size())){
        throw std::runtime_error("write_device_value: index out of bounds");
    }
    T* dst = thrust::raw_pointer_cast(vec.data()) + idx;
    cudaError_t err = cudaMemcpy(dst, &value, sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        throw std::runtime_error(std::string("cudaMemcpy failed: ") + cudaGetErrorString(err));
    }
}

struct AtomState { int type; real q; };

static void set_atom_identity(System& sys, DeviceAtoms& d_atoms, int i, int newType_1based, real newQ, int ntypes){
    if (i < 0 || i >= sys.natoms) throw std::runtime_error("set_atom_identity: atom index out of range");
    if (newType_1based < 1 || newType_1based > ntypes) throw std::runtime_error("set_atom_identity: type out of range");
    sys.atoms[i].type = newType_1based;
    sys.atoms[i].q    = newQ;
    write_device_value(d_atoms.q, i, newQ);
    int devType = newType_1based - 1;
    write_device_value(d_atoms.type, i, devType);
}

static inline real total_energy(const DeviceAtoms& d_atoms,
                                const DeviceParams& d_params,
                                const DeviceCSR& d_neigh,
                                const Box& box,
                                PMEPlan* plan,
                                PMEEnergyComponents* comps=nullptr){
    real E_lj = compute_lj_sr_energy_gpu(d_atoms, d_params, d_neigh, box);
    real E_c = 0.0;
    if (d_params.coulombtype == "coul_cut"){
        E_c = compute_coul_sr_energy_gpu(d_atoms, d_params, d_neigh, box);
    } else if (d_params.coulombtype == "pme"){
        if (!plan) throw std::runtime_error("total_energy: PME plan is null");
        E_c = pmeEnergy(*plan, d_atoms, d_neigh, box, comps);
    } else {
        throw std::runtime_error("total_energy: unsupported coulombtype");
    }
    return E_lj + E_c;
}

static inline int choose_one(std::mt19937& rng, const std::vector<int>& pool){
    std::uniform_int_distribution<int> uni(0, static_cast<int>(pool.size()) - 1);
    return pool[uni(rng)];
}

MCPlan build_plan(const System& sys, const Params& p){
    MCPlan out;
    const int N = sys.natoms;
    out.idx_nh_n.reserve(N);
    out.idx_nh_p.reserve(N);
    out.idx_w.reserve(N);
    int cl_total = 0;

    for (int i = 0; i < N; ++i){
        int t = sys.atoms[i].type;
        if      (t == p.NH_N_type) out.idx_nh_n.push_back(i);
        else if (t == p.NH_P_type) out.idx_nh_p.push_back(i);
        else if (t == p.W_type)    out.idx_w.push_back(i);
        else if (t == p.Cl_type)   ++cl_total;
    }

    int nh_total = static_cast<int>(out.idx_nh_n.size() + out.idx_nh_p.size());
    if (nh_total == 0) throw std::runtime_error("[plan] No NH sites found.");

    int target_count = static_cast<int>(std::floor(p.target_protonation_pct / 100.0 * nh_total));
    int current_prot = static_cast<int>(out.idx_nh_p.size());
    out.to_flip = target_count - current_prot;
    if (out.to_flip < 0) throw std::runtime_error("[plan] Target < current; de-protonation not supported.");
    if (static_cast<int>(out.idx_nh_n.size()) < out.to_flip)
        throw std::runtime_error("[plan] Not enough NH_N to flip.");

    real z_max_site = 0.0;
    for (int i: out.idx_nh_n) z_max_site = std::max(z_max_site, sys.atoms[i].z);
    for (int i: out.idx_nh_p) z_max_site = std::max(z_max_site, sys.atoms[i].z);
    out.z_cut = z_max_site + p.w_z_extra_nm;

    std::vector<int> w_filtered; w_filtered.reserve(out.idx_w.size());
    for (int j: out.idx_w){ if (sys.atoms[j].z < out.z_cut) w_filtered.push_back(j); }
    out.idx_w.swap(w_filtered);

    if (static_cast<int>(out.idx_w.size()) < out.to_flip){
        std::ostringstream oss; oss << "[plan] Not enough W below z<" << std::setprecision(3) << out.z_cut
                                    << " nm: need " << out.to_flip << ", have " << out.idx_w.size();
        throw std::runtime_error(oss.str());
    }

    mc_report() << "\n--- Plan summary ---\n";
    mc_report() << "[NH] total=" << nh_total
              << ", NH_N=" << out.idx_nh_n.size()
              << ", NH_P=" << out.idx_nh_p.size() << "\n";
    mc_report() << std::fixed << std::setprecision(3)
              << "[W/Cl] W_total(z-filtered)=" << out.idx_w.size()
              << ", Cl_total=" << cl_total
              << ", z_max_site=" << z_max_site
              << ", z_cut=" << out.z_cut << " nm\n";
    mc_report() << std::setprecision(2)
              << "[target] pct=" << p.target_protonation_pct
              << ", to_flip=" << out.to_flip
              << " remaining=" << std::max(0, out.to_flip) << "\n\n"<< std::endl;
    return out;
}

Phase1Result apply_direct_protonation(System& sys,
                                      DeviceAtoms& d_atoms,
                                      const DeviceParams& d_params,
                                      const DeviceCSR& d_neigh,
                                      const Box& box,
                                      PMEPlan* plan,
                                      const Params& p,
                                      const MCPlan& mcplan){
    Phase1Result R; R.nh_n_remaining = mcplan.idx_nh_n; R.w_remaining = mcplan.idx_w;

    int need = mcplan.to_flip;
    if (need <= 0){ R.E = total_energy(d_atoms, d_params, d_neigh, box, plan); return R; }

    if ((int)R.nh_n_remaining.size() < need || (int)R.w_remaining.size() < need)
        throw std::runtime_error("[direct] pools smaller than to_flip");

    for (int k=0;k<need;++k){
        int i_nh = R.nh_n_remaining[k];
        int j_w  = R.w_remaining[k];
        set_atom_identity(sys, d_atoms, i_nh, p.NH_P_type, +1.0, d_params.ntypes);
        set_atom_identity(sys, d_atoms, j_w,  p.Cl_type,   -1.0, d_params.ntypes);
        R.nh_p_mc.push_back(i_nh);
        R.cl_mc.push_back(j_w);
    }
    R.nh_n_remaining.erase(R.nh_n_remaining.begin(), R.nh_n_remaining.begin()+need);
    R.w_remaining.erase(R.w_remaining.begin(), R.w_remaining.begin()+need);

    R.attempted = need; R.accepted = need;
    R.E = total_energy(d_atoms, d_params, d_neigh, box, plan);
    mc_report() << std::setprecision(6)
              << "[direct] done: applied direct flips=" << need << " E=" << R.E
              << " remaining=" << std::max(0, need - R.accepted) << std::endl;
    return R;
}

Phase2Result phase2_swap_jitter(System& sys,
                                DeviceAtoms& d_atoms,
                                const DeviceParams& d_params,
                                const DeviceCSR& d_neigh,
                                const Box& box,
                                PMEPlan* plan,
                                const Params& p,
                                const MCPlan& mcplan,
                                const Phase1Result& ph1,
                                std::mt19937& rng){
    Phase2Result R;
    R.E = total_energy(d_atoms, d_params, d_neigh, box, plan);
    R.best_E = R.E;

    std::vector<int> nh_p    = ph1.nh_p_mc;
    std::vector<int> nh_n    = ph1.nh_n_remaining;
    std::vector<int> w_cand  = ph1.w_remaining;
    std::vector<int> cl_pool = ph1.cl_mc;

    auto filter_under_z = [&](std::vector<int>& pool){
        pool.erase(std::remove_if(pool.begin(), pool.end(), [&](int idx){
            return idx < 0 || idx >= sys.natoms || sys.atoms[idx].z >= mcplan.z_cut;
        }), pool.end());
    };
    filter_under_z(w_cand);
    filter_under_z(cl_pool);

    mc_report() << "[phase2] pools: NH_P=" << nh_p.size()
                << " NH_N=" << nh_n.size()
                << " Cl="   << cl_pool.size()
                << " W="    << w_cand.size()
                << " (z_cut=" << mcplan.z_cut << ")\n";

    auto can_four = [&](){
        return !nh_p.empty() && !nh_n.empty() && !w_cand.empty() && !cl_pool.empty();
    };
    auto can_wc = [&](){
        return !w_cand.empty() && !cl_pool.empty();
    };

    bool prefer_wc = nh_n.empty();
    if (!prefer_wc && !can_four() && can_wc()) prefer_wc = true;
    if (!prefer_wc && !can_four() && !can_wc()){
        mc_report() << "[phase2] no valid proposals (no 4-pt and no wc-only). Stop.\n";
        return R;
    }
    if (prefer_wc && !can_wc()){
        mc_report() << "[phase2] wc-only requested but pools empty; Stop.\n";
        return R;
    }

    std::uniform_real_distribution<real> uni(0.0, 1.0);
    const real best_eps = static_cast<real>(1e-9);
    int since_best = 0;

    for (int step = 0; step < p.max_attempts; ++step){
        bool mode_four = (!prefer_wc) && can_four();
        bool mode_wc   = prefer_wc || (!mode_four && can_wc());

        if (!mode_four && !mode_wc){
            mc_report() << "[phase2] no valid proposals; Stop.\n";
            break;
        }

        ++R.attempted;

        if (mode_four){
            int iP  = choose_one(rng, nh_p);
            int iN  = choose_one(rng, nh_n);
            int jCl = choose_one(rng, cl_pool);
            int jW  = choose_one(rng, w_cand);

            AtomState oP{sys.atoms[iP].type,  sys.atoms[iP].q};
            AtomState oN{sys.atoms[iN].type,  sys.atoms[iN].q};
            AtomState oC{sys.atoms[jCl].type, sys.atoms[jCl].q};
            AtomState oW{sys.atoms[jW].type,  sys.atoms[jW].q};

            set_atom_identity(sys, d_atoms, iP,  p.NH_N_type, 0.0,  d_params.ntypes);
            set_atom_identity(sys, d_atoms, iN,  p.NH_P_type, +1.0, d_params.ntypes);
            set_atom_identity(sys, d_atoms, jCl, p.W_type,    0.0,  d_params.ntypes);
            set_atom_identity(sys, d_atoms, jW,  p.Cl_type,  -1.0,  d_params.ntypes);

            real E_trial = total_energy(d_atoms, d_params, d_neigh, box, plan);
            real dE = E_trial - R.E;
            real expo = -p.beta * dE;
            real clamped = clamp_real(expo, static_cast<real>(-40.0), static_cast<real>(40.0));
            bool accept = (dE <= static_cast<real>(0.0)) || (std::exp(clamped) > uni(rng));

            if (accept){
                ++R.accepted;
                R.E = E_trial;
                if (R.E < R.best_E - best_eps) { R.best_E = R.E; since_best = 0; }
                else                           { ++since_best; }

                nh_p.erase(std::find(nh_p.begin(), nh_p.end(), iP)); nh_n.push_back(iP);
                nh_n.erase(std::find(nh_n.begin(), nh_n.end(), iN)); nh_p.push_back(iN);

                cl_pool.erase(std::find(cl_pool.begin(), cl_pool.end(), jCl)); w_cand.push_back(jCl);
                w_cand.erase(std::find(w_cand.begin(), w_cand.end(), jW));     cl_pool.push_back(jW);
            } else {
                set_atom_identity(sys, d_atoms, iP,  oP.type, oP.q, d_params.ntypes);
                set_atom_identity(sys, d_atoms, iN,  oN.type, oN.q, d_params.ntypes);
                set_atom_identity(sys, d_atoms, jCl, oC.type, oC.q, d_params.ntypes);
                set_atom_identity(sys, d_atoms, jW,  oW.type, oW.q, d_params.ntypes);
                ++since_best;
            }
        } else {
            int jCl = choose_one(rng, cl_pool);
            int jW  = choose_one(rng, w_cand);

            AtomState oC{sys.atoms[jCl].type, sys.atoms[jCl].q};
            AtomState oW{sys.atoms[jW].type,  sys.atoms[jW].q};

            set_atom_identity(sys, d_atoms, jCl, p.W_type,  0.0,  d_params.ntypes);
            set_atom_identity(sys, d_atoms, jW,  p.Cl_type, -1.0, d_params.ntypes);

            real E_trial = total_energy(d_atoms, d_params, d_neigh, box, plan);
            real dE = E_trial - R.E;
            real expo = -p.beta * dE;
            real clamped = clamp_real(expo, static_cast<real>(-40.0), static_cast<real>(40.0));
            bool accept = (dE <= static_cast<real>(0.0)) || (std::exp(clamped) > uni(rng));

            if (accept){
                ++R.accepted;
                R.E = E_trial;
                if (R.E < R.best_E - best_eps) { R.best_E = R.E; since_best = 0; }
                else                           { ++since_best; }

                cl_pool.erase(std::find(cl_pool.begin(), cl_pool.end(), jCl)); w_cand.push_back(jCl);
                w_cand.erase(std::find(w_cand.begin(), w_cand.end(), jW));     cl_pool.push_back(jW);
            } else {
                set_atom_identity(sys, d_atoms, jCl, oC.type, oC.q, d_params.ntypes);
                set_atom_identity(sys, d_atoms, jW,  oW.type, oW.q, d_params.ntypes);
                ++since_best;
            }
        }

        if (R.attempted % std::max(1, p.energy_interval) == 0){
            bool log_wc = (!mode_four && mode_wc);
            real acc = (R.attempted > 0)
                ? static_cast<real>(R.accepted) / static_cast<real>(R.attempted)
                : static_cast<real>(0.0);
            mc_report() << std::setprecision(6)
                        << "[phase2] step=" << R.attempted
                        << " E=" << R.E
                        << " accepted=" << R.accepted
                        << " acc=" << std::setprecision(3) << acc
                        << " since_best=" << since_best << "/" << p.patience
                        << (log_wc ? " mode=wc-only\n" : " mode=4-pt\n");
        }

        if (since_best >= p.patience){
            mc_report() << "[phase2] patience reached: "
                        << since_best << "/" << p.patience << ". Stop.\n";
            break;
        }
    }

    return R;
}

PhaseGCResult phase_gc_mu_control(System& sys,
                                  DeviceAtoms& d_atoms,
                                  const DeviceParams& d_params,
                                  const DeviceCSR& d_neigh,
                                  const Box& box,
                                  PMEPlan* plan,
                                  Params& p,
                                  const MCPlan& mcplan,
                                  std::mt19937& rng,
                                  Phase1Result* seed_out){
    PhaseGCResult R{};

    std::vector<int> nh_n = mcplan.idx_nh_n;
    std::vector<int> nh_p = mcplan.idx_nh_p;
    std::vector<int> w    = mcplan.idx_w;
    std::vector<int> cl;
    cl.reserve(sys.natoms);
    for (int j = 0; j < sys.natoms; ++j){
        if (sys.atoms[j].type == p.Cl_type && sys.atoms[j].z < mcplan.z_cut){
            cl.push_back(j);
        }
    }

    R.Nh = static_cast<int>(nh_n.size() + nh_p.size());
    R.Nprot = static_cast<int>(nh_p.size());
    R.target_count = static_cast<int>(std::floor(p.target_protonation_pct / 100.0 * R.Nh));
    R.E = total_energy(d_atoms, d_params, d_neigh, box, plan);

    real fstar = clamp_real(static_cast<real>(p.target_protonation_pct / 100.0), static_cast<real>(1e-6), static_cast<real>(1.0 - 1e-6));
    p.mu_eff = (real(1.0) / p.beta) * std::log(fstar / (static_cast<real>(1.0) - fstar));
    real mu_bound = static_cast<real>(10.0) / p.beta;
    p.mu_eff = clamp_real(p.mu_eff, -mu_bound, mu_bound);

    if (R.Nprot == R.target_count){
        R.reached_target = true;
        R.mu_eff = p.mu_eff;
        mc_report() << "[sgmc] already at target: Nprot=" << R.Nprot << "/" << R.Nh
                    << " target=" << R.target_count << " mu/kT=" << p.beta * p.mu_eff
                    << " E=" << R.E << " remaining=0" << std::endl;
        return R;
    }

    const int gc_block = (p.gc_block_fixed > 0 ? p.gc_block_fixed : 2000);
    const int Amax = (p.sgmc_epoch_attempt_max > 0 ? p.sgmc_epoch_attempt_max : 100000);
    const int R0 = std::max(0, R.target_count - R.Nprot);
    const real frac = (p.sgmc_nacc_frac_round > static_cast<real>(0.0)) ? p.sgmc_nacc_frac_round : static_cast<real>(0.10);
    const int Nacc_target = std::max(1, static_cast<int>(std::ceil(static_cast<double>(frac) * static_cast<double>(R0))));

    std::uniform_real_distribution<real> uni(0.0, 1.0);
    std::uniform_real_distribution<real> u01(0.0, 1.0);

    for (int epoch = 0;; ++epoch){
        int acc_since_mu = 0;
        int att_since_mu = 0;

        while (acc_since_mu < Nacc_target && att_since_mu < Amax){
            int block_attempted = 0;
            int block_accepted = 0;

            for (int t = 0; t < gc_block; ++t){
                long long a = static_cast<long long>(nh_n.size());
                long long b = static_cast<long long>(nh_p.size());
                long long c = static_cast<long long>(w.size());
                long long d = static_cast<long long>(cl.size());
                long long w_fwd = a * c;
                long long w_bwd = b * d;
                long long S_before = a * c + b * d;
                bool can_fwd = (w_fwd > 0);
                bool can_bwd = (w_bwd > 0);

                bool forward;
                if (can_fwd && can_bwd){
                    real sum = static_cast<real>(w_fwd + w_bwd);
                    forward = (u01(rng) < static_cast<real>(w_fwd) / sum);
                } else if (can_fwd){
                    forward = true;
                } else if (can_bwd){
                    forward = false;
                } else {
                    mc_report() << "[sgmc] no valid pairs to flip; stop." << std::endl;
                    R.mu_eff = p.mu_eff;
                    return R;
                }

                int i = -1;
                int j = -1;
                if (forward){
                    i = choose_one(rng, nh_n);
                    j = choose_one(rng, w);
                } else {
                    i = choose_one(rng, nh_p);
                    j = choose_one(rng, cl);
                }

                AtomState oi{sys.atoms[i].type, sys.atoms[i].q};
                AtomState oj{sys.atoms[j].type, sys.atoms[j].q};

                if (forward){
                    set_atom_identity(sys, d_atoms, i, p.NH_P_type, +1.0, d_params.ntypes);
                    set_atom_identity(sys, d_atoms, j, p.Cl_type, -1.0, d_params.ntypes);
                } else {
                    set_atom_identity(sys, d_atoms, i, p.NH_N_type, 0.0, d_params.ntypes);
                    set_atom_identity(sys, d_atoms, j, p.W_type, 0.0, d_params.ntypes);
                }

                real E_trial = total_energy(d_atoms, d_params, d_neigh, box, plan);
                real dE = E_trial - R.E;
                int s = forward ? +1 : -1;
                long long S_after;
                if (forward){
                    S_after = (a - 1) * (c - 1) + (b + 1) * (d + 1);
                } else {
                    S_after = (a + 1) * (c + 1) + (b - 1) * (d - 1);
                }
                double logH = 0.0;
                if (S_before > 0 && S_after > 0){
                    logH = std::log(static_cast<double>(S_before)) - std::log(static_cast<double>(S_after));
                }
                double expo_eff = static_cast<double>(-p.beta * (dE - s * p.mu_eff)) + logH;
                expo_eff = std::clamp(expo_eff, -40.0, 40.0);
                bool accept = (expo_eff >= 0.0) || (std::exp(expo_eff) > static_cast<double>(uni(rng)));

                ++R.attempted;
                ++block_attempted;
                ++att_since_mu;

                if (accept){
                    ++R.accepted;
                    ++block_accepted;
                    ++acc_since_mu;
                    if (forward) ++R.accepted_forward;
                    else         ++R.accepted_backward;
                    R.E = E_trial;

                    if (forward){
                        nh_n.erase(std::find(nh_n.begin(), nh_n.end(), i));
                        nh_p.push_back(i);
                        w.erase(std::find(w.begin(), w.end(), j));
                        cl.push_back(j);
                        ++R.Nprot;
                    } else {
                        nh_p.erase(std::find(nh_p.begin(), nh_p.end(), i));
                        nh_n.push_back(i);
                        cl.erase(std::find(cl.begin(), cl.end(), j));
                        w.push_back(j);
                        --R.Nprot;
                    }

                    if (R.Nprot == R.target_count){
                        R.reached_target = true;
                        R.mu_eff = p.mu_eff;
                        mc_report() << std::setprecision(6)
                                    << "[sgmc] reached target in epoch=" << epoch
                                    << " step_in_block=" << (t + 1)
                                    << " Nprot=" << R.Nprot << "/" << R.Nh
                                    << " mu/kT=" << p.beta * p.mu_eff
                                    << " E=" << R.E
                                    << " fwd_acc=" << R.accepted_forward
                                    << " bwd_acc=" << R.accepted_backward
                                    << " remaining=0" << std::endl;
                        if (seed_out){
                            seed_out->nh_p_mc.clear();
                            seed_out->nh_n_remaining.clear();
                            seed_out->w_remaining.clear();
                            seed_out->cl_mc.clear();
                            for (int idx = 0; idx < sys.natoms; ++idx){
                                const int t = sys.atoms[idx].type;
                                const real zz = sys.atoms[idx].z;
                                if      (t == p.NH_P_type) seed_out->nh_p_mc.push_back(idx);
                                else if (t == p.NH_N_type) seed_out->nh_n_remaining.push_back(idx);
                                else if (t == p.W_type){
                                    if (zz < mcplan.z_cut) seed_out->w_remaining.push_back(idx);
                                } else if (t == p.Cl_type){
                                    if (zz < mcplan.z_cut) seed_out->cl_mc.push_back(idx);
                                }
                            }
                            seed_out->E = R.E;
                        }
                        return R;
                    }
                } else {
                    set_atom_identity(sys, d_atoms, i, oi.type, oi.q, d_params.ntypes);
                    set_atom_identity(sys, d_atoms, j, oj.type, oj.q, d_params.ntypes);
                }
            }

            real acc_rate_block = (block_attempted > 0) ? static_cast<real>(block_accepted) / static_cast<real>(block_attempted) : static_cast<real>(0.0);
            int remaining_epoch = std::max(0, R.target_count - R.Nprot);
            mc_report() << std::setprecision(6)
                        << "[sgmc] chunk acc=" << acc_since_mu << "/" << Nacc_target
                        << " attempts=" << att_since_mu
                        << " acc_rate=" << acc_rate_block
                        << " fwd_total=" << R.accepted_forward
                        << " bwd_total=" << R.accepted_backward
                        << " gc_block=" << gc_block
                        << " remaining=" << remaining_epoch
                        << std::endl;
        }

        real f = (R.Nh > 0) ? static_cast<real>(R.Nprot) / static_cast<real>(R.Nh) : static_cast<real>(0.0);
        real f0 = clamp_real(fstar, static_cast<real>(1e-6), static_cast<real>(1.0 - 1e-6));
        real f1 = clamp_real(f,     static_cast<real>(1e-6), static_cast<real>(1.0 - 1e-6));
        real logit_diff = std::log(f1 / (static_cast<real>(1.0) - f1)) - std::log(f0 / (static_cast<real>(1.0) - f0));
        real eta = (p.mu_eta > static_cast<real>(0.0)) ? p.mu_eta : static_cast<real>(0.3);
        bool early_stop = (att_since_mu >= Amax) && (acc_since_mu < Nacc_target);
        real w_info = (Nacc_target > 0)
                        ? clamp_real(static_cast<real>(acc_since_mu) / static_cast<real>(std::max(1, Nacc_target)),
                                     static_cast<real>(0.0), static_cast<real>(1.0))
                        : static_cast<real>(1.0);
        real mu_step = (static_cast<real>(1.0) / p.beta) * eta * logit_diff * w_info;
        p.mu_eff -= mu_step;

        R.blocks_done = epoch + 1;

        mc_report() << std::setprecision(6)
                    << "[sgmc] mu-update#" << (epoch + 1)
                    << " cause=" << (early_stop ? "Amax" : "Nacc")
                    << " mu/kT=" << p.beta * p.mu_eff
                    << " f=" << f
                    << " acc=" << acc_since_mu << "/" << Nacc_target
                    << " attempts=" << att_since_mu << "/" << Amax
                    << " fwd_total=" << R.accepted_forward
                    << " bwd_total=" << R.accepted_backward
                    << " remaining=" << std::max(0, R.target_count - R.Nprot)
                    << " R0=" << R0
                    << " w_info=" << w_info << std::endl;
    }

    R.mu_eff = p.mu_eff;
    return R;
}
