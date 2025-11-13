#include "io.hpp"
#include "neighbor.hpp"
#include "energy.hpp"
#include "device_data.hpp"
#include "pme.hpp"
#include "mc.hpp"
#include "report.hpp"
#include <thrust/host_vector.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <random>
#include <cuda_runtime.h>

 

static DeviceAtoms upload_atoms_to_device(const System& sys){
    DeviceAtoms d;
    d.natoms = sys.natoms;
    if (d.natoms<=0){
        d.x.clear(); d.y.clear(); d.z.clear();
        d.q.clear(); d.type.clear();
        return d;
    }
    thrust::host_vector<real> hx(d.natoms), hy(d.natoms), hz(d.natoms), hq(d.natoms);
    thrust::host_vector<int> htype(d.natoms);
    for (int i=0;i<d.natoms;++i){
        hx[i]=sys.atoms[i].x;
        hy[i]=sys.atoms[i].y;
        hz[i]=sys.atoms[i].z;
        hq[i]=sys.atoms[i].q;
        htype[i]=sys.atoms[i].type-1;
    }
    d.x.assign(hx.begin(), hx.end());
    d.y.assign(hy.begin(), hy.end());
    d.z.assign(hz.begin(), hz.end());
    d.q.assign(hq.begin(), hq.end());
    d.type.assign(htype.begin(), htype.end());
    return d;
}

static DeviceParams upload_params_to_device(const Params& p){
    DeviceParams dp;
    dp.ntypes = p.ntypes;
    dp.rc = p.rc;
    dp.epsilon_r = p.epsilon_r;
    dp.lj_shift = p.lj_shift;
    dp.coulombtype = p.coulombtype;
    dp.ewald_rtol = p.ewald_rtol;
    dp.ewald_alpha = p.ewald_alpha;
    dp.pme_spacing = p.pme_spacing;
    dp.pme_order = p.pme_order;
    dp.pme_3dc_enabled = p.pme_3dc_enabled;
    dp.pme_3dc_zfac = p.pme_3dc_zfac;
    if (!p.lj_eps.empty()){
        dp.lj_eps.assign(p.lj_eps.begin(), p.lj_eps.end());
        dp.lj_sig.assign(p.lj_sig.begin(), p.lj_sig.end());
    }
    if (p.lj_shift && !p.lj_shift_tbl.empty()){
        dp.lj_shift_tbl.assign(p.lj_shift_tbl.begin(), p.lj_shift_tbl.end());
    } else {
        dp.lj_shift_tbl.clear();
    }
    return dp;
}

int main(int argc, char** argv){
    if (argc < 3){
        std::cerr << "Usage: " << argv[0] << " in.data params.in [--write-data out.data]\n";
        return 1;
    }
    std::string data_path = argv[1];
    std::string params_path = argv[2];
    std::string write_data_path;

    for (int i=3; i<argc; ++i){
        std::string arg = argv[i];
        if (arg == "--write-data"){
            if (i+1 >= argc){
                std::cerr << "--write-data requires a path\n";
                return 1;
            }
            write_data_path = argv[++i];
        } else {
            std::cerr << "Unrecognized argument: " << arg << "\n";
            return 1;
        }
    }

    PMEPlan plan;
    bool planActive = false;
    auto destroyPlan = [&](){
        if (planActive){
            pmeDestroy(&plan);
            planActive = false;
        }
    };

    try{
        ReportFileGuard report_guard("report.txt");
        System sys = read_lammps_full(data_path);
        if (sys.ntypes<=0){ int tmax=0; for(const auto& a: sys.atoms) tmax=std::max(tmax, a.type); sys.ntypes = tmax; }
        Params p = read_params(params_path, sys.ntypes);

        // Wrap positions
        wrap_positions(sys);
        Box writeBox = sys.box;

        if (p.pme_3dc_enabled){
            real zfac = (p.pme_3dc_zfac < 2.0) ? real(2.0) : p.pme_3dc_zfac;
            real oldLz = sys.box.Lz;
            sys.box.Lz = oldLz * zfac;
            mc_report() << std::setprecision(3)
                      << "[3dc] Enabling slab correction: zfac=" << zfac
                      << ", Lz: " << oldLz << " -> " << sys.box.Lz << " (nm)\n";
        }

        // Build 1-2 adjacency (CSR)
        CSR bonds12 = build_bond_adjacency(sys);

        // Upload atoms once and keep on device
        DeviceAtoms d_atoms = upload_atoms_to_device(sys);
        // Upload bonds adjacency once for neighbor exclusion
        DeviceCSR d_bonds = copy_csr_to_device(bonds12);
        // Build neighbors (half) directly on device
        DeviceCSR d_neigh = build_half_neighbors_gpu(d_atoms, sys.box, d_bonds, p.rc);
        // Upload LJ parameter tables once
        DeviceParams d_params = upload_params_to_device(p);

        if (p.coulombtype == "pme"){
            pmeCreate(&plan, d_params, sys.box, p.rc);
            planActive = true;
            mc_report() << "PME grid: " << plan.nx << " x " << plan.ny << " x " << plan.nz
                      << " order=" << plan.order
                      << " alpha=" << plan.alpha
                      << " spacing≈" << sys.box.Lx / plan.nx << "\n";
        }

        mc_report().setf(std::ios::fixed);
        mc_report() << "Atoms: " << sys.natoms << "\n";
        mc_report() << std::setprecision(3)
                  << "Box: Lx=" << sys.box.Lx << " Ly=" << sys.box.Ly << " Lz=" << sys.box.Lz << "\n";
        mc_report() << std::setprecision(4)
                  << "cutoff: " << p.rc << "\n";
        mc_report() << "Neighbor pairs (half): " << d_neigh.nnz << "\n";
        mc_report() << "coulombtype: " << p.coulombtype << " epsilon_r: " << p.epsilon_r << "\n";

        // GPU LJ energy
        real E_lj = compute_lj_sr_energy_gpu(d_atoms, d_params, d_neigh, sys.box);
        mc_report() << std::setprecision(6)
                  << "E_LJ (SR) = " << E_lj << "\n";

        real E_coul = 0.0;
        PMEEnergyComponents comps;
        if (p.coulombtype == "coul_cut"){
            E_coul = compute_coul_sr_energy_gpu(d_atoms, d_params, d_neigh, sys.box);
            mc_report() << std::setprecision(6)
                      << "E_Coulomb (SR, cutoff) = " << E_coul << "\n";
        } else if (p.coulombtype == "pme"){
            E_coul = pmeEnergy(plan, d_atoms, d_neigh, sys.box, &comps);
            mc_report() << std::setprecision(6)
                      << "E_Coulomb (PME) = " << E_coul
                      << "  [real=" << comps.real_space
                      << ", k=" << comps.recip_space
                      << ", self=" << comps.self_term
                      << ", qcorr=" << comps.qcorr_term << "]\n";
        } else {
            destroyPlan();
            std::cerr << "Error: Unsupported coulombtype: " << p.coulombtype << "\n";
            return 2;
        }

        real totalEnergy = E_lj + E_coul;
        mc_report() << std::setprecision(6)
                  << "E_total = " << totalEnergy << "\n";

        PMEPlan* planPtr = planActive ? &plan : nullptr;

        std::mt19937 rng(static_cast<unsigned>(p.rng_seed));
        MCPlan mcplan = build_plan(sys, p);

        Phase1Result ph1;
        const bool use_direct_phase1 = p.target_protonation_pct > static_cast<real>(p.direct_pct_threshold);
        if (use_direct_phase1){
            ph1 = apply_direct_protonation(sys, d_atoms, d_params, d_neigh, sys.box, planPtr, p, mcplan);
            write_lammps_full("phase1.data", sys, writeBox);
            mc_report() << "[write] phase1.data written\n";
        } else {
            PhaseGCResult gc_result = phase_gc_mu_control(sys, d_atoms, d_params, d_neigh, sys.box, planPtr, p, mcplan, rng, &ph1);
            ph1.attempted = gc_result.attempted;
            ph1.accepted = gc_result.accepted;
            ph1.E = gc_result.E;

            write_lammps_full("phase1.data", sys, writeBox);
            mc_report() << "[write] phase1.data written\n";
        }

        Phase2Result ph2;
        if (p.phase2_wc){
            ph2 = phase2_swap_jitter(sys, d_atoms, d_params, d_neigh, sys.box, planPtr,
                                     p, mcplan, ph1, rng);
        } else {
            mc_report() << "[phase2] skipped (phase2_wc=false)\n";
            ph2.E = ph1.E;
        }

        write_lammps_full("phase2.data", sys, writeBox);
        mc_report() << "[write] phase2.data written\n";

        mc_report() << std::setprecision(6)
                  << "[summary] E0/after phase1/after phase2 = "
                  << totalEnergy
                  << " / " << ph1.E
                  << " / " << ph2.E << "\n";

        if (!write_data_path.empty()){
            write_lammps_full(write_data_path, sys, writeBox);
            mc_report() << "[write] Wrote updated data file to " << write_data_path << "\n";
        }

        destroyPlan();
        // no dump to file per request
    } catch (const std::exception& e){
        destroyPlan();
        std::cerr << "Error: " << e.what() << "\n"; return 2;
    }
    return 0;
}
