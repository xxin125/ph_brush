#include "io.hpp"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <iomanip>

static inline std::string strip(const std::string& s){
    auto is_not_space = [](unsigned char c){ return !std::isspace(c); };
    auto it = std::find_if(s.begin(), s.end(), is_not_space);
    if (it == s.end()) return "";
    auto rit = std::find_if(s.rbegin(), s.rend(), is_not_space);
    return std::string(it, rit.base());
}

static inline real parse_real(const std::string& s){
    return static_cast<real>(std::stod(s));
}

static bool parse_bool(std::string s){
    auto trim = [](std::string& x){
        size_t a = x.find_first_not_of(" \t\r\n");
        size_t b = x.find_last_not_of(" \t\r\n");
        if (a == std::string::npos){ x.clear(); return; }
        x = x.substr(a, b - a + 1);
    };
    trim(s);
    std::string t;
    t.reserve(s.size());
    for (char c : s){
        t.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (t=="1" || t=="true" || t=="yes" || t=="on") return true;
    if (t=="0" || t=="false" || t=="no"  || t=="off") return false;
    throw std::runtime_error("invalid boolean: " + s);
}

static inline bool starts_with(const std::string& s, const std::string& p){
    return s.rfind(p, 0) == 0;
}

static double erfinv_newton(double x){
    if (x <= -1.0) return -std::numeric_limits<double>::infinity();
    if (x >= 1.0) return std::numeric_limits<double>::infinity();
    const double a = 0.147;
    double ln = std::log(1.0 - x*x);
    double tt1 = 2.0/(M_PI*a) + ln/2.0;
    double tt2 = ln/a;
    double inside = std::sqrt(tt1*tt1 - tt2);
    double initial = std::sqrt(std::max(0.0, inside - tt1));
    if (x < 0) initial = -initial;
    double val = initial;
    for (int i=0;i<3;++i){
        double err = std::erf(val) - x;
        double deriv = 2.0/std::sqrt(M_PI) * std::exp(-val*val);
        val -= err / deriv;
    }
    return val;
}

static double erfc_inv(double y){
    if (y <= 0.0) return std::numeric_limits<double>::infinity();
    if (y >= 2.0) return -std::numeric_limits<double>::infinity();
    double x = 1.0 - y;
    return erfinv_newton(x);
}

static double calc_ewald_alpha(double rc, double rtol){
    if (rtol <= 0.0) rtol = 1e-5;
    double clamped = std::clamp(rtol, 1e-12, 0.5);
    double inv = erfc_inv(clamped);
    return inv / rc;
}

System read_lammps_full(const std::string& path){
    System sys;
    std::ifstream in(path);
    if(!in) throw std::runtime_error("Failed to open LAMMPS data: "+path);
    std::string line;
    int natoms=0, nbonds=0, ntypes=0;
    real xlo=0,xhi=0,ylo=0,yhi=0,zlo=0,zhi=0;
    // First pass: header
    while (std::getline(in, line)){
        std::string s = strip(line);
        if (s.empty()) continue;
        if (s.find(" atoms")!=std::string::npos && s.find("atom types")==std::string::npos) {
            std::istringstream iss(s); iss>>natoms;
        } else if (s.find(" bonds")!=std::string::npos && s.find("bond types")==std::string::npos) {
            std::istringstream iss(s); iss>>nbonds;
        } else if (s.find(" atom types")!=std::string::npos) {
            std::istringstream iss(s); iss>>ntypes;
        } else if (s.find("xlo xhi")!=std::string::npos) {
            std::istringstream iss(s); iss>>xlo>>xhi;
        } else if (s.find("ylo yhi")!=std::string::npos) {
            std::istringstream iss(s); iss>>ylo>>yhi;
        } else if (s.find("zlo zhi")!=std::string::npos) {
            std::istringstream iss(s); iss>>zlo>>zhi;
        } else if (starts_with(s, "Atoms")) break;
    }
    if (natoms<=0) throw std::runtime_error("No atoms in header");
    sys.natoms=natoms; sys.nbonds=nbonds; sys.ntypes=ntypes;
    sys.box.Lx = xhi-xlo; sys.box.Ly = yhi-ylo; sys.box.Lz = zhi-zlo;

    // Rewind and read sections
    in.clear(); in.seekg(0);
    bool inAtoms=false, inBonds=false;
    sys.atoms.reserve(natoms);
    sys.bonds.reserve(nbonds);
    std::unordered_map<int,int> id2idx; id2idx.reserve(natoms*2);
    while (std::getline(in, line)){
        std::string s = strip(line);
        if (s.empty()) continue;
        if (starts_with(s, "Atoms")) { inAtoms=true; inBonds=false; continue; }
        if (starts_with(s, "Bonds")) { inAtoms=false; inBonds=true; continue; }
        if (starts_with(s, "Angles") || starts_with(s, "Dihedrals") || starts_with(s, "Impropers") || starts_with(s, "Velocities")) { inAtoms=false; inBonds=false; continue; }
        if (inAtoms){
            std::vector<std::string> t; std::istringstream iss(s); std::string tok; while(iss>>tok) t.push_back(tok);
            if (t.size() < 7) continue;
            Atom a;
            a.id=std::stoi(t[0]);
            a.mol=std::stoi(t[1]);
            a.type=std::stoi(t[2]);
            a.q=parse_real(t[3]);
            a.x=parse_real(t[4]);
            a.y=parse_real(t[5]);
            a.z=parse_real(t[6]);
            id2idx[a.id] = (int)sys.atoms.size();
            sys.atoms.push_back(a);
        } else if (inBonds){
            std::vector<std::string> t; std::istringstream iss(s); std::string tok; while(iss>>tok) t.push_back(tok);
            if (t.size() < 4) continue;
            Bond b; b.id=std::stoi(t[0]); b.type=std::stoi(t[1]);
            int ai = std::stoi(t[2]); int aj = std::stoi(t[3]);
            auto iti = id2idx.find(ai), itj = id2idx.find(aj);
            if (iti==id2idx.end() || itj==id2idx.end()) continue;
            b.i = iti->second; b.j = itj->second;
            sys.bonds.push_back(b);
        }
    }
    if ((int)sys.atoms.size() != natoms) throw std::runtime_error("Atom count mismatch in Atoms section");
    return sys;
}

CSR build_bond_adjacency(const System& sys){
    const int N = sys.natoms;
    std::vector<std::vector<int>> tmp(N);
    for (const auto& b: sys.bonds){
        tmp[b.i].push_back(b.j);
        tmp[b.j].push_back(b.i);
    }
    CSR out; out.head.assign(N+1,0);
    for (int i=0;i<N;++i){
        auto& v = tmp[i];
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        out.head[i+1] = out.head[i] + (int)v.size();
    }
    out.list.resize(out.head[N]);
    for (int i=0;i<N;++i){
        int off = out.head[i];
        const auto& v = tmp[i];
        for (size_t k=0;k<v.size();++k) out.list[off+k] = v[k];
    }
    return out;
}

Params read_params(const std::string& path, int ntypes){
    Params p; p.rc=0.0; p.ntypes = ntypes; p.lj_eps.assign(ntypes*ntypes, 0.0); p.lj_sig.assign(ntypes*ntypes, 0.0); p.lj_shift=true; p.epsilon_r=1.0; p.coulombtype="coul_cut";
    std::ifstream in(path);
    if(!in) throw std::runtime_error("Failed to open params: "+path);
    std::string line;
    auto normkey = [](std::string s){ for(char& c:s) if(c=='-') c='_'; return s; };
    while (std::getline(in, line)){
        std::string s = strip(line);
        if (s.empty() || s[0]=='#') continue;
        auto eq = s.find('=');
        std::string key, val;
        if (eq != std::string::npos){ key = normkey(strip(s.substr(0,eq))); val = strip(s.substr(eq+1)); }
        if (!key.empty()){
            if (key=="cutoff_nm") p.rc = parse_real(val);
            else if (key=="lj_shift"){
                std::string v=val; for(char& c:v) c=std::tolower(c);
                p.lj_shift = (v=="1"||v=="true"||v=="yes"||v=="on");
            } else if (key=="epsilon_r"){
                p.epsilon_r = parse_real(val);
            } else if (key=="coulombtype"){
                std::string v=val; for(char& c:v) c=std::tolower(c);
                if (v=="coul_cut" || v=="coulcut") p.coulombtype = "coul_cut";
                else if (v=="pme") p.coulombtype = "pme";
                else p.coulombtype = "unknown";
            } else if (key=="ewald_rtol"){
                p.ewald_rtol = parse_real(val);
            } else if (key=="ewald_alpha"){
                p.ewald_alpha = parse_real(val);
            } else if (key=="pme_spacing"){
                p.pme_spacing = parse_real(val);
            } else if (key=="pme_order"){
                p.pme_order = std::stoi(val);
            } else if (key=="pme_3dc"){
                std::istringstream iss(val);
                std::string flag;
                real zfac = p.pme_3dc_zfac;
                if (!(iss >> flag)) flag = "yes";
                if (iss.good()) iss >> zfac;
                std::string low = flag;
                for (char& c : low) c = std::tolower(static_cast<unsigned char>(c));
                p.pme_3dc_enabled = (low=="yes" || low=="true" || low=="on" || low=="1");
                if (zfac < 2.0) zfac = 2.0;
                p.pme_3dc_zfac = zfac;
            } else if (key=="nh_n_type") {
                p.NH_N_type = std::stoi(val);
            } else if (key=="nh_p_type") {
                p.NH_P_type = std::stoi(val);
            } else if (key=="water_type" || key=="w_type") {
                p.W_type = std::stoi(val);
            } else if (key=="cl_type") {
                p.Cl_type = std::stoi(val);
            } else if (key=="target_pct" || key=="target_protonation_pct") {
                p.target_protonation_pct = parse_real(val);
            } else if (key=="rng_seed") {
                p.rng_seed = std::stoi(val);
            } else if (key=="energy_interval") {
                p.energy_interval = std::stoi(val);
            } else if (key=="patience") {
                p.patience = std::stoi(val);
            } else if (key=="max_attempts") {
                p.max_attempts = std::stoi(val);
            } else if (key=="beta") {
                p.beta = parse_real(val);
            } else if (key=="w_z_extra_nm") {
                p.w_z_extra_nm = parse_real(val);
            } else if (key=="direct_pct_threshold" || key=="direct_protonation_pct_threshold") {
                p.direct_pct_threshold = std::stoi(val);
            } else if (key=="mu_eta") {
                p.mu_eta = parse_real(val);
            } else if (key=="gc_block_fixed") {
                p.gc_block_fixed = std::stoi(val);
            } else if (key=="sgmc_epoch_attempt_max" || key=="epoch_attempt_max") {
                p.sgmc_epoch_attempt_max = std::stoi(val);
            } else if (key=="sgmc_nacc_frac_round" || key=="nacc_frac_round") {
                p.sgmc_nacc_frac_round = parse_real(val);
            } else if (key=="phase2_wc" || key=="phase2_WC") {
                p.phase2_wc = parse_bool(val);
            }
            // ignore other keyed entries
        } else if (starts_with(s, "lj ")){
            std::istringstream iss(s); std::string tag; int ti,tj; real eps,sig; iss>>tag>>ti>>tj>>eps>>sig;
            if (ti<1||tj<1||ti>ntypes||tj>ntypes) throw std::runtime_error("LJ type index out of bounds");
            int I=(ti-1)*ntypes+(tj-1), J=(tj-1)*ntypes+(ti-1);
            p.lj_eps[I]=p.lj_eps[J]=eps;
            p.lj_sig[I]=p.lj_sig[J]=sig;
        }
    }
    if (p.rc<=0.0) throw std::runtime_error("cutoff_nm must be > 0 in params");
    if (p.pme_spacing <= 0.0) p.pme_spacing = 0.12;
    if (p.pme_order <= 0) p.pme_order = 4;
    if (p.coulombtype == "pme"){
        if (p.ewald_alpha <= 0.0){
            p.ewald_alpha = static_cast<real>(calc_ewald_alpha(static_cast<double>(p.rc),
                                                               static_cast<double>(p.ewald_rtol)));
        }
    } else {
        p.ewald_alpha = 0.0;
    }
    // Precompute LJ shift at rc if requested
    p.lj_shift_tbl.assign(ntypes*ntypes, 0.0);
    if (p.lj_shift){
        for (int i=0;i<ntypes;++i){
            for (int j=0;j<ntypes;++j){
                real eps=p.lj_eps[i*ntypes+j], sig=p.lj_sig[i*ntypes+j];
                if (eps>0.0 && sig>0.0){
                    real sr = sig / p.rc; real sr2=sr*sr; real sr6=sr2*sr2*sr2; real sr12=sr6*sr6;
                    p.lj_shift_tbl[i*ntypes+j] = real(4.0)*eps*(sr12 - sr6);
                }
            }
        }
    }
    if (p.direct_pct_threshold < 10 || p.direct_pct_threshold > 100){
        throw std::runtime_error("direct_pct_threshold must be between 10 and 100");
    }
    if (p.gc_block_fixed <= 0) p.gc_block_fixed = 2000;
    if (p.sgmc_epoch_attempt_max <= 0) p.sgmc_epoch_attempt_max = 100000;
    if (p.sgmc_nacc_frac_round <= static_cast<real>(0.0)) p.sgmc_nacc_frac_round = static_cast<real>(0.10);
    if (p.mu_eta <= static_cast<real>(0.0)) p.mu_eta = static_cast<real>(0.3);
    auto ensure_type = [&](const char* name, int typeId){
        if (typeId < 1 || typeId > ntypes){
            std::ostringstream oss;
            oss << name << " (" << typeId << ") must be between 1 and " << ntypes;
            throw std::runtime_error(oss.str());
        }
    };
    ensure_type("nh_n_type", p.NH_N_type);
    ensure_type("nh_p_type", p.NH_P_type);
    ensure_type("w_type",    p.W_type);
    ensure_type("cl_type",   p.Cl_type);
    return p;
}

void wrap_positions(System& sys){
    const real Lx=sys.box.Lx, Ly=sys.box.Ly, Lz=sys.box.Lz;
    auto wrap = [](real r, real L){
        real w = r - std::floor(r / L) * L;
        if (w < 0) w += L;
        if (w >= L) w -= L;
        return w;
    };
    for (auto& a: sys.atoms){
        a.x = wrap(a.x, Lx);
        a.y = wrap(a.y, Ly);
        a.z = wrap(a.z, Lz);
    }
}

static inline int max_type_id(const std::vector<Bond>& bonds){
    int maxType = 0;
    for (const auto& b : bonds){
        maxType = std::max(maxType, b.type);
    }
    return maxType;
}

void write_lammps_full(const std::string& path, const System& sys, const Box& boxForOutput){
    std::ofstream out(path);
    if (!out){
        throw std::runtime_error("Failed to open output data file: " + path);
    }
    out.setf(std::ios::fixed);
    out << "# Generated by lj-only MC\n\n";
    out << sys.natoms << " atoms\n";
    out << sys.nbonds << " bonds\n\n";
    out << sys.ntypes << " atom types\n";
    int nbondtypes = max_type_id(sys.bonds);
    if (nbondtypes > 0){
        out << nbondtypes << " bond types\n";
    }
    out << "\n";
    out << std::setprecision(12);
    out << 0.0 << " " << boxForOutput.Lx << " xlo xhi\n";
    out << 0.0 << " " << boxForOutput.Ly << " ylo yhi\n";
    out << 0.0 << " " << boxForOutput.Lz << " zlo zhi\n\n";

    out << "Atoms # full\n\n";
    out << std::setprecision(8);
    for (const auto& a : sys.atoms){
        out << a.id << " "
            << a.mol << " "
            << a.type << " "
            << a.q << " "
            << a.x << " "
            << a.y << " "
            << a.z << "\n";
    }
    out << "\n";
    if (!sys.bonds.empty()){
        out << "Bonds\n\n";
        for (const auto& b : sys.bonds){
            out << b.id << " "
                << b.type << " "
                << sys.atoms[b.i].id << " "
                << sys.atoms[b.j].id << "\n";
        }
        out << "\n";
    }
}
