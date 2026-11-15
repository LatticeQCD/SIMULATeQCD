/*
 * Mobius domain-wall measurements on one gauge configuration, in the conventions and file formats of the GPT
 * production scripts (DSDRMDWFruns/Measurements), so that the existing analysis scripts read the output unchanged
 * and both codes agree configuration by configuration (parity suite: LatticeSimulateMDWF/parity):
 *
 *   mres_screening = 1: point source at `source`, mass `mres_mass`, 12 solves (MDWFMeasurements.h):
 *       <out_prefix>_mres_spatial.tsv       as measure_mres_screening_mdwf.py (J5qP, PP, mres per coordinate)
 *       <out_prefix>_screening_spatial.tsv  connected C_G for `screening_channels` (old = the 22 MesonCorr channels)
 *     along `directions` (any of x, y, z, t; GPT's default "spatial" is z = spatial_z, corr_dir 2).
 *   condensate = 1: `nrv` Z2 noise vectors (hash noise, seed string `noise_seed`, default "<label>_noise"), for each
 *     flavour of `condensate_masses` ("light:0.00084525,strange:0.04122"):
 *       <out_prefix>_condensates.tsv        as chiral-condensate_4d_clean_light_strange.py (I_id, I_raw,
 *                                           chiral_condensate_bare_per_V, P5_raw/pseudoscalar_loop_per_V if
 *                                           measure_pseudoscalar, S2_raw/chi_conn_*_per_V if measure_connected)
 *     A flavour may carry its own vector count, "light:0.00084525:40,strange:0.04122:10" (default nrv); vector i uses
 *     the same noise for every flavour. S2_raw only for the first `nrv_connected` vectors (-1 = all).
 *     slice_dirs (e.g. "xyz", default "" = off):
 *       <out_prefix>_condensate_slices.tsv  as measure_condensate_slices_mdwf.py (I_id, I_raw, P5_raw with dir "all",
 *                                           L_raw / L5_raw = eta^+ S eta / eta^+ g5 S eta summed over x_mu slices)
 *
 *   screening_n2max > 0 (2026-11-12, MDWFScreeningMomentum.h; default 0 = off, output above unchanged): from the same
 *     point-source solves, the connected correlators of `screening_mom_channels` (default "minimal" = I, 5, X, Y, Z, T)
 *     along every slicing direction of `screening_mom_dirs` (default "xyz") at every transverse momentum n
 *     (n_mu = 0, |n|^2 <= screening_n2max, including n = 0):
 *       <out_prefix>_screening_mom.tsv      label conf mass Ls M5 b5 c5 src_x..src_t direction corr_dir nx ny nz n2
 *                                           channel coord re im;  sum_{x_mu = x0_mu + coord} G(x) exp(i p.(x - x0))
 *   slice_n2max > 0 (2026-11-12; default 0 = off): the noise loops of slice_dirs also at transverse momenta
 *     0 < |n|^2 <= slice_n2max (both n and -n):
 *       <out_prefix>_condensate_slices_mom.tsv  label flavor mass rnd_vec_idx operator dir nx ny nz coord real imag;
 *                                           L_raw / L5_raw = sum_{x with x_mu = coord} exp(i p.x) eta^+ (g5) S eta
 *
 *   pion_dynamics = 1: point source `piondyn_source` ("random" = hash of piondyn_seed and label, or "x,y,z,t"), mass
 *     `piondyn_mass` (default mres_mass), 12 solves (MDWFPionDynamics.h):
 *       <out_prefix>_piondyn.tsv            as measure_charge_spectral_mdwf.py (= measure_pion_dynamics_mdwf.py + rho/a1):
 *                                           temporal P, A0, V0, Vi, Ai correlators with local, J5q and Mobius conserved
 *                                           currents at |n|^2 <= piondyn_n2max; along z (p = 0) PP, SS, A0A0, J5qP, AzP,
 *                                           AzcP, V0V0, V0cV0, A0cA0 and the V/A/Vc/Ac tensor diagonals (2026-11-07)
 *
 * c5 = b5 - 1 (MDWFMobiusMapping.h). Solver: mixed_precision = 1 (single-precision CG with double reliable updates,
 * same stopping rule on the true residual) or 0 (double). force_halos = 0 (default): no halo buffers on one GPU.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFMeasurements.h"
#include "../experimental/mdwf/MDWFPionDynamics.h"
#include "../experimental/mdwf/MDWFScreeningMomentum.h"
#include "../experimental/mdwf/MDWFLowModes.h"
#include "../experimental/mdwf/MDWFMixedPrecisionSolver.h"
#include "../gauge/gaugeAction.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

class MDWFMeasureParameters : public LatticeParameters {
public:
    Parameter<double> M5;
    Parameter<double> b5;
    Parameter<double> csw;
    Parameter<int> ls;
    Parameter<int> antiperiodic_t;
    Parameter<std::string> start;
    Parameter<int> seed;
    Parameter<std::string> label;
    Parameter<std::string> out_prefix;
    Parameter<int> force_halos;
    Parameter<int> mixed_precision;
    Parameter<double> reliable_delta;
    Parameter<int> max_iter;

    Parameter<int> mres_screening;
    Parameter<double> mres_mass;
    Parameter<int, 4> source;
    Parameter<std::string> directions;
    Parameter<std::string> screening_channels;
    Parameter<double> screening_precision;
    Parameter<int> screening_n2max;
    Parameter<std::string> screening_mom_dirs;
    Parameter<std::string> screening_mom_channels;

    Parameter<int> condensate;
    Parameter<std::string> condensate_masses;
    Parameter<int> nrv;
    Parameter<int> i_start;
    Parameter<std::string> noise_seed;
    Parameter<int> measure_connected;
    Parameter<int> measure_pseudoscalar;
    Parameter<double> condensate_precision;
    Parameter<int> nrv_connected;
    Parameter<std::string> slice_dirs;
    Parameter<int> slice_n2max;

    Parameter<int> pion_dynamics;
    Parameter<double> piondyn_mass;
    Parameter<std::string> piondyn_source;
    Parameter<std::string> piondyn_seed;
    Parameter<int> piondyn_n2max;
    Parameter<double> piondyn_precision;

    Parameter<int> lma;
    Parameter<int> lma_nmodes;
    Parameter<double> lma_precision;
    Parameter<double> lma_tol;
    Parameter<int> lma_nbasis;
    Parameter<int> lma_nkeep;
    Parameter<int> lma_max_restarts;
    Parameter<int> lma_selftest;

    MDWFMeasureParameters() {
        addDefault(M5, "M5", 1.8);
        addDefault(b5, "b5", 1.9);
        addDefault(csw, "c_sw", 0.0);
        addDefault(ls, "ls", 16);
        addDefault(antiperiodic_t, "antiperiodic_t", 1);
        addDefault(start, "start", std::string("random"));
        addDefault(seed, "seed", 20261007);
        addDefault(label, "label", std::string(""));
        addDefault(out_prefix, "out_prefix", std::string(""));
        addDefault(force_halos, "force_halos", 0);
        addDefault(mixed_precision, "mixed_precision", 1);
        addDefault(reliable_delta, "reliable_delta", 0.1);
        addDefault(max_iter, "max_iter", 50000);

        addDefault(mres_screening, "mres_screening", 1);
        addDefault(mres_mass, "mres_mass", 0.00084525);
        static const int defaultSource[4] = {0, 0, 0, 0};
        addDefault(source, "source", defaultSource);
        addDefault(directions, "directions", std::string("z"));
        addDefault(screening_channels, "screening_channels", std::string("old"));
        addDefault(screening_precision, "screening_precision", 1e-12);
        addDefault(screening_n2max, "screening_n2max", 0);
        addDefault(screening_mom_dirs, "screening_mom_dirs", std::string("xyz"));
        addDefault(screening_mom_channels, "screening_mom_channels", std::string("minimal"));

        addDefault(condensate, "condensate", 1);
        addDefault(condensate_masses, "condensate_masses", std::string("light:0.00084525,strange:0.04122"));
        addDefault(nrv, "nrv", 10);
        addDefault(i_start, "i_start", 0);
        addDefault(noise_seed, "noise_seed", std::string(""));
        addDefault(measure_connected, "measure_connected", 1);
        addDefault(measure_pseudoscalar, "measure_pseudoscalar", 1);
        addDefault(condensate_precision, "condensate_precision", 1e-8);
        addDefault(nrv_connected, "nrv_connected", -1);
        addDefault(slice_dirs, "slice_dirs", std::string(""));
        addDefault(slice_n2max, "slice_n2max", 0);

        addDefault(pion_dynamics, "pion_dynamics", 0);
        addDefault(piondyn_mass, "piondyn_mass", 0.0);
        addDefault(piondyn_source, "piondyn_source", std::string("random"));
        addDefault(piondyn_seed, "piondyn_seed", std::string("piondyn"));
        addDefault(piondyn_n2max, "piondyn_n2max", 4);
        addDefault(piondyn_precision, "piondyn_precision", 1e-8);

        addDefault(lma, "lma", 0);
        addDefault(lma_nmodes, "lma_nmodes", 16);
        addDefault(lma_precision, "lma_precision", 1e-10);
        addDefault(lma_tol, "lma_tol", 1e-8);
        addDefault(lma_nbasis, "lma_nbasis", 0);
        addDefault(lma_nkeep, "lma_nkeep", 0);
        addDefault(lma_max_restarts, "lma_max_restarts", 50);
        addDefault(lma_selftest, "lma_selftest", 0);
    }
};

struct MDWFMeasureFlavour {
    std::string name;
    double mass;
    int nrv;   // -1: the global nrv
};

// "light:0.00084525:40,strange:0.04122" -> [(light, 0.00084525, 40), (strange, 0.04122, -1)]
std::vector<MDWFMeasureFlavour> mdwfParseMasses(const std::string &raw) {
    std::vector<MDWFMeasureFlavour> masses;
    std::istringstream in(raw);
    std::string item;
    while (std::getline(in, item, ',')) {
        if (item.empty()) {
            continue;
        }
        std::vector<std::string> f;
        std::istringstream fields(item);
        std::string field;
        while (std::getline(fields, field, ':')) {
            f.push_back(field);
        }
        if (f.size() == 1) {
            f.insert(f.begin(), "mass" + std::to_string(masses.size()));
        }
        if (f.size() < 2 || f.size() > 3) {
            throw std::runtime_error(stdLogger.fatal("mdwfMeasure: bad flavour '", item, "' in '", raw, "'"));
        }
        masses.push_back({f[0], std::stod(f[1]), f.size() == 3 ? std::stoi(f[2]) : -1});
    }
    if (masses.empty()) {
        throw std::runtime_error(stdLogger.fatal("mdwfMeasure: no masses in '", raw, "'"));
    }
    return masses;
}

std::string mdwfG16(double v) {   // Python f"{v:.16g}"
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.16g", v);
    return buf;
}

std::string mdwfE18(double v) {   // Python f"{v:.18e}"
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.18e", v);
    return buf;
}

template<size_t Ls, bool Mixed>
void runMDWFMeasure(CommunicationBase &commBase, MDWFMeasureParameters &param) {
    const size_t HaloDepth = 2;
    using Solver = std::conditional_t<Mixed, MDWFMobiusEvenOddMixedSolver<HaloDepth, HaloDepth, Ls>,
                                      MDWFMobiusEvenOddSolver<double, HaloDepth, HaloDepth, Ls>>;
    using Meas = MDWFResidualMassMeasurement<HaloDepth, Ls, Solver>;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    typedef GIndexer<All, HaloDepth> GInd;
    const LatticeData lat = GInd::getLatData();

    Gauge gauge(commBase, "MDWF_meas_gauge");
    Gauge fermionGauge(commBase, "MDWF_meas_fgauge");
    std::string conf = param.start();
    if (param.GaugefileName.isSet()) {
        conf = param.GaugefileName();
        rootLogger.info("mdwfMeasure: reading NERSC configuration ", conf);
        gauge.readconf_nersc(conf);
    } else if (param.start() == "unit") {
        gauge.one();
    } else {
        grnd_state<false> h_rand;
        grnd_state<true> d_rand;
        h_rand.make_rng_state(param.seed());
        d_rand = h_rand;
        gauge.random(d_rand.state);
    }
    gauge.updateAll();
    GaugeAction<double, true, HaloDepth, R18> action(gauge);
    const double plaquette = static_cast<double>(action.plaquette());
    mdwfFermionGaugeField<HaloDepth>(fermionGauge, gauge, param.antiperiodic_t() != 0);

    std::string label = param.label();
    if (label.empty()) {
        const size_t slash = conf.find_last_of('/');
        label = slash == std::string::npos ? conf : conf.substr(slash + 1);
    }
    const std::string prefix = param.out_prefix().empty() ? param.measurements_dir() + "/" + label
                                                          : param.out_prefix();
    const double c5 = param.b5() - 1.0;
    rootLogger.info("mdwfMeasure: ", lat.globLX, "x", lat.globLY, "x", lat.globLZ, "x", lat.globLT, ", Ls = ", Ls,
                    ", M5 = ", param.M5(), ", b5 = ", param.b5(), ", c5 = ", c5, ", plaquette = ", plaquette,
                    ", label ", label, ", output prefix ", prefix, Mixed ? ", mixed precision" : ", double precision");

    // Common header columns of GPT's TSVs.
    auto physics = [&](double mass) {
        return label + "\t" + conf + "\t" + mdwfG16(mass) + "\t" + std::to_string(Ls) + "\t" + mdwfG16(param.M5())
               + "\t" + mdwfG16(param.b5()) + "\t" + mdwfG16(c5);
    };

    MDWFSpinor<double, true, All, HaloDepth, Ls> store(commBase, "MDWF_meas_store");

    if (param.mres_screening() != 0) {
        const std::array<int, 4> x0 = {param.source[0], param.source[1], param.source[2], param.source[3]};
        const std::vector<std::string> channels = mdwfParseChannels(param.screening_channels());
        std::vector<int> dirs;
        for (const char d : param.directions()) {
            const size_t mu = std::string("xyzt").find(d);
            if (mu == std::string::npos) {
                throw std::runtime_error(stdLogger.fatal("mdwfMeasure: directions must be letters of xyzt, got ",
                                                         param.directions()));
            }
            dirs.push_back(static_cast<int>(mu));
        }
        const char *dirLabel[4] = {"spatial_x", "spatial_y", "spatial_z", "temporal_t"};
        Meas meas(fermionGauge, param.M5(), param.mres_mass(), param.b5(), param.csw(), "MDWF_meas_pt");
        if constexpr (Mixed) {
            meas.solver().setDelta(param.reliable_delta());
        }
        // Transverse-momentum screening correlators (MDWFScreeningMomentum.h), contractions only.
        const bool mom = param.screening_n2max() > 0;
        std::vector<int> momDirs;
        std::vector<std::string> momChannels;
        std::unique_ptr<LatticeContainer<true, COMPLEX(double)>> momAcc, momRed;
        std::unique_ptr<MDWFScreeningMomentum<HaloDepth, Ls>> momMeas;
        if (mom) {
            for (const char d : param.screening_mom_dirs()) {
                const size_t mu = std::string("xyz").find(d);
                if (mu == std::string::npos) {
                    throw std::runtime_error(stdLogger.fatal("mdwfMeasure: screening_mom_dirs must be letters of xyz, "
                                                             "got ", param.screening_mom_dirs()));
                }
                momDirs.push_back(static_cast<int>(mu));
            }
            momChannels = mdwfParseChannels(param.screening_mom_channels());
            momAcc.reset(new LatticeContainer<true, COMPLEX(double)>(commBase, "MDWF_sm_acc"));
            momRed.reset(new LatticeContainer<true, COMPLEX(double)>(commBase, "MDWF_sm_red"));
            momMeas.reset(new MDWFScreeningMomentum<HaloDepth, Ls>(*momAcc, *momRed, momChannels));
        }
        const MDWFPointMeasurementResult r = mdwfMeasurePointSource<HaloDepth, Ls>(
            meas, store, x0, channels, param.max_iter(), param.screening_precision(),
            mom ? std::function<void(int)>([&](int) { momMeas->addColour(store); }) : std::function<void(int)>());
        rootLogger.info("mdwfMeasure: point source (", x0[0], ",", x0[1], ",", x0[2], ",", x0[3], "), mass ",
                        param.mres_mass(), ": 12 solves, ", r.iterations, " iterations, max residue ", r.maxResidue,
                        ", ", r.seconds, " s; ", channels.size(), " channels");
        if (commBase.IamRoot()) {
            const std::string src = std::to_string(x0[0]) + "\t" + std::to_string(x0[1]) + "\t"
                                    + std::to_string(x0[2]) + "\t" + std::to_string(x0[3]);
            std::ofstream mres(prefix + "_mres_spatial.tsv"), scr(prefix + "_screening_spatial.tsv");
            if (!mres || !scr) {
                throw std::runtime_error(stdLogger.fatal("mdwfMeasure cannot write ", prefix, "_*.tsv"));
            }
            mres << "label\tconf\tmass\tLs\tM5\tb5\tc5\tsrc_x\tsrc_y\tsrc_z\tsrc_t\tdirection\tcorr_dir\tcoord\tJ5qP"
                    "\tPP\tmres\n";
            scr << "label\tconf\tmass\tLs\tM5\tb5\tc5\tsrc_x\tsrc_y\tsrc_z\tsrc_t\tdirection\tcorr_dir\tcoord\tchannel"
                   "\tcorr\n";
            for (const int mu : dirs) {
                const std::string head = physics(param.mres_mass()) + "\t" + src + "\t" + dirLabel[mu] + "\t"
                                         + std::to_string(mu) + "\t";
                for (size_t n = 0; n < r.pp[mu].size(); n++) {
                    mres << head << n << "\t" << mdwfE18(r.j5q[mu][n]) << "\t" << mdwfE18(r.pp[mu][n]) << "\t"
                         << mdwfE18(r.j5q[mu][n] / r.pp[mu][n]) << "\n";
                }
                for (size_t ch = 0; ch < channels.size(); ch++) {
                    for (size_t n = 0; n < r.channels[ch][mu].size(); n++) {
                        scr << head << n << "\t" << channels[ch] << "\t" << mdwfE18(r.channels[ch][mu][n]) << "\n";
                    }
                }
            }
        }
        if (mom) {
            // Projections (all ranks), then the file (root). Check: n = 0 real part vs the screening_spatial sums.
            const std::string src = std::to_string(x0[0]) + "\t" + std::to_string(x0[1]) + "\t"
                                    + std::to_string(x0[2]) + "\t" + std::to_string(x0[3]);
            std::ostringstream body;
            double maxDev = 0.0;
            int nChecked = 0;
            size_t nrows = 0;
            for (const int mu : momDirs) {
                const std::vector<std::array<int, 3>> moms = mdwfTransverseMomenta(mu, param.screening_n2max(), true);
                for (size_t ch = 0; ch < momChannels.size(); ch++) {
                    for (const auto &n : moms) {
                        const std::vector<COMPLEX(double)> c = momMeas->project(static_cast<int>(ch), mu, n, x0);
                        const int n2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
                        if (n2 == 0) {
                            const auto it = std::find(channels.begin(), channels.end(), momChannels[ch]);
                            if (it != channels.end()) {
                                const auto &ref = r.channels[it - channels.begin()][mu];   // global coordinate
                                const int N = static_cast<int>(ref.size());
                                double scale = 1e-300, dev = 0.0;
                                for (int k = 0; k < N; k++) {
                                    scale = std::max(scale, std::abs(ref[k]));
                                }
                                for (int k = 0; k < N; k++) {
                                    dev = std::max(dev, std::abs(real(c[k]) - ref[(x0[mu] + k) % N]));
                                }
                                maxDev = std::max(maxDev, dev / scale);
                                nChecked++;
                            }
                        }
                        const std::string head = physics(param.mres_mass()) + "\t" + src + "\t" + dirLabel[mu] + "\t"
                                                 + std::to_string(mu) + "\t" + std::to_string(n[0]) + "\t"
                                                 + std::to_string(n[1]) + "\t" + std::to_string(n[2]) + "\t"
                                                 + std::to_string(n2) + "\t" + momChannels[ch] + "\t";
                        for (size_t k = 0; k < c.size(); k++) {
                            body << head << k << "\t" << mdwfE18(real(c[k])) << "\t" << mdwfE18(imag(c[k])) << "\n";
                            nrows++;
                        }
                    }
                }
            }
            rootLogger.info("mdwfMeasure: screening momentum: ", nrows, " rows (", momChannels.size(),
                            " channels, directions ", param.screening_mom_dirs(), ", n2max ", param.screening_n2max(),
                            "); n = 0 check against screening_spatial: ", nChecked,
                            " correlators, max |dev| / max |C| = ", maxDev);
            if (nChecked > 0 && !(maxDev < 1e-10)) {
                throw std::runtime_error(stdLogger.fatal("mdwfMeasure: screening momentum n = 0 projection differs "
                                                         "from screening_spatial by ", maxDev));
            }
            if (commBase.IamRoot()) {
                const std::string file = prefix + "_screening_mom.tsv";
                std::ofstream out(file + ".part");
                out << "label\tconf\tmass\tLs\tM5\tb5\tc5\tsrc_x\tsrc_y\tsrc_z\tsrc_t\tdirection\tcorr_dir\tnx\tny\tnz"
                       "\tn2\tchannel\tcoord\tre\tim\n"
                    << body.str();
                out.close();
                if (!out || std::rename((file + ".part").c_str(), file.c_str()) != 0) {
                    throw std::runtime_error(stdLogger.fatal("mdwfMeasure cannot write ", file));
                }
            }
        }
    }

    if (param.pion_dynamics() != 0) {
        const double mass = param.piondyn_mass.isSet() ? param.piondyn_mass() : param.mres_mass();
        std::array<int, 4> x0{};
        const int ext[4] = {static_cast<int>(lat.globLX), static_cast<int>(lat.globLY), static_cast<int>(lat.globLZ),
                            static_cast<int>(lat.globLT)};
        if (param.piondyn_source() == "random") {
            // Not GPT's numpy generator: a SplitMix64 stream from FNV-1a("<piondyn_seed>:<label>").
            uint64_t h = mdwfSeedFromString(param.piondyn_seed() + ":" + label);
            for (int i = 0; i < 4; i++) {
                h = mdwfSplitMix64(h);
                x0[i] = static_cast<int>(h % static_cast<uint64_t>(ext[i]));
            }
        } else {
            std::string raw = param.piondyn_source();   // "x,y,z,t" or GPT's "x.y.z.t" (no spaces: one token)
            std::replace(raw.begin(), raw.end(), ',', ' ');
            std::replace(raw.begin(), raw.end(), '.', ' ');
            std::istringstream in(raw);
            for (int i = 0; i < 4; i++) {
                if (!(in >> x0[i]) || x0[i] < 0 || x0[i] >= ext[i]) {
                    throw std::runtime_error(stdLogger.fatal("mdwfMeasure: piondyn_source must be random or x,y,z,t"
                                                             " inside the lattice, got ", param.piondyn_source()));
                }
            }
        }
        rootLogger.info("mdwfMeasure: pion dynamics, source (", x0[0], ",", x0[1], ",", x0[2], ",", x0[3], ") (",
                        param.piondyn_source(), "), mass ", mass, ", n2max ", param.piondyn_n2max());
        Meas meas(fermionGauge, param.M5(), mass, param.b5(), param.csw(), "MDWF_meas_pd");
        if constexpr (Mixed) {
            meas.solver().setDelta(param.reliable_delta());
        }
        using Spinor5 = MDWFSpinor<double, true, All, HaloDepth, Ls>;
        Spinor5 q0(commBase, "MDWF_pd_q0"), q1(commBase, "MDWF_pd_q1"), q2(commBase, "MDWF_pd_q2"),
            q3(commBase, "MDWF_pd_q3");
        LatticeContainer<true, COMPLEX(double)> acc(commBase, "MDWF_pd_acc"), red(commBase, "MDWF_pd_red");
        const MDWFPionResult r = mdwfMeasurePionDynamics<HaloDepth, Ls>(
            meas, fermionGauge, {&q0, &q1, &q2, &q3}, acc, red, x0, mass, param.b5(), param.piondyn_n2max(),
            param.max_iter(), param.piondyn_precision());
        rootLogger.info("mdwfMeasure: pion dynamics: 12 solves, ", r.iterations, " iterations, max residue ",
                        r.maxResidue, ", ", r.seconds, " s; ", r.rows.size(), " rows; max PCAC deviation ",
                        r.maxPcacDeviation, "; max axial-charge WI deviation ", r.maxPcacDeviationA0,
                        "; max z-PCAC deviation ", r.maxPcacDeviationZ, "; VzcVz conservation ", r.maxVzConservation);
        if (commBase.IamRoot()) {
            const std::string file = prefix + "_piondyn.tsv";
            std::ofstream out(file + ".part");
            if (!out) {
                throw std::runtime_error(stdLogger.fatal("mdwfMeasure cannot write ", file, ".part"));
            }
            out << "label\tconf\tmass\tLs\tM5\tb5\tc5\tsrc_x\tsrc_y\tsrc_z\tsrc_t\tdirection\tn2\tn_vec\tchannel\tcoord"
                   "\tre\tim\n";
            const std::string head = physics(mass) + "\t" + std::to_string(x0[0]) + "\t" + std::to_string(x0[1]) + "\t"
                                     + std::to_string(x0[2]) + "\t" + std::to_string(x0[3]) + "\t";
            for (const MDWFPionRow &row : r.rows) {
                out << head << row.direction << "\t" << row.n2 << "\t" << row.nvec << "\t" << row.channel << "\t"
                    << row.coord << "\t" << mdwfE18(real(row.value)) << "\t" << mdwfE18(imag(row.value)) << "\n";
            }
            out.close();
            if (!out || std::rename((file + ".part").c_str(), file.c_str()) != 0) {
                throw std::runtime_error(stdLogger.fatal("mdwfMeasure cannot write ", file));
            }
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // Low modes and low-mode averaging (MDWFLowModes.h, branch mdwf-lma). Modes of the light operator at mres_mass;
    // they are kept for the low-mode-exact light loops of the condensate block.
    // ------------------------------------------------------------------------------------------------------------
    auto parseDirs = [](const std::string &raw, const char *what) {
        std::vector<int> out;
        for (const char d : raw) {
            const size_t mu = std::string("xyzt").find(d);
            if (mu == std::string::npos) {
                throw std::runtime_error(stdLogger.fatal("mdwfMeasure: ", what, " must be letters of xyzt, got ", raw));
            }
            out.push_back(static_cast<int>(mu));
        }
        return out;
    };
    std::unique_ptr<Meas> lmaMeas;
    std::unique_ptr<MDWFLowModes<HaloDepth, Ls, Solver>> lma;
    std::vector<double> lmaMu;
    std::array<std::array<std::vector<COMPLEX(double)>, 2>, 4> lmaExact;   // [mu][Gamma = 1, g5][n]
    std::vector<std::pair<std::string, double>> lmaChecks;
    if (param.lma() != 0) {
        const auto t0 = std::chrono::steady_clock::now();
        lmaMeas.reset(new Meas(fermionGauge, param.M5(), param.mres_mass(), param.b5(), param.csw(), "MDWF_meas_lma"));
        if constexpr (Mixed) {
            lmaMeas->solver().setDelta(param.reliable_delta());
        }
        MDWFLowModeParameters lp;
        lp.nModes = param.lma_nmodes();
        lp.precision = param.lma_precision();
        lp.tol = param.lma_tol();
        lp.nBasis = param.lma_nbasis();
        lp.nKeep = param.lma_nkeep();
        lp.maxRestarts = param.lma_max_restarts();
        lp.maxIter = param.max_iter();
        lma.reset(new MDWFLowModes<HaloDepth, Ls, Solver>(*lmaMeas, store, param.mres_mass(), lp));
        lma->compute(mdwfSeedFromString(label + "_lma"));
        const int N = lma->size();
        double maxRes = 0.0;
        for (const auto &in : lma->info) {
            lmaMu.push_back(in.mu);
            maxRes = std::max(maxRes, in.residual);
        }
        const double tModes = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        rootLogger.info("mdwfMeasure: LMA: ", N, " modes, ", lma->solves(), " solves, ", lma->iterations(),
                        " iterations, ", tModes, " s, max mode residual ", maxRes);
        lmaChecks.emplace_back("nmodes", N);
        lmaChecks.emplace_back("solves_modes", lma->solves());
        lmaChecks.emplace_back("iterations_modes", lma->iterations());
        lmaChecks.emplace_back("seconds_modes", tModes);
        lmaChecks.emplace_back("max_mode_residual", maxRes);

        const std::array<int, 4> x0 = {param.source[0], param.source[1], param.source[2], param.source[3]};
        const std::vector<std::string> channels = mdwfParseChannels(param.screening_channels());
        const std::vector<int> dirs = parseDirs(param.directions(), "directions");
        const std::vector<std::array<COMPLEX(double), 12>> psiAt = lma->valuesAt(x0);
        std::vector<int> nSub;
        for (int ns : {N / 4, N / 2, N}) {
            if (ns > 0 && (nSub.empty() || ns != nSub.back())) {
                nSub.push_back(ns);
            }
        }
        const double V4 = static_cast<double>(lat.globvol4);
        const char *dirLabel[4] = {"spatial_x", "spatial_y", "spatial_z", "temporal_t"};
        std::ofstream modesOut, lmaOut;
        if (commBase.IamRoot()) {
            modesOut.open(prefix + "_lowmodes.tsv");
            lmaOut.open(prefix + "_lma_screening.tsv");
            if (!modesOut || !lmaOut) {
                throw std::runtime_error(stdLogger.fatal("mdwfMeasure cannot write ", prefix, "_lowmodes/_lma_*.tsv"));
            }
            modesOut << "label\tconf\tmass\tLs\tM5\tb5\tc5\tmode\tmu\tchirality\tlambda\tm_res_mode\tresidual\n";
            for (int k = 0; k < N; k++) {
                const auto &in = lma->info[k];
                modesOut << physics(param.mres_mass()) << "\t" << k << "\t" << mdwfE18(in.mu) << "\t"
                         << mdwfE18(in.chirality) << "\t" << mdwfE18(in.lambda) << "\t" << mdwfE18(in.mresMode) << "\t"
                         << mdwfE18(in.residual) << "\n";
            }
            lmaOut << "label\tconf\tmass\tLs\tM5\tb5\tc5\tsrc_x\tsrc_y\tsrc_z\tsrc_t\tdirection\tcorr_dir\tcoord\tchannel"
                      "\tnmodes\tLL_pt\tLL_a2a\n";
        }
        {   // per-mode z profiles psi^+ psi and psi^+ g5 psi (basis-independent per mode; cross-code check with GPT)
            const std::vector<std::vector<COMPLEX(double)>> dz = lma->spinDensities(2, true);
            const MDWFSpinMatrix one = mdwfSpinIdentity(), g5m = mdwfGammaMatrix("5");
            if (commBase.IamRoot()) {
                std::ofstream prof(prefix + "_lowmode_profiles.tsv");
                prof << "label\tmode\tdir\tcoord\tpsidag_psi\tpsidag_g5_psi\n";
                for (int k = 0; k < N; k++) {
                    const std::vector<COMPLEX(double)> p1 = mdwfLmaBilinear(dz, N, k, k, one);
                    const std::vector<COMPLEX(double)> p5 = mdwfLmaBilinear(dz, N, k, k, g5m);
                    for (size_t n = 0; n < p1.size(); n++) {
                        prof << label << "\t" << k << "\tz\t" << n << "\t" << mdwfE18(real(p1[n])) << "\t"
                             << mdwfE18(real(p5[n])) << "\n";
                    }
                }
            }
        }
        double maxImag = 0.0, selftestPoint = 0.0;
        const std::string src = std::to_string(x0[0]) + "\t" + std::to_string(x0[1]) + "\t" + std::to_string(x0[2])
                                + "\t" + std::to_string(x0[3]);
        for (const int mu : dirs) {
            const std::vector<std::vector<COMPLEX(double)>> d = lma->spinDensities(mu, false);
            std::vector<MDWFLmaChannelResult> res;
            for (const std::string &ch : channels) {
                res.push_back(mdwfLmaChannel(d, N, lmaMu, psiAt, ch, V4, nSub));
                maxImag = std::max(maxImag, res.back().maxImag);
            }
            if (param.lma_selftest() != 0) {
                const std::vector<std::vector<double>> ex = mdwfLmaExplicitPointLL(*lma, store, psiAt, channels, mu);
                for (size_t ch = 0; ch < channels.size(); ch++) {
                    double num = 0.0, den = 0.0;
                    for (size_t n = 0; n < ex[ch].size(); n++) {
                        num = std::max(num, std::abs(ex[ch][n] - res[ch].pointLL.back()[n]));
                        den = std::max(den, std::abs(ex[ch][n]));
                    }
                    selftestPoint = std::max(selftestPoint, num / std::max(den, 1e-300));
                }
            }
            if (commBase.IamRoot()) {
                const std::string head = physics(param.mres_mass()) + "\t" + src + "\t" + dirLabel[mu] + "\t"
                                         + std::to_string(mu) + "\t";
                for (size_t ch = 0; ch < channels.size(); ch++) {
                    for (size_t s = 0; s < nSub.size(); s++) {
                        for (size_t n = 0; n < res[ch].pointLL[s].size(); n++) {
                            lmaOut << head << n << "\t" << channels[ch] << "\t" << nSub[s] << "\t"
                                   << mdwfE18(res[ch].pointLL[s][n]) << "\t" << mdwfE18(res[ch].a2aLL[s][n]) << "\n";
                        }
                    }
                }
            }
        }
        for (const int mu : parseDirs(param.slice_dirs(), "slice_dirs")) {
            lmaExact[mu] = mdwfLmaExactLoops(lma->spinDensities(mu, true), N, lmaMu);
        }
        lmaChecks.emplace_back("max_imag_over_real_LL", maxImag);
        if (param.lma_selftest() != 0) {
            lmaChecks.emplace_back("selftest_pointLL_explicit_vs_formula", selftestPoint);
            rootLogger.info("mdwfMeasure: LMA self-test: point LL explicit vs formula, max relative deviation ",
                            selftestPoint);
        }
        lmaChecks.emplace_back("seconds_lma_total",
                               std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    if (param.condensate() != 0) {
        auto masses = mdwfParseMasses(param.condensate_masses());
        const std::string seedString = param.noise_seed().empty() ? label + "_noise" : param.noise_seed();
        const uint64_t seed = mdwfSeedFromString(seedString);
        const bool connected = param.measure_connected() != 0, pseudo = param.measure_pseudoscalar() != 0;
        const double V4 = static_cast<double>(lat.globvol4);
        int maxNrv = 0;
        for (auto &m : masses) {
            if (m.nrv < 0) {
                m.nrv = param.nrv();
            }
            maxNrv = std::max(maxNrv, m.nrv);
        }
        std::vector<int> sliceDirs;
        for (const char d : param.slice_dirs()) {
            const size_t mu = std::string("xyzt").find(d);
            if (mu == std::string::npos) {
                throw std::runtime_error(stdLogger.fatal("mdwfMeasure: slice_dirs must be letters of xyzt, got ",
                                                         param.slice_dirs()));
            }
            sliceDirs.push_back(static_cast<int>(mu));
        }
        LatticeContainer<true, double> reduction(commBase);
        reduction.adjustSize(lat.vol4);
        std::vector<std::unique_ptr<Meas>> meas;
        for (size_t f = 0; f < masses.size(); f++) {
            meas.emplace_back(new Meas(fermionGauge, param.M5(), masses[f].mass, param.b5(), param.csw(),
                                       "MDWF_meas_f" + std::to_string(f) + "x"));
            if constexpr (Mixed) {
                meas.back()->solver().setDelta(param.reliable_delta());
            }
        }
        const bool sliceMom = param.slice_n2max() > 0 && !sliceDirs.empty();
        std::vector<std::vector<std::array<int, 3>>> sliceMoms(4);
        LatticeContainer<true, COMPLEX(double)> momRed(commBase, "MDWF_sm_loops");
        if (sliceMom) {
            for (const int mu : sliceDirs) {
                if (mu > 2) {
                    throw std::runtime_error(stdLogger.fatal("mdwfMeasure: slice_n2max needs spatial slice_dirs"));
                }
                sliceMoms[mu] = mdwfTransverseMomenta(mu, param.slice_n2max(), false);
            }
            momRed.adjustSize(lat.vol4);
        }
        double maxSliceMomDev = 0.0;
        std::ofstream out, slices, slicesMom;
        if (commBase.IamRoot() && sliceMom) {
            slicesMom.open(prefix + "_condensate_slices_mom.tsv");
            if (!slicesMom) {
                throw std::runtime_error(stdLogger.fatal("mdwfMeasure cannot write ", prefix,
                                                         "_condensate_slices_mom.tsv"));
            }
            slicesMom << "label\tflavor\tmass\trnd_vec_idx\toperator\tdir\tnx\tny\tnz\tcoord\treal\timag\n";
        }
        if (commBase.IamRoot() && !sliceDirs.empty()) {
            slices.open(prefix + "_condensate_slices.tsv");
            if (!slices) {
                throw std::runtime_error(stdLogger.fatal("mdwfMeasure cannot write ", prefix,
                                                         "_condensate_slices.tsv"));
            }
            slices << "label\tflavor\tmass\trnd_vec_idx\toperator\tdir\tcoord\treal\timag\n";
            if (lma) {   // exact low-mode loops once per configuration (rnd_vec_idx -1)
                for (const auto &m : masses) {
                    if (std::abs(m.mass - param.mres_mass()) > 1e-12 * std::max(1.0, std::abs(param.mres_mass()))) {
                        continue;
                    }
                    const std::string sh = label + "\t" + m.name + "\t" + mdwfG16(m.mass) + "\t-1\t";
                    for (const int mu : sliceDirs) {
                        for (int g = 0; g < (pseudo ? 2 : 1); g++) {
                            for (size_t n = 0; n < lmaExact[mu][g].size(); n++) {
                                slices << sh << (g ? "L5_lowexact" : "L_lowexact") << "\t" << "xyzt"[mu] << "\t" << n
                                       << "\t" << mdwfE18(real(lmaExact[mu][g][n])) << "\t"
                                       << mdwfE18(imag(lmaExact[mu][g][n])) << "\n";
                            }
                        }
                    }
                }
            }
        }
        if (commBase.IamRoot()) {
            out.open(prefix + "_condensates.tsv");
            if (!out) {
                throw std::runtime_error(stdLogger.fatal("mdwfMeasure cannot write ", prefix, "_condensates.tsv"));
            }
            out << "label\tconf\tflavor\tmass\tLs\tM5\tb5\tc5\tNsig\tNtau\tV4\tprecision\trng_type\tseed"
                   "\trnd_vec_idx\toperator\treal\timag\n";
        }
        double maxSliceDev = 0.0;
        bool lmaLoopTested = false;
        double lmaLoopTest = 0.0;
        for (int i = param.i_start(); i < param.i_start() + maxNrv; i++) {
            bool noiseWritten = false;
            const bool conn = connected && (param.nrv_connected() < 0 || i < param.i_start() + param.nrv_connected());
            for (size_t f = 0; f < masses.size(); f++) {
                if (i >= param.i_start() + masses[f].nrv) {
                    continue;
                }
                const MDWFCondensateVector v = mdwfMeasureCondensateVector<HaloDepth, Ls>(
                    *meas[f], store, seed, static_cast<uint64_t>(i), !noiseWritten, conn, param.max_iter(),
                    param.condensate_precision());
                noiseWritten = true;
                rootLogger.info("mdwfMeasure: ", masses[f].name, " vector ", i, ": tr S = ", v.raw,
                                conn ? ", tr S^2 = " : "", conn ? v.s2 : COMPLEX(double)(0.0, 0.0), ", ",
                                v.iterations, " iterations, ", v.seconds, " s");
                // Slice-resolved loops from the store (column 0 = eta, column 1 = S eta), as GPT's slice script.
                std::vector<std::pair<std::string, std::vector<COMPLEX(double)>>> sliceRows;
                const bool lmaLoops = lma && std::abs(masses[f].mass - param.mres_mass())
                                                 <= 1e-12 * std::max(1.0, std::abs(param.mres_mass()));
                std::vector<COMPLEX(double)> lmaC;   // c_k = psi_k^+ g5 eta of this vector
                for (const int mu : sliceDirs) {
                    std::array<std::vector<COMPLEX(double)>, 2> low;
                    if (lmaLoops) {
                        low = mdwfLmaNoiseLowPart(*lma, store, 0, mu, lmaC);
                        if (param.lma_selftest() != 0 && !lmaLoopTested) {
                            // S_L eta into store column 3, then the direct slice loops (test L2, loops)
                            for (int k = 0; k < lma->size(); k++) {
                                store.template iterateOverBulk<BLOCKSIZE>(MDWFColumnAxpy<HaloDepth, Ls>(
                                    store, *lma->psi[k], 3, lmaC[k] / lmaMu[k], k == 0));
                            }
                            double num = 0.0, den = 0.0;
                            for (int g = 0; g < 2; g++) {
                                const std::vector<COMPLEX(double)> dir = mdwfColumnInnerSlices<HaloDepth, Ls>(
                                    reduction, store, 0, 3, g == 1, mu);
                                for (size_t n = 0; n < dir.size(); n++) {
                                    num = std::max(num, abs(dir[n] - low[g][n]));
                                    den = std::max(den, abs(dir[n]));
                                }
                            }
                            lmaLoopTest = num / std::max(den, 1e-300);
                            lmaLoopTested = true;
                            rootLogger.info("mdwfMeasure: LMA self-test: noise low part direct vs formula, max "
                                            "relative deviation ", lmaLoopTest);
                        }
                    }
                    for (int g5 = 0; g5 < (pseudo ? 2 : 1); g5++) {
                        std::vector<COMPLEX(double)> sl = mdwfColumnInnerSlices<HaloDepth, Ls>(
                            reduction, store, 0, 1, g5 == 1, mu);
                        COMPLEX(double) sum(0.0, 0.0);
                        for (const auto &z : sl) {
                            sum += z;
                        }
                        const COMPLEX(double) total = g5 ? v.p5 : v.raw;
                        maxSliceDev = std::max(maxSliceDev, abs(sum - total) / std::max(abs(total), 1e-300));
                        if (sliceMom) {
                            // n = 0 through the momentum path must reproduce the slice loops
                            const std::vector<COMPLEX(double)> z0 = mdwfLoopMomentumSlices<HaloDepth, Ls>(
                                momRed, store, 0, 1, g5 == 1, mu, {0, 0, 0});
                            double scale = 1e-300, dev = 0.0;
                            for (size_t k = 0; k < sl.size(); k++) {
                                scale = std::max(scale, abs(sl[k]));
                                dev = std::max(dev, abs(z0[k] - sl[k]));
                            }
                            maxSliceMomDev = std::max(maxSliceMomDev, dev / scale);
                            for (const auto &n : sliceMoms[mu]) {
                                const std::vector<COMPLEX(double)> lm = mdwfLoopMomentumSlices<HaloDepth, Ls>(
                                    momRed, store, 0, 1, g5 == 1, mu, n);
                                if (slicesMom.is_open()) {
                                    const std::string sh = label + "\t" + masses[f].name + "\t"
                                                           + mdwfG16(masses[f].mass) + "\t" + std::to_string(i) + "\t"
                                                           + (g5 ? "L5_raw\t" : "L_raw\t") + "xyzt"[mu] + "\t"
                                                           + std::to_string(n[0]) + "\t" + std::to_string(n[1]) + "\t"
                                                           + std::to_string(n[2]) + "\t";
                                    for (size_t k = 0; k < lm.size(); k++) {
                                        slicesMom << sh << k << "\t" << mdwfE18(real(lm[k])) << "\t"
                                                  << mdwfE18(imag(lm[k])) << "\n";
                                    }
                                }
                            }
                        }
                        if (lmaLoops) {
                            std::vector<COMPLEX(double)> lsl(sl.size());
                            for (size_t n = 0; n < sl.size(); n++) {
                                lsl[n] = sl[n] - low[g5][n] + lmaExact[mu][g5][n];
                            }
                            sliceRows.emplace_back(std::string(g5 ? "L5_raw_lma\t" : "L_raw_lma\t") + "xyzt"[mu],
                                                   std::move(lsl));
                        }
                        sliceRows.emplace_back(std::string(g5 ? "L5_raw\t" : "L_raw\t") + "xyzt"[mu], std::move(sl));
                    }
                }
                if (slicesMom.is_open()) {
                    slicesMom.flush();
                }
                if (!commBase.IamRoot()) {
                    continue;
                }
                if (slices.is_open()) {
                    const std::string sh = label + "\t" + masses[f].name + "\t" + mdwfG16(masses[f].mass) + "\t"
                                           + std::to_string(i) + "\t";
                    auto all = [&](const char *op, COMPLEX(double) z) {
                        slices << sh << op << "\tall\t-1\t" << mdwfE18(real(z)) << "\t" << mdwfE18(imag(z)) << "\n";
                    };
                    all("I_id", v.id);
                    all("I_raw", v.raw);
                    if (pseudo) {
                        all("P5_raw", v.p5);
                    }
                    for (const auto &rowSet : sliceRows) {
                        for (size_t n = 0; n < rowSet.second.size(); n++) {
                            slices << sh << rowSet.first << "\t" << n << "\t" << mdwfE18(real(rowSet.second[n]))
                                   << "\t" << mdwfE18(imag(rowSet.second[n])) << "\n";
                        }
                    }
                    slices.flush();
                }
                const std::string head = label + "\t" + conf + "\t" + masses[f].name + "\t"
                                         + mdwfG16(masses[f].mass) + "\t" + std::to_string(Ls) + "\t"
                                         + mdwfG16(param.M5()) + "\t" + mdwfG16(param.b5()) + "\t" + mdwfG16(c5)
                                         + "\t" + std::to_string(lat.globLX) + "\t" + std::to_string(lat.globLT)
                                         + "\t" + std::to_string(lat.globvol4) + "\t"
                                         + (Mixed ? "mixed" : "double") + "\tz2hash\t" + seedString + "\t"
                                         + std::to_string(i) + "\t";
                auto row = [&](const char *op, COMPLEX(double) z) {
                    out << head << op << "\t" << mdwfE18(real(z)) << "\t" << mdwfE18(imag(z)) << "\n";
                };
                row("I_id", v.id);
                row("I_raw", v.raw);
                row("chiral_condensate_bare_per_V", v.raw / V4);
                if (pseudo) {
                    row("P5_raw", v.p5);
                    row("pseudoscalar_loop_per_V", v.p5 / V4);
                }
                if (conn) {
                    row("S2_raw", v.s2);
                    row("chi_conn_plus_per_V", v.s2 / V4);
                    row("chi_conn_minus_per_V", -1.0 * v.s2 / V4);
                }
                out.flush();
            }
        }
        if (lma && param.lma_selftest() != 0 && lmaLoopTested) {
            lmaChecks.emplace_back("selftest_noise_lowpart_direct_vs_formula", lmaLoopTest);
        }
        if (!sliceDirs.empty()) {
            rootLogger.info("mdwfMeasure: slice loops along ", param.slice_dirs(),
                            ", max relative deviation of the slice sums from the 4D traces ", maxSliceDev);
        }
        if (sliceMom) {
            rootLogger.info("mdwfMeasure: slice loops at transverse momenta |n|^2 <= ", param.slice_n2max(),
                            " (", sliceMoms[sliceDirs[0]].size(), " momenta per direction); n = 0 check against the"
                            " slice loops: max |dev| / max |L| = ", maxSliceMomDev);
            if (!(maxSliceMomDev < 1e-10)) {
                throw std::runtime_error(stdLogger.fatal("mdwfMeasure: momentum loops at n = 0 differ from the slice "
                                                         "loops by ", maxSliceMomDev));
            }
        }
    }
    if (lma && commBase.IamRoot()) {
        std::ofstream chk(prefix + "_lma_checks.tsv");
        chk << "label\tkey\tvalue\n";
        for (const auto &kv : lmaChecks) {
            chk << label << "\t" << kv.first << "\t" << mdwfE18(kv.second) << "\n";
        }
    }
    rootLogger.info("mdwfMeasure: done");
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);
        MDWFMeasureParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfMeasure.param", argc, argv);
        commBase.forceHalos(param.force_halos() != 0);
        commBase.init(param.nodeDim());
        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);

        const bool mixed = param.mixed_precision() != 0;
        if (param.ls() == 8) {
            mixed ? runMDWFMeasure<8, true>(commBase, param) : runMDWFMeasure<8, false>(commBase, param);
        } else if (param.ls() == 12) {
            mixed ? runMDWFMeasure<12, true>(commBase, param) : runMDWFMeasure<12, false>(commBase, param);
        } else if (param.ls() == 16) {
            mixed ? runMDWFMeasure<16, true>(commBase, param) : runMDWFMeasure<16, false>(commBase, param);
        } else if (param.ls() == 24) {
            mixed ? runMDWFMeasure<24, true>(commBase, param) : runMDWFMeasure<24, false>(commBase, param);
        } else {
            throw std::runtime_error(stdLogger.fatal("mdwfMeasure supports ls = 8, 12, 16 or 24, got ", param.ls()));
        }
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
