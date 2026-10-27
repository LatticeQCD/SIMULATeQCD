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
 * c5 = b5 - 1 (MDWFMobiusMapping.h). Solver: mixed_precision = 1 (single-precision CG with double reliable updates,
 * same stopping rule on the true residual) or 0 (double). force_halos = 0 (default): no halo buffers on one GPU.
 *
 * Usage (from the build's testing directory):  ./mdwfMeasure <param file> [key=value ...]
 * Parameters: parameter/tests/mdwfMeasure.param. Single rank; ls = 8, 16 or 24.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFMeasurements.h"
#include "../experimental/mdwf/MDWFMixedPrecisionSolver.h"
#include "../gauge/gaugeAction.h"

#include <algorithm>
#include <array>
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
        const MDWFPointMeasurementResult r = mdwfMeasurePointSource<HaloDepth, Ls>(
            meas, store, x0, channels, param.max_iter(), param.screening_precision());
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
        std::ofstream out, slices;
        if (commBase.IamRoot() && !sliceDirs.empty()) {
            slices.open(prefix + "_condensate_slices.tsv");
            if (!slices) {
                throw std::runtime_error(stdLogger.fatal("mdwfMeasure cannot write ", prefix,
                                                         "_condensate_slices.tsv"));
            }
            slices << "label\tflavor\tmass\trnd_vec_idx\toperator\tdir\tcoord\treal\timag\n";
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
                for (const int mu : sliceDirs) {
                    for (int g5 = 0; g5 < (pseudo ? 2 : 1); g5++) {
                        std::vector<COMPLEX(double)> sl = mdwfColumnInnerSlices<HaloDepth, Ls>(
                            reduction, store, 0, 1, g5 == 1, mu);
                        COMPLEX(double) sum(0.0, 0.0);
                        for (const auto &z : sl) {
                            sum += z;
                        }
                        const COMPLEX(double) total = g5 ? v.p5 : v.raw;
                        maxSliceDev = std::max(maxSliceDev, abs(sum - total) / std::max(abs(total), 1e-300));
                        sliceRows.emplace_back(std::string(g5 ? "L5_raw\t" : "L_raw\t") + "xyzt"[mu], std::move(sl));
                    }
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
        if (!sliceDirs.empty()) {
            rootLogger.info("mdwfMeasure: slice loops along ", param.slice_dirs(),
                            ", max relative deviation of the slice sums from the 4D traces ", maxSliceDev);
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
        } else if (param.ls() == 16) {
            mixed ? runMDWFMeasure<16, true>(commBase, param) : runMDWFMeasure<16, false>(commBase, param);
        } else if (param.ls() == 24) {
            mixed ? runMDWFMeasure<24, true>(commBase, param) : runMDWFMeasure<24, false>(commBase, param);
        } else {
            throw std::runtime_error(stdLogger.fatal("mdwfMeasure supports ls = 8, 16 or 24, got ", param.ls()));
        }
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
