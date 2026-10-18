/*
 * MDWF 2+1 flavour HMC run (step 5 of the MDWF RHMC plan: <exp(-Delta H)> = 1
 * over Metropolis trajectories with the one-flavour RHMC strange quark).
 *
 * Runs MDWFTwoPlusOneHmc (Wilson gauge action, or tree-level Symanzik with
 * symanzik_gauge = 1, as S_g = -(3 beta/5) symanzik(); Mobius clover Pauli-Villars
 * two-flavour light pseudofermion at mf and one-flavour Pauli-Villars RHMC
 * strange pseudofermion at ms, sampling
 * det(M_l^\dagger M_l / M_1^\dagger M_1) det(M_s^\dagger M_s / M_1^\dagger M_1)^(1/2))
 * with the Sexton-Weingarten two-scale leapfrog for n_traj trajectories with
 * Metropolis accept/reject. Every trajectory appends one line to
 * <measurements_dir>/<output_name>:
 *
 *   traj  accepted  deltaH  exp(-deltaH)  plaquette  kinetic_before  gauge_before
 *   fermion_before  fermion_forces  seconds
 *
 * The strange-quark approximations are AlgRemez approximations on the
 * intervals given in the parameter file (lambda_low_s/high_s for
 * M_s^\dagger M_s, lambda_low_pv/high_pv for M_1^\dagger M_1). With
 * check_steps > 0, Lanczos (check_steps steps) estimates both spectral ranges
 * on the start configuration, and the run stops if an estimate lies outside
 * its interval; the same estimate on the final configuration is reported.
 * Lanczos approaches lambda_min from above and lambda_max from below, so keep
 * a margin.
 *
 * The first n_nometro (<= n_therm) trajectories are always accepted; the
 * trajectories after the first n_therm give the acceptance rate,
 * <exp(-Delta H)> (naive and blocked errors), <Delta H> versus <Delta H^2>/2,
 * and the mean plaquette. Correct sampling requires <exp(-Delta H)> = 1
 * within errors. With Gaugefile_out set, the configuration is written into
 * measurements_dir every save_every trajectories (if > 0) and at the end.
 * With max_seconds > 0 the run starts no further trajectory once the elapsed
 * wall-clock time plus 1.25 times the longest trajectory so far would exceed
 * max_seconds, then writes the summary and configuration as usual, so a
 * chained batch segment ends cleanly before its time limit.
 * Ensemble streams: traj_offset = n numbers the trajectories n+1, n+2, ... (a chained segment continuing a stream;
 * the output file is then appended to instead of overwritten), and save_series = 1 writes every saved
 * configuration to its own file <Gaugefile_out>.<trajectory> instead of overwriting Gaugefile_out.
 *
 * Fermion boundary conditions in time: antiperiodic_t = 1 gives antiperiodic
 * temporal BCs (required for finite-temperature ensembles; MDWFHmc.h and
 * MDWFFermionBoundary.h), default 0 = periodic (as all runs before it). The
 * Lanczos interval checks then measure the operator with those BCs.
 *
 * Hasenbusch: with even_odd = 1, hasenbusch_masses = m_1 ... m_k (strictly between mf and pv_mass) splits the light
 * pair into k + 1 ratio factors (MDWFEvenOddHasenbuschTwoFlavorFermionAction), each with its own pseudofermion;
 * Multi-level integrator: term_level = one level per fermion force term (Hasenbusch factors lightest first, then the
 * strange RHMC; 0 = coarsest, `fermion_steps` steps), level_substeps = steps per step of the level above for levels
 * 1, 2, ...; the gauge force sits below the finest level with gauge_substeps. Unset = the two-scale integrator.
 * md_scheme / gauge_scheme: 0 leapfrog, 1 OMF2 (omf2_lambda), 2 OMF2 force-gradient (GPT's OMF2_force_gradient) per
 * step of the fermion levels / gauge level (MDWFHmcParameters). MD time: t_GPT = sqrt(2) t here (GPT pi = c_a T_a with
 * c_a ~ N(0,1); here P = p_a lambda_a with p_a ~ N(0,1/2)), so GPT tau = 1 is tau = 0.70710678.
 *
 * With even_odd = 1 the run uses MDWFEvenOddTwoPlusOneHmc (EVEN_ODD_DESIGN.md):
 * even-site pseudofermions on the Schur complement Mhat^+ Mhat, plus the
 * log det M_oo ratio term for c_sw != 0. The strange-quark intervals then refer
 * to Mhat^+ Mhat (a different, better conditioned spectrum than M^+ M), and the
 * Lanczos checks measure Mhat^+ Mhat.
 *
 * DSDR: with even_odd = 1, dsdr_twists = eps_0 < eps_1 < ... < eps_k multiplies the weight by the DSDR ladder
 * prod_i det(D_W^+ D_W + eps_i^2) / det(D_W^+ D_W + eps_(i+1)^2), D_W = D_W(-dsdr_M5) the 4D Wilson operator
 * (MDWFDSDRFermionActions.h; GPT's tm_ratios eps_i:eps_(i+1)), dsdr_M5 <= 0 meaning M5. Its k force terms come after
 * the Hasenbusch factors and the strange RHMC in term_level.
 *
 * Reversibility: reversibility_check = n > 0 runs, instead of the trajectories, n checks on the start configuration
 * (fresh momenta and pseudofermions each; integrate tau, flip P, integrate tau, flip P) with the full action and
 * solver settings (mixed precision, chronological guesses), prints max |U_back - U_0|, max |P_back - P_0| (infnorm
 * per link) and H_back - H_0, and exits without writing output.
 *
 * conf_double = 1 writes Gaugefile_out with full 3x3 links in double precision (default: 2 rows, single).
 *
 * Gauge action: rectangle_gauge = 1 uses S_g = -(beta/3) [(1 - 8 c1) sum Re tr U_p + c1 sum Re tr U_rect] with
 * c1 = gauge_c1 (Iwasaki -0.331, GPT's normalization of beta) instead of symanzik_gauge / Wilson.
 *
 * Usage (from the build's testing directory):
 *   ./mdwfTwoPlusOneHmcRun <param file> [key=value ...]
 * Parameters: see parameter/tests/mdwfTwoPlusOneHmcRun.param. Ls is a compile-time
 * template: Ls = 8 (default) for all paths, Ls = 16 for the DSDR path only. Single rank; test-only scaffold, not a production RHMC.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFDSDRFermionActions.h"
#include "../experimental/mdwf/MDWFEofaFermionAction.h"
#include "../experimental/mdwf/MDWFEvenOddFermionActions.h"
#include "../experimental/mdwf/MDWFRhmcFermionActions.h"
#include "../experimental/mdwf/MDWFSpectralBounds.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

class MDWFTwoPlusOneRunParameters : public LatticeParameters {
public:
    Parameter<double> M5;
    Parameter<double> mf;
    Parameter<double> ms;
    Parameter<double> b5;
    Parameter<double> csw;
    Parameter<double> pv_mass;
    Parameter<double> lambda_low_s;
    Parameter<double> lambda_high_s;
    Parameter<double> lambda_low_pv;
    Parameter<double> lambda_high_pv;
    Parameter<double> action_error;
    Parameter<double> force_error;
    Parameter<int> max_order;
    Parameter<int> remez_digits;
    Parameter<int> strange_eofa;          // 1: EOFA strange action (GPT exact_one_flavor_ratio) instead of the RHMC
    Parameter<double> eofa_lambda_low;    // EOFA heatbath: x^(-1/2) approximation interval and error
    Parameter<double> eofa_lambda_high;
    Parameter<double> eofa_error;
    Parameter<int> eofa_check_heatbath;
    Parameter<int> force_halos;   // 1: halo buffers and exchanges even on one rank (old behaviour); 0: periodic wrap
    Parameter<int> check_steps;
    Parameter<int> symanzik_gauge;
    Parameter<int> rectangle_gauge;
    Parameter<double> gauge_c1;
    DynamicParameter<double> dsdr_twists;
    Parameter<double> dsdr_M5;
    Parameter<int> antiperiodic_t;
    Parameter<int> Ls;
    Parameter<int> mixed_precision;
    Parameter<double> mixed_delta;
    Parameter<double> force_precision;
    Parameter<int> chrono_size;
    DynamicParameter<double> hasenbusch_masses;
    Parameter<int> light_form;            // light ratio factors: 1 = GPT form (default), 0 = original SQ form
    DynamicParameter<int> term_level;
    DynamicParameter<int> level_substeps;
    Parameter<int> md_scheme;
    Parameter<int> gauge_scheme;
    Parameter<double> omf2_lambda;
    Parameter<int> even_odd;
    Parameter<double> tau;
    Parameter<int> fermion_steps;
    Parameter<int> gauge_substeps;
    Parameter<int> n_traj;
    Parameter<int> n_therm;
    Parameter<int> n_nometro;
    Parameter<int> seed;
    Parameter<int> max_iter;
    Parameter<double> precision;
    Parameter<int> block_size;
    Parameter<int> save_every;
    Parameter<int> conf_double;
    Parameter<double> max_seconds;
    Parameter<int> traj_offset;
    Parameter<int> save_series;
    Parameter<int> reversibility_check;
    Parameter<std::string> output_name;

    MDWFTwoPlusOneRunParameters() {
        addDefault(M5, "M5", 1.8);
        addDefault(mf, "mf", 0.1);
        addDefault(ms, "ms", 0.2);
        addDefault(b5, "b5", 1.5);
        addDefault(csw, "c_sw", 0.5);
        addDefault(pv_mass, "pv_mass", 1.0);
        addDefault(lambda_low_s, "lambda_low_s", 0.04);
        addDefault(lambda_high_s, "lambda_high_s", 180.0);
        addDefault(lambda_low_pv, "lambda_low_pv", 0.2);
        addDefault(lambda_high_pv, "lambda_high_pv", 180.0);
        addDefault(action_error, "action_error", 1e-12);
        addDefault(force_error, "force_error", 1e-8);
        addDefault(max_order, "max_order", 30);
        addDefault(remez_digits, "remez_digits", 50);
        addDefault(strange_eofa, "strange_eofa", 0);
        addDefault(eofa_lambda_low, "eofa_lambda_low", 1.0);
        addDefault(eofa_lambda_high, "eofa_lambda_high", 12.0);
        addDefault(eofa_error, "eofa_error", 1e-10);
        addDefault(eofa_check_heatbath, "eofa_check_heatbath", 1);
        addDefault(force_halos, "force_halos", 1);
        addDefault(check_steps, "check_steps", 500);
        addDefault(symanzik_gauge, "symanzik_gauge", 0);
        addDefault(rectangle_gauge, "rectangle_gauge", 0);
        addDefault(gauge_c1, "gauge_c1", -0.331);
        addOptional(dsdr_twists, "dsdr_twists");
        addDefault(dsdr_M5, "dsdr_M5", 0.0);
        addDefault(antiperiodic_t, "antiperiodic_t", 0);
        addDefault(Ls, "Ls", 8);
        addDefault(mixed_precision, "mixed_precision", 0);
        addDefault(mixed_delta, "mixed_delta", 0.1);
        addDefault(force_precision, "force_precision", 0.0);
        addDefault(chrono_size, "chrono_size", 0);
        addOptional(hasenbusch_masses, "hasenbusch_masses");
        addDefault(light_form, "light_form", 1);
        addOptional(term_level, "term_level");
        addOptional(level_substeps, "level_substeps");
        addDefault(md_scheme, "md_scheme", 0);
        addDefault(gauge_scheme, "gauge_scheme", -1);
        addDefault(omf2_lambda, "omf2_lambda", 0.18);
        addDefault(even_odd, "even_odd", 0);
        addDefault(tau, "tau", 0.5);
        addDefault(fermion_steps, "fermion_steps", 20);
        addDefault(gauge_substeps, "gauge_substeps", 8);
        addDefault(n_traj, "n_traj", 40);
        addDefault(n_therm, "n_therm", 5);
        addDefault(n_nometro, "n_nometro", 0);
        addDefault(seed, "seed", 20260928);
        addDefault(max_iter, "max_iter", 20000);
        addDefault(precision, "precision", 1e-10);
        addDefault(block_size, "block_size", 5);
        addDefault(save_every, "save_every", 5);
        addDefault(conf_double, "conf_double", 0);
        addDefault(max_seconds, "max_seconds", 0.0);
        addDefault(traj_offset, "traj_offset", 0);
        addDefault(save_series, "save_series", 0);
        addDefault(reversibility_check, "reversibility_check", 0);
        addDefault(output_name, "output_name", std::string("mdwfTwoPlusOneHmcRun.dat"));
    }
};

struct MDWFTwoPlusOneRunStatistic {
    double mean;
    double naiveError;
    double blockedError;
};

MDWFTwoPlusOneRunStatistic mdwfTwoPlusOneRunStatistic(const std::vector<double> &values, int blockSize) {
    MDWFTwoPlusOneRunStatistic stat{0.0, 0.0, 0.0};
    const size_t n = values.size();
    if (n == 0) {
        return stat;
    }
    for (double v : values) {
        stat.mean += v;
    }
    stat.mean /= static_cast<double>(n);
    if (n > 1) {
        double var = 0.0;
        for (double v : values) {
            var += (v - stat.mean) * (v - stat.mean);
        }
        stat.naiveError = std::sqrt(var / static_cast<double>(n - 1) / static_cast<double>(n));
    }
    const size_t b = static_cast<size_t>(std::max(1, blockSize));
    const size_t nBlocks = n / b;
    if (nBlocks > 1) {
        std::vector<double> blockMeans(nBlocks, 0.0);
        for (size_t block = 0; block < nBlocks; block++) {
            for (size_t i = 0; i < b; i++) {
                blockMeans[block] += values[block * b + i];
            }
            blockMeans[block] /= static_cast<double>(b);
        }
        double blockMean = 0.0;
        for (double v : blockMeans) {
            blockMean += v;
        }
        blockMean /= static_cast<double>(nBlocks);
        double var = 0.0;
        for (double v : blockMeans) {
            var += (v - blockMean) * (v - blockMean);
        }
        stat.blockedError = std::sqrt(var / static_cast<double>(nBlocks - 1) / static_cast<double>(nBlocks));
    }
    return stat;
}

// Lanczos estimate of the spectral range of M(mass)^\dagger M(mass); true if it lies inside [low, high].
template<class Adapter, class Spinor>
bool mdwfTwoPlusOneRunLanczosReport(Adapter &adapter, Spinor &start, double mass, double low, double high, int steps,
                                    uint4 *randState, const std::string &label, const std::string &name) {
    start.gauss(randState);
    start.updateAll();
    const MDWFLanczosResult result = mdwfLanczosExtremes(adapter, start, steps, {steps / 2, steps}, name + "_lanczos");
    const MDWFLanczosCheckpoint &last = result.checkpoints.back();
    const bool inside = last.lambda_min >= low && last.lambda_max <= high;
    for (const MDWFLanczosCheckpoint &c : result.checkpoints) {
        rootLogger.info("MDWF 2+1 HMC run spectral check (", label, ", mass = ", mass, "): Lanczos steps = ", c.steps,
                        ", lambda_min = ", c.lambda_min, ", lambda_max = ", c.lambda_max);
    }
    rootLogger.info("MDWF 2+1 HMC run spectral check (", label, ", mass = ", mass, "): interval [", low, ", ", high,
                    "], margins lambda_min / low = ", last.lambda_min / low, ", high / lambda_max = ",
                    high / last.lambda_max, ", inside = ", inside);
    return inside;
}

// Lanczos range of M^+ M (evenOdd = false) or of Mhat^+ Mhat on the even sites (evenOdd = true), the operator the
// strange-quark approximation of the chosen action acts on; true if inside [low, high].
template<size_t HaloDepth, size_t Ls>
bool mdwfTwoPlusOneRunCheckBounds(CommunicationBase &commBase, Gaugefield<double, true, HaloDepth, R18> &gauge,
                                  const MDWFHmcParameters &param, double mass, double low, double high, int steps,
                                  uint4 *randState, const std::string &label, const std::string &name, bool evenOdd) {
    if (evenOdd) {
        using Types = MDWFEvenOddTypes<HaloDepth, Ls>;
        typename Types::EvenOdd eo(gauge, param.M5, mass, param.b5, param.csw, name + "_eo");
        typename Types::NormalOp normal(eo, commBase, name + "_nrm");
        typename Types::Adapter adapter(normal);
        typename Types::SpinorE start(commBase, name + "_starte");
        eo.refresh();
        return mdwfTwoPlusOneRunLanczosReport(adapter, start, mass, low, high, steps, randState,
                                              label + ", Mhat^+Mhat", name);
    }
    using Forward = MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Adjoint = MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Normal = MDWFNormalOperator<Forward, Adjoint>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, Normal>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;

    Forward forward(gauge, param.M5, mass, param.b5, param.csw, name + "_forward");
    Adjoint adjoint(gauge, param.M5, mass, param.b5, param.csw, name + "_adjoint");
    Normal normal(commBase, forward, adjoint, name + "_normal");
    Adapter adapter(normal);
    Spinor start(commBase, name + "_start");
    return mdwfTwoPlusOneRunLanczosReport(adapter, start, mass, low, high, steps, randState, label + ", M^+M", name);
}

// EOFA strange (strange_eofa = 1): light Hasenbusch ladder + EOFA (+ DSDR ladder). Force terms as with the RHMC.
template<size_t HaloDepth, size_t Ls>
using MDWFEvenOddHasenbuschEofaTwoPlusOneFermionAction =
    MDWFSumFermionAction<HaloDepth, MDWFEvenOddHasenbuschTwoFlavorFermionAction<HaloDepth, Ls>,
                         MDWFEofaRatioFermionAction<HaloDepth, Ls>>;

template<size_t HaloDepth, size_t Ls>
using MDWFEvenOddHasenbuschDSDREofaTwoPlusOneHmc =
    MDWFHmcDriver<HaloDepth, Ls,
                  MDWFSumFermionAction<HaloDepth, MDWFEvenOddHasenbuschEofaTwoPlusOneFermionAction<HaloDepth, Ls>,
                                       MDWFDSDRFermionAction<HaloDepth>>>;

// GPT's sector labels for the force terms (Hasenbusch lightest first, strange, DSDR): mdwf_<m>_<m'>, eofa_ or
// rhmc_<ms>_<pv>, tm_<eps>_<eps'> (python's :g formatting).
inline std::vector<std::string> mdwfTwoPlusOneRunSectorLabels(const MDWFHmcParameters &param, bool eofa) {
    auto g = [](double x) {
        std::ostringstream s;
        s << x;
        return s.str();
    };
    std::vector<std::string> labels;
    std::vector<double> m{param.mf};
    m.insert(m.end(), param.hasenbusch_masses.begin(), param.hasenbusch_masses.end());
    m.push_back(param.pv_mass);
    for (size_t i = 0; i + 1 < m.size(); i++) {
        labels.push_back("mdwf_" + g(m[i]) + "_" + g(m[i + 1]));
    }
    labels.push_back((eofa ? "eofa_" : "rhmc_") + g(param.rhmc.ms) + "_" + g(param.pv_mass));
    for (size_t i = 0; i + 1 < param.dsdr_twists.size(); i++) {
        labels.push_back("tm_" + g(param.dsdr_twists[i]) + "_" + g(param.dsdr_twists[i + 1]));
    }
    return labels;
}

// The strange action of the fermion action (the second summand, or the first summand's second with DSDR).
template<class Fermion>
auto &mdwfTwoPlusOneRunStrange(Fermion &fermion) {
    if constexpr (std::is_same<std::decay_t<decltype(fermion.second())>, MDWFDSDRFermionAction<2>>::value) {
        return fermion.first().second();
    } else {
        return fermion.second();
    }
}

template<size_t Ls, class Hmc, bool EvenOdd>
void runMDWFTwoPlusOneHmcRun(CommunicationBase &commBase, MDWFTwoPlusOneRunParameters &runParam) {
    const size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    const auto runStart = std::chrono::steady_clock::now();

    if (!runParam.beta.isSet()) {
        throw std::runtime_error(stdLogger.fatal("MDWF 2+1 HMC run requires beta in the parameter file"));
    }

    MDWFHmcParameters param{};
    param.beta = runParam.beta();
    param.M5 = runParam.M5();
    param.mf = runParam.mf();
    param.b5 = runParam.b5();
    param.csw = runParam.csw();
    {
        const LatticeData lat = GIndexer<All, HaloDepth>::getLatData();
        if (param.csw != 0.0 && lat.vol4 != lat.globvol4) {
            throw std::runtime_error(stdLogger.fatal("MDWF 2+1 HMC run: c_sw != 0 (clover all-link force) is single-rank "
                                                     "only; use c_sw = 0 on several ranks"));
        }
    }
    param.pv_mass = runParam.pv_mass();
    param.tau = runParam.tau();
    param.steps = runParam.fermion_steps();
    param.gauge_substeps = runParam.gauge_substeps();
    param.max_iter = runParam.max_iter();
    param.precision = runParam.precision();
    param.mixed_precision = runParam.mixed_precision() != 0;
    param.mixed_delta = runParam.mixed_delta();
    param.force_precision = runParam.force_precision();
    param.chrono_size = runParam.chrono_size();
    if (param.chrono_size > 0 && !param.mixed_precision) {
        throw std::runtime_error(stdLogger.fatal("chrono_size > 0 needs mixed_precision = 1"));
    }
    param.symanzik_gauge = runParam.symanzik_gauge() != 0;
    param.rectangle_gauge = runParam.rectangle_gauge() != 0;
    param.gauge_c1 = runParam.gauge_c1();
    if (param.symanzik_gauge && param.rectangle_gauge) {
        throw std::runtime_error(stdLogger.fatal("MDWF 2+1 HMC run: set symanzik_gauge or rectangle_gauge, not both"));
    }
    if (runParam.dsdr_twists.isSet()) {
        param.dsdr_twists = runParam.dsdr_twists.get();
    }
    param.dsdr_M5 = runParam.dsdr_M5();
    param.antiperiodic_t = runParam.antiperiodic_t() != 0;
    param.light_form = runParam.light_form();
    if (runParam.hasenbusch_masses.isSet()) {
        param.hasenbusch_masses = runParam.hasenbusch_masses.get();
    }
    if (runParam.term_level.isSet()) {
        param.term_level = runParam.term_level.get();
    }
    if (runParam.level_substeps.isSet()) {
        param.level_substeps = runParam.level_substeps.get();
    }
    param.md_scheme = runParam.md_scheme();
    param.gauge_scheme = runParam.gauge_scheme();
    param.omf2_lambda = runParam.omf2_lambda();
    param.rhmc.ms = runParam.ms();
    param.rhmc.lambda_low_s = runParam.lambda_low_s();
    param.rhmc.lambda_high_s = runParam.lambda_high_s();
    param.rhmc.lambda_low_pv = runParam.lambda_low_pv();
    param.rhmc.lambda_high_pv = runParam.lambda_high_pv();
    param.rhmc.action_error = runParam.action_error();
    param.rhmc.force_error = runParam.force_error();
    param.rhmc.max_order = runParam.max_order();
    param.rhmc.digits = runParam.remez_digits();
    const bool eofa = runParam.strange_eofa() != 0;
    param.eofa.lambda_low = runParam.eofa_lambda_low();
    param.eofa.lambda_high = runParam.eofa_lambda_high();
    param.eofa.error = runParam.eofa_error();
    param.eofa.check_heatbath = runParam.eofa_check_heatbath() != 0;
    param.eofa.max_order = runParam.max_order();
    param.eofa.digits = runParam.remez_digits();

    const int nTraj = runParam.n_traj();
    const int nTherm = runParam.n_therm();
    const int nNoMetro = runParam.n_nometro();
    const int checkSteps = runParam.check_steps();
    const int saveEvery = runParam.save_every();
    if (nNoMetro > nTherm) {
        throw std::runtime_error(stdLogger.fatal("MDWF 2+1 HMC run requires n_nometro <= n_therm, so that trajectories "
                                                 "without accept/reject never enter the statistics"));
    }
    const std::string outputPath = runParam.measurements_dir() + "/" + runParam.output_name();
    const bool saveConf = runParam.GaugefileName_out.isSet();
    const std::string confPath = saveConf ? runParam.measurements_dir() + "/" + runParam.GaugefileName_out() : "";
    // conf_double = 1: full 3x3 links in double precision (read back unchanged, e.g. by GPT); default 2 rows, single.
    const int confRows = runParam.conf_double() != 0 ? 3 : 2;
    const int confPrec = runParam.conf_double() != 0 ? 2 : 1;
    // writeconf_nersc only logs a failed write; a segment that cannot save has no restart point, so stop instead.
    // The probe catches a bad path (e.g. an absolute Gaugefile_out, which is relative to measurements_dir) at start.
    auto requireFile = [&](const std::string &path, const char *what) {
        commBase.globalBarrier();
        if (!std::ifstream(path).good()) {
            throw std::runtime_error(stdLogger.fatal("MDWF 2+1 HMC run: ", what, " ", path, " was not written "
                                                     "(Gaugefile_out is relative to measurements_dir)"));
        }
    };
    auto writeConf = [&](Gauge &field, const std::string &path) {
        field.writeconf_nersc(path, confRows, confPrec);
        requireFile(path, "configuration");
    };
    if (saveConf) {
        const std::string probe = confPath + ".writeprobe";
        if (commBase.IamRoot()) { std::ofstream(probe) << "probe\n"; }
        requireFile(probe, "write probe");
        commBase.globalBarrier();
        if (commBase.IamRoot()) { std::remove(probe.c_str()); }
    }

    if (!param.hasenbusch_masses.empty()) {
        std::ostringstream ladder;
        ladder << param.mf;
        for (const double m : param.hasenbusch_masses) {
            ladder << " < " << m;
        }
        ladder << " < " << param.pv_mass;
        rootLogger.info("MDWF 2+1 HMC run: light Hasenbusch ladder ", ladder.str(), " (",
                        param.hasenbusch_masses.size() + 1, " ratio factors), form ",
                        param.light_form == 1 ? "GPT phi^+ B^+ (A A^+)^-1 B phi" : "SIMULATeQCD phi^+ B (A^+ A)^-1 B^+ phi");
    }
    if (!param.term_level.empty()) {
        std::ostringstream levels;
        for (const int l : param.term_level) {
            levels << l << " ";
        }
        levels << "| level substeps ";
        for (const int n : param.level_substeps) {
            levels << n << " ";
        }
        rootLogger.info("MDWF 2+1 HMC run: multi-level integrator, term levels ", levels.str(), "| gauge substeps ",
                        param.gauge_substeps);
    }
    {
        const char *schemeName[] = {"leapfrog", "OMF2", "OMF2 force-gradient"};
        const int gs = param.gauge_scheme < 0 ? param.md_scheme : param.gauge_scheme;
        if (param.md_scheme < 0 || param.md_scheme > 2 || gs < 0 || gs > 2) {
            throw std::runtime_error(stdLogger.fatal("MDWF 2+1 HMC run: md_scheme / gauge_scheme must be 0, 1 or 2"));
        }
        rootLogger.info("MDWF 2+1 HMC run: integrator scheme ", schemeName[param.md_scheme], " (fermion levels), ",
                        schemeName[gs], " (gauge level), OMF2 lambda = ", param.omf2_lambda,
                        "; tau in GPT units = sqrt(2) tau");
    }
    if (!param.dsdr_twists.empty()) {
        std::ostringstream twists;
        for (size_t i = 0; i < param.dsdr_twists.size(); i++) {
            twists << (i ? " < " : "") << param.dsdr_twists[i];
        }
        rootLogger.info("MDWF 2+1 HMC run: DSDR ladder, twists ", twists.str(), " (", param.dsdr_twists.size() - 1,
                        " factors), D_W(-M5) with M5 = ", param.dsdr_M5 > 0.0 ? param.dsdr_M5 : param.M5);
    }
    rootLogger.info("MDWF 2+1 HMC run: ", EvenOdd ? "even/odd preconditioned (Mhat^+ Mhat, even-site pseudofermions)"
                                                  : "unpreconditioned (M^+ M)", " actions");
    const std::string gaugeName = param.rectangle_gauge
                                      ? "plaquette + rectangle (c1 = " + std::to_string(param.gauge_c1) + ")"
                                      : (param.symanzik_gauge ? "Symanzik" : "Wilson");
    rootLogger.info("MDWF 2+1 HMC run: ", gaugeName, " beta = ", param.beta, ", M5 = ", param.M5, ", mf = ", param.mf,
                    ", ms = ", param.rhmc.ms, ", pv_mass = ", param.pv_mass, ", b5 = ", param.b5,
                    ", c5 = ", param.b5 - 1.0, ", c_sw = ", param.csw, ", Ls = ", Ls, ", tau = ", param.tau,
                    ", fermion steps = ", param.steps, ", gauge substeps = ", param.gauge_substeps,
                    ", n_traj = ", nTraj, ", n_therm = ", nTherm, ", n_nometro = ", nNoMetro,
                    ", seed = ", runParam.seed(), ", solver precision = ", param.precision,
                    ", fermion BCs in time: ", param.antiperiodic_t ? "antiperiodic" : "periodic",
                    ", output = ", outputPath);
    if (eofa) {
        rootLogger.info("MDWF 2+1 HMC run strange: EOFA ratio det M(ms) / det M(pv_mass) (GPT exact_one_flavor_ratio), "
                        "heatbath x^(-1/2) on [", param.eofa.lambda_low, ", ", param.eofa.lambda_high, "], error ",
                        param.eofa.error, ", heatbath check ", param.eofa.check_heatbath);
    } else {
        rootLogger.info("MDWF 2+1 HMC run strange approximations: M_s^dagger M_s interval [", param.rhmc.lambda_low_s,
                        ", ", param.rhmc.lambda_high_s, "], M_1^dagger M_1 interval [", param.rhmc.lambda_low_pv, ", ",
                        param.rhmc.lambda_high_pv, "], action error ", param.rhmc.action_error, ", force error ",
                        param.rhmc.force_error, ", max order ", param.rhmc.max_order, ", ", param.rhmc.digits,
                        " digits");
    }

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(runParam.seed());
    d_rand = h_rand;
    std::mt19937_64 acceptRng(static_cast<unsigned long>(runParam.seed()) + 1UL);

    Gauge gauge(commBase, "MDWF_2p1_run_gauge");
    if (runParam.GaugefileName.isSet()) {
        rootLogger.info("MDWF 2+1 HMC run: reading start configuration ", runParam.GaugefileName());
        gauge.readconf_nersc(runParam.GaugefileName());
    } else {
        rootLogger.info("MDWF 2+1 HMC run: cold (unit) start");
        gauge.one();
    }
    gauge.updateAll();

    // Lanczos checks on the operator the fermion action uses (with the fermion BCs).
    MDWFFermionGauge<HaloDepth> checkGauge(commBase, gauge, param.antiperiodic_t, "MDWF_2p1_run_bc_links");
    if (checkSteps > 0 && !eofa) {
        const bool insideS = mdwfTwoPlusOneRunCheckBounds<HaloDepth, Ls>(
            commBase, checkGauge.get(), param, param.rhmc.ms, param.rhmc.lambda_low_s, param.rhmc.lambda_high_s, checkSteps,
            d_rand.state, "start", "MDWF_2p1_run_check_start_s", EvenOdd);
        const bool insidePv = mdwfTwoPlusOneRunCheckBounds<HaloDepth, Ls>(
            commBase, checkGauge.get(), param, param.pv_mass, param.rhmc.lambda_low_pv, param.rhmc.lambda_high_pv, checkSteps,
            d_rand.state, "start", "MDWF_2p1_run_check_start_pv", EvenOdd);
        if (!insideS || !insidePv) {
            throw std::runtime_error(stdLogger.fatal("MDWF 2+1 HMC run: the start configuration's spectrum is outside "
                                                     "the approximation intervals; widen them in the parameter file"));
        }
    }

    Hmc hmc(commBase, gauge, param, d_rand.state);
    const auto &strange = mdwfTwoPlusOneRunStrange(hmc.fermion());
    if constexpr (std::is_same<std::decay_t<decltype(strange)>, MDWFEofaRatioFermionAction<HaloDepth, Ls>>::value) {
        rootLogger.info("MDWF 2+1 HMC run EOFA heatbath approximation order ", strange.inverseSqrt().order,
                        " (error ", strange.inverseSqrt().max_relative_error, ")");
    } else {
        rootLogger.info("MDWF 2+1 HMC run strange approximation orders: A^(1/4) ", strange.quarterS().order,
                        " (error ", strange.quarterS().max_relative_error, "), A^(-1/2) ", strange.halfS().order, " (",
                        strange.halfS().max_relative_error, "), force ", strange.halfSForce().order, " (",
                        strange.halfSForce().max_relative_error, "); B^(1/4) ", strange.quarterPv().order, " (",
                        strange.quarterPv().max_relative_error, "), force ", strange.quarterPvForce().order, " (",
                        strange.quarterPvForce().max_relative_error, ")");
    }

    if (runParam.reversibility_check() > 0) {
        for (int i = 1; i <= runParam.reversibility_check(); i++) {
            const auto startTime = std::chrono::steady_clock::now();
            const MDWFHmcReversibilityResult r = hmc.reversibilityCheck();
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
            rootLogger.info(std::setprecision(12), "MDWF 2+1 HMC run reversibility check ", i, ": H_0 = ",
                            r.start.total(), ", forward Delta H = ", r.forward.total() - r.start.total(),
                            ", H_back - H_0 = ", r.back.total() - r.start.total(), " (kinetic ",
                            r.back.kinetic - r.start.kinetic, ", gauge ", r.back.gauge - r.start.gauge, ", fermion ",
                            r.back.fermion - r.start.fermion, "), max link diff = ", r.max_link_diff,
                            ", max momentum diff = ", r.max_momentum_diff, ", fermion forces = ", r.force_evaluations,
                            ", seconds = ", seconds);
        }
        return;
    }

    std::ofstream out;
    if (commBase.IamRoot()) {
        const bool append = runParam.traj_offset() > 0;
        out.open(outputPath, std::ios::out | (append ? std::ios::app : std::ios::trunc));
        if (!out) {
            throw std::runtime_error(stdLogger.fatal("MDWF 2+1 HMC run cannot open output file ", outputPath));
        }
        if (!append) {
            out << "# traj accepted deltaH exp(-deltaH) plaquette kinetic_before gauge_before fermion_before "
                   "fermion_forces seconds" << std::endl;
        }
        out << std::setprecision(12);
    }

    std::vector<double> deltaH;
    std::vector<double> expMinusDeltaH;
    std::vector<double> accepted;
    std::vector<double> plaquette;

    const double maxSeconds = runParam.max_seconds();
    double longestTrajectory = 0.0;
    int completed = 0;
    for (int traj = 1; traj <= nTraj; traj++) {
        const double forceSecondsBefore = hmc.fermionForceSeconds();
        const double heatbathSecondsBefore = hmc.heatbathSeconds();
        const double actionSecondsBefore = hmc.actionSeconds();
        const std::vector<double> termSecondsBefore = hmc.termForceSeconds();
        const std::vector<double> termNorm2Before = hmc.termForceNorm2Sum();
        const std::vector<double> termNorm2SqBefore = hmc.termForceNorm2SumSq();
        const std::vector<int> termEvalsBefore = hmc.termForceEvaluations();
        const MDWFForceStorageTiming storageBefore = mdwfForceStorageTiming();
        const auto startTime = std::chrono::steady_clock::now();
        // Max over ranks, so that all ranks take the same stop decision (otherwise a deadlock in the next collective).
        const double elapsed = commBase.reduceMax(std::chrono::duration<double>(startTime - runStart).count());
        if (maxSeconds > 0.0 && traj > 1 && elapsed + 1.25 * longestTrajectory > maxSeconds) {
            rootLogger.info("MDWF 2+1 HMC run: stopping before trajectory ", traj, " (elapsed ", elapsed,
                            " s, longest trajectory ", longestTrajectory, " s, max_seconds ", maxSeconds, ")");
            break;
        }
        const MDWFHmcTrajectoryResult result = hmc.trajectory(traj > nNoMetro, acceptRng);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
        longestTrajectory = std::max(longestTrajectory, commBase.reduceMax(seconds));
        completed = traj;
        GaugeAction<double, true, HaloDepth, R18> action(gauge);
        const double plaq = static_cast<double>(action.plaquette());
        const double expMdH = std::exp(-result.delta_h);

        const double forceSeconds = hmc.fermionForceSeconds() - forceSecondsBefore;
        const double storageSeconds = mdwfForceStorageTiming().seconds - storageBefore.seconds;
        const long storageTerms = mdwfForceStorageTiming().terms - storageBefore.terms;
        rootLogger.info("MDWF 2+1 HMC run trajectory ", traj, " cost: fermion forces ", forceSeconds, " s, of which host force "
                        "storage (incl. term vectors) ", storageSeconds, " s for ", storageTerms, " terms, solves and the "
                        "rest ", forceSeconds - storageSeconds, " s; outside the fermion forces (heatbath, actions, gauge "
                        "updates) ", seconds - forceSeconds, " s: heatbath ", hmc.heatbathSeconds() - heatbathSecondsBefore,
                        " s, actions (before + after) ", hmc.actionSeconds() - actionSecondsBefore, " s");
        if (!hmc.termForceSeconds().empty()) {
            std::ostringstream terms;
            for (size_t t = 0; t < hmc.termForceSeconds().size(); t++) {
                const double before = t < termSecondsBefore.size() ? termSecondsBefore[t] : 0.0;
                terms << " " << t << ":" << hmc.termForceSeconds()[t] - before;
            }
            rootLogger.info("MDWF 2+1 HMC run trajectory ", traj, " force seconds per term (Hasenbusch lightest first, "
                            "strange, DSDR):", terms.str());
            // Per sector, in GPT's labels and units (its "<label> <i> force norm2/sites = mean +- std" lines).
            const std::vector<std::string> labels = mdwfTwoPlusOneRunSectorLabels(param, eofa);
            for (size_t t = 0; t < hmc.termForceSeconds().size(); t++) {
                const auto diff = [&](const auto &now, const auto &before) {
                    return static_cast<double>(now[t]) - (t < before.size() ? static_cast<double>(before[t]) : 0.0);
                };
                const double n = diff(hmc.termForceEvaluations(), termEvalsBefore);
                const double mean = n > 0 ? diff(hmc.termForceNorm2Sum(), termNorm2Before) / n : 0.0;
                const double var = n > 0 ? diff(hmc.termForceNorm2SumSq(), termNorm2SqBefore) / n - mean * mean : 0.0;
                rootLogger.info("MDWF 2+1 HMC run trajectory ", traj, " sector ",
                                t < labels.size() ? labels[t] : "term" + std::to_string(t), " force norm2/sites = ",
                                mean, " +- ", std::sqrt(std::max(var, 0.0)), " (", n, " evaluations, ",
                                diff(hmc.termForceSeconds(), termSecondsBefore), " s)");
            }
        }
        rootLogger.info("MDWF 2+1 HMC run trajectory ", traj, ": Delta H = ", result.delta_h,
                        ", exp(-Delta H) = ", expMdH, ", accepted = ", result.accepted,
                        ", plaquette = ", plaq, ", fermion forces = ", result.force_evaluations,
                        ", seconds = ", seconds);
        if (commBase.IamRoot()) {
            out << traj + runParam.traj_offset() << " " << (result.accepted ? 1 : 0) << " " << result.delta_h << " " << expMdH << " "
                << plaq << " " << result.before.kinetic << " " << result.before.gauge << " "
                << result.before.fermion << " " << result.force_evaluations << " " << seconds << std::endl;
        }

        if (traj > nTherm) {
            deltaH.push_back(result.delta_h);
            expMinusDeltaH.push_back(expMdH);
            accepted.push_back(result.accepted ? 1.0 : 0.0);
            plaquette.push_back(plaq);
        }
        if (saveConf && saveEvery > 0 && (traj + runParam.traj_offset()) % saveEvery == 0
            && (traj < nTraj || runParam.save_series() != 0)) {
            const std::string path = runParam.save_series() != 0
                                     ? confPath + "." + std::to_string(traj + runParam.traj_offset()) : confPath;
            rootLogger.info("MDWF 2+1 HMC run: writing configuration after trajectory ",
                            traj + runParam.traj_offset(), " to ", path);
            writeConf(gauge, path);
        }
    }

    std::vector<double> halfDeltaH2(deltaH.size());
    for (size_t i = 0; i < deltaH.size(); i++) {
        halfDeltaH2[i] = 0.5 * deltaH[i] * deltaH[i];
    }
    const int blockSize = runParam.block_size();
    const MDWFTwoPlusOneRunStatistic expStat = mdwfTwoPlusOneRunStatistic(expMinusDeltaH, blockSize);
    const MDWFTwoPlusOneRunStatistic dHStat = mdwfTwoPlusOneRunStatistic(deltaH, blockSize);
    const MDWFTwoPlusOneRunStatistic halfDH2Stat = mdwfTwoPlusOneRunStatistic(halfDeltaH2, blockSize);
    const MDWFTwoPlusOneRunStatistic accStat = mdwfTwoPlusOneRunStatistic(accepted, blockSize);
    const MDWFTwoPlusOneRunStatistic plaqStat = mdwfTwoPlusOneRunStatistic(plaquette, blockSize);
    const double expError = std::max(expStat.naiveError, expStat.blockedError);
    const double expPull = expError > 0.0 ? (expStat.mean - 1.0) / expError : 0.0;

    rootLogger.info("MDWF 2+1 HMC run summary (", completed, " of ", nTraj, " trajectories run; statistics over ",
                    deltaH.size(), " after ", nTherm, " thermalization, block size ", blockSize, "):");
    rootLogger.info("  acceptance = ", accStat.mean, " +- ", accStat.naiveError);
    rootLogger.info("  <exp(-Delta H)> = ", expStat.mean, " +- ", expStat.naiveError, " (naive), +- ",
                    expStat.blockedError, " (blocked); (mean - 1) / error = ", expPull);
    rootLogger.info("  <Delta H> = ", dHStat.mean, " +- ", std::max(dHStat.naiveError, dHStat.blockedError),
                    ", <Delta H^2>/2 = ", halfDH2Stat.mean, " +- ",
                    std::max(halfDH2Stat.naiveError, halfDH2Stat.blockedError));
    rootLogger.info("  <plaquette> = ", plaqStat.mean, " +- ", plaqStat.naiveError, " (naive), +- ",
                    plaqStat.blockedError, " (blocked)");
    rootLogger.info("  total fermion forces = ", hmc.forceEvaluations(), ", gauge updates = ", hmc.gaugeUpdates());
    for (size_t t = 0; t < hmc.termForceEvaluations().size(); t++) {
        const int level = param.term_level.empty() ? 0 : param.term_level.at(t);
        rootLogger.info("  force term ", t, " (level ", level, "): ", hmc.termForceEvaluations()[t],
                        " evaluations, largest rms force ", hmc.termMaxForceRms()[t]);
    }

    if (saveConf) {
        // save_series: the final configuration is numbered too (restart point of the next segment).
        const std::string path = runParam.save_series() != 0
                                 ? confPath + "." + std::to_string(completed + runParam.traj_offset()) : confPath;
        rootLogger.info("MDWF 2+1 HMC run: writing final configuration ", path);
        writeConf(gauge, path);
    }

    if (checkSteps > 0) {
        checkGauge.sync();
        mdwfTwoPlusOneRunCheckBounds<HaloDepth, Ls>(commBase, checkGauge.get(), param, param.rhmc.ms, param.rhmc.lambda_low_s,
                                                    param.rhmc.lambda_high_s, checkSteps, d_rand.state, "final",
                                                    "MDWF_2p1_run_check_final_s", EvenOdd);
        mdwfTwoPlusOneRunCheckBounds<HaloDepth, Ls>(commBase, checkGauge.get(), param, param.pv_mass, param.rhmc.lambda_low_pv,
                                                    param.rhmc.lambda_high_pv, checkSteps, d_rand.state, "final",
                                                    "MDWF_2p1_run_check_final_pv", EvenOdd);
    }
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);

        MDWFTwoPlusOneRunParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfTwoPlusOneHmcRun.param", argc, argv);
        commBase.forceHalos(param.force_halos() != 0);
        commBase.init(param.nodeDim());

        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);

        const bool hasenbusch = param.hasenbusch_masses.isSet() && param.hasenbusch_masses.numberValues() > 0;
        const bool dsdr = param.dsdr_twists.isSet() && param.dsdr_twists.numberValues() > 0;
        if (hasenbusch && param.even_odd() == 0) {
            throw std::runtime_error(stdLogger.fatal("hasenbusch_masses needs even_odd = 1"));
        }
        if (dsdr && param.even_odd() == 0) {
            throw std::runtime_error(stdLogger.fatal("dsdr_twists needs even_odd = 1"));
        }
        if (param.Ls() != 8 && !(dsdr && param.Ls() == 16)) {
            throw std::runtime_error(stdLogger.fatal("Ls = ", param.Ls(), " not instantiated (Ls = 8, or Ls = 16 with dsdr_twists)"));
        }
        if (param.strange_eofa() != 0 && !(dsdr && hasenbusch)) {
            throw std::runtime_error(stdLogger.fatal("strange_eofa = 1 is instantiated with hasenbusch_masses and "
                                                     "dsdr_twists (even_odd = 1, c_sw = 0)"));
        }
        if (param.strange_eofa() != 0 && param.Ls() == 16) {
            runMDWFTwoPlusOneHmcRun<16, MDWFEvenOddHasenbuschDSDREofaTwoPlusOneHmc<HaloDepth, 16>, true>(commBase, param);
        } else if (param.strange_eofa() != 0) {
            runMDWFTwoPlusOneHmcRun<8, MDWFEvenOddHasenbuschDSDREofaTwoPlusOneHmc<HaloDepth, 8>, true>(commBase, param);
        } else if (dsdr && param.Ls() == 16) {
            runMDWFTwoPlusOneHmcRun<16, MDWFEvenOddHasenbuschDSDRTwoPlusOneHmc<HaloDepth, 16>, true>(commBase, param);
        } else if (dsdr) {
            runMDWFTwoPlusOneHmcRun<8, MDWFEvenOddHasenbuschDSDRTwoPlusOneHmc<HaloDepth, 8>, true>(commBase, param);
        } else if (param.even_odd() != 0 && hasenbusch) {
            runMDWFTwoPlusOneHmcRun<8, MDWFEvenOddHasenbuschTwoPlusOneHmc<HaloDepth, 8>, true>(commBase, param);
        } else if (param.even_odd() != 0) {
            runMDWFTwoPlusOneHmcRun<8, MDWFEvenOddTwoPlusOneHmc<HaloDepth, 8>, true>(commBase, param);
        } else {
            runMDWFTwoPlusOneHmcRun<8, MDWFTwoPlusOneHmc<HaloDepth, 8>, false>(commBase, param);
        }
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
