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
 *
 * Usage (from the build's testing directory):
 *   ./mdwfTwoPlusOneHmcRun <param file> [key=value ...]
 * Parameters: see parameter/tests/mdwfTwoPlusOneHmcRun.param. Ls = 8 is fixed
 * at compile time. Single rank; test-only scaffold, not a production RHMC.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFRhmcFermionActions.h"
#include "../experimental/mdwf/MDWFSpectralBounds.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>
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
    Parameter<int> check_steps;
    Parameter<int> symanzik_gauge;
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
        addDefault(check_steps, "check_steps", 500);
        addDefault(symanzik_gauge, "symanzik_gauge", 0);
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
template<size_t HaloDepth, size_t Ls>
bool mdwfTwoPlusOneRunCheckBounds(CommunicationBase &commBase, Gaugefield<double, true, HaloDepth, R18> &gauge,
                                  const MDWFHmcParameters &param, double mass, double low, double high, int steps,
                                  uint4 *randState, const std::string &label, const std::string &name) {
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

template<size_t Ls>
void runMDWFTwoPlusOneHmcRun(CommunicationBase &commBase, MDWFTwoPlusOneRunParameters &runParam) {
    const size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using Hmc = MDWFTwoPlusOneHmc<HaloDepth, Ls>;

    if (!runParam.beta.isSet()) {
        throw std::runtime_error(stdLogger.fatal("MDWF 2+1 HMC run requires beta in the parameter file"));
    }

    MDWFHmcParameters param{};
    param.beta = runParam.beta();
    param.M5 = runParam.M5();
    param.mf = runParam.mf();
    param.b5 = runParam.b5();
    param.csw = runParam.csw();
    param.pv_mass = runParam.pv_mass();
    param.tau = runParam.tau();
    param.steps = runParam.fermion_steps();
    param.gauge_substeps = runParam.gauge_substeps();
    param.max_iter = runParam.max_iter();
    param.precision = runParam.precision();
    param.symanzik_gauge = runParam.symanzik_gauge() != 0;
    param.rhmc.ms = runParam.ms();
    param.rhmc.lambda_low_s = runParam.lambda_low_s();
    param.rhmc.lambda_high_s = runParam.lambda_high_s();
    param.rhmc.lambda_low_pv = runParam.lambda_low_pv();
    param.rhmc.lambda_high_pv = runParam.lambda_high_pv();
    param.rhmc.action_error = runParam.action_error();
    param.rhmc.force_error = runParam.force_error();
    param.rhmc.max_order = runParam.max_order();
    param.rhmc.digits = runParam.remez_digits();

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

    rootLogger.info("MDWF 2+1 HMC run: ", param.symanzik_gauge ? "Symanzik" : "Wilson", " beta = ", param.beta, ", M5 = ", param.M5, ", mf = ", param.mf,
                    ", ms = ", param.rhmc.ms, ", pv_mass = ", param.pv_mass, ", b5 = ", param.b5,
                    ", c5 = ", param.b5 - 1.0, ", c_sw = ", param.csw, ", Ls = ", Ls, ", tau = ", param.tau,
                    ", fermion steps = ", param.steps, ", gauge substeps = ", param.gauge_substeps,
                    ", n_traj = ", nTraj, ", n_therm = ", nTherm, ", n_nometro = ", nNoMetro,
                    ", seed = ", runParam.seed(), ", solver precision = ", param.precision,
                    ", output = ", outputPath);
    rootLogger.info("MDWF 2+1 HMC run strange approximations: M_s^dagger M_s interval [", param.rhmc.lambda_low_s,
                    ", ", param.rhmc.lambda_high_s, "], M_1^dagger M_1 interval [", param.rhmc.lambda_low_pv, ", ",
                    param.rhmc.lambda_high_pv, "], action error ", param.rhmc.action_error, ", force error ",
                    param.rhmc.force_error, ", max order ", param.rhmc.max_order, ", ", param.rhmc.digits, " digits");

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

    if (checkSteps > 0) {
        const bool insideS = mdwfTwoPlusOneRunCheckBounds<HaloDepth, Ls>(
            commBase, gauge, param, param.rhmc.ms, param.rhmc.lambda_low_s, param.rhmc.lambda_high_s, checkSteps,
            d_rand.state, "start", "MDWF_2p1_run_check_start_s");
        const bool insidePv = mdwfTwoPlusOneRunCheckBounds<HaloDepth, Ls>(
            commBase, gauge, param, param.pv_mass, param.rhmc.lambda_low_pv, param.rhmc.lambda_high_pv, checkSteps,
            d_rand.state, "start", "MDWF_2p1_run_check_start_pv");
        if (!insideS || !insidePv) {
            throw std::runtime_error(stdLogger.fatal("MDWF 2+1 HMC run: the start configuration's spectrum is outside "
                                                     "the approximation intervals; widen them in the parameter file"));
        }
    }

    Hmc hmc(commBase, gauge, param, d_rand.state);
    const auto &strange = hmc.fermion().second();
    rootLogger.info("MDWF 2+1 HMC run strange approximation orders: A^(1/4) ", strange.quarterS().order, " (error ",
                    strange.quarterS().max_relative_error, "), A^(-1/2) ", strange.halfS().order, " (",
                    strange.halfS().max_relative_error, "), force ", strange.halfSForce().order, " (",
                    strange.halfSForce().max_relative_error, "); B^(1/4) ", strange.quarterPv().order, " (",
                    strange.quarterPv().max_relative_error, "), force ", strange.quarterPvForce().order, " (",
                    strange.quarterPvForce().max_relative_error, ")");

    std::ofstream out;
    if (commBase.IamRoot()) {
        out.open(outputPath, std::ios::out | std::ios::trunc);
        if (!out) {
            throw std::runtime_error(stdLogger.fatal("MDWF 2+1 HMC run cannot open output file ", outputPath));
        }
        out << "# traj accepted deltaH exp(-deltaH) plaquette kinetic_before gauge_before fermion_before "
               "fermion_forces seconds" << std::endl;
        out << std::setprecision(12);
    }

    std::vector<double> deltaH;
    std::vector<double> expMinusDeltaH;
    std::vector<double> accepted;
    std::vector<double> plaquette;

    for (int traj = 1; traj <= nTraj; traj++) {
        const auto startTime = std::chrono::steady_clock::now();
        const MDWFHmcTrajectoryResult result = hmc.trajectory(traj > nNoMetro, acceptRng);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
        GaugeAction<double, true, HaloDepth, R18> action(gauge);
        const double plaq = static_cast<double>(action.plaquette());
        const double expMdH = std::exp(-result.delta_h);

        rootLogger.info("MDWF 2+1 HMC run trajectory ", traj, ": Delta H = ", result.delta_h,
                        ", exp(-Delta H) = ", expMdH, ", accepted = ", result.accepted,
                        ", plaquette = ", plaq, ", fermion forces = ", result.force_evaluations,
                        ", seconds = ", seconds);
        if (commBase.IamRoot()) {
            out << traj << " " << (result.accepted ? 1 : 0) << " " << result.delta_h << " " << expMdH << " "
                << plaq << " " << result.before.kinetic << " " << result.before.gauge << " "
                << result.before.fermion << " " << result.force_evaluations << " " << seconds << std::endl;
        }

        if (traj > nTherm) {
            deltaH.push_back(result.delta_h);
            expMinusDeltaH.push_back(expMdH);
            accepted.push_back(result.accepted ? 1.0 : 0.0);
            plaquette.push_back(plaq);
        }
        if (saveConf && saveEvery > 0 && traj % saveEvery == 0 && traj < nTraj) {
            rootLogger.info("MDWF 2+1 HMC run: writing configuration after trajectory ", traj, " to ", confPath);
            gauge.writeconf_nersc(confPath);
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

    rootLogger.info("MDWF 2+1 HMC run summary (", deltaH.size(), " trajectories after ", nTherm,
                    " thermalization, block size ", blockSize, "):");
    rootLogger.info("  acceptance = ", accStat.mean, " +- ", accStat.naiveError);
    rootLogger.info("  <exp(-Delta H)> = ", expStat.mean, " +- ", expStat.naiveError, " (naive), +- ",
                    expStat.blockedError, " (blocked); (mean - 1) / error = ", expPull);
    rootLogger.info("  <Delta H> = ", dHStat.mean, " +- ", std::max(dHStat.naiveError, dHStat.blockedError),
                    ", <Delta H^2>/2 = ", halfDH2Stat.mean, " +- ",
                    std::max(halfDH2Stat.naiveError, halfDH2Stat.blockedError));
    rootLogger.info("  <plaquette> = ", plaqStat.mean, " +- ", plaqStat.naiveError, " (naive), +- ",
                    plaqStat.blockedError, " (blocked)");
    rootLogger.info("  total fermion forces = ", hmc.forceEvaluations(), ", gauge updates = ", hmc.gaugeUpdates());

    if (saveConf) {
        rootLogger.info("MDWF 2+1 HMC run: writing final configuration ", confPath);
        gauge.writeconf_nersc(confPath);
    }

    if (checkSteps > 0) {
        mdwfTwoPlusOneRunCheckBounds<HaloDepth, Ls>(commBase, gauge, param, param.rhmc.ms, param.rhmc.lambda_low_s,
                                                    param.rhmc.lambda_high_s, checkSteps, d_rand.state, "final",
                                                    "MDWF_2p1_run_check_final_s");
        mdwfTwoPlusOneRunCheckBounds<HaloDepth, Ls>(commBase, gauge, param, param.pv_mass, param.rhmc.lambda_low_pv,
                                                    param.rhmc.lambda_high_pv, checkSteps, d_rand.state, "final",
                                                    "MDWF_2p1_run_check_final_pv");
    }
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);

        MDWFTwoPlusOneRunParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfTwoPlusOneHmcRun.param", argc, argv);
        commBase.init(param.nodeDim());

        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);

        runMDWFTwoPlusOneHmcRun<8>(commBase, param);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
