/*
 * MDWF Pauli-Villars two-flavour HMC run (step 3 of the MDWF RHMC plan:
 * <exp(-Delta H)> = 1 over Metropolis trajectories).
 *
 * Runs MDWFPauliVillarsTwoFlavorHmc (Wilson gauge action, Mobius clover
 * det(M_f^\dagger M_f) / det(M_1^\dagger M_1)) with the Sexton-Weingarten two-scale
 * leapfrog for n_traj trajectories with Metropolis accept/reject. Every
 * trajectory appends one line to <measurements_dir>/<output_name>:
 *
 *   traj  accepted  deltaH  exp(-deltaH)  plaquette  kinetic_before  gauge_before
 *   fermion_before  fermion_forces  seconds
 *
 * The first n_nometro (<= n_therm) trajectories are always accepted so the
 * field can leave the cold start; they never enter the statistics.
 * After the run, the trajectories after the first n_therm give the acceptance
 * rate, <exp(-Delta H)> (naive and blocked standard errors), <Delta H> versus
 * <Delta H^2>/2 (equal to leading order for an exact area-preserving,
 * reversible integrator), and the mean plaquette. Correct sampling requires
 * <exp(-Delta H)> = 1 within errors.
 *
 * Usage (from the build's testing directory):
 *   ./mdwfPauliVillarsHmcRun <param file> [key=value ...]
 * Parameters: see parameter/tests/mdwfPauliVillarsHmcRun.param. Start is cold
 * (unit) unless Gaugefile (NERSC) is set; Gaugefile_out saves the final
 * configuration into measurements_dir. Ls = 8 is fixed at compile time.
 * Single rank; test-only scaffold, not a production RHMC.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFHmc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

class MDWFHmcRunParameters : public LatticeParameters {
public:
    Parameter<double> M5;
    Parameter<double> mf;
    Parameter<double> b5;
    Parameter<double> csw;
    Parameter<double> pv_mass;
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
    Parameter<std::string> output_name;

    MDWFHmcRunParameters() {
        addDefault(M5, "M5", 1.8);
        addDefault(mf, "mf", 0.1);
        addDefault(b5, "b5", 1.5);
        addDefault(csw, "c_sw", 0.5);
        addDefault(pv_mass, "pv_mass", 1.0);
        addDefault(tau, "tau", 0.5);
        addDefault(fermion_steps, "fermion_steps", 10);
        addDefault(gauge_substeps, "gauge_substeps", 8);
        addDefault(n_traj, "n_traj", 60);
        addDefault(n_therm, "n_therm", 20);
        addDefault(n_nometro, "n_nometro", 5);
        addDefault(seed, "seed", 20260925);
        addDefault(max_iter, "max_iter", 20000);
        addDefault(precision, "precision", 1e-10);
        addDefault(block_size, "block_size", 5);
        addDefault(output_name, "output_name", std::string("mdwfPauliVillarsHmcRun.dat"));
    }
};

struct MDWFHmcRunStatistic {
    double mean;
    double naiveError;
    double blockedError;
};

MDWFHmcRunStatistic mdwfHmcRunStatistic(const std::vector<double> &values, int blockSize) {
    MDWFHmcRunStatistic stat{0.0, 0.0, 0.0};
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

template<size_t Ls>
void runMDWFPauliVillarsHmcRun(CommunicationBase &commBase, MDWFHmcRunParameters &runParam) {
    const size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using Hmc = MDWFPauliVillarsTwoFlavorHmc<HaloDepth, Ls>;

    if (!runParam.beta.isSet()) {
        throw std::runtime_error(stdLogger.fatal("MDWF HMC run requires beta in the parameter file"));
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

    const int nTraj = runParam.n_traj();
    const int nTherm = runParam.n_therm();
    const int nNoMetro = runParam.n_nometro();
    if (nNoMetro > nTherm) {
        throw std::runtime_error(stdLogger.fatal("MDWF HMC run requires n_nometro <= n_therm, so that trajectories "
                                                 "without accept/reject never enter the statistics"));
    }
    const std::string outputPath = runParam.measurements_dir() + "/" + runParam.output_name();

    rootLogger.info("MDWF HMC run: Wilson beta = ", param.beta, ", M5 = ", param.M5, ", mf = ", param.mf,
                    ", pv_mass = ", param.pv_mass, ", b5 = ", param.b5, ", c5 = ", param.b5 - 1.0,
                    ", c_sw = ", param.csw, ", Ls = ", Ls, ", tau = ", param.tau,
                    ", fermion steps = ", param.steps, ", gauge substeps = ", param.gauge_substeps,
                    ", n_traj = ", nTraj, ", n_therm = ", nTherm, ", n_nometro = ", nNoMetro,
                    ", seed = ", runParam.seed(),
                    ", solver precision = ", param.precision, ", output = ", outputPath);

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(runParam.seed());
    d_rand = h_rand;
    std::mt19937_64 acceptRng(static_cast<unsigned long>(runParam.seed()) + 1UL);

    Gauge gauge(commBase, "MDWF_hmc_run_gauge");
    if (runParam.GaugefileName.isSet()) {
        rootLogger.info("MDWF HMC run: reading start configuration ", runParam.GaugefileName());
        gauge.readconf_nersc(runParam.GaugefileName());
    } else {
        rootLogger.info("MDWF HMC run: cold (unit) start");
        gauge.one();
    }
    gauge.updateAll();

    Hmc hmc(commBase, gauge, param, d_rand.state);

    std::ofstream out;
    if (commBase.IamRoot()) {
        out.open(outputPath, std::ios::out | std::ios::trunc);
        if (!out) {
            throw std::runtime_error(stdLogger.fatal("MDWF HMC run cannot open output file ", outputPath));
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
        // The first n_nometro trajectories skip accept/reject so the field can leave the cold start.
        const MDWFHmcTrajectoryResult result = hmc.trajectory(traj > nNoMetro, acceptRng);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
        GaugeAction<double, true, HaloDepth, R18> action(gauge);
        const double plaq = static_cast<double>(action.plaquette());
        const double expMdH = std::exp(-result.delta_h);

        rootLogger.info("MDWF HMC run trajectory ", traj, ": Delta H = ", result.delta_h,
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
    }

    std::vector<double> halfDeltaH2(deltaH.size());
    for (size_t i = 0; i < deltaH.size(); i++) {
        halfDeltaH2[i] = 0.5 * deltaH[i] * deltaH[i];
    }
    const int blockSize = runParam.block_size();
    const MDWFHmcRunStatistic expStat = mdwfHmcRunStatistic(expMinusDeltaH, blockSize);
    const MDWFHmcRunStatistic dHStat = mdwfHmcRunStatistic(deltaH, blockSize);
    const MDWFHmcRunStatistic halfDH2Stat = mdwfHmcRunStatistic(halfDeltaH2, blockSize);
    const MDWFHmcRunStatistic accStat = mdwfHmcRunStatistic(accepted, blockSize);
    const MDWFHmcRunStatistic plaqStat = mdwfHmcRunStatistic(plaquette, blockSize);
    const double expError = std::max(expStat.naiveError, expStat.blockedError);
    const double expPull = expError > 0.0 ? (expStat.mean - 1.0) / expError : 0.0;

    rootLogger.info("MDWF HMC run summary (", deltaH.size(), " trajectories after ", nTherm,
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

    if (runParam.GaugefileName_out.isSet()) {
        const std::string confPath = runParam.measurements_dir() + "/" + runParam.GaugefileName_out();
        rootLogger.info("MDWF HMC run: writing final configuration ", confPath);
        gauge.writeconf_nersc(confPath);
    }
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);

        MDWFHmcRunParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfPauliVillarsHmcRun.param", argc, argv);
        commBase.init(param.nodeDim());

        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);

        runMDWFPauliVillarsHmcRun<8>(commBase, param);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
