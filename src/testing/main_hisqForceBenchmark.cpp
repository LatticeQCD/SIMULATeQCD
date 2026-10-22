/*
 * main_hisqForceBenchmark.cpp
 *
 * Large-lattice timing run for the optimized HISQ force.
 */

#include "../simulateqcd.h"
#include "../modules/hisq/hisqSmearing.h"
#include "../modules/hisq/hisqForce.h"

#include <string>
#include <vector>

#define PREC double

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);

    StopWatch<true> timer;
    CommunicationBase commBase(&argc, &argv);

    std::string forceOutput;
    bool randomGauge = false;
    std::vector<char *> parameterArgv{argv[0]};
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--force-output") {
            if (++i >= argc) {
                rootLogger.error("--force-output requires a file name");
                return 2;
            }
            forceOutput = argv[i];
        } else if (std::string(argv[i]) == "--random-gauge") {
            randomGauge = true;
        } else {
            parameterArgv.push_back(argv[i]);
        }
    }

    int parameterArgc = static_cast<int>(parameterArgv.size());
    RhmcParameters rhmc_param;
    rhmc_param.readfile(commBase, "../parameter/tests/hisqForce_bench.param", parameterArgc, parameterArgv.data());

    commBase.init(rhmc_param.nodeDim());

    const size_t HaloDepth = 0;
    const size_t HaloDepthSpin = 4;

    typedef GIndexer<All, HaloDepth> GInd;
    initIndexer(HaloDepth, rhmc_param, commBase);

    RationalCoeff rat;
    rat.readfile(commBase, rhmc_param.rat_file());

    Gaugefield<PREC, false, HaloDepth> force_host(commBase);
    Gaugefield<PREC, true, HaloDepth, R18> gauge(commBase);
    Gaugefield<PREC, true, HaloDepth> gaugeLvl2(commBase);
    Gaugefield<PREC, true, HaloDepth, U3R14> gaugeNaik(commBase);
    Gaugefield<PREC, true, HaloDepth> force(commBase);
    Spinorfield<PREC, true, Even, HaloDepthSpin> SpinorIn(commBase);

    grnd_state<true> d_rand;
    initialize_rng(rhmc_param.seed(), d_rand);

    if (randomGauge) {
        rootLogger.info(
            "HISQ force input: RANDOM gauge configuration "
            "(synthetic volume-scaling point; not thermalized)");
        gauge.random(d_rand.state);
    } else {
        rootLogger.info(
            "HISQ force input: THERMALIZED gauge configuration ",
            rhmc_param.GaugefileName());

        gauge.readconf_nersc(rhmc_param.GaugefileName());
    }
    gauge.updateAll();

    HisqSmearing<PREC, true, HaloDepth, R18> smearing(gauge, gaugeLvl2, gaugeNaik);
    smearing.SmearAll();

    AdvancedMultiShiftCG<PREC, 14> CG;

    HisqDSlash<PREC, true, Even, HaloDepth, HaloDepthSpin, 1> dslash(gaugeLvl2, gaugeNaik, 0.0);
    HisqDSlash<PREC, true, Even, HaloDepth, HaloDepthSpin, 14> dslash_multi(gaugeLvl2, gaugeNaik, 0.0);

    HisqForce<PREC, true, HaloDepth, HaloDepthSpin, R18, true> forceCalculator(
        gauge, force, CG, dslash, dslash_multi, rhmc_param, rat, smearing);

    timer.start();
    forceCalculator.TestForce(SpinorIn, force, d_rand);
    timer.stop();

    rootLogger.info("HISQ force time: ", sformat("%.6fs", timer.seconds()));

    force_host = force;
    SU3<PREC> result = force_host.getAccessor().getLink(GInd::getSiteMu(0, 0, 0, 3, 3));

    rootLogger.info("Time: ", timer);
    rootLogger.info("Force:");
    rootLogger.info(result.getLink00(), result.getLink01(), result.getLink02());
    rootLogger.info(result.getLink10(), result.getLink11(), result.getLink12());
    rootLogger.info(result.getLink20(), result.getLink21(), result.getLink22());

    if (!forceOutput.empty()) {
        force.writeconf_nersc(forceOutput, 3, 2);
        rootLogger.info("Complete HISQ force written to ", forceOutput);
    }

    return 0;
}
