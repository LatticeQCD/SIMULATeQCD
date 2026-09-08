/*
 * One-trajectory RHMC benchmark. Initialization and I/O are intentionally
 * outside the timed region.
 */

#include "../simulateqcd.h"
#include "../modules/rhmc/rhmc.h"
#include "../modules/observables/polyakovLoop.h"

#include <string>
#include <vector>

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);

    std::string gaugeOutput;
    bool reverse = false;
    std::vector<char *> parameterArgv{argv[0]};
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--gauge-output") {
            if (++i >= argc) {
                rootLogger.error("--gauge-output requires a file name");
                return 2;
            }
            gaugeOutput = argv[i];
        } else if (argument == "--reverse") {
            reverse = true;
        } else {
            parameterArgv.push_back(argv[i]);
        }
    }

    int parameterArgc = static_cast<int>(parameterArgv.size());
    RhmcParameters param;
    param.readfile(commBase, "../parameter/tests/rhmcBenchmark.param", parameterArgc, parameterArgv.data());

    RationalCoeff rat;
    rat.readfile(commBase, param.rat_file());
    rat.check_rat(param);

    commBase.init(param.nodeDim(), param.gpuTopo());

    using floatT = float;
    constexpr size_t HaloDepth = 2;

    initIndexer(4, param, commBase);

    Gaugefield<floatT, true, HaloDepth> gauge(commBase);
    grnd_state<true> d_rand;

    initialize_rng(param.seed(), d_rand);

    gauge.readconf_nersc("../test_conf/l528f21b6315m00282m0759_001.1610");
    gauge.updateAll();
    gauge.su3latunitarize();

    PolyakovLoop<floatT, true, HaloDepth, R18> polyakovLoop(gauge);
    GaugeAction<floatT, true, HaloDepth, R18> gaugeAction(gauge);

    rootLogger.info("Initial plaquette: ", sformatScientific(gaugeAction.plaquette()));
    rootLogger.info("Initial rectangle: ", sformatScientific(gaugeAction.rectangle()));
    rootLogger.info("Initial Polyakov loop: ", sformatScientific(polyakovLoop.getPolyakovLoop()));

    rhmc<floatT, true, HaloDepth> HMC(param, rat, gauge, d_rand.state);
    HMC.init_ratapprox();

    StopWatch<true> timer;
    timer.start();
    int accepted = HMC.update(!param.always_acc(), reverse);
    timer.stop();

    rootLogger.info("RHMC trajectory time: ", sformat("%.3fs", timer.seconds()));
    rootLogger.info("RHMC trajectory status: ", accepted ? "ACCEPTED" : "REJECTED");
    rootLogger.info("Final plaquette: ", sformatScientific(gaugeAction.plaquette()));
    rootLogger.info("Final rectangle: ", sformatScientific(gaugeAction.rectangle()));
    rootLogger.info("Final Polyakov loop: ", sformatScientific(polyakovLoop.getPolyakovLoop()));

    if (!gaugeOutput.empty()) {
        gauge.writeconf_nersc(gaugeOutput, 3, 2);
        rootLogger.info("Final RHMC gauge field written to ", gaugeOutput);
    }

    return 0;
}
