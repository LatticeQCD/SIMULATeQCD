/*
 * Full-field comparison for the one-step deterministic RHMC diagnostic.
 */

#include "../simulateqcd.h"
#include "../modules/rhmc/rhmcParameters.h"
#include "testing.h"

#include <string>

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);

    if (argc != 3) {
        rootLogger.error("Usage: rhmcTrajectoryCompare FIELD_A FIELD_B");
        return 2;
    }

    const std::string fieldA(argv[1]);
    const std::string fieldB(argv[2]);

    CommunicationBase commBase(&argc, &argv);
    RhmcParameters param;
    param.readfile(commBase, "../parameter/tests/rhmcBenchmark.param");
    commBase.init(param.nodeDim(), param.gpuTopo());

    using floatT = float;
    constexpr size_t HaloDepth = 2;

    initIndexer(4, param, commBase);

    Gaugefield<floatT, true, HaloDepth, R18> legacyGauge(commBase);
    Gaugefield<floatT, true, HaloDepth, R18> recursiveGauge(commBase);

    legacyGauge.readconf_nersc(fieldA);
    recursiveGauge.readconf_nersc(fieldB);

    const bool pass = compare_fields<floatT, HaloDepth, true, R18>(legacyGauge, recursiveGauge, 1e-6);

    if (!pass) {
        rootLogger.error("RHMC gauge-field comparison: FAIL");
        return 1;
    }

    rootLogger.info(CoutColors::green, "RHMC gauge-field comparison: PASS", CoutColors::reset);
    return 0;
}
