/*
 * One-shot full-field comparison of serialized legacy and recursive HISQ forces.
 */

#include "../simulateqcd.h"
#include "../modules/rhmc/rhmcParameters.h"
#include "testing.h"

#include <string>
#include <vector>

#define PREC double

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);
    std::string legacyPath = "force_large_legacy";
    std::string recursivePath = "force_large_recursive";
    std::vector<char *> parameterArgv{argv[0]};
    for (int i = 1; i < argc; ++i) {
        const std::string option(argv[i]);
        if (option == "--legacy-force" || option == "--recursive-force") {
            if (++i >= argc) {
                rootLogger.error("Large-lattice legacy vs recursive HISQ force: FAIL (missing field path)");
                return 2;
            }
            (option == "--legacy-force" ? legacyPath : recursivePath) = argv[i];
        } else {
            parameterArgv.push_back(argv[i]);
        }
    }
    int parameterArgc = static_cast<int>(parameterArgv.size());
    RhmcParameters rhmc_param;
    rhmc_param.readfile(commBase, "../parameter/tests/hisqForce_bench.param", parameterArgc, parameterArgv.data());

    commBase.init(rhmc_param.nodeDim());

    const size_t HaloDepth = 0;
    initIndexer(HaloDepth, rhmc_param, commBase);

    Gaugefield<PREC, true, HaloDepth, R18> legacyForce(commBase);
    Gaugefield<PREC, true, HaloDepth, R18> recursiveForce(commBase);

    legacyForce.readconf_nersc(legacyPath);
    recursiveForce.readconf_nersc(recursivePath);

    const bool pass = compare_fields<PREC, HaloDepth, true, R18>(legacyForce, recursiveForce, 1e-8);

    if (!pass) {
        rootLogger.error("Large-lattice legacy vs recursive HISQ force: FAIL");
        return 1;
    }

    rootLogger.info(CoutColors::green, "Large-lattice legacy vs recursive HISQ force: PASS", CoutColors::reset);
    return 0;
}
