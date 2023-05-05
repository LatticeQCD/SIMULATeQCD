/*
 * main_createCorrelatorNorm.cpp
 *
 * D. Clarke
 *
 * The way the correlator is coded, it needs a normalization vector that is slow to calculate. My workaround is to save
 * this vector to a file, created by this script, that will be read by the correlator main later. Each lattice geometry
 * requires a new normalization vector.
 *
 */

#include "../SIMULATeQCD.h"

#define PREC double

struct corrParam : LatticeParameters {
    Parameter<std::string>  domain;

    corrParam() {
        addOptional(domain,"Domain");
    }
};

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);
    const size_t HaloDepth  = 0;

    rootLogger.info("Initialization.");
    corrParam param;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../test_parameter/correlatorTest.param", argc, argv);
    commBase.init(param.nodeDim());
    initIndexer(HaloDepth,param,commBase);
    typedef GIndexer<All,HaloDepth> GInd;
    CorrelatorTools<PREC,false,HaloDepth> corrTools;

    std::string domain = param.domain();

    corrTools.createNorm(domain,commBase);

    return 0;
}

