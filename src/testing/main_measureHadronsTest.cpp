/* 
 * main_measureHadronsTest.cu                                                               
 * 
 * Luis Altenkort, 6 Jan 2021
 * 
 * Routine to measure hadronic correlators.
 *
 */

#include "../SIMULATeQCD.h"
#include "../modules/measureHadrons/measureHadrons.h"

#define USE_GPU true
//define precision
#if SINGLEPREC
#define PREC float
#else
#define PREC double
#endif


int main(int argc, char *argv[]) {
    try {
        stdLogger.setVerbosity(TRACE);

        CommunicationBase commBase(&argc, &argv);
        measureHadronsParam<PREC> lp;
        lp.readfile(commBase, "../parameter/tests/measureHadronsTest.param", argc, argv);
        commBase.init(lp.nodeDim());
        lp.check_for_nonsense();

        const size_t HaloDepth = 2; //FIXME what halodepth do we actually need?
        const size_t HaloDepthSpin = 4; //FIXME and what do we need here?
        const size_t NStacks = 1; //TODO add support for multiple sources at the same time i.e. nstacks>1

        initIndexer(HaloDepth, lp, commBase);

        Gaugefield<PREC, USE_GPU, HaloDepth> gauge(commBase);
        if (lp.use_unit_conf()){
            rootLogger.info("Using unit configuration for tests/benchmarks");
            gauge.one();
        } else {
            rootLogger.info("Read configuration");
            gauge.readconf_nersc(lp.GaugefileName());
        }
        gauge.updateAll();

        //! Check plaquette
        GaugeAction<PREC, USE_GPU, HaloDepth> gAction(gauge);
        PREC plaq;
        plaq = gAction.plaquette();
        rootLogger.info("plaquette: " ,  plaq);


        measureHadrons<PREC, USE_GPU, HaloDepth, HaloDepthSpin, Even, NStacks, R18> mesons(commBase, lp, gauge);
        mesons.compute_HISQ_correlators();

        //TODO compare results against BielefeldGPUCode (put those in refValues_measureHadrons.h)
        // and give clear pass/fail output

        //TODO in the end add this script to the compound testing script and to the bash test_run script

    }

    catch (const std::runtime_error &error) {
        return -1;
    }
    return 0;
}

