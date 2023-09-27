/*
 * main_measureHadrons.cpp
 *
 * Luis Altenkort, 6 Jan 2021
 *
 * Main application for measuring hadronic correlators.
 *
 */

#include "../simulateqcd.h"
#include "../modules/measureHadrons/measureHadrons.h"

#define USE_GPU true
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

        const size_t HaloDepth = 2; //! reason: HISQ Dslash
        const size_t HaloDepthSpin = 4; //! reason: HISQ Dslash
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

        {
            measureHadrons<PREC, USE_GPU, HaloDepth, HaloDepthSpin, Even, NStacks, R18> mesons(commBase, lp, gauge);
            mesons.compute_HISQ_correlators();
            mesons.write_correlators_to_file();
        }

        //!Additional features we want to have:
        //TODO create a test routine that checks if the program works for all possible parameters on a small configuration
        //     using results from the BielefeldGPUCode as a reference
        //TODO add the option for multiple sources (maybe first one after another and then in parallel using stacked
        // spinorfields and multiRHS inverter?)
        //TODO add an option to choose the boundary conditions for the fermions (?)
    }

    catch (const std::runtime_error &error) {
        return -1;
    }
    return 0;
}

