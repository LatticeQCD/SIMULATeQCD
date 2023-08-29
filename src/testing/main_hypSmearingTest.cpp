/*
 * main_HypSmearingTest.cu
 *
 */

#include "../simulateqcd.h"
#include "../modules/hyp/hypSmearing.h"

#define PREC double
#define MY_BLOCKSIZE 256
#define USE_GPU true

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);
    LatticeParameters param;
    StopWatch<true> timer;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/tests/hypSmearingTest.param", argc, argv);
    commBase.init(param.nodeDim());

    const size_t HaloDepth = 0;

    rootLogger.info("Initialize Lattice");
    typedef GIndexer<All,HaloDepth> GInd;
    initIndexer(HaloDepth,param,commBase);

    rootLogger.info("Initialize Gaugefield");
    Gaugefield<PREC, true, HaloDepth> gauge_in(commBase);
    Gaugefield<PREC, true, HaloDepth> gauge_out(commBase);

    timer.start();
    HypSmearing<PREC, USE_GPU, HaloDepth,R18> smearing(gauge_in);
    timer.stop();
    rootLogger.info("Time for initializing HypSmearing: ", timer);
    timer.reset();

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(12345);
    d_rand = h_rand;
    gauge_in.random(d_rand.state);

    timer.start();
    smearing.SmearAll(gauge_out);

    timer.stop();
    rootLogger.info("Time for full smearing: ", timer);
    gauge_out.writeconf_nersc("../test_conf/pgpu_hyp_smearing.nersc");

    //! Check plaquette
    GaugeAction<PREC, USE_GPU, HaloDepth> gAction(gauge_out);
    PREC plaq;
    plaq = gAction.plaquette();
    rootLogger.info("plaquette: " ,  plaq);

    int faults = 0;
    if(std::abs(plaq - 0.592294660376)>1E-7)faults+=1;

    rootLogger.info("abs(plaquette-plaquette_expected): ", std::abs(plaq-0.592294660376));

    if (faults == 0) {
        rootLogger.info(CoutColors::green, "Hyp test passed!", CoutColors::reset);
    } else {
        rootLogger.error("Hyp test failed!");
        return 1;
    }
    return 0;
}
