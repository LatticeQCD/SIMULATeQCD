#include "../simulateqcd.h"
#include "../modules/dslash/dslash.h"
#include "../modules/rhmc/rhmcParameters.h"

int main(int argc, char **argv) {

    stdLogger.setVerbosity(DEBUG);

    CommunicationBase commBase(&argc, &argv);

    RhmcParameters param;
    param.readfile(commBase, "../parameter/profiling/inverterProfile.param", argc, argv);

    RationalCoeff rat;
    rat.readfile(commBase, param.rat_file());

    commBase.init(param.nodeDim(), param.gpuTopo());

    const int HaloDepthSpin = 4;
    const int HaloDepth = 2;
    initIndexer(HaloDepthSpin, param, commBase);


    Gaugefield<float, true, HaloDepth, R18> gauge_smeared(commBase);
    Gaugefield<float, true, HaloDepth, U3R14> gauge_naik(commBase);

    gauge_smeared.one();
    gauge_naik.one();

    grnd_state<true> d_rand;
    initialize_rng(579238, d_rand);

    typedef Spinorfield<float, true, Even, HaloDepthSpin, 1> spin_t;
    typedef Spinorfield<float, true, Even, HaloDepthSpin, 14> spin14_t;
    
    spin_t spinorIn(commBase);
    spin_t spinorOut(commBase);
    spin_t spinorCheck(commBase);

    spinorIn.gauss(d_rand.state);

    ConjugateGradient<float, 1> cg;
    rootLogger.info("Run Inversion");

    HisqDSlash<float, true, Even, HaloDepth, HaloDepthSpin, 1> dslash(gauge_smeared, gauge_naik, param.m_ud());
    
    StopWatch<true> timer;
    //roctracer_start();
    timer.start();

    cg.invert_res_replace(dslash, spinorOut, spinorIn, param.cgMax(), param.residue(), 0.1);
    timer.stop();
    //roctracer_stop();
    dslash.applyMdaggM(spinorCheck, spinorOut);

    spinorOut = spinorCheck - spinorIn;


    COMPLEX(double) err_abs(0.0);
    err_abs = spinorOut.dotProduct(spinorOut);
    rootLogger.info("Absolute error of [D^{dagger}D * (D^{dagger}D)^{-1}] * phi - phi = )",err_abs);

    return 0;
}
