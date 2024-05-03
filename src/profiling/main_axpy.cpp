#include "../simulateqcd.h"

int main(int argc, char *argv[]) {
    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);
    LatticeParameters param;

    param.readfile(commBase, "../parameter/profiling/mrhsDSlashProf.param", argc, argv);

    commBase.init(param.nodeDim());
    commBase.forceHalos(true);

    const size_t HaloDepth = 4;
    initIndexer(HaloDepth, param,commBase);
    grnd_state<true> rand;
    initialize_rng(1337,rand);
    
    StopWatch<true> timer;
    Spinorfield<float, true, Even, HaloDepth, 14> spinor(commBase);
    Spinorfield<float, true, Even, HaloDepth, 14> x(commBase);
    SimpleArray<float, 14> B(1.0);
    spinor.gauss(rand.state);
    x.gauss(rand.state);


    timer.start();
    spinor.axpyThisLoop(B, x, 14);
    timer.stop();

    rootLogger.info("axpyThisLoop timing: ", timer);
    return 0;
}