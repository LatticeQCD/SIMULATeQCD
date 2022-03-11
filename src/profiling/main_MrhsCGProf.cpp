#include "../SIMULATeQCD.h"
#include "../modules/dslash/dslash.h"
#include "../modules/HISQ/hisqSmearing.h"


template<class floatT, Layout LatLayout, size_t NStacks, bool onDevice>
void test_cg(CommunicationBase &commBase) {
    StopWatch<onDevice> timer;

    const int HaloDepth = 2;
    const int HaloDepthSpin = 4;
    const int maxIteration = 50;
    const double residual = 1e-8;
    floatT mass = 0.02;

    Gaugefield<floatT, onDevice, HaloDepth, R14> gauge(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, R18> gauge_smeared(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, U3R14> gauge_naik(commBase);

    HisqSmearing<floatT, onDevice, HaloDepth> smearing(gauge, gauge_smeared, gauge_naik);

    grnd_state<onDevice> rand;
    initialize_rng(1337, rand);

    gauge.random(rand.state);

    gpuError_t gpuErr = gpuGetLastError();
    if (gpuErr)
        rootLogger.info("Error creating random gauge field!");

    gauge.updateAll();

    gpuErr = gpuGetLastError();
    if (gpuErr)
        rootLogger.info("Error performing HaloUpdate!");

    smearing.SmearAll();

    gpuErr = gpuGetLastError();
    if (gpuErr)
        rootLogger.info("Error in HISQ smearing!");

    
    rootLogger.info("Initialize spinors");
    Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks> spinorIn(commBase);
    Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks> spinorOut(commBase);
    Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks> spinorTmp(commBase);
    gpuErr = gpuGetLastError();
    if (gpuErr)
        GpuError("error in spinor Initialization", gpuErr);

    rootLogger.info("Randomize spinors");
    spinorIn.gauss(rand.state);
    

    
    gpuErr = gpuGetLastError();
    if (gpuErr)
        rootLogger.info("error in gaussian spinors");

    rootLogger.info("Initialize DSlash");
    HisqDSlash<floatT, onDevice, LatLayout, HaloDepth, HaloDepthSpin, NStacks> dslash(gauge_smeared, gauge_naik, mass);
    
    gpuErr = gpuGetLastError();
    if (gpuErr)
        rootLogger.info("Error in Initialization of DSlash");

    ConjugateGradient<floatT, NStacks> MRHS_CG;

    rootLogger.info("Running CG");
    timer.start();

    MRHS_CG.invert(dslash, spinorOut, spinorIn, maxIteration, residual);

    timer.stop();    
    
    

    rootLogger.info("Time for MRHS-CG Inversion with " , NStacks, "-RHS: " , timer);
}

int main(int argc, char **argv) {

    stdLogger.setVerbosity(INFO);
    CommunicationBase commBase(&argc, &argv);

    LatticeParameters param;
    param.readfile(commBase, "../parameter/profiling/MrhsCGProfile.param", argc, argv);

    const int LatDim[] =  {param.latDim[0],param.latDim[1],param.latDim[2],param.latDim[3]};

    param.latDim.set(LatDim);

    commBase.init(param.nodeDim());

    const int HaloDepthSpin = 4;
    initIndexer(HaloDepthSpin,param,commBase);

    test_cg<float, Even, 1, true>(commBase);
    test_cg<float, Even, 2, true>(commBase);
    test_cg<float, Even, 3, true>(commBase);
    test_cg<float, Even, 4, true>(commBase);
    test_cg<float, Even, 5, true>(commBase);
    test_cg<float, Even, 6, true>(commBase);
    test_cg<float, Even, 7, true>(commBase);
    test_cg<float, Even, 8, true>(commBase);
    test_cg<float, Even, 9, true>(commBase);
    test_cg<float, Even, 10, true>(commBase);
    test_cg<float, Even, 11, true>(commBase);
    test_cg<float, Even, 12, true>(commBase);

}
