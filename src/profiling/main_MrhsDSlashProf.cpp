/* 
 * main_MrhsDSlashProf.cpp
 * 
 */

#include "../SIMULATeQCD.h"
#include "../modules/dslash/dslash.h"
#include "../modules/rhmc/rhmcParameters.h"
#include "../modules/HISQ/hisqSmearing.h"


//the Dslash test function. Please start reading here.
template<class floatT, Layout LatLayout, Layout LatLayoutRHS, size_t NStacks, bool onDevice>
void test_dslash(CommunicationBase &commBase, int Vol){

    StopWatch<true> timer;

    const int HaloDepth = 2;
    const int HaloDepthSpin = 4;
    typedef GIndexer<LatLayout, HaloDepth> GInd;

    Gaugefield<floatT, onDevice, HaloDepth, R14> gauge(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, R18> gauge_smeared(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, U3R14> gauge_Naik(commBase);

    HisqSmearing<floatT, onDevice, HaloDepth> smearing(gauge, gauge_smeared, gauge_Naik);
    rootLogger.info("Starting Test with " ,  NStacks ,  " Stacks");
    rootLogger.info("Initialize random state");
    grnd_state<false> h_rand;
    grnd_state<onDevice> d_rand;

    h_rand.make_rng_state(1337);
    d_rand = h_rand;

    rootLogger.info("Generate configuration");
    gauge.random(d_rand.state);
    gpuError_t gpuErr = gpuGetLastError();
    if (gpuErr)
        rootLogger.info("Error in random gauge field");

    gauge.updateAll();
    gpuErr = gpuGetLastError();
    if (gpuErr)
        rootLogger.info("Error updateAll");

    smearing.SmearAll();
    gpuErr = gpuGetLastError();
    if (gpuErr)
        rootLogger.info("Error in smearing");

    rootLogger.info("Initialize spinors");
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> spinorIn(commBase);
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> spinorSave(commBase);
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> spinorOut(commBase);
    gpuErr = gpuGetLastError();
    if (gpuErr)
        GpuError("Error in spinor initialization", gpuErr);

    rootLogger.info("Randomize spinors");
    spinorIn.gauss(d_rand.state);

    spinorSave = spinorIn;
    gpuErr = gpuGetLastError();
    if (gpuErr)
        rootLogger.info("Error in gaussian spinors");

    rootLogger.info("Initialize DSlash");
    HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, NStacks> dslash(gauge_smeared, gauge_Naik, 0.0);
    gpuErr = gpuGetLastError();
    if (gpuErr)
        rootLogger.info("Error in initialization of DSlash");
    
    for (int i = 0; i < 500; ++i) {
        timer.start();
        dslash.applyMdaggM(spinorOut, spinorIn, false);
        timer.stop();
        spinorIn=spinorSave;
    }
     
    rootLogger.info("Time for 500 applications of multiRHS Dslash: " ,  timer);
  
    float EOfactor = ((LatLayout == Even || LatLayout == Odd) ? 0.5 : 1.0);
  
    float TFlops = NStacks * Vol * EOfactor * 500 * 2316 /(timer.milliseconds() * 1e-3)*1e-12;
    rootLogger.info("Achieved TFLOP/s " ,  TFlops);
}



int main(int argc, char **argv) {

    stdLogger.setVerbosity(INFO);
    CommunicationBase commBase(&argc, &argv);

    LatticeParameters param;
    param.readfile(commBase, "../parameter/profiling/MrhsDSlashProf.param", argc, argv);

 
    const int LatDim[] = {param.latDim[0],param.latDim[1],param.latDim[2],param.latDim[3]};

    int Vol = LatDim[0]*LatDim[1]*LatDim[2]*LatDim[3];

    param.latDim.set(LatDim);

    commBase.init(param.nodeDim());

    const int HaloDepthSpin = 4;
    initIndexer(HaloDepthSpin,param, commBase);
    stdLogger.setVerbosity(INFO);

    rootLogger.info("-------------------------------------");
    rootLogger.info("Running on Device");
    rootLogger.info("-------------------------------------");
    rootLogger.info("Testing Even - Odd");
    rootLogger.info("------------------");
    test_dslash<float, Even, Odd, 1, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 2, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 3, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 4, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 5, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 6, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 7, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 8, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 9, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 10, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 11, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 12, true>(commBase, Vol);
}
