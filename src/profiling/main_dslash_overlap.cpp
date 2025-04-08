#include "../simulateqcd.h"
#include "../modules/dslash/dslash.h"

template<class floatT, Layout LatLayout, Layout LatLayoutRHS, size_t NStacks, bool onDevice>
void test_dslash(CommunicationBase &commBase, int Vol) {
    StopWatch<true> timer;

    const int HaloDepth = 2;
    const int HaloDepthSpin = 4;

    Gaugefield<floatT, onDevice, HaloDepth, R18> gauge_smeared(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, U3R14> gauge_naik(commBase);

    rootLogger.info("Initialize random state");
    grnd_state<false> h_rand;
    grnd_state<onDevice> d_rand;

    h_rand.make_rng_state(1337);
    d_rand = h_rand;

    rootLogger.info("Generate configuration");
    gauge_smeared.random(d_rand.state);
    gauge_naik.random(d_rand.state);
    
    gpuError_t gpuErr = gpuGetLastError();
    if (gpuErr)
        rootLogger.info("Error in random gauge field");

    gauge_smeared.updateAll();
    gauge_naik.updateAll();
    
    gpuErr = gpuGetLastError();
    if (gpuErr)
        rootLogger.info("Error updateAll");

    rootLogger.info("Initialize spinors");

    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> spinorIn(commBase);
    Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks> spinorOut(commBase);

    gpuErr = gpuGetLastError();
    if (gpuErr)
        GpuError("Error in spinor initialization", gpuErr);

    rootLogger.info("Randomize spinors");
    spinorIn.gauss(d_rand.state);

    gpuErr = gpuGetLastError();
    if (gpuErr)
        rootLogger.info("Error in gaussian spinors");

    HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, NStacks> dslash(gauge_smeared, gauge_naik, 0.0);
    gpuErr = gpuGetLastError();
    if (gpuErr)
        rootLogger.info("Error in initialization of DSlash");


    float EOfactor = ((LatLayout == Even || LatLayout == Odd) ? 0.5 : 1.0);


    timer.reset();
    timer.start();
    for (int i = 0; i < 50; ++i) {
        dslash.Dslash(spinorOut,spinorIn,true);
    }
    timer.stop();
    rootLogger.info("Time for 50 applications of Dslash (serial comms): ", timer);
    float TFlops = NStacks * Vol * EOfactor * 50 * 1146 /(timer.milliseconds() * 1e-3)*1e-12;
    rootLogger.info("Achieved TFLOP/s ", TFlops);

    timer.reset();
    timer.start();
    for (int i =0; i < 50; ++i) {
        spinorIn.updateAll(COMM_START | Hyperplane);
        dslash.Dslash_center(spinorOut,spinorIn);
        spinorIn.updateAll(COMM_FINISH | Hyperplane);
        dslash.Dslash_halo(spinorOut, spinorIn);
    }

    timer.stop();
    rootLogger.info("Time for 50 applications of Dslash (overlapping comms): ", timer);
    TFlops = NStacks * Vol * EOfactor * 50 * 1146 /(timer.milliseconds() * 1e-3)*1e-12;
    rootLogger.info("Achieved TFLOP/s ", TFlops);

    
}

int main(int argc, char **argv) {
    stdLogger.setVerbosity(INFO);
    CommunicationBase commBase(&argc, &argv);

    LatticeParameters param;
    param.readfile(commBase, "../parameter/profiling/mrhsDSlashProf.param", argc, argv);


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
    rootLogger.info("Testing Even - Odd - FP32");
    rootLogger.info("------------------");
    test_dslash<float, Even, Odd, 1, true>(commBase, Vol);
    rootLogger.info("-------------------------------------");
    rootLogger.info("Testing Even - Odd - FP64");
    rootLogger.info("------------------");
    test_dslash<double, Even, Odd, 1, true>(commBase, Vol);
    
}