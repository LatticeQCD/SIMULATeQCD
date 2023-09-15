/*
 * main_MrhsDSlashProf.cpp
 *
 */

#include "../simulateqcd.h"
#include "../modules/dslash/dslash.h"
#include "../modules/rhmc/rhmcParameters.h"
#include "../modules/hisq/hisqSmearing.h"


//the Dslash test function. Please start reading here.
template<class floatT, Layout LatLayout, Layout LatLayoutRHS, size_t NStacks, size_t NStacks_cached, bool onDevice>
void test_dslash(CommunicationBase &commBase, int Vol){

    StopWatch<true> timer;

    const int HaloDepth = 2;
    const int HaloDepthSpin = 4;
   
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

    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks * NStacks_cached> spinorIn2(commBase);
    Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks * NStacks_cached> spinorOut2(commBase);
    Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks * NStacks_cached> spinorOut_ref(commBase);
    
    Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks> spinorOut3(commBase);
    
    gpuErr = gpuGetLastError();
    if (gpuErr)
        GpuError("Error in spinor initialization", gpuErr);

    rootLogger.info("Randomize spinors");
    spinorIn2.gauss(d_rand.state);
    gpuErr = gpuGetLastError();
    if (gpuErr)
        rootLogger.info("Error in gaussian spinors");

    rootLogger.info("Initialize DSlash");
    HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, NStacks*NStacks_cached, NStacks_cached> dslash(gauge_smeared, gauge_Naik, 0.0);
    HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, NStacks*NStacks_cached> dslash2(gauge_smeared, gauge_Naik, 0.0);
    gpuErr = gpuGetLastError();
    if (gpuErr)
        rootLogger.info("Error in initialization of DSlash");


    float EOfactor = ((LatLayout == Even || LatLayout == Odd) ? 0.5 : 1.0);


    timer.reset();
    for (int i = 0; i < 5; ++i) {
        timer.start();
        dslash.Dslash(spinorOut2,spinorIn2,false);
        timer.stop();

    }


    rootLogger.info("Time for 5 applications of multiRHS Dslash (thread+block version): ", timer);
    float TFlops = NStacks * NStacks_cached * Vol * EOfactor * 5 * 1146 /(timer.milliseconds() * 1e-3)*1e-12;
    rootLogger.info("Achieved TFLOP/s ", TFlops, " with ", NStacks, " Stacks (thread loop) and ", NStacks_cached, " Stacks (blockDim.y)");

    dslash.Dslash_stacked(spinorOut2,spinorIn2,false);
    SimpleArray<COMPLEX(double),NStacks*NStacks_cached> dot(0.0);

    dslash2.Dslash(spinorOut_ref, spinorIn2,false);

    spinorOut_ref = spinorOut_ref - spinorOut2;

    dot = spinorOut_ref.dotProductStacked(spinorOut_ref);
    
    for (size_t i = 0; i < NStacks*NStacks_cached; i++) {                                                                                                                                                                      
         rootLogger.info("Testing for correctness: dot prod of difference = ", dot[i]);                                                                                                                           
     }    
   

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
    rootLogger.info("Testing Even - Odd");
    rootLogger.info("------------------");
#ifdef USE_TILED_MULTIRHS
    rootLogger.info("--------------------------------------");
    rootLogger.info("--------TESTING 1 STACK---------------");
    rootLogger.info("--------------------------------------");    
    test_dslash<float, Even, Odd, 1, 1, true>(commBase, Vol);

    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 2 STACKS--------------");
    rootLogger.info("--------------------------------------");  
    test_dslash<float, Even, Odd, 1, 2, true>(commBase, Vol); 
    test_dslash<float, Even, Odd, 2, 1, true>(commBase, Vol);

    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 3 STACKS--------------");
    rootLogger.info("--------------------------------------");  
    test_dslash<float, Even, Odd, 1, 3, true>(commBase, Vol); 
    test_dslash<float, Even, Odd, 3, 1, true>(commBase, Vol);
    
    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 4 STACKS--------------");
    rootLogger.info("--------------------------------------");  
    test_dslash<float, Even, Odd, 1, 4, true>(commBase, Vol); 
    test_dslash<float, Even, Odd, 2, 2, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 1, 4, true>(commBase, Vol);


    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 5 STACKS--------------");
    rootLogger.info("--------------------------------------");  
    test_dslash<float, Even, Odd, 1, 5, true>(commBase, Vol); 
    test_dslash<float, Even, Odd, 5, 1, true>(commBase, Vol);
    


    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 6 STACKS--------------");
    rootLogger.info("--------------------------------------");  
    test_dslash<float, Even, Odd, 1, 6, true>(commBase, Vol); 
    test_dslash<float, Even, Odd, 2, 3, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 3, 2, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 6, 1, true>(commBase, Vol);



    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 8 STACKS--------------");
    rootLogger.info("--------------------------------------");
    test_dslash<float, Even, Odd, 2, 4, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 4, 2, true>(commBase, Vol);

    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 10 STACKS-------------");
    rootLogger.info("--------------------------------------");
    test_dslash<float, Even, Odd, 2, 5, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 5, 2, true>(commBase, Vol);
    


    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 12 STACKS-------------");
    rootLogger.info("--------------------------------------");
    test_dslash<float, Even, Odd, 2, 6, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 3, 4, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 4, 3, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 6, 2, true>(commBase, Vol);



    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 15 STACKS-------------");
    rootLogger.info("--------------------------------------");  
    test_dslash<float, Even, Odd, 3, 5, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 5, 3, true>(commBase, Vol);

    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 16 STACKS-------------");
    rootLogger.info("--------------------------------------");  
    test_dslash<float, Even, Odd, 2, 8, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 4, 4, true>(commBase, Vol);


#else
    rootLogger.info("--------------------------------------");
    rootLogger.info("--------TESTING 1 STACK---------------");
    rootLogger.info("--------------------------------------");    
    test_dslash<float, Even, Odd, 1, 1, true>(commBase, Vol);

    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 2 STACKS--------------");
    rootLogger.info("--------------------------------------");  
    test_dslash<float, Even, Odd, 1, 2, true>(commBase, Vol); 
    test_dslash<float, Even, Odd, 2, 1, true>(commBase, Vol);
    
    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 4 STACKS--------------");
    rootLogger.info("--------------------------------------");  
    test_dslash<float, Even, Odd, 1, 4, true>(commBase, Vol); 
    test_dslash<float, Even, Odd, 2, 2, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 4, 1, true>(commBase, Vol);


    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 12 STACKS-------------");
    rootLogger.info("--------------------------------------");
    test_dslash<float, Even, Odd, 2, 6, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 3, 4, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 4, 3, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 6, 2, true>(commBase, Vol);
#endif

}
