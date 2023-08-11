/*
 * main_MrhsDSlashProf.cpp
 *
 */

#include "../SIMULATeQCD.h"
#include "../modules/dslash/dslash.h"
#include "../modules/rhmc/rhmcParameters.h"
#include "../modules/HISQ/hisqSmearing.h"


//the Dslash test function. Please start reading here.
template<class floatT, Layout LatLayout, Layout LatLayoutRHS, size_t NStacks, size_t NStacks_cached, bool onDevice>
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
    HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, NStacks> dslash(gauge_smeared, gauge_Naik, 0.0);
    HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, NStacks*NStacks_cached> dslash2(gauge_smeared, gauge_Naik, 0.0);
    gpuErr = gpuGetLastError();
    if (gpuErr)
        rootLogger.info("Error in initialization of DSlash");

    // for (int i = 0; i < 5; ++i) {
    //     timer.start();
    //     dslash.Dslash(spinorOut3, spinorIn, false);
    //     timer.stop();
    //     // spinorIn=spinorSave;
    // }

    // rootLogger.info("Time for 5 applications of multiRHS Dslash: " ,  timer);

    float EOfactor = ((LatLayout == Even || LatLayout == Odd) ? 0.5 : 1.0);

//     // float TFlops = NStacks * Vol * EOfactor * 5 * 2316 /(timer.milliseconds() * 1e-3)*1e-12;
//    float TFlops = NStacks * Vol * EOfactor * 5 * 1146 /(timer.milliseconds() * 1e-3)*1e-12;
   
    // rootLogger.info("Achieved TFLOP/s " ,  TFlops);


    timer.reset();
    for (int i = 0; i < 5; ++i) {
        timer.start();
        dslash.template Dslash_stacked<NStacks_cached>(spinorOut2,spinorIn2,false);
        timer.stop();

    }


    rootLogger.info("Time for 5 applications of multiRHS Dslash (thread+block version): ", timer);
    float TFlops = NStacks * NStacks_cached * Vol * EOfactor * 5 * 1146 /(timer.milliseconds() * 1e-3)*1e-12;
    rootLogger.info("Achieved TFLOP/s ", TFlops, " with ", NStacks, " Stacks (thread loop) and ", NStacks_cached, " Stacks (blockDim.y)");

    dslash.template Dslash_stacked<NStacks_cached>(spinorOut2,spinorIn2,false);
    SimpleArray<GCOMPLEX(double),NStacks*NStacks_cached> dot(0.0);

    dslash2.Dslash(spinorOut_ref, spinorIn2,false);

    spinorOut_ref = spinorOut_ref - spinorOut2;

    dot = spinorOut_ref.dotProductStacked(spinorOut_ref);
    
    for (int i = 0; i < NStacks*NStacks_cached; i++) {                                                                                                                                                                      
         rootLogger.info("Testing for correctness: dot prod of difference = ", dot[i]);                                                                                                                           
     }    
    
    // timer.reset();
    // for (int i = 0; i < 5; ++i) {
    //     timer.start();
    //     dslash.Dslash_threadRHS(spinorOut3,spinorIn,false);
    //     timer.stop();

    // }


    // rootLogger.info("Time for 5 applications of multiRHS Dslash (pure thread version): ", timer);
    // TFlops = NStacks * Vol * EOfactor * 5 * 1146 /(timer.milliseconds() * 1e-3)*1e-12;
    // rootLogger.info("Achieved TFLOP/s ", TFlops);

    // SimpleArray<GCOMPLEX(double), NStacks> dot2(0.0);
    // dslash.Dslash(spinorOut4, spinorIn, false);


    // spinorOut4 = spinorOut4 - spinorOut3;
    // dot2 = spinorOut4.dotProductStacked(spinorOut4);


    // for (int i = 0; i < NStacks; i++) {
    //     rootLogger.info("Tesing for correcting dot(difference) = ", dot2[i]);
    // }

    // dslash.Dslash(spinorOut2,spinorIn,false);
    // SimpleArray<GCOMPLEX(double), NStacks> dot(0.0);
    // SimpleArray<GCOMPLEX(double), NStacks> dot2(0.0);
    
    // // dslash.Dslash_stackloop(spinorOut3,spinorIn,false);
    // // dot2 = spinorOut3.dotProductStacked(spinorOut3);

    
    // spinorOut2 = spinorOut2 - spinorOut3;
    // dot = spinorOut2.dotProductStacked(spinorOut2);

    //  for (int i = 0; i < NStacks; i++) {
    //     rootLogger.info("Testing for correctness: dot(difference 1-3) = ", dot[i]);
    // }   


    // dslash.Dslash(spinorOut2,spinorIn,false);
    // dslash.template Dslash_stacked<NStacks_cached>(spinorOut3,spinorIn,false);
    // spinorOut2 = spinorOut2 - spinorOut3;

    // dot = spinorOut2.dotProductStacked(spinorOut2);
    // for (int i = 0; i < NStacks; i++) {
    //     rootLogger.info("Testing for correctness: dot(difference 1-2) = ", dot[i]);
    // }  

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
    rootLogger.info("--------Testing 7 STACKS--------------");
    rootLogger.info("--------------------------------------");  
    test_dslash<float, Even, Odd, 1, 7, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 7, 1, true>(commBase, Vol);



    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 8 STACKS--------------");
    rootLogger.info("--------------------------------------");  
    test_dslash<float, Even, Odd, 1, 8, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 2, 4, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 4, 2, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 8, 1, true>(commBase, Vol);

    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 9 STACKS--------------");
    rootLogger.info("--------------------------------------");  
    test_dslash<float, Even, Odd, 1, 9, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 9, 1, true>(commBase, Vol);

    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 10 STACKS-------------");
    rootLogger.info("--------------------------------------");  
    test_dslash<float, Even, Odd, 1, 10, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 2, 5, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 5, 2, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 10, 1, true>(commBase, Vol);
    
    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 11 STACKS-------------");
    rootLogger.info("--------------------------------------");  
    test_dslash<float, Even, Odd, 1, 11, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 11, 1, true>(commBase, Vol);


    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 12 STACKS-------------");
    rootLogger.info("--------------------------------------");  
    test_dslash<float, Even, Odd, 1, 12, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 2, 6, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 3, 4, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 4, 3, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 6, 2, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 12, 1, true>(commBase, Vol);


    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 14 STACKS-------------");
    rootLogger.info("--------------------------------------");  
    test_dslash<float, Even, Odd, 1, 14, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 2, 7, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 7, 2, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 14, 1, true>(commBase, Vol);


    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 15 STACKS-------------");
    rootLogger.info("--------------------------------------");  
    test_dslash<float, Even, Odd, 1, 15, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 3, 5, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 5, 3, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 15, 1, true>(commBase, Vol);

    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 16 STACKS-------------");
    rootLogger.info("--------------------------------------");  
    test_dslash<float, Even, Odd, 1, 16, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 2, 8, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 4, 4, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 8, 2, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 16, 1, true>(commBase, Vol);


    rootLogger.info("--------------------------------------");
    rootLogger.info("--------Testing 32 STACKS-------------");
    rootLogger.info("--------------------------------------");  
    test_dslash<float, Even, Odd, 1, 32, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 2, 16, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 4, 8, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 8, 4, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 16, 2, true>(commBase, Vol);
    test_dslash<float, Even, Odd, 32, 1, true>(commBase, Vol);



}
