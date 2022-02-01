/* 
 * main_half_prec_dslash_profile.cu                                                               
 * 
 * Dennis Bollweg
 * 
 */

#include "../SIMULATeQCD.h"
#include "../modules/dslash/dslash.h"
#include "../modules/rhmc/rhmcParameters.h"
#include "../modules/HISQ/hisqSmearing.h"

//the Dslash test function. Please start reading here.
template<class floatT, Layout LatLayout, Layout LatLayoutRHS, size_t NStacks, bool onDevice>
void test_dslash(CommunicationBase &commBase, int Vol){

    //Initialization as usual

    StopWatch<true> timer;

    const int HaloDepth = 2;
    const int HaloDepthSpin = 4;
    typedef GIndexer<LatLayout, HaloDepth> GInd;

    //Our gaugefield
    Gaugefield<floatT, onDevice, HaloDepth, R14> gauge(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, R18> gauge_smeared(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, U3R14> gauge_Naik(commBase);

    Gaugefield<__half, true, HaloDepth, R18> gauge_smeared_half(commBase);
    Gaugefield<__half, true, HaloDepth, U3R14> gauge_Naik_half(commBase);
    
    HisqSmearing<floatT, onDevice, HaloDepth> smearing(gauge, gauge_smeared, gauge_Naik);
    rootLogger.info("Starting Test with " ,  NStacks ,  " Stacks");
    rootLogger.info("Initialize random state");
    grnd_state<false> h_rand;
    grnd_state<onDevice> d_rand;

    h_rand.make_rng_state(1337);
    d_rand = h_rand;

    rootLogger.info("gen conf");

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
        
    gauge_smeared_half.convert_precision<floatT>(gauge_smeared);
    gauge_Naik_half.convert_precision<floatT>(gauge_Naik);

    rootLogger.info("Initialize spinors");
    Spinorfield<__half, true, LatLayoutRHS, HaloDepthSpin, NStacks> spinorIn(commBase);
    Spinorfield<__half, true, LatLayoutRHS, HaloDepthSpin, NStacks> spinorSave(commBase);
    Spinorfield<__half, true, LatLayoutRHS, HaloDepthSpin, NStacks> spinorOut(commBase);

    gpuErr = gpuGetLastError();
    if (gpuErr)
        GpuError("error in spinor Initialization", gpuErr);

    rootLogger.info("Randomize spinors");
    spinorIn.gauss(d_rand.state);
    spinorSave = spinorIn;
    gpuErr = gpuGetLastError();
    if (gpuErr)
        rootLogger.info("error in gaussian spinors");


    rootLogger.info("Initialize DSlash");
    HisqDSlash<__half, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, NStacks> dslash(gauge_smeared_half, gauge_Naik_half, 0.0);
    
    gpuErr = gpuGetLastError();
    if (gpuErr)
        rootLogger.info("Error in Initialization of DSlash");

   
    for (int i = 0; i < 500; ++i) {
        timer.start();
        dslash.applyMdaggM(spinorOut, spinorIn, false);
        timer.stop();
        spinorIn=spinorSave;
    }
   
    


    Spinorfield<floatT, true, LatLayoutRHS, HaloDepthSpin, NStacks> spinorTest(commBase);
    spinorTest.template convert_precision<__half>(spinorOut);
    SimpleArray<GCOMPLEX(double), NStacks> dot(0);
    SimpleArray<double, NStacks> norm(0.0);
    dot = spinorTest.dotProductStacked(spinorTest);
    norm = real<double>(dot);
    SimpleArray<double, NStacks> Vol_Stack(Vol);
    norm = norm / Vol_Stack;
    
    rootLogger.info("Time for 500 applications of multiRHS Dslash: " ,  timer);
    float EOfactor = ((LatLayout == Even || LatLayout == Odd) ? 0.5 : 1.0);
 
    float TFlops = NStacks * Vol * EOfactor * 500 * 2316 /timer.seconds()*1e-12;
    rootLogger.info("Achieved TFLOP/s " ,  TFlops);

    for (int i = 0; i < NStacks; i++) {
        rootLogger.info("dot [" ,  i , "] = ",  norm[i]);
    }

}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);
        CommunicationBase commBase(&argc, &argv);

        LatticeParameters param;
        param.readfile(commBase, "../parameter/tests/InverterTest.param", argc, argv);
        const int LatDim[] = {param.latDim[0], param.latDim[1], param.latDim[2], param.latDim[3]};


        int Vol = LatDim[0] * LatDim[1] * LatDim[2] * LatDim[3];


        param.latDim.set(LatDim);

        commBase.init(param.nodeDim());

        const int HaloDepthSpin = 4;
        initIndexer(HaloDepthSpin, param, commBase);
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
    catch (const std::runtime_error &error) {
        return 1;
    }
    return 0;
}
