/* 
 * main_half_prec_dslash_test.cu                                                               
 * 
 * Dennis Bollweg
 * 
 */

#include "../SIMULATeQCD.h"
#include "../modules/dslash/dslash.h"
#include "../modules/rhmc/rhmcParameters.h"
#include "../modules/HISQ/hisqSmearing.h"

/*
template<class floatT_source, class floatT_target, bool onDevice, size_t HaloDepth, CompressionType comp>
struct convert_precision {
    gaugeAccessor<floatT_source,comp> gAcc_source;

    convert_precision(Gaugefield<floatT_source, onDevice, HaloDepth, comp> &gaugeIn) : gAcc_source(gaugeIn.getAccessor()) {}

    __device__ __host__ GSU3<floatT_target> operator()(gSiteMu site) {
        GSU3<floatT_source> source = gAcc_source.getLink(site);
        GSU3<floatT_target> target(source);
        return target;
    }
    };

template<class floatT_source, class floatT_target, bool onDevice, Layout LatLayout, size_t HaloDepthSpin, size_t NStacks>
struct convert_spinor_precision {
    gVect3arrayAcc<floatT_source> spinor_source;

    convert_spinor_precision(Spinorfield<floatT_source, onDevice, LatLayout, HaloDepthSpin, NStacks> &spinorIn) : spinor_source(spinorIn.getAccessor()) {}

    __device__ __host__ gVect3<floatT_target> operator()(gSite site) {
        gVect3<floatT_source> source = spinor_source.getElement(site);
        gVect3<floatT_target> target(source);
        return target;
    }
};
*/
//the Dslash test function. Please start reading here.
template<class floatT, Layout LatLayout, Layout LatLayoutRHS, size_t NStacks, bool onDevice>
void test_dslash(CommunicationBase &commBase, int Vol){

    //Initialization as usual

    gpuEvent_t start, stop;
    gpuEventCreate(&start);
    gpuEventCreate(&stop);

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
    rootLogger.info() << "Starting Test with " << NStacks << " Stacks";
    rootLogger.info() << "Initialize random state";
    grnd_state<false> h_rand;
    grnd_state<onDevice> d_rand;

    h_rand.make_rng_state(1337);
    d_rand = h_rand;

    rootLogger.info() << "gen conf";

    // gauge.readconf_nersc("../test_conf/gauge12750");
    gauge.random(d_rand.state);

    gpuError_t gpuErr = gpuGetLastError();
        if (gpuErr)
            // GpuError("Error in random gauge field", gpuErr);
            rootLogger.info() << "Error in random gauge field";

    gauge.updateAll();

    gpuErr = gpuGetLastError();
        if (gpuErr)
            // GpuError("Error in updateAll", gpuErr);
            rootLogger.info() << "Error updateAll";

    smearing.SmearAll();

    gpuErr = gpuGetLastError();
    if (gpuErr)
        // GpuError("error in smearing", gpuErr);
        rootLogger.info() << "Error in smearing";
    //convert_precision<float,__half,HaloDepth,R18> convert_struct(gauge_smeared);
        
    gauge_smeared_half.convert_precision<floatT>(gauge_smeared);
    gauge_Naik_half.convert_precision<floatT>(gauge_Naik);

//gauge_smeared.iterateOverBulkAllMu(convert_precision<__half,floatT,true, HaloDepth,R18>(gauge_smeared_half));
    rootLogger.info() << "Initialize spinors";
    Spinorfield<__half, true, LatLayoutRHS, HaloDepthSpin, NStacks> spinorIn(commBase);
    Spinorfield<__half, true, LatLayoutRHS, HaloDepthSpin, NStacks> spinorSave(commBase);
    Spinorfield<__half, true, LatLayoutRHS, HaloDepthSpin, NStacks> spinorOut(commBase);

    gpuErr = gpuGetLastError();
        if (gpuErr)
            GpuError("error in spinor Initialization", gpuErr);

    rootLogger.info() << "Randomize spinors";
    spinorIn.gauss(d_rand.state);
    spinorSave = spinorIn;
    gpuErr = gpuGetLastError();
        if (gpuErr)
           // GpuError("error in gaussian spinors", gpuErr);
            rootLogger.info() << "error in gaussian spinors";


    rootLogger.info() << "Initialize DSlash";
    HisqDSlash<__half, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, NStacks> dslash(gauge_smeared_half, gauge_Naik_half, 0.0);
    
    gpuErr = gpuGetLastError();
        if (gpuErr)
            // GpuError("error in Initialization of DSlash", gpuErr);
            rootLogger.info() << "Error in Initialization of DSlash";

    gpuEventRecord(start);
    for (int i = 0; i < 500; ++i)
    {
        // spinorIn.updateAll(COMM_BOTH | Hyperplane);
        dslash.applyMdaggM(spinorOut, spinorIn, false);
        spinorIn=spinorSave;
    }
    gpuEventRecord(stop);
    gpuEventSynchronize(stop);
    float milliseconds = 0;
    gpuEventElapsedTime(&milliseconds, start, stop);
    


    Spinorfield<floatT, true, LatLayoutRHS, HaloDepthSpin, NStacks> spinorTest(commBase);
    spinorTest.template convert_precision<__half>(spinorOut);
    SimpleArray<GCOMPLEX(double), NStacks> dot(0);
    SimpleArray<double, NStacks> norm(0.0);
    dot = spinorTest.dotProductStacked(spinorTest);
    norm = real<double>(dot);
    SimpleArray<double, NStacks> Vol_Stack(Vol);
    norm = norm / Vol_Stack;
    
    // gpuProfilerStop();
    rootLogger.info() << "Time for 500 applications of multiRHS Dslash: " << milliseconds;
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wint-in-bool-context"
    float EOfactor = (LatLayout == (Even || Odd) ? 0.5 : 1.0);
    #pragma GCC diagnostic pop
    float TFlops = NStacks * Vol * EOfactor * 500 * 2316 /(milliseconds * 1e-3)*1e-12;
    rootLogger.info() << "Achieved TFLOP/s " << TFlops;

    for (int i = 0; i < NStacks; i++) {
        rootLogger.info() << "dot [" << i <<"] = "<< norm[i];
    }

}

int main(int argc, char **argv) {

    stdLogger.setVerbosity(INFO);
    CommunicationBase commBase(&argc, &argv);

    LatticeParameters param;
    param.readfile(commBase, "../parameter/tests/InverterTest.param", argc, argv);
    const int LatDim[] = {96, 96, 96, 16};
    
    
    // const int NodeDim[] = {1, 1, 1, 1};
    int Vol = LatDim[0]*LatDim[1]*LatDim[2]*LatDim[3];
    
    
    param.latDim.set(LatDim);
    // param.nodeDim.set(NodeDim);

    commBase.init(param.nodeDim());

    const int HaloDepthSpin = 4;
    initIndexer(HaloDepthSpin,param, commBase);
    stdLogger.setVerbosity(INFO);

    rootLogger.info() << "-------------------------------------";
    rootLogger.info() << "Running on Device";
    rootLogger.info() << "-------------------------------------";
    rootLogger.info() << "Testing Even - Odd";
    rootLogger.info() << "------------------";
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
    

/// Apparently the host has trouble to store a configuration.
//    rootLogger.info() << "-------------------------------------";
//    rootLogger.info() << "Running on Host";
//    rootLogger.info() << "-------------------------------------";
//    rootLogger.info() << "Testing Even - Odd";
//    rootLogger.info() << "------------------";
//    test_dslash<float, Even, Odd, 1, false>(commBase);
//    rootLogger.info() << "------------------";
//    rootLogger.info() << "Testing Odd - Even";
//    rootLogger.info() << "------------------";
//    test_dslash<float, Odd, Even, 1, false>(commBase);
}


template<Layout LatLayout, size_t HaloDepth>
size_t getGlobalIndex(LatticeDimensions coord) {
    typedef GIndexer<LatLayout, HaloDepth> GInd;

    LatticeData lat = GInd::getLatData();
    LatticeDimensions globCoord = lat.globalPos(coord);

    return globCoord[0] + globCoord[1] * lat.globLX + globCoord[2] * lat.globLX * lat.globLY +
           globCoord[3] * lat.globLX * lat.globLY * lat.globLZ;
}