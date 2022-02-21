/* 
 * main_dotProduct.cu                                                               
 * 
 * D. Bollweg
 * 
 */

#include "../SIMULATeQCD.h"
#include "../modules/rhmc/rhmcParameters.h"

template<class floatT, Layout LatLayout, bool onDevice>
void run_func_nostacks(CommunicationBase &commBase) {

    const int HaloDepthSpin = 4;

    StopWatch<true> timer;

    grnd_state<onDevice> d_rand;
    initialize_rng(13333+2130, d_rand);

    Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayout>(), HaloDepthSpin, 1> spinorIn(commBase);

    rootLogger.info("Randomize spinors");
    spinorIn.gauss(d_rand.state);
    GCOMPLEX(double) dot(0.0,0.0);
    
    timer.start();
    dot = spinorIn.dotProduct(spinorIn);

    timer.stop();

    rootLogger.info("dot " ,  dot);

    rootLogger.info("Time for dot-product: " ,  timer);
}


template<class floatT, Layout LatLayout, size_t NStacks, bool onDevice>
void run_func(CommunicationBase &commBase) {

    const int HaloDepthSpin = 4;
    
    StopWatch<true> timer;

    grnd_state<onDevice> d_rand;
    initialize_rng(13333+2130, d_rand);

    Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayout>(), HaloDepthSpin, NStacks> spinorIn(commBase);

    rootLogger.info("Randomize spinors");
    spinorIn.gauss(d_rand.state);
    SimpleArray<GCOMPLEX(double), NStacks> dot(0.0);
    
    timer.start();
    dot = spinorIn.dotProductStacked(spinorIn);
    timer.stop();

    for (size_t i = 0; i < NStacks; ++i) {
        rootLogger.info("dot " ,  dot[i]);
    }

    rootLogger.info("Time for dot-product: " , timer);
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(DEBUG);
        
        CommunicationBase commBase(&argc, &argv);
        RhmcParameters param;
    
        param.readfile(commBase, "../parameter/tests/MixedPrecInverterTest.param", argc, argv);
        
        RationalCoeff rat;
        
        rat.readfile(commBase, param.rat_file());
        
        commBase.init(param.nodeDim(), param.gpuTopo());

        const int HaloDepthSpin = 4;
        initIndexer(HaloDepthSpin, param, commBase);
        
        run_func_nostacks<float, Even, true>(commBase);
        run_func_nostacks<double, Even, true>(commBase);
        
        run_func<float, Even, 1, true>(commBase);
        run_func<double, Even, 1, true>(commBase);
        run_func<float, Even, 2, true>(commBase);
        run_func<double, Even, 2, true>(commBase);
        run_func<float, Even, 4, true>(commBase);
        run_func<double, Even, 4, true>(commBase);
        run_func<float, Even, 8, true>(commBase);
        run_func<double, Even, 8, true>(commBase);
        run_func<float, Even, 10, true>(commBase);
        run_func<double, Even, 10, true>(commBase);
        
        return 0;
    }
    catch (const std::runtime_error &error){
        rootLogger.error("There has been a runtime error!");
        return -1;
    }
}
