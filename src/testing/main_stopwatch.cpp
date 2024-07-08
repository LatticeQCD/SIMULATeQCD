#include "../simulateqcd.h"
#include <chrono>
#include<thread>

int main(int argc, char **argv) {
    stdLogger.setVerbosity(DEBUG);

    
    LatticeParameters param;
    const int  LatDim[]   = {20,20,20,20};
    const int  NodeDim[]  = {1 ,1 ,1 ,1};
    const int HaloDepth = 1;
    param.latDim.set(LatDim);
    param.nodeDim.set(NodeDim);
    CommunicationBase commBase(&argc, &argv);
    commBase.init(param.nodeDim());
    initIndexer(HaloDepth,param,commBase);

    StopWatch<true> timer;

    timer.start();

    std::this_thread::sleep_for(std::chrono::minutes(2));
    std::this_thread::sleep_for(std::chrono::seconds(50));

    timer.stop();

    rootLogger.info("Time elapsed: ", timer);

    return 0;
    
}