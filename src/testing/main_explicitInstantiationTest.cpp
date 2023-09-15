#include "../simulateqcd.h"
#include "explicitInstantiationTest.h"

int main(int argc, char *argv[]){

    stdLogger.setVerbosity(DEBUG);

    LatticeParameters param;
    const int LatDim[] = {20, 20, 20, 20};
    const int NodeDim[] = {1, 1, 1, 1};
    param.latDim.set(LatDim);
    param.nodeDim.set(NodeDim);

    CommunicationBase commBase(&argc, &argv);
    commBase.init(param.nodeDim());

    float a = 3;
    float b = 9;
    
    TestClass1<float,1> test1;
    TestClass2 test2;

    test1.add(a,b);
    test2.sub(a,b);
    
    testFunc<float,1>(a);

    rootLogger.info(CoutColors::green,"Test passed!",CoutColors::reset);
    return 0;
}
