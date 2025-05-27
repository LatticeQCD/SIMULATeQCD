#include "../simulateqcd.h"
#include "../modules/observables/taylorMeasurement.h"

int main(int argc, char *argv[]){

    stdLogger.setVerbosity(INFO);

    TaylorMeasurementParameters param;
    CommunicationBase commBase(&argc, &argv);
 
    const int NodeDim[] = {1, 1, 1, 1};
    param.nodeDim.set(NodeDim);
    commBase.init(param.nodeDim());



    const size_t HaloDepthGauge = 2; // >= 1 for multi gpu
    const size_t HaloDepthSpin = 4;
    const size_t NStacks = 1; // NOTE: this only works for NStacks=8 after the blocksize fix
    // typedef float floatT; // Define the precision here
    typedef float PREC;

    Eigenpairs<PREC,true,Even,HaloDepthGauge,HaloDepthSpin,NStacks> eigenpairs(commBase);
    eigenpairs.fillRandom(10);
    eigenpairs.writeEvNersc("testEvFile");
    eigenpairs.updateAll();


    Eigenpairs<PREC,true,Even,HaloDepthGauge,HaloDepthSpin,NStacks> eigenpairs2(commBase);
    eigenpairs2.readEvNersc("testEvFile", 10);
    eigenpairs2.updateAll();

    for (int idx = 0; idx < 0; idx++) {

    }
}