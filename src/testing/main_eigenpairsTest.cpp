#include "../simulateqcd.h"
#include "../modules/observables/taylorMeasurement.h"

int main(int argc, char *argv[]){

    stdLogger.setVerbosity(INFO);

    TaylorMeasurementParameters param;
    CommunicationBase commBase(&argc, &argv);
    
    // try reading parameter file from the same directory 
    rootLogger.info("Reading parameter file \"TaylorMeasurement.param\" from the current working directory.");
    param.readfile(commBase, "../parameter/applications/TaylorMeasurement.param", argc, argv);
 
    
    commBase.init(param.nodeDim());


    const size_t HaloDepthGauge = 2; // >= 1 for multi gpu
    const size_t HaloDepthSpin = 4;
    const size_t NStacks = 1; // NOTE: this only works for NStacks=8 after the blocksize fix
    const int numVec = 1;
    typedef float floatT; // Define the precision here
    typedef float PREC;

    initIndexer(HaloDepthGauge, param, commBase);

    Eigenpairs<PREC,true,All,HaloDepthGauge,HaloDepthSpin,NStacks> eigenpairs0(commBase);
    eigenpairs0.fillRandom(numVec);
    eigenpairs0.writeEvNersc("testEvFile");
    eigenpairs0.updateAll();


    Eigenpairs<PREC,true,All,HaloDepthGauge,HaloDepthSpin,NStacks> eigenpairs1(commBase);
    eigenpairs1.readEvNersc("testEvFile", numVec);
    eigenpairs1.updateAll();

    double lambda_diff;
    Spinorfield<floatT,true,All,HaloDepthSpin,NStacks> spinor0(commBase);
    Spinorfield<floatT,true,All,HaloDepthSpin,NStacks> spinor1(commBase);
    Spinorfield<floatT,true,All,HaloDepthSpin,NStacks> spinor_diff(commBase);

    for (int idx = 0; idx < numVec; idx++) {
        rootLogger.info("pair ", idx);

        spinor0 = eigenpairs0.spinor_vec[idx];
        rootLogger.info("spinor0= ", spinor0.realdotProduct(spinor0));

        spinor1 = eigenpairs1.spinor_vec[idx];
        rootLogger.info("spinor1= ", spinor1.realdotProduct(spinor1));

        spinor_diff = spinor0;
        spinor_diff -= spinor1;
        rootLogger.info("spinor_diff= ", spinor_diff.realdotProduct(spinor_diff));

        lambda_diff = eigenpairs0.lambda_vec[idx] - eigenpairs1.lambda_vec[idx];
        rootLogger.info("lambda_diff= ", lambda_diff);
    }
}