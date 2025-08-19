#include "../simulateqcd.h"
#include "../modules/observables/taylorMeasurement.h"

int main(int argc, char *argv[]){

    stdLogger.setVerbosity(INFO);

    LatticeParameters param;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/tests/dslashTest.param", argc, argv);
    
    commBase.init(param.nodeDim());


    const size_t HaloDepthGauge = 2; // >= 1 for multi gpu
    const size_t HaloDepthSpin = 4;
    const size_t NStacks = 1; // NOTE: this only works for NStacks=8 after the blocksize fix
    const int numVec = 2;
    typedef float floatT; // Define the precision here
    typedef float PREC;

    initIndexer(HaloDepthGauge, param, commBase);

    Eigenpairs<PREC,true,Even,HaloDepthGauge,HaloDepthSpin,NStacks> eigenpairsWrite(commBase);
    eigenpairsWrite.fillRandom(numVec);
    eigenpairsWrite.writeEigenpairsFile("testEigenpairsFile", 0, ENDIAN_AUTO);
    eigenpairsWrite.updateAll();


    Eigenpairs<PREC,true,Even,HaloDepthGauge,HaloDepthSpin,NStacks> eigenpairsRead(commBase);
    eigenpairsRead.readEigenpairsFile("testEigenpairsFile");
    eigenpairsRead.updateAll();

    // double lambdaDiff;
    // Spinorfield<floatT,true,Even,HaloDepthSpin,NStacks> spinorDiff(commBase);

    // for (int idx = 0; idx < numVec; idx++) {
    //     rootLogger.info("pair ", idx+1);

    //     spinorDiff = eigenpairsWrite.spinor_vec[idx];
    //     rootLogger.info("spinorWrite=", spinorDiff.realdotProduct(spinorDiff));

    //     spinorDiff = eigenpairsRead.spinor_vec[idx];
    //     rootLogger.info("spinorRead=", spinorDiff.realdotProduct(spinorDiff));

    //     spinorDiff -= eigenpairsWrite.spinor_vec[idx];
    //     rootLogger.info("spinorDiff=", spinorDiff.realdotProduct(spinorDiff));

    //     lambdaDiff = eigenpairsWrite.lambda_vec[idx];
    //     rootLogger.info("lambdaWrite=", lambdaDiff);

    //     lambdaDiff = eigenpairsRead.lambda_vec[idx];
    //     rootLogger.info("lambdaRead=", lambdaDiff);

    //     lambdaDiff -= eigenpairsWrite.lambda_vec[idx];
    //     rootLogger.info("lambdaDiff=", lambdaDiff);
    // }
}