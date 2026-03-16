#include "../simulateqcd.h"
#include "../modules/observables/taylorMeasurement.h"

int main(int argc, char *argv[]){

    stdLogger.setVerbosity(INFO);

    TaylorMeasurementParameters param;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/applications/TaylorMeasurement.param", argc, argv);
    
    commBase.init(param.nodeDim());

    const size_t HaloDepthGauge = 2; // >= 1 for multi gpu
    const size_t HaloDepthSpin = 4;
    const size_t NStacks = 1; // NOTE: this only works for NStacks=8 after the blocksize fix
    const int numVec = 10;
    typedef float floatT; // Define the precision here

    initIndexer(HaloDepthGauge, param, commBase);

    Gaugefield<floatT,true,HaloDepthGauge> gauge(commBase);      /// gauge field
    rootLogger.info("Read configuration from ", param.GaugefileName());
    gauge.readconf_nersc(param.GaugefileName());
    gauge.updateAll();

    Eigenpairs<floatT,true,Even,HaloDepthGauge,HaloDepthSpin,NStacks> eigenpairsWrite(commBase);
    // eigenpairsWrite.lanczos(numVec);
    // eigenpairsWrite.tester(gauge);
    eigenpairsWrite.fillRandom(numVec);
    eigenpairsWrite.writeEigenpairsSequential("testEigenpairsFile", 0, ENDIAN_AUTO);

    Eigenpairs<floatT,true,Even,HaloDepthGauge,HaloDepthSpin,NStacks> eigenpairsRead(commBase);
    eigenpairsRead.readEigenpairsSequential("testEigenpairsFile");
    eigenpairsRead.updateAll();

    double lambdaWrite;
    double lambdaRead;
    double lambdaDiff;
    Spinorfield<floatT,true,Even,HaloDepthSpin,NStacks> spinorWrite(commBase);
    Spinorfield<floatT,true,Even,HaloDepthSpin,NStacks> spinorRead(commBase);
    Spinorfield<floatT,true,Even,HaloDepthSpin,NStacks> spinorDiff(commBase);

    int steps = 5;
    double stepSize = static_cast<double>(eigenpairsRead.SpinorCount() - 1) / (steps - 1);

    for (int idx = 0; idx < steps; idx++) {
        int index = static_cast<int>(idx * stepSize);
        rootLogger.info("Check Eigenpair with index " ,  index);

        eigenpairsWrite.getEigenPair(spinorWrite, lambdaWrite, idx);
        rootLogger.info("spinorWrite=", spinorWrite.realdotProduct(spinorWrite));
        rootLogger.info("lambdaWrite=", lambdaWrite);

        eigenpairsRead.getEigenPair(spinorRead, lambdaRead, idx);
        rootLogger.info("spinorRead=", spinorRead.realdotProduct(spinorRead));
        rootLogger.info("lambdaRead=", lambdaRead);

        spinorDiff = spinorWrite - spinorRead;
        lambdaDiff = lambdaWrite - lambdaRead;
        rootLogger.info("spinorDiff=", spinorDiff.realdotProduct(spinorDiff));
        rootLogger.info("lambdaDiff=", lambdaDiff);
    }
}
