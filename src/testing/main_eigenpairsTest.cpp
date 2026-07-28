#include "../simulateqcd.h"
#include "../modules/observables/taylorMeasurement.h"
#include <fstream>
#include <iomanip>

int main(int argc, char *argv[]){

    stdLogger.setVerbosity(INFO);

    TaylorMeasurementParameters param;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/applications/TaylorMeasurement.param", argc, argv);
    
    commBase.init(param.nodeDim());

    const size_t HaloDepthGauge = 2; // >= 1 for multi gpu
    const size_t HaloDepthSpin = 4;
    const size_t NStacks = 1; // NOTE: this only works for NStacks=8 after the blocksize fix
    const int numVec = 5;
    typedef float floatT; // Define the precision here

    initIndexer(HaloDepthGauge, param, commBase);

    Gaugefield<floatT,true,HaloDepthGauge,R18> gauge(commBase);      /// gauge field
    rootLogger.info("Read configuration from ", param.GaugefileName());
    gauge.readconf_nersc(param.GaugefileName());
    gauge.updateAll();

    Gaugefield<floatT,true,HaloDepthGauge,R18> gauge_smeared(commBase);
    Gaugefield<floatT,true,HaloDepthGauge,U3R14> gauge_Naik(commBase);
    HisqSmearing<floatT, true, HaloDepthGauge, R18, R18, R18, U3R14> smearing(gauge, gauge_smeared, gauge_Naik);
    smearing.SmearAll();

    HisqDSlash<floatT,true,Even,HaloDepthGauge,HaloDepthSpin,NStacks> dslash(gauge_smeared, gauge_Naik, 0.0, 0.0);
    
    Eigenpairs<floatT,true,Even,HaloDepthGauge,HaloDepthSpin,NStacks> eigenpairsWrite(commBase);
    TRLanRestartParams lanczosParams;
    lanczosParams.krylovDim = 384;
    lanczosParams.thickRestartDim = 128;
    lanczosParams.maxRestarts = 5;
    lanczosParams.residualTol = 1e-6;
    lanczosParams.breakdownTol = 1e-12;
    lanczosParams.seed = 1234;
    lanczosParams.failOnNoConvergence = false;


    lanczosParams.chebyshev.enabled = false;
    lanczosParams.exponential.enabled = true;
    lanczosParams.exponential.order = 26;
    lanczosParams.exponential.alpha = 9.0;
    lanczosParams.exponential.beta = 1.0;

    eigenpairsWrite.lanczos(dslash, numVec, lanczosParams);
    eigenpairsWrite.checkEigenValueEquation(dslash, 0.0, 1e-5);

    if (commBase.IamRoot()) {
        std::ofstream evout("simqcd_eigenvalues.txt");
        evout << std::setprecision(17);

        for (int idx = 0; idx < eigenpairsWrite.SpinorCount(); idx++) {
            const double lambda = eigenpairsWrite.getEigenValue(idx);
            rootLogger.info("SIMQCD_EIGENVALUE ", idx, " = ", lambda);
            evout << idx << " " << lambda << "\n";
        }
    }
    eigenpairsWrite.writeEigenpairsToFile("testEigenpairsFile", 0, ENDIAN_AUTO);

    Eigenpairs<floatT,true,Even,HaloDepthGauge,HaloDepthSpin,NStacks> eigenpairsRead(commBase);
    eigenpairsRead.readEigenpairsFromFile("testEigenpairsFile");
    eigenpairsRead.updateAll();
    
    Spinorfield<floatT,true,Even,HaloDepthSpin,NStacks> spinorWrite(commBase);
    Spinorfield<floatT,true,Even,HaloDepthSpin,NStacks> spinorRead(commBase);
    Spinorfield<floatT,true,Even,HaloDepthSpin,NStacks> spinorDiff(commBase);

    for (int idx = 0; idx < eigenpairsRead.SpinorCount(); idx++) {
        eigenpairsWrite.getEigenSpinor(spinorWrite, idx);
        eigenpairsRead.getEigenSpinor(spinorRead, idx);

        spinorDiff = spinorWrite - spinorRead;
        if (spinorDiff.realdotProduct(spinorDiff) > 1e-10) {
            rootLogger.warn("Eigenpair with index ", idx, " differs between written and read version! Norm of difference: ", spinorDiff.realdotProduct(spinorDiff));
        } else {
            rootLogger.info("Eigenpair with index ", idx, " matches between written and read version. Norm of difference: ", spinorDiff.realdotProduct(spinorDiff));
        }
    }

    for (int idx = 0; idx < eigenpairsRead.SpinorCount(); idx++) {
        double lambdaWrite = eigenpairsWrite.getEigenValue(idx);
        double lambdaRead = eigenpairsRead.getEigenValue(idx);

        double lambdaDiff = lambdaWrite - lambdaRead;
        if (std::abs(lambdaDiff) > 1e-10) {
            rootLogger.warn("Eigenvalue with index ", idx, " differs between written and read version! Difference: ", lambdaDiff);
        } else {
            rootLogger.info("Eigenvalue with index ", idx, " matches between written and read version. Difference: ", lambdaDiff);
        }
    }
}
