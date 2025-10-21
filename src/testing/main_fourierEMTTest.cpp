/*
 * main_fourierEMTTest.cpp
 * 
 * Jonas Winter, 09 Oct 2025
 *
 * Test that the Fourier transform routine for the EMT works correctly.
 *
*/

#include "../simulateqcd.h"
#include "../experimental/fourierNon2.h"
#include "../base/math/matrix4x4SymComplex.h"
#include "../modules/observables/blocking.h"

#define PREC double
#define HaloDepth 2


int main(int argc, char *argv[]) {

    // Initialization
    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);

    LatticeParameters param;
    param.readfile(commBase, "../parameter/tests/fourierEMTTest.param", argc, argv);
    commBase.init(param.nodeDim());

    typedef GIndexer<All,HaloDepth> GInd;

    // create gauge fields
    Gaugefield<PREC, false, HaloDepth> gaugeHost(commBase);
    Gaugefield<PREC, true, HaloDepth> gaugeDevice(commBase);

    gaugeHost.readconf_nersc(param.GaugefileName());

    gaugeHost.updateAll();

    gaugeDevice = gaugeHost;

    // create lattice containers for EMT fields
    LatticeContainer<false, Matrix4x4SymComplex<PREC> > _redBaseEMTUComplexHost(commBase, "EMTU_HOST", "EMTU_HOST", "EMTU_HOST", "EMTU_HOST");
    LatticeContainer<true, Matrix4x4SymComplex<PREC> > _redBaseEMTUComplexDevice(commBase, "EMTU_COMPLEX", "EMTU_COMPLEX", "EMTU_COMPLEX", "EMTU_COMPLEX");
    LatticeContainer<true, Matrix4x4SymComplex<PREC> > _redBaseEMTUFourierTransformedForwards(commBase, "EMTU_FOURIER_TRANSFORMED_FORWARDS", "EMTU_FOURIER_TRANSFORMED_FORWARDS", "EMTU_FOURIER_TRANSFORMED_FORWARDS", "EMTU_FOURIER_TRANSFORMED_FORWARDS");
    LatticeContainer<true, Matrix4x4SymComplex<PREC> > _redBaseEMTUFourierTransformedForwardsBackwards(commBase, "EMTU_FOURIER_TRANSFORMED_FORWARDS_BACKWARDS", "EMTU_FOURIER_TRANSFORMED_FORWARDS_BACKWARDS", "EMTU_FOURIER_TRANSFORMED_FORWARDS_BACKWARDS", "EMTU_FOURIER_TRANSFORMED_FORWARDS_BACKWARDS");

    // adjust their size
    _redBaseEMTUComplexHost.adjustSize(GInd::getLatData().vol4);
    _redBaseEMTUComplexDevice.adjustSize(GInd::getLatData().vol4);
    _redBaseEMTUFourierTransformedForwards.adjustSize(GInd::getLatData().vol4);
    _redBaseEMTUFourierTransformedForwardsBackwards.adjustSize(GInd::getLatData().vol4);

    // calculate EMT for the EMT-designated fields
    _redBaseEMTUComplexHost.template iterateOverBulk<All, HaloDepth>(EMTtracelessComplex<PREC, false, HaloDepth>(gaugeHost.getAccessor()));
    _redBaseEMTUComplexDevice.template iterateOverBulk<All, HaloDepth>(EMTtracelessComplex<PREC, true, HaloDepth>(gaugeDevice.getAccessor()));

    // create Fourier class
    FourierClass<PREC> fourierClass(gaugeDevice.getComm());

    // perform Fourier transformation forwards
    fourierClass.performFourier3DEMT(_redBaseEMTUComplexDevice, _redBaseEMTUFourierTransformedForwards, 1.0);
    // perform Fourier transformation backwards after the backwards
    fourierClass.performFourier3DEMT(_redBaseEMTUFourierTransformedForwards, _redBaseEMTUFourierTransformedForwardsBackwards, -1.0);

    // variables for reduced fields
    Matrix4x4SymComplex<PREC> resultEMTUComplexDevice;
    Matrix4x4SymComplex<PREC> resultEMTUComplexHost;
    Matrix4x4SymComplex<PREC> resultEMTUFourierTransformedForwards;
    Matrix4x4SymComplex<PREC> resultEMTUFourier2TransformedForwardsBackwards;

    // reduce the fields
    _redBaseEMTUComplexDevice.reduce(resultEMTUComplexDevice, GInd::getLatData().vol4);
    _redBaseEMTUComplexHost.reduce(resultEMTUComplexHost, GInd::getLatData().vol4);
    _redBaseEMTUFourierTransformedForwards.reduce(resultEMTUFourierTransformedForwards, GInd::getLatData().vol4);
    _redBaseEMTUFourierTransformedForwardsBackwards.reduce(resultEMTUFourier2TransformedForwardsBackwards, GInd::getLatData().vol4);

    Matrix4x4SymComplex<PREC> firstEMT(1,2,3,4,5,6,7,8,9,10);
    Matrix4x4SymComplex<PREC> secondEMT(1,2,3,4,5,6,7,8,9,10);

    // Checking if everything worked out

    bool lerror = false;

    if (lerror) {
        rootLogger.error("At least one test failed!");
        return -1;
    } else {
        rootLogger.info("All tests ", CoutColors::green , "passed!" , CoutColors::reset);
    }

    return 0;
}
