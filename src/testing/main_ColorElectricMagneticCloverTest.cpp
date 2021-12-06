/* 
 * main_ColorElectricMagneticCloverTest.cpp                                                               
 *
 * Hai Tao Shu, 7 Apr 2021
 *
 */

#include "../SIMULATeQCD.h"
#include "../modules/observables/ColorElectricCorr.h"
#include "../modules/observables/ColorMagneticCorr.h"

#define PREC double
#define ON_DEVICE true

int main(int argc, char *argv[]) {

    const size_t HaloDepth = 1;

    ///Initialize Base
    typedef GIndexer<All,HaloDepth> GInd;
    stdLogger.setVerbosity(INFO);
    MicroTimer                              timer;
    LatticeParameters                       lp;
    CommunicationBase                       commBase(&argc, &argv);
    lp.readfile(commBase, "../parameter/tests/ColorElectricMagneticCloverTest.param", argc, argv);
    commBase.init(lp.nodeDim());
    initIndexer(HaloDepth,lp,commBase);
    Gaugefield<PREC,ON_DEVICE,HaloDepth>     gauge(commBase);

    const size_t Ntau  = GInd::getLatData().lt;

    rootLogger.info("Read configuration" ,  lp.GaugefileName());
    gauge.readconf_nersc(lp.GaugefileName());

    /// Calculate and report ColorElectricCorr.
    ColorElectricCorr<PREC,ON_DEVICE,HaloDepth> CEC(gauge);
    ColorMagneticCorr<PREC,ON_DEVICE,HaloDepth> CMC(gauge);
    timer.start();
    gauge.updateAll();

    std::vector<GCOMPLEX(PREC)> resultColElecCorSl_clover;
    std::vector<GCOMPLEX(PREC)> resultColMagnCorSl_clover;

    timer.start();
    resultColElecCorSl_clover = CEC.getColorElectricCorr_clover();
    timer.stop();
    rootLogger.info("Time for clover color-electric corrs: " ,  timer);
    timer.reset();
    timer.start();
    resultColMagnCorSl_clover = CMC.getColorMagneticCorr_clover();
    timer.stop();
    rootLogger.info("Time for clover color-magnetic corrs: " ,  timer);


    return 0;
}

