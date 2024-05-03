/*
 * main_ColorElectricMagneticCloverBenchmark.cpp
 *
 * Hai Tao Shu, 7 Apr 2021
 *
 */

#include "../simulateqcd.h"
#include "../modules/observables/colorElectricCorr.h"
#include "../modules/observables/colorMagneticCorr.h"

#define PREC double
#define ON_DEVICE true

int main(int argc, char *argv[]) {

    const size_t HaloDepth = 1;

    ///Initialize Base
    typedef GIndexer<All,HaloDepth> GInd;
    stdLogger.setVerbosity(INFO);
    StopWatch<true> timer;
    LatticeParameters lp;
    CommunicationBase commBase(&argc, &argv);
    lp.readfile(commBase, "../parameter/tests/colorElectricMagneticCloverTest.param", argc, argv);
    commBase.init(lp.nodeDim());
    initIndexer(HaloDepth,lp,commBase);
    Gaugefield<PREC,ON_DEVICE,HaloDepth> gauge(commBase);

    const size_t Ntau  = GInd::getLatData().lt;

    rootLogger.info("Read configuration" ,  lp.GaugefileName());
    gauge.readconf_nersc(lp.GaugefileName());

    /// Calculate and report ColorElectricCorr.
    ColorElectricCorr<PREC,ON_DEVICE,HaloDepth> CEC(gauge);
    ColorMagneticCorr<PREC,ON_DEVICE,HaloDepth> CMC(gauge);
    timer.start();
    gauge.updateAll();

    std::vector<COMPLEX(PREC)> resultColElecCorSl_clover;
    std::vector<COMPLEX(PREC)> resultColMagnCorSl_clover;

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

