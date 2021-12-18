/*
 * main_ColorElectricCorrTest.cu
 *
 * v1.0: L. Altenkort, 14 Jan 2019
 *
 * Tests the Color-Electric Correlator (ColorElectricCorr) using the multi-GPU framework. Initialization copied from
 * main_polyakovloop.cu.
 * Read sketch from right to left (time advances to the left, space advances to the top)
 *          <----   <------  ^ <---^
 *          |  -  |          |  -  |   +  flipped = "going downwards" + "going upwards"
 * <------  v <---v          <----
 *
 */
#include "../SIMULATeQCD.h"
#include "../modules/observables/ColorElectricCorr.h"

#define PREC double
#define ON_DEVICE true

int main(int argc, char *argv[]) {

    const size_t HaloDepth = 1;

    ///Initialize Base
    typedef GIndexer<All,HaloDepth> GInd;
    stdLogger.setVerbosity(INFO);
    StopWatch                              timer;
    LatticeParameters                       lp;
    CommunicationBase                       commBase(&argc, &argv);
    lp.readfile(commBase, "../parameter/tests/ColorElectricCorrTest.param", argc, argv);
    commBase.init(lp.nodeDim());
    initIndexer(HaloDepth,lp,commBase);
    Gaugefield<PREC,ON_DEVICE,HaloDepth>     gauge(commBase);

    const size_t Ntau  = GInd::getLatData().lt;

    rootLogger.info("Reference values in this test come from l328f21b6285m0009875m0790a_019.995");
    rootLogger.info("Read configuration" ,  lp.GaugefileName());
    gauge.readconf_nersc(lp.GaugefileName());

    /// Calculate and report ColorElectricCorr.
    ColorElectricCorr<PREC,ON_DEVICE,HaloDepth> CEC(gauge);
    //ColorElectricMagneticCorr<PREC, ON_DEVICE, HaloDepth> CECM(gauge);
    rootLogger.info("Calculating color electric correlator (ColorElectricCorr)...");
    timer.start();
    gauge.updateAll();
    std::vector<GCOMPLEX(PREC)> result = CEC.getColorElectricCorr_naive();
    //std::vector<Matrix4x4Sym<PREC>> result = CECM.getColorElectricMagneticCorr();
    timer.stop();
    rootLogger.info("Time for ColorElectricCorr: " ,  timer);

    ///reference for l328f21b6285m0009875m0790a_019.995
    std::vector<PREC> reference_result = { -0.0036785027192919143857, -0.0018776993246428395554, -0.00098076311144967278852,
                                           -0.00086320431721050657099 };
    for ( PREC& val : reference_result ){
        val /= -6; //! this is to reflect the normalization, which is not included in the above values
    }

    bool failed = false;
    PREC tolerance = 1e-7;
    rootLogger.info("====== Results are said to be equal to reference if difference is less than 1e-7 ======");
    for(size_t i = 0; i < Ntau/2; ++i) {
        rootLogger.info(std::setprecision(20) ,  "ColorElectricCorr(" ,  i+1 ,  ") =     " ,  real(result[i]));
        //rootLogger.info(std::setprecision(20) ,  "ColorElectricCorr(" ,  i+1 ,  ") =     " ,  result[i].elems[0]);
        if ( !(isApproximatelyEqual(real(result[i]), reference_result[i], tolerance)) ){
        //if ( !(isApproximatelyEqual(result[i].elems[0], reference_result[i], tolerance)) ){
            failed = true;
            rootLogger.info( "ColorElectricCorr_ref(" ,  i+1 ,  ") = " ,  reference_result[i]);
            rootLogger.info( "Difference to reference is  " ,  std::fabs(real(result[i])
            //rootLogger.info() ,   "Difference to reference is  " ,  std::fabs(result[i].elems[0]
            -reference_result[i]));
            rootLogger.error("--- Failed for dt = " ,  i+1 ,  "! ---");

        } else {
            rootLogger.info( "ColorElectricCorr_ref(" ,  i+1 ,  ") = " ,  reference_result[i]);
            rootLogger.info( "Difference to reference is  " ,  std::fabs(real(result[i])
            //rootLogger.info() ,   "Difference to reference is  " ,  std::fabs(result[i].elems[0]
                                                                              -reference_result[i]));
            rootLogger.info("--- Passed for dt = " ,  i+1 ,  "! ---");
        }
    }
    if (failed){
        rootLogger.info("======== ",  CoutColors::red ,  "FAILED" ,  CoutColors::reset
                           ,  " TESTS FOR COLOR-ELECTRIC CORRELATOR!!! ========");

    } else {
        rootLogger.info("======== ALL TESTS FOR COLOR-ELECTRIC CORRELATOR ",  CoutColors::green , "PASSED!"
                          ,  CoutColors::reset ,  " ========");
    }

    return 0;
}

