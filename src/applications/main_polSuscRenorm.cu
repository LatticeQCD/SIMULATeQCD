/* 
 * main_polSuscRenorm.cu
 *
 * D. Clarke
 *
 * GPU code to remove short distance contributions from Polyakov loop susceptibility.
 *
 */

#include "../modules/observables/PolyakovLoop.h"

#include <iostream>
#include <unistd.h>

#define PREC double 
#define MY_BLOCKSIZE 256

/// Custom parameter structure for this program.
template<class floatT>
struct ploopParam : LatticeParameters {
    Parameter<std::string>  cml;
    Parameter<std::string>  cms;
    ploopParam() {
        addOptional(cml,"mlight");
        addOptional(cms,"mstrange");
    }
};

int main(int argc, char *argv[]) {

    PREC norm, suscL, suscT;
    GCOMPLEX(PREC) suscA, PLoop;

    /// General initialization
    stdLogger.setVerbosity(INFO);
    const size_t HaloDepth  = 0;
    rootLogger.info("Initialization...");
    CommunicationBase commBase(&argc, &argv);
    ploopParam<PREC> param;
    param.readfile(commBase, "../parameter/applications/polSuscRenorm.param", argc, argv);
    commBase.init(param.nodeDim());
    initIndexer(HaloDepth,param,commBase);
    StopWatch timer;

    Gaugefield<PREC,true,HaloDepth>      gauge(commBase);   /// gauge field
    PolyakovLoop<PREC,true,HaloDepth>    ploopClass(gauge); /// for measuring Polyakov loops
    CorrelatorTools<PREC,true,HaloDepth> corrTools;         /// for measuring correlators

    if( (corrTools.Nx != corrTools.Ny) || (corrTools.Ny != corrTools.Nz) ) {
        throw std::runtime_error(stdLogger.fatal("Need Nx=Ny=Nz."));
    }

    rootLogger.info("Read configuration " ,  param.GaugefileName());
    gauge.readconf_nersc(param.GaugefileName());
    gauge.updateAll();

    CorrField<false,GSU3<PREC>>       thermalWilsonLine(commBase, corrTools.vol3);
    Correlator<false,GCOMPLEX(PREC)>  ABareSusc(commBase, corrTools.USr2max);
    Correlator<false,PREC>            LBareSusc(commBase, corrTools.USr2max);
    Correlator<false,PREC>            TBareSusc(commBase, corrTools.USr2max);
    Correlator<false,PREC>            CPUnorm(commBase, corrTools.USr2max);

    LatticeContainerAccessor _thermalWilsonLine(thermalWilsonLine.getAccessor());
    LatticeContainerAccessor _ABareSusc(ABareSusc.getAccessor());
    LatticeContainerAccessor _LBareSusc(LBareSusc.getAccessor());
    LatticeContainerAccessor _TBareSusc(TBareSusc.getAccessor());
    LatticeContainerAccessor _CPUnorm(CPUnorm.getAccessor());

    ploopClass.PloopInArray(_thermalWilsonLine);

    timer.start();
    corrTools.correlateAt<GSU3<PREC>,GCOMPLEX(PREC),trAxtrBt<PREC>>("spatial", thermalWilsonLine, thermalWilsonLine, CPUnorm, ABareSusc);
    /// Here we have A and B the same operator, so XYswapSymmetry == True here.
    corrTools.correlateAt<GSU3<PREC>,PREC,trReAxtrReB<PREC>>("spatial", thermalWilsonLine, thermalWilsonLine, CPUnorm, LBareSusc, true);
    corrTools.correlateAt<GSU3<PREC>,PREC,trImAxtrImB<PREC>>("spatial", thermalWilsonLine, thermalWilsonLine, CPUnorm, TBareSusc, true);
    timer.stop();

    for(int ir2=0; ir2<corrTools.USr2max+1; ir2++) {
        _CPUnorm.getValue<PREC>(ir2,norm);
        if(norm > 0) {
            _ABareSusc.getValue<GCOMPLEX(PREC)>(ir2,suscA);
            _LBareSusc.getValue<PREC>(ir2,suscL);
            _TBareSusc.getValue<PREC>(ir2,suscT);
            rootLogger.info("r**2, A, L, T : " ,  ir2 ,  ": " ,  real(suscA) ,  ": " ,  suscL ,  ": " ,  suscT);
        }
    }

    rootLogger.info("Time to calculate bare susceptibility correlators: " ,  timer);

    PLoop = ploopClass.getPolyakovLoop();

    rootLogger.info(" P = " ,  PLoop);

    return 0;
}

