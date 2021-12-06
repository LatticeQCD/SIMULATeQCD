/* 
 * main_gfixplcTest.cu
 *
 * D. Clarke
 *
 * Quick test of some Gauge fixing functions and Polyakov loop correlator measurements. The test compares the output
 * of this program to Olaf's master thesis program, which is assumed to be correct.
 *
 */

#include "../SIMULATeQCD.h"
#include "../modules/gaugeFixing/gfix.h"
#include "../modules/gaugeFixing/PolyakovLoopCorrelator.h"

#include <iostream>
#include <unistd.h>
#include <vector>

#include "refValues_gfixplc.h"

#define PREC double 
#define MY_BLOCKSIZE 256

/// Create our own parameter structure, which inherits from LatticeParameters.
template<class floatT>
struct gfixParam : LatticeParameters {
    Parameter<floatT>       gtolerance;
    Parameter<floatT>       ptolerance;
    Parameter<int,1>        maxgfsteps;
    Parameter<int,1>        numunit;
    Parameter<std::string>  cml;
    Parameter<std::string>  cms;

    gfixParam() {
        addDefault (gtolerance,"gtolerance",1e-6);
        addDefault (ptolerance,"ptolerance",1e-3);
        addDefault (maxgfsteps,"maxgfsteps",1000);
        addDefault (numunit   ,"numunit"   ,20);
        addOptional(cml       ,"mlight");
        addOptional(cms       ,"mstrange");
    }
};

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);
    const size_t HaloDepth  = 0;

    /// Read in parameters and initialize communication base.
    rootLogger.info("Initialization");
    gfixParam<PREC> param;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/tests/gfixplcTest.param", argc, argv);
    const int  ngfstepMAX = param.maxgfsteps();
    const int  nunit      = param.numunit();
    const PREC gtol       = param.gtolerance();
    const PREC tolp       = param.ptolerance();
    commBase.init(param.nodeDim());
    initIndexer(HaloDepth,param,commBase);
    typedef GIndexer<All,HaloDepth> GInd;

    /// More initialization.
    MicroTimer                        timer;
    int                               ngfstep = 0;          /// # of gauge fixing steps
    PREC                              gftheta = 1e10;       /// gauge fixing theta
    PREC                              act     = 1.0;        /// gauge fix action after OR update
    Gaugefield<PREC,true,HaloDepth>   gauge(commBase);      /// gauge field
    GaugeFixing<PREC,true,HaloDepth>  GFixing(gauge);       /// gauge fixing class
    CorrelatorTools<PREC,true,HaloDepth>  corrTools;        /// general correlator class
    PolyakovLoopCorrelator<PREC,true,HaloDepth> PLC(gauge); /// class for Polyakov loop correlators

    /// Read the configuration. Remember a halo exchange is needed every time the gauge field changes.
    rootLogger.info("Read configuration");
    gauge.readconf_nersc("../test_conf/l328f21b6285m0009875m0790a_019.995");
    gauge.updateAll();

    /// ----------------------------------------------------------------------------------------------------GAUGE FIXING
    rootLogger.info("GAUGE FIXING...");
    timer.start();

    bool lerror=false;
    PREC adif  =0.;
    PREC tdif  =0.;
    while ( (ngfstep<ngfstepMAX) && (gftheta>gtol) ) {
        /// Compute starting GF functional and update the lattice.
        GFixing.gaugefixOR();
        /// Due to the nature of the update, we have to re-unitarize every so often.
        if ( (ngfstep%nunit) == 0 ) {
            gauge.su3latunitarize();
        }
        /// Compute GF functional difference, compute theta, and report to user.
        act    =GFixing.getAction();
        gftheta=GFixing.getTheta();
        adif=abs(act    -control_act[ngfstep]);
        tdif=abs(gftheta-control_tha[ngfstep]);
        if ( adif>abs(tolp*control_act[ngfstep]) || tdif>abs(tolp*control_tha[ngfstep]) ) {
            rootLogger.error("Large functional or theta difference!");
            rootLogger.info(control_act[ngfstep] ,  "  " ,  act);
            rootLogger.info(control_tha[ngfstep] ,  "  " ,  gftheta);
            lerror=true;
        }
        ngfstep+=1;
    }

    /// Final reunitarization.
    gauge.su3latunitarize();

    /// Report time to user.
    timer.stop();
    rootLogger.info("Time to gauge fix: " ,  timer);
    timer.reset();

    /// --------------------------------------------------------------------------------------POLYAKOV LOOP CORRELATIONS
    const int distmax=corrTools.distmax;
    const int pvol3=corrTools.pvol3;
    std::vector<PREC> vec_plca(distmax);
    std::vector<PREC> vec_plc1(distmax);
    std::vector<PREC> vec_plc8(distmax);
    std::vector<int>  vec_factor(distmax);
    std::vector<int>  vec_weight(pvol3);
    corrTools.getFactorArray(vec_factor,vec_weight);

    /// Calculation of Polyakov loop correlators.
    timer.start();
    rootLogger.info("CALCULATING CORRELATORS...");

    rootLogger.info("RUN FAST CORRELATOR TEST");
    PLC.PLCtoArrays(vec_plca, vec_plc1, vec_plc8, vec_factor, vec_weight, true);

    int controlindex=0;
    PREC difa=0.;
    PREC dif1=0.;
    PREC dif8=0.;
    for (int r2=0 ; r2<distmax ; r2++) {
        if (vec_factor[r2]>0) {
            difa=abs(vec_plca[r2]-control_plca[controlindex]);
            dif1=abs(vec_plc1[r2]-control_plc1[controlindex]);
            dif8=abs(vec_plc8[r2]-control_plc8[controlindex]);
            if ( difa>abs(tolp*control_plca[controlindex]) || dif1>abs(tolp*control_plc1[controlindex])
                                                           || dif8>abs(tolp*control_plc8[controlindex]) ) {
                rootLogger.error("Large correlator difference!");
                rootLogger.info(control_plca[controlindex] ,  "  " ,  vec_plca[r2]);
                rootLogger.info(control_plc1[controlindex] ,  "  " ,  vec_plc1[r2]);
                rootLogger.info(control_plc8[controlindex] ,  "  " ,  vec_plc8[r2]);
                lerror=true;
            }
            controlindex+=1;
        }
    }

    timer.stop();
    rootLogger.info("Time to measure correlations: " ,  timer);
    timer.reset();
    timer.start();

    rootLogger.info("RUN GENERAL CORRELATOR TEST");
    PLC.PLCtoArrays(vec_plca, vec_plc1, vec_plc8, vec_factor, vec_weight, false);

    controlindex=0;
    difa=0.;
    dif1=0.;
    dif8=0.;
    for (int r2=0 ; r2<distmax ; r2++) {
        if (vec_factor[r2]>0) {
            difa=abs(vec_plca[r2]-control_plca[controlindex]);
            dif1=abs(vec_plc1[r2]-control_plc1[controlindex]);
            dif8=abs(vec_plc8[r2]-control_plc8[controlindex]);
            if ( difa>abs(tolp*control_plca[controlindex]) || dif1>abs(tolp*control_plc1[controlindex])
                 || dif8>abs(tolp*control_plc8[controlindex]) ) {
                rootLogger.error("Large correlator difference!");
                rootLogger.info(control_plca[controlindex] ,  "  " ,  vec_plca[r2]);
                rootLogger.info(control_plc1[controlindex] ,  "  " ,  vec_plc1[r2]);
                rootLogger.info(control_plc8[controlindex] ,  "  " ,  vec_plc8[r2]);
                lerror=true;
            }
            controlindex+=1;
        }
    }

    timer.stop();
    rootLogger.info("Time to measure correlations: " ,  timer);

    if(lerror) {
      rootLogger.error("At least one test " ,  CoutColors::red ,  "failed!" ,  CoutColors::reset);
    } else {
      rootLogger.info("All tests " ,  CoutColors::green ,  "passed!" ,  CoutColors::reset);
    }

    return 0;
}

