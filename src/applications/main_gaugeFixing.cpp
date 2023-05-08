/* 
 * main_gaugeFixing.cpp
 *
 * D. Clarke
 *
 * GPU code to fix a configuration to the Coulomb gauge. Computes average, singlet, and octet Polyakov loop correlations
 * as well as Wilson line correlators. Will also compute average thermal Wilson line, which is sometimes needed to normalize
 * aforementioned Polyakov loop correlators to the long-distance behavior. This main does not work for multiGPU, because it 
 * is hard to find a way to calculate correlations efficiently on multiGPU.
 *
 * For the Polyakov loop correlations, this program uses functions defined in the header file. There are more general
 * functions to measure such correlators in ../math/correlators.h, but these functions are slower because they call a
 * kernel for each type of correlation, rather than one kernel to compute three.
 *
 */


#include "../SIMULATeQCD.h"
#include "../modules/gaugeFixing/gfix.h"
#include "../modules/observables/PolyakovLoop.h"
#include "../modules/gaugeFixing/PolyakovLoopCorrelator.h"
#include "../modules/observables/WilsonLineCorrelator.h"

#define PREC double 

template<class floatT>
struct gfixParam : LatticeParameters {
    Parameter<floatT>      gtolerance;
    Parameter<int,1>       maxgfsteps;
    Parameter<int,1>       numunit;
    Parameter<std::string> cml;
    Parameter<std::string> cms;
    Parameter<std::string> measurements_dir;
    Parameter<std::string> SavedConfName;
    Parameter<bool>        PolyakovLoopCorr;
    Parameter<bool>        WilsonLineCorr;
    Parameter<bool>        ThermalWilsonLine;
    Parameter<bool>        SaveConfig;

    gfixParam() {
        addDefault (gtolerance,"gtolerance",1e-6);
        addDefault (maxgfsteps,"maxgfsteps",1000);
        addDefault (numunit   ,"numunit"   ,20);
        addOptional(cml       ,"mlight");
        addOptional(cms       ,"mstrange");
        addDefault(PolyakovLoopCorr , "PolyakovLoopCorr" , false);
        addDefault(WilsonLineCorr   , "WilsonLineCorr"   , false);
        addDefault(ThermalWilsonLine, "ThermalWilsonLine", false);
        addDefault(SaveConfig       , "SaveConfig"       , false);
        add(measurements_dir, "measurements_dir");
        add(SavedConfName   , "SavedConfName");
    }
};


int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);
    const size_t HaloDepth  = 0;

    /// Read in parameters and initialize communication base.
    rootLogger.info("Initialization");
    CommunicationBase commBase(&argc, &argv);
    gfixParam<PREC> param;
    param.readfile(commBase, "../parameter/applications/gaugeFixing.param", argc, argv);
    commBase.init(param.nodeDim());
    initIndexer(HaloDepth,param,commBase);
    typedef GIndexer<All,HaloDepth> GInd;
    const int  ngfstepMAX = param.maxgfsteps();
    const int  nunit      = param.numunit();
    const PREC gtol       = param.gtolerance();

    StopWatch<true> timer;
    int        ngfstep = 0;          /// # of gauge fixing steps
    PREC       gftheta = 1e10;       /// gauge fixing theta
    PREC       act1    = 1.0;        /// gauge fix action before OR update
    PREC       act2    = 1.0;        /// gauge fix action after OR update
    Gaugefield<PREC,true,HaloDepth> gauge(commBase);        /// gauge field
    GaugeFixing<PREC,true,HaloDepth> GFixing(gauge);        /// gauge fixing class
    CorrelatorTools<PREC,true,HaloDepth> Corrs;             /// general correlator class
    PolyakovLoopCorrelator<PREC,true,HaloDepth> PLC(gauge); /// class for Polyakov loop correlators
    PolyakovLoop<PREC,true,HaloDepth> ploopClass(gauge);    /// class for measuring Polyakov loops
    WilsonLineCorrelator<PREC,true,HaloDepth> WLC(gauge);   /// class for Polyakov loop correlators

    ///prepare output file
    std::stringstream plcfilename,wlcfilename,cbeta,cstream;
    plcfilename << param.measurements_dir() << "plc_l" << param.latDim[0] << param.latDim[3] << "f21";
    wlcfilename << param.measurements_dir() << "wlc_l" << param.latDim[0] << param.latDim[3] << "f21";

    if (param.beta.isSet()) {
        cbeta << std::setw(4) << (int)(1000*param.beta());
        plcfilename << "b" << cbeta.str();
        wlcfilename << "b" << cbeta.str();
    }
    if (param.cml.isSet()) {
        plcfilename << "m" << param.cml();
        wlcfilename << "m" << param.cml();
    }
    if (param.cms.isSet()) {
        plcfilename << "m" << param.cms();
        wlcfilename << "m" << param.cms();
    }
    if (param.streamName.isSet()) {
        cstream.fill('0');
        cstream << std::setw(3) << param.streamName();
        plcfilename << "a_" << cstream.str();
        wlcfilename << "a_" << cstream.str();
    }
    if (param.confnumber.isSet()) {
        plcfilename << "." << param.confnumber();
        wlcfilename << "." << param.confnumber();
    }
    plcfilename << ".d";
    wlcfilename << ".d";

    std::ofstream plcresultfile;
    if ( param.PolyakovLoopCorr() ) {
        plcresultfile.open(plcfilename.str());
        rootLogger.info("PolyakovLoopCorr OUTPUT TO FILE: " ,  plcfilename.str());
    }

    std::ofstream wlcresultfile;
    if ( param.WilsonLineCorr() ) {
        wlcresultfile.open(wlcfilename.str());
        rootLogger.info("WilsonLineCorr OUTPUT TO FILE: " ,  wlcfilename.str());
    }

    /// Read the configuration. Remember a halo exchange is needed every time the gauge field changes.
    rootLogger.info("Read configuration");
    gauge.readconf_nersc(param.GaugefileName());

    /// Measure the Polyakov loop and report to user.
    GCOMPLEX(PREC) ploop = ploopClass.getPolyakovLoop();
    rootLogger.info("# POLYAKOV LOOP :: " ,  ploop);


    /// ----------------------------------------------------------------------------------------------------GAUGE FIXING


    rootLogger.info("GAUGE FIXING...");
    if (commBase.MyRank()==0) std::cout << "\nngfstep\t  act\t         act diff\ttheta\n";
    timer.start();

    while ( (ngfstep<ngfstepMAX) && (gftheta>gtol) ) {
        /// Compute starting GF functional and update the lattice.
        act1=GFixing.getAction();
        GFixing.gaugefixOR();
        /// Due to the nature of the update, we have to re-unitarize every so often.
        if ( (ngfstep%nunit) == 0 ) {
            gauge.su3latunitarize();
        }
        /// Compute GF functional difference, compute theta, and report to user.
        act2   =GFixing.getAction();
        gftheta=GFixing.getTheta();
        if (commBase.MyRank()==0) {
            std::cout << std::setw(7) << ngfstep << "  " << std::setw(13) << std::scientific << act2
                                                 << "  " << std::setw(13) << std::scientific << fabs(act2-act1)
                                                 << "  " << std::setw(13) << std::scientific << gftheta << std::endl;
        }
        ngfstep+=1;
    }

    /// Final reunitarization.
    gauge.su3latunitarize();

    /// Report time to user.
    timer.stop();
    if (commBase.MyRank()==0) std::cout << "\n";
    rootLogger.info("Time to gauge fix: " ,  timer);

    /// Optionally save configuration.
    if ( param.SaveConfig() ) gauge.writeconf_nersc(param.SavedConfName(), 2, 2);

    /// ------------------------------------------------------------------------------------ AVERAGE THERMAL WILSON LINE

    
    if ( param.ThermalWilsonLine() ) {
        rootLogger.info("CALCULATING AVERAGE THERMAL WILSON LINE...");
    }


    /// ---------------------------------------------------------------------------------------POLYAKOV LOOP CORRELATORS
    if ( param.PolyakovLoopCorr() ) {
        rootLogger.info("CALCULATING POLYAKOVLOOP CORRELATORS...");
        timer.start();
        std::ofstream plcresultfile;
        plcresultfile.open(plcfilename.str());
        const int distmax=Corrs.distmax;
        const int pvol3=Corrs.pvol3;
        std::vector<PREC> vec_plca(distmax);
        std::vector<PREC> vec_plc1(distmax);
        std::vector<PREC> vec_plc8(distmax);
        std::vector<int>  vec_factor(distmax);
        std::vector<int>  vec_weight(pvol3);
        Corrs.getFactorArray(vec_factor,vec_weight);

        if (commBase.MyRank()==0) plcresultfile << "#  r**2\t  plca\t         plc1\t        plc8\n";

        /// Calculation of Polyakov loop correlators.
        PLC.PLCtoArrays(vec_plca, vec_plc1, vec_plc8, vec_factor, vec_weight, true);

        /// Write final results to output file.
        for (int dx=0 ; dx<distmax ; dx++) {
            if (vec_factor[dx]>0) {
                if (commBase.MyRank()==0) {
                    plcresultfile << std::setw(7) << dx
                                  << "  " << std::setw(13) << std::scientific << vec_plca[dx]
                                  << "  " << std::setw(13) << std::scientific << vec_plc1[dx]
                                  << "  " << std::setw(13) << std::scientific << vec_plc8[dx] << std::endl;
                }
            }
        }
        plcresultfile.close();
        rootLogger.info("POLYAKOVLOOP CORRELATORS MEASURED");
        timer.stop();
        rootLogger.info("Time to measure polyakovloop correlations: " ,  timer);
    }

    /// -----------------------------------------------------------------------------------------WILSON LINE CORRELATORS
    if ( param.WilsonLineCorr() ) {
        rootLogger.info("CALCULATING WILSONLINE CORRELATORS...");
        timer.start();
        std::ofstream wlcresultfile;
        wlcresultfile.open(wlcfilename.str());
        const int distmax=Corrs.distmax;
        const int pvol3=Corrs.pvol3;
        std::vector<PREC> vec_wlca_full(distmax*param.latDim[3]);
        std::vector<PREC> vec_wlc1_full(distmax*param.latDim[3]);
        std::vector<PREC> vec_wlc8_full(distmax*param.latDim[3]);
        std::vector<int>  vec_factor(distmax);
        std::vector<int>  vec_weight(pvol3);
        Corrs.getFactorArray(vec_factor,vec_weight);

        if (commBase.MyRank()==0) wlcresultfile << "#  r**2\t   dtau\t  wlca\t         wlc1\t        wlc8\n";

        /// Calculation of wilsonline correlators.
        WLC.WLCtoArrays(vec_wlca_full, vec_wlc1_full, vec_wlc8_full, vec_factor, vec_weight, true);

        /// Write final results to output file.
        for (int dtau=1; dtau<=param.latDim[3];dtau++) {
            for (int dx=0 ; dx<distmax ; dx++) {
                if (vec_factor[dx]>0) {
                    if (commBase.MyRank()==0) {
                        wlcresultfile << std::setw(7) << dx << "\t    " << dtau
                                      << "  " << std::setw(13) << std::scientific << vec_wlca_full[(dtau-1)*distmax+dx]
                                      << "  " << std::setw(13) << std::scientific << vec_wlc1_full[(dtau-1)*distmax+dx]
                                      << "  " << std::setw(13) << std::scientific << vec_wlc8_full[(dtau-1)*distmax+dx] << std::endl;
                    }
                }
            }
        }
        wlcresultfile.close();
        rootLogger.info("WILSONLINE CORRELATORS MEASURED");
        timer.stop();
        rootLogger.info("Time to measure wilsonline correlations: " ,  timer);
    }

    return 0;
}

