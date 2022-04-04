/* 
 * main_gfixTestMulti.cu
 *
 * D. Clarke
 *
 * Test some gauge-dependent quantities to verify that single GPU runs are consistent with multiGPU runs. The tested
 * quantities include the gauge-fixing theta and <tr U>
 *
 */

#include "../SIMULATeQCD.h"
#include "../modules/gaugeFixing/gfix.h"
#include "testing.h"

#define PREC double 

/// Get tr U for each link
template<class floatT,size_t HaloDepth>
struct CalcTrU{

    gaugeAccessor<floatT> gaugeAccessor;

    CalcTrU(Gaugefield<floatT,true,HaloDepth> &gauge) : gaugeAccessor(gauge.getAccessor()){}

    __device__ __host__ floatT operator()(gSite site) {

        typedef GIndexer<All, HaloDepth> GInd;
        GSU3<floatT> temp;
        floatT result = 0.;
        for (int mu = 0; mu < 4; mu++) {
            gSiteMu siteMu = GInd::getSiteMu(site,mu);
            result+=tr_d(gaugeAccessor.getLink(siteMu));
        }
        return result/4;
    }
};

/// Function getting <tr U> using the above kernel
template<class floatT, size_t HaloDepth>
floatT getTrU(Gaugefield<floatT,true, HaloDepth> &gauge, LatticeContainer<true,floatT> &redBase){

    typedef GIndexer<All,HaloDepth> GInd;
    const size_t elems = GInd::getLatData().vol4;
    redBase.adjustSize(elems);

    redBase.template iterateOverBulk<All, HaloDepth>(CalcTrU<floatT, HaloDepth>(gauge));
    floatT trU;
    redBase.reduce(trU, elems);

    trU /= (GInd::getLatData().globalLattice().mult()*12);
    return trU;
}


int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);
    const size_t HaloDepth = 2;

    /// None of these parameters should be changed.
    rootLogger.info("Initialization");
    LatticeParameters param;
    const int  LatDim[]   = {32,32,32,8};
    const int  ngfstepMAX = 30;
    const int  nunit      = 20;
    const PREC gtol       = 1e-6;
    const PREC tolp       = 1e-14; 
    param.latDim.set(LatDim);
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/tests/gfixTestMulti.param", argc, argv);
    commBase.init(param.nodeDim());
    initIndexer(HaloDepth,param,commBase);
    typedef GIndexer<All,HaloDepth> GInd;

    /// More initialization.
    int                               ngfstep = 0;           /// # of gauge fixing steps
    PREC                              gftheta = 1e10;        /// gauge fixing theta
    PREC                              act1    = 1.0;         /// gauge fix action before OR update
    PREC                              act2    = 1.0;         /// gauge fix action after OR update
    PREC                              traceU  = 0.;          /// average trace of a link
    Gaugefield<PREC,false,HaloDepth>  refgauge(commBase);    /// gauge field created by gfixTestSingle to compare
    Gaugefield<PREC,true,HaloDepth>   gauge(commBase);       /// gauge field created by gfixTestMulti
    Gaugefield<PREC,false,HaloDepth>  hostgauge(commBase);   /// above gauge field copied to host for comparison
    GaugeFixing<PREC,true,HaloDepth>  GFixing(gauge);        /// gauge fixing class

    LatticeContainer<true,PREC> redBase(commBase);
    redBase.adjustSize(GInd::getLatData().vol4);

    /// Read the configuration. Remember a halo exchange is needed every time the gauge field changes.
    rootLogger.info("Read configuration");
    gauge.readconf_nersc("../test_conf/l328f21b6285m0009875m0790a_019.995");
    gauge.updateAll();

    /// ----------------------------------------------------------------------------------------------------GAUGE FIXING
    rootLogger.info("GAUGE FIXING...");

    bool lpassed=true;
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
        traceU =getTrU<PREC,HaloDepth>(gauge, redBase);
        if (commBase.MyRank()==0) {
            std::cout << std::setw(7) << ngfstep << "  " << std::setw(13) << std::scientific << act2
                                                 << "  " << std::setw(13) << std::scientific << fabs(act2-act1)
                                                 << "  " << std::setw(13) << std::scientific << gftheta
                                                 << "  " << std::setw(13) << std::scientific << traceU << std::endl;
        }
        ngfstep+=1;
    }

    /// Final reunitarization.
    gauge.su3latunitarize();

    refgauge.readconf_nersc("gfixTestSingle_conf");
    hostgauge=gauge;

    if(compare_fields<PREC,HaloDepth,false,R18>(hostgauge,refgauge,tolp)) {
        rootLogger.info("Direct link check (read) " ,  CoutColors::green ,  "passed." ,  CoutColors::reset);
    } else {
        rootLogger.info("Direct link check (read) " ,  CoutColors::red ,  "failed." ,  CoutColors::reset);
        lpassed=false;
    }

    /// Close up shop.
    rootLogger.info("==============================");
    if (lpassed) {
        rootLogger.info("All tests " ,  CoutColors::green ,  "passed!" ,  CoutColors::reset);
    } else {
        rootLogger.error("At least one test failed!");
        return -1;
    }
    rootLogger.info("==============================");

    return 0;
}

