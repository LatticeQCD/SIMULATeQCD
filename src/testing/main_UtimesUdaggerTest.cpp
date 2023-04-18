/*
 * main_UtimesUdagger.cpp
 *
 * Hai-Tao Shu, 5 May 2020
 *
 * Read in the test conf and calculate UxUdagger and see whether it gives unity.
 *
 */

#include "../SIMULATeQCD.h"

#define PREC double

/// Get tr_d(UxUdagger) for each link
template<class floatT,size_t HaloDepth>
struct CalcTrUtimesUdagger{

    gaugeAccessor<floatT> gaugeAccessor;

    CalcTrUtimesUdagger(Gaugefield<floatT,true,HaloDepth> &gauge) : gaugeAccessor(gauge.getAccessor()){}

    __device__ __host__ floatT operator()(gSite site) {

        typedef GIndexer<All, HaloDepth> GInd;
        GSU3<floatT> temp;
        floatT result = 0.;
        for (int mu = 0; mu < 4; mu++) {
            gSiteMu siteMu = GInd::getSiteMu(site,mu);
            result+=tr_d(gaugeAccessor.getLink(siteMu)*dagger(gaugeAccessor.getLink(siteMu)));
        }
        return result/3;
    }
};

/// Function getting <tr UtimesUdagger> using the above kernel
template<class floatT, size_t HaloDepth>
floatT getTrUtimesUdagger(Gaugefield<floatT,true, HaloDepth> &gauge, LatticeContainer<true,floatT> &redBase){

    typedef GIndexer<All,HaloDepth> GInd;
    const size_t elems = GInd::getLatData().vol4;
    redBase.adjustSize(elems);

    redBase.template iterateOverBulk<All, HaloDepth>(CalcTrUtimesUdagger<floatT, HaloDepth>(gauge));
    floatT trUUdagger;
    redBase.reduce(trUUdagger, elems);

    trUUdagger /= (GInd::getLatData().globalLattice().mult()*4);
    return trUUdagger;
}

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);
    const size_t HaloDepth = 0;

    /// None of these parameters should be changed.
    rootLogger.info("Initialization");
    LatticeParameters param;
    const int  LatDim[]   = {20,20,20,20};
    const int  NodeDim[]  = {1 ,1 ,1 ,1};
    param.latDim.set(LatDim);
    param.nodeDim.set(NodeDim);
    CommunicationBase commBase(&argc, &argv);
    commBase.init(param.nodeDim());
    initIndexer(HaloDepth,param,commBase);
    typedef GIndexer<All,HaloDepth> GInd;

    PREC                              traceU  = 0.;      /// average trace of a link
    Gaugefield<PREC,true,HaloDepth>   gauge(commBase);   /// gauge field
    LatticeContainer<true,PREC> redBase(commBase);
    redBase.adjustSize(GInd::getLatData().vol4);

    /// Read the configuration. Remember a halo exchange is needed every time the gauge field changes.
    rootLogger.info("Read configuration");
    gauge.readconf_nersc("../test_conf/l20t20b06498a_nersc.302500");
    gauge.updateAll();

    traceU =getTrUtimesUdagger<PREC,HaloDepth>(gauge, redBase);

    rootLogger.info(std::scientific ,  std::setprecision(14)  ,  "Trace of (UtimesUdagger) is: " ,  traceU);

    if (fabs(traceU-1.0)>1.e-10) {
        rootLogger.error("UtimesUdagger TEST: failed!");
        return -1;
    } else {
        rootLogger.info("UtimesUdagger TEST: passed");
    }

    return 0;
}

