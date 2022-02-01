/* 
 * main_FieldStrengthTensorBenchmark.cu                                                               
 * 
 * Hai Tao Shu
 * 
 */

#include "../SIMULATeQCD.h"

#define PREC double
#define ON_DEVICE true

template<class floatT, bool onDevice, size_t HaloDepth>
struct FieldStrengthTensorTemporalKernel{

    gaugeAccessor<floatT> gaugeAccessor;
    FieldStrengthTensor<floatT, HaloDepth, onDevice, R18> FT;

    FieldStrengthTensorTemporalKernel(Gaugefield<floatT,onDevice,HaloDepth> &gauge) : gaugeAccessor(gauge.getAccessor()), FT(gauge.getAccessor()) {}
    __device__ __host__ GSU3<floatT> operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;

        GSU3<floatT> Ft; 
        Ft = FT(site, 0, 3) + FT(site, 1, 3) + FT(site, 2, 3); 
        return Ft;
    }
};  

template<class floatT, bool onDevice, size_t HaloDepth>
struct FieldStrengthTensorSpatialKernel{

    gaugeAccessor<floatT> gaugeAccessor;
    FieldStrengthTensor<floatT, HaloDepth, onDevice, R18> FT;

    FieldStrengthTensorSpatialKernel(Gaugefield<floatT,onDevice,HaloDepth> &gauge) : gaugeAccessor(gauge.getAccessor()), FT(gauge.getAccessor()) { }
    __device__ __host__ GSU3<floatT> operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;

        GSU3<floatT> Fs;    
        Fs = FT(site, 0, 1) + FT(site, 0, 2) + FT(site, 1, 2);   
        return Fs;
    }   
}; 

template<class floatT, bool onDevice, size_t HaloDepth>
GSU3<floatT> getFmunu_spatial(Gaugefield<floatT,true, HaloDepth> &gauge, LatticeContainer<true,GSU3<floatT>> &redBase){

    typedef GIndexer<All,HaloDepth> GInd;
    const size_t elems = GInd::getLatData().vol4;
    redBase.adjustSize(elems);

    redBase.template iterateOverBulk<All, HaloDepth>(FieldStrengthTensorSpatialKernel<floatT,onDevice,HaloDepth>(gauge));
    GSU3<floatT> Fmunu;
    redBase.reduce(Fmunu, elems);

    Fmunu /= (GInd::getLatData().globalLattice().mult()*4);
    return Fmunu;
}

template<class floatT, bool onDevice, size_t HaloDepth>
GSU3<floatT> getFmunu_temporal(Gaugefield<floatT,true, HaloDepth> &gauge, LatticeContainer<true,GSU3<floatT>> &redBase){

    typedef GIndexer<All,HaloDepth> GInd;
    const size_t elems = GInd::getLatData().vol4;
    redBase.adjustSize(elems);

    redBase.template iterateOverBulk<All, HaloDepth>(FieldStrengthTensorTemporalKernel<floatT,onDevice,HaloDepth>(gauge));
    GSU3<floatT> Fmunu;
    redBase.reduce(Fmunu, elems);

    Fmunu /= (GInd::getLatData().globalLattice().mult()*4);
    return Fmunu;
}

int main(int argc, char *argv[]) {
    try {
        const size_t HaloDepth = 1;

        ///Initialize Base
        typedef GIndexer<All, HaloDepth> GInd;
        stdLogger.setVerbosity(INFO);
        StopWatch<true> timer;
        LatticeParameters lp;
        CommunicationBase commBase(&argc, &argv);
        lp.readfile(commBase, "../parameter/tests/FieldStrengthTensorTest.param", argc, argv);
        commBase.init(lp.nodeDim());
        initIndexer(HaloDepth, lp, commBase);
        Gaugefield<PREC, ON_DEVICE, HaloDepth> gauge(commBase);

        rootLogger.info("Reference values in this test come from l328f21b6285m0009875m0790a_019.995");
        rootLogger.info("Read configuration", lp.GaugefileName());
        gauge.readconf_nersc(lp.GaugefileName());

        LatticeContainer<true, GSU3<PREC>> redBase(commBase);
        redBase.adjustSize(GInd::getLatData().vol4);

        GSU3<PREC> F_temporal;
        GSU3<PREC> F_spatial;
        timer.start();
        F_temporal = getFmunu_temporal<PREC, HaloDepth>(gauge, redBase);
        timer.stop();
        rootLogger.info("Time for temporal Fmunu computation: ", timer);
        timer.reset();
        timer.start();
        F_spatial = getFmunu_spatial<PREC, HaloDepth>(gauge, redBase);
        timer.stop();
        rootLogger.info("Time for spatial Fmunu computation: ", timer);
    }
    catch (const std::runtime_error &error) {
        return 1;
    }
    return 0;
}

