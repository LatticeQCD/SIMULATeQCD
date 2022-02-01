/* 
 * main_half_prec_math_test.cu                                                               
 * 
 * Dennis Bollweg
 * 
 */

#include "../SIMULATeQCD.h"

template<class floatT, size_t HaloDepth, CompressionType comp = R18>
struct simple_add {
    gaugeAccessor<floatT, comp> gAcc;

    simple_add(Gaugefield<floatT, true, HaloDepth,comp> &gaugeIn) :
        gAcc(gaugeIn.getAccessor()) {}

    __host__ __device__ GSU3<floatT> operator()(gSiteMu site) {
        typedef GIndexer<All, HaloDepth> GInd;
        GSU3<floatT> temp;
        GSU3<floatT> temp2 = gAcc.getLink(site);
        temp = temp2+temp2+temp2;
        return temp;
    }
};

template<class floatT, size_t HaloDepth, CompressionType comp = R18>
struct simple_mult {
    gaugeAccessor<floatT,comp> gAcc;

    simple_mult(Gaugefield<floatT, true, HaloDepth, comp> &gaugeIn) :
        gAcc(gaugeIn.getAccessor()) {}

    __host__ __device__ GSU3<floatT> operator()(gSiteMu site) {
        typedef GIndexer<All, HaloDepth> GInd;
        GSU3<floatT> temp;
        GSU3<floatT> temp2 = gAcc.getLink(site);
        temp = temp2*temp2;
        return temp;
    }
};

template<class floatT, size_t HaloDepth, CompressionType comp>
struct convert_to_half {
    gaugeAccessor<floatT,comp> gAcc_source;

    convert_to_half(Gaugefield<floatT, true, HaloDepth, comp> &gaugeIn) : gAcc_source(gaugeIn.getAccessor()) {}

    __device__ __host__ GSU3<__half> operator()(gSiteMu site) {
        GSU3<floatT> source = gAcc_source.getLink(site);
        GSU3<__half> target(source);
        return target;
    }
};

template<class floatT, size_t HaloDepth, CompressionType comp>
struct simple_matvec_mult {
    gaugeAccessor<floatT,comp> gAcc;
    gVect3arrayAcc<floatT> spinorAcc;

    simple_matvec_mult(Gaugefield<floatT, true, HaloDepth, comp> &gaugeIn, Spinorfield<floatT, true, All, 0, 1> &spinorIn) : gAcc(gaugeIn.getAccessor()), spinorAcc(spinorIn.getAccessor()) {}

    __device__ __host__ gVect3<floatT> operator()(gSite site) {
        typedef GIndexer<All,0> GInd;
        gVect3<floatT> tmp(0.0);
        for (int mu = 0; mu < 4; mu++) {
            tmp += gAcc.getLink(GInd::getSiteMu(site,mu)) * spinorAcc.getElement(site);
        }

        return tmp;
    }
};

int main(int argc, char *argv[]) {
    try {
        stdLogger.setVerbosity(INFO);
    
        const size_t HaloDepth=0;
    
        rootLogger.info("Initialization");
    
        LatticeParameters param;
        const int LatDim[] = {20,20,20,20};
        const int NodeDim[] = {1, 1, 1, 1};
        param.latDim.set(LatDim);
        param.nodeDim.set(NodeDim);
    
        CommunicationBase commBase(&argc, &argv);
        commBase.init(param.nodeDim());
    
        initIndexer(HaloDepth,param,commBase);
    
        StopWatch<true> timer;
    
        Gaugefield<float,true,HaloDepth,R18> gauge(commBase);
        Gaugefield<float,true,HaloDepth,R18> dummy(commBase);
        Gaugefield<float,false,HaloDepth,R18> gauge_host(commBase);
        grnd_state<false> h_rand;
        grnd_state<true> d_rand;
    
        h_rand.make_rng_state(1337);
        d_rand = h_rand;
      
        gauge.random(d_rand.state);
    
        Gaugefield<__half,true,HaloDepth,R18> gauge_half(commBase);
        gauge_half.iterateOverBulkAllMu(convert_to_half(gauge));
        dummy.template convert_precision<__half>(gauge_half);
        gauge_host = dummy;
        Spinorfield<__half,true,All,0,1> spinor_half(commBase);
        Spinorfield<__half,true,All,0,1> spinor_out(commBase);
        spinor_half.one();
        spinor_out.iterateOverBulk(simple_matvec_mult<__half,HaloDepth,R18>(gauge_half,spinor_half));
    
        timer.start();
    
        typedef GIndexer<All,HaloDepth> GInd;
        gSite site1 = GInd::getSite(0,0,1,1);
        gaugeAccessor<float,R18> gAcc = gauge_host.getAccessor();
        rootLogger.info("initial values: ");
        GSU3<float> test = gAcc.getLink<float>(GInd::getSiteMu(site1, 3));
        rootLogger.info(test.getLink00() ,  " " , test.getLink01() ,  " " , test.getLink02());
        rootLogger.info(test.getLink10() ,  " " , test.getLink11() ,  " " ,  test.getLink12());
        rootLogger.info(test.getLink20() ,  " " , test.getLink21() ,  " " ,  test.getLink22());
        
        for (int i = 0; i < 2; i++) {
            gauge_half.iterateOverBulkAllMu(simple_add<__half,HaloDepth,R18>(gauge_half));
        }
    
        dummy.template convert_precision<__half>(gauge_half);
        gauge_host = dummy;
        
        test = gAcc.getLink<float>(GInd::getSiteMu(site1, 3));
        rootLogger.info("half precision values: ");
        
        rootLogger.info(test.getLink00() ,  " " , test.getLink01() ,  " " , test.getLink02());
        rootLogger.info(test.getLink10() ,  " " , test.getLink11() ,  " " ,  test.getLink12());
        rootLogger.info(test.getLink20() ,  " " , test.getLink21() ,  " " ,  test.getLink22());
        
        timer.stop();
        rootLogger.info("Time for 2 applications of 2 Adds (half): " ,  timer);
        
        timer.reset();
        timer.start();
        for (int i = 0; i < 2; i++) {
            gauge.iterateOverBulkAllMu(simple_add<float,HaloDepth,R18>(gauge));
        }
        gauge_host = gauge;
        
        test = gAcc.getLink<float>(GInd::getSiteMu(site1, 3));
        rootLogger.info("single precision values: ");
        
        rootLogger.info(test.getLink00() ,  " " , test.getLink01() ,  " " , test.getLink02());
        rootLogger.info(test.getLink10() ,  " " , test.getLink11() ,  " " ,  test.getLink12());
        rootLogger.info(test.getLink20() ,  " " , test.getLink21() ,  " " ,  test.getLink22());
    
        timer.stop();
        rootLogger.info("Time for 2 applications of 2 Adds (float): " ,  timer);
    
        
        timer.reset();
        timer.start();
        for (int i = 0; i < 2; i++) {
            gauge_half.iterateOverBulkAllMu(simple_mult<__half,HaloDepth,R18>(gauge_half));
        }
        
        timer.stop();
        rootLogger.info("Time for 2 applications of 6 mults (half): " ,  timer);
    
        timer.reset();
        timer.start();
    
        dummy.template convert_precision<__half>(gauge_half);
        gauge_host = dummy;
        
        test = gAcc.getLink<float>(GInd::getSiteMu(site1, 3));
        rootLogger.info("half precision values: ");
        
        rootLogger.info(test.getLink00() ,  " " , test.getLink01() ,  " " , test.getLink02());
        rootLogger.info(test.getLink10() ,  " " , test.getLink11() ,  " " ,  test.getLink12());
        rootLogger.info(test.getLink20() ,  " " , test.getLink21() ,  " " ,  test.getLink22());
        
        
        for (int i = 0; i < 2; i++) {
            gauge.iterateOverBulkAllMu(simple_mult<float,HaloDepth,R18>(gauge));
        }
        
        timer.stop();
        rootLogger.info("Time for 2 applications of 6 mults (float): " ,  timer);
        
        gauge_host = gauge;
        
        test = gAcc.getLink<float>(GInd::getSiteMu(site1, 3));
        rootLogger.info("single precision values: ");
        
        rootLogger.info(test.getLink00() ,  " " , test.getLink01() ,  " " , test.getLink02());
        rootLogger.info(test.getLink10() ,  " " , test.getLink11() ,  " " ,  test.getLink12());
        rootLogger.info(test.getLink20() ,  " " , test.getLink21() ,  " " ,  test.getLink22());
        return 0;
    }
    catch (const std::runtime_error &error) {
        return 1;
    }
}
