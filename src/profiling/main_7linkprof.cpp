#include "../simulateqcd.h"
#include "../gauge/constructs/hisqForceConstructs.h"


template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part, size_t term = 0>
class contribution_7link {
private:
    SU3Accessor<floatT, comp> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;
    floatT _c7 = 1/48./8.;
public:
    contribution_7link(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceInm);
    __host__ __device__ SU3<floatT> operator() (gSiteMu siteMu);
};

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int Part, size_t term>
contribution_7link<floatT, onDevice, HaloDepth, comp, Part, term>::contribution_7link(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceIn) : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int Part, size_t term>
__host__ __device__ SU3<floatT> contribution_7link<floatT, onDevice, HaloDepth, comp, Part, term>::operator() (gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);
    switch (Part) {
    case 1:
        return sevenLinkContribution_1<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    case 2:
        return sevenLinkContribution_2<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    case 3:
        return sevenLinkContribution_3<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    case 4:
        return sevenLinkContribution_4<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    case 5:
        return sevenLinkContribution_5<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    case 6:
        return sevenLinkContribution_6<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    case 7:
        return sevenLinkContribution_7<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    default:
        return su3_zero<floatT>();
    }

}



template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part>
class contribution_5link {
private:
    SU3Accessor<floatT,comp> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;
    floatT _c5=1/8./8.;
public:
    contribution_5link(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceIn);
    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};


template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part>
contribution_5link<floatT, onDevice, HaloDepth, comp, part>::contribution_5link(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceIn) : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part>
__host__ __device__ SU3<floatT> contribution_5link<floatT, onDevice, HaloDepth, comp, part>::operator() (gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);
    switch (part) {
    case 1: return fiveLinkContribution_11<floatT, HaloDepth, comp>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
    case 2: return fiveLinkContribution_12<floatT, HaloDepth, comp>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
    case 3: return fiveLinkContribution_13<floatT, HaloDepth, comp>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
    case 4: return fiveLinkContribution_14<floatT, HaloDepth, comp>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
    // case 5: return fiveLinkContribution_20<floatT, HaloDepth, comp>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
    // case 6: return fiveLinkContribution_30<floatT, HaloDepth, comp>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
    default: return su3_zero<floatT>();
    }
}



template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part, size_t term>
class contribution_5link_large {
private:
    SU3Accessor<floatT,comp> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;
    floatT _c5=1/8./8.;
public:
    contribution_5link_large(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceIn);
    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};


template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part, size_t term>
contribution_5link_large<floatT, onDevice, HaloDepth, comp, part, term>::contribution_5link_large(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceIn) : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part, size_t term>
__host__ __device__ SU3<floatT> contribution_5link_large<floatT, onDevice, HaloDepth, comp, part, term>::operator() (gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);
    switch (part) {
        case 5: return fiveLinkContribution_20<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
        case 6: return fiveLinkContribution_30<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
    default: return su3_zero<floatT>();
    }
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, bool lvl1>
class contribution_3link {
private:
    SU3Accessor<floatT,comp> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;
    SmearingParameters<floatT> _smParams = (lvl1 ? getLevel1Params<floatT>() : getLevel2Params<floatT>());
public:
    contribution_3link(Gaugefield<floatT, onDevice, HaloDepth,comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceIn);
    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};


template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, bool lvl1>
contribution_3link<floatT, onDevice, HaloDepth, comp, lvl1>::contribution_3link(Gaugefield<floatT, onDevice,HaloDepth,comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceIn) : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, bool lvl1>
__host__ __device__ SU3<floatT> contribution_3link<floatT, onDevice, HaloDepth, comp, lvl1>::operator() (gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);
    return threeLinkContribution<floatT, HaloDepth, comp>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _smParams);
}


template <class PREC, size_t HaloDepth>
void run_force(CommunicationBase &commBase) {
        Gaugefield<PREC, true, HaloDepth, R18> gauge(commBase);
        Gaugefield<PREC, true, HaloDepth> force(commBase);
        Gaugefield<PREC, true, HaloDepth> forceOut(commBase);
        Gaugefield<PREC, true, HaloDepth> dummy(commBase);
        grnd_state<true> rand;

        initialize_rng(1337, rand);
        gauge.random(rand.state);
        force.random(rand.state);
        gauge.updateAll();
        force.updateAll();

        StopWatch<true> timer;
        forceOut = su3_zero<PREC>();
        timer.start();

        dummy.template iterateOverBulkAllMu<64>(contribution_3link<PREC,true, HaloDepth, R18, true>(gauge,force));
        forceOut = forceOut + dummy;
        timer.stop();
        rootLogger.info("Time for 3link contribution: ", timer);
        timer.reset();
        timer.start();
        static_for<1,5>::apply([&] (auto part) {
            dummy.template iterateOverBulkAllMu<64>(contribution_5link<PREC,true, HaloDepth, R18, part>(gauge,force));
            forceOut = forceOut + dummy;
        });

        static_for<5,7>::apply([&] (auto part) {
            static_for<0,4>::apply([&](auto term) {
            dummy.template iterateOverBulkAllMu<64>(contribution_5link_large<PREC,true, HaloDepth, R18, part, term>(gauge,force));
            forceOut = forceOut + dummy;
            });
        });

        timer.stop();
        rootLogger.info("Time for 5link contribution: ", timer);
        timer.reset();
        timer.start();
        static_for<1,8>::apply([&] (auto part) {
            static_for<0,8>::apply([&](auto term) {
            dummy.template iterateOverBulkAllMu<64>(contribution_7link<PREC, true, HaloDepth, R18, part, term>(gauge,force));
            forceOut = forceOut + dummy;
            });
        });
        
    
        timer.stop();
        rootLogger.info("Time for 7link contribution: ", timer);
};

template <class PREC, size_t HaloDepth>
void run_shortforce(CommunicationBase &commBase) {
        Gaugefield<PREC, true, HaloDepth, R18> gauge(commBase);
        Gaugefield<PREC, true, HaloDepth> force(commBase);
        Gaugefield<PREC, true, HaloDepth> forceOut(commBase);
        Gaugefield<PREC, true, HaloDepth> dummy(commBase);
        grnd_state<true> rand;

        initialize_rng(1337, rand);
        gauge.random(rand.state);
        force.random(rand.state);
        gauge.updateAll();
        force.updateAll();

        StopWatch<true> timer;
        forceOut = su3_zero<PREC>();
        timer.start();

        dummy.template iterateOverBulkAllMu<64>(contribution_3link<PREC,true, HaloDepth, R18, true>(gauge,force));
        forceOut = forceOut + dummy;
        timer.stop();
        rootLogger.info("Time for 3link contribution: ", timer);
        timer.reset();
        timer.start();
        static_for<1,2>::apply([&] (auto part) {
            dummy.template iterateOverBulkAllMu<64>(contribution_5link<PREC,true, HaloDepth, R18, part>(gauge,force));
            forceOut = forceOut + dummy;
        });
        timer.stop();
        rootLogger.info("Time for 5link contribution: ", timer);
        timer.reset();
        timer.start();
        static_for<1,2>::apply([&] (auto part) {
            static_for<0,1>::apply([&](auto term) {
            dummy.template iterateOverBulkAllMu<64>(contribution_7link<PREC, true, HaloDepth, R18, part, term>(gauge,force));
            forceOut = forceOut + dummy;
            });
        });
        
    
        timer.stop();
        rootLogger.info("Time for 7link contribution: ", timer);
};
int main(int argc, char *argv[]) {
    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);
    LatticeParameters rhmc_param;

    rhmc_param.readfile(commBase, "../parameter/profiling/mrhsDSlashProf.param", argc, argv);

    commBase.init(rhmc_param.nodeDim());

    const size_t HaloDepth = 2;
    initIndexer(HaloDepth, rhmc_param,commBase);

#if false
    rootLogger.info("Running test in single precision");
    run_shortforce<float, HaloDepth>(commBase);

    rootLogger.info("Running test in double precision");
    run_shortforce<double, HaloDepth>(commBase);    
#else
    rootLogger.info("Running test in single precision");
    run_force<float, HaloDepth>(commBase);

    rootLogger.info("Running test in double precision");
    run_force<double, HaloDepth>(commBase);  
#endif

    return 0;

}