// #include "wrapper/gpu_wrapper.h"
// #include "wrapper/"
#pragma once
#include "../define.h"

#include "math/operators.h"
#include "../base/indexer/haloIndexer.h"
#include <sycl/sycl.hpp>

#define DEFAULT_NBLOCKS 64
#define DEFAULT_NBLOCKS_LOOP 128
#define DEFAULT_NBLOCKS_CONST 256

template<bool onDevice, class Accessor>
class RunFunctors {
    public:
    virtual Accessor getAccessor() const = 0;

    template<unsigned WorkGroupSize = DEFAULT_NBLOCKS_CONST, typename CalcReadInd, typename CalcWriteInd, typename Object>
    void iterateWithConstObject(Object ob, CalcReadInd calcReadInd, CalcWriteInd calcWriteInd,
    const size_t elems_x, sycl::queue q);

    template<unsigned WorkGroupSize = DEFAULT_NBLOCKS, typename CalcReadInd, typename CalcWriteInd, typename Functor>
    void iterateFunctor(Functor Op, CalcReadInd calcReadInd, CalcWriteInd calcWriteInd, 
    const size_t elems_x, sycl::queue q);


    template<size_t Nloops, unsigned WorkGroupSize = DEFAULT_NBLOCKS_LOOP, typename CalcReadInd, typename CalcWriteInd, typename Functor>
    void iterateFunctorLoop(Functor op, CalcReadInd calcReadInd, CalcWriteInd calcWriteInd, 
    const size_t elems_x, sycl::queue q, size_t Nmax = Nloops);

};

template<bool onDevice, class Accessor>
template<unsigned WorkGroupSize, typename CalcReadInd, typename CalcWriteInd, typename Object>
void RunFunctors<onDevice, Accessor>::iterateWithConstObject(Object ob, CalcReadInd calcReadInd, CalcWriteInd calcWriteInd,
    const size_t elems_x, sycl::queue q) {

    q.submit([&] (sycl::handler& cgh) {
        size_t WorkGroups = static_cast<size_t> (ceilf(static_cast<float> (elems_x) / static_cast<float> (WorkGroupSize)));
        size_t iteration_limit = WorkGroups * WorkGroupSize;

        auto iteration_range = sycl::nd_range{sycl::range<1>{iteration_limit}, sycl::range<1>{WorkGroupSize}};
        auto acc = getAccessor();

        cgh.parallel_for(iteration_range, [=](sycl::nd_item<1> itm) {
            size_t i = itm.get_local_id()+itm.get_group().get_group_id(0)*itm.get_local_range(0);

            if (i > elems_x) return;

            auto site = calcReadInd(i);
            acc.setElement(calcWriteInd(site), ob);
        });

    }).wait();


}



template<bool onDevice, class Accessor>
template<unsigned WorkGroupSize, typename CalcReadInd, typename CalcWriteInd, typename Functor>
void RunFunctors<onDevice, Accessor>::iterateFunctor(Functor op, CalcReadInd calcReadInd, 
    CalcWriteInd calcWriteInd, const size_t elems_x, sycl::queue q) {
    

    q.submit([&] (sycl::handler& cgh) {

        size_t WorkGroups = static_cast<size_t> (ceilf(static_cast<float> (elems_x) / static_cast<float> (WorkGroupSize)));
        size_t iteration_limit = WorkGroups * WorkGroupSize;

        auto iteration_range = sycl::nd_range{sycl::range<1> {iteration_limit}, sycl::range<1>{WorkGroupSize}};
        auto acc = getAccessor(); //implicit capture of 'this' is not allowed inside sycl kernel so we get hold of a local copy here

        cgh.parallel_for(iteration_range, [=](sycl::nd_item<1> itm) {
            size_t i = itm.get_local_id()+itm.get_group().get_group_id(0)*itm.get_local_range(0);
            
            if (i > elems_x) return;

            auto site = calcReadInd(i);
            acc.setElement(calcWriteInd(site), op(site));
        });

    }).wait();
}

template<bool onDevice, class Accessor>
template<size_t Nloops, unsigned WorkGroupSize, typename CalcReadInd, typename CalcWriteInd, typename Functor>
void RunFunctors<onDevice, Accessor>::iterateFunctorLoop(Functor op, CalcReadInd calcReadInd,
CalcWriteInd calcWriteInd, const size_t elems_x, sycl::queue q, size_t Nmax) {
    
    if (Nmax > Nloops) {
        throw std::runtime_error(stdLogger.fatal("Nmax larger than Nloops!"));
    }

    q.submit([&] (sycl::handler& cgh) {
        
        size_t WorkGroups = static_cast<size_t> (ceilf(static_cast<float> (elems_x) / static_cast<float> (WorkGroupSize)));
        size_t iteration_limit = WorkGroups * WorkGroupSize;
        auto iteration_range = sycl::nd_range{sycl::range<1> {iteration_limit}, sycl::range<1>{WorkGroupSize}};
        auto acc = getAccessor();

        cgh.parallel_for(iteration_range, [=] (sycl::nd_item<1> itm) {
            size_t i = itm.get_local_id() + itm.get_group().get_group_id(0)*itm.get_local_range(0);
            
            if (i > elems_x) return;

            auto site = calcReadInd(i);
            op.initialize(site);

            for (size_t loopIdx = 0; loopIdx < Nloops; loopIdx++) {
                acc.setElement(calcWriteInd(site, loopIdx), op(site, loopIdx));
            }
        });
    }).wait();
}

template<bool onDevice, size_t WorkGroupSize = DEFAULT_NBLOCKS, typename CalcReadInd, typename Functor>
void iterateFunctorNoReturn(Functor op, CalcReadInd calcReadInd, const size_t elems_x, sycl::queue q) {


    q.submit([&] (sycl::handler& cgh) {

        size_t WorkGroups = static_cast<size_t> (ceilf(static_cast<float> (elems_x) / static_cast<float> (WorkGroupSize)));
        size_t iteration_limit = WorkGroups * WorkGroupSize;

        auto iteration_range = sycl::nd_range{sycl::range<1> {iteration_limit}, sycl::range<1>{WorkGroupSize}};
       
        cgh.parallel_for(iteration_range, [=](sycl::nd_item<1> itm) {
            size_t i = itm.get_local_id()+itm.get_group().get_group_id(0)*itm.get_local_range(0);
            
            if (i > elems_x) return;

            auto site = calcReadInd(i);
//            acc.setElement(calcWriteInd(site), op(site));
            op(site);
        });

    }).wait();
}


template<bool onDevice, size_t WorkGroupSize = DEFAULT_NBLOCKS, typename CalcReadWriteInd, typename Functor, class Accessor>
void iterateFunctorComm(Functor op, Accessor acc, CalcReadWriteInd calcReadWriteInd, const size_t subHaloSize, const size_t elems_x, sycl::queue q) {

    q.submit([&] (sycl::handler& cgh) {

        size_t WorkGroups = static_cast<size_t> (ceilf(static_cast<float> (elems_x) / static_cast<float> (WorkGroupSize)));
        size_t iteration_limit = WorkGroups * WorkGroupSize;

        auto iteration_range = sycl::nd_range{sycl::range<1> {iteration_limit}, sycl::range<1>{WorkGroupSize}};
        // auto acc = getAccessor(); //implicit capture of 'this' is not allowed inside sycl kernel so we get hold of a local copy here

        cgh.parallel_for(iteration_range, [=](sycl::nd_item<1> itm) {
            size_t i = itm.get_local_id()+itm.get_group().get_group_id(0)*itm.get_local_range(0);
            
            if (i > elems_x) return;

            size_t HaloMuIndex = i;
            size_t HaloIndex = HaloMuIndex % subHaloSize;
            size_t mu = HaloMuIndex / subHaloSize;

            auto site = calcReadWriteInd(HaloIndex,mu);
            acc.setElement(site, op(site));
        });

    }).wait();
}

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteFull {

    template<typename... Args>
    inline gSite operator()(Args... args) const {
        gSite site = GIndexer<LatticeLayout, HaloDepth>::getSiteFull(args...);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSite {
    template<typename... Args>
    inline gSite operator()(Args... args) const {
        gSite site = GIndexer<LatticeLayout, HaloDepth>::getSite(args...);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteInnerBulk {
    typedef HaloIndexer<LatticeLayout, HaloDepth> HInd;
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;

    inline gSite operator()(size_t index) {
        // size_t index = blockdim.x*blockidx.x+threadidx.x;
        sitexyzt coord = HInd::getCenterCoord(index);
        gSite site = GInd::getSite(coord.x, coord.y, coord.z, coord.t);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteInnerBulkStack {
    typedef HaloIndexer<LatticeLayout, HaloDepth> HInd;
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;

    inline gSiteStack operator()(size_t index, size_t idx_y) {
        // size_t index = blockdim.x*blockidx.x+threadidx.x;
        sitexyzt coord = HInd::getCenterCoord(index);
        gSiteStack site = GInd::getSiteStack(coord.x, coord.y, coord.z, coord.t, idx_y);
        return site;
    }
};


template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteHalo {
    typedef HaloIndexer<LatticeLayout, HaloDepth> HInd;
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;

    inline gSite operator()(size_t index) {
        // size_t index = blockdim.x*blockidx.x+threadidx.x;
        sitexyzt coord = HInd::getInnerCoord(index);
        gSite site = GInd::getSite(coord.x, coord.y, coord.z, coord.t);
        return site;
    }
};


template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteHaloStack {
    typedef HaloIndexer<LatticeLayout, HaloDepth> HInd;
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;

    inline gSiteStack operator()(size_t index, size_t idx_y) {
        // size_t index = blockdim.x*blockidx.x+threadidx.x;
        sitexyzt coord = HInd::getInnerCoord(index);
        gSiteStack site = GInd::getSiteStack(coord.x, coord.y, coord.z, coord.t, idx_y);
        return site;
    }
};
//Fix below this commment


struct CalcGSiteHaloLookup {
    gSite* HaloSites;

    CalcGSiteHaloLookup(gSite* HalSites) : HaloSites(HalSites) {}

    inline  gSite operator()(size_t index) {
        // size_t index = blockdim.x*blockidx.x+threadidx.x;
        gSite site = HaloSites[index];
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteSpatialFull {
    template<typename... Args>
    inline  gSite operator()(Args... args) {
        gSite site = GIndexer<LatticeLayout, HaloDepth>::getSiteSpatialFull(args...);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteSpatial {
    template<typename... Args>
    inline  gSite operator()(Args... args) {
        gSite site = GIndexer<LatticeLayout, HaloDepth>::getSiteSpatial(args...);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteStack {
    template<typename... Args>
    inline  gSiteStack operator()(Args... args) {
        gSiteStack site = GIndexer<LatticeLayout, HaloDepth>::getSiteStack(args...);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteStackFull {
    template<typename... Args>
    inline  gSiteStack operator()(Args... args) {
        gSiteStack site = GIndexer<LatticeLayout, HaloDepth>::getSiteStackFull(args...);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteAllMu {
    template<typename... Args>
    inline  gSiteMu operator()(Args... args) {
        gSiteMu site = GIndexer<LatticeLayout, HaloDepth>::getSiteMu(args...);
        return site;
    }
};

template<uint8_t mu, Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteAtMu {
    template<typename... Args>
    inline  gSiteMu operator()(Args... args) {
        gSiteMu site = GIndexer<LatticeLayout, HaloDepth>::getSiteMu(args..., mu);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteAllMuFull {

    template<typename... Args>
    inline  gSiteMu operator()(Args... args) {
        gSiteMu site = GIndexer<LatticeLayout, HaloDepth>::getSiteMuFull(args...);
        return site;
    }
};

template<uint8_t mu, Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteAtMuFull {
    template<typename... Args>
    inline  gSiteMu operator()(Args... args) {
        gSiteMu site = GIndexer<LatticeLayout, HaloDepth>::getSiteMuFull(args..., mu);
        return site;
    }
};

template<size_t stack, Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteAtStackFull {
    template<typename... Args>
    inline  gSiteStack operator()(Args... args) {
        gSiteStack site = GIndexer<LatticeLayout, HaloDepth>::getSiteStackFull(args..., stack);
        return site;
    }
};

template<size_t stack, Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteAtStack {
    template<typename... Args>
    inline  gSiteStack operator()(Args... args) {
        gSiteStack site = GIndexer<LatticeLayout, HaloDepth>::getSiteStack(args..., stack);
        return site;
    }
};

//! When you want to run over the Odd part of an object with Layout=All. We need an offset to do that.
template<size_t stack, Layout LatticeLayout, size_t HaloDepth>
struct CalcOddGSiteAtStack {
    template<typename... Args>
    inline  gSiteStack operator()(Args... args) {
        gSiteStack site = GIndexer<LatticeLayout, HaloDepth>::getSiteStackOdd(args..., stack);
        return site;
    }
};


template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteLoopMu {
    template<typename... Args>
    inline  gSite operator()(Args... args) {
        gSite site = GIndexer<LatticeLayout, HaloDepth>::getSite(args...);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteLoopStack {
    template<typename... Args>
    inline  gSite operator()(Args... args) {
        gSite site = GIndexer<LatticeLayout, HaloDepth>::getSite(args...);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteLoopMuFull {
    template<typename... Args>
    inline  gSite operator()(Args... args) {
        gSite site = GIndexer<LatticeLayout, HaloDepth>::getSiteFull(args...);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteLoopStackFull {
    template<typename... Args>
    inline  gSite operator()(Args... args) {
        gSite site = GIndexer<LatticeLayout, HaloDepth>::getSiteFull(args...);
        return site;
    }
};

//! use this if you don't actually need to read in from any site, for example when initializing point sources
template<Layout LatticeLayout, size_t HaloDepth>
struct ReadDummy {
    template<typename... Args> inline  gSite operator()(__attribute__((unused)) Args... args) {
        return GIndexer<LatticeLayout, HaloDepth>::getSite(99999,99999,99999,99999);
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct WriteAtLoopMu {
    inline  gSiteMu operator()(const gSite &site, size_t mu) {
        return GIndexer<LatticeLayout, HaloDepth>::getSiteMu(site, mu);
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct WriteAtLoopStack {
    inline  gSiteStack operator()(const gSite &site, size_t stack) {
        return GIndexer<LatticeLayout, HaloDepth>::getSiteStack(site, stack);
    }
};



struct WriteAtRead {
    inline  gSite operator()(const gSite &site) const {
        return site;
    }
};



struct WriteAtReadStack {
    inline  gSiteStack operator()(const gSiteStack &site) {
        return site;
    }
};

struct WriteAtReadMu {
    inline  gSiteMu operator()(const gSiteMu &siteMu) {
        return siteMu;
    }
};

//! Writes to the same fixed site regardless of input! You probably want to call this with <blocksize=1> and elems=1.
//! Useful for constructing point sources.
template<Layout LatticeLayout, size_t HaloDepth>
struct WriteAtFixedSite {
    const gSite _fixed_site;
    explicit WriteAtFixedSite(const gSite mysite) : _fixed_site(mysite) {}
    inline  gSite operator()(__attribute__((unused)) const gSite dummy) {
        return _fixed_site;
    }
};


template<bool onDevice, Layout LatticeLayout, size_t HaloDepth, unsigned WorkGroupSize = 64, typename Functor>
void iterateOverFullAllMu(Functor op) {
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;
    CalcGSiteAllMuFull<LatticeLayout, HaloDepth> calcGSiteAllMuFull;
    iterateFunctorNoReturn<onDevice, WorkGroupSize>(op, calcGSiteAllMuFull, GInd::getLatData().vol4Full, 4);
}

template<bool onDevice, Layout LatticeLayout, size_t HaloDepth, unsigned WorkGroupSize = 64, typename Functor>
void iterateOverBulkAllMu(Functor op) {
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;
    CalcGSiteAllMu<LatticeLayout, HaloDepth> calcGSiteAllMu;
    iterateFunctorNoReturn<onDevice, WorkGroupSize>(op, calcGSiteAllMu, GInd::getLatData().vol4, 4);
}

template<bool onDevice, Layout LatticeLayout, size_t HaloDepth, uint8_t mu, unsigned WorkGroupSize = 256, typename Functor>
void iterateOverFullAtMu(Functor op) {
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;
    CalcGSiteAtMuFull<mu, LatticeLayout, HaloDepth> calcGSiteAtMuFull;
    iterateFunctorNoReturn<onDevice, WorkGroupSize>(op, calcGSiteAtMuFull, GInd::getLatData().vol4Full);
}

template<bool onDevice, Layout LatticeLayout, size_t HaloDepth, uint8_t mu, unsigned WorkGroupSize = 256, typename Functor>
void iterateOverBulkAtMu(Functor op) {
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;
    CalcGSiteAtMu<mu, LatticeLayout, HaloDepth> calcGSiteAtMu;
    iterateFunctorNoReturn<onDevice, WorkGroupSize>(op, calcGSiteAtMu, GInd::getLatData().vol4);
}

template<bool onDevice, Layout LatticeLayout, size_t HaloDepth, unsigned WorkGroupSize = 256, typename Functor>
void iterateOverBulk(Functor op) {
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;
    CalcGSite< LatticeLayout, HaloDepth> calcGSite;
    if(LatticeLayout == All){
        iterateFunctorNoReturn<onDevice, WorkGroupSize>(op, calcGSite, GInd::getLatData().vol4);
    }
    else{
        iterateFunctorNoReturn<onDevice, WorkGroupSize>(op, calcGSite, GInd::getLatData().vol4/2);
    }
}
