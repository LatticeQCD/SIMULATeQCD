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

    template<unsigned WgSize = DEFAULT_NBLOCKS_CONST, typename CalcReadInd, typename CalcWriteInd, typename Object>
    void iterateWithConstObject(Object ob, CalcReadInd calcReadInd, CalcWriteInd calcWriteInd,
    const size_t elems_x, sycl::queue q);

    template<unsigned WgSize = DEFAULT_NBLOCKS, typename CalcReadInd, typename CalcWriteInd, typename Functor>
    void iterateFunctor(Functor Op, CalcReadInd calcReadInd, CalcWriteInd calcWriteInd, 
    const size_t elems_x, sycl::queue q);


    template<size_t Nloops, unsigned WgSize = DEFAULT_NBLOCKS_LOOP, typename CalcReadInd, typename CalcWriteInd, typename Functor>
    void iterateFunctorLoop(Functor op, CalcReadInd calcReadInd, CalcWriteInd calcWriteInd, 
    const size_t elems_x, sycl::queue q, size_t Nmax = Nloops);

};

template<bool onDevice, class Accessor>
template<unsigned WgSize, typename CalcReadInd, typename CalcWriteInd, typename Object>
void RunFunctors<onDevice, Accessor>::iterateWithConstObject(Object ob, CalcReadInd calcReadInd, CalcWriteInd calcWriteInd,
    const size_t elems_x, sycl::queue q) {

    q.submit([&] (sycl::handler& cgh) {

        auto iteration_range = sycl::nd_range{sycl::range<1>{elems_x}, sycl::range<1>{DEFAULT_NBLOCKS_CONST}};
        auto acc = getAccessor();

        cgh.parallel_for(iteration_range, [=](sycl::nd_item<1> itm) {
            size_t i = itm.get_local_id()+itm.get_group().get_group_id(0)*itm.get_local_range(0);

            auto site = calcReadInd(i);
            acc.setElement(CalcWriteInd(site), ob);
        });

    }).wait();


}



template<bool onDevice, class Accessor>
template<unsigned WgSize, typename CalcReadInd, typename CalcWriteInd, typename Functor>
void RunFunctors<onDevice, Accessor>::iterateFunctor(Functor op, CalcReadInd calcReadInd, 
    CalcWriteInd calcWriteInd, const size_t elems_x, sycl::queue q) {
    

    q.submit([&] (sycl::handler& cgh) {

        auto iteration_range = sycl::nd_range{sycl::range<1> {elems_x}, sycl::range<1>{DEFAULT_NBLOCKS}};
        auto acc = getAccessor(); //implicit capture of 'this' is not allowed inside sycl kernel so we get hold of a local copy here

        cgh.parallel_for(iteration_range, [=](sycl::nd_item<1> itm) {
            size_t i = itm.get_local_id()+itm.get_group().get_group_id(0)*itm.get_local_range(0);
            
            auto site = calcReadInd(i);
            acc.setElement(calcWriteInd(site), op(site));
        });

    }).wait();
}

template<bool onDevice, class Accessor>
template<size_t Nloops, unsigned WgSize, typename CalcReadInd, typename CalcWriteInd, typename Functor>
void RunFunctors<onDevice, Accessor>::iterateFunctorLoop(Functor op, CalcReadInd calcReadInd,
CalcWriteInd calcWriteInd, const size_t elems_x, sycl::queue q, size_t Nmax) {
    
    if (Nmax > Nloops) {
        throw std::runtime_error(stdLogger.fatal("Nmax larger than Nloops!"));
    }

    q.submit([&] (sycl::handler& cgh) {
        auto iteration_range = sycl::nd_range{sycl::range<1> {elems_x}, sycl::range<1>{DEFAULT_NBLOCKS}};
        auto acc = getAccessor();

        cgh.parallel_for(iteration_range, [=] (sycl::nd_item<1> itm) {
            size_t i = itm.get_local_id() + itm.get_group().get_group_id(0)*itm.get_local_range(0);

            auto site = calcReadInd(i);
            op.initialize(site);

            for (size_t loopIdx = 0; loopIdx < Nloops; loopIdx++) {
                acc.setElement(calcWriteInd(site, loopIdx), op(site, loopIdx));
            }
        });
    }).wait();
}

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSite {
    template<typename... Args>
    inline gSite operator()(Args... args) const {
        gSite site = GIndexer<LatticeLayout, HaloDepth>::getSite(args...);
        return site;
    }
};


struct WriteAtRead {
    inline __host__ __device__ gSite operator()(const gSite &site) const {
        return site;
    }
};

