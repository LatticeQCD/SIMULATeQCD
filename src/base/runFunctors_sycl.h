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
/*
    template<unsigned WgSize = DEFAULT_NBLOCKS_CONST, typename CalcReadInd, typename CalcWriteInd, typename Object>
    void iterateWithConstObject(Object ob, CalcReadInd calcReadInd, CalcWriteInd, calcWriteInd,
    const size_t elems_x, const size_t elems_y = 1, const size_t elems_z = 1, sycl::queue q = default_queue);
*/
    template<unsigned WgSize = DEFAULT_NBLOCKS, typename CalcReadInd, typename CalcWriteInd, typename Functor>
    void iterateFunctor(Functor Op, CalcReadInd calcReadInd, CalcWriteInd calcWriteInd, 
    const size_t elems_x, sycl::queue q);

/*
    template<size_t Nloops, unsigned WgSize = DEFAULT_NBLOCKS_LOOP, typename CalcReadInd, typename CalcWriteInd, typename Functor>
    void iterateFunctorLoop(Functor op, CalcReadInd calcReadInd, CalcWriteInd calcWriteInd, 
    const size_t elems_x, const size_t elems_y = 1, const size_t elems_z =1, sycl::queue q = default_queue);
*/
    private:
    sycl::queue default_queue;
};

template<bool onDevice, class Accessor>
template<unsigned WgSize, typename CalcReadInd, typename CalcWriteInd, typename Functor>
void RunFunctors<onDevice, Accessor>::iterateFunctor(Functor op, CalcReadInd calcReadInd, 
    CalcWriteInd calcWriteInd, const size_t elems_x, sycl::queue q) {
    

    q.submit([&] (sycl::handler& cgh) {
        auto iteration_range = sycl::nd_range{sycl::range<1> {elems_x}, sycl::range<1>{DEFAULT_NBLOCKS}};
        auto acc = getAccessor();
        cgh.parallel_for(iteration_range, [=](sycl::nd_item<1> itm) {
            size_t i = itm.get_local_id()+itm.get_group().get_group_id(0)*itm.get_local_range(0);
            
            auto site = calcReadInd(i);
            acc.setElement(calcWriteInd(site), op(site));
            // performFunctor(getAccessor(), op, calcReadInd, calcWriteInd, elems_x);
        });
    });
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
// template<typename Accessor, typename Functor, typename CalcReadInd, typename CalcWriteInd>
// void performFunctor(Accessor res, Functor op, CalcReadInd calcReadInd, CalcWriteInd calcWriteInd, const size_t size_x) {

//     // size_t i = blockDim.x * blockIdx.x + threadIdx.x;
//     size_t i = 
//     if (i >= size_x) {
//         return;
//     }

//     //Site can be anything. Therefore auto
// #ifdef USE_CUDA
//     auto site = calcReadInd(blockDim, blockIdx, threadIdx);
// #elif defined USE_HIP
// auto site = calcReadInd(dim3(blockDim), GetUint3(dim3(blockIdx)), GetUint3(dim3(threadIdx)));
// #endif
//     auto site = calcReadInd()
//     res.setElement(calcWriteInd(site), op(site));
// }