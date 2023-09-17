#ifndef _runFunctors_h_

#define _runFunctors_h_
#include "wrapper/gpu_wrapper.h"
#include "../define.h"
#include "../base/gutils.h"
#include "math/operators.h"
#include "../base/communication/communicationBase.h"

#include "../base/indexer/HaloIndexer.h"

#define DEFAULT_NBLOCKS  128
#define DEFAULT_NBLOCKS_LOOP  128
#define DEFAULT_NBLOCKS_CONST  256

template<bool onDevice, class Accessor>
class RunFunctors {
public:
    virtual Accessor getAccessor() const = 0;

    template<unsigned BlockSize = DEFAULT_NBLOCKS_CONST, typename CalcReadInd, typename CalcWriteInd, typename Object>
    void iterateWithConstObject(Object ob, CalcReadInd calcReadInd, CalcWriteInd calcWriteInd,
            const size_t elems_x, const size_t elems_y = 1, const size_t elems_z = 1,gpuStream_t stream = (gpuStream_t)nullptr);


    template<unsigned BlockSize = DEFAULT_NBLOCKS, typename CalcReadInd, typename CalcWriteInd, typename Functor>
    void iterateFunctor(Functor op, CalcReadInd calcReadInd, CalcWriteInd calcWriteInd,
            const size_t elems_x, const size_t elems_y = 1, const size_t elems_z = 1,gpuStream_t stream = (gpuStream_t)nullptr);



    //ranluo
    template<unsigned BlockSize = DEFAULT_NBLOCKS, typename CalcReadInd, typename Functor>
    void iterateFunctor_half(Functor op, CalcReadInd calcReadInd, 
            const size_t elems_x, const size_t elems_y = 1, const size_t elems_z = 1,gpuStream_t stream = (gpuStream_t)nullptr);

    template<unsigned BlockSize = DEFAULT_NBLOCKS, typename CalcReadInd, typename Functor>
    void iterateFunctor_single(Functor op, CalcReadInd calcReadInd,
            const size_t elems_x, const size_t elems_y = 1, const size_t elems_z = 1,gpuStream_t stream = (gpuStream_t)nullptr);


    template<size_t Nloops, unsigned BlockSize = DEFAULT_NBLOCKS_LOOP, typename CalcReadInd, typename CalcWriteInd, typename Functor>
    void iterateFunctorLoop(Functor op, CalcReadInd calcReadInd, CalcWriteInd calcWriteInd,
            const size_t elems_x, const size_t elems_y = 1, const size_t elems_z = 1,gpuStream_t stream = (gpuStream_t)nullptr, size_t Nmax=Nloops);
};

#ifdef __GPUCC__

#ifdef USE_HIP_AMD

__host__ __device__ static inline HIP_vector_type<unsigned int, 3> GetUint3(dim3 Idx){

        return HIP_vector_type<unsigned int, 3>(Idx.x, Idx.y, Idx.z);

};
#elif defined USE_HIP_NVIDIA
__host__ __device__ static dim3 GetUint3(dim3 Idx){

        return Idx;

};
#endif


template<typename Accessor, typename Functor, typename CalcReadInd, typename CalcWriteInd>
__global__ void performFunctor(Accessor res, Functor op, CalcReadInd calcReadInd, CalcWriteInd calcWriteInd, const size_t size_x) {

    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= size_x) {
        return;
    }

    //Site can be anything. Therefore auto
#ifdef USE_CUDA
    auto site = calcReadInd(blockDim, blockIdx, threadIdx);
#elif defined USE_HIP
auto site = calcReadInd(dim3(blockDim), GetUint3(dim3(blockIdx)), GetUint3(dim3(threadIdx)));
#endif
    res.setElement(calcWriteInd(site), op(site));
}




//ranluo
template<typename Accessor, typename Functor, typename CalcReadInd>
__global__ void half_Dslash(Accessor res, Functor op, CalcReadInd calcReadInd, const size_t size_x)
{
    if (blockDim.x * blockIdx.x + threadIdx.x >= size_x) {
        return;
    }
    auto site = calcReadInd(blockDim, blockIdx, threadIdx);
    res.setElement(site, op(site));
}

template<typename Accessor, typename Functor, typename CalcReadInd>
__global__ void single_Dslash(Accessor res, Functor op, CalcReadInd calcReadInd, const size_t size_x)
{
    if (blockDim.x * blockIdx.x + threadIdx.x >= size_x) {
        return;
    }
    auto site = calcReadInd(blockDim, blockIdx, threadIdx);
    res.setElement(site, op(site));
}


template<size_t Nloops, typename Accessor, typename Functor, typename CalcReadInd, typename CalcWriteInd>
__global__ void performFunctorLoop(Accessor res, Functor op, CalcReadInd calcReadInd,
        CalcWriteInd calcWriteInd, const size_t size_x, size_t Nmax=Nloops) {

    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= size_x) {
        return;
    }

    //Site can be anything. Therefore auto

#ifdef USE_CUDA
    auto site = calcReadInd(blockDim, blockIdx, threadIdx);
#elif defined USE_HIP
    auto site = calcReadInd(dim3(blockDim), GetUint3(dim3(blockIdx)), GetUint3(dim3(threadIdx)));
#endif
    op.initialize(site);
#ifdef USE_CUDA
#pragma unroll
#endif
    for (size_t loopIdx = 0; loopIdx < Nloops; loopIdx++){
        if(loopIdx >= Nmax) break;
        res.setElement(calcWriteInd(site, loopIdx), op(site, loopIdx));
    }
}


template<typename Accessor, typename Object, typename CalcReadInd, typename CalcWriteInd>
__global__ void performCopyConstObject(Accessor res, Object ob, CalcReadInd calcReadInd,
        CalcWriteInd calcWriteInd, const size_t size_x) {

    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= size_x) {
        return;
    }

    //Site can be anything. Therefore auto

#ifdef USE_CUDA
    auto site = calcReadInd(blockDim, blockIdx, threadIdx);
#elif defined USE_HIP
    auto site = calcReadInd(dim3(blockDim), GetUint3(dim3(blockIdx)), GetUint3(dim3(threadIdx)));
#endif
    res.setElement(calcWriteInd(site), ob);
}
#endif


template<bool onDevice, class Accessor>
template<unsigned BlockSize, typename CalcReadInd, typename CalcWriteInd, typename Functor>
void RunFunctors<onDevice, Accessor>::iterateFunctor(Functor op, CalcReadInd calcReadInd,
                                                                                   CalcWriteInd calcWriteInd,
                                                                                   const size_t elems_x,
                                                                                   const size_t elems_y,
                                                                                   const size_t elems_z,
                                                                                   __attribute__((unused)) gpuStream_t stream){

    dim3 blockDim;

    blockDim.x = BlockSize;
    blockDim.y = elems_y;
    blockDim.z = elems_z;

    //Grid only in x direction!
    const dim3 gridDim = static_cast<int> (ceilf(static_cast<float> (elems_x)
                / static_cast<float> (blockDim.x)));

    if (onDevice) {
#ifdef __GPUCC__

#ifdef USE_CUDA
        performFunctor<<< gridDim, blockDim,0, stream >>> (getAccessor(), op, calcReadInd, calcWriteInd, elems_x);
#elif defined USE_HIP
        hipLaunchKernelGGL(performFunctor, dim3(gridDim), dim3(blockDim), 0, stream , getAccessor(), op, calcReadInd, calcWriteInd, elems_x);
#endif
        gpuError_t gpuErr = gpuGetLastError();
        if (gpuErr)
            GpuError("performFunctor: Failed to launch kernel", gpuErr);
#else 
        static_assert(!onDevice, "Functor construction not available for device code outside .cpp files");
#endif
    } else {
        auto resAcc = getAccessor();
        uint3 blockIdx;
        blockIdx.y = 0;
        blockIdx.z = 0;

        uint3 threadIdx;
        for (size_t block_x = 0; block_x < gridDim.x; block_x++) {
            blockIdx.x = block_x;

            for (size_t thread_x = 0; thread_x < blockDim.x; thread_x++) {
                threadIdx.x = thread_x;

                for (size_t thread_y = 0; thread_y < blockDim.y; thread_y++) {
                    threadIdx.y = thread_y;

                    for (size_t thread_z = 0; thread_z < blockDim.z; thread_z++) {
                        threadIdx.z = thread_z;

                        auto site = calcReadInd(blockDim, blockIdx, threadIdx);

                        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
                        if (i >= elems_x) {
                            continue;
                        }
                        resAcc.setElement(calcWriteInd(site), op(site));
                    }
                }
            }
        }
    }
}





//ranluo
template<bool onDevice, class Accessor>
template<unsigned BlockSize, typename CalcReadInd, typename Functor>
void RunFunctors<onDevice, Accessor>::iterateFunctor_half(Functor op, CalcReadInd calcReadInd,
                                                                                   const size_t elems_x,
                                                                                   const size_t elems_y,
                                                                                   const size_t elems_z,
                                                                                   __attribute__((unused)) gpuStream_t stream){

    dim3 blockDim;

    blockDim.x = BlockSize;
    blockDim.y = elems_y;
    blockDim.z = elems_z;

    //Grid only in x direction!
    const dim3 gridDim = static_cast<int> (ceilf(static_cast<float> (elems_x)
                / static_cast<float> (blockDim.x)));

    if (onDevice) {
#ifdef __GPUCC__

#ifdef USE_CUDA
        half_Dslash<<< gridDim, blockDim,0, stream >>> (getAccessor(), op, calcReadInd, elems_x);
#elif defined USE_HIP
        hipLaunchKernelGGL(performFunctor, dim3(gridDim), dim3(blockDim), 0, stream , getAccessor(), op, calcReadInd, calcWriteInd, elems_x);
#endif
        gpuError_t gpuErr = gpuGetLastError();
        if (gpuErr)
            GpuError("performFunctor: Failed to launch kernel", gpuErr);
#else 
        static_assert(!onDevice, "Functor construction not available for device code outside .cpp files");
#endif
    }
}

template<bool onDevice, class Accessor>
template<unsigned BlockSize, typename CalcReadInd, typename Functor>
void RunFunctors<onDevice, Accessor>::iterateFunctor_single(Functor op, CalcReadInd calcReadInd,
                                                                                   const size_t elems_x,
                                                                                   const size_t elems_y,
                                                                                   const size_t elems_z,
                                                                                   __attribute__((unused)) gpuStream_t stream){

    dim3 blockDim;

    blockDim.x = BlockSize;
    blockDim.y = elems_y;
    blockDim.z = elems_z;

    //Grid only in x direction!
    const dim3 gridDim = static_cast<int> (ceilf(static_cast<float> (elems_x)
                / static_cast<float> (blockDim.x)));

    if (onDevice) {
#ifdef __GPUCC__

#ifdef USE_CUDA
        single_Dslash<<< gridDim, blockDim,0, stream >>> (getAccessor(), op, calcReadInd, elems_x);
#elif defined USE_HIP
        hipLaunchKernelGGL(performFunctor, dim3(gridDim), dim3(blockDim), 0, stream , getAccessor(), op, calcReadInd, calcWriteInd, elems_x);
#endif
        gpuError_t gpuErr = gpuGetLastError();
        if (gpuErr)
            GpuError("performFunctor: Failed to launch kernel", gpuErr);
#else 
        static_assert(!onDevice, "Functor construction not available for device code outside .cpp files");
#endif
    }
}




template<bool onDevice, class Accessor>
template<size_t Nloops, unsigned BlockSize, typename CalcReadInd, typename CalcWriteInd, typename Functor>
void RunFunctors<onDevice, Accessor>::iterateFunctorLoop(Functor op,
    CalcReadInd calcReadInd, CalcWriteInd calcWriteInd, const size_t elems_x, const size_t elems_y, const size_t elems_z,__attribute__((unused)) gpuStream_t stream, size_t Nmax) {

    dim3 blockDim;

    blockDim.x = BlockSize;
    blockDim.y = elems_y;
    blockDim.z = elems_z;

    if (Nmax > Nloops)
    {
        throw std::runtime_error(stdLogger.fatal("Nmax larger than Nloops!"));
    }


    //Grid only in x direction!
    const dim3 gridDim = static_cast<int> (ceilf(static_cast<float> (elems_x)
                / static_cast<float> (blockDim.x)));

    if (onDevice) {
#ifdef __GPUCC__

#ifdef USE_CUDA
        performFunctorLoop<Nloops> <<< gridDim, blockDim, 0, stream >>> (getAccessor(), op, calcReadInd, calcWriteInd, elems_x, Nmax);
#elif defined USE_HIP
        hipLaunchKernelGGL((performFunctorLoop<Nloops>), dim3(gridDim), dim3(blockDim), 0, stream , getAccessor(), op, calcReadInd, calcWriteInd, elems_x,         Nmax);
#endif

        gpuError_t gpuErr = gpuGetLastError();
        if (gpuErr)
            GpuError("performFunctorLoop: Failed to launch kernel", gpuErr);
#else
        static_assert(!onDevice, "Functor construction not available for device code outside .cpp files");
#endif
    } else {
        auto resAcc = getAccessor();
        uint3 blockIdx;
        blockIdx.y = 0;
        blockIdx.z = 0;

        uint3 threadIdx;
        for (size_t block_x = 0; block_x < gridDim.x; block_x++) {
            blockIdx.x = block_x;

            for (size_t thread_x = 0; thread_x < blockDim.x; thread_x++) {
                threadIdx.x = thread_x;

                for (size_t thread_y = 0; thread_y < blockDim.y; thread_y++) {
                    threadIdx.y = thread_y;

                    for (size_t thread_z = 0; thread_z < blockDim.z; thread_z++) {
                        threadIdx.z = thread_z;

                        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
                        if (i >= elems_x) {
                            continue;
                        }
                        auto site = calcReadInd(blockDim, blockIdx, threadIdx);

                        op.initialize(site);

                        for (size_t loopIdx = 0; loopIdx < Nloops; loopIdx++){
                            if(loopIdx >= Nmax) break; 
                                resAcc.setElement(calcWriteInd(site, loopIdx), op(site, loopIdx));
                        }
                    }
                }
            }
        }
    }
}


template<bool onDevice, class Accessor>
template<unsigned BlockSize, typename CalcReadInd, typename CalcWriteInd, typename Object>
void RunFunctors<onDevice, Accessor>::iterateWithConstObject(Object ob, CalcReadInd calcReadInd,
        CalcWriteInd calcWriteInd,
        const size_t elems_x,
        const size_t elems_y,
        const size_t elems_z,
        __attribute__((unused)) gpuStream_t stream ){

    dim3 blockDim;

    blockDim.x = BlockSize;
    blockDim.y = elems_y;
    blockDim.z = elems_z;

    //Grid only in x direction!
    const dim3 gridDim = static_cast<int> (ceilf(static_cast<float> (elems_x)
                / static_cast<float> (blockDim.x)));

    if (onDevice) {
#ifdef __GPUCC__

#ifdef USE_CUDA
        performCopyConstObject<<< gridDim, blockDim,0, stream >>> (getAccessor(), ob, calcReadInd, calcWriteInd, elems_x);
#elif defined USE_HIP
        hipLaunchKernelGGL((performCopyConstObject), dim3(gridDim), dim3(blockDim), 0, stream , getAccessor(), ob, calcReadInd, calcWriteInd, elems_x);
#endif


        gpuError_t gpuErr = gpuGetLastError();
        if (gpuErr)
            GpuError("CopyFromConst: Failed to launch kernel", gpuErr);
#else
        static_assert(!onDevice, "Functor construction not available for device code outside .cpp files");
#endif
    } else {
        auto resAcc = getAccessor();
        uint3 blockIdx;
        blockIdx.y = 0;
        blockIdx.z = 0;

        uint3 threadIdx;
        for (size_t block_x = 0; block_x < gridDim.x; block_x++) {
            blockIdx.x = block_x;

            for (size_t thread_x = 0; thread_x < blockDim.x; thread_x++) {
                threadIdx.x = thread_x;

                for (size_t thread_y = 0; thread_y < blockDim.y; thread_y++) {
                    threadIdx.y = thread_y;

                    for (size_t thread_z = 0; thread_z < blockDim.z; thread_z++) {
                        threadIdx.z = thread_z;

                        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
                        if (i >= elems_x) {
                            continue;
                        }
                        auto site = calcReadInd(blockDim, blockIdx, threadIdx);

                        resAcc.setElement(calcWriteInd(site), ob);
                    }
                }
            }
        }
    }
}


#ifdef __GPUCC__

template<typename Functor, typename CalcReadInd>
__global__ void performFunctorNoReturn(Functor op, CalcReadInd calcReadInd, const size_t size_x) {

    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= size_x) {
        return;
    }

    //Site can be anything. Therefore auto
#ifdef USE_CUDA
    auto site = calcReadInd(blockDim, blockIdx, threadIdx);
#elif defined USE_HIP
    auto site = calcReadInd(dim3(blockDim), GetUint3(dim3(blockIdx)), GetUint3(dim3(threadIdx)));
#endif
    op(site);
}

#endif


template<bool onDevice, size_t BlockSize = DEFAULT_NBLOCKS, typename CalcReadInd, typename Functor>
void iterateFunctorNoReturn(Functor op, CalcReadInd calcReadInd, const size_t elems_x,
        const size_t elems_y = 1,
        const size_t elems_z = 1,
        __attribute__((unused)) gpuStream_t stream = (gpuStream_t)nullptr){

    dim3 blockDim;

    blockDim.x = BlockSize;
    blockDim.y = elems_y;
    blockDim.z = elems_z;

    //Grid only in x direction!
    const dim3 gridDim = static_cast<int> (ceilf(static_cast<float> (elems_x)
                / static_cast<float> (blockDim.x)));

    if (onDevice) {
#ifdef __GPUCC__


#ifdef USE_CUDA
        performFunctorNoReturn<<< gridDim, blockDim, 0, stream >>> (op, calcReadInd, elems_x);
#elif defined USE_HIP
        hipLaunchKernelGGL((performFunctorNoReturn), dim3(gridDim), dim3(blockDim), 0, stream , op, calcReadInd, elems_x);
#endif

        gpuError_t gpuErr = gpuGetLastError();
        if (gpuErr)
            GpuError("performFunctor: Failed to launch kernel", gpuErr);
#else
        static_assert(!onDevice, "Functor construction not available for device code outside .cpp files");
#endif
    } else {
        uint3 blockIdx;
        blockIdx.y = 0;
        blockIdx.z = 0;

        uint3 threadIdx;
        for (size_t block_x = 0; block_x < gridDim.x; block_x++) {
            blockIdx.x = block_x;

            for (size_t thread_x = 0; thread_x < blockDim.x; thread_x++) {
                threadIdx.x = thread_x;

                for (size_t thread_y = 0; thread_y < blockDim.y; thread_y++) {
                    threadIdx.y = thread_y;

                    for (size_t thread_z = 0; thread_z < blockDim.z; thread_z++) {
                        threadIdx.z = thread_z;

                        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
                        if (i >= elems_x) {
                            continue;
                        }
                        auto site = calcReadInd(blockDim, blockIdx, threadIdx);
                        op(site);
                    }
                }
            }
        }
    }
}

#ifdef __GPUCC__

template<typename Functor, typename CalcReadWriteInd, class Accessor>
__global__ void performFunctorComm(Functor op, Accessor acc,  CalcReadWriteInd calcReadWriteInd, const size_t subHaloSize, const size_t size_x ) {

    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= size_x) {
        return;
    }

    size_t HaloMuIndex = blockDim.x * blockIdx.x + threadIdx.x;
    size_t HaloIndex = HaloMuIndex % subHaloSize;
    size_t mu = HaloMuIndex / subHaloSize;

    auto site = calcReadWriteInd(HaloIndex,mu);
    acc.setElement(site, op(site));
}

#endif


template<bool onDevice, size_t BlockSize = DEFAULT_NBLOCKS, typename CalcReadWriteInd, typename Functor, class Accessor>
void iterateFunctorComm(Functor op, Accessor acc, CalcReadWriteInd calcReadWriteInd, const size_t subHaloSize, const size_t elems_x,
                                   const size_t elems_y = 1,
                                   const size_t elems_z = 1,
                                   __attribute__((unused)) gpuStream_t stream = (gpuStream_t)nullptr){

    dim3 blockDim;

    blockDim.x = BlockSize;
    blockDim.y = elems_y;
    blockDim.z = elems_z;

    //Grid only in x direction!
    const dim3 gridDim = static_cast<int> (ceilf(static_cast<float> (elems_x)
                                                 / static_cast<float> (blockDim.x)));

    if (onDevice) {
#ifdef __GPUCC__

#ifdef USE_CUDA
        performFunctorComm<<< gridDim, blockDim, 0, stream >>> (op, acc, calcReadWriteInd, subHaloSize, elems_x);
#elif defined USE_HIP
        hipLaunchKernelGGL((performFunctorComm), dim3(gridDim), dim3(blockDim), 0, stream , op, acc, calcReadWriteInd, subHaloSize, elems_x);
#endif

        gpuError_t gpuErr = gpuGetLastError();
        if (gpuErr)
            GpuError("performFunctor: Failed to launch kernel", gpuErr);
#else
        static_assert(!onDevice, "Functor construction not available for device code outside .cpp files");
#endif
    } else {
        uint3 blockIdx;
        blockIdx.y = 0;
        blockIdx.z = 0;

        uint3 threadIdx;
        for (size_t block_x = 0; block_x < gridDim.x; block_x++) {
            blockIdx.x = block_x;

            for (size_t thread_x = 0; thread_x < blockDim.x; thread_x++) {
                threadIdx.x = thread_x;

                for (size_t thread_y = 0; thread_y < blockDim.y; thread_y++) {
                    threadIdx.y = thread_y;

                    for (size_t thread_z = 0; thread_z < blockDim.z; thread_z++) {
                        threadIdx.z = thread_z;

                        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
                        if (i >= elems_x) {
                            continue;
                        }
                        size_t HaloMuIndex = blockDim.x * blockIdx.x + threadIdx.x;
                        size_t HaloIndex = HaloMuIndex % subHaloSize;
                        size_t mu = HaloMuIndex / subHaloSize;

                        auto site = calcReadWriteInd(HaloIndex,mu);
                        acc.setElement(site, op(site));
                    }
                }
            }
        }
    }
}

//Simple functions to calculate lattice indices. These functions are usually passed to 
//constructing Kernels (See spinorfield.h) as argument

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteFull {

    template<typename... Args>
    inline __host__ __device__ gSite operator()(Args... args) {
        gSite site = GIndexer<LatticeLayout, HaloDepth>::getSiteFull(args...);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSite {
    template<typename... Args>
    inline __host__ __device__ gSite operator()(Args... args) {
        gSite site = GIndexer<LatticeLayout, HaloDepth>::getSite(args...);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteSpatialFull {
    template<typename... Args>
    inline __host__ __device__ gSite operator()(Args... args) {
        gSite site = GIndexer<LatticeLayout, HaloDepth>::getSiteSpatialFull(args...);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteSpatial {
    template<typename... Args>
    inline __host__ __device__ gSite operator()(Args... args) {
        gSite site = GIndexer<LatticeLayout, HaloDepth>::getSiteSpatial(args...);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteStack {
    template<typename... Args>
    inline __host__ __device__ gSiteStack operator()(Args... args) {
        return GIndexer<LatticeLayout, HaloDepth>::getSiteStack(args...);
    }
};



//Revised by ranluo
template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteStack_Center {
    template<typename... Args>
    inline __host__ __device__ gSiteStack operator()(Args... args) {
        return GIndexer<LatticeLayout, HaloDepth>::getSiteStack_Center(args...);
    }
};

//Revised by ranluo
template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteStack_InnerHalo {
    template<typename... Args>
    inline __host__ __device__ gSiteStack operator()(Args... args) {
        return GIndexer<LatticeLayout, HaloDepth>::getSiteStack_InnerHalo(args...);
    }
};



template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteStackFull {
    template<typename... Args>
    inline __host__ __device__ gSiteStack operator()(Args... args) {
        gSiteStack site = GIndexer<LatticeLayout, HaloDepth>::getSiteStackFull(args...);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteAllMu {
    template<typename... Args>
    inline __host__ __device__ gSiteMu operator()(Args... args) {
        gSiteMu site = GIndexer<LatticeLayout, HaloDepth>::getSiteMu(args...);
        return site;
    }
};

template<uint8_t mu, Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteAtMu {
    template<typename... Args>
    inline __host__ __device__ gSiteMu operator()(Args... args) {
        gSiteMu site = GIndexer<LatticeLayout, HaloDepth>::getSiteMu(args..., mu);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteAllMuFull {

    template<typename... Args>
    inline __host__ __device__ gSiteMu operator()(Args... args) {
        gSiteMu site = GIndexer<LatticeLayout, HaloDepth>::getSiteMuFull(args...);
        return site;
    }
};

template<uint8_t mu, Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteAtMuFull {
    template<typename... Args>
    inline __host__ __device__ gSiteMu operator()(Args... args) {
        gSiteMu site = GIndexer<LatticeLayout, HaloDepth>::getSiteMuFull(args..., mu);
        return site;
    }
};

template<size_t stack, Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteAtStackFull {
    template<typename... Args>
    inline __host__ __device__ gSiteStack operator()(Args... args) {
        gSiteStack site = GIndexer<LatticeLayout, HaloDepth>::getSiteStackFull(args..., stack);
        return site;
    }
};

template<size_t stack, Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteAtStack {
    template<typename... Args>
    inline __host__ __device__ gSiteStack operator()(Args... args) {
        gSiteStack site = GIndexer<LatticeLayout, HaloDepth>::getSiteStack(args..., stack);
        return site;
    }
};

//! When you want to run over the Odd part of an object with Layout=All. We need an offset to do that.
template<size_t stack, Layout LatticeLayout, size_t HaloDepth>
struct CalcOddGSiteAtStack {
    template<typename... Args>
    inline __host__ __device__ gSiteStack operator()(Args... args) {
        gSiteStack site = GIndexer<LatticeLayout, HaloDepth>::getSiteStackOdd(args..., stack);
        return site;
    }
};


template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteLoopMu {
    template<typename... Args>
    inline __host__ __device__ gSite operator()(Args... args) {
        gSite site = GIndexer<LatticeLayout, HaloDepth>::getSite(args...);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteLoopStack {
    template<typename... Args>
    inline __host__ __device__ gSite operator()(Args... args) {
        gSite site = GIndexer<LatticeLayout, HaloDepth>::getSite(args...);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteLoopMuFull {
    template<typename... Args>
    inline __host__ __device__ gSite operator()(Args... args) {
        gSite site = GIndexer<LatticeLayout, HaloDepth>::getSiteFull(args...);
        return site;
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct CalcGSiteLoopStackFull {
    template<typename... Args>
    inline __host__ __device__ gSite operator()(Args... args) {
        gSite site = GIndexer<LatticeLayout, HaloDepth>::getSiteFull(args...);
        return site;
    }
};

//! use this if you don't actually need to read in from any site, for example when initializing point sources
template<Layout LatticeLayout, size_t HaloDepth>
struct ReadDummy {
    template<typename... Args> inline __host__ __device__ gSite operator()(__attribute__((unused)) Args... args) {
        return GIndexer<LatticeLayout, HaloDepth>::getSite(99999,99999,99999,99999);
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct WriteAtLoopMu {
    inline __host__ __device__ gSiteMu operator()(const gSite &site, size_t mu) {
        return GIndexer<LatticeLayout, HaloDepth>::getSiteMu(site, mu);
    }
};

template<Layout LatticeLayout, size_t HaloDepth>
struct WriteAtLoopStack {
    inline __host__ __device__ gSiteStack operator()(const gSite &site, size_t stack) {
        return GIndexer<LatticeLayout, HaloDepth>::getSiteStack(site, stack);
    }
};

struct WriteAtRead {
    inline __host__ __device__ gSite operator()(const gSite &site) {
        return site;
    }
};

struct WriteAtReadStack {
    inline __host__ __device__ gSiteStack operator()(const gSiteStack &site) {
        return site;
    }
};

struct WriteAtReadMu {
    inline __host__ __device__ gSiteMu operator()(const gSiteMu &siteMu) {
        return siteMu;
    }
};

//! Writes to the same fixed site regardless of input! You probably want to call this with <blocksize=1> and elems=1.
//! Useful for constructing point sources.
template<Layout LatticeLayout, size_t HaloDepth>
struct WriteAtFixedSite {
    const gSite _fixed_site;
    explicit WriteAtFixedSite(const gSite mysite) : _fixed_site(mysite) {}
    inline __host__ __device__ gSite operator()(__attribute__((unused)) const gSite dummy) {
        return _fixed_site;
    }
};


template<bool onDevice, Layout LatticeLayout, size_t HaloDepth, unsigned BlockSize = 64, typename Functor>
void iterateOverFullAllMu(Functor op) {
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;
    CalcGSiteAllMuFull<LatticeLayout, HaloDepth> calcGSiteAllMuFull;
    iterateFunctorNoReturn<onDevice, BlockSize>(op, calcGSiteAllMuFull, GInd::getLatData().vol4Full, 4);
}

template<bool onDevice, Layout LatticeLayout, size_t HaloDepth, unsigned BlockSize = 64, typename Functor>
void iterateOverBulkAllMu(Functor op) {
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;
    CalcGSiteAllMu<LatticeLayout, HaloDepth> calcGSiteAllMu;
    iterateFunctorNoReturn<onDevice, BlockSize>(op, calcGSiteAllMu, GInd::getLatData().vol4, 4);
}

template<bool onDevice, Layout LatticeLayout, size_t HaloDepth, uint8_t mu, unsigned BlockSize = 256, typename Functor>
void iterateOverFullAtMu(Functor op) {
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;
    CalcGSiteAtMuFull<mu, LatticeLayout, HaloDepth> calcGSiteAtMuFull;
    iterateFunctorNoReturn<onDevice, BlockSize>(op, calcGSiteAtMuFull, GInd::getLatData().vol4Full);
}

template<bool onDevice, Layout LatticeLayout, size_t HaloDepth, uint8_t mu, unsigned BlockSize = 256, typename Functor>
void iterateOverBulkAtMu(Functor op) {
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;
    CalcGSiteAtMu<mu, LatticeLayout, HaloDepth> calcGSiteAtMu;
    iterateFunctorNoReturn<onDevice, BlockSize>(op, calcGSiteAtMu, GInd::getLatData().vol4);
}

template<bool onDevice, Layout LatticeLayout, size_t HaloDepth, unsigned BlockSize = 256, typename Functor>
void iterateOverBulk(Functor op) {
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;
    CalcGSite< LatticeLayout, HaloDepth> calcGSite;
    if(LatticeLayout == All){
        iterateFunctorNoReturn<onDevice, BlockSize>(op, calcGSite, GInd::getLatData().vol4);
    }
    else{
        iterateFunctorNoReturn<onDevice, BlockSize>(op, calcGSite, GInd::getLatData().vol4/2);
    }
}

#endif
