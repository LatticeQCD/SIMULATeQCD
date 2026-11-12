/*
 * Stack-in-grid launch of the MDWF 5D kernels: default on; MDWF_STACK_GRID=0 restores SIMULATeQCD's launch.
 *
 * SIMULATeQCD launches a stacked kernel with blockDim = (BlockSize / Ls sites, Ls stacks) (RunFunctors::iterateFunctor,
 * iterateFunctorNoReturn, Spinorfield::iterateOverBulk). The spinor memory is stack-major SoA
 * (_elements[c][site + stack * sizeh]), so a warp reads only BlockSize / Ls consecutive sites per stack: at Ls = 24 with
 * the hop block size 32 one site (24-thread blocks: 8 idle lanes per warp, 8 of 32 bytes per memory sector used), at
 * Ls = 16 two sites. Here instead every block holds mdwfStackGridBlock() consecutive sites at ONE stack, the stack is
 * blockIdx.x and the site block blockIdx.y, so the Ls blocks of the same sites run back to back (CUDA issues blocks
 * x-fastest): full warps, coalesced loads and stores for any Ls, and the gauge links of those sites are still in L2 for
 * the next stack (a site-block-fastest order would stream the whole local gauge field, ~150 MB single at 32x32x16x8 >
 * 50 MB L2, once per stack). More than 65535 site blocks (gridDim.y limit): site blocks in x, stacks in y.
 * (Grid / GPT store s fastest in memory and run s fastest in the thread index; SIMULATeQCD stores s slowest, so the
 * closest match without a new layout is coalescing within a stack + stack-adjacent blocks.)
 *
 * Pure re-mapping: every thread computes the same (site, stack) element with the same code, and no MDWF kernel shares
 * data between threads (no __shared__ / __syncthreads), so the results are bit-identical to the default launch.
 * MDWF_STACK_GRID_BLOCK = 32 / 64 / 128 / 256 sets the block size (default 128).
 */

#pragma once

#include "../../spinor/spinorfield.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

#define MDWF_STACK_GRID_MAX_BLOCK 256

// Settable at run time (the operator benchmark switches it); initial value from the environment: on unless
// MDWF_STACK_GRID=0. Covered: Wilson hops (mdwfOverlapHop), fused Schur kernels, the mixed-CG vector updates and the
// single-reduction 5D dot product.
inline bool &mdwfStackGrid() {
    static bool on = [] {
        const char *env = std::getenv("MDWF_STACK_GRID");
        return env == nullptr || std::string(env) != "0";
    }();
    return on;
}

inline unsigned &mdwfStackGridBlock() {
    static unsigned block = [] {
        const char *env = std::getenv("MDWF_STACK_GRID_BLOCK");
        const unsigned b = env != nullptr ? static_cast<unsigned>(std::atoi(env)) : 128u;
        return (b == 32 || b == 64 || b == 128 || b == 256) ? b : 128u;
    }();
    return block;
}

// (i, s) -> the i-th site of the layout at stack s (as CalcGSiteStack).
template<Layout LatLayout, size_t HaloDepth>
struct MDWFStackGridSite {
    __device__ __host__ gSiteStack operator()(size_t i, size_t s) const {
        return GIndexer<LatLayout, HaloDepth>::getSiteStack(i, s);
    }
};

// (i, s) -> the i-th site of a site list at stack s (as MDWFCalcListSiteStack).
template<Layout LatLayout, size_t HaloDepth>
struct MDWFStackGridListSite {
    const unsigned *list;
    __device__ __host__ gSiteStack operator()(size_t i, size_t s) const {
        typedef GIndexer<LatLayout, HaloDepth> GInd;
        return GInd::getSiteStack(GInd::getSite(static_cast<size_t>(list[i])), s);
    }
};

#ifdef __GPUCC__
template<class Accessor, class Functor, class Calc>
__global__ void __launch_bounds__(MDWF_STACK_GRID_MAX_BLOCK)
    mdwfStackGridKernel(Accessor res, Functor op, Calc calc, const size_t n, const bool stackFastest) {
    const size_t b = stackFastest ? blockIdx.y : blockIdx.x;
    const size_t i = b * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    gSiteStack site = calc(i, stackFastest ? blockIdx.x : blockIdx.y);   // non-const: some functors take gSiteStack &
    res.setElement(site, op(site));
}

template<class Functor, class Calc>
__global__ void __launch_bounds__(MDWF_STACK_GRID_MAX_BLOCK)
    mdwfStackGridKernelNoReturn(Functor op, Calc calc, const size_t n, const bool stackFastest) {
    const size_t b = stackFastest ? blockIdx.y : blockIdx.x;
    const size_t i = b * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    gSiteStack site = calc(i, stackFastest ? blockIdx.x : blockIdx.y);
    op(site);
}
#endif

// out(site, s) = op(site, s) for the n sites of calc and all stacks s < stacks.
template<class Spinor, class Functor, class Calc>
void mdwfStackGridLaunch(Spinor &out, Functor op, Calc calc, size_t n, size_t stacks,
                         gpuStream_t stream = (gpuStream_t) nullptr) {
#ifdef __GPUCC__
    const unsigned block = mdwfStackGridBlock();
    const size_t nb = (n + block - 1) / block;
    const bool stackFastest = nb <= 65535;
    const dim3 grid = stackFastest ? dim3(static_cast<unsigned>(stacks), static_cast<unsigned>(nb))
                                   : dim3(static_cast<unsigned>(nb), static_cast<unsigned>(stacks));
    mdwfStackGridKernel<<<grid, block, 0, stream>>>(out.getAccessor(), op, calc, n, stackFastest);
    const gpuError_t err = gpuGetLastError();
    if (err) {
        GpuError("mdwfStackGridLaunch: kernel launch failed", err);
    }
#else
    (void)out; (void)op; (void)calc; (void)n; (void)stacks; (void)stream;
    throw std::runtime_error("mdwfStackGridLaunch: GPU build only");
#endif
}

// op(site, s), no output field (the functor writes itself).
template<class Functor, class Calc>
void mdwfStackGridLaunchNoReturn(Functor op, Calc calc, size_t n, size_t stacks,
                                 gpuStream_t stream = (gpuStream_t) nullptr) {
#ifdef __GPUCC__
    const unsigned block = mdwfStackGridBlock();
    const size_t nb = (n + block - 1) / block;
    const bool stackFastest = nb <= 65535;
    const dim3 grid = stackFastest ? dim3(static_cast<unsigned>(stacks), static_cast<unsigned>(nb))
                                   : dim3(static_cast<unsigned>(nb), static_cast<unsigned>(stacks));
    mdwfStackGridKernelNoReturn<<<grid, block, 0, stream>>>(op, calc, n, stackFastest);
    const gpuError_t err = gpuGetLastError();
    if (err) {
        GpuError("mdwfStackGridLaunchNoReturn: kernel launch failed", err);
    }
#else
    (void)op; (void)calc; (void)n; (void)stacks; (void)stream;
    throw std::runtime_error("mdwfStackGridLaunchNoReturn: GPU build only");
#endif
}
