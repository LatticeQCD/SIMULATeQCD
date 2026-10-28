/*
 * LatticeContainer.cpp
 *
 */
#include "latticeContainer.h"

#ifdef USE_CUDA
#include <cub/cub.cuh>
#define gpucub cub
#elif defined USE_HIP
#include <hipcub/hipcub.hpp>
#define gpucub hipcub
#endif

#ifdef USE_CPU
#include <vector>

//! Sums fixed-size chunks in parallel and adds the partial sums in chunk order, so the
//! result does not depend on the number of OpenMP threads.
template<class floatT>
static floatT cpuSum(const floatT *values, size_t size) {
    constexpr size_t chunk = 4096;
    const size_t nChunks = (size + chunk - 1) / chunk;
    std::vector<floatT> partial(nChunks, floatT(0.0));
#pragma omp parallel for schedule(static)
    for (size_t c = 0; c < nChunks; ++c) {
        const size_t end = (c + 1) * chunk < size ? (c + 1) * chunk : size;
        floatT sum = floatT(0.0);
        for (size_t i = c * chunk; i < end; ++i) sum += values[i];
        partial[c] = sum;
    }
    floatT sum = floatT(0.0);
    for (size_t c = 0; c < nChunks; ++c) sum += partial[c];
    return sum;
}

//! Maximum that propagates NaNs, so downstream finite-value checks can reject them.
template<class floatT>
static floatT cpuMax(const floatT *values, size_t size) {
    floatT result = values[0];
#pragma omp parallel
    {
        floatT local = values[0];
#pragma omp for schedule(static) nowait
        for (size_t i = 1; i < size; ++i) {
            if (values[i] != values[i] || values[i] > local) local = values[i];
        }
#pragma omp critical
        if (local != local || (result == result && local > result)) result = local;
    }
    return result;
}
#endif


template<class floatT>
gpuError_t CubReduce(void *helpArr, size_t *temp_storage_bytes, floatT *Arr, floatT *out, size_t size, gpuStream_t stream) {

#ifdef USE_CPU
    (void)stream;
    if (!helpArr) { *temp_storage_bytes = 1; return gpuSuccess; }
    cpuProfile::Scope profile("CubReduce (sum)");
    *out = cpuSum(Arr, size);
    return gpuSuccess;
#else
    return gpucub::DeviceReduce::Sum(helpArr, *temp_storage_bytes, static_cast<floatT *>(Arr), out,
                                  size, stream);
#endif
}

template<class floatT>
gpuError_t CubReduceMax(void *helpArr, size_t *temp_storage_bytes, void *Arr, floatT *out, size_t size) {

#ifdef USE_CPU
    if (!helpArr) { *temp_storage_bytes = 1; return gpuSuccess; }
    if (!size) return cpuInvalidValue;
    cpuProfile::Scope profile("CubReduceMax");
    *out = cpuMax(static_cast<floatT*>(Arr), size);
    return gpuSuccess;
#else
    return gpucub::DeviceReduce::Max(helpArr, *temp_storage_bytes, static_cast<floatT *>(Arr), out,
                                  size);
#endif
}

template<class floatT>
gpuError_t
CubReduceStacked(void *helpArr, size_t *temp_storage_bytes, void *Arr, void *out, int Nt, void *StackOffsets) {

#ifdef USE_CPU
    if (!helpArr) { *temp_storage_bytes = 1; return gpuSuccess; }
    cpuProfile::Scope profile("CubReduceStacked");
    const auto* offsets = static_cast<size_t*>(StackOffsets);
    const auto* values = static_cast<floatT*>(Arr);
    auto* result = static_cast<floatT*>(out);
    for (int stack=0; stack<Nt; ++stack)
        result[stack] = cpuSum(values + offsets[stack], offsets[stack+1] - offsets[stack]);
    return gpuSuccess;
#else
    return gpucub::DeviceSegmentedReduce::Sum(helpArr, *temp_storage_bytes, static_cast<floatT *>(Arr),
                                           static_cast<floatT *>(out),
                                           Nt, static_cast<size_t*>(StackOffsets), static_cast<size_t*>(StackOffsets) + 1);
#endif
}

#define CLASS_INIT(floatT) \
template gpuError_t CubReduce<floatT>(void * helpArr, size_t *temp_storage_bytes, floatT* Arr, floatT* out, size_t size, gpuStream_t stream); \
template gpuError_t CubReduceStacked<floatT>(void * helpArr, size_t *temp_storage_bytes, void * Arr, void* out, int Nt, void *StackOffsets); \


CLASS_INIT(float)

CLASS_INIT(double)

CLASS_INIT(int)

CLASS_INIT(COMPLEX(float))

CLASS_INIT(COMPLEX(double))

CLASS_INIT(Matrix4x4Sym<double>)

CLASS_INIT(Matrix4x4Sym<float>)

CLASS_INIT(SU3<double>)

CLASS_INIT(SU3<float>)

#define CLASS_INITMAX(floatT) \
template gpuError_t CubReduceMax<floatT>(void * helpArr, size_t *temp_storage_bytes, void * Arr, floatT* out, size_t size);

CLASS_INITMAX(float)

CLASS_INITMAX(double)

CLASS_INITMAX(int)

