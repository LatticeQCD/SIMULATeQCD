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


template<class floatT>
gpuError_t CubReduce(void *helpArr, size_t *temp_storage_bytes, floatT *Arr, floatT *out, size_t size, gpuStream_t stream) {

#ifdef USE_CPU
    (void)stream;
    if (!helpArr) { *temp_storage_bytes = 1; return gpuSuccess; }
    floatT sum = floatT(0.0);
    for (size_t i=0; i<size; ++i) sum += Arr[i];
    *out = sum;
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
    auto* values = static_cast<floatT*>(Arr);
    floatT result = values[0];
    for (size_t i=1; i<size; ++i) {
        // Preserve NaNs so downstream finite-value checks can reject them.
        if (values[i] != values[i] || values[i] > result) result = values[i];
    }
    *out = result;
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
    const auto* offsets = static_cast<size_t*>(StackOffsets);
    const auto* values = static_cast<floatT*>(Arr);
    auto* result = static_cast<floatT*>(out);
    for (int stack=0; stack<Nt; ++stack) {
        result[stack] = floatT(0.0);
        for (size_t i=offsets[stack]; i<offsets[stack+1]; ++i)
            result[stack] += values[i];
    }
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

