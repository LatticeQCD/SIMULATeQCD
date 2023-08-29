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
gpuError_t CubReduce(void *helpArr, size_t *temp_storage_bytes, floatT *Arr, floatT *out, size_t size) {

    return gpucub::DeviceReduce::Sum(helpArr, *temp_storage_bytes, static_cast<floatT *>(Arr), out,
                                  size);
}

template<class floatT>
gpuError_t CubReduceMax(void *helpArr, size_t *temp_storage_bytes, void *Arr, floatT *out, size_t size) {

    return gpucub::DeviceReduce::Max(helpArr, *temp_storage_bytes, static_cast<floatT *>(Arr), out,
                                  size);
}

template<class floatT>
gpuError_t
CubReduceStacked(void *helpArr, size_t *temp_storage_bytes, void *Arr, void *out, int Nt, void *StackOffsets) {

    return gpucub::DeviceSegmentedReduce::Sum(helpArr, *temp_storage_bytes, static_cast<floatT *>(Arr),
                                           static_cast<floatT *>(out),
                                           Nt, static_cast<size_t*>(StackOffsets), static_cast<size_t*>(StackOffsets) + 1);
}

#define CLASS_INIT(floatT) \
template gpuError_t CubReduce<floatT>(void * helpArr, size_t *temp_storage_bytes, floatT* Arr, floatT* out, size_t size); \
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

