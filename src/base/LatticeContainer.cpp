/* 
 * LatticeContainer.cpp                                                               
 * 
 */
#include "LatticeContainer.h"

#ifdef USE_CUDA
#include <cub/cub.cuh>
#define gpucub cub
#elif defined USE_HIP
#include <hipcub/hipcub.hpp>
#define gpucub hipcub
#endif

#ifndef USE_CPU_ONLY
template<class floatT>
GPUERROR_T CubReduce(void *helpArr, size_t *temp_storage_bytes, floatT *Arr, floatT *out, size_t size) {

    return gpucub::DeviceReduce::Sum(helpArr, *temp_storage_bytes, static_cast<floatT *>(Arr), out,
                                  size);
}

template<class floatT>
GPUERROR_T CubReduceMax(void *helpArr, size_t *temp_storage_bytes, void *Arr, floatT *out, size_t size) {

    return gpucub::DeviceReduce::Max(helpArr, *temp_storage_bytes, static_cast<floatT *>(Arr), out,
                                  size);
}

template<class floatT>
GPUERROR_T
CubReduceStacked(void *helpArr, size_t *temp_storage_bytes, void *Arr, void *out, int Nt, void *StackOffsets) {

    return gpucub::DeviceSegmentedReduce::Sum(helpArr, *temp_storage_bytes, static_cast<floatT *>(Arr),
                                           static_cast<floatT *>(out),
                                           Nt, static_cast<size_t*>(StackOffsets), static_cast<size_t*>(StackOffsets) + 1);
}

#define CLASS_INIT(floatT) \
template GPUERROR_T CubReduce<floatT>(void * helpArr, size_t *temp_storage_bytes, floatT* Arr, floatT* out, size_t size); \
template GPUERROR_T CubReduceStacked<floatT>(void * helpArr, size_t *temp_storage_bytes, void * Arr, void* out, int Nt, void *StackOffsets); \


CLASS_INIT(float)

CLASS_INIT(double)

CLASS_INIT(int)

CLASS_INIT(GCOMPLEX(float))

CLASS_INIT(GCOMPLEX(double))

CLASS_INIT(Matrix4x4Sym<double>)

CLASS_INIT(Matrix4x4Sym<float>)

CLASS_INIT(GSU3<double>)

CLASS_INIT(GSU3<float>)

#define CLASS_INITMAX(floatT) \
template gpuError_t CubReduceMax<floatT>(void * helpArr, size_t *temp_storage_bytes, void * Arr, floatT* out, size_t size);

CLASS_INITMAX(float)

CLASS_INITMAX(double)

CLASS_INITMAX(int)

#endif
