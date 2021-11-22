#include "LatticeContainer.h"
#define CUB_NS_QUALIFIER

#include "../../cub/cub/cub.cuh"


template<class floatT>
gpuError_t CubReduce(void *helpArr, size_t *temp_storage_bytes, floatT *Arr, floatT *out, size_t size) {

    return cub::DeviceReduce::Sum(helpArr, *temp_storage_bytes, static_cast<floatT *>(Arr), out,
                                  size);
}

template<class floatT>
gpuError_t CubReduceMax(void *helpArr, size_t *temp_storage_bytes, void *Arr, floatT *out, size_t size) {

    return cub::DeviceReduce::Max(helpArr, *temp_storage_bytes, static_cast<floatT *>(Arr), out,
                                  size);
}

template<class floatT>
gpuError_t
CubReduceStacked(void *helpArr, size_t *temp_storage_bytes, void *Arr, void *out, int Nt, void *StackOffsets) {

    return cub::DeviceSegmentedReduce::Sum(helpArr, *temp_storage_bytes, static_cast<floatT *>(Arr),
                                           static_cast<floatT *>(out),
                                           Nt, static_cast<size_t*>(StackOffsets), static_cast<size_t*>(StackOffsets) + 1);
}

#define CLASS_INIT(floatT) \
template gpuError_t CubReduce<floatT>(void * helpArr, size_t *temp_storage_bytes, floatT* Arr, floatT* out, size_t size); \
template gpuError_t CubReduceStacked<floatT>(void * helpArr, size_t *temp_storage_bytes, void * Arr, void* out, int Nt, void *StackOffsets); \


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

