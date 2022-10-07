/* 
 * gutils.h                                                               
 * 
 * Utility functions usable in GPUs. 
 * 
 */
#ifndef UTIL_H
#define UTIL_H

#include "../define.h"
#include "wrapper/gpu_wrapper.h"
#include <iostream>
#include <math.h>

/**
 * Utility function to calculate quotient and remainder of
 * nominator / denominator.
 */
__host__ __device__ void inline divmod(int nominator, int denominator,
        int &quotient, int &remainder) {
    quotient  = nominator / denominator;
    remainder = nominator - (quotient * denominator);
}
__host__ __device__ void inline divmod(size_t nominator, size_t denominator,
        size_t &quotient, size_t &remainder) {
    quotient  = nominator / denominator;
    remainder = nominator - (quotient * denominator);
}

__host__ void inline compute_dim3(dim3 &blockDim, dim3 &gridDim,
        const size_t elems, const size_t blockSize) {
    blockDim = blockSize;
    gridDim  = static_cast<int>(ceilf(static_cast<float>(elems) / static_cast<float>(blockDim.x)));
}

/**
 * Utility class to report errors in GPU code.
 */
class GpuError {
public:
    explicit GpuError(gpuError_t err);
  
    GpuError(const char *warn, gpuError_t err);
  
    gpuError_t getError();
  
    const std::string getErrorMessage();

private:
    gpuError_t gpuErr;
};

/**
 * Utility method for speedy testing of whether a number is odd
 */
__device__ __host__ inline bool isOdd(int cand) { return (cand & 0x1); }




#endif /* UTIL_H */
