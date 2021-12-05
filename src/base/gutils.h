#ifndef UTIL_H
#define UTIL_H

#include "../define.h"
#include "wrapper/gpu_wrapper.h"
#include <chrono>
#include <ctime>
#include <iostream>
#include <math.h>

/**
 * Utility function to calculate quotient and remainder of
 * nominator / denominator.
 */
__host__ __device__ void inline divmod(int nominator, int denominator,
                                       int &quotient, int &remainder) {
  quotient = nominator / denominator;
  remainder = nominator - (quotient * denominator);
}
__host__ __device__ void inline divmod(size_t nominator, size_t denominator,
                                       size_t &quotient, size_t &remainder) {
  quotient = nominator / denominator;
  remainder = nominator - (quotient * denominator);
}

__host__ void inline compute_dim3(dim3 &blockDim, dim3 &gridDim,
                                  const size_t elems, const size_t blockSize) {
  blockDim = blockSize;
  gridDim = static_cast<int>(
      ceilf(static_cast<float>(elems) / static_cast<float>(blockDim.x)));
}

/**
 * Utility class to report errors in CUDA code.
 */
class GpuError {
public:
  explicit GpuError(gpuError_t err);

  GpuError(const char *warn, gpuError_t err);

  gpuError_t getError();

  const char *getErrorMessage();

private:
  gpuError_t gpuErr;
};

/**
 * Utility method for speedy testing of whether a number is odd
 */
__device__ __host__ inline bool isOdd(int cand) { return (cand & 0x1); }

class GpuStopWatch {
    float _elapsed = 0;

#ifdef __CUDACC__
    gpuEvent_t _start_time, _stop_time;

    public:
    GpuStopWatch() {
        gpuEventCreate(&_start_time);
        gpuEventCreate(&_stop_time);
    }

    ~GpuStopWatch() {
        gpuEventDestroy(_start_time);
        gpuEventDestroy(_stop_time);
    }

    void start() { gpuEventRecord(_start_time, 0); }

    double stop() {
        gpuEventRecord(_stop_time, 0);
        gpuEventSynchronize(_stop_time);
        float time;
        gpuEventElapsedTime(&time, _start_time, _stop_time);
        _elapsed += time;
        return _elapsed;
    }
#else
    public:
    std::chrono::high_resolution_clock::time_point _start_time, _stop_time;

    void start() { _start_time = std::chrono::high_resolution_clock::now(); }

    double stop() {
        float time;
        _stop_time = std::chrono::high_resolution_clock::now();
        time = std::chrono::duration_cast<std::chrono::microseconds>(_stop_time -
                _start_time)
            .count();
        _elapsed += time;
        return _elapsed;
    }
#endif

    void reset() { _elapsed = 0; }

    void print(std::string text) {
        rootLogger.info("Time for " + text + " " ,  _elapsed ,  "ms");
    }

    inline friend std::ostream &operator<<(std::ostream &stream,
            const GpuStopWatch &rhs);
};

std::ostream &operator<<(std::ostream &stream, const GpuStopWatch &rhs) {
    stream << "Time = " << rhs._elapsed << "ms";
    return stream;
}

#endif /* UTIL_H */

