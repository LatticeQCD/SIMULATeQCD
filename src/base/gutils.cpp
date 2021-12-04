
#include "gutils.h"

GpuError::GpuError(gpuError_t err) : gpuErr(err) {
  throw std::runtime_error(stdLogger.fatal("A CUDA error occured: ", getErrorMessage()));
}

GpuError::GpuError(const char *warn, gpuError_t err) : gpuErr(err) {
  throw std::runtime_error(stdLogger.fatal("A CUDA error occured: ", warn, ": ", getErrorMessage()));
}

gpuError_t GpuError::getError() { return gpuErr; }

const char *GpuError::getErrorMessage() {
  return gpuGetErrorString(getError());
}
