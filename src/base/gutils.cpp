/*
 * gutils.cpp
 *
 */

#include "gutils.h"

GpuError::GpuError(gpuError_t err) : gpuErr(err) {
  throw std::runtime_error(stdLogger.fatal("A GPU error occured: ", getErrorMessage()));
}

GpuError::GpuError(const char *warn, gpuError_t err) : gpuErr(err) {
  throw std::runtime_error(stdLogger.fatal("A GPU error occured: ", warn, ": ", getErrorMessage()));
}

gpuError_t GpuError::getError() { return gpuErr; }

const std::string GpuError::getErrorMessage() {
    std::string err_name = gpuGetErrorName(getError());
    std::string err_msg = gpuGetErrorString(getError());
    std::string err = err_msg + " ( " + err_name + " )";
    return err;
}
