//
// Created by Lukas Mazur on 03.06.20.
//

#pragma once
#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <utility>


#define gpuError_t                       cudaError_t
#define gpuDeviceProp                    cudaDeviceProp
#define gpuEvent_t                       cudaEvent_t
#define gpuIpcEventHandle_t              cudaIpcEventHandle_t
#define gpuIpcMemHandle_t                cudaIpcMemHandle_t
#define gpuStream_t                      cudaStream_t

#define gpuComputeModeDefault            cudaComputeModeDefault
#define gpuDeviceCanAccessPeer           cudaDeviceCanAccessPeer
#define gpuDeviceSynchronize             cudaDeviceSynchronize
#define gpuEventCreate                   cudaEventCreate
#define gpuEventCreateWithFlags          cudaEventCreateWithFlags
#define gpuEventDestroy                  cudaEventDestroy
#define gpuEventDisableTiming            cudaEventDisableTiming
#define gpuEventElapsedTime              cudaEventElapsedTime
#define gpuEventInterprocess             cudaEventInterprocess
#define gpuEventQuery                    cudaEventQuery
#define gpuEventRecord                   cudaEventRecord
#define gpuEventSynchronize              cudaEventSynchronize
#define gpuEventDefault                  cudaEventDefault
#define gpuFree                          cudaFree
#define gpuFreeHost                      cudaFreeHost
#define gpuGetDeviceCount                cudaGetDeviceCount
#define gpuGetDeviceProperties           cudaGetDeviceProperties
#define gpuGetErrorName                  cudaGetErrorName
#define gpuGetErrorString                cudaGetErrorString
#define gpuGetLastError                  cudaGetLastError
#define gpuIpcCloseMemHandle             cudaIpcCloseMemHandle
#define gpuIpcGetMemHandle               cudaIpcGetMemHandle
#define gpuIpcMemLazyEnablePeerAccess    cudaIpcMemLazyEnablePeerAccess
#define gpuIpcOpenEventHandle            cudaIpcOpenEventHandle
#define gpuIpcGetEventHandle             cudaIpcGetEventHandle
#define gpuIpcOpenMemHandle              cudaIpcOpenMemHandle
#define gpuMalloc                        cudaMalloc
#define gpuMallocHost                    cudaMallocHost
#define gpuMemGetInfo                    cudaMemGetInfo
#define gpuMemcpy                        cudaMemcpy
#define gpuMemcpyToSymbol                cudaMemcpyToSymbol
#define gpuMemcpyAsync                   cudaMemcpyAsync
#define gpuMemcpyDefault                 cudaMemcpyDefault
#define gpuMemcpyDeviceToDevice          cudaMemcpyDeviceToDevice
#define gpuMemcpyDeviceToHost            cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice            cudaMemcpyHostToDevice
#define gpuMemcpyHostToHost              cudaMemcpyHostToHost
#define gpuMemset                        cudaMemset
#define gpuProfilerStart                 cudaProfilerStart
#define gpuProfilerStop                  cudaProfilerStop
#define gpuSetDevice                     cudaSetDevice
#define gpuStreamCreate                  cudaStreamCreate
#define gpuStreamDestroy                 cudaStreamDestroy
#define gpuStreamSynchronize             cudaStreamSynchronize
#define gpuStreamWaitEvent               cudaStreamWaitEvent
#define gpuSuccess                       cudaSuccess


#ifdef __CUDA_ARCH__
#define __GPU_ARCH__                     __CUDA_ARCH__
#endif

#ifdef __CUDACC__
#define __GPUCC__                        __CUDACC__
#endif


#elif defined USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_fp16.h>
#include <utility>


#define gpuError_t                       hipError_t
#define gpuDeviceProp                    hipDeviceProp_t
#define gpuEvent_t                       hipEvent_t
#define gpuIpcEventHandle_t              hipIpcEventHandle_t
#define gpuIpcMemHandle_t                hipIpcMemHandle_t
#define gpuStream_t                      hipStream_t

#define gpuComputeModeDefault            hipComputeModeDefault
#define gpuDeviceCanAccessPeer           hipDeviceCanAccessPeer
#define gpuDeviceSynchronize             hipDeviceSynchronize
#define gpuEventCreate                   hipEventCreate
#define gpuEventCreateWithFlags          hipEventCreateWithFlags
#define gpuEventDestroy                  hipEventDestroy
#define gpuEventDisableTiming            hipEventDisableTiming
#define gpuEventElapsedTime              hipEventElapsedTime
#define gpuEventInterprocess             hipEventInterprocess
#define gpuEventQuery                    hipEventQuery
#define gpuEventRecord                   hipEventRecord
#define gpuEventSynchronize              hipEventSynchronize
#define gpuEventDefault                  hipEventDefault
#define gpuFree                          hipFree
#define gpuFreeHost                      hipHostFree
#define gpuGetDeviceCount                hipGetDeviceCount
#define gpuGetDeviceProperties           hipGetDeviceProperties
#define gpuGetErrorName                  hipGetErrorName
#define gpuGetErrorString                hipGetErrorString
#define gpuGetLastError                  hipGetLastError
#define gpuIpcCloseMemHandle             hipIpcCloseMemHandle
#define gpuIpcGetMemHandle               hipIpcGetMemHandle
#define gpuIpcMemLazyEnablePeerAccess    hipIpcMemLazyEnablePeerAccess
#define gpuIpcOpenEventHandle            hipDummyFunction2//cudaIpcOpenEventHandle
#define gpuIpcGetEventHandle             hipDummyFunction1//cudaIpcGetEventHandle
#define gpuIpcOpenMemHandle              hipIpcOpenMemHandle
#define gpuMalloc                        hipMalloc
#define gpuMallocHost                    hipHostMalloc
#define gpuMemGetInfo                    hipMemGetInfo
#define gpuMemcpy                        hipMemcpy
#define gpuMemcpyToSymbol                hipMemcpyToSymbol
#define gpuMemcpyAsync                   hipMemcpyAsync
#define gpuMemcpyDefault                 hipMemcpyDefault
#define gpuMemcpyDeviceToDevice          hipMemcpyDeviceToDevice
#define gpuMemcpyDeviceToHost            hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice            hipMemcpyHostToDevice
#define gpuMemcpyHostToHost              hipMemcpyHostToHost
#define gpuMemset                        hipMemset
#define gpuProfilerStart                 hipProfilerStart
#define gpuProfilerStop                  hipProfilerStop
#define gpuSetDevice                     hipSetDevice
#define gpuStreamCreate                  hipStreamCreate
#define gpuStreamDestroy                 hipStreamDestroy
#define gpuStreamSynchronize             hipStreamSynchronize
#define gpuStreamWaitEvent               hipStreamWaitEvent
#define gpuSuccess                       hipSuccess

// As soon as HIP supports these two functions below, we need to replace them!
//
[[nodiscard]] inline gpuError_t hipDummyFunction1(__attribute__((unused)) gpuIpcEventHandle_t* handle, __attribute__((unused)) hipEvent_t event) {
    __attribute__((unused)) hipError_t temp = hipErrorUnknown;
    return temp;
}

[[nodiscard]] inline gpuError_t hipDummyFunction2(__attribute__((unused)) gpuEvent_t* event, __attribute__((unused)) gpuIpcEventHandle_t handle) {
    __attribute__((unused)) hipError_t temp = hipErrorUnknown;
    return temp;
}

#ifdef __HIP_DEVICE_COMPILE__
#define __GPU_ARCH__                     __HIP_DEVICE_COMPILE__
#endif

#ifdef __HIPCC__
#define __GPUCC__                        __HIPCC__
#endif

#endif

