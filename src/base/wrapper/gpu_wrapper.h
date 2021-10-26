//
// Created by Lukas Mazur on 03.06.20.
//

#ifndef GPU_WRAPPER_H
#define GPU_WRAPPER_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
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
#define gpuEventDestroy                  cudaEventDestroy
#define gpuEventDisableTiming            cudaEventDisableTiming
#define gpuEventElapsedTime              cudaEventElapsedTime
#define gpuEventInterprocess             cudaEventInterprocess
#define gpuEventQuery                    cudaEventQuery
#define gpuEventRecord                   cudaEventRecord
#define gpuEventSynchronize              cudaEventSynchronize
#define gpuFree                          cudaFree
#define gpuFreeHost                      cudaFreeHost
#define gpuGetDeviceCount                cudaGetDeviceCount
#define gpuGetDeviceProperties           cudaGetDeviceProperties
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



#endif //GPU_WRAPPER_H
