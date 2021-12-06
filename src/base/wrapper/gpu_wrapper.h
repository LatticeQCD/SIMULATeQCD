//
// Created by Lukas Mazur on 03.06.20.
//

#ifndef GPU_WRAPPER_H
#define GPU_WRAPPER_H
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
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
#define gpuFree                          hipFree
#define gpuFreeHost                      hipHostFree
#define gpuGetDeviceCount                hipGetDeviceCount
#define gpuGetDeviceProperties           hipGetDeviceProperties
#define gpuGetErrorString                hipGetErrorString
#define gpuGetLastError                  hipGetLastError
#define gpuIpcCloseMemHandle             hipIpcCloseMemHandle
#define gpuIpcGetMemHandle               hipIpcGetMemHandle
#define gpuIpcMemLazyEnablePeerAccess    hipIpcMemLazyEnablePeerAccess
#define gpuIpcOpenEventHandle            cudaDummyFunction2//cudaIpcOpenEventHandle
#define gpuIpcGetEventHandle             cudaDummyFunction1//cudaIpcGetEventHandle
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

inline gpuError_t cudaDummyFunction1(gpuIpcEventHandle_t* handle, hipEvent_t event) {
    hipError_t temp = hipErrorUnknown;
    return temp;
}

inline gpuError_t cudaDummyFunction2(gpuEvent_t* event, gpuIpcEventHandle_t handle) {
    hipError_t temp = hipErrorUnknown;
    return temp;
}
#endif //GPU_WRAPPER_H
