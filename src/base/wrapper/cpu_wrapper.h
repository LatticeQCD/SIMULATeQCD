#pragma once

// Synchronous host implementation of the runtime operations used by HISQ.
// onDevice=true remains the compute allocation domain, but contains host RAM.
// Do not define __GPUCC__ or __GPU_ARCH__: CPU code must use host indexers.
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>
#include "cpuProfile.h"

#define __host__
#define __device__
#define __constant__
#define __forceinline__ inline

struct uint3 { unsigned x, y, z; };
struct uint4 { unsigned x, y, z, w; };
inline constexpr uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) {
    return {x,y,z,w};
}
struct dim3 {
    unsigned x, y, z;
    constexpr dim3(unsigned a=1, unsigned b=1, unsigned c=1): x(a), y(b), z(c) {}
    constexpr dim3(uint3 a): x(a.x), y(a.y), z(a.z) {}
    constexpr operator uint3() const { return {x,y,z}; }
};
template<class T> struct CpuVec2 {
    T x, y;
    template<class U> explicit operator CpuVec2<U>() const {
        return {static_cast<U>(x), static_cast<U>(y)};
    }
};
using float2 = CpuVec2<float>;
using double2 = CpuVec2<double>;
struct float3 { float x, y, z; };

// The shared headers mention half precision even when it is not instantiated.
// No half-precision executable or mixed-half solver is enabled in this backend.
#if !defined(__FLT16_MANT_DIG__)
#error "CPU prototype requires a compiler with _Float16 (e.g. GCC 12 on x86-64)"
#endif
using __half = _Float16;
using __half2 = CpuVec2<__half>;
inline __half __float2half(float x) { return static_cast<__half>(x); }
inline float __half2float(__half x) { return static_cast<float>(x); }

using gpuError_t = int;
inline constexpr gpuError_t gpuSuccess = 0;
inline constexpr gpuError_t cpuInvalidValue = 1;
inline constexpr gpuError_t cpuOutOfMemory = 2;
inline constexpr gpuError_t cpuUnsupported = 3;
enum CpuMemcpyKind { gpuMemcpyDefault, gpuMemcpyHostToHost,
    gpuMemcpyHostToDevice, gpuMemcpyDeviceToHost, gpuMemcpyDeviceToDevice };
struct CpuStream {};
using gpuStream_t = CpuStream*;
struct CpuEvent { std::chrono::steady_clock::time_point time; };
using gpuEvent_t = CpuEvent*;
inline constexpr gpuEvent_t gpuEventDefault = nullptr;
inline constexpr unsigned gpuEventDisableTiming = 1, gpuEventInterprocess = 2;
inline constexpr unsigned gpuIpcMemLazyEnablePeerAccess = 1;
struct gpuIpcMemHandle_t { unsigned char unused[64]; };
struct gpuIpcEventHandle_t { unsigned char unused[64]; };
inline constexpr int gpuComputeModeDefault = 0;
struct gpuDeviceProp {
    int computeMode=0, major=0, minor=0, pciBusID=0, unifiedAddressing=0;
    char name[32] = "CPU";
};

inline const char* gpuGetErrorName(gpuError_t e) {
    switch(e) {
        case gpuSuccess: return "success";
        case cpuInvalidValue: return "invalid CPU runtime argument";
        case cpuOutOfMemory: return "host allocation failed";
        default: return "operation unsupported by CPU backend";
    }
}
inline const char* gpuGetErrorString(gpuError_t e) { return gpuGetErrorName(e); }
inline gpuError_t gpuGetLastError() { return gpuSuccess; }
inline gpuError_t gpuDeviceSynchronize() { return gpuSuccess; }
// 64-byte alignment: one AVX-512 register / cache line. One extra line of slack lets
// vector loads that cover a few unused elements past the last one stay inside the block.
inline constexpr size_t cpuAllocAlignment = 64;
inline gpuError_t gpuMalloc(void** p, size_t n) {
    if (!p) return cpuInvalidValue;
    const size_t rounded = (n + cpuAllocAlignment - 1) / cpuAllocAlignment * cpuAllocAlignment;
    *p = std::aligned_alloc(cpuAllocAlignment, rounded + cpuAllocAlignment);
    return *p ? gpuSuccess : cpuOutOfMemory;
}
inline gpuError_t gpuMallocHost(void** p, size_t n) { return gpuMalloc(p,n); }
inline gpuError_t gpuFree(void* p) { std::free(p); return gpuSuccess; }
inline gpuError_t gpuFreeHost(void* p) { return gpuFree(p); }
inline gpuError_t gpuMemcpy(void* dst, const void* src, size_t n, CpuMemcpyKind) {
    if (!n) return gpuSuccess;
    cpuProfile::Scope profile("gpuMemcpy");
    if (!dst || !src) return cpuInvalidValue;
    auto* d = static_cast<unsigned char*>(dst);
    const auto* s = static_cast<const unsigned char*>(src);
    // Large non-overlapping copies (field copies) are split over the OpenMP threads.
    constexpr size_t chunk = size_t(1) << 18;
    if (n >= 4 * chunk && (d + n <= s || s + n <= d)) {
        const size_t nChunks = (n + chunk - 1) / chunk;
#pragma omp parallel for schedule(static)
        for (size_t c = 0; c < nChunks; ++c) {
            const size_t begin = c * chunk;
            std::memcpy(d + begin, s + begin, begin + chunk < n ? chunk : n - begin);
        }
        return gpuSuccess;
    }
    std::memmove(dst,src,n);
    return gpuSuccess;
}
inline gpuError_t gpuMemcpyAsync(void* d, const void* s, size_t n,
                                CpuMemcpyKind k, gpuStream_t = nullptr) {
    return gpuMemcpy(d,s,n,k);
}
inline gpuError_t gpuMemset(void* d, int value, size_t n) {
    if (!n) return gpuSuccess;
    if (!d) return cpuInvalidValue;
    std::memset(d,value,n);
    return gpuSuccess;
}
template<class Symbol>
inline gpuError_t gpuMemcpyToSymbol(Symbol& dst, const void* src, size_t n,
                                    size_t offset, CpuMemcpyKind k) {
    if (offset > sizeof(Symbol) || n > sizeof(Symbol)-offset) return cpuInvalidValue;
    return gpuMemcpy(reinterpret_cast<unsigned char*>(&dst)+offset,src,n,k);
}
inline gpuError_t gpuStreamCreate(gpuStream_t* s) {
    *s = new(std::nothrow) CpuStream;
    return *s ? gpuSuccess : cpuOutOfMemory;
}
inline gpuError_t gpuStreamDestroy(gpuStream_t s) { delete s; return gpuSuccess; }
inline gpuError_t gpuStreamSynchronize(gpuStream_t) { return gpuSuccess; }
inline gpuError_t gpuEventCreateWithFlags(gpuEvent_t* e, unsigned flags) {
    if (flags & gpuEventInterprocess) return cpuUnsupported;
    *e = new(std::nothrow) CpuEvent{};
    return *e ? gpuSuccess : cpuOutOfMemory;
}
inline gpuError_t gpuEventCreate(gpuEvent_t* e) { return gpuEventCreateWithFlags(e,0); }
inline gpuError_t gpuEventDestroy(gpuEvent_t e) { delete e; return gpuSuccess; }
inline gpuError_t gpuEventRecord(gpuEvent_t e, gpuStream_t = nullptr) {
    if (!e) return cpuInvalidValue;
    e->time = std::chrono::steady_clock::now();
    return gpuSuccess;
}
inline gpuError_t gpuEventSynchronize(gpuEvent_t) { return gpuSuccess; }
inline gpuError_t gpuEventQuery(gpuEvent_t) { return gpuSuccess; }
inline gpuError_t gpuStreamWaitEvent(gpuStream_t, gpuEvent_t, unsigned) { return gpuSuccess; }
inline gpuError_t gpuEventElapsedTime(float* ms, gpuEvent_t a, gpuEvent_t b) {
    if (!ms || !a || !b) return cpuInvalidValue;
    *ms = std::chrono::duration<float,std::milli>(b->time-a->time).count();
    return gpuSuccess;
}
// GPU discovery and IPC must never silently succeed on CPU builds.
inline gpuError_t gpuGetDeviceCount(int* n) { *n=0; return gpuSuccess; }
inline gpuError_t gpuGetDeviceProperties(gpuDeviceProp*, int) { return cpuUnsupported; }
inline gpuError_t gpuSetDevice(int) { return cpuUnsupported; }
inline gpuError_t gpuDeviceCanAccessPeer(int* yes, int, int) { *yes=0; return gpuSuccess; }
inline gpuError_t gpuIpcGetMemHandle(gpuIpcMemHandle_t*, void*) { return cpuUnsupported; }
inline gpuError_t gpuIpcGetEventHandle(gpuIpcEventHandle_t*, gpuEvent_t) { return cpuUnsupported; }
inline gpuError_t gpuIpcOpenMemHandle(void**, gpuIpcMemHandle_t, unsigned) { return cpuUnsupported; }
inline gpuError_t gpuIpcOpenEventHandle(gpuEvent_t*, gpuIpcEventHandle_t) { return cpuUnsupported; }
inline gpuError_t gpuIpcCloseMemHandle(void*) { return cpuUnsupported; }
