
#pragma once

#include <string>
#include <string_view>
#include "gpu_wrapper.h"
#include "../stopWatch.h"
#include "../utilities/parseObjectName.h"

inline void markerBegin(std::string_view marker_name, std::string_view group_name){
#ifdef USE_MARKER
    #ifdef USE_CUDA
        nvtxRangePush((std::string(group_name) + ": " + std::string(marker_name)).c_str());
    #elif defined USE_HIP
	    roctxRangePush((std::string(group_name) + ": " + std::string(marker_name)).c_str());
    #endif
#endif

}

inline void markerEnd(){
#ifdef USE_MARKER
    #ifdef USE_CUDA
        nvtxRangePop();
    #elif defined USE_HIP
        roctxRangePop();
    #endif
#endif
}

template<bool onDevice>
class AutoTimer {
    StopWatch<onDevice> internalTimer;
    std::string_view marker_name;
    std::string_view group_name;
    public:
    AutoTimer<onDevice>(std::string_view mn, std::string_view gn) : marker_name(mn), group_name(gn) {
        if (rootLogger.getVerbosity() == TRACE) {
        internalTimer.start();
        markerBegin(marker_name, group_name);
        }
    }

    ~AutoTimer<onDevice>(){
        if (rootLogger.getVerbosity() == TRACE) {
        markerEnd();
        internalTimer.stop();
        rootLogger.trace(std::string(group_name), ": ", std::string(marker_name), "took ", internalTimer);
        }
    }
};