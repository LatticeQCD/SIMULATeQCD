
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

template<bool onDevice, LogLevel level>
class AutoTimer {
    StopWatch<onDevice> internalTimer;
    std::string_view marker_name;
    std::string_view group_name;
    bool timer_started = false;
    bool marker_started = false;
    public:
    AutoTimer<onDevice,level>(std::string_view mn, std::string_view gn) : marker_name(mn), group_name(gn) {
        if (rootLogger.getVerbosity() == level) {
            timer_started=true;
            marker_started=true;
            internalTimer.start();
            markerBegin(marker_name, group_name);
        }
    }

    ~AutoTimer<onDevice,level>(){
        if (marker_started) {
            markerEnd();
        }
        if (timer_started) {
            internalTimer.stop();
            rootLogger.template message<level>(std::string(group_name), ": ", std::string(marker_name), "took ", internalTimer);
        }
    }
};