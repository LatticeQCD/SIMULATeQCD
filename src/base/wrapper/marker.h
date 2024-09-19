
#pragma once

#include <string>
#include <string_view>
#include "gpu_wrapper.h"

inline void markerBegin(std::string_view marker_name, std::string_view group_name){
#ifdef USE_MARKER
    #ifdef USE_CUDA
        nvtxRangePush((std::string(group_name) + ": " + std::string(marker_name)).c_str());
    #elif USE_HIP
	    roctxRangePush((std::string(group_name) + ": " + std::string(marker_name)).c_str());
    #endif
#endif

}

inline void markerEnd(){
#ifdef USE_MARKER
    #ifdef USE_CUDA
        nvtxRangePop();
    #elif USE_HIP
        roctxRangePop();
    #endif
#endif
}
