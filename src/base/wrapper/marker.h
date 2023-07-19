
#pragma once

#include <string>
#include <string_view>
#include "gpu_wrapper.h"

inline void markerBegin(std::string_view marker_name, std::string_view group_name){
#ifdef USE_MARKER
    #ifdef USE_CUDA
        nvtxRangePush((std::string(group_name) + ": " + std::string(marker_name)).c_str());
    #elif defined USE_HIP
	auto ret = roctxRangePush((std::string(group_name) + ": " + std::string(marker_name)).c_str());
    std::cout << "roctxRangePush (" << ret << ")" <<": group_name " <<std::string(group_name) << ", marker_name " << std::string(marker_name) << std::endl;
    #endif
#endif

}

inline void markerEnd(){
#ifdef USE_MARKER
    #ifdef USE_CUDA
        nvtxRangePop();
    #elif defined USE_HIP
        auto ret = roctxRangePop();
        std::cout << "roctxRangePop (" << ret << ")" << std::endl;
    #endif
#endif
}
