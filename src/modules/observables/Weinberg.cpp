#include "Weinberg.h"

/*WBDensKernel(gaugeAccessor<floatT> gAcc, floatT * WBDensArray)
	compute the topological charge density q(x) given by
		q(x)= 1/4pi^2 * tr(  F_(3,0) * F_(1,2)
					  	   + F_(3,1) * F_(2,0)
					  	   + F_(3,2) * F_(0,1) )
 */

template<class floatT, bool onDevice, size_t HaloDepth>
template<bool onDeviceRet, bool improved, bool improved_O6>
MemoryAccessor Weinberg<floatT, onDevice, HaloDepth>::WBField() {
    if (recompute) {
        _redBase.template iterateOverBulk<All, HaloDepth>(
                WBDensKernel<floatT, HaloDepth, onDevice, improved, improved_O6>(_gauge));
    }

    if (onDevice) {
        if (onDeviceRet) {
            return _redBase.getAccessor();
        } else {
            HostMemPointer->template copyFrom<onDevice>(_redBase.getMemPointer(), GInd::getLatData().vol4* sizeof(floatT));
            return MemoryAccessor(HostMemPointer->getPointer());
        }
    } else {
        if (!onDeviceRet) {
            return _redBase.getAccessor();
        } else {
            DevMemPointer->template copyFrom<onDevice>(_redBase.getMemPointer(), GInd::getLatData().vol4* sizeof(floatT));
            return MemoryAccessor(DevMemPointer->getPointer());
        }
    }
}

template<class floatT, bool onDevice, size_t HaloDepth>
template<bool improved, bool improved_O6>
floatT Weinberg<floatT, onDevice, HaloDepth>::WB() {
    if (recompute) {
        _redBase.template iterateOverBulk<All, HaloDepth>(
                WBDensKernel<floatT, HaloDepth, onDevice, improved, improved_O6>(_gauge));
    }
    floatT WB = 0;
    _redBase.reduce(WB, GInd::getLatData().vol4);
    return WB;
}

template<class floatT, bool onDevice, size_t HaloDepth>
template<bool improved, bool improved_O6>
void Weinberg<floatT, onDevice, HaloDepth>::WBTimeSlices(std::vector<floatT> &result) {

    if (recompute) {
        _redBase.template iterateOverTimeslices<All, HaloDepth>(
                WBDensKernel<floatT, HaloDepth, onDevice, improved, improved_O6>(_gauge));
    }
    /// Reduce the array on each GPU and also globally on all CPU's
    _redBase.reduceTimeSlices(result);
}


#define CLASS_INIT(floatT, HALO) \
template class Weinberg<floatT,true,HALO>; \

INIT_PH(CLASS_INIT)

#define CLASS_INIT2(floatT, HALO) \
template MemoryAccessor Weinberg<floatT, true,HALO>::WBField<true,false>(); \
template floatT Weinberg<floatT, true,HALO>::WB<true,false>(); \
template void Weinberg<floatT, true,HALO>::WBTimeSlices<true,false>(std::vector<floatT> &result); \
template MemoryAccessor Weinberg<floatT, true,HALO>::WBField<false,true>(); \
template floatT Weinberg<floatT, true,HALO>::WB<false,true>(); \
template void Weinberg<floatT, true,HALO>::WBTimeSlices<false,true>(std::vector<floatT> &result); \
template MemoryAccessor Weinberg<floatT, true,HALO>::WBField<false,false>(); \
template floatT Weinberg<floatT, true,HALO>::WB<false,false>(); \
template void Weinberg<floatT, true,HALO>::WBTimeSlices<false,false>(std::vector<floatT> &result); \

INIT_PH(CLASS_INIT2)

/*
template MemoryAccessor Weinberg<floatT, true,HALO>::WBField<true>(); \
template floatT Weinberg<floatT, true,HALO>::WB<true>(); \
template void Weinberg<floatT, true,HALO>::WBTimeSlices<true>(std::vector<floatT> &result); \
template MemoryAccessor Weinberg<floatT, true,HALO>::WBField<false>(); \
template floatT Weinberg<floatT, true,HALO>::WB<false>(); \
template void Weinberg<floatT, true,HALO>::WBTimeSlices<false>(std::vector<floatT> &result); \
*/
