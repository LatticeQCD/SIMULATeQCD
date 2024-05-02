#include "Topology.h"

/*topChargeDensKernel(gaugeAccessor<floatT> gAcc, floatT * topChargeDensArray)
	compute the topological charge density q(x) given by
		q(x)= 1/4pi^2 * tr(  F_(3,0) * F_(1,2)
					  	   + F_(3,1) * F_(2,0)
					  	   + F_(3,2) * F_(0,1) )
 */

template<class floatT, bool onDevice, size_t HaloDepth>
template<bool onDeviceRet, bool improved, bool improved_O6>
MemoryAccessor Topology<floatT, onDevice, HaloDepth>::topChargeField() {
    if (recompute) {
        _redBase.template iterateOverBulk<All, HaloDepth>(
                topChargeDensKernel<floatT, HaloDepth, onDevice, improved, improved_O6>(_gauge));
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
floatT Topology<floatT, onDevice, HaloDepth>::topCharge() {
    if (recompute) {
        _redBase.template iterateOverBulk<All, HaloDepth>(
                topChargeDensKernel<floatT, HaloDepth, onDevice, improved, improved_O6>(_gauge));
    }
    floatT topCharge = 0;
    _redBase.reduce(topCharge, GInd::getLatData().vol4);
    return topCharge;
}

template<class floatT, bool onDevice, size_t HaloDepth>
template<bool improved, bool improved_O6>
void Topology<floatT, onDevice, HaloDepth>::topChargeTimeSlices(std::vector<floatT> &result) {

    if (recompute) {
        _redBase.template iterateOverTimeslices<All, HaloDepth>(
                topChargeDensKernel<floatT, HaloDepth, onDevice, improved, improved_O6>(_gauge));
    }
    /// Reduce the array on each GPU and also globally on all CPU's
    _redBase.reduceTimeSlices(result);
}


#define CLASS_INIT(floatT, HALO) \
template class Topology<floatT,true,HALO>; \

INIT_PH(CLASS_INIT)

#define CLASS_INIT2(floatT, HALO) \
template MemoryAccessor Topology<floatT, true,HALO>::topChargeField<true,false>(); \
template floatT Topology<floatT, true,HALO>::topCharge<true,false>(); \
template void Topology<floatT, true,HALO>::topChargeTimeSlices<true,false>(std::vector<floatT> &result); \
template MemoryAccessor Topology<floatT, true,HALO>::topChargeField<false,true>(); \
template floatT Topology<floatT, true,HALO>::topCharge<false,true>(); \
template void Topology<floatT, true,HALO>::topChargeTimeSlices<false,true>(std::vector<floatT> &result); \
template MemoryAccessor Topology<floatT, true,HALO>::topChargeField<false,false>(); \
template floatT Topology<floatT, true,HALO>::topCharge<false,false>(); \
template void Topology<floatT, true,HALO>::topChargeTimeSlices<false,false>(std::vector<floatT> &result); \

INIT_PH(CLASS_INIT2)

/*
template MemoryAccessor Topology<floatT, true,HALO>::topChargeField<true>(); \
template floatT Topology<floatT, true,HALO>::topCharge<true>(); \
template void Topology<floatT, true,HALO>::topChargeTimeSlices<true>(std::vector<floatT> &result); \
template MemoryAccessor Topology<floatT, true,HALO>::topChargeField<false>(); \
template floatT Topology<floatT, true,HALO>::topCharge<false>(); \
template void Topology<floatT, true,HALO>::topChargeTimeSlices<false>(std::vector<floatT> &result); \
*/
