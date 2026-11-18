#include "topology.h"

/*topChargeDensKernel(SU3Accessor<floatT> gAcc, floatT * topChargeDensArray)
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


template<class floatT, bool onDevice, size_t HaloDepth>
template<bool improved, bool improved_O6>
void Topology<floatT, onDevice, HaloDepth>::topChargeSlices(std::array<std::vector<floatT>, 4> &result) {
    // Density in bulk order (the layout of topCharge(); topChargeTimeSlices uses a time-slice layout).
    _redBase.template iterateOverBulk<All, HaloDepth>(
            topChargeDensKernel<floatT, HaloDepth, onDevice, improved, improved_O6>(_gauge));
    recompute = true;   // never let a later dontRecomputeField() reuse this layout by accident
    LatticeData lat = GInd::getLatData();
    const size_t vol4 = lat.vol4;
    HostMemPointer->adjustSize(vol4 * sizeof(floatT));
    HostMemPointer->template copyFrom<onDevice>(_redBase.getMemPointer(), vol4 * sizeof(floatT));
    MemoryAccessor acc(HostMemPointer->getPointer());
    const size_t ext[4] = {(size_t) lat.globLX, (size_t) lat.globLY, (size_t) lat.globLZ, (size_t) lat.globLT};
    const size_t off[4] = {0, ext[0], ext[0] + ext[1], ext[0] + ext[1] + ext[2]};
    std::vector<double> buf(off[3] + ext[3], 0.0);
    for (size_t i = 0; i < vol4; i++) {
        floatT q;
        acc.getValue<floatT>(i, q);
        const sitexyzt g = lat.globalPos(GInd::getSite(i).coord);
        buf[off[0] + g.x] += q;
        buf[off[1] + g.y] += q;
        buf[off[2] + g.z] += q;
        buf[off[3] + g.t] += q;
    }
    _gauge.getComm().reduce(buf.data(), (int) buf.size());
    for (int mu = 0; mu < 4; mu++) {
        result[mu].assign(buf.begin() + off[mu], buf.begin() + off[mu] + ext[mu]);
    }
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
template void Topology<floatT, true,HALO>::topChargeSlices<true,false>(std::array<std::vector<floatT>, 4> &result); \
template void Topology<floatT, true,HALO>::topChargeSlices<false,false>(std::array<std::vector<floatT>, 4> &result); \

INIT_PH(CLASS_INIT2)

/*
template MemoryAccessor Topology<floatT, true,HALO>::topChargeField<true>(); \
template floatT Topology<floatT, true,HALO>::topCharge<true>(); \
template void Topology<floatT, true,HALO>::topChargeTimeSlices<true>(std::vector<floatT> &result); \
template MemoryAccessor Topology<floatT, true,HALO>::topChargeField<false>(); \
template floatT Topology<floatT, true,HALO>::topCharge<false>(); \
template void Topology<floatT, true,HALO>::topChargeTimeSlices<false>(std::vector<floatT> &result); \
*/
