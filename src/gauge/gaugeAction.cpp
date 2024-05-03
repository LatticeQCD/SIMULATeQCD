//
// Created by Lukas Mazur on 22.06.18.
//

#include "gaugeAction.h"
#include "../define.h"
#include "../base/math/complex.h"
#include "../base/gutils.h"
#include "../base/math/su3array.h"
#include "../modules/observables/fieldStrengthTensor.h"

#include <stdint.h>
#include <stdio.h>
#include "gauge_kernels.cpp"

template<class floatT, bool onDevice,size_t HaloDepth, CompressionType comp>
template<bool onDeviceRet>
MemoryAccessor GaugeAction<floatT, onDevice, HaloDepth,comp>::getField() {
    if (onDevice) {
        if (onDeviceRet) {
            return _redBase.getAccessor();
        }
        else {
            HostMemPointer->template copyFrom<onDevice>(_redBase.getMemPointer(), sizeof(floatT)*GInd::getLatData().vol4);
            return MemoryAccessor(HostMemPointer->getPointer());
        }
    } else {
        if (!onDeviceRet) {
            return _redBase.getAccessor();
        } else {
            DevMemPointer->template copyFrom<onDevice>(_redBase.getMemPointer(), sizeof(floatT)*GInd::getLatData().vol4);
            return MemoryAccessor(DevMemPointer->getPointer());
        }
    }
}

template<class floatT, bool onDevice,size_t HaloDepth, CompressionType comp>
template<bool onDeviceRet>
MemoryAccessor GaugeAction<floatT, onDevice, HaloDepth,comp>::getPlaquetteField() {
    if (recompute) {
        _redBase.template iterateOverBulk<All, HaloDepth>(
                plaquetteKernel<floatT, onDevice, HaloDepth,comp>(_gauge));
    }
    return getField<onDeviceRet>();
}

template<class floatT, bool onDevice,size_t HaloDepth, CompressionType comp>
template<bool onDeviceRet>
MemoryAccessor GaugeAction<floatT, onDevice, HaloDepth,comp>::getCloverField() {
    if (recompute) {
        _redBase.template iterateOverBulk<All, HaloDepth>(
                cloverKernel<floatT, onDevice, HaloDepth,comp>(_gauge));
    }
    return getField<onDeviceRet>();
}
template<class floatT, bool onDevice,size_t HaloDepth, CompressionType comp>
template<bool onDeviceRet>
MemoryAccessor GaugeAction<floatT, onDevice, HaloDepth,comp>::getRectangleField() {
    if (recompute) {
        _redBase.template iterateOverBulk<All, HaloDepth>(
                rectangleKernel<floatT, onDevice, HaloDepth,comp>(_gauge));
    }
    return getField<onDeviceRet>();
}


template<class floatT, bool onDevice,size_t HaloDepth, CompressionType comp>
__host__ floatT GaugeAction<floatT, onDevice, HaloDepth,comp>::barePlaquette() {

    if (recompute) {
        _redBase.template iterateOverBulk<All, HaloDepth>(
                plaquetteKernel<floatT, onDevice, HaloDepth,comp>(_gauge));
    }
    floatT plaq = 0;
    _redBase.reduce(plaq, GInd::getLatData().vol4);
    return plaq;
}

template<class floatT, bool onDevice,size_t HaloDepth, CompressionType comp>
__host__ floatT GaugeAction<floatT, onDevice, HaloDepth,comp>::barePlaquetteSS() {

//    if (recompute) {
        _redBase.template iterateOverBulk<All, HaloDepth>(
                plaquetteKernelSS<floatT, onDevice, HaloDepth,comp>(_gauge));
//    }
    floatT plaq = 0;
    _redBase.reduce(plaq, GInd::getLatData().vol4);
    return plaq;
}


template<class floatT, bool onDevice,size_t HaloDepth, CompressionType comp>
__host__ floatT GaugeAction<floatT, onDevice, HaloDepth,comp>::bareUtauMinusUsigma() {

    if (recompute) {
        _redBase.template iterateOverBulk<All, HaloDepth>(
                UtauMinusUsigmaKernel<floatT, onDevice, HaloDepth,comp>(_gauge));
    }
    floatT subtraction = 0;
    _redBase.reduce(subtraction, GInd::getLatData().vol4);
    return subtraction;
}


template<class floatT, bool onDevice,size_t HaloDepth, CompressionType comp>
__host__ floatT GaugeAction<floatT, onDevice, HaloDepth,comp>::bareClover() {

    if (recompute) {
        _redBase.template iterateOverBulk<All, HaloDepth>(
                cloverKernel<floatT, onDevice, HaloDepth,comp>(_gauge));
    }
    floatT clov = 0;
    _redBase.reduce(clov, GInd::getLatData().vol4);
    return clov;
}

template<class floatT, bool onDevice,size_t HaloDepth, CompressionType comp>
__host__ floatT GaugeAction<floatT, onDevice, HaloDepth,comp>::bareRectangle() {

    if (recompute) {
        _redBase.template iterateOverBulk<All, HaloDepth>(
                rectangleKernel<floatT, onDevice, HaloDepth,comp>(_gauge));
    }
    floatT rect = 0;
    _redBase.reduce(rect, GInd::getLatData().vol4);
    return rect;
}

template<class floatT, bool onDevice,size_t HaloDepth, CompressionType comp>
void GaugeAction<floatT, onDevice, HaloDepth,comp>::cloverTimeSlices(std::vector<floatT> &result) {

    if (recompute) {
        _redBase.template iterateOverTimeslices<All, HaloDepth>(
                cloverKernel<floatT, onDevice, HaloDepth,comp>(_gauge));
    }
    _redBase.reduceTimeSlices(result);
    LatticeDimensions lat = GInd::getLatData().globalLattice();
    for(auto &elem : result){
        elem/=(2.0 * lat[0]*lat[1]*lat[2]);
    }
}

#define CLASS_INIT(floatT,HALO,COMP) \
template class GaugeAction<floatT,true,HALO,COMP>; \
template class GaugeAction<floatT,false,HALO,COMP>; \
template MemoryAccessor GaugeAction<floatT, true,HALO,COMP>::getCloverField<false>(); \

INIT_PHC(CLASS_INIT)

