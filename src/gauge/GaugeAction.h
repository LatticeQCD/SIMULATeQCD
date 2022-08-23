//
// Created by Lukas Mazur on 22.06.18.
//

#ifndef ENERGYDENSITY_H
#define ENERGYDENSITY_H

#include "../base/LatticeContainer.h"
#include "../modules/observables/FieldStrengthTensor.h"
#include "gaugefield.h"

/// Don't forget to exchange Halos before computing any of these observables!
template<class floatT, bool onDevice, size_t HaloDepth,CompressionType comp = R18>
class GaugeAction {
private:
    typedef GIndexer<All, HaloDepth> GInd;
    LatticeContainer<onDevice, floatT> _redBase;
    Gaugefield<floatT, onDevice, HaloDepth,comp> &_gauge;

    bool recompute;
    gMemoryPtr<onDevice> DevMemPointer;
    gMemoryPtr<false> HostMemPointer;

    template<bool onDeviceRet>
    MemoryAccessor getField();

    HOST floatT barePlaquette();
    HOST floatT bareUtauMinusUsigma();
    HOST floatT bareClover();
    HOST floatT bareRectangle();

    HOST floatT barePlaquetteSS();


public:
    GaugeAction(Gaugefield<floatT, onDevice, HaloDepth,comp> &gaugefield) :
            _redBase(gaugefield.getComm()), _gauge(gaugefield), recompute(true),
            DevMemPointer(MemoryManagement::getMemAt<onDevice>("DevMemPointer")),
            HostMemPointer(MemoryManagement::getMemAt<false>("HostMemPointer"))
    {
        _redBase.adjustSize(GInd::getLatData().vol4);
    }

    ~GaugeAction() {}

    template<bool onDeviceRet>
    MemoryAccessor getPlaquetteField();

    template<bool onDeviceRet>
    MemoryAccessor getCloverField();

    template<bool onDeviceRet>
    MemoryAccessor getRectangleField();

    floatT plaquette() {
        // using definition Sg=\beta \sum_n \sum_{\mu\nu}  (1 - 1/3*ReTrU_{\mu\nu})
        // calculates the mean value of 1/3*ReTrU_{\mu\nu} (without mu,nu running)
        // 18 = 3*6. 3 from beta/3, 6 from 6 distinct plaq on each site
        return barePlaquette()/(GInd::getLatData().globalLattice().mult() * 18);
    }

    floatT plaquetteSS() {
        return barePlaquetteSS()/(GInd::getLatData().globalLattice().mult() * 9);
    }

    floatT UtauMinusUsigma() {
        return bareUtauMinusUsigma()/(GInd::getLatData().globalLattice().mult() * 18);
    }

    floatT rectangle() {
        return bareRectangle()/(GInd::getLatData().globalLattice().mult() * 36);
    }

    floatT wilson() {
        /// ????
        return 3*barePlaquette();
    }

    floatT clover() {
        return bareClover() / (2.0 * GInd::getLatData().globalLattice().mult());
    }

    floatT symanzik() {
        floatT tmp = (5.0 / 3.0) * barePlaquette();
        tmp += (-1.0 / 12.0)* bareRectangle();
        return tmp / 3.0;
    }

    floatT symanzik_staggered() {
        floatT tmp = -(5.0 / 3.0) * barePlaquette();
        tmp += (-1.0 / 12.0)* bareRectangle();
        return tmp / 3.0;
    }

    void cloverTimeSlices(std::vector<floatT> &result);

    void dontRecomputeField(){recompute = false;}
    void recomputeField(){recompute = true;}
};


#endif //ENERGYDENSITY_H
