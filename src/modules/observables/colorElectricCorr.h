/*
 * colorElectricCorr.h
 *
 * v1.0: L. Altenkort, 28 Jan 2019
 *
 * Measure Color-Electric Correlator (ColorElectricCorr) using the multi-GPU framework. Read sketch from right to left
 * (time advances to the left, space advances to the top)
 *          <----   <------  ^ <---^
 *          |  -  |          |  -  |   +  flipped = "going downwards" + "going upwards"
 * <------  v <---v          <----
 *
 */

#pragma once

#include "../../base/latticeContainer.h"
#include "../../gauge/gaugefield.h"

/// Class for the ColorElectricCorr. (cf. numerator of Eq. (4.2) in arXiv:0901.1195. Get denominator via
/// PolyakovLoop class)
template<class floatT,bool onDevice,size_t HaloDepth, CompressionType comp = R18>
class ColorElectricCorr{
protected:
    LatticeContainer<onDevice,COMPLEX(floatT)> _redBase;
    Gaugefield<floatT, onDevice, HaloDepth, comp> &_gauge;

private:
    typedef GIndexer<All, HaloDepth> GInd;
    const size_t Ntau = GInd::getLatData().lt;
    const size_t elems = GInd::getLatData().vol4;
    const size_t vol = GInd::getLatData().globvol4;

public:
    explicit ColorElectricCorr(Gaugefield<floatT, onDevice, HaloDepth, comp> &gaugefield) :
            _redBase(gaugefield.getComm()),
            _gauge(gaugefield){
        _redBase.adjustSize(GInd::getLatData().vol4);
    }

    ///call this to get the color electric correlator G_E(dt). Don't forget to exchange halos before this!
    std::vector<COMPLEX(floatT)> getColorElectricCorr_naive();
    std::vector<COMPLEX(floatT)> getColorElectricCorr_clover();
};

