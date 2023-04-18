/*
 * PolyakovLoop.h
 *
 * L. Mazur, 22 Jun 2018
 *
 */

#pragma once

#include "../../base/LatticeContainer.h"
#include "../../gauge/gaugefield.h"
#include "../../base/math/gcomplex.h"
#include "../../base/math/correlators.h"

/// Class for calculating Polyakov loops. Make sure you exchange halos before measurements.
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp = R18>
class PolyakovLoop {
protected:
    LatticeContainer<onDevice,GCOMPLEX(floatT)> _redBase;
    Gaugefield<floatT, onDevice, HaloDepth, comp> &_gauge;

private:
    typedef GIndexer<All, HaloDepth> GInd;
    const size_t elems = GInd::getLatData().vol3;
    const size_t spatialvol = GInd::getLatData().globvol3;

public:
    PolyakovLoop(Gaugefield<floatT, onDevice, HaloDepth, comp> &gaugefield) : _redBase(gaugefield.getComm()), _gauge(gaugefield) {
        _redBase.adjustSize(elems);
    }

    GCOMPLEX(floatT) getPolyakovLoop();          /// Return reduced Polyakov loop
    void  PloopInArray(MemoryAccessor _ploop);   /// Store untraced Polyakov loop in array _ploop
};