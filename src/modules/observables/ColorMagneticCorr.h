/*
 * ColorMagneticCorr.h
 *
 * v1.0: Hai-Tao Shu, 28 Oct 2020
 *
 * Measure Color-Magnetic Correlator using the multi-GPU framework. Read sketch from right to left
 *
 *              <-----------
 *             /|          ^|
 *            / |         / |
 *           /  |        /  |
 *          /   V       /   ^
 *          | - /       | - /
 *          |  /        |  /
 *          | /         | /
 * <------  v           |<      + flipped
 *
 * B_0=F_{12}=U_1(\vec{x})U_2(\vec{x}+\hat{1})-U_2(\vec{x})U_1(\vec{x}+\hat{2})
 * B_1=F_{20}=U_2(\vec{x})U_0(\vec{x}+\hat{2})-U_0(\vec{x})U_2(\vec{x}+\hat{0})
 * B_2=F_{01}=U_0(\vec{x})U_1(\vec{x}+\hat{0})-U_1(\vec{x})U_0(\vec{x}+\hat{1})
 *
 * similar to EE correlators but no shift for tau in the "square"
 */

#pragma once

#include "../../base/LatticeContainer.h"
#include "../../gauge/gaugefield.h"

template<class floatT,bool onDevice,size_t HaloDepth, CompressionType comp = R18>
class ColorMagneticCorr{
protected:
    LatticeContainer<onDevice,GCOMPLEX(floatT)> _redBase;
    Gaugefield<floatT, onDevice, HaloDepth, comp> &_gauge;

private:
    typedef GIndexer<All, HaloDepth> GInd;
    const size_t Ntau = GInd::getLatData().lt;
    const size_t elems = GInd::getLatData().vol4;
    const size_t vol = GInd::getLatData().globvol4;

public:
    ColorMagneticCorr(Gaugefield<floatT, onDevice, HaloDepth, comp> &gaugefield) :
    _redBase(gaugefield.getComm()),
    _gauge(gaugefield){
        _redBase.adjustSize(GInd::getLatData().vol4);
    }

    std::vector<GCOMPLEX(floatT)> getColorMagneticCorr_naive();
    std::vector<GCOMPLEX(floatT)> getColorMagneticCorr_clover();
};