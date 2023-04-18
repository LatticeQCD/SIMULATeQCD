//
// Created by Hai-Tao Shu on 11.11.2020
//

#pragma once

#include "../../base/math/correlators.h"
#include "../observables/PolyakovLoop.h"
#include "../gaugeFixing/PolyakovLoopCorrelator.h"


template<class floatT, bool onDevice, size_t HaloDepth>
class WilsonLineCorrelator : protected CorrelatorTools<floatT,onDevice,HaloDepth> {
protected:
    Gaugefield<floatT, onDevice, HaloDepth> &_gauge;
private:
    typedef GIndexer<All, HaloDepth> GInd;
    const int _elems = GInd::getLatData().vol3;
    floatT wlcaoff, wlc1off, wlc8off, wlcaon, wlc1on, wlc8on;

public:

    //construct the class
    WilsonLineCorrelator(Gaugefield<floatT, onDevice, HaloDepth> &gaugefield) : CorrelatorTools<floatT, onDevice, HaloDepth>(), _gauge(gaugefield) {}

    void WLCtoArrays(std::vector<floatT> &vec_wlca_full, std::vector<floatT> &vec_wlc1_full, std::vector<floatT> &vec_wlc8_full,
                     std::vector<int> &vec_factor , std::vector<int> &vec_weight , bool fastWLC);

};