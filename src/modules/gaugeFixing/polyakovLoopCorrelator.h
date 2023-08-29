/*
 * polyakovLoopCorrelator.h
 *
 * D. Clarke, 27 Apr 2020
 *
 */
#pragma once
#include "../../base/math/correlators.h"
#include "../observables/polyakovLoop.h"

template<class floatT, bool onDevice, size_t HaloDepth>
class PolyakovLoopCorrelator : protected CorrelatorTools<floatT,onDevice,HaloDepth> {
protected:
    Gaugefield<floatT, onDevice, HaloDepth>   &_gauge;

private:
    floatT plcaoff, plc1off, plc8off, plcaon, plc1on, plc8on;

public:

    PolyakovLoopCorrelator(Gaugefield<floatT, onDevice, HaloDepth> &gaugefield) : CorrelatorTools<floatT, onDevice, HaloDepth>(), _gauge(gaugefield) {}

    /// Calculate Polyakov loop correlators from _ploopGPU and put the results in the right arrays.
    void PLCtoArrays(std::vector<floatT> &vec_plca, std::vector<floatT> &vec_plc1, std::vector<floatT> &vec_plc8,
                     std::vector<int> &vec_factor , std::vector<int> &vec_weight , bool fastPLC);

};

/// Compute average, singlet, and octet Polyakov loop correlation contributions.
/// INTENT: IN--pol1, pol2; OUT--addplca, addplc1, addplc8
template<class floatT>
__host__ __device__ void inline plc_contrib(SU3<floatT> pol1, SU3<floatT> pol2, floatT &addplca,
                                            floatT &addplc1,floatT &addplc8) {
    floatT tmpa  = real(tr_c(pol1)*tr_c(dagger(pol2)));
    addplca     += tmpa;
    floatT tmps  = tr_d(pol1,dagger(pol2));
    addplc1     += tmps;
    addplc8     += 0.125*tmpa - 0.04166666666*tmps;
}

