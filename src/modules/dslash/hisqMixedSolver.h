/*
 * hisqMixedSolver.h
 *
 * Mixed-precision multi-shift solver for the HISQ M^dagger M: keeps single-precision copies of
 * the smeared (fat and Naik) links of a double-precision HisqDSlash, refreshes them at every
 * solve (the links change during molecular dynamics), and runs MixedPrecMultiShiftCG.
 * Memory is allocated on first use.
 */
#pragma once

#include "dslash.h"
#include "../inverter/mixedMultiShiftCG.h"
#include <memory>

template<typename floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, size_t NStacks>
class HisqMixedMultiShiftSolver {
    struct FloatLinks {
        Gaugefield<float, onDevice, HaloDepth, R18> smeared;
        Gaugefield<float, onDevice, HaloDepth, U3R14> naik;
        HisqDSlash<float, onDevice, Even, HaloDepth, HaloDepthSpin, 1> dslash;
        explicit FloatLinks(CommunicationBase &comm)
            : smeared(comm, "MixedSolverSmearedFloat"), naik(comm, "MixedSolverNaikFloat"),
              dslash(smeared, naik, 0.0) {}
    };
    std::unique_ptr<FloatLinks> _float;
    MixedPrecMultiShiftCG<floatT, onDevice, Even, HaloDepthSpin, NStacks> _cg;

public:
    //! Same arguments and result as AdvancedMultiShiftCG<floatT, NStacks>::invert with the
    //! (massless) HisqDSlash dslash; delta sets how often the residual is recomputed in double.
    void invert(HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, 1> &dslash,
                Spinorfield<floatT, onDevice, Even, HaloDepthSpin, NStacks> &spinorOut,
                Spinorfield<floatT, onDevice, Even, HaloDepthSpin, 1> &spinorIn,
                SimpleArray<floatT, NStacks> sigma, const int max_iter, const double precision,
                const double delta) {
        if (!_float) _float = std::make_unique<FloatLinks>(spinorIn.getComm());
        _float->smeared.convert_precision(dslash.getGaugeSmeared());
        _float->naik.convert_precision(dslash.getGaugeNaik());
        _cg.invert(dslash, _float->dslash, spinorOut, spinorIn, sigma, max_iter, precision, delta);
    }
};
