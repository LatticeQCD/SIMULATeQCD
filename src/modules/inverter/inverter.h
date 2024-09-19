/*
 * inverter.h
 *
 */

#pragma once

#include "../../gauge/gaugefield.h"
#include "../../spinor/spinorfield.h"
#include "../../spinor/eigenpairs.h"
#include "../../base/math/simpleArray.h"

// /// Abstract base class for all kind of linear operators that shall enter the inversion
// template <typename Vector>
// class LinearOperator{
// public:
//     virtual void applyMdaggM(Vector&, const Vector&, bool update = true) = 0;
// };


/// Class for multiple right hand side inversion. NStacks is the number of right hand sides. The objects to be inverted
/// must have the member function vec.dotProductStacked(vec&) and must to be able to use the operator syntax.
template<typename floatT, size_t NStacks = 1>
class ConjugateGradient{
public:

    template <typename Spinor_t>
    void invert(LinearOperator<Spinor_t>& dslash, Spinor_t& spinorOut, Spinor_t& spinorIn, int max_iter, double precision);

    template <typename Spinor_t>
    void invert_new(LinearOperator<Spinor_t>& dslash, Spinor_t& spinorOut, const Spinor_t& spinorIn, const int max_iter, const double precision);

    template <typename eigenpairs, typename Spinor_t>
    void invert_deflation(LinearOperator<Spinor_t>& dslash, Spinor_t& spinorOut, 
        const Spinor_t& spinorIn, 
        eigenpairs& eigen,
        const int max_iter, const double precision);

    template <typename Spinor_t>
    void invert_res_replace(LinearOperator<Spinor_t>& dslash, Spinor_t& spinorOut, const Spinor_t& spinorIn,
                            const int max_iter, const double precision, double delta);

    template <typename Spinor_t, typename Spinor_t_half>
    void invert_mixed(LinearOperator<Spinor_t>& dslash, LinearOperator<Spinor_t_half>& dslash_inner, Spinor_t& spinorOut, const Spinor_t& spinorIn,
                      const int max_iter, const double precision, double delta);
};


template <typename floatT, bool onDevice, Layout LatLayout, int HaloDepth, size_t NStacks>
class MultiShiftCG {
public:
    void invert(LinearOperator<Spinorfield<floatT, onDevice, LatLayout, HaloDepth, 1>>& dslash,
                Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>& spinorOut,
                Spinorfield<floatT, onDevice, LatLayout, HaloDepth, 1>& spinorIn,
                SimpleArray<floatT, NStacks> sigma, int max_iter, double precision);
};


/// This is a reimplementation of the old BielefeldGPUcode MultishiftCG, the cool feature here is that parts of the
/// stack that have already converged are not updated any more in the CGM iterations. Might produce better results for
/// the rational approximations. Not finished yet.
template <typename floatT, size_t NStacks = 14>
class AdvancedMultiShiftCG {
public:
    template <typename SpinorIn_t, typename SpinorOut_t>
    void invert(LinearOperator<SpinorIn_t>& dslash, SpinorOut_t& spinorOut, const SpinorIn_t& spinorIn,
                 SimpleArray<floatT, NStacks> sigma, const int max_iter, const double precision);
};

