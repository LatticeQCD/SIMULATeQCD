/*
 * MDWF pseudofermion/action scaffold.
 *
 * This file adds only small wrappers around the already isolated coupled-5D
 * rational operator.  It does not generate random noise, define determinant
 * powers, call RHMC/HMC, compute forces, or assign heatbath/action/force
 * coefficient semantics.  Coefficients are supplied explicitly by the caller.
 */

#pragma once

#include "MDWFRationalOperator.h"

#include <cmath>
#include <string>

template<class floatT>
struct MDWFPseudofermionHeatbathResult {
    MDWFCoupledMultiShiftCGResults<floatT> rational_result;
    double noise_norm2;
    double pseudofermion_norm2;
};

template<class floatT>
struct MDWFRationalActionResult {
    MDWFCoupledMultiShiftCGResults<floatT> rational_result;
    COMPLEX(double) action;
    double action_real;
    double action_imag;
};

template<class floatT, class CoupledAdapter, size_t BlockSize = 64>
MDWFPseudofermionHeatbathResult<floatT> applyMDWFPseudofermionHeatbath(
    CoupledAdapter &adapter,
    typename CoupledAdapter::Spinor &pseudofermion,
    typename CoupledAdapter::Spinor &noise,
    const MDWFRationalCoefficients<floatT> &coefficients,
    int max_iter,
    double precision,
    const std::string &name = "MDWF_pseudofermion_heatbath") {

    MDWFRationalOperator<floatT, CoupledAdapter, BlockSize> rational_operator(
        coefficients, max_iter, precision, name);
    MDWFCoupledMultiShiftCGResults<floatT> rational_result
        = rational_operator.apply(adapter, pseudofermion, noise, true);

    return {
        rational_result,
        adapter.norm2(noise),
        adapter.norm2(pseudofermion)
    };
}

template<class floatT, class CoupledAdapter, size_t BlockSize = 64>
MDWFRationalActionResult<floatT> computeMDWFRationalAction(
    CoupledAdapter &adapter,
    typename CoupledAdapter::Spinor &action_workspace,
    typename CoupledAdapter::Spinor &field,
    const MDWFRationalCoefficients<floatT> &coefficients,
    int max_iter,
    double precision,
    const std::string &name = "MDWF_rational_action") {

    MDWFRationalOperator<floatT, CoupledAdapter, BlockSize> rational_operator(
        coefficients, max_iter, precision, name);
    MDWFCoupledMultiShiftCGResults<floatT> rational_result
        = rational_operator.apply(adapter, action_workspace, field, true);

    const COMPLEX(double) action = adapter.dotProduct5D(field, action_workspace);
    return {
        rational_result,
        action,
        real<double>(action),
        imag<double>(action)
    };
}
