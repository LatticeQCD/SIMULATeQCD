/*
 * MDWF analytic force-contraction check scaffold.
 *
 * This helper compares a single-link analytic action derivative against the
 * finite-difference action derivative from MDWFFiniteDifferenceHarness.  It is
 * deliberately not a production force accumulator: it does not allocate gauge
 * force, update momenta, call RHMC/HMC, touch HISQ, or define final integrator
 * force-sign conventions.
 */

#pragma once

#include "MDWFFiniteDifferenceHarness.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

template<class floatT>
struct MDWFAnalyticForceContractionResult {
    double finite_difference_derivative;
    double analytic_derivative;
    double absolute_difference;
    double relative_difference;
    bool passed;
};

template<class floatT>
__host__ __device__ double mdwfContractSingleLinkActionDerivative(
    const SU3<floatT> &single_link_action_derivative,
    const MDWFFiniteDifferenceProbe<floatT> &probe) {

    const SU3<floatT> generator = mdwfFiniteDifferenceGenerator<floatT>(probe.generator_id);
    return static_cast<double>(real(tr_c(generator, single_link_action_derivative)));
}

template<class floatT>
__host__ __device__ SU3<floatT> mdwfSingleLinkLinearActionDerivativeMatrix(
    const SU3<floatT> &base_link,
    const SU3<floatT> &linear_action_matrix,
    MDWFFiniteDifferenceMultiplicationSide multiplication_side) {

    if (multiplication_side == MDWFFiniteDifferenceMultiplicationSide::Right) {
        return linear_action_matrix * base_link;
    }
    return base_link * linear_action_matrix;
}

template<class floatT>
MDWFAnalyticForceContractionResult<floatT> compareMDWFAnalyticForceContraction(
    const MDWFFiniteDifferenceResult<floatT> &finite_difference,
    double analytic_derivative,
    double absolute_tolerance,
    double relative_tolerance) {

    if (absolute_tolerance < 0.0
        || relative_tolerance < 0.0
        || !std::isfinite(absolute_tolerance)
        || !std::isfinite(relative_tolerance)
        || !std::isfinite(analytic_derivative)
        || !std::isfinite(finite_difference.derivative)) {
        throw std::runtime_error("MDWF analytic force contraction check requires finite inputs");
    }

    const double absolute_difference = std::abs(
        finite_difference.derivative - analytic_derivative);
    const double scale = std::max(1.0, std::abs(finite_difference.derivative));
    const double relative_difference = absolute_difference / scale;
    const bool passed = finite_difference.converged
                        && absolute_difference <= absolute_tolerance
                        && relative_difference <= relative_tolerance;

    return {
        finite_difference.derivative,
        analytic_derivative,
        absolute_difference,
        relative_difference,
        passed
    };
}
