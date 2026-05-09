/*
 * MDWF finite-difference action-derivative harness.
 *
 * This helper evaluates
 *
 *     [S(U_+) - S(U_-)] / (2 epsilon)
 *
 * for a single-link gauge perturbation.  It deliberately does not allocate or
 * accumulate gauge force, update momenta, call HMC/RHMC, touch HISQ, or define
 * final force sign conventions.
 */

#pragma once

#include "MDWFPseudofermionAction.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

enum class MDWFFiniteDifferenceMultiplicationSide {
    Left,
    Right
};

template<class floatT>
struct MDWFFiniteDifferenceProbe {
    int x;
    int y;
    int z;
    int t;
    uint8_t mu;
    int generator_id;
    floatT epsilon;
    MDWFFiniteDifferenceMultiplicationSide multiplication_side;
};

template<class floatT>
struct MDWFFiniteDifferenceActionValue {
    double action_real;
    double action_imag;
    double max_shifted_residual;
    bool converged;
};

template<class floatT>
struct MDWFFiniteDifferenceResult {
    MDWFFiniteDifferenceActionValue<floatT> plus;
    MDWFFiniteDifferenceActionValue<floatT> minus;
    double derivative;
    double action_imag_relative;
    double max_shifted_residual;
    bool converged;
};

template<class floatT>
MDWFFiniteDifferenceActionValue<floatT> makeMDWFFiniteDifferenceActionValue(
    const MDWFRationalActionResult<floatT> &action_result) {

    double max_residue = 0.0;
    for (size_t term = 0; term < action_result.rational_result.shifts.size(); term++) {
        max_residue = std::max(max_residue, action_result.rational_result.shifts[term].residue);
    }

    return {
        action_result.action_real,
        action_result.action_imag,
        max_residue,
        action_result.rational_result.converged()
    };
}

template<class floatT>
__host__ __device__ SU3<floatT> mdwfFiniteDifferenceGenerator(int generator_id) {
    const COMPLEX(floatT) zero(0.0, 0.0);
    const COMPLEX(floatT) one(1.0, 0.0);
    const COMPLEX(floatT) minus_one(-1.0, 0.0);
    const COMPLEX(floatT) plus_i(0.0, 1.0);
    const COMPLEX(floatT) minus_i(0.0, -1.0);

    if (generator_id == 1) {
        return SU3<floatT>(
            zero, one, zero,
            minus_one, zero, zero,
            zero, zero, zero);
    }

    if (generator_id == 2) {
        return SU3<floatT>(
            zero, zero, zero,
            zero, zero, one,
            zero, minus_one, zero);
    }

    return SU3<floatT>(
        plus_i, zero, zero,
        zero, minus_i, zero,
        zero, zero, zero);
}

template<class floatT, CompressionType comp>
struct MDWFSingleLinkPerturbation {
    SU3Accessor<floatT, comp> gauge_in;
    MDWFFiniteDifferenceProbe<floatT> probe;
    int sign;

    MDWFSingleLinkPerturbation(const SU3Accessor<floatT, comp> &gauge_in_in,
                               const MDWFFiniteDifferenceProbe<floatT> &probe_in,
                               int sign_in)
        : gauge_in(gauge_in_in),
          probe(probe_in),
          sign(sign_in) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu) {
        SU3<floatT> link = gauge_in.getLink(siteMu);

        if (siteMu.coord.x == probe.x
            && siteMu.coord.y == probe.y
            && siteMu.coord.z == probe.z
            && siteMu.coord.t == probe.t
            && siteMu.mu == probe.mu) {
            const floatT signed_epsilon = static_cast<floatT>(sign) * probe.epsilon;
            const SU3<floatT> generator = mdwfFiniteDifferenceGenerator<floatT>(probe.generator_id);
            const SU3<floatT> perturbation = su3_exp(signed_epsilon * generator);

            if (probe.multiplication_side == MDWFFiniteDifferenceMultiplicationSide::Right) {
                return link * perturbation;
            }
            return perturbation * link;
        }

        return link;
    }
};

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void applyMDWFSingleLinkPerturbation(
    Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_out,
    const Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_in,
    const MDWFFiniteDifferenceProbe<floatT> &probe,
    int sign) {

    if (probe.epsilon <= static_cast<floatT>(0.0)
        || !std::isfinite(static_cast<double>(probe.epsilon))) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF finite-difference probe requires positive finite epsilon"));
    }
    if (probe.mu >= 4) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF finite-difference probe requires mu in [0, 3], got ", static_cast<int>(probe.mu)));
    }
    if (sign != 1 && sign != -1) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF finite-difference perturbation sign must be +1 or -1"));
    }

    gauge_out.template iterateOverFullAllMu<>(
        MDWFSingleLinkPerturbation<floatT, comp>(gauge_in.getAccessor(), probe, sign));
    gauge_out.updateAll();
}

template<class floatT,
         bool onDevice,
         size_t HaloDepth,
         CompressionType comp,
         class ActionEvaluator>
MDWFFiniteDifferenceResult<floatT> evaluateMDWFFiniteDifferenceAction(
    Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_plus,
    Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_minus,
    const Gaugefield<floatT, onDevice, HaloDepth, comp> &base_gauge,
    const MDWFFiniteDifferenceProbe<floatT> &probe,
    ActionEvaluator &action_evaluator) {

    applyMDWFSingleLinkPerturbation(gauge_plus, base_gauge, probe, 1);
    applyMDWFSingleLinkPerturbation(gauge_minus, base_gauge, probe, -1);

    MDWFFiniteDifferenceActionValue<floatT> plus = action_evaluator(gauge_plus);
    MDWFFiniteDifferenceActionValue<floatT> minus = action_evaluator(gauge_minus);

    const double derivative = (plus.action_real - minus.action_real)
                              / (2.0 * static_cast<double>(probe.epsilon));
    const double plus_scale = std::max(1.0, std::abs(plus.action_real));
    const double minus_scale = std::max(1.0, std::abs(minus.action_real));
    const double action_imag_relative = std::max(
        std::abs(plus.action_imag) / plus_scale,
        std::abs(minus.action_imag) / minus_scale);
    const double max_shifted_residual = std::max(plus.max_shifted_residual,
                                                 minus.max_shifted_residual);

    return {
        plus,
        minus,
        derivative,
        action_imag_relative,
        max_shifted_residual,
        plus.converged && minus.converged
    };
}
