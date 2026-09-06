/*
 * MDWF test-only direction-independent contraction helpers.
 *
 * These helpers provide an explicit eight-element su(3) basis, reconstruct the
 * unique algebra matrix representing measured directional derivatives, and
 * apply a selected-link perturbation along an arbitrary algebra direction.
 * They do not define a production raw-bilinear projection, ipdot convention,
 * HMC sign, all-link force accumulator, or MPI ownership implementation.
 */

#pragma once

#include "MDWFFiniteDifferenceHarness.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

static constexpr size_t MDWFDirectionIndependentBasisSize = 8;

template<class floatT>
SU3<floatT> mdwfDirectionIndependentBasisGenerator(size_t generator_index) {
    const COMPLEX(floatT) zero(0.0, 0.0);
    const COMPLEX(floatT) one(1.0, 0.0);
    const COMPLEX(floatT) minus_one(-1.0, 0.0);
    const COMPLEX(floatT) plus_i(0.0, 1.0);
    const COMPLEX(floatT) minus_i(0.0, -1.0);
    const COMPLEX(floatT) minus_two_i(0.0, -2.0);

    if (generator_index == 0) {
        return SU3<floatT>(
            zero, one, zero,
            minus_one, zero, zero,
            zero, zero, zero);
    }
    if (generator_index == 1) {
        return SU3<floatT>(
            zero, plus_i, zero,
            plus_i, zero, zero,
            zero, zero, zero);
    }
    if (generator_index == 2) {
        return SU3<floatT>(
            zero, zero, one,
            zero, zero, zero,
            minus_one, zero, zero);
    }
    if (generator_index == 3) {
        return SU3<floatT>(
            zero, zero, plus_i,
            zero, zero, zero,
            plus_i, zero, zero);
    }
    if (generator_index == 4) {
        return SU3<floatT>(
            zero, zero, zero,
            zero, zero, one,
            zero, minus_one, zero);
    }
    if (generator_index == 5) {
        return SU3<floatT>(
            zero, zero, zero,
            zero, zero, plus_i,
            zero, plus_i, zero);
    }
    if (generator_index == 6) {
        return SU3<floatT>(
            plus_i, zero, zero,
            zero, minus_i, zero,
            zero, zero, zero);
    }
    if (generator_index == 7) {
        return SU3<floatT>(
            plus_i, zero, zero,
            zero, plus_i, zero,
            zero, zero, minus_two_i);
    }

    throw std::runtime_error(stdLogger.fatal(
        "MDWF direction-independent basis generator index must be in [0, 7], got ",
        generator_index));
}

template<class floatT>
std::array<SU3<floatT>, MDWFDirectionIndependentBasisSize>
mdwfDirectionIndependentBasis() {
    std::array<SU3<floatT>, MDWFDirectionIndependentBasisSize> basis;
    for (size_t generator = 0;
         generator < MDWFDirectionIndependentBasisSize;
         generator++) {
        basis[generator]
            = mdwfDirectionIndependentBasisGenerator<floatT>(generator);
    }
    return basis;
}

using MDWFDirectionIndependentGramMatrix
    = std::array<
        std::array<double, MDWFDirectionIndependentBasisSize>,
        MDWFDirectionIndependentBasisSize>;

template<class floatT>
MDWFDirectionIndependentGramMatrix mdwfMeasureDirectionIndependentGramMatrix(
    const std::array<SU3<floatT>, MDWFDirectionIndependentBasisSize> &basis) {

    MDWFDirectionIndependentGramMatrix gram{};
    for (size_t row = 0; row < MDWFDirectionIndependentBasisSize; row++) {
        for (size_t column = 0;
             column < MDWFDirectionIndependentBasisSize;
             column++) {
            gram[row][column]
                = static_cast<double>(real(tr_c(basis[row], basis[column])));
        }
    }
    return gram;
}

template<class floatT>
struct MDWFDirectionIndependentReconstruction {
    SU3<floatT> matrix;
    std::array<double, MDWFDirectionIndependentBasisSize> coefficients;
    double minimum_absolute_pivot;
};

template<class floatT>
MDWFDirectionIndependentReconstruction<floatT>
mdwfReconstructDirectionIndependentMatrix(
    const std::array<SU3<floatT>, MDWFDirectionIndependentBasisSize> &basis,
    const MDWFDirectionIndependentGramMatrix &gram,
    const std::array<double, MDWFDirectionIndependentBasisSize> &contractions) {

    std::array<
        std::array<double, MDWFDirectionIndependentBasisSize + 1>,
        MDWFDirectionIndependentBasisSize> augmented{};

    for (size_t row = 0; row < MDWFDirectionIndependentBasisSize; row++) {
        for (size_t column = 0;
             column < MDWFDirectionIndependentBasisSize;
             column++) {
            augmented[row][column] = gram[row][column];
        }
        augmented[row][MDWFDirectionIndependentBasisSize] = contractions[row];
    }

    double minimum_absolute_pivot = 0.0;
    for (size_t column = 0;
         column < MDWFDirectionIndependentBasisSize;
         column++) {
        size_t pivot_row = column;
        double pivot_abs = std::abs(augmented[pivot_row][column]);
        for (size_t row = column + 1;
             row < MDWFDirectionIndependentBasisSize;
             row++) {
            const double candidate_abs = std::abs(augmented[row][column]);
            if (candidate_abs > pivot_abs) {
                pivot_abs = candidate_abs;
                pivot_row = row;
            }
        }

        if (!std::isfinite(pivot_abs) || pivot_abs <= 1e-14) {
            throw std::runtime_error(stdLogger.fatal(
                "MDWF direction-independent Gram system is singular at column ",
                column, " with pivot ", pivot_abs));
        }
        if (pivot_row != column) {
            std::swap(augmented[pivot_row], augmented[column]);
        }
        minimum_absolute_pivot
            = column == 0
              ? pivot_abs
              : std::min(minimum_absolute_pivot, pivot_abs);

        const double pivot = augmented[column][column];
        for (size_t entry = column;
             entry <= MDWFDirectionIndependentBasisSize;
             entry++) {
            augmented[column][entry] /= pivot;
        }

        for (size_t row = 0; row < MDWFDirectionIndependentBasisSize; row++) {
            if (row == column) {
                continue;
            }
            const double factor = augmented[row][column];
            for (size_t entry = column;
                 entry <= MDWFDirectionIndependentBasisSize;
                 entry++) {
                augmented[row][entry] -= factor * augmented[column][entry];
            }
        }
    }

    std::array<double, MDWFDirectionIndependentBasisSize> coefficients{};
    SU3<floatT> matrix = su3_zero<floatT>();
    for (size_t row = 0; row < MDWFDirectionIndependentBasisSize; row++) {
        coefficients[row]
            = augmented[row][MDWFDirectionIndependentBasisSize];
        matrix += static_cast<floatT>(coefficients[row]) * basis[row];
    }

    return {
        matrix,
        coefficients,
        minimum_absolute_pivot
    };
}

template<class floatT, CompressionType comp>
struct MDWFSingleLinkExplicitDirectionPerturbation {
    SU3Accessor<floatT, comp> gauge_in;
    MDWFFiniteDifferenceProbe<floatT> probe;
    SU3<floatT> direction;
    int sign;

    MDWFSingleLinkExplicitDirectionPerturbation(
        const SU3Accessor<floatT, comp> &gauge_in_in,
        const MDWFFiniteDifferenceProbe<floatT> &probe_in,
        const SU3<floatT> &direction_in,
        int sign_in)
        : gauge_in(gauge_in_in),
          probe(probe_in),
          direction(direction_in),
          sign(sign_in) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu site_mu) {
        const SU3<floatT> link = gauge_in.getLink(site_mu);

        if (site_mu.coord.x == probe.x
            && site_mu.coord.y == probe.y
            && site_mu.coord.z == probe.z
            && site_mu.coord.t == probe.t
            && site_mu.mu == probe.mu) {
            const floatT signed_epsilon
                = static_cast<floatT>(sign) * probe.epsilon;
            const SU3<floatT> perturbation
                = su3_exp(signed_epsilon * direction);

            if (probe.multiplication_side
                == MDWFFiniteDifferenceMultiplicationSide::Right) {
                return link * perturbation;
            }
            return perturbation * link;
        }

        return link;
    }
};

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void applyMDWFSingleLinkExplicitDirectionPerturbation(
    Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_out,
    const Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_in,
    const MDWFFiniteDifferenceProbe<floatT> &probe,
    const SU3<floatT> &direction,
    int sign) {

    const double direction_norm
        = -static_cast<double>(real(tr_c(direction, direction)));
    const double anti_hermitian_violation
        = static_cast<double>(infnorm(direction + dagger(direction)));
    const double trace_violation
        = static_cast<double>(abs(tr_c(direction)));

    if (probe.epsilon <= static_cast<floatT>(0.0)
        || !std::isfinite(static_cast<double>(probe.epsilon))) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF explicit-direction probe requires positive finite epsilon"));
    }
    if (probe.mu >= 4) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF explicit-direction probe requires mu in [0, 3], got ",
            static_cast<int>(probe.mu)));
    }
    if (!std::isfinite(direction_norm) || direction_norm <= 0.0
        || anti_hermitian_violation > 1e-12
        || trace_violation > 1e-12) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF explicit direction must be finite, nonzero, anti-Hermitian, "
            "and traceless: norm = ", direction_norm,
            ", antiHermitianViolation = ", anti_hermitian_violation,
            ", traceViolation = ", trace_violation));
    }
    if (sign != 1 && sign != -1) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF explicit-direction perturbation sign must be +1 or -1"));
    }

    gauge_out.template iterateOverBulkAllMu<>(
        MDWFSingleLinkExplicitDirectionPerturbation<floatT, comp>(
            gauge_in.getAccessor(), probe, direction, sign));
    gauge_out.updateAll();
}

template<class floatT,
         bool onDevice,
         size_t HaloDepth,
         CompressionType comp,
         class ActionEvaluator>
MDWFFiniteDifferenceResult<floatT>
evaluateMDWFSingleLinkExplicitDirectionFiniteDifference(
    Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_plus,
    Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_minus,
    const Gaugefield<floatT, onDevice, HaloDepth, comp> &base_gauge,
    const MDWFFiniteDifferenceProbe<floatT> &probe,
    const SU3<floatT> &direction,
    ActionEvaluator &action_evaluator) {

    applyMDWFSingleLinkExplicitDirectionPerturbation(
        gauge_plus, base_gauge, probe, direction, 1);
    applyMDWFSingleLinkExplicitDirectionPerturbation(
        gauge_minus, base_gauge, probe, direction, -1);

    const MDWFFiniteDifferenceActionValue<floatT> plus
        = action_evaluator(gauge_plus);
    const MDWFFiniteDifferenceActionValue<floatT> minus
        = action_evaluator(gauge_minus);
    const double derivative
        = (plus.action_real - minus.action_real)
          / (2.0 * static_cast<double>(probe.epsilon));
    const double plus_scale = std::max(1.0, std::abs(plus.action_real));
    const double minus_scale = std::max(1.0, std::abs(minus.action_real));

    return {
        plus,
        minus,
        derivative,
        std::max(std::abs(plus.action_imag) / plus_scale,
                 std::abs(minus.action_imag) / minus_scale),
        std::max(plus.max_shifted_residual, minus.max_shifted_residual),
        plus.converged && minus.converged
    };
}
