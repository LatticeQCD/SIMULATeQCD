/*
 * Minimal coupled-5D MDWF multishift-CG scaffold.
 *
 * This is intentionally isolated from the existing inverter and RHMC modules.
 * It exposes a multishift interface for systems
 *
 *     (A + sigma_i) x_i = b
 *
 * while preserving the coupled 5D inner product where Spinorfield stacks are
 * the physical fifth dimension Ls.  This first scaffold is correctness-first:
 * it solves each shift independently with the coupled-5D CG recurrence.  A
 * later optimization may replace the internals with a simultaneous multishift
 * recurrence after the interface and mock-SPD tests are stable.
 */

#pragma once

#include "MDWFCoupledSolverAdapter.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

template<class floatT>
struct MDWFCoupledMultiShiftCGResult {
    int iterations;
    double residue;
    bool converged;
};

template<class floatT>
struct MDWFCoupledMultiShiftCGResults {
    std::vector<MDWFCoupledMultiShiftCGResult<floatT>> shifts;

    bool converged() const {
        for (const auto &result : shifts) {
            if (!result.converged) {
                return false;
            }
        }
        return true;
    }
};

template<class floatT, class CoupledAdapter, size_t BlockSize = 64>
class MDWFCoupledMultiShiftCG {
public:
    using Spinor = typename CoupledAdapter::Spinor;

    MDWFCoupledMultiShiftCGResults<floatT> invert(CoupledAdapter &adapter,
                                                  const std::vector<Spinor *> &spinor_out,
                                                  const Spinor &spinor_in,
                                                  const std::vector<floatT> &sigma,
                                                  int max_iter,
                                                  double precision,
                                                  bool update = true) const {
        if (spinor_out.size() != sigma.size() || sigma.empty()) {
            throw std::runtime_error(stdLogger.fatal(
                "MDWF coupled multishift CG requires matching nonempty solution and shift vectors"));
        }

        MDWFCoupledMultiShiftCGResults<floatT> results;
        results.shifts.reserve(sigma.size());

        for (size_t shift = 0; shift < sigma.size(); shift++) {
            if (spinor_out[shift] == nullptr) {
                throw std::runtime_error(stdLogger.fatal("MDWF coupled multishift CG received a null solution pointer"));
            }
            if (!std::isfinite(static_cast<double>(sigma[shift])) || sigma[shift] < static_cast<floatT>(0.0)) {
                throw std::runtime_error(stdLogger.fatal(
                    "MDWF coupled multishift CG requires finite nonnegative shifts, got sigma = ", sigma[shift]));
            }

            results.shifts.push_back(invertShift(
                adapter, *spinor_out[shift], spinor_in, sigma[shift], max_iter, precision, update));
        }

        return results;
    }

private:
    MDWFCoupledMultiShiftCGResult<floatT> invertShift(CoupledAdapter &adapter,
                                                     Spinor &spinor_out,
                                                     const Spinor &spinor_in,
                                                     floatT sigma,
                                                     int max_iter,
                                                     double precision,
                                                     bool update) const {
        Spinor residual(spinor_in.getComm(), "MDWF_coupled_multishift_cg_residual");
        Spinor search(spinor_in.getComm(), "MDWF_coupled_multishift_cg_search");
        Spinor operator_search(spinor_in.getComm(), "MDWF_coupled_multishift_cg_operator_search");

        spinor_out = static_cast<floatT>(0.0) * spinor_in;
        residual = spinor_in;
        search = residual;

        const double source_norm = adapter.norm2(residual);
        const double target_norm = precision * precision * std::max(source_norm, 1.0);
        double residual_norm = source_norm;

        if (residual_norm <= target_norm) {
            if (update) {
                spinor_out.updateAll();
            }
            return {0, std::sqrt(residual_norm / std::max(source_norm, 1.0)), true};
        }

        for (int iteration = 0; iteration < max_iter; iteration++) {
            search.updateAll();
            adapter.apply(operator_search, search, false);
            operator_search.template axpyThisB<BlockSize>(sigma, search);

            const double denominator = real<double>(adapter.dotProduct5D(search, operator_search));
            if (!std::isfinite(denominator) || denominator <= 0.0) {
                throw std::runtime_error(stdLogger.fatal(
                    "MDWF coupled multishift CG encountered a nonpositive search denominator"));
            }

            const floatT alpha = static_cast<floatT>(residual_norm / denominator);
            spinor_out.template axpyThisB<BlockSize>(alpha, search);
            residual.template axpyThisB<BlockSize>(-alpha, operator_search);

            const double next_residual_norm = adapter.norm2(residual);
            if (next_residual_norm <= target_norm) {
                if (update) {
                    spinor_out.updateAll();
                }
                return {iteration + 1,
                        std::sqrt(next_residual_norm / std::max(source_norm, 1.0)),
                        true};
            }

            const floatT beta = static_cast<floatT>(next_residual_norm / residual_norm);
            search *= COMPLEX(floatT)(beta, 0.0);
            search += residual;
            residual_norm = next_residual_norm;
        }

        if (update) {
            spinor_out.updateAll();
        }
        return {max_iter, std::sqrt(residual_norm / std::max(source_norm, 1.0)), false};
    }
};
