/*
 * Minimal coupled-5D MDWF CG scaffold.
 *
 * This is intentionally isolated from the existing inverter module.  It uses
 * MDWFCoupledSolverAdapter vector-space operations, where Ls stacks are summed
 * into one coupled 5D inner product rather than treated as independent RHS.
 */

#pragma once

#include "MDWFCoupledSolverAdapter.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

template<class floatT>
struct MDWFCoupledCGResult {
    int iterations;
    double residue;
    bool converged;
};

template<class floatT, class CoupledAdapter, size_t BlockSize = 32>
class MDWFCoupledCG {
public:
    using Spinor = typename CoupledAdapter::Spinor;

    MDWFCoupledCGResult<floatT> invert(CoupledAdapter &adapter,
                                       Spinor &spinor_out,
                                       const Spinor &spinor_in,
                                       int max_iter,
                                       double precision,
                                       bool update = true) const {
        Spinor residual(spinor_in.getComm(), "MDWF_coupled_cg_residual");
        Spinor search(spinor_in.getComm(), "MDWF_coupled_cg_search");
        Spinor operator_search(spinor_in.getComm(), "MDWF_coupled_cg_operator_search");

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

            const double denominator = real<double>(adapter.dotProduct5D(search, operator_search));
            if (!std::isfinite(denominator) || std::abs(denominator) <= 0.0) {
                throw std::runtime_error(stdLogger.fatal("MDWF coupled CG encountered zero search denominator"));
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
