/*
 * Coupled-5D MDWF solver-adapter scaffold.
 *
 * This adapter defines the vector-space operations a future MDWF Krylov solver
 * should use when Spinorfield stacks represent the physical fifth dimension Ls.
 * It intentionally aggregates stacked dot products into one coupled 5D scalar
 * and does not call, inherit from, or plug into the existing multi-RHS CG code.
 */

#pragma once

#include "MDWFLinearOperator.h"

#include <vector>

template<class floatT,
         size_t HaloDepthGauge,
         size_t HaloDepthSpin,
         size_t Ls,
         class LinearOperatorT = MDWFLinearOperator<floatT, HaloDepthGauge, HaloDepthSpin, Ls>>
class MDWFCoupledSolverAdapter {
public:
    using Operator = LinearOperatorT;
    using Spinor = typename Operator::Spinor;

private:
    Operator &_linear_operator;

public:
    explicit MDWFCoupledSolverAdapter(Operator &linear_operator)
        : _linear_operator(linear_operator) {}

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        _linear_operator.apply(spinor_out, spinor_in, update);
    }

    COMPLEX(double) dotProduct5D(Spinor &left, const Spinor &right) {
        std::vector<COMPLEX(double)> stack_dots = left.dotProductStacked(right);

        COMPLEX(double) result = 0.0;
        for (size_t stack = 0; stack < Ls; stack++) {
            result += stack_dots[stack];
        }
        return result;
    }

    double realDotProduct5D(Spinor &left, const Spinor &right) {
        std::vector<COMPLEX(double)> stack_dots = left.dotProductStacked(right);

        double result = 0.0;
        for (size_t stack = 0; stack < Ls; stack++) {
            result += real<double>(stack_dots[stack]);
        }
        return result;
    }

    double norm2(Spinor &spinor) {
        return realDotProduct5D(spinor, spinor);
    }

    void invert(Spinor &, const Spinor &, int, double) = delete;
};
