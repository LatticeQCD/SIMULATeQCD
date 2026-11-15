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
#include "MDWFStackGrid.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

// One reduction over all Ls stacks for the coupled 5D dot products (instead of one per stack): on unless
// MDWF_SINGLE_REDUCTION=0.
inline bool &mdwfSingleReduction() {
    static bool on = [] {
        const char *env = std::getenv("MDWF_SINGLE_REDUCTION");
        return env == nullptr || std::string(env) != "0";
    }();
    return on;
}

template<class floatT, size_t elems>
struct MDWFDot5DFunctor {
    VectArrayAcc<floatT, elems> _a, _b;

    template<class Spinor>
    MDWFDot5DFunctor(const Spinor &a, const Spinor &b) : _a(a.getAccessor()), _b(b.getAccessor()) {}

    __device__ __host__ COMPLEX(double) operator()(gSiteStack &site) {
        return _a.template getElement<double>(site) * _b.template getElement<double>(site);
    }
};

// Operator::readsInputHalo if declared, else true.
template<class Op>
constexpr auto mdwfOperatorReadsInputHalo(int) -> decltype(Op::readsInputHalo, bool()) {
    return Op::readsInputHalo;
}
template<class Op>
constexpr bool mdwfOperatorReadsInputHalo(long) {
    return true;
}

// Halo update of a solver's operator input: adapter.prepareInput(v) if the adapter has it, else mdwfUpdateHalo(v).
template<class Adapter, class Spinor>
auto mdwfPrepareInputImpl(Adapter &adapter, Spinor &v, int) -> decltype(adapter.prepareInput(v), void()) {
    adapter.prepareInput(v);
}
template<class Adapter, class Spinor>
void mdwfPrepareInputImpl(Adapter &, Spinor &v, long) {
    mdwfUpdateHalo(v);
}
template<class Adapter, class Spinor>
void mdwfPrepareInput(Adapter &adapter, Spinor &v) {
    mdwfPrepareInputImpl(adapter, v, 0);
}

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
    std::shared_ptr<LatticeContainer<true, COMPLEX(double)>> _reduction;

public:
    explicit MDWFCoupledSolverAdapter(Operator &linear_operator)
        : _linear_operator(linear_operator) {}

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        _linear_operator.apply(spinor_out, spinor_in, update);
    }

    // Halo update of an operator input before apply(), skipped for operators that declare readsInputHalo = false
    // (they never read the input's halo, e.g. the Schur normal operator: one exchange less per CG iteration).
    void prepareInput(Spinor &spinor_in) {
        if constexpr (mdwfOperatorReadsInputHalo<Operator>(0)) {
            mdwfUpdateHalo(spinor_in);
        }
    }

    COMPLEX(double) dotProduct5D(Spinor &left, const Spinor &right) {
        if (mdwfSingleReduction()) {
            using T = MDWFSpinorTraits<Spinor>;
            if (!_reduction) {
                _reduction = std::make_shared<LatticeContainer<true, COMPLEX(double)>>(left.getComm());
            }
            const size_t n = left.getNumberElements();
            _reduction->adjustSize(n);
            if (mdwfStackGrid()) {
                mdwfStackGridLaunch(*_reduction, MDWFDot5DFunctor<typename T::floatT, T::elems>(left, right),
                                    MDWFStackGridSite<T::layout, T::halo>{}, left.getNumberLatticePoints(), T::stacks);
            } else {
                _reduction->template iterateOverBulkStacked<T::layout, T::halo, T::stacks>(
                    MDWFDot5DFunctor<typename T::floatT, T::elems>(left, right));
            }
            COMPLEX(double) result = 0.0;
            _reduction->reduce(result, n);
            return result;
        }
        std::vector<COMPLEX(double)> stack_dots = left.dotProductStacked(right);

        COMPLEX(double) result = 0.0;
        for (size_t stack = 0; stack < Ls; stack++) {
            result += stack_dots[stack];
        }
        return result;
    }

    double realDotProduct5D(Spinor &left, const Spinor &right) {
        if (mdwfSingleReduction()) {
            return real<double>(dotProduct5D(left, right));
        }
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

    // Real part of the sum over all (site, stack) of op(site), op launched on the stack grid over the sites of
    // `like` and written into the same reduction container, with the same CubReduce as dotProduct5D: an op that
    // returns the per-element value of MDWFDot5DFunctor gives a result bit-identical to dotProduct5D. Used to fuse a
    // vector update with the norm of its result (needs the stack grid and the single reduction).
    template<class Functor>
    double realReduce5D(Spinor &like, Functor op) {
        using T = MDWFSpinorTraits<Spinor>;
        if (!_reduction) {
            _reduction = std::make_shared<LatticeContainer<true, COMPLEX(double)>>(like.getComm());
        }
        const size_t n = like.getNumberElements();
        _reduction->adjustSize(n);
        mdwfStackGridLaunch(*_reduction, op, MDWFStackGridSite<T::layout, T::halo>{}, like.getNumberLatticePoints(),
                            T::stacks);
        COMPLEX(double) result = 0.0;
        _reduction->reduce(result, n);
        return real<double>(result);
    }

    void invert(Spinor &, const Spinor &, int, double) = delete;
};
