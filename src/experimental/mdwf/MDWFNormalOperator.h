/*
 * Explicit MDWF normal-operator scaffold.
 *
 * For a coupled 5D MDWF operator M, the Hermitian positive-definite CG
 * operator must be a normal form
 *
 *     N = M^\dagger M
 *
 * with the coupled 5D inner product used by MDWFCoupledSolverAdapter.  This
 * wrapper does not define the MDWF adjoint.  It only composes two explicitly
 * supplied operators, a forward operator M and an adjoint operator M^\dagger.
 * The resulting operator is HPD only if the supplied adjoint is mathematically
 * correct and M has no null vectors on the solved subspace.
 */

#pragma once

#include <string>

template<class ForwardOperatorT, class AdjointOperatorT>
class MDWFNormalOperator {
public:
    using Spinor = typename ForwardOperatorT::Spinor;

private:
    ForwardOperatorT &_forward;
    AdjointOperatorT &_adjoint;
    Spinor _tmp;

public:
    MDWFNormalOperator(CommunicationBase &commBase,
                       ForwardOperatorT &forward,
                       AdjointOperatorT &adjoint,
                       std::string name = "MDWF_normal_operator")
        : _forward(forward),
          _adjoint(adjoint),
          _tmp(commBase, name + "_tmp") {}

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        _forward.apply(_tmp, spinor_in, true);
        _adjoint.apply(spinor_out, _tmp, update);
    }
};
