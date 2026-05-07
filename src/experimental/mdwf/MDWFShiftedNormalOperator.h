/*
 * Explicit shifted MDWF normal-operator scaffold.
 *
 * This wrapper applies
 *
 *     (M^\dagger M + sigma) x
 *
 * using an already supplied normal operator.  It is intended for isolated
 * solver and future rational-approximation tests only; it does not define a
 * multishift solver and does not hook into RHMC/HMC.
 */

#pragma once

#include <string>

template<class NormalOperatorT, class floatT, size_t BlockSize = 64>
class MDWFShiftedNormalOperator {
public:
    using Spinor = typename NormalOperatorT::Spinor;

private:
    NormalOperatorT &_normal;
    floatT _sigma;

public:
    explicit MDWFShiftedNormalOperator(NormalOperatorT &normal,
                                       floatT sigma,
                                       std::string = "MDWF_shifted_normal_operator")
        : _normal(normal),
          _sigma(sigma) {}

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        _normal.apply(spinor_out, spinor_in, false);
        spinor_out.template axpyThisB<BlockSize>(_sigma, spinor_in);

        if (update) {
            spinor_out.updateAll();
        }
    }

    floatT sigma() const {
        return _sigma;
    }
};
