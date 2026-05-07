/*
 * Non-solving MDWF linear-operator scaffold.
 *
 * This wrapper owns the MDWF operator temporaries and presents a named apply()
 * entry point for the coupled 5D operator.  It intentionally does not inherit
 * from LinearOperator<Spinor> yet: existing CG/MRHS code interprets stacks as
 * independent right-hand sides, while MDWF uses stacks as the coupled fifth
 * dimension Ls.
 */

#pragma once

#include "MDWFOperator.h"

#include <string>

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
class MDWFLinearOperator {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;
    using Gauge = Gaugefield<floatT, true, HaloDepthGauge, R18>;

private:
    MDWFFifthDimCoefficients<floatT> _fifth_coeff;
    floatT _mass;
    floatT _csw;
    MDWFOperatorWorkspace<floatT, HaloDepthGauge, HaloDepthSpin, Ls> _workspace;

public:
    MDWFLinearOperator(Gauge &gauge,
                       MDWFFifthDimCoefficients<floatT> fifth_coeff,
                       floatT mass,
                       floatT csw = 0.0,
                       std::string name = "MDWF_linear_operator")
        : _fifth_coeff(fifth_coeff),
          _mass(mass),
          _csw(csw),
          _workspace(gauge, name + "_workspace") {}

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        _workspace.applyClover(spinor_out, spinor_in, _fifth_coeff, _mass, _csw, update);
    }

    void applyMdaggM(Spinor &, const Spinor &, bool = true) = delete;

    void setFifthDimCoefficients(MDWFFifthDimCoefficients<floatT> fifth_coeff) {
        _fifth_coeff = fifth_coeff;
    }

    void setMass(floatT mass) {
        _mass = mass;
    }

    void setCsw(floatT csw) {
        _csw = csw;
    }

    floatT mass() const {
        return _mass;
    }

    floatT csw() const {
        return _csw;
    }
};
