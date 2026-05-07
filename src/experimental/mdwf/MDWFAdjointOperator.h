/*
 * Explicit adjoint for the current MDWF operator scaffold.
 *
 * This header defines a separately supplied M^\dagger for use in the normal
 * form N = M^\dagger M.  The 4D Wilson/clover part uses gamma5 hermiticity,
 * while the fifth-direction part is the explicit transpose-adjoint of the
 * current MDWFFifthDimCoupling coefficients and projectors.
 */

#pragma once

#include "MDWFOperator.h"

#include <string>

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t Ls>
void applyMDWFGamma5(MDWFSpinor<floatT, onDevice, LatLayout, HaloDepth, Ls> &spinor_out,
                     const MDWFSpinor<floatT, onDevice, LatLayout, HaloDepth, Ls> &spinor_in,
                     bool update = false) {
    spinor_out.template iterateOverBulk<BLOCKSIZE>(
        gamma5<floatT, LatLayout, HaloDepth, Ls>(spinor_in));

    if (update) {
        spinor_out.updateAll();
    }
}

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct MDWFFifthDimAdjointCoupling {
    static_assert(Ls > 1, "MDWF fifth-direction adjoint coupling requires Ls > 1");

    Vect12ArrayAcc<floatT> spinor_in;
    MDWFFifthDimCoefficients<floatT> coeff;

    typedef GIndexer<LatLayout, HaloDepth> GInd;

    template<bool onDevice>
    MDWFFifthDimAdjointCoupling(const MDWFSpinor<floatT, onDevice, LatLayout, HaloDepth, Ls> &spinor_in_in,
                                MDWFFifthDimCoefficients<floatT> coeff_in)
        : spinor_in(spinor_in_in.getAccessor()),
          coeff(coeff_in) {}

    __device__ __host__ Vect12<floatT> operator()(gSiteStack site) {
        const size_t s = site.stack;
        const bool receives_forward_boundary = (s == 0);
        const bool receives_backward_boundary = (s + 1 == Ls);
        const size_t s_backward = receives_forward_boundary ? Ls - 1 : s - 1;
        const size_t s_forward = receives_backward_boundary ? 0 : s + 1;

        const floatT from_backward_coeff = receives_forward_boundary ? coeff.forward_boundary : coeff.forward_hop;
        const floatT from_forward_coeff = receives_backward_boundary ? coeff.backward_boundary : coeff.backward_hop;

        Vect12<floatT> out = coeff.diagonal * spinor_in.getElement(site);
        out += from_backward_coeff * mdwfProjectMinus(spinor_in.getElement(GInd::getSiteStack(site, s_backward)));
        out += from_forward_coeff * mdwfProjectPlus(spinor_in.getElement(GInd::getSiteStack(site, s_forward)));

        return out;
    }
};

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t Ls>
void applyMDWFFifthDimAdjointCoupling(MDWFSpinor<floatT, onDevice, LatLayout, HaloDepth, Ls> &spinor_out,
                                      const MDWFSpinor<floatT, onDevice, LatLayout, HaloDepth, Ls> &spinor_in,
                                      MDWFFifthDimCoefficients<floatT> coeff,
                                      bool update = false) {
    spinor_out.template iterateOverBulk<>(
        MDWFFifthDimAdjointCoupling<floatT, LatLayout, HaloDepth, Ls>(spinor_in, coeff));

    if (update) {
        spinor_out.updateAll();
    }
}

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
void applyMDWFAdjointCloverOperator(MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_out,
                                    Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
                                    MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &gamma5_in,
                                    MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &wilson_gamma5_out,
                                    MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &wilson_part,
                                    MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &fifth_part,
                                    MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &wilson_tmp,
                                    Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_upper,
                                    Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_lower,
                                    Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_inv_upper,
                                    Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_inv_lower,
                                    const MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_in,
                                    MDWFFifthDimCoefficients<floatT> fifth_coeff,
                                    floatT mass,
                                    floatT csw = 0.0,
                                    bool update = false) {
    applyMDWFGamma5<floatT, true, All, HaloDepthSpin, Ls>(gamma5_in, spinor_in, true);

    applyMDWFCloverWilsonSlice<floatT, HaloDepthGauge, HaloDepthSpin, Ls>(
        wilson_gamma5_out, gauge, wilson_tmp, fmunu_upper, fmunu_lower,
        fmunu_inv_upper, fmunu_inv_lower, gamma5_in, mass, csw, false);

    applyMDWFGamma5<floatT, true, All, HaloDepthSpin, Ls>(wilson_part, wilson_gamma5_out, false);

    applyMDWFFifthDimAdjointCoupling<floatT, true, All, HaloDepthSpin, Ls>(
        fifth_part, spinor_in, fifth_coeff, false);

    spinor_out = wilson_part;
    spinor_out += fifth_part;

    if (update) {
        spinor_out.updateAll();
    }
}

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
class MDWFAdjointOperatorWorkspace {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;
    using CloverField = Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1>;

private:
    Gaugefield<floatT, true, HaloDepthGauge, R18> &_gauge;
    Spinor _gamma5_in;
    Spinor _wilson_gamma5_out;
    Spinor _wilson_part;
    Spinor _fifth_part;
    Spinor _wilson_tmp;
    CloverField _fmunu_upper;
    CloverField _fmunu_lower;
    CloverField _fmunu_inv_upper;
    CloverField _fmunu_inv_lower;

public:
    explicit MDWFAdjointOperatorWorkspace(Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
                                          std::string name = "MDWF_adjoint_operator_workspace")
        : _gauge(gauge),
          _gamma5_in(gauge.getComm(), name + "_gamma5_in"),
          _wilson_gamma5_out(gauge.getComm(), name + "_wilson_gamma5_out"),
          _wilson_part(gauge.getComm(), name + "_wilson_part"),
          _fifth_part(gauge.getComm(), name + "_fifth_part"),
          _wilson_tmp(gauge.getComm(), name + "_wilson_tmp"),
          _fmunu_upper(gauge.getComm(), name + "_fmunu_upper"),
          _fmunu_lower(gauge.getComm(), name + "_fmunu_lower"),
          _fmunu_inv_upper(gauge.getComm(), name + "_fmunu_inv_upper"),
          _fmunu_inv_lower(gauge.getComm(), name + "_fmunu_inv_lower") {}

    void applyClover(Spinor &spinor_out,
                     const Spinor &spinor_in,
                     MDWFFifthDimCoefficients<floatT> fifth_coeff,
                     floatT mass,
                     floatT csw = 0.0,
                     bool update = false) {
        applyMDWFAdjointCloverOperator<floatT, HaloDepthGauge, HaloDepthSpin, Ls>(
            spinor_out, _gauge, _gamma5_in, _wilson_gamma5_out, _wilson_part, _fifth_part,
            _wilson_tmp, _fmunu_upper, _fmunu_lower, _fmunu_inv_upper, _fmunu_inv_lower,
            spinor_in, fifth_coeff, mass, csw, update);
    }
};

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
class MDWFAdjointLinearOperator {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;
    using Gauge = Gaugefield<floatT, true, HaloDepthGauge, R18>;

private:
    MDWFFifthDimCoefficients<floatT> _fifth_coeff;
    floatT _mass;
    floatT _csw;
    MDWFAdjointOperatorWorkspace<floatT, HaloDepthGauge, HaloDepthSpin, Ls> _workspace;

public:
    MDWFAdjointLinearOperator(Gauge &gauge,
                              MDWFFifthDimCoefficients<floatT> fifth_coeff,
                              floatT mass,
                              floatT csw = 0.0,
                              std::string name = "MDWF_adjoint_linear_operator")
        : _fifth_coeff(fifth_coeff),
          _mass(mass),
          _csw(csw),
          _workspace(gauge, name + "_workspace") {}

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        _workspace.applyClover(spinor_out, spinor_in, _fifth_coeff, _mass, _csw, update);
    }

    floatT mass() const {
        return _mass;
    }

    floatT csw() const {
        return _csw;
    }
};
