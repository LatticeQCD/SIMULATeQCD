/*
 * Test-only general-Mobius (RBC/UKQCD convention) physical parameter mapping
 * and forward/adjoint operator composition, following
 * PHYSICAL_OPERATOR_MAPPING.md Section 2.3.
 *
 * This is plain (not zMobius) Mobius: constant b5, c5 per slice, with the
 * project's chosen RBC/UKQCD convention c5 = b5 - 1. It does not implement
 * s-dependent bs[s]/cs[s], clover (csw != 0) for this path, or a production
 * M5/mf/b5 choice. It does not change MDWFFifthDimCoupling,
 * MDWFFifthDimAdjointCoupling, MDWFWilsonSlice, MDWFOperator,
 * MDWFAdjointOperator.h, or MDWFPhysicalMapping.h: the composition below
 * calls them unmodified.
 *
 * The construction, transcribed from Grid's CayleyFermion5D (M5D, Meooe5D,
 * M), is:
 *
 *   Din  = MDWFFifthDimCoupling(psi; diagonal = b5, forward_hop = c5,
 *                                    backward_hop = c5,
 *                                    forward_boundary  = -mf * c5,
 *                                    backward_boundary = -mf * c5)
 *   chi  = MDWFWilsonSlice(Din; mass = mdwfShamirKernelMass(M5), csw = 0)
 *   chi += MDWFFifthDimCoupling(psi; mdwfShamirFifthDimCoefficients(mf))
 *
 * The second coupling call has diagonal = 1, so it already contributes psi
 * once (Grid's separate "+= psi" step, regrouped by associativity into this
 * one call); do not add psi again separately, or it is double-counted.
 *
 * At b5 = 1 (so c5 = 0), Din reduces exactly to psi (forward_hop =
 * backward_hop = 0, boundaries = 0), so this reduces identically to the
 * Shamir case already implemented in MDWFPhysicalMapping.h /
 * mdwfShamirFifthDimMappingTest.
 */

#pragma once

#include "MDWFAdjointOperator.h"
#include "MDWFPhysicalMapping.h"
#include "MDWFWilsonSlice.h"

#include <string>

template<class floatT>
__host__ __device__ inline MDWFFifthDimCoefficients<floatT> mdwfMobiusDinCoefficients(
    floatT b5, floatT c5, floatT mf) {
    return MDWFFifthDimCoefficients<floatT>(
        b5,             // diagonal
        c5,             // forward_hop
        c5,             // backward_hop
        -mf * c5,       // forward_boundary
        -mf * c5);      // backward_boundary
}

template<class floatT>
struct MDWFMobiusOperatorParameters {
    floatT mass;
    floatT b5;
    floatT c5;
    MDWFFifthDimCoefficients<floatT> dinCoeff;
    MDWFFifthDimCoefficients<floatT> shiftCoeff;

    // c5 = b5 - 1 enforces the project's chosen RBC/UKQCD convention
    // (b5 - c5 = 1); it is not a mathematical requirement of the
    // construction itself.
    __host__ __device__ MDWFMobiusOperatorParameters(floatT M5, floatT mf, floatT b5_in)
        : mass(mdwfShamirKernelMass(M5)),
          b5(b5_in),
          c5(b5_in - static_cast<floatT>(1.0)),
          dinCoeff(mdwfMobiusDinCoefficients(b5_in, b5_in - static_cast<floatT>(1.0), mf)),
          shiftCoeff(mdwfShamirFifthDimCoefficients(mf)) {}
};

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
void applyMDWFMobiusOperator(MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_out,
                             Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
                             MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &din,
                             MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &dw_din,
                             MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &wilson_tmp,
                             MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &shift_part,
                             const MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_in,
                             const MDWFMobiusOperatorParameters<floatT> &params,
                             bool update = false) {
    // Din is read spatially (x,y,z,t neighbors) by the Wilson kernel below,
    // so its halo must be refreshed here even though the fifth-direction
    // coupling itself only needs same-site stack neighbors.
    applyMDWFFifthDimCoupling<floatT, true, All, HaloDepthSpin, Ls>(
        din, spinor_in, params.dinCoeff, true);

    applyMDWFWilsonSlice<floatT, HaloDepthGauge, HaloDepthSpin, Ls>(
        dw_din, gauge, wilson_tmp, din, params.mass, static_cast<floatT>(0.0));

    // shiftCoeff has diagonal = 1, so this term already includes the "+psi"
    // identity contribution (Grid's axpby step) together with the shift; do
    // not add spinor_in again separately, or psi would be double-counted.
    applyMDWFFifthDimCoupling<floatT, true, All, HaloDepthSpin, Ls>(
        shift_part, spinor_in, params.shiftCoeff);

    spinor_out = dw_din;
    spinor_out += shift_part;

    if (update) {
        spinor_out.updateAll();
    }
}

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
class MDWFMobiusOperatorWorkspace {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;

private:
    Gaugefield<floatT, true, HaloDepthGauge, R18> &_gauge;
    Spinor _din;
    Spinor _dw_din;
    Spinor _wilson_tmp;
    Spinor _shift_part;

public:
    explicit MDWFMobiusOperatorWorkspace(Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
                                         std::string name = "MDWF_mobius_operator_workspace")
        : _gauge(gauge),
          _din(gauge.getComm(), name + "_din"),
          _dw_din(gauge.getComm(), name + "_dw_din"),
          _wilson_tmp(gauge.getComm(), name + "_wilson_tmp"),
          _shift_part(gauge.getComm(), name + "_shift_part") {}

    void apply(Spinor &spinor_out,
               const Spinor &spinor_in,
               const MDWFMobiusOperatorParameters<floatT> &params,
               bool update = false) {
        applyMDWFMobiusOperator<floatT, HaloDepthGauge, HaloDepthSpin, Ls>(
            spinor_out, _gauge, _din, _dw_din, _wilson_tmp, _shift_part,
            spinor_in, params, update);
    }
};

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
class MDWFMobiusLinearOperator {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;
    using Gauge = Gaugefield<floatT, true, HaloDepthGauge, R18>;

private:
    MDWFMobiusOperatorParameters<floatT> _params;
    MDWFMobiusOperatorWorkspace<floatT, HaloDepthGauge, HaloDepthSpin, Ls> _workspace;

public:
    MDWFMobiusLinearOperator(Gauge &gauge, floatT M5, floatT mf, floatT b5,
                             std::string name = "MDWF_mobius_linear_operator")
        : _params(M5, mf, b5),
          _workspace(gauge, name + "_workspace") {}

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        _workspace.apply(spinor_out, spinor_in, _params, update);
    }

    const MDWFMobiusOperatorParameters<floatT> &params() const {
        return _params;
    }

    floatT mass() const {
        return _params.mass;
    }
};

/*
 * General-Mobius adjoint, by the same composition principle as the forward
 * operator: M = (D_W o Din) + Shift, so M^dagger = Din^dagger o D_W^dagger
 * + Shift^dagger. D_W^dagger = gamma5 D_W gamma5 is the standard Wilson
 * gamma5-Hermiticity already used by MDWFAdjointOperator.h; Din^dagger and
 * Shift^dagger both reuse the existing, already-validated
 * MDWFFifthDimAdjointCoupling (the formal transpose-adjoint of
 * MDWFFifthDimCoupling for any five coefficients), applied to the two
 * coefficient sets from MDWFMobiusOperatorParameters. No new adjoint
 * machinery is introduced.
 *
 * At b5 = 1 (c5 = 0), Din^dagger is the identity (same reasoning as the
 * forward operator), so this reduces exactly to the existing Shamir
 * adjoint (MDWFAdjointLinearOperator / applyMDWFAdjointCloverOperator at
 * csw = 0).
 */
template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
void applyMDWFMobiusAdjointOperator(MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_out,
                                    Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
                                    MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &gamma5_in,
                                    MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &wilson_gamma5_out,
                                    MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &wilson_tmp,
                                    MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &dw_dagger,
                                    MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &din_dagger,
                                    MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &shift_dagger,
                                    const MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_in,
                                    const MDWFMobiusOperatorParameters<floatT> &params,
                                    bool update = false) {
    // D_W^dagger(x) = gamma5(D_W(gamma5(x))); gamma5_in needs its halo
    // refreshed before the spatially-neighboring Wilson kernel reads it.
    applyMDWFGamma5<floatT, true, All, HaloDepthSpin, Ls>(gamma5_in, spinor_in, true);
    applyMDWFWilsonSlice<floatT, HaloDepthGauge, HaloDepthSpin, Ls>(
        wilson_gamma5_out, gauge, wilson_tmp, gamma5_in, params.mass, static_cast<floatT>(0.0));
    applyMDWFGamma5<floatT, true, All, HaloDepthSpin, Ls>(dw_dagger, wilson_gamma5_out);

    // (D_W o Din)^dagger = Din^dagger o D_W^dagger. Din^dagger only reads
    // same-site stack neighbors, so no additional halo update is needed here.
    applyMDWFFifthDimAdjointCoupling<floatT, true, All, HaloDepthSpin, Ls>(
        din_dagger, dw_dagger, params.dinCoeff);

    applyMDWFFifthDimAdjointCoupling<floatT, true, All, HaloDepthSpin, Ls>(
        shift_dagger, spinor_in, params.shiftCoeff);

    spinor_out = din_dagger;
    spinor_out += shift_dagger;

    if (update) {
        spinor_out.updateAll();
    }
}

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
class MDWFMobiusAdjointOperatorWorkspace {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;

private:
    Gaugefield<floatT, true, HaloDepthGauge, R18> &_gauge;
    Spinor _gamma5_in;
    Spinor _wilson_gamma5_out;
    Spinor _wilson_tmp;
    Spinor _dw_dagger;
    Spinor _din_dagger;
    Spinor _shift_dagger;

public:
    explicit MDWFMobiusAdjointOperatorWorkspace(Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
                                                std::string name = "MDWF_mobius_adjoint_operator_workspace")
        : _gauge(gauge),
          _gamma5_in(gauge.getComm(), name + "_gamma5_in"),
          _wilson_gamma5_out(gauge.getComm(), name + "_wilson_gamma5_out"),
          _wilson_tmp(gauge.getComm(), name + "_wilson_tmp"),
          _dw_dagger(gauge.getComm(), name + "_dw_dagger"),
          _din_dagger(gauge.getComm(), name + "_din_dagger"),
          _shift_dagger(gauge.getComm(), name + "_shift_dagger") {}

    void apply(Spinor &spinor_out,
               const Spinor &spinor_in,
               const MDWFMobiusOperatorParameters<floatT> &params,
               bool update = false) {
        applyMDWFMobiusAdjointOperator<floatT, HaloDepthGauge, HaloDepthSpin, Ls>(
            spinor_out, _gauge, _gamma5_in, _wilson_gamma5_out, _wilson_tmp,
            _dw_dagger, _din_dagger, _shift_dagger, spinor_in, params, update);
    }
};

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
class MDWFMobiusAdjointLinearOperator {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;
    using Gauge = Gaugefield<floatT, true, HaloDepthGauge, R18>;

private:
    MDWFMobiusOperatorParameters<floatT> _params;
    MDWFMobiusAdjointOperatorWorkspace<floatT, HaloDepthGauge, HaloDepthSpin, Ls> _workspace;

public:
    MDWFMobiusAdjointLinearOperator(Gauge &gauge, floatT M5, floatT mf, floatT b5,
                                    std::string name = "MDWF_mobius_adjoint_linear_operator")
        : _params(M5, mf, b5),
          _workspace(gauge, name + "_workspace") {}

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        _workspace.apply(spinor_out, spinor_in, _params, update);
    }

    const MDWFMobiusOperatorParameters<floatT> &params() const {
        return _params;
    }

    floatT mass() const {
        return _params.mass;
    }
};

/*
 * General-Mobius clover extension, added purely alongside the c_sw = 0
 * classes above: they are untouched, so the already cluster-validated
 * mdwfMobiusFifthDimMappingTest / mdwfMobiusAdjointMappingTest behavior is
 * unaffected.
 *
 * Per the project's clover guardrail ("route clover only through the
 * existing Wilson-kernel path; do not duplicate clover storage or alter
 * MDWF fifth-direction coupling"), the only change from
 * applyMDWFMobiusOperator / applyMDWFMobiusAdjointOperator is that the single
 * D_W(Din) (forward) / D_W^dagger(gamma5_in) (adjoint) step now calls the
 * existing applyMDWFCloverWilsonSlice instead of applyMDWFWilsonSlice, with
 * an explicit csw. Din/Din^dagger, the fifth-direction shift term, and the
 * gamma5-Hermiticity composition are unchanged. At csw = 0,
 * applyMDWFCloverWilsonSlice already reproduces applyMDWFWilsonSlice exactly
 * (the Stage 5/6 mdwfCloverCsw0Test regression, generic in its input), so
 * this is expected to reduce identically to the existing Mobius classes;
 * mdwfMobiusCloverMappingTest checks that expectation explicitly rather than
 * assuming it.
 */
template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
void applyMDWFMobiusCloverOperator(MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_out,
                                   Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
                                   MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &din,
                                   MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &dw_din,
                                   MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &wilson_tmp,
                                   Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_upper,
                                   Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_lower,
                                   Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_inv_upper,
                                   Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_inv_lower,
                                   MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &shift_part,
                                   const MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_in,
                                   const MDWFMobiusOperatorParameters<floatT> &params,
                                   floatT csw = 0.0,
                                   bool update = false) {
    applyMDWFFifthDimCoupling<floatT, true, All, HaloDepthSpin, Ls>(
        din, spinor_in, params.dinCoeff, true);

    applyMDWFCloverWilsonSlice<floatT, HaloDepthGauge, HaloDepthSpin, Ls>(
        dw_din, gauge, wilson_tmp, fmunu_upper, fmunu_lower, fmunu_inv_upper, fmunu_inv_lower,
        din, params.mass, csw);

    applyMDWFFifthDimCoupling<floatT, true, All, HaloDepthSpin, Ls>(
        shift_part, spinor_in, params.shiftCoeff);

    spinor_out = dw_din;
    spinor_out += shift_part;

    if (update) {
        spinor_out.updateAll();
    }
}

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
class MDWFMobiusCloverOperatorWorkspace {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;
    using CloverField = Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1>;

private:
    Gaugefield<floatT, true, HaloDepthGauge, R18> &_gauge;
    Spinor _din;
    Spinor _dw_din;
    Spinor _wilson_tmp;
    Spinor _shift_part;
    CloverField _fmunu_upper;
    CloverField _fmunu_lower;
    CloverField _fmunu_inv_upper;
    CloverField _fmunu_inv_lower;

public:
    explicit MDWFMobiusCloverOperatorWorkspace(Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
                                               std::string name = "MDWF_mobius_clover_operator_workspace")
        : _gauge(gauge),
          _din(gauge.getComm(), name + "_din"),
          _dw_din(gauge.getComm(), name + "_dw_din"),
          _wilson_tmp(gauge.getComm(), name + "_wilson_tmp"),
          _shift_part(gauge.getComm(), name + "_shift_part"),
          _fmunu_upper(gauge.getComm(), name + "_fmunu_upper"),
          _fmunu_lower(gauge.getComm(), name + "_fmunu_lower"),
          _fmunu_inv_upper(gauge.getComm(), name + "_fmunu_inv_upper"),
          _fmunu_inv_lower(gauge.getComm(), name + "_fmunu_inv_lower") {}

    void apply(Spinor &spinor_out,
               const Spinor &spinor_in,
               const MDWFMobiusOperatorParameters<floatT> &params,
               floatT csw = 0.0,
               bool update = false) {
        applyMDWFMobiusCloverOperator<floatT, HaloDepthGauge, HaloDepthSpin, Ls>(
            spinor_out, _gauge, _din, _dw_din, _wilson_tmp,
            _fmunu_upper, _fmunu_lower, _fmunu_inv_upper, _fmunu_inv_lower,
            _shift_part, spinor_in, params, csw, update);
    }
};

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
class MDWFMobiusCloverLinearOperator {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;
    using Gauge = Gaugefield<floatT, true, HaloDepthGauge, R18>;

private:
    MDWFMobiusOperatorParameters<floatT> _params;
    floatT _csw;
    MDWFMobiusCloverOperatorWorkspace<floatT, HaloDepthGauge, HaloDepthSpin, Ls> _workspace;

public:
    MDWFMobiusCloverLinearOperator(Gauge &gauge, floatT M5, floatT mf, floatT b5, floatT csw = 0.0,
                                   std::string name = "MDWF_mobius_clover_linear_operator")
        : _params(M5, mf, b5),
          _csw(csw),
          _workspace(gauge, name + "_workspace") {}

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        _workspace.apply(spinor_out, spinor_in, _params, _csw, update);
    }

    const MDWFMobiusOperatorParameters<floatT> &params() const {
        return _params;
    }

    floatT mass() const {
        return _params.mass;
    }

    floatT csw() const {
        return _csw;
    }
};

/*
 * General-Mobius clover adjoint: same composition principle as
 * applyMDWFMobiusAdjointOperator, with D_W^dagger replaced by its
 * clover-capable form (gamma5 . applyMDWFCloverWilsonSlice(csw) . gamma5),
 * matching how MDWFAdjointOperator.h's applyMDWFAdjointCloverOperator
 * extends the Shamir adjoint. Din^dagger and Shift^dagger are unchanged
 * (MDWFFifthDimAdjointCoupling has no csw dependence).
 */
template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
void applyMDWFMobiusAdjointCloverOperator(MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_out,
                                          Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
                                          MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &gamma5_in,
                                          MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &wilson_gamma5_out,
                                          MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &wilson_tmp,
                                          Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_upper,
                                          Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_lower,
                                          Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_inv_upper,
                                          Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> &fmunu_inv_lower,
                                          MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &dw_dagger,
                                          MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &din_dagger,
                                          MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &shift_dagger,
                                          const MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &spinor_in,
                                          const MDWFMobiusOperatorParameters<floatT> &params,
                                          floatT csw = 0.0,
                                          bool update = false) {
    applyMDWFGamma5<floatT, true, All, HaloDepthSpin, Ls>(gamma5_in, spinor_in, true);
    applyMDWFCloverWilsonSlice<floatT, HaloDepthGauge, HaloDepthSpin, Ls>(
        wilson_gamma5_out, gauge, wilson_tmp, fmunu_upper, fmunu_lower, fmunu_inv_upper, fmunu_inv_lower,
        gamma5_in, params.mass, csw);
    applyMDWFGamma5<floatT, true, All, HaloDepthSpin, Ls>(dw_dagger, wilson_gamma5_out);

    applyMDWFFifthDimAdjointCoupling<floatT, true, All, HaloDepthSpin, Ls>(
        din_dagger, dw_dagger, params.dinCoeff);

    applyMDWFFifthDimAdjointCoupling<floatT, true, All, HaloDepthSpin, Ls>(
        shift_dagger, spinor_in, params.shiftCoeff);

    spinor_out = din_dagger;
    spinor_out += shift_dagger;

    if (update) {
        spinor_out.updateAll();
    }
}

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
class MDWFMobiusAdjointCloverOperatorWorkspace {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;
    using CloverField = Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1>;

private:
    Gaugefield<floatT, true, HaloDepthGauge, R18> &_gauge;
    Spinor _gamma5_in;
    Spinor _wilson_gamma5_out;
    Spinor _wilson_tmp;
    Spinor _dw_dagger;
    Spinor _din_dagger;
    Spinor _shift_dagger;
    CloverField _fmunu_upper;
    CloverField _fmunu_lower;
    CloverField _fmunu_inv_upper;
    CloverField _fmunu_inv_lower;

public:
    explicit MDWFMobiusAdjointCloverOperatorWorkspace(
        Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
        std::string name = "MDWF_mobius_adjoint_clover_operator_workspace")
        : _gauge(gauge),
          _gamma5_in(gauge.getComm(), name + "_gamma5_in"),
          _wilson_gamma5_out(gauge.getComm(), name + "_wilson_gamma5_out"),
          _wilson_tmp(gauge.getComm(), name + "_wilson_tmp"),
          _dw_dagger(gauge.getComm(), name + "_dw_dagger"),
          _din_dagger(gauge.getComm(), name + "_din_dagger"),
          _shift_dagger(gauge.getComm(), name + "_shift_dagger"),
          _fmunu_upper(gauge.getComm(), name + "_fmunu_upper"),
          _fmunu_lower(gauge.getComm(), name + "_fmunu_lower"),
          _fmunu_inv_upper(gauge.getComm(), name + "_fmunu_inv_upper"),
          _fmunu_inv_lower(gauge.getComm(), name + "_fmunu_inv_lower") {}

    void apply(Spinor &spinor_out,
               const Spinor &spinor_in,
               const MDWFMobiusOperatorParameters<floatT> &params,
               floatT csw = 0.0,
               bool update = false) {
        applyMDWFMobiusAdjointCloverOperator<floatT, HaloDepthGauge, HaloDepthSpin, Ls>(
            spinor_out, _gauge, _gamma5_in, _wilson_gamma5_out, _wilson_tmp,
            _fmunu_upper, _fmunu_lower, _fmunu_inv_upper, _fmunu_inv_lower,
            _dw_dagger, _din_dagger, _shift_dagger, spinor_in, params, csw, update);
    }
};

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
class MDWFMobiusCloverAdjointLinearOperator {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;
    using Gauge = Gaugefield<floatT, true, HaloDepthGauge, R18>;

private:
    MDWFMobiusOperatorParameters<floatT> _params;
    floatT _csw;
    MDWFMobiusAdjointCloverOperatorWorkspace<floatT, HaloDepthGauge, HaloDepthSpin, Ls> _workspace;

public:
    MDWFMobiusCloverAdjointLinearOperator(Gauge &gauge, floatT M5, floatT mf, floatT b5, floatT csw = 0.0,
                                          std::string name = "MDWF_mobius_clover_adjoint_linear_operator")
        : _params(M5, mf, b5),
          _csw(csw),
          _workspace(gauge, name + "_workspace") {}

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        _workspace.apply(spinor_out, spinor_in, _params, _csw, update);
    }

    const MDWFMobiusOperatorParameters<floatT> &params() const {
        return _params;
    }

    floatT mass() const {
        return _params.mass;
    }

    floatT csw() const {
        return _csw;
    }
};
