/*
 * Test-only general-Mobius (RBC/UKQCD convention) physical parameter mapping
 * and forward-operator composition, following PHYSICAL_OPERATOR_MAPPING.md
 * Section 2.3.
 *
 * This is plain (not zMobius) Mobius: constant b5, c5 per slice, with the
 * project's chosen RBC/UKQCD convention c5 = b5 - 1. It does not implement
 * s-dependent bs[s]/cs[s], clover (csw != 0) for this path, the adjoint of
 * the general Mobius operator, or a production M5/mf/b5 choice. It does not
 * change MDWFFifthDimCoupling, MDWFWilsonSlice, MDWFOperator, or
 * MDWFPhysicalMapping.h: the composition below calls them unmodified.
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
