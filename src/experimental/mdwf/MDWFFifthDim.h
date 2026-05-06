/*
 * Fifth-direction-only MDWF coupling.
 *
 * This header does not call the 4D Wilson kernel, does not apply clover, and
 * does not integrate with a solver.  It only couples neighboring Ls slices
 * stored in the MDWFSpinor stack dimension.
 */

#pragma once

#include "MDWFSpinor.h"
#include "../fullSpinor.h"

template<class floatT>
__host__ __device__ inline Vect12<floatT> mdwfProjectPlus(const Vect12<floatT> &spinor) {
    Vect12<floatT> out(0.0);

    for (size_t i = 0; i < 6; i++) {
        out.data[i] = spinor.data[i];
    }
    return out;
}

template<class floatT>
__host__ __device__ inline Vect12<floatT> mdwfProjectMinus(const Vect12<floatT> &spinor) {
    Vect12<floatT> out(0.0);

    for (size_t i = 6; i < 12; i++) {
        out.data[i] = spinor.data[i];
    }
    return out;
}

template<class floatT>
struct MDWFFifthDimCoefficients {
    floatT diagonal;
    floatT forward_hop;
    floatT backward_hop;
    floatT forward_boundary;
    floatT backward_boundary;

    __host__ __device__ MDWFFifthDimCoefficients(floatT diagonal_in,
                                                 floatT forward_hop_in,
                                                 floatT backward_hop_in,
                                                 floatT forward_boundary_in,
                                                 floatT backward_boundary_in)
        : diagonal(diagonal_in),
          forward_hop(forward_hop_in),
          backward_hop(backward_hop_in),
          forward_boundary(forward_boundary_in),
          backward_boundary(backward_boundary_in) {}
};

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct MDWFFifthDimCoupling {
    static_assert(Ls > 1, "MDWF fifth-direction coupling requires Ls > 1");

    Vect12ArrayAcc<floatT> spinor_in;
    MDWFFifthDimCoefficients<floatT> coeff;

    typedef GIndexer<LatLayout, HaloDepth> GInd;

    template<bool onDevice>
    MDWFFifthDimCoupling(const MDWFSpinor<floatT, onDevice, LatLayout, HaloDepth, Ls> &spinor_in_in,
                         MDWFFifthDimCoefficients<floatT> coeff_in)
        : spinor_in(spinor_in_in.getAccessor()),
          coeff(coeff_in) {}

    __device__ __host__ Vect12<floatT> operator()(gSiteStack site) {
        const size_t s = site.stack;
        const bool at_forward_boundary = (s + 1 == Ls);
        const bool at_backward_boundary = (s == 0);
        const size_t s_forward = at_forward_boundary ? 0 : s + 1;
        const size_t s_backward = at_backward_boundary ? Ls - 1 : s - 1;

        const floatT forward_coeff = at_forward_boundary ? coeff.forward_boundary : coeff.forward_hop;
        const floatT backward_coeff = at_backward_boundary ? coeff.backward_boundary : coeff.backward_hop;

        Vect12<floatT> out = coeff.diagonal * spinor_in.getElement(site);
        out += forward_coeff * mdwfProjectMinus(spinor_in.getElement(GInd::getSiteStack(site, s_forward)));
        out += backward_coeff * mdwfProjectPlus(spinor_in.getElement(GInd::getSiteStack(site, s_backward)));

        return out;
    }
};

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t Ls>
void applyMDWFFifthDimCoupling(MDWFSpinor<floatT, onDevice, LatLayout, HaloDepth, Ls> &spinor_out,
                               const MDWFSpinor<floatT, onDevice, LatLayout, HaloDepth, Ls> &spinor_in,
                               MDWFFifthDimCoefficients<floatT> coeff,
                               bool update = false) {
    spinor_out.template iterateOverBulk<>(
        MDWFFifthDimCoupling<floatT, LatLayout, HaloDepth, Ls>(spinor_in, coeff));

    if (update) {
        spinor_out.updateAll();
    }
}
