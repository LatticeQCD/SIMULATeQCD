/*
 * Test-only Shamir-kernel physical parameter mapping.
 *
 * Maps (M5, mf) onto the existing MDWF scaffold's abstract per-slice Wilson
 * kernel mass and MDWFFifthDimCoefficients, following the Shamir special case
 * (b5 = 1, c5 = 0) derived in PHYSICAL_OPERATOR_MAPPING.md:
 *
 *   mass              = 4 - M5          (from mass = m_std + 4, m_std = -M5)
 *   diagonal          = 1
 *   forward_hop       = -1
 *   backward_hop      = -1
 *   forward_boundary  = +mf
 *   backward_boundary = +mf
 *
 * This header does not select a production M5/mf, does not implement general
 * Mobius (nonzero c5, which needs a gauge-dependent fifth-direction coupling
 * and is out of scope), and does not change MDWFFifthDimCoupling,
 * MDWFOperator, or any existing scaffold behavior.
 */

#pragma once

#include "MDWFFifthDim.h"

/*
 * Wilson kernel mass argument for D_W(-M5). gamma5DiracWilson and the clover
 * path (preCalcFmunu + DiracWilsonEvenEven2/EvenOdd2, src/experimental/DWilson.h)
 * compute (D_W psi)(x) = mass psi(x) - (1/2) sum_mu [(1 - gamma_mu) U psi(x+mu)
 * + (1 + gamma_mu) U^dagger psi(x-mu)] (the kernel starts from 2 mass psi and
 * halves the whole sum), so D_W(p = 0) = mass - 4 = m_std and mass = m_std + 4 =
 * 4 - M5. The earlier 2 - M5/2 read the kernel as 2 mass psi - (1/2) sum and
 * simulated D_W(p = 0) = -(2 + M5/2), i.e. an effective M5 of 2.9 for M5 = 1.8,
 * outside the domain-wall window (TODO.md). mdwfWilsonKernelNormalizationTest
 * pins D_W(p = 0) = -M5 on a unit gauge field.
 */
template<class floatT>
__host__ __device__ inline floatT mdwfShamirKernelMass(floatT M5) {
    return static_cast<floatT>(4.0) - M5;
}

template<class floatT>
__host__ __device__ inline MDWFFifthDimCoefficients<floatT> mdwfShamirFifthDimCoefficients(floatT mf) {
    return MDWFFifthDimCoefficients<floatT>(
        static_cast<floatT>(1.0),   // diagonal
        static_cast<floatT>(-1.0),  // forward_hop
        static_cast<floatT>(-1.0),  // backward_hop
        mf,                         // forward_boundary
        mf);                        // backward_boundary
}

template<class floatT>
struct MDWFShamirOperatorParameters {
    floatT mass;
    MDWFFifthDimCoefficients<floatT> fifth_coeff;

    __host__ __device__ MDWFShamirOperatorParameters(floatT M5, floatT mf)
        : mass(mdwfShamirKernelMass<floatT>(M5)),
          fifth_coeff(mdwfShamirFifthDimCoefficients<floatT>(mf)) {}
};
