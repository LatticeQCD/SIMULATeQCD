/*
 * Test-only Shamir-kernel physical parameter mapping.
 *
 * Maps (M5, mf) onto the existing MDWF scaffold's abstract per-slice Wilson
 * kernel mass and MDWFFifthDimCoefficients, following the Shamir special case
 * (b5 = 1, c5 = 0) derived in PHYSICAL_OPERATOR_MAPPING.md:
 *
 *   mass              = 2 - M5 / 2      (from mass = (m_std + 4) / 2, m_std = -M5)
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

template<class floatT>
__host__ __device__ inline floatT mdwfShamirKernelMass(floatT M5) {
    return static_cast<floatT>(2.0) - static_cast<floatT>(0.5) * M5;
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
