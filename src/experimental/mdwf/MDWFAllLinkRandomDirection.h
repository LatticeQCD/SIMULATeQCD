/*
 * MDWF all-link deterministic-direction test helpers.
 *
 * These helpers construct and apply a reproducible anti-Hermitian traceless
 * direction to every owned bulk gauge link.  They are for finite-difference
 * and direction-dependent accumulator tests only; they do not define a
 * production force, projection, momentum update, or HMC sign.
 */

#pragma once

#include "MDWFOneLinkForceAccumulatorMock.h"

#include <cmath>
#include <stdexcept>

template<class floatT, size_t HaloDepth>
__host__ __device__ SU3<floatT> mdwfAllLinkDeterministicDirection(
    const gSiteMu &site_mu) {

    typedef GIndexer<All, HaloDepth> GInd;
    const sitexyzt global = GInd::getLatData().globalPos(site_mu.coord);
    const int hash = global.x
                     + 3 * global.y
                     + 5 * global.z
                     + 7 * global.t
                     + 11 * static_cast<int>(site_mu.mu);
    const int generator_id = hash % 3;
    const floatT sign = ((global.x + global.y + global.z + global.t
                          + static_cast<int>(site_mu.mu)) % 2 == 0)
                        ? static_cast<floatT>(1.0)
                        : static_cast<floatT>(-1.0);
    const floatT inverse_sqrt_two = static_cast<floatT>(0.70710678118654752440);

    return sign * inverse_sqrt_two
           * mdwfFiniteDifferenceGenerator<floatT>(generator_id);
}

template<class floatT, size_t HaloDepth, CompressionType comp>
struct MDWFAllLinkPerturbation {
    SU3Accessor<floatT, comp> gauge_in;
    floatT epsilon;
    int sign;

    MDWFAllLinkPerturbation(
        const SU3Accessor<floatT, comp> &gauge_in_in,
        floatT epsilon_in,
        int sign_in)
        : gauge_in(gauge_in_in),
          epsilon(epsilon_in),
          sign(sign_in) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu site_mu) {
        const SU3<floatT> direction
            = mdwfAllLinkDeterministicDirection<floatT, HaloDepth>(site_mu);
        const SU3<floatT> perturbation = su3_exp(
            static_cast<floatT>(sign) * epsilon * direction);
        return perturbation * gauge_in.getLink(site_mu);
    }
};

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void applyMDWFAllLinkPerturbation(
    Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_out,
    const Gaugefield<floatT, onDevice, HaloDepth, comp> &gauge_in,
    floatT epsilon,
    int sign) {

    if (epsilon <= static_cast<floatT>(0.0)
        || !std::isfinite(static_cast<double>(epsilon))) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF all-link perturbation requires positive finite epsilon"));
    }
    if (sign != 1 && sign != -1) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF all-link perturbation sign must be +1 or -1"));
    }

    gauge_out.template iterateOverBulkAllMu<>(
        MDWFAllLinkPerturbation<floatT, HaloDepth, comp>(
            gauge_in.getAccessor(), epsilon, sign));
    gauge_out.updateAll();
}

template<class floatT>
struct MDWFAllLinkZeroMatrix {
    __host__ __device__ SU3<floatT> operator()(gSiteMu) {
        return su3_zero<floatT>();
    }
};
