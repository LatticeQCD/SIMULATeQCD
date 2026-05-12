/*
 * MDWF one-link force-accumulator mock.
 *
 * This helper writes one selected link of a caller-supplied gauge-force-like
 * field and validates the scalar contraction convention against the existing
 * analytic force-contraction helper.  It deliberately does not implement a
 * production force kernel, projection convention, momentum update, RHMC/HMC
 * wiring, HISQ reuse, smearing, or all-link accumulation.
 */

#pragma once

#include "MDWFAnalyticForceContractionCheck.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

template<class floatT>
struct MDWFOneLinkForceAccumulatorMockResult {
    double contracted_derivative;
    double selected_link_difference;
    double max_off_probe_norm;
    size_t selected_link_count;
};

template<class floatT>
__host__ __device__ bool mdwfOneLinkForceAccumulatorMockMatchesProbe(
    const gSiteMu &site_mu,
    const MDWFFiniteDifferenceProbe<floatT> &probe) {

    return site_mu.mu == probe.mu
           && site_mu.coord.x == probe.x
           && site_mu.coord.y == probe.y
           && site_mu.coord.z == probe.z
           && site_mu.coord.t == probe.t;
}

template<class floatT>
struct MDWFOneLinkForceAccumulatorMockFunctor {
    MDWFFiniteDifferenceProbe<floatT> probe;
    SU3<floatT> selected_link_action_derivative;

    MDWFOneLinkForceAccumulatorMockFunctor(
        const MDWFFiniteDifferenceProbe<floatT> &probe_in,
        const SU3<floatT> &selected_link_action_derivative_in)
        : probe(probe_in),
          selected_link_action_derivative(selected_link_action_derivative_in) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu site_mu) {
        if (mdwfOneLinkForceAccumulatorMockMatchesProbe(site_mu, probe)) {
            return selected_link_action_derivative;
        }
        return su3_zero<floatT>();
    }
};

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void writeMDWFOneLinkForceAccumulatorMock(
    Gaugefield<floatT, onDevice, HaloDepth, comp> &force_out,
    const MDWFFiniteDifferenceProbe<floatT> &probe,
    const SU3<floatT> &selected_link_action_derivative) {

    if (probe.mu >= 4) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF one-link force-accumulator mock requires mu in [0, 3], got ",
            static_cast<int>(probe.mu)));
    }

    force_out.template iterateOverBulkAllMu<>(
        MDWFOneLinkForceAccumulatorMockFunctor<floatT>(
            probe, selected_link_action_derivative));
    force_out.updateAll();
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
MDWFOneLinkForceAccumulatorMockResult<floatT> inspectMDWFOneLinkForceAccumulatorMock(
    Gaugefield<floatT, onDevice, HaloDepth, comp> &force,
    const MDWFFiniteDifferenceProbe<floatT> &probe,
    const SU3<floatT> &expected_selected_link_action_derivative,
    CommunicationBase &commBase,
    const std::string &name = "MDWF_one_link_force_accumulator_mock") {

    typedef GIndexer<All, HaloDepth> GInd;

    Gaugefield<floatT, false, HaloDepth, comp> forceHost(commBase, name + "_host");
    forceHost = force;
    SU3Accessor<floatT, comp> forceAcc = forceHost.getAccessor();

    double contracted_derivative = 0.0;
    double selected_link_difference = 0.0;
    double max_off_probe_norm = 0.0;
    size_t selected_link_count = 0;

    for (size_t site_index = 0; site_index < GInd::getLatData().vol4; site_index++) {
        const gSite site = GInd::getSite(site_index);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const gSiteMu site_mu = GInd::getSiteMu(site, mu);
            const SU3<floatT> force_link = forceAcc.getLink(site_mu);

            if (mdwfOneLinkForceAccumulatorMockMatchesProbe(site_mu, probe)) {
                selected_link_count++;
                contracted_derivative = mdwfContractSingleLinkActionDerivative(
                    force_link, probe);
                selected_link_difference = std::max(
                    selected_link_difference,
                    static_cast<double>(infnorm(force_link
                                                - expected_selected_link_action_derivative)));
            } else {
                max_off_probe_norm = std::max(
                    max_off_probe_norm,
                    static_cast<double>(infnorm(force_link)));
            }
        }
    }

    return {
        contracted_derivative,
        selected_link_difference,
        max_off_probe_norm,
        selected_link_count
    };
}
