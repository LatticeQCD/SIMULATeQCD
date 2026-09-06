/*
 * MDWF test-only single-rank all-link direction-independent storage.
 *
 * The first scaffolds accumulate raw Wilson and optional clover matrices in
 * explicit bulk buffers and overwrite a caller-supplied host R18 destination
 * with one projected total per bulk link.  They do not read the previous
 * destination, refresh output halos, define ipdot, choose an HMC sign, or
 * define MPI ownership.
 */

#pragma once

#include "MDWFAllLinkWilsonContraction.h"
#include "MDWFCloverAllLinkContraction.h"
#include "MDWFRationalCoefficientAdapter.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

template<class floatT>
struct MDWFAllLinkDirectionIndependentStorageResult {
    std::vector<SU3<floatT>> raw_wilson;
    std::vector<SU3<floatT>> raw_clover;
    std::vector<size_t> wilson_term_additions;
    std::vector<size_t> clover_term_additions;
    std::vector<size_t> clover_path_additions;
    std::vector<size_t> finalize_counts;
    size_t bulk_links;
};

template<size_t HaloDepth, size_t Ls, class Workspace>
MDWFAllLinkDirectionIndependentStorageResult<double>
overwriteMDWFWilsonAllLinkDirectionIndependentStorageCsw0(
    Gaugefield<double, false, HaloDepth, R18> &destination,
    const Gaugefield<double, false, HaloDepth, R18> &gauge,
    const Workspace &workspace,
    const MDWFRationalCoefficients<double> &force_coefficients,
    CommunicationBase &comm_base,
    const std::string &name
        = "MDWF_wilson_all_link_direction_independent_storage_csw0") {

    typedef GIndexer<All, HaloDepth> GInd;
    using HostSpinor = MDWFSpinor<double, false, All, HaloDepth, Ls>;

    const LatticeData lat = GInd::getLatData();
    if (lat.vol4 != lat.globvol4) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF all-link direction-independent storage is single-rank "
            "only: local volume = ", lat.vol4,
            ", global volume = ", lat.globvol4));
    }
    if (workspace.size() != force_coefficients.numerator.size()) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF all-link direction-independent storage workspace/coefficient "
            "size mismatch: workspace = ", workspace.size(),
            ", numerators = ", force_coefficients.numerator.size()));
    }

    const size_t bulk_links = 4 * lat.vol4;
    MDWFAllLinkDirectionIndependentStorageResult<double> result{
        std::vector<SU3<double>>(bulk_links, su3_zero<double>()),
        std::vector<SU3<double>>(bulk_links, su3_zero<double>()),
        std::vector<size_t>(bulk_links, 0),
        std::vector<size_t>(bulk_links, 0),
        std::vector<size_t>(bulk_links, 0),
        std::vector<size_t>(bulk_links, 0),
        bulk_links
    };
    const SU3Accessor<double, R18> gauge_acc = gauge.getAccessor();

    for (size_t term = 0; term < workspace.size(); term++) {
        HostSpinor chi_host(
            comm_base, name + "_chi_" + std::to_string(term));
        HostSpinor eta_host(
            comm_base, name + "_eta_" + std::to_string(term));
        chi_host = workspace.chi(term);
        eta_host = workspace.eta(term);
        const Vect12ArrayAcc<double> chi_acc = chi_host.getAccessor();
        const Vect12ArrayAcc<double> eta_acc = eta_host.getAccessor();
        const double rational_weight
            = -2.0 * force_coefficients.numerator[term];

        for (size_t site_index = 0; site_index < lat.vol4; site_index++) {
            const gSite site = GInd::getSite(site_index);
            for (uint8_t mu = 0; mu < 4; mu++) {
                const size_t bulk_link = site.isite * 4 + mu;
                result.raw_wilson[bulk_link]
                    += rational_weight
                       * mdwfWilsonLeftRawContractionMatrixTerm<
                           HaloDepth, Ls>(
                           chi_acc, eta_acc, gauge_acc, site, mu);
                result.wilson_term_additions[bulk_link]++;
            }
        }
    }

    SU3Accessor<double, R18> destination_acc = destination.getAccessor();
    for (size_t site_index = 0; site_index < lat.vol4; site_index++) {
        const gSite site = GInd::getSite(site_index);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const size_t bulk_link = site.isite * 4 + mu;
            SU3<double> projected_total
                = result.raw_wilson[bulk_link]
                  + result.raw_clover[bulk_link];
            projected_total.TA();
            destination_acc.setLink(
                GInd::getSiteMu(site, mu), projected_total);
            result.finalize_counts[bulk_link]++;
        }
    }

    return result;
}

template<size_t HaloDepth, size_t Ls, class Workspace>
MDWFAllLinkDirectionIndependentStorageResult<double>
overwriteMDWFCloverAllLinkDirectionIndependentStorageNonzero(
    Gaugefield<double, false, HaloDepth, R18> &destination,
    const Gaugefield<double, false, HaloDepth, R18> &gauge,
    const Workspace &workspace,
    const MDWFRationalCoefficients<double> &force_coefficients,
    double csw,
    CommunicationBase &comm_base,
    const std::string &name
        = "MDWF_clover_all_link_direction_independent_storage_nonzero") {

    typedef GIndexer<All, HaloDepth> GInd;
    using HostSpinor = MDWFSpinor<double, false, All, HaloDepth, Ls>;

    const LatticeData lat = GInd::getLatData();
    if (lat.vol4 != lat.globvol4) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF nonzero-c_sw all-link direction-independent storage is "
            "single-rank only: local volume = ", lat.vol4,
            ", global volume = ", lat.globvol4));
    }
    if (!std::isfinite(csw) || csw == 0.0) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF nonzero-c_sw all-link direction-independent storage "
            "requires finite nonzero c_sw, got ", csw));
    }
    if (workspace.size() != force_coefficients.numerator.size()) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF nonzero-c_sw all-link direction-independent storage "
            "workspace/coefficient size mismatch: workspace = ",
            workspace.size(), ", numerators = ",
            force_coefficients.numerator.size()));
    }

    const size_t bulk_links = 4 * lat.vol4;
    MDWFAllLinkDirectionIndependentStorageResult<double> result{
        std::vector<SU3<double>>(bulk_links, su3_zero<double>()),
        std::vector<SU3<double>>(bulk_links, su3_zero<double>()),
        std::vector<size_t>(bulk_links, 0),
        std::vector<size_t>(bulk_links, 0),
        std::vector<size_t>(bulk_links, 0),
        std::vector<size_t>(bulk_links, 0),
        bulk_links
    };
    const SU3Accessor<double, R18> gauge_acc = gauge.getAccessor();

    for (size_t term = 0; term < workspace.size(); term++) {
        HostSpinor chi_host(
            comm_base, name + "_chi_" + std::to_string(term));
        HostSpinor eta_host(
            comm_base, name + "_eta_" + std::to_string(term));
        chi_host = workspace.chi(term);
        eta_host = workspace.eta(term);
        const Vect12ArrayAcc<double> chi_acc = chi_host.getAccessor();
        const Vect12ArrayAcc<double> eta_acc = eta_host.getAccessor();
        const double rational_weight
            = -2.0 * force_coefficients.numerator[term];
        std::vector<size_t> term_clover_path_additions(bulk_links, 0);

        for (size_t site_index = 0; site_index < lat.vol4; site_index++) {
            const gSite site = GInd::getSite(site_index);
            for (uint8_t mu = 0; mu < 4; mu++) {
                const size_t bulk_link = site.isite * 4 + mu;
                result.raw_wilson[bulk_link]
                    += rational_weight
                       * mdwfWilsonLeftRawContractionMatrixTerm<
                           HaloDepth, Ls>(
                           chi_acc, eta_acc, gauge_acc, site, mu);
                result.wilson_term_additions[bulk_link]++;
            }

            mdwfAllLinkCloverAccumulateLeftRawContractionSite<
                HaloDepth, Ls>(
                gauge_acc, chi_acc, eta_acc, site, csw,
                rational_weight, result.raw_clover,
                term_clover_path_additions);
        }

        for (size_t bulk_link = 0;
             bulk_link < bulk_links;
             bulk_link++) {
            result.clover_path_additions[bulk_link]
                += term_clover_path_additions[bulk_link];
            if (term_clover_path_additions[bulk_link] > 0) {
                result.clover_term_additions[bulk_link]++;
            }
        }
    }

    SU3Accessor<double, R18> destination_acc = destination.getAccessor();
    for (size_t site_index = 0; site_index < lat.vol4; site_index++) {
        const gSite site = GInd::getSite(site_index);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const size_t bulk_link = site.isite * 4 + mu;
            SU3<double> projected_total
                = result.raw_wilson[bulk_link]
                  + result.raw_clover[bulk_link];
            projected_total.TA();
            destination_acc.setLink(
                GInd::getSiteMu(site, mu), projected_total);
            result.finalize_counts[bulk_link]++;
        }
    }

    return result;
}
