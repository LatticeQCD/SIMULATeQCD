/*
 * MDWF c_sw = 0 Wilson all-link direction-independent storage scaffold.
 *
 * This single-rank test accumulates one raw Wilson matrix per owned bulk link,
 * projects exactly once, and overwrites a full-R18 host destination.  It
 * validates zero-filled, nonzero-sentinel, and repeated-sentinel destinations
 * against an independent deterministic-direction scalar contraction.  It
 * does not define output halos, an additive caller API, ipdot, an HMC sign, or
 * MPI ownership.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFAdjointOperator.h"
#include "../experimental/mdwf/MDWFAllLinkDirectionIndependentStorage.h"
#include "../experimental/mdwf/MDWFFermionForceWorkspace.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"
#include "../experimental/mdwf/MDWFRationalCoefficientAdapter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFWilsonAllLinkDirectionIndependentStorageCsw0Source {
    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t component = 0; component < 12; component++) {
            out.data[component] = COMPLEX(floatT)(
                static_cast<floatT>(0.5)
                + static_cast<floatT>(site.stack + 1)
                + static_cast<floatT>(0.001)
                  * static_cast<floatT>(site.isite + component + 1),
                static_cast<floatT>(0.017) * static_cast<floatT>(component + 1)
                - static_cast<floatT>(0.0025)
                  * static_cast<floatT>(site.stack));
        }
        return out;
    }
};

template<class floatT>
struct MDWFAllLinkDirectionIndependentStorageSentinel {
    __host__ __device__ SU3<floatT> operator()(gSiteMu site_mu) {
        const floatT scale
            = static_cast<floatT>(3.0)
              + static_cast<floatT>(0.25)
                * static_cast<floatT>(site_mu.mu);
        return scale * SU3<floatT>(
            COMPLEX(floatT)(1.0, 0.5),
            COMPLEX(floatT)(0.25, -0.75),
            COMPLEX(floatT)(-0.5, 0.125),
            COMPLEX(floatT)(0.875, 0.375),
            COMPLEX(floatT)(-0.25, 1.25),
            COMPLEX(floatT)(0.625, -0.5),
            COMPLEX(floatT)(-0.375, 0.75),
            COMPLEX(floatT)(0.5, 0.25),
            COMPLEX(floatT)(1.5, -0.625));
    }
};

template<class floatT>
bool mdwfAllLinkDirectionIndependentStorageMatrixFinite(
    const SU3<floatT> &matrix) {

    for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; column++) {
            const COMPLEX(floatT) entry = matrix(row, column);
            if (!std::isfinite(static_cast<double>(real(entry)))
                || !std::isfinite(static_cast<double>(imag(entry)))) {
                return false;
            }
        }
    }
    return true;
}

template<size_t Ls>
void runMDWFWilsonAllLinkDirectionIndependentStorageCsw0Test(
    CommunicationBase &comm_base) {

    const size_t HaloDepth = 2;
    const double mass = 4.0;
    const double csw = 0.0;
    typedef GIndexer<All, HaloDepth> GInd;

    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using ForwardOperator
        = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using AdjointOperator
        = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using NormalOperator
        = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using Workspace
        = MDWFFermionForceWorkspace<double, HaloDepth, HaloDepth, Ls,
                                    NormalOperator, ForwardOperator>;
    using Spinor = typename NormalOperator::Spinor;
    using HostSpinor = MDWFSpinor<double, false, All, HaloDepth, Ls>;
    using StorageResult
        = MDWFAllLinkDirectionIndependentStorageResult<double>;

    const LatticeData lat = GInd::getLatData();
    if (lat.vol4 != lat.globvol4) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Wilson all-link direction-independent storage scaffold is "
            "single-rank only: local volume = ", lat.vol4,
            ", global volume = ", lat.globvol4));
    }

    MDWFExplicitRationalInput<double> force_input{
        "wilson_all_link_direction_independent_storage_csw0_force_coefficients",
        MDWFRationalCoefficientRole::Force,
        0.0,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    const MDWFRationalCoefficients<double> force_coefficients
        = makeMDWFRationalCoefficients(force_input);

    Gauge gauge(comm_base,
                "MDWF_wilson_all_link_direction_independent_storage_csw0_gauge");
    grnd_state<false> host_random;
    grnd_state<true> device_random;
    host_random.make_rng_state(20260514);
    device_random = host_random;
    gauge.random(device_random.state);
    gauge.updateAll();

    Spinor field(
        comm_base,
        "MDWF_wilson_all_link_direction_independent_storage_csw0_field");
    field.template iterateOverBulk<>(
        FillMDWFWilsonAllLinkDirectionIndependentStorageCsw0Source<
            double, All, HaloDepth, Ls>());
    field.updateAll();

    const MDWFFifthDimCoefficients<double> fifth_coefficients(
        1.0, -0.05, -0.05, 0.0, 0.0);
    ForwardOperator forward(
        gauge, fifth_coefficients, mass, csw,
        "MDWF_wilson_all_link_direction_independent_storage_csw0_forward");
    AdjointOperator adjoint(
        gauge, fifth_coefficients, mass, csw,
        "MDWF_wilson_all_link_direction_independent_storage_csw0_adjoint");
    NormalOperator normal(
        comm_base, forward, adjoint,
        "MDWF_wilson_all_link_direction_independent_storage_csw0_normal");
    Workspace workspace;
    workspace.prepare(
        normal, forward, field, force_coefficients, 512, 1e-8,
        "MDWF_wilson_all_link_direction_independent_storage_csw0_workspace");

    double max_force_workspace_residual = 0.0;
    for (const auto &info : workspace.shiftInfo()) {
        max_force_workspace_residual = std::max(
            max_force_workspace_residual, info.residue);
    }

    HostGauge gauge_host(
        comm_base,
        "MDWF_wilson_all_link_direction_independent_storage_csw0_gauge_host");
    gauge_host = gauge;
    const SU3Accessor<double, R18> gauge_accessor = gauge_host.getAccessor();

    const size_t expected_links = 4 * lat.vol4;
    std::vector<double> direct_link_derivatives(expected_links, 0.0);

    for (size_t term = 0; term < workspace.size(); term++) {
        HostSpinor chi_host(
            comm_base,
            "MDWF_wilson_all_link_direction_independent_storage_csw0_chi_"
            + std::to_string(term));
        HostSpinor eta_host(
            comm_base,
            "MDWF_wilson_all_link_direction_independent_storage_csw0_eta_"
            + std::to_string(term));
        chi_host = workspace.chi(term);
        eta_host = workspace.eta(term);
        const Vect12ArrayAcc<double> chi_accessor = chi_host.getAccessor();
        const Vect12ArrayAcc<double> eta_accessor = eta_host.getAccessor();
        const double rational_weight
            = -2.0 * force_coefficients.numerator[term];

        for (size_t site_index = 0; site_index < lat.vol4; site_index++) {
            const gSite site = GInd::getSite(site_index);
            for (uint8_t mu = 0; mu < 4; mu++) {
                const gSiteMu site_mu = GInd::getSiteMu(site, mu);
                const SU3<double> direction
                    = mdwfAllLinkDeterministicDirection<
                        double, HaloDepth>(site_mu);
                const size_t bulk_link = site.isite * 4 + mu;
                direct_link_derivatives[bulk_link]
                    += rational_weight
                       * mdwfAllLinkWilsonContractionTerm<
                           HaloDepth, Ls>(
                           chi_accessor, eta_accessor, gauge_accessor,
                           site, mu, direction);
            }
        }
    }

    HostGauge zero_destination(
        comm_base,
        "MDWF_wilson_all_link_direction_independent_storage_csw0_zero");
    zero_destination.template iterateOverFullAllMu<>(
        MDWFAllLinkZeroMatrix<double>());

    HostGauge sentinel_reference(
        comm_base,
        "MDWF_wilson_all_link_direction_independent_storage_csw0_sentinel_ref");
    sentinel_reference.template iterateOverFullAllMu<>(
        MDWFAllLinkDirectionIndependentStorageSentinel<double>());
    HostGauge sentinel_destination(
        comm_base,
        "MDWF_wilson_all_link_direction_independent_storage_csw0_sentinel");
    sentinel_destination = sentinel_reference;

    const StorageResult zero_result
        = overwriteMDWFWilsonAllLinkDirectionIndependentStorageCsw0<
            HaloDepth, Ls>(
            zero_destination, gauge_host, workspace, force_coefficients,
            comm_base, "MDWF_wilson_all_link_storage_zero");
    const StorageResult sentinel_result
        = overwriteMDWFWilsonAllLinkDirectionIndependentStorageCsw0<
            HaloDepth, Ls>(
            sentinel_destination, gauge_host, workspace, force_coefficients,
            comm_base, "MDWF_wilson_all_link_storage_sentinel");

    HostGauge sentinel_first_result(
        comm_base,
        "MDWF_wilson_all_link_direction_independent_storage_csw0_sentinel_first");
    sentinel_first_result = sentinel_destination;
    const StorageResult repeated_result
        = overwriteMDWFWilsonAllLinkDirectionIndependentStorageCsw0<
            HaloDepth, Ls>(
            sentinel_destination, gauge_host, workspace, force_coefficients,
            comm_base, "MDWF_wilson_all_link_storage_repeated");

    const std::array<const StorageResult *, 3> results{{
        &zero_result, &sentinel_result, &repeated_result
    }};
    for (const StorageResult *result : results) {
        if (result->bulk_links != expected_links
            || result->raw_wilson.size() != expected_links
            || result->raw_clover.size() != expected_links
            || result->wilson_term_additions.size() != expected_links
            || result->clover_term_additions.size() != expected_links
            || result->clover_path_additions.size() != expected_links
            || result->finalize_counts.size() != expected_links) {
            throw std::runtime_error(stdLogger.fatal(
                "MDWF c_sw = 0 Wilson all-link direction-independent storage "
                "returned an invalid bulk-buffer shape"));
        }
    }

    const SU3Accessor<double, R18> zero_accessor
        = zero_destination.getAccessor();
    const SU3Accessor<double, R18> sentinel_reference_accessor
        = sentinel_reference.getAccessor();
    const SU3Accessor<double, R18> sentinel_first_accessor
        = sentinel_first_result.getAccessor();
    const SU3Accessor<double, R18> sentinel_repeated_accessor
        = sentinel_destination.getAccessor();

    size_t inspected_links = 0;
    size_t overwritten_sentinel_links = 0;
    size_t invalid_links = 0;
    size_t missing_finalize_links = 0;
    size_t duplicate_finalize_links = 0;
    size_t missing_wilson_term_links = 0;
    size_t duplicate_wilson_term_links = 0;
    size_t unexpected_clover_term_links = 0;
    size_t unexpected_clover_path_links = 0;
    double direct_total = 0.0;
    double raw_total = 0.0;
    double stored_total = 0.0;
    double max_direct_raw_abs_diff = 0.0;
    double max_direct_stored_abs_diff = 0.0;
    double max_zero_sentinel_output_diff = 0.0;
    double max_repeated_output_diff = 0.0;
    double max_zero_sentinel_raw_diff = 0.0;
    double max_repeated_raw_diff = 0.0;
    double max_raw_clover_norm = 0.0;
    double max_anti_hermitian_violation = 0.0;
    double max_trace_violation = 0.0;
    double max_raw_norm = 0.0;
    double max_stored_norm = 0.0;
    double max_raw_invisible_component_norm = 0.0;

    for (size_t site_index = 0; site_index < lat.vol4; site_index++) {
        const gSite site = GInd::getSite(site_index);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const gSiteMu site_mu = GInd::getSiteMu(site, mu);
            const size_t bulk_link = site.isite * 4 + mu;
            const SU3<double> direction
                = mdwfAllLinkDeterministicDirection<
                    double, HaloDepth>(site_mu);
            const SU3<double> raw_matrix
                = zero_result.raw_wilson[bulk_link];
            const SU3<double> stored_matrix
                = zero_accessor.getLink(site_mu);
            const SU3<double> sentinel_reference_matrix
                = sentinel_reference_accessor.getLink(site_mu);
            const SU3<double> sentinel_first_matrix
                = sentinel_first_accessor.getLink(site_mu);
            const SU3<double> sentinel_repeated_matrix
                = sentinel_repeated_accessor.getLink(site_mu);
            const double direct_derivative
                = direct_link_derivatives[bulk_link];
            const double raw_derivative
                = real(tr_c(direction, raw_matrix));
            const double stored_derivative
                = real(tr_c(direction, stored_matrix));

            inspected_links++;
            direct_total += direct_derivative;
            raw_total += raw_derivative;
            stored_total += stored_derivative;
            max_direct_raw_abs_diff = std::max(
                max_direct_raw_abs_diff,
                std::abs(direct_derivative - raw_derivative));
            max_direct_stored_abs_diff = std::max(
                max_direct_stored_abs_diff,
                std::abs(direct_derivative - stored_derivative));
            max_zero_sentinel_output_diff = std::max(
                max_zero_sentinel_output_diff,
                static_cast<double>(infnorm(
                    stored_matrix - sentinel_first_matrix)));
            max_repeated_output_diff = std::max(
                max_repeated_output_diff,
                static_cast<double>(infnorm(
                    sentinel_first_matrix - sentinel_repeated_matrix)));
            max_zero_sentinel_raw_diff = std::max(
                max_zero_sentinel_raw_diff,
                static_cast<double>(infnorm(
                    zero_result.raw_wilson[bulk_link]
                    - sentinel_result.raw_wilson[bulk_link])));
            max_repeated_raw_diff = std::max(
                max_repeated_raw_diff,
                static_cast<double>(infnorm(
                    sentinel_result.raw_wilson[bulk_link]
                    - repeated_result.raw_wilson[bulk_link])));
            max_raw_clover_norm = std::max(
                max_raw_clover_norm,
                static_cast<double>(infnorm(
                    zero_result.raw_clover[bulk_link])));
            max_anti_hermitian_violation = std::max(
                max_anti_hermitian_violation,
                static_cast<double>(infnorm(
                    stored_matrix + dagger(stored_matrix))));
            max_trace_violation = std::max(
                max_trace_violation,
                static_cast<double>(abs(tr_c(stored_matrix))));
            max_raw_norm = std::max(
                max_raw_norm,
                static_cast<double>(infnorm(raw_matrix)));
            max_stored_norm = std::max(
                max_stored_norm,
                static_cast<double>(infnorm(stored_matrix)));
            max_raw_invisible_component_norm = std::max(
                max_raw_invisible_component_norm,
                static_cast<double>(infnorm(
                    raw_matrix - stored_matrix)));

            if (infnorm(
                    sentinel_reference_matrix - sentinel_first_matrix)
                > 1e-12) {
                overwritten_sentinel_links++;
            }
            for (const StorageResult *result : results) {
                if (result->finalize_counts[bulk_link] == 0) {
                    missing_finalize_links++;
                }
                if (result->finalize_counts[bulk_link] > 1) {
                    duplicate_finalize_links++;
                }
                if (result->wilson_term_additions[bulk_link]
                    < workspace.size()) {
                    missing_wilson_term_links++;
                }
                if (result->wilson_term_additions[bulk_link]
                    > workspace.size()) {
                    duplicate_wilson_term_links++;
                }
                if (result->clover_term_additions[bulk_link] != 0) {
                    unexpected_clover_term_links++;
                }
                if (result->clover_path_additions[bulk_link] != 0) {
                    unexpected_clover_path_links++;
                }
            }
            if (!std::isfinite(direct_derivative)
                || !mdwfAllLinkDirectionIndependentStorageMatrixFinite(
                    raw_matrix)
                || !mdwfAllLinkDirectionIndependentStorageMatrixFinite(
                    stored_matrix)
                || !mdwfAllLinkDirectionIndependentStorageMatrixFinite(
                    sentinel_first_matrix)
                || !mdwfAllLinkDirectionIndependentStorageMatrixFinite(
                    sentinel_repeated_matrix)) {
                invalid_links++;
            }
        }
    }

    const double direct_raw_total_abs_diff
        = std::abs(direct_total - raw_total);
    const double direct_stored_total_abs_diff
        = std::abs(direct_total - stored_total);
    const double contraction_tolerance
        = 1e-10 * std::max(1.0, std::abs(direct_total));

    if (mdwfRationalCoefficientRoleName(force_input.role) != "force"
        || !workspace.converged()
        || max_force_workspace_residual > 1e-8
        || inspected_links != expected_links
        || overwritten_sentinel_links != expected_links
        || invalid_links != 0
        || missing_finalize_links != 0
        || duplicate_finalize_links != 0
        || missing_wilson_term_links != 0
        || duplicate_wilson_term_links != 0
        || unexpected_clover_term_links != 0
        || unexpected_clover_path_links != 0
        || max_direct_raw_abs_diff > contraction_tolerance
        || max_direct_stored_abs_diff > contraction_tolerance
        || direct_raw_total_abs_diff > contraction_tolerance
        || direct_stored_total_abs_diff > contraction_tolerance
        || max_zero_sentinel_output_diff > 1e-12
        || max_repeated_output_diff > 1e-12
        || max_zero_sentinel_raw_diff > 1e-12
        || max_repeated_raw_diff > 1e-12
        || max_raw_clover_norm > 1e-12
        || max_anti_hermitian_violation > 1e-12
        || max_trace_violation > 1e-12
        || max_raw_norm <= 1e-12
        || max_stored_norm <= 1e-12
        || max_raw_invisible_component_norm <= 1e-12
        || !std::isfinite(direct_total)
        || !std::isfinite(raw_total)
        || !std::isfinite(stored_total)) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF c_sw = 0 Wilson all-link direction-independent storage "
            "test failed: Ls = ", Ls,
            ", c_sw = ", csw,
            ", inspectedLinks = ", inspected_links,
            ", expectedLinks = ", expected_links,
            ", overwrittenSentinelLinks = ", overwritten_sentinel_links,
            ", invalidLinks = ", invalid_links,
            ", missingFinalizeLinks = ", missing_finalize_links,
            ", duplicateFinalizeLinks = ", duplicate_finalize_links,
            ", missingWilsonTermLinks = ", missing_wilson_term_links,
            ", duplicateWilsonTermLinks = ", duplicate_wilson_term_links,
            ", unexpectedCloverTermLinks = ",
            unexpected_clover_term_links,
            ", unexpectedCloverPathLinks = ",
            unexpected_clover_path_links,
            ", direct = ", direct_total,
            ", raw = ", raw_total,
            ", stored = ", stored_total,
            ", directRawTotalAbsDiff = ", direct_raw_total_abs_diff,
            ", directStoredTotalAbsDiff = ",
            direct_stored_total_abs_diff,
            ", maxDirectRawAbsDiff = ", max_direct_raw_abs_diff,
            ", maxDirectStoredAbsDiff = ", max_direct_stored_abs_diff,
            ", maxZeroSentinelOutputDiff = ",
            max_zero_sentinel_output_diff,
            ", maxRepeatedOutputDiff = ", max_repeated_output_diff,
            ", maxZeroSentinelRawDiff = ", max_zero_sentinel_raw_diff,
            ", maxRepeatedRawDiff = ", max_repeated_raw_diff,
            ", maxRawCloverNorm = ", max_raw_clover_norm,
            ", maxAntiHermitianViolation = ",
            max_anti_hermitian_violation,
            ", maxTraceViolation = ", max_trace_violation,
            ", maxRawNorm = ", max_raw_norm,
            ", maxStoredNorm = ", max_stored_norm,
            ", maxRawInvisibleComponentNorm = ",
            max_raw_invisible_component_norm,
            ", forceMaxResidual = ", max_force_workspace_residual));
    }

    rootLogger.info(
        "MDWF c_sw = 0 Wilson all-link direction-independent storage test "
        "passed with Ls = ", Ls,
        ", links = ", inspected_links,
        ", rationalTermsPerLink = ", workspace.size(),
        ", finalizationsPerLink = 1",
        ", overwrittenSentinelLinks = ", overwritten_sentinel_links,
        ", missingFinalizeLinks = ", missing_finalize_links,
        ", duplicateFinalizeLinks = ", duplicate_finalize_links,
        ", missingWilsonTermLinks = ", missing_wilson_term_links,
        ", duplicateWilsonTermLinks = ", duplicate_wilson_term_links,
        ", unexpectedCloverTermLinks = ", unexpected_clover_term_links,
        ", unexpectedCloverPathLinks = ", unexpected_clover_path_links,
        ", direct = ", direct_total,
        ", raw = ", raw_total,
        ", stored = ", stored_total,
        ", directRawTotalAbsDiff = ", direct_raw_total_abs_diff,
        ", directStoredTotalAbsDiff = ", direct_stored_total_abs_diff,
        ", maxDirectRawAbsDiff = ", max_direct_raw_abs_diff,
        ", maxDirectStoredAbsDiff = ", max_direct_stored_abs_diff,
        ", maxZeroSentinelOutputDiff = ", max_zero_sentinel_output_diff,
        ", maxRepeatedOutputDiff = ", max_repeated_output_diff,
        ", maxZeroSentinelRawDiff = ", max_zero_sentinel_raw_diff,
        ", maxRepeatedRawDiff = ", max_repeated_raw_diff,
        ", maxRawCloverNorm = ", max_raw_clover_norm,
        ", maxAntiHermitianViolation = ", max_anti_hermitian_violation,
        ", maxTraceViolation = ", max_trace_violation,
        ", maxRawNorm = ", max_raw_norm,
        ", maxStoredNorm = ", max_stored_norm,
        ", maxRawInvisibleComponentNorm = ",
        max_raw_invisible_component_norm,
        ", forceMaxResidual = ", max_force_workspace_residual);
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);

        LatticeParameters param;
        CommunicationBase comm_base(&argc, &argv, true);
        param.readfile(
            comm_base, "../parameter/tests/mdwfFifthDimTest.param", argc, argv);
        comm_base.init(param.nodeDim());

        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, comm_base);

        runMDWFWilsonAllLinkDirectionIndependentStorageCsw0Test<8>(
            comm_base);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
