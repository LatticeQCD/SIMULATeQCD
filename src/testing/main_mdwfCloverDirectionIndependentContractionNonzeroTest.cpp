/*
 * MDWF nonzero-c_sw selected-link direction-independent contraction test.
 *
 * This reconstructs separate test-only Wilson, clover, and total su(3)
 * contraction matrices from an explicit eight-direction basis.  A held-out
 * mixed direction validates each split component, their matrix sum, and the
 * total centered action finite difference.  It also derives one left-oriented
 * raw clover path matrix and independently derives the raw Wilson matrix.
 * After all slice and rational accumulation, it projects each component and
 * their raw sum exactly once, verifying TA(B_W + B_C) = TA(B_W) + TA(B_C)
 * against the independent Wilson, clover, and total reconstructions.  It does
 * not select a production force field, ipdot convention, HMC sign, all-link
 * matrix accumulator, or MPI ownership implementation.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFAdjointOperator.h"
#include "../experimental/mdwf/MDWFAllLinkWilsonContraction.h"
#include "../experimental/mdwf/MDWFCloverAllLinkContraction.h"
#include "../experimental/mdwf/MDWFDirectionIndependentContraction.h"
#include "../experimental/mdwf/MDWFFermionForceWorkspace.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"
#include "../experimental/mdwf/MDWFRationalCoefficientAdapter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFCloverDirectionIndependentNonzeroSource {
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

template<size_t HaloDepth, size_t Ls>
class MDWFCloverDirectionIndependentNonzeroActionEvaluator {
public:
    using ForwardOperator
        = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using AdjointOperator
        = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using Adapter
        = MDWFCoupledSolverAdapter<
            double, HaloDepth, HaloDepth, Ls, NormalOperator>;
    using Spinor = typename NormalOperator::Spinor;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;

private:
    CommunicationBase &_comm_base;
    Spinor &_field;
    MDWFRationalCoefficients<double> _coefficients;
    MDWFFifthDimCoefficients<double> _fifth_coeff;
    double _mass;
    double _csw;
    int _max_iter;
    double _precision;

public:
    MDWFCloverDirectionIndependentNonzeroActionEvaluator(
        CommunicationBase &comm_base,
        Spinor &field,
        const MDWFRationalCoefficients<double> &coefficients,
        MDWFFifthDimCoefficients<double> fifth_coeff,
        double mass,
        double csw,
        int max_iter,
        double precision)
        : _comm_base(comm_base),
          _field(field),
          _coefficients(coefficients),
          _fifth_coeff(fifth_coeff),
          _mass(mass),
          _csw(csw),
          _max_iter(max_iter),
          _precision(precision) {}

    MDWFFiniteDifferenceActionValue<double> operator()(Gauge &gauge) {
        ForwardOperator forward(
            gauge, _fifth_coeff, _mass, _csw,
            "MDWF_clover_direction_independent_nonzero_action_forward");
        AdjointOperator adjoint(
            gauge, _fifth_coeff, _mass, _csw,
            "MDWF_clover_direction_independent_nonzero_action_adjoint");
        NormalOperator normal(
            _comm_base, forward, adjoint,
            "MDWF_clover_direction_independent_nonzero_action_normal");
        Adapter adapter(normal);
        Spinor action_workspace(
            _comm_base,
            "MDWF_clover_direction_independent_nonzero_action_workspace");

        const MDWFRationalActionResult<double> action_result
            = computeMDWFRationalAction<double, Adapter>(
                adapter, action_workspace, _field, _coefficients,
                _max_iter, _precision,
                "MDWF_clover_direction_independent_nonzero_action");

        return makeMDWFFiniteDifferenceActionValue(action_result);
    }
};

struct MDWFCloverDirectionIndependentDerivatives {
    double wilson;
    double clover;

    double total() const {
        return wilson + clover;
    }
};

template<size_t Ls>
void runMDWFCloverDirectionIndependentContractionNonzeroTest(
    CommunicationBase &comm_base) {

    const size_t HaloDepth = 2;
    const double mass = 4.0;
    const double csw = 0.5;
    typedef GIndexer<All, HaloDepth> GInd;

    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using ForwardOperator
        = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using AdjointOperator
        = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using Workspace
        = MDWFFermionForceWorkspace<
            double, HaloDepth, HaloDepth, Ls, NormalOperator, ForwardOperator>;
    using Spinor = typename NormalOperator::Spinor;
    using HostSpinor = MDWFSpinor<double, false, All, HaloDepth, Ls>;

    const LatticeData lat = GInd::getLatData();
    if (lat.vol4 != lat.globvol4) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF nonzero-c_sw direction-independent contraction test is "
            "single-rank only: local volume = ", lat.vol4,
            ", global volume = ", lat.globvol4));
    }

    const MDWFExplicitRationalInput<double> action_input{
        "clover_direction_independent_nonzero_action_coefficients",
        MDWFRationalCoefficientRole::Action,
        0.125,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    const MDWFExplicitRationalInput<double> force_input{
        "clover_direction_independent_nonzero_force_coefficients",
        MDWFRationalCoefficientRole::Force,
        0.0,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    const MDWFRationalCoefficients<double> action_coefficients
        = makeMDWFRationalCoefficients(action_input);
    const MDWFRationalCoefficients<double> force_coefficients
        = makeMDWFRationalCoefficients(force_input);

    Gauge base_gauge(
        comm_base, "MDWF_clover_direction_independent_nonzero_base_gauge");
    Gauge gauge_plus(
        comm_base, "MDWF_clover_direction_independent_nonzero_gauge_plus");
    Gauge gauge_minus(
        comm_base, "MDWF_clover_direction_independent_nonzero_gauge_minus");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260514);
    d_rand = h_rand;
    base_gauge.random(d_rand.state);
    base_gauge.updateAll();

    Spinor field(
        comm_base, "MDWF_clover_direction_independent_nonzero_field");
    field.template iterateOverBulk<>(
        FillMDWFCloverDirectionIndependentNonzeroSource<
            double, All, HaloDepth, Ls>());
    field.updateAll();
    Spinor force_field(
        comm_base,
        "MDWF_clover_direction_independent_nonzero_force_field");
    force_field = field;
    force_field.updateAll();

    const MDWFFifthDimCoefficients<double> fifth_coeff(
        1.0, -0.05, -0.05, 0.0, 0.0);
    const MDWFFiniteDifferenceProbe<double> probe{
        2, 2, 2, 2,
        1,
        0,
        1e-4,
        MDWFFiniteDifferenceMultiplicationSide::Left
    };
    const gSite selected_site
        = GInd::getSite(probe.x, probe.y, probe.z, probe.t);
    const gSiteMu selected_link
        = GInd::getSiteMu(selected_site, probe.mu);

    const std::array<SU3<double>, MDWFDirectionIndependentBasisSize> basis
        = mdwfDirectionIndependentBasis<double>();
    const MDWFDirectionIndependentGramMatrix gram
        = mdwfMeasureDirectionIndependentGramMatrix(basis);

    double gram_minimum_diagonal = std::numeric_limits<double>::infinity();
    double gram_maximum_diagonal = -std::numeric_limits<double>::infinity();
    double gram_maximum_off_diagonal = 0.0;
    double gram_maximum_asymmetry = 0.0;
    bool gram_entries_finite = true;
    double basis_minimum_norm = std::numeric_limits<double>::infinity();
    double basis_maximum_norm = 0.0;
    double basis_maximum_anti_hermitian_violation = 0.0;
    double basis_maximum_trace_violation = 0.0;
    for (size_t row = 0; row < MDWFDirectionIndependentBasisSize; row++) {
        const double basis_norm = -gram[row][row];
        basis_minimum_norm = std::min(basis_minimum_norm, basis_norm);
        basis_maximum_norm = std::max(basis_maximum_norm, basis_norm);
        basis_maximum_anti_hermitian_violation = std::max(
            basis_maximum_anti_hermitian_violation,
            static_cast<double>(infnorm(basis[row] + dagger(basis[row]))));
        basis_maximum_trace_violation = std::max(
            basis_maximum_trace_violation,
            static_cast<double>(abs(tr_c(basis[row]))));

        for (size_t column = 0;
             column < MDWFDirectionIndependentBasisSize;
             column++) {
            gram_entries_finite
                = gram_entries_finite
                  && std::isfinite(gram[row][column]);
            gram_maximum_asymmetry = std::max(
                gram_maximum_asymmetry,
                std::abs(gram[row][column] - gram[column][row]));
            if (row == column) {
                gram_minimum_diagonal
                    = std::min(gram_minimum_diagonal, gram[row][column]);
                gram_maximum_diagonal
                    = std::max(gram_maximum_diagonal, gram[row][column]);
            } else {
                gram_maximum_off_diagonal = std::max(
                    gram_maximum_off_diagonal,
                    std::abs(gram[row][column]));
            }
        }
    }

    std::array<double, MDWFDirectionIndependentBasisSize>
        mixed_coefficients{{
            0.37, -0.21, 0.43, 0.19,
            -0.31, 0.27, 0.41, -0.17
        }};
    SU3<double> mixed_direction = su3_zero<double>();
    for (size_t generator = 0;
         generator < MDWFDirectionIndependentBasisSize;
         generator++) {
        mixed_direction
            += mixed_coefficients[generator] * basis[generator];
    }
    const double mixed_raw_norm
        = -static_cast<double>(real(tr_c(mixed_direction, mixed_direction)));
    if (!std::isfinite(mixed_raw_norm) || mixed_raw_norm <= 0.0) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF nonzero-c_sw mixed direction has invalid raw norm ",
            mixed_raw_norm));
    }
    const double mixed_inverse_norm = 1.0 / std::sqrt(mixed_raw_norm);
    mixed_direction = mixed_inverse_norm * mixed_direction;
    for (double &coefficient : mixed_coefficients) {
        coefficient *= mixed_inverse_norm;
    }
    const double mixed_direction_norm
        = -static_cast<double>(real(tr_c(mixed_direction, mixed_direction)));
    const double mixed_anti_hermitian_violation = static_cast<double>(
        infnorm(mixed_direction + dagger(mixed_direction)));
    const double mixed_trace_violation
        = static_cast<double>(abs(tr_c(mixed_direction)));

    MDWFCloverDirectionIndependentNonzeroActionEvaluator<HaloDepth, Ls>
        action_evaluator(
            comm_base, field, action_coefficients, fifth_coeff,
            mass, csw, 512, 1e-8);
    std::array<
        MDWFFiniteDifferenceResult<double>,
        MDWFDirectionIndependentBasisSize> basis_finite_differences;
    double max_action_residual = 0.0;
    double max_action_imag_relative = 0.0;
    bool finite_differences_converged = true;

    for (size_t generator = 0;
         generator < MDWFDirectionIndependentBasisSize;
         generator++) {
        basis_finite_differences[generator]
            = evaluateMDWFSingleLinkExplicitDirectionFiniteDifference(
                gauge_plus, gauge_minus, base_gauge, probe,
                basis[generator], action_evaluator);
        max_action_residual = std::max(
            max_action_residual,
            basis_finite_differences[generator].max_shifted_residual);
        max_action_imag_relative = std::max(
            max_action_imag_relative,
            basis_finite_differences[generator].action_imag_relative);
        finite_differences_converged
            = finite_differences_converged
              && basis_finite_differences[generator].converged;
    }
    const MDWFFiniteDifferenceResult<double> mixed_finite_difference
        = evaluateMDWFSingleLinkExplicitDirectionFiniteDifference(
            gauge_plus, gauge_minus, base_gauge, probe,
            mixed_direction, action_evaluator);
    max_action_residual = std::max(
        max_action_residual,
        mixed_finite_difference.max_shifted_residual);
    max_action_imag_relative = std::max(
        max_action_imag_relative,
        mixed_finite_difference.action_imag_relative);
    finite_differences_converged
        = finite_differences_converged && mixed_finite_difference.converged;

    HostGauge gauge_host(
        comm_base, "MDWF_clover_direction_independent_nonzero_gauge_host");
    HostGauge mixed_plus_host(
        comm_base,
        "MDWF_clover_direction_independent_nonzero_mixed_plus_host");
    gauge_host = base_gauge;
    mixed_plus_host = gauge_plus;
    const SU3Accessor<double, R18> gauge_acc = gauge_host.getAccessor();
    const SU3Accessor<double, R18> mixed_plus_acc
        = mixed_plus_host.getAccessor();
    size_t perturbed_link_count = 0;
    double selected_link_perturbation = 0.0;
    double max_off_probe_perturbation = 0.0;

    for (size_t site_index = 0; site_index < lat.vol4; site_index++) {
        const gSite site = GInd::getSite(site_index);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const gSiteMu site_mu = GInd::getSiteMu(site, mu);
            const double link_difference = static_cast<double>(infnorm(
                mixed_plus_acc.getLink(site_mu)
                - gauge_acc.getLink(site_mu)));
            const bool matches_probe
                = site_mu.coord.x == probe.x
                  && site_mu.coord.y == probe.y
                  && site_mu.coord.z == probe.z
                  && site_mu.coord.t == probe.t
                  && site_mu.mu == probe.mu;
            if (link_difference > 1e-12) {
                perturbed_link_count++;
            }
            if (matches_probe) {
                selected_link_perturbation = link_difference;
            } else {
                max_off_probe_perturbation = std::max(
                    max_off_probe_perturbation, link_difference);
            }
        }
    }

    ForwardOperator forward(
        base_gauge, fifth_coeff, mass, csw,
        "MDWF_clover_direction_independent_nonzero_forward");
    AdjointOperator adjoint(
        base_gauge, fifth_coeff, mass, csw,
        "MDWF_clover_direction_independent_nonzero_adjoint");
    NormalOperator normal(
        comm_base, forward, adjoint,
        "MDWF_clover_direction_independent_nonzero_normal");
    Workspace workspace;
    workspace.prepare(
        normal, forward, force_field, force_coefficients, 512, 1e-8,
        "MDWF_clover_direction_independent_nonzero_workspace");

    double max_force_workspace_residual = 0.0;
    for (const auto &info : workspace.shiftInfo()) {
        max_force_workspace_residual = std::max(
            max_force_workspace_residual, info.residue);
    }

    std::array<
        MDWFCloverDirectionIndependentDerivatives,
        MDWFDirectionIndependentBasisSize> basis_derivatives{};
    MDWFCloverDirectionIndependentDerivatives mixed_derivatives{0.0, 0.0};
    SU3<double> raw_wilson_matrix = su3_zero<double>();
    SU3<double> raw_clover_matrix = su3_zero<double>();

    for (size_t term = 0; term < workspace.size(); term++) {
        HostSpinor chi_host(
            comm_base,
            "MDWF_clover_direction_independent_nonzero_chi_"
                + std::to_string(term));
        HostSpinor eta_host(
            comm_base,
            "MDWF_clover_direction_independent_nonzero_eta_"
                + std::to_string(term));
        chi_host = workspace.chi(term);
        eta_host = workspace.eta(term);
        const Vect12ArrayAcc<double> chi_acc = chi_host.getAccessor();
        const Vect12ArrayAcc<double> eta_acc = eta_host.getAccessor();
        const double rational_weight
            = -2.0 * force_coefficients.numerator[term];

        for (size_t generator = 0;
             generator < MDWFDirectionIndependentBasisSize;
             generator++) {
            basis_derivatives[generator].wilson
                += rational_weight
                   * mdwfAllLinkWilsonContractionTerm<HaloDepth, Ls>(
                       chi_acc, eta_acc, gauge_acc, selected_site,
                       probe.mu, basis[generator]);
            basis_derivatives[generator].clover
                += rational_weight
                   * mdwfSelectedLinkCloverContractionTerm<HaloDepth, Ls>(
                       gauge_acc, chi_acc, eta_acc, selected_link,
                       basis[generator], probe.multiplication_side, csw);
        }

        mixed_derivatives.wilson
            += rational_weight
               * mdwfAllLinkWilsonContractionTerm<HaloDepth, Ls>(
                   chi_acc, eta_acc, gauge_acc, selected_site,
                   probe.mu, mixed_direction);
        mixed_derivatives.clover
            += rational_weight
               * mdwfSelectedLinkCloverContractionTerm<HaloDepth, Ls>(
                   gauge_acc, chi_acc, eta_acc, selected_link,
                   mixed_direction, probe.multiplication_side, csw);
        raw_clover_matrix
            += rational_weight
               * mdwfSelectedLinkCloverLeftRawContractionMatrixTerm<
                   HaloDepth, Ls>(
                   gauge_acc, chi_acc, eta_acc, selected_link, csw);
        raw_wilson_matrix
            += rational_weight
               * mdwfWilsonLeftRawContractionMatrixTerm<HaloDepth, Ls>(
                   chi_acc, eta_acc, gauge_acc, selected_site, probe.mu);
    }

    std::array<double, MDWFDirectionIndependentBasisSize>
        wilson_basis_contractions{};
    std::array<double, MDWFDirectionIndependentBasisSize>
        clover_basis_contractions{};
    std::array<double, MDWFDirectionIndependentBasisSize>
        total_basis_contractions{};
    double analytic_scale = 1.0;
    for (size_t generator = 0;
         generator < MDWFDirectionIndependentBasisSize;
         generator++) {
        wilson_basis_contractions[generator]
            = basis_derivatives[generator].wilson;
        clover_basis_contractions[generator]
            = basis_derivatives[generator].clover;
        total_basis_contractions[generator]
            = basis_derivatives[generator].total();
        analytic_scale = std::max(
            analytic_scale,
            std::max(std::abs(basis_derivatives[generator].wilson),
                     std::max(std::abs(basis_derivatives[generator].clover),
                              std::abs(basis_derivatives[generator].total()))));
    }
    analytic_scale = std::max(
        analytic_scale,
        std::max(std::abs(mixed_derivatives.wilson),
                 std::max(std::abs(mixed_derivatives.clover),
                          std::abs(mixed_derivatives.total()))));
    const double algebra_tolerance = 1e-10 * analytic_scale;

    const MDWFDirectionIndependentReconstruction<double>
        wilson_reconstruction = mdwfReconstructDirectionIndependentMatrix(
            basis, gram, wilson_basis_contractions);
    const MDWFDirectionIndependentReconstruction<double>
        clover_reconstruction = mdwfReconstructDirectionIndependentMatrix(
            basis, gram, clover_basis_contractions);
    const MDWFDirectionIndependentReconstruction<double>
        total_reconstruction = mdwfReconstructDirectionIndependentMatrix(
            basis, gram, total_basis_contractions);
    const SU3<double> raw_total_matrix
        = raw_wilson_matrix + raw_clover_matrix;
    SU3<double> projected_raw_wilson_matrix = raw_wilson_matrix;
    projected_raw_wilson_matrix.TA();
    SU3<double> projected_raw_clover_matrix = raw_clover_matrix;
    projected_raw_clover_matrix.TA();
    SU3<double> projected_raw_total_matrix = raw_total_matrix;
    projected_raw_total_matrix.TA();
    const SU3<double> projected_component_sum
        = projected_raw_wilson_matrix + projected_raw_clover_matrix;

    const SU3<double> split_matrix_sum
        = wilson_reconstruction.matrix + clover_reconstruction.matrix;
    const double split_matrix_sum_abs_diff = static_cast<double>(infnorm(
        total_reconstruction.matrix - split_matrix_sum));
    double max_basis_wilson_reconstruction_abs_diff = 0.0;
    double max_basis_clover_reconstruction_abs_diff = 0.0;
    double max_basis_total_reconstruction_abs_diff = 0.0;
    double max_basis_raw_wilson_abs_diff = 0.0;
    double max_basis_projected_wilson_abs_diff = 0.0;
    double max_basis_raw_clover_abs_diff = 0.0;
    double max_basis_projected_clover_abs_diff = 0.0;
    double max_basis_raw_total_abs_diff = 0.0;
    double max_basis_projected_total_abs_diff = 0.0;
    double max_basis_split_abs_diff = 0.0;
    double max_basis_finite_difference_abs_diff = 0.0;
    double max_basis_finite_difference_rel_diff = 0.0;
    bool basis_finite_differences_passed = true;
    bool basis_values_finite = true;

    for (size_t generator = 0;
         generator < MDWFDirectionIndependentBasisSize;
         generator++) {
        const double reconstructed_wilson = static_cast<double>(
            real(tr_c(basis[generator], wilson_reconstruction.matrix)));
        const double reconstructed_clover = static_cast<double>(
            real(tr_c(basis[generator], clover_reconstruction.matrix)));
        const double reconstructed_total = static_cast<double>(
            real(tr_c(basis[generator], total_reconstruction.matrix)));
        const double raw_wilson = static_cast<double>(
            real(tr_c(basis[generator], raw_wilson_matrix)));
        const double projected_wilson = static_cast<double>(
            real(tr_c(basis[generator], projected_raw_wilson_matrix)));
        const double raw_clover = static_cast<double>(
            real(tr_c(basis[generator], raw_clover_matrix)));
        const double projected_clover = static_cast<double>(
            real(tr_c(basis[generator], projected_raw_clover_matrix)));
        const double raw_total = static_cast<double>(
            real(tr_c(basis[generator], raw_total_matrix)));
        const double projected_total = static_cast<double>(
            real(tr_c(basis[generator], projected_raw_total_matrix)));
        const double raw_wilson_abs_diff = std::abs(
            raw_wilson - basis_derivatives[generator].wilson);
        const double projected_wilson_abs_diff = std::abs(
            projected_wilson - basis_derivatives[generator].wilson);
        const double raw_clover_abs_diff = std::abs(
            raw_clover - basis_derivatives[generator].clover);
        const double projected_clover_abs_diff = std::abs(
            projected_clover - basis_derivatives[generator].clover);
        const double raw_total_abs_diff = std::abs(
            raw_total - basis_derivatives[generator].total());
        const double projected_total_abs_diff = std::abs(
            projected_total - basis_derivatives[generator].total());
        const double split_abs_diff = std::abs(
            basis_derivatives[generator].total()
            - basis_derivatives[generator].wilson
            - basis_derivatives[generator].clover);
        const double finite_difference_abs_diff = std::abs(
            basis_finite_differences[generator].derivative
            - basis_derivatives[generator].total());
        const double finite_difference_scale = std::max(
            1.0,
            std::abs(basis_finite_differences[generator].derivative));
        const double finite_difference_rel_diff
            = finite_difference_abs_diff / finite_difference_scale;
        const double finite_difference_tolerance
            = 5e-3 + 5e-4 * finite_difference_scale;

        max_basis_wilson_reconstruction_abs_diff = std::max(
            max_basis_wilson_reconstruction_abs_diff,
            std::abs(reconstructed_wilson
                     - basis_derivatives[generator].wilson));
        max_basis_clover_reconstruction_abs_diff = std::max(
            max_basis_clover_reconstruction_abs_diff,
            std::abs(reconstructed_clover
                     - basis_derivatives[generator].clover));
        max_basis_total_reconstruction_abs_diff = std::max(
            max_basis_total_reconstruction_abs_diff,
            std::abs(reconstructed_total
                     - basis_derivatives[generator].total()));
        max_basis_raw_wilson_abs_diff = std::max(
            max_basis_raw_wilson_abs_diff, raw_wilson_abs_diff);
        max_basis_projected_wilson_abs_diff = std::max(
            max_basis_projected_wilson_abs_diff,
            projected_wilson_abs_diff);
        max_basis_raw_clover_abs_diff = std::max(
            max_basis_raw_clover_abs_diff, raw_clover_abs_diff);
        max_basis_projected_clover_abs_diff = std::max(
            max_basis_projected_clover_abs_diff,
            projected_clover_abs_diff);
        max_basis_raw_total_abs_diff = std::max(
            max_basis_raw_total_abs_diff, raw_total_abs_diff);
        max_basis_projected_total_abs_diff = std::max(
            max_basis_projected_total_abs_diff,
            projected_total_abs_diff);
        max_basis_split_abs_diff = std::max(
            max_basis_split_abs_diff, split_abs_diff);
        max_basis_finite_difference_abs_diff = std::max(
            max_basis_finite_difference_abs_diff,
            finite_difference_abs_diff);
        max_basis_finite_difference_rel_diff = std::max(
            max_basis_finite_difference_rel_diff,
            finite_difference_rel_diff);
        basis_finite_differences_passed
            = basis_finite_differences_passed
              && finite_difference_abs_diff <= finite_difference_tolerance;
        basis_values_finite
            = basis_values_finite
              && std::isfinite(basis_derivatives[generator].wilson)
              && std::isfinite(basis_derivatives[generator].clover)
              && std::isfinite(basis_derivatives[generator].total())
              && std::isfinite(reconstructed_wilson)
              && std::isfinite(reconstructed_clover)
              && std::isfinite(reconstructed_total)
              && std::isfinite(raw_wilson)
              && std::isfinite(projected_wilson)
              && std::isfinite(raw_clover)
              && std::isfinite(projected_clover)
              && std::isfinite(raw_total)
              && std::isfinite(projected_total)
              && std::isfinite(
                  basis_finite_differences[generator].derivative);

        rootLogger.info(
            "MDWF nonzero-c_sw direction-independent basis probe with "
            "generator = ", generator,
            ", norm = ", -gram[generator][generator],
            ", finiteDifference = ",
            basis_finite_differences[generator].derivative,
            ", Wilson = ", basis_derivatives[generator].wilson,
            ", clover = ", basis_derivatives[generator].clover,
            ", analytic = ", basis_derivatives[generator].total(),
            ", reconstructedWilson = ", reconstructed_wilson,
            ", reconstructedClover = ", reconstructed_clover,
            ", reconstructedTotal = ", reconstructed_total,
            ", rawWilson = ", raw_wilson,
            ", projectedRawWilson = ", projected_wilson,
            ", rawClover = ", raw_clover,
            ", projectedRawClover = ", projected_clover,
            ", rawTotal = ", raw_total,
            ", projectedRawTotal = ", projected_total,
            ", finiteDifferenceAbsDiff = ", finite_difference_abs_diff,
            ", finiteDifferenceRelDiff = ", finite_difference_rel_diff,
            ", rawWilsonAbsDiff = ", raw_wilson_abs_diff,
            ", projectedRawWilsonAbsDiff = ",
            projected_wilson_abs_diff,
            ", rawCloverAbsDiff = ", raw_clover_abs_diff,
            ", projectedRawCloverAbsDiff = ",
            projected_clover_abs_diff,
            ", rawTotalAbsDiff = ", raw_total_abs_diff,
            ", projectedRawTotalAbsDiff = ",
            projected_total_abs_diff);
    }

    const double mixed_reconstructed_wilson = static_cast<double>(
        real(tr_c(mixed_direction, wilson_reconstruction.matrix)));
    const double mixed_reconstructed_clover = static_cast<double>(
        real(tr_c(mixed_direction, clover_reconstruction.matrix)));
    const double mixed_reconstructed_total = static_cast<double>(
        real(tr_c(mixed_direction, total_reconstruction.matrix)));
    const double mixed_raw_wilson = static_cast<double>(
        real(tr_c(mixed_direction, raw_wilson_matrix)));
    const double mixed_projected_wilson = static_cast<double>(
        real(tr_c(mixed_direction, projected_raw_wilson_matrix)));
    const double mixed_raw_clover = static_cast<double>(
        real(tr_c(mixed_direction, raw_clover_matrix)));
    const double mixed_projected_clover = static_cast<double>(
        real(tr_c(mixed_direction, projected_raw_clover_matrix)));
    const double mixed_raw_total = static_cast<double>(
        real(tr_c(mixed_direction, raw_total_matrix)));
    const double mixed_projected_total = static_cast<double>(
        real(tr_c(mixed_direction, projected_raw_total_matrix)));
    MDWFCloverDirectionIndependentDerivatives mixed_basis_linear{0.0, 0.0};
    for (size_t generator = 0;
         generator < MDWFDirectionIndependentBasisSize;
         generator++) {
        mixed_basis_linear.wilson
            += mixed_coefficients[generator]
               * basis_derivatives[generator].wilson;
        mixed_basis_linear.clover
            += mixed_coefficients[generator]
               * basis_derivatives[generator].clover;
    }

    const double mixed_finite_difference_abs_diff = std::abs(
        mixed_finite_difference.derivative - mixed_derivatives.total());
    const double mixed_finite_difference_scale = std::max(
        1.0, std::abs(mixed_finite_difference.derivative));
    const double mixed_finite_difference_rel_diff
        = mixed_finite_difference_abs_diff / mixed_finite_difference_scale;
    const double mixed_finite_difference_tolerance
        = 5e-3 + 5e-4 * mixed_finite_difference_scale;
    const double mixed_wilson_reconstruction_abs_diff = std::abs(
        mixed_reconstructed_wilson - mixed_derivatives.wilson);
    const double mixed_clover_reconstruction_abs_diff = std::abs(
        mixed_reconstructed_clover - mixed_derivatives.clover);
    const double mixed_total_reconstruction_abs_diff = std::abs(
        mixed_reconstructed_total - mixed_derivatives.total());
    const double mixed_raw_wilson_abs_diff = std::abs(
        mixed_raw_wilson - mixed_derivatives.wilson);
    const double mixed_projected_wilson_abs_diff = std::abs(
        mixed_projected_wilson - mixed_derivatives.wilson);
    const double mixed_raw_clover_abs_diff = std::abs(
        mixed_raw_clover - mixed_derivatives.clover);
    const double mixed_projected_clover_abs_diff = std::abs(
        mixed_projected_clover - mixed_derivatives.clover);
    const double mixed_raw_total_abs_diff = std::abs(
        mixed_raw_total - mixed_derivatives.total());
    const double mixed_projected_total_abs_diff = std::abs(
        mixed_projected_total - mixed_derivatives.total());
    const double mixed_wilson_linearity_abs_diff = std::abs(
        mixed_basis_linear.wilson - mixed_derivatives.wilson);
    const double mixed_clover_linearity_abs_diff = std::abs(
        mixed_basis_linear.clover - mixed_derivatives.clover);
    const double mixed_total_linearity_abs_diff = std::abs(
        mixed_basis_linear.total() - mixed_derivatives.total());

    const double wilson_matrix_norm
        = static_cast<double>(infnorm(wilson_reconstruction.matrix));
    const double clover_matrix_norm
        = static_cast<double>(infnorm(clover_reconstruction.matrix));
    const double total_matrix_norm
        = static_cast<double>(infnorm(total_reconstruction.matrix));
    const double raw_wilson_matrix_norm
        = static_cast<double>(infnorm(raw_wilson_matrix));
    const double raw_wilson_invisible_component_norm
        = static_cast<double>(infnorm(
            raw_wilson_matrix - projected_raw_wilson_matrix));
    const double projected_raw_wilson_matrix_norm
        = static_cast<double>(infnorm(projected_raw_wilson_matrix));
    const double projected_raw_wilson_matrix_abs_diff
        = static_cast<double>(infnorm(
            projected_raw_wilson_matrix - wilson_reconstruction.matrix));
    const double raw_clover_matrix_norm
        = static_cast<double>(infnorm(raw_clover_matrix));
    const double raw_clover_invisible_component_norm
        = static_cast<double>(infnorm(
            raw_clover_matrix - projected_raw_clover_matrix));
    const double projected_raw_clover_matrix_norm
        = static_cast<double>(infnorm(projected_raw_clover_matrix));
    const double projected_raw_clover_matrix_abs_diff
        = static_cast<double>(infnorm(
            projected_raw_clover_matrix - clover_reconstruction.matrix));
    const double projected_raw_clover_anti_hermitian_violation
        = static_cast<double>(infnorm(
            projected_raw_clover_matrix
            + dagger(projected_raw_clover_matrix)));
    const double projected_raw_clover_trace_violation
        = static_cast<double>(abs(tr_c(projected_raw_clover_matrix)));
    const double raw_total_matrix_norm
        = static_cast<double>(infnorm(raw_total_matrix));
    const double raw_total_invisible_component_norm
        = static_cast<double>(infnorm(
            raw_total_matrix - projected_raw_total_matrix));
    const double projected_raw_total_matrix_norm
        = static_cast<double>(infnorm(projected_raw_total_matrix));
    const double projected_raw_total_matrix_abs_diff
        = static_cast<double>(infnorm(
            projected_raw_total_matrix - total_reconstruction.matrix));
    const double projection_linearity_abs_diff
        = static_cast<double>(infnorm(
            projected_raw_total_matrix - projected_component_sum));
    const double projected_component_sum_abs_diff
        = static_cast<double>(infnorm(
            projected_component_sum - split_matrix_sum));
    const double projected_raw_max_anti_hermitian_violation = std::max(
        static_cast<double>(infnorm(
            projected_raw_wilson_matrix
            + dagger(projected_raw_wilson_matrix))),
        std::max(
            projected_raw_clover_anti_hermitian_violation,
            static_cast<double>(infnorm(
                projected_raw_total_matrix
                + dagger(projected_raw_total_matrix)))));
    const double projected_raw_max_trace_violation = std::max(
        static_cast<double>(abs(tr_c(projected_raw_wilson_matrix))),
        std::max(
            projected_raw_clover_trace_violation,
            static_cast<double>(abs(tr_c(projected_raw_total_matrix)))));
    const double max_matrix_anti_hermitian_violation = std::max(
        static_cast<double>(infnorm(
            wilson_reconstruction.matrix
            + dagger(wilson_reconstruction.matrix))),
        std::max(
            static_cast<double>(infnorm(
                clover_reconstruction.matrix
                + dagger(clover_reconstruction.matrix))),
            static_cast<double>(infnorm(
                total_reconstruction.matrix
                + dagger(total_reconstruction.matrix)))));
    const double max_matrix_trace_violation = std::max(
        static_cast<double>(abs(tr_c(wilson_reconstruction.matrix))),
        std::max(
            static_cast<double>(abs(tr_c(clover_reconstruction.matrix))),
            static_cast<double>(abs(tr_c(total_reconstruction.matrix)))));
    const double minimum_absolute_pivot = std::min(
        wilson_reconstruction.minimum_absolute_pivot,
        std::min(clover_reconstruction.minimum_absolute_pivot,
                 total_reconstruction.minimum_absolute_pivot));
    const bool mixed_values_finite
        = std::isfinite(mixed_finite_difference.derivative)
          && std::isfinite(mixed_derivatives.wilson)
          && std::isfinite(mixed_derivatives.clover)
          && std::isfinite(mixed_derivatives.total())
          && std::isfinite(mixed_reconstructed_wilson)
          && std::isfinite(mixed_reconstructed_clover)
          && std::isfinite(mixed_reconstructed_total)
          && std::isfinite(mixed_raw_wilson)
          && std::isfinite(mixed_projected_wilson)
          && std::isfinite(mixed_raw_clover)
          && std::isfinite(mixed_projected_clover)
          && std::isfinite(mixed_raw_total)
          && std::isfinite(mixed_projected_total)
          && std::isfinite(mixed_basis_linear.wilson)
          && std::isfinite(mixed_basis_linear.clover)
          && std::isfinite(mixed_basis_linear.total());
    const bool scalar_diagnostics_finite
        = std::isfinite(max_action_residual)
          && std::isfinite(max_force_workspace_residual)
          && std::isfinite(max_action_imag_relative)
          && std::isfinite(selected_link_perturbation)
          && std::isfinite(max_off_probe_perturbation)
          && std::isfinite(basis_minimum_norm)
          && std::isfinite(basis_maximum_norm)
          && std::isfinite(basis_maximum_anti_hermitian_violation)
          && std::isfinite(basis_maximum_trace_violation)
          && std::isfinite(minimum_absolute_pivot)
          && std::isfinite(algebra_tolerance)
          && std::isfinite(split_matrix_sum_abs_diff)
          && std::isfinite(mixed_direction_norm)
          && std::isfinite(mixed_anti_hermitian_violation)
          && std::isfinite(mixed_trace_violation)
          && std::isfinite(max_matrix_anti_hermitian_violation)
          && std::isfinite(max_matrix_trace_violation)
          && std::isfinite(raw_wilson_matrix_norm)
          && std::isfinite(raw_wilson_invisible_component_norm)
          && std::isfinite(projected_raw_wilson_matrix_norm)
          && std::isfinite(projected_raw_wilson_matrix_abs_diff)
          && std::isfinite(raw_clover_matrix_norm)
          && std::isfinite(raw_clover_invisible_component_norm)
          && std::isfinite(projected_raw_clover_matrix_norm)
          && std::isfinite(projected_raw_clover_matrix_abs_diff)
          && std::isfinite(
              projected_raw_clover_anti_hermitian_violation)
          && std::isfinite(projected_raw_clover_trace_violation)
          && std::isfinite(raw_total_matrix_norm)
          && std::isfinite(raw_total_invisible_component_norm)
          && std::isfinite(projected_raw_total_matrix_norm)
          && std::isfinite(projected_raw_total_matrix_abs_diff)
          && std::isfinite(projection_linearity_abs_diff)
          && std::isfinite(projected_component_sum_abs_diff)
          && std::isfinite(
              projected_raw_max_anti_hermitian_violation)
          && std::isfinite(projected_raw_max_trace_violation);

    if (mdwfRationalCoefficientRoleName(action_input.role) != "action"
        || mdwfRationalCoefficientRoleName(force_input.role) != "force"
        || probe.multiplication_side
           != MDWFFiniteDifferenceMultiplicationSide::Left
        || !finite_differences_converged
        || !workspace.converged()
        || max_action_residual > 1e-8
        || max_force_workspace_residual > 1e-8
        || max_action_imag_relative > 1e-8
        || perturbed_link_count != 1
        || selected_link_perturbation <= 1e-12
        || max_off_probe_perturbation > 1e-12
        || !gram_entries_finite
        || gram_minimum_diagonal >= 0.0
        || gram_maximum_diagonal >= 0.0
        || gram_maximum_asymmetry > 1e-12
        || basis_minimum_norm <= 0.0
        || basis_maximum_anti_hermitian_violation > 1e-12
        || basis_maximum_trace_violation > 1e-12
        || minimum_absolute_pivot <= 1e-14
        || !basis_values_finite
        || !mixed_values_finite
        || !scalar_diagnostics_finite
        || !basis_finite_differences_passed
        || max_basis_wilson_reconstruction_abs_diff > algebra_tolerance
        || max_basis_clover_reconstruction_abs_diff > algebra_tolerance
        || max_basis_total_reconstruction_abs_diff > algebra_tolerance
        || max_basis_raw_wilson_abs_diff > algebra_tolerance
        || max_basis_projected_wilson_abs_diff > algebra_tolerance
        || max_basis_raw_clover_abs_diff > algebra_tolerance
        || max_basis_projected_clover_abs_diff > algebra_tolerance
        || max_basis_raw_total_abs_diff > algebra_tolerance
        || max_basis_projected_total_abs_diff > algebra_tolerance
        || max_basis_split_abs_diff > algebra_tolerance
        || split_matrix_sum_abs_diff > algebra_tolerance
        || std::abs(mixed_direction_norm - 1.0) > 1e-12
        || mixed_anti_hermitian_violation > 1e-12
        || mixed_trace_violation > 1e-12
        || mixed_finite_difference_abs_diff
           > mixed_finite_difference_tolerance
        || mixed_wilson_reconstruction_abs_diff > algebra_tolerance
        || mixed_clover_reconstruction_abs_diff > algebra_tolerance
        || mixed_total_reconstruction_abs_diff > algebra_tolerance
        || mixed_raw_wilson_abs_diff > algebra_tolerance
        || mixed_projected_wilson_abs_diff > algebra_tolerance
        || mixed_raw_clover_abs_diff > algebra_tolerance
        || mixed_projected_clover_abs_diff > algebra_tolerance
        || mixed_raw_total_abs_diff > algebra_tolerance
        || mixed_projected_total_abs_diff > algebra_tolerance
        || mixed_wilson_linearity_abs_diff > algebra_tolerance
        || mixed_clover_linearity_abs_diff > algebra_tolerance
        || mixed_total_linearity_abs_diff > algebra_tolerance
        || !std::isfinite(wilson_matrix_norm)
        || !std::isfinite(clover_matrix_norm)
        || !std::isfinite(total_matrix_norm)
        || wilson_matrix_norm <= 0.0
        || clover_matrix_norm <= 0.0
        || total_matrix_norm <= 0.0
        || max_matrix_anti_hermitian_violation > 1e-12
        || max_matrix_trace_violation > 1e-12
        || raw_wilson_matrix_norm <= 0.0
        || raw_wilson_invisible_component_norm <= 1e-12
        || projected_raw_wilson_matrix_norm <= 0.0
        || projected_raw_wilson_matrix_abs_diff > algebra_tolerance
        || raw_clover_matrix_norm <= 0.0
        || raw_clover_invisible_component_norm <= 1e-12
        || projected_raw_clover_matrix_norm <= 0.0
        || projected_raw_clover_matrix_abs_diff > algebra_tolerance
        || projected_raw_clover_anti_hermitian_violation > 1e-12
        || projected_raw_clover_trace_violation > 1e-12
        || raw_total_matrix_norm <= 0.0
        || raw_total_invisible_component_norm <= 1e-12
        || projected_raw_total_matrix_norm <= 0.0
        || projected_raw_total_matrix_abs_diff > algebra_tolerance
        || projection_linearity_abs_diff > algebra_tolerance
        || projected_component_sum_abs_diff > algebra_tolerance
        || projected_raw_max_anti_hermitian_violation > 1e-12
        || projected_raw_max_trace_violation > 1e-12) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF nonzero-c_sw direction-independent contraction test failed: "
            "Ls = ", Ls,
            ", c_sw = ", csw,
            ", link = (", probe.x, ", ", probe.y, ", ", probe.z, ", ",
            probe.t, "; mu = ", static_cast<int>(probe.mu), ")",
            ", basisDirections = ", MDWFDirectionIndependentBasisSize,
            ", perturbedLinks = ", perturbed_link_count,
            ", selectedLinkPerturbation = ", selected_link_perturbation,
            ", maxOffProbePerturbation = ", max_off_probe_perturbation,
            ", gramDiagonalRange = [", gram_minimum_diagonal, ", ",
            gram_maximum_diagonal, "]",
            ", gramMaxOffDiagonal = ", gram_maximum_off_diagonal,
            ", gramMaxAsymmetry = ", gram_maximum_asymmetry,
            ", gramMinAbsPivot = ", minimum_absolute_pivot,
            ", basisNormRange = [", basis_minimum_norm, ", ",
            basis_maximum_norm, "]",
            ", basisMaxAntiHermitianViolation = ",
            basis_maximum_anti_hermitian_violation,
            ", basisMaxTraceViolation = ",
            basis_maximum_trace_violation,
            ", maxBasisFiniteDifferenceAbsDiff = ",
            max_basis_finite_difference_abs_diff,
            ", maxBasisFiniteDifferenceRelDiff = ",
            max_basis_finite_difference_rel_diff,
            ", maxBasisWilsonReconstructionAbsDiff = ",
            max_basis_wilson_reconstruction_abs_diff,
            ", maxBasisCloverReconstructionAbsDiff = ",
            max_basis_clover_reconstruction_abs_diff,
            ", maxBasisTotalReconstructionAbsDiff = ",
            max_basis_total_reconstruction_abs_diff,
            ", maxBasisRawWilsonAbsDiff = ",
            max_basis_raw_wilson_abs_diff,
            ", maxBasisProjectedWilsonAbsDiff = ",
            max_basis_projected_wilson_abs_diff,
            ", maxBasisRawCloverAbsDiff = ",
            max_basis_raw_clover_abs_diff,
            ", maxBasisProjectedCloverAbsDiff = ",
            max_basis_projected_clover_abs_diff,
            ", maxBasisRawTotalAbsDiff = ",
            max_basis_raw_total_abs_diff,
            ", maxBasisProjectedTotalAbsDiff = ",
            max_basis_projected_total_abs_diff,
            ", splitMatrixSumAbsDiff = ", split_matrix_sum_abs_diff,
            ", mixedFiniteDifference = ",
            mixed_finite_difference.derivative,
            ", mixedWilson = ", mixed_derivatives.wilson,
            ", mixedClover = ", mixed_derivatives.clover,
            ", mixedAnalytic = ", mixed_derivatives.total(),
            ", mixedReconstructedWilson = ", mixed_reconstructed_wilson,
            ", mixedReconstructedClover = ", mixed_reconstructed_clover,
            ", mixedReconstructedTotal = ", mixed_reconstructed_total,
            ", mixedRawWilson = ", mixed_raw_wilson,
            ", mixedProjectedRawWilson = ", mixed_projected_wilson,
            ", mixedRawClover = ", mixed_raw_clover,
            ", mixedProjectedRawClover = ", mixed_projected_clover,
            ", mixedRawTotal = ", mixed_raw_total,
            ", mixedProjectedRawTotal = ", mixed_projected_total,
            ", mixedFiniteDifferenceAbsDiff = ",
            mixed_finite_difference_abs_diff,
            ", mixedFiniteDifferenceRelDiff = ",
            mixed_finite_difference_rel_diff,
            ", mixedWilsonReconstructionAbsDiff = ",
            mixed_wilson_reconstruction_abs_diff,
            ", mixedCloverReconstructionAbsDiff = ",
            mixed_clover_reconstruction_abs_diff,
            ", mixedTotalReconstructionAbsDiff = ",
            mixed_total_reconstruction_abs_diff,
            ", mixedRawWilsonAbsDiff = ",
            mixed_raw_wilson_abs_diff,
            ", mixedProjectedRawWilsonAbsDiff = ",
            mixed_projected_wilson_abs_diff,
            ", mixedRawCloverAbsDiff = ",
            mixed_raw_clover_abs_diff,
            ", mixedProjectedRawCloverAbsDiff = ",
            mixed_projected_clover_abs_diff,
            ", mixedRawTotalAbsDiff = ", mixed_raw_total_abs_diff,
            ", mixedProjectedRawTotalAbsDiff = ",
            mixed_projected_total_abs_diff,
            ", mixedWilsonLinearityAbsDiff = ",
            mixed_wilson_linearity_abs_diff,
            ", mixedCloverLinearityAbsDiff = ",
            mixed_clover_linearity_abs_diff,
            ", mixedTotalLinearityAbsDiff = ",
            mixed_total_linearity_abs_diff,
            ", matrixNorms = [", wilson_matrix_norm, ", ",
            clover_matrix_norm, ", ", total_matrix_norm, "]",
            ", maxMatrixAntiHermitianViolation = ",
            max_matrix_anti_hermitian_violation,
            ", maxMatrixTraceViolation = ", max_matrix_trace_violation,
            ", rawWilsonMatrixNorm = ", raw_wilson_matrix_norm,
            ", rawWilsonInvisibleComponentNorm = ",
            raw_wilson_invisible_component_norm,
            ", projectedRawWilsonMatrixNorm = ",
            projected_raw_wilson_matrix_norm,
            ", projectedRawWilsonMatrixAbsDiff = ",
            projected_raw_wilson_matrix_abs_diff,
            ", rawCloverMatrixNorm = ", raw_clover_matrix_norm,
            ", rawCloverInvisibleComponentNorm = ",
            raw_clover_invisible_component_norm,
            ", projectedRawCloverMatrixNorm = ",
            projected_raw_clover_matrix_norm,
            ", projectedRawCloverMatrixAbsDiff = ",
            projected_raw_clover_matrix_abs_diff,
            ", projectedRawCloverAntiHermitianViolation = ",
            projected_raw_clover_anti_hermitian_violation,
            ", projectedRawCloverTraceViolation = ",
            projected_raw_clover_trace_violation,
            ", rawTotalMatrixNorm = ", raw_total_matrix_norm,
            ", rawTotalInvisibleComponentNorm = ",
            raw_total_invisible_component_norm,
            ", projectedRawTotalMatrixNorm = ",
            projected_raw_total_matrix_norm,
            ", projectedRawTotalMatrixAbsDiff = ",
            projected_raw_total_matrix_abs_diff,
            ", projectionLinearityAbsDiff = ",
            projection_linearity_abs_diff,
            ", projectedComponentSumAbsDiff = ",
            projected_component_sum_abs_diff,
            ", projectedRawMaxAntiHermitianViolation = ",
            projected_raw_max_anti_hermitian_violation,
            ", projectedRawMaxTraceViolation = ",
            projected_raw_max_trace_violation,
            ", actionMaxResidual = ", max_action_residual,
            ", forceMaxResidual = ", max_force_workspace_residual,
            ", actionImagRel = ", max_action_imag_relative));
    }

    rootLogger.info(
        "MDWF nonzero-c_sw direction-independent contraction test passed with "
        "Ls = ", Ls,
        ", c_sw = ", csw,
        ", link = (", probe.x, ", ", probe.y, ", ", probe.z, ", ",
        probe.t, "; mu = ", static_cast<int>(probe.mu), ")",
        ", basisDirections = ", MDWFDirectionIndependentBasisSize,
        ", mixedDirections = ", 1,
        ", perturbedLinks = ", perturbed_link_count,
        ", selectedLinkPerturbation = ", selected_link_perturbation,
        ", maxOffProbePerturbation = ", max_off_probe_perturbation,
        ", gramDiagonalRange = [", gram_minimum_diagonal, ", ",
        gram_maximum_diagonal, "]",
        ", gramMaxOffDiagonal = ", gram_maximum_off_diagonal,
        ", gramMinAbsPivot = ", minimum_absolute_pivot,
        ", basisNormRange = [", basis_minimum_norm, ", ",
        basis_maximum_norm, "]",
        ", maxBasisFiniteDifferenceAbsDiff = ",
        max_basis_finite_difference_abs_diff,
        ", maxBasisFiniteDifferenceRelDiff = ",
        max_basis_finite_difference_rel_diff,
        ", maxBasisWilsonReconstructionAbsDiff = ",
        max_basis_wilson_reconstruction_abs_diff,
        ", maxBasisCloverReconstructionAbsDiff = ",
        max_basis_clover_reconstruction_abs_diff,
        ", maxBasisTotalReconstructionAbsDiff = ",
        max_basis_total_reconstruction_abs_diff,
        ", maxBasisRawWilsonAbsDiff = ",
        max_basis_raw_wilson_abs_diff,
        ", maxBasisProjectedWilsonAbsDiff = ",
        max_basis_projected_wilson_abs_diff,
        ", maxBasisRawCloverAbsDiff = ",
        max_basis_raw_clover_abs_diff,
        ", maxBasisProjectedCloverAbsDiff = ",
        max_basis_projected_clover_abs_diff,
        ", maxBasisRawTotalAbsDiff = ",
        max_basis_raw_total_abs_diff,
        ", maxBasisProjectedTotalAbsDiff = ",
        max_basis_projected_total_abs_diff,
        ", splitMatrixSumAbsDiff = ", split_matrix_sum_abs_diff,
        ", mixedFiniteDifference = ", mixed_finite_difference.derivative,
        ", mixedWilson = ", mixed_derivatives.wilson,
        ", mixedClover = ", mixed_derivatives.clover,
        ", mixedAnalytic = ", mixed_derivatives.total(),
        ", mixedReconstructedWilson = ", mixed_reconstructed_wilson,
        ", mixedReconstructedClover = ", mixed_reconstructed_clover,
        ", mixedReconstructedTotal = ", mixed_reconstructed_total,
        ", mixedRawWilson = ", mixed_raw_wilson,
        ", mixedProjectedRawWilson = ", mixed_projected_wilson,
        ", mixedRawClover = ", mixed_raw_clover,
        ", mixedProjectedRawClover = ", mixed_projected_clover,
        ", mixedRawTotal = ", mixed_raw_total,
        ", mixedProjectedRawTotal = ", mixed_projected_total,
        ", mixedFiniteDifferenceAbsDiff = ",
        mixed_finite_difference_abs_diff,
        ", mixedFiniteDifferenceRelDiff = ",
        mixed_finite_difference_rel_diff,
        ", mixedWilsonReconstructionAbsDiff = ",
        mixed_wilson_reconstruction_abs_diff,
        ", mixedCloverReconstructionAbsDiff = ",
        mixed_clover_reconstruction_abs_diff,
        ", mixedTotalReconstructionAbsDiff = ",
        mixed_total_reconstruction_abs_diff,
        ", mixedRawWilsonAbsDiff = ", mixed_raw_wilson_abs_diff,
        ", mixedProjectedRawWilsonAbsDiff = ",
        mixed_projected_wilson_abs_diff,
        ", mixedRawCloverAbsDiff = ", mixed_raw_clover_abs_diff,
        ", mixedProjectedRawCloverAbsDiff = ",
        mixed_projected_clover_abs_diff,
        ", mixedRawTotalAbsDiff = ", mixed_raw_total_abs_diff,
        ", mixedProjectedRawTotalAbsDiff = ",
        mixed_projected_total_abs_diff,
        ", mixedWilsonLinearityAbsDiff = ",
        mixed_wilson_linearity_abs_diff,
        ", mixedCloverLinearityAbsDiff = ",
        mixed_clover_linearity_abs_diff,
        ", mixedTotalLinearityAbsDiff = ",
        mixed_total_linearity_abs_diff,
        ", matrixNorms = [", wilson_matrix_norm, ", ",
        clover_matrix_norm, ", ", total_matrix_norm, "]",
        ", rawWilsonMatrixNorm = ", raw_wilson_matrix_norm,
        ", rawWilsonInvisibleComponentNorm = ",
        raw_wilson_invisible_component_norm,
        ", projectedRawWilsonMatrixNorm = ",
        projected_raw_wilson_matrix_norm,
        ", projectedRawWilsonMatrixAbsDiff = ",
        projected_raw_wilson_matrix_abs_diff,
        ", rawCloverMatrixNorm = ", raw_clover_matrix_norm,
        ", rawCloverInvisibleComponentNorm = ",
        raw_clover_invisible_component_norm,
        ", projectedRawCloverMatrixNorm = ",
        projected_raw_clover_matrix_norm,
        ", projectedRawCloverMatrixAbsDiff = ",
        projected_raw_clover_matrix_abs_diff,
        ", rawTotalMatrixNorm = ", raw_total_matrix_norm,
        ", rawTotalInvisibleComponentNorm = ",
        raw_total_invisible_component_norm,
        ", projectedRawTotalMatrixNorm = ",
        projected_raw_total_matrix_norm,
        ", projectedRawTotalMatrixAbsDiff = ",
        projected_raw_total_matrix_abs_diff,
        ", projectionLinearityAbsDiff = ",
        projection_linearity_abs_diff,
        ", projectedComponentSumAbsDiff = ",
        projected_component_sum_abs_diff,
        ", actionMaxResidual = ", max_action_residual,
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

        runMDWFCloverDirectionIndependentContractionNonzeroTest<8>(comm_base);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
