/*
 * MDWF c_sw = 0 selected-link direction-independent Wilson contraction test.
 *
 * This reconstructs one test-local su(3) matrix from Wilson directional
 * derivatives measured along an explicit eight-element basis.  It solves the
 * measured basis Gram system and validates a held-out mixed direction against
 * both the direct analytic contraction and a centered action finite
 * difference.  It also derives one left-oriented raw Wilson bilinear, applies
 * SU3::TA() exactly once after all slice/rational accumulation, and compares
 * the result with the independent reconstruction.  It does not define a
 * production force field, ipdot convention, HMC sign, all-link accumulator,
 * or MPI force path.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFAdjointOperator.h"
#include "../experimental/mdwf/MDWFAllLinkWilsonContraction.h"
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
struct FillMDWFWilsonDirectionIndependentCsw0Source {
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
class MDWFWilsonDirectionIndependentCsw0ActionEvaluator {
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
    int _max_iter;
    double _precision;

public:
    MDWFWilsonDirectionIndependentCsw0ActionEvaluator(
        CommunicationBase &comm_base,
        Spinor &field,
        const MDWFRationalCoefficients<double> &coefficients,
        MDWFFifthDimCoefficients<double> fifth_coeff,
        double mass,
        int max_iter,
        double precision)
        : _comm_base(comm_base),
          _field(field),
          _coefficients(coefficients),
          _fifth_coeff(fifth_coeff),
          _mass(mass),
          _max_iter(max_iter),
          _precision(precision) {}

    MDWFFiniteDifferenceActionValue<double> operator()(Gauge &gauge) {
        const double csw = 0.0;

        ForwardOperator forward(
            gauge, _fifth_coeff, _mass, csw,
            "MDWF_wilson_direction_independent_csw0_action_forward");
        AdjointOperator adjoint(
            gauge, _fifth_coeff, _mass, csw,
            "MDWF_wilson_direction_independent_csw0_action_adjoint");
        NormalOperator normal(
            _comm_base, forward, adjoint,
            "MDWF_wilson_direction_independent_csw0_action_normal");
        Adapter adapter(normal);
        Spinor action_workspace(
            _comm_base,
            "MDWF_wilson_direction_independent_csw0_action_workspace");

        const MDWFRationalActionResult<double> action_result
            = computeMDWFRationalAction<double, Adapter>(
                adapter, action_workspace, _field, _coefficients,
                _max_iter, _precision,
                "MDWF_wilson_direction_independent_csw0_action");

        return makeMDWFFiniteDifferenceActionValue(action_result);
    }
};

template<size_t Ls>
void runMDWFWilsonDirectionIndependentContractionCsw0Test(
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
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using Workspace
        = MDWFFermionForceWorkspace<
            double, HaloDepth, HaloDepth, Ls, NormalOperator, ForwardOperator>;
    using Spinor = typename NormalOperator::Spinor;
    using HostSpinor = MDWFSpinor<double, false, All, HaloDepth, Ls>;

    const LatticeData lat = GInd::getLatData();
    if (lat.vol4 != lat.globvol4) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF direction-independent Wilson contraction test is single-rank "
            "only: local volume = ", lat.vol4,
            ", global volume = ", lat.globvol4));
    }

    const MDWFExplicitRationalInput<double> action_input{
        "wilson_direction_independent_csw0_action_coefficients",
        MDWFRationalCoefficientRole::Action,
        0.125,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    const MDWFExplicitRationalInput<double> force_input{
        "wilson_direction_independent_csw0_force_coefficients",
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
        comm_base, "MDWF_wilson_direction_independent_csw0_base_gauge");
    Gauge gauge_plus(
        comm_base, "MDWF_wilson_direction_independent_csw0_gauge_plus");
    Gauge gauge_minus(
        comm_base, "MDWF_wilson_direction_independent_csw0_gauge_minus");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260514);
    d_rand = h_rand;
    base_gauge.random(d_rand.state);
    base_gauge.updateAll();

    Spinor field(
        comm_base, "MDWF_wilson_direction_independent_csw0_field");
    field.template iterateOverBulk<>(
        FillMDWFWilsonDirectionIndependentCsw0Source<
            double, All, HaloDepth, Ls>());
    field.updateAll();
    Spinor force_field(
        comm_base, "MDWF_wilson_direction_independent_csw0_force_field");
    force_field = field;
    force_field.updateAll();

    const MDWFFifthDimCoefficients<double> fifth_coeff(
        1.0, -0.05, -0.05, 0.0, 0.0);
    const MDWFFiniteDifferenceProbe<double> probe{
        1, 2, 3, 0,
        1,
        0,
        1e-4,
        MDWFFiniteDifferenceMultiplicationSide::Left
    };

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
            "MDWF mixed direction has invalid raw norm ", mixed_raw_norm));
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

    MDWFWilsonDirectionIndependentCsw0ActionEvaluator<HaloDepth, Ls>
        action_evaluator(
            comm_base, field, action_coefficients, fifth_coeff,
            mass, 512, 1e-8);
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
        comm_base, "MDWF_wilson_direction_independent_csw0_gauge_host");
    HostGauge mixed_plus_host(
        comm_base, "MDWF_wilson_direction_independent_csw0_mixed_plus_host");
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
        "MDWF_wilson_direction_independent_csw0_forward");
    AdjointOperator adjoint(
        base_gauge, fifth_coeff, mass, csw,
        "MDWF_wilson_direction_independent_csw0_adjoint");
    NormalOperator normal(
        comm_base, forward, adjoint,
        "MDWF_wilson_direction_independent_csw0_normal");
    Workspace workspace;
    workspace.prepare(
        normal, forward, force_field, force_coefficients, 512, 1e-8,
        "MDWF_wilson_direction_independent_csw0_workspace");

    double max_force_workspace_residual = 0.0;
    for (const auto &info : workspace.shiftInfo()) {
        max_force_workspace_residual = std::max(
            max_force_workspace_residual, info.residue);
    }

    const gSite link_site
        = GInd::getSite(probe.x, probe.y, probe.z, probe.t);

    std::array<double, MDWFDirectionIndependentBasisSize>
        basis_analytic_derivatives{};
    double mixed_analytic_derivative = 0.0;
    SU3<double> raw_wilson_matrix = su3_zero<double>();

    for (size_t term = 0; term < workspace.size(); term++) {
        HostSpinor chi_host(
            comm_base,
            "MDWF_wilson_direction_independent_csw0_chi_"
                + std::to_string(term));
        HostSpinor eta_host(
            comm_base,
            "MDWF_wilson_direction_independent_csw0_eta_"
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
            basis_analytic_derivatives[generator]
                += rational_weight
                   * mdwfAllLinkWilsonContractionTerm<HaloDepth, Ls>(
                       chi_acc, eta_acc, gauge_acc, link_site,
                       probe.mu, basis[generator]);
        }
        mixed_analytic_derivative
            += rational_weight
               * mdwfAllLinkWilsonContractionTerm<HaloDepth, Ls>(
                   chi_acc, eta_acc, gauge_acc, link_site,
                   probe.mu, mixed_direction);
        raw_wilson_matrix
            += rational_weight
               * mdwfWilsonLeftRawContractionMatrixTerm<HaloDepth, Ls>(
                   chi_acc, eta_acc, gauge_acc, link_site, probe.mu);
    }

    const MDWFDirectionIndependentReconstruction<double> reconstruction
        = mdwfReconstructDirectionIndependentMatrix(
            basis, gram, basis_analytic_derivatives);
    SU3<double> projected_raw_wilson_matrix = raw_wilson_matrix;
    projected_raw_wilson_matrix.TA();

    double max_basis_reconstruction_abs_diff = 0.0;
    double max_basis_raw_matrix_abs_diff = 0.0;
    double max_basis_projected_matrix_abs_diff = 0.0;
    double max_basis_finite_difference_abs_diff = 0.0;
    double max_basis_finite_difference_rel_diff = 0.0;
    bool basis_finite_differences_passed = true;
    bool basis_derivatives_finite = true;

    for (size_t generator = 0;
         generator < MDWFDirectionIndependentBasisSize;
         generator++) {
        const double reconstructed_derivative = static_cast<double>(
            real(tr_c(basis[generator], reconstruction.matrix)));
        const double raw_matrix_derivative = static_cast<double>(
            real(tr_c(basis[generator], raw_wilson_matrix)));
        const double projected_matrix_derivative = static_cast<double>(
            real(tr_c(basis[generator], projected_raw_wilson_matrix)));
        const double reconstruction_abs_diff = std::abs(
            reconstructed_derivative
            - basis_analytic_derivatives[generator]);
        const double raw_matrix_abs_diff = std::abs(
            raw_matrix_derivative
            - basis_analytic_derivatives[generator]);
        const double projected_matrix_abs_diff = std::abs(
            projected_matrix_derivative
            - basis_analytic_derivatives[generator]);
        const double finite_difference_abs_diff = std::abs(
            basis_finite_differences[generator].derivative
            - basis_analytic_derivatives[generator]);
        const double finite_difference_scale = std::max(
            1.0,
            std::abs(basis_finite_differences[generator].derivative));
        const double finite_difference_rel_diff
            = finite_difference_abs_diff / finite_difference_scale;
        const double finite_difference_tolerance
            = 5e-3 + 5e-4 * finite_difference_scale;

        max_basis_reconstruction_abs_diff = std::max(
            max_basis_reconstruction_abs_diff, reconstruction_abs_diff);
        max_basis_raw_matrix_abs_diff = std::max(
            max_basis_raw_matrix_abs_diff, raw_matrix_abs_diff);
        max_basis_projected_matrix_abs_diff = std::max(
            max_basis_projected_matrix_abs_diff,
            projected_matrix_abs_diff);
        max_basis_finite_difference_abs_diff = std::max(
            max_basis_finite_difference_abs_diff,
            finite_difference_abs_diff);
        max_basis_finite_difference_rel_diff = std::max(
            max_basis_finite_difference_rel_diff,
            finite_difference_rel_diff);
        basis_finite_differences_passed
            = basis_finite_differences_passed
              && finite_difference_abs_diff <= finite_difference_tolerance;
        basis_derivatives_finite
            = basis_derivatives_finite
              && std::isfinite(
                  basis_finite_differences[generator].derivative)
              && std::isfinite(basis_analytic_derivatives[generator])
              && std::isfinite(reconstructed_derivative)
              && std::isfinite(raw_matrix_derivative)
              && std::isfinite(projected_matrix_derivative);

        rootLogger.info(
            "MDWF c_sw = 0 Wilson direction-independent basis probe with "
            "generator = ", generator,
            ", norm = ", -gram[generator][generator],
            ", finiteDifference = ",
            basis_finite_differences[generator].derivative,
            ", analytic = ", basis_analytic_derivatives[generator],
            ", reconstructed = ", reconstructed_derivative,
            ", rawMatrix = ", raw_matrix_derivative,
            ", projectedRawMatrix = ", projected_matrix_derivative,
            ", finiteDifferenceAbsDiff = ", finite_difference_abs_diff,
            ", finiteDifferenceRelDiff = ", finite_difference_rel_diff,
            ", reconstructionAbsDiff = ", reconstruction_abs_diff,
            ", rawMatrixAbsDiff = ", raw_matrix_abs_diff,
            ", projectedRawMatrixAbsDiff = ",
            projected_matrix_abs_diff);
    }

    const double mixed_reconstructed_derivative = static_cast<double>(
        real(tr_c(mixed_direction, reconstruction.matrix)));
    const double mixed_raw_matrix_derivative = static_cast<double>(
        real(tr_c(mixed_direction, raw_wilson_matrix)));
    const double mixed_projected_matrix_derivative = static_cast<double>(
        real(tr_c(mixed_direction, projected_raw_wilson_matrix)));
    double mixed_basis_linear_derivative = 0.0;
    for (size_t generator = 0;
         generator < MDWFDirectionIndependentBasisSize;
         generator++) {
        mixed_basis_linear_derivative
            += mixed_coefficients[generator]
               * basis_analytic_derivatives[generator];
    }

    const double mixed_finite_difference_abs_diff = std::abs(
        mixed_finite_difference.derivative - mixed_analytic_derivative);
    const double mixed_finite_difference_scale = std::max(
        1.0, std::abs(mixed_finite_difference.derivative));
    const double mixed_finite_difference_rel_diff
        = mixed_finite_difference_abs_diff / mixed_finite_difference_scale;
    const double mixed_finite_difference_tolerance
        = 5e-3 + 5e-4 * mixed_finite_difference_scale;
    const double mixed_reconstruction_abs_diff = std::abs(
        mixed_reconstructed_derivative - mixed_analytic_derivative);
    const double mixed_raw_matrix_abs_diff = std::abs(
        mixed_raw_matrix_derivative - mixed_analytic_derivative);
    const double mixed_projected_matrix_abs_diff = std::abs(
        mixed_projected_matrix_derivative - mixed_analytic_derivative);
    const double mixed_linearity_abs_diff = std::abs(
        mixed_basis_linear_derivative - mixed_analytic_derivative);
    const double algebra_tolerance
        = 1e-10 * std::max(1.0, std::abs(mixed_analytic_derivative));

    const double reconstructed_matrix_norm
        = static_cast<double>(infnorm(reconstruction.matrix));
    const double reconstructed_matrix_anti_hermitian_violation
        = static_cast<double>(
            infnorm(reconstruction.matrix + dagger(reconstruction.matrix)));
    const double reconstructed_matrix_trace_violation
        = static_cast<double>(abs(tr_c(reconstruction.matrix)));
    const double raw_matrix_norm
        = static_cast<double>(infnorm(raw_wilson_matrix));
    const double projected_raw_matrix_norm
        = static_cast<double>(infnorm(projected_raw_wilson_matrix));
    const double raw_invisible_component_norm = static_cast<double>(
        infnorm(raw_wilson_matrix - projected_raw_wilson_matrix));
    const double projected_raw_matrix_abs_diff = static_cast<double>(
        infnorm(projected_raw_wilson_matrix - reconstruction.matrix));
    const double projected_raw_matrix_anti_hermitian_violation
        = static_cast<double>(infnorm(
            projected_raw_wilson_matrix
            + dagger(projected_raw_wilson_matrix)));
    const double projected_raw_matrix_trace_violation
        = static_cast<double>(abs(tr_c(projected_raw_wilson_matrix)));

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
        || reconstruction.minimum_absolute_pivot <= 1e-14
        || !basis_derivatives_finite
        || !basis_finite_differences_passed
        || max_basis_reconstruction_abs_diff > algebra_tolerance
        || max_basis_raw_matrix_abs_diff > algebra_tolerance
        || max_basis_projected_matrix_abs_diff > algebra_tolerance
        || std::abs(mixed_direction_norm - 1.0) > 1e-12
        || mixed_anti_hermitian_violation > 1e-12
        || mixed_trace_violation > 1e-12
        || !std::isfinite(mixed_finite_difference.derivative)
        || !std::isfinite(mixed_analytic_derivative)
        || !std::isfinite(mixed_reconstructed_derivative)
        || !std::isfinite(mixed_raw_matrix_derivative)
        || !std::isfinite(mixed_projected_matrix_derivative)
        || !std::isfinite(mixed_basis_linear_derivative)
        || mixed_finite_difference_abs_diff
           > mixed_finite_difference_tolerance
        || mixed_reconstruction_abs_diff > algebra_tolerance
        || mixed_raw_matrix_abs_diff > algebra_tolerance
        || mixed_projected_matrix_abs_diff > algebra_tolerance
        || mixed_linearity_abs_diff > algebra_tolerance
        || !std::isfinite(reconstructed_matrix_norm)
        || reconstructed_matrix_norm <= 0.0
        || reconstructed_matrix_anti_hermitian_violation > 1e-12
        || reconstructed_matrix_trace_violation > 1e-12
        || !std::isfinite(raw_matrix_norm)
        || !std::isfinite(projected_raw_matrix_norm)
        || !std::isfinite(raw_invisible_component_norm)
        || !std::isfinite(projected_raw_matrix_abs_diff)
        || raw_matrix_norm <= 0.0
        || projected_raw_matrix_norm <= 0.0
        || raw_invisible_component_norm <= 1e-12
        || projected_raw_matrix_abs_diff > algebra_tolerance
        || projected_raw_matrix_anti_hermitian_violation > 1e-12
        || projected_raw_matrix_trace_violation > 1e-12) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF c_sw = 0 Wilson direction-independent contraction test "
            "failed: Ls = ", Ls,
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
            ", gramEntriesFinite = ", gram_entries_finite,
            ", gramMinAbsPivot = ",
            reconstruction.minimum_absolute_pivot,
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
            ", maxBasisReconstructionAbsDiff = ",
            max_basis_reconstruction_abs_diff,
            ", maxBasisRawMatrixAbsDiff = ",
            max_basis_raw_matrix_abs_diff,
            ", maxBasisProjectedMatrixAbsDiff = ",
            max_basis_projected_matrix_abs_diff,
            ", basisDerivativesFinite = ", basis_derivatives_finite,
            ", mixedDirectionNorm = ", mixed_direction_norm,
            ", mixedFiniteDifference = ",
            mixed_finite_difference.derivative,
            ", mixedAnalytic = ", mixed_analytic_derivative,
            ", mixedReconstructed = ", mixed_reconstructed_derivative,
            ", mixedRawMatrix = ", mixed_raw_matrix_derivative,
            ", mixedProjectedRawMatrix = ",
            mixed_projected_matrix_derivative,
            ", mixedBasisLinear = ", mixed_basis_linear_derivative,
            ", mixedFiniteDifferenceAbsDiff = ",
            mixed_finite_difference_abs_diff,
            ", mixedFiniteDifferenceRelDiff = ",
            mixed_finite_difference_rel_diff,
            ", mixedReconstructionAbsDiff = ",
            mixed_reconstruction_abs_diff,
            ", mixedRawMatrixAbsDiff = ", mixed_raw_matrix_abs_diff,
            ", mixedProjectedRawMatrixAbsDiff = ",
            mixed_projected_matrix_abs_diff,
            ", mixedLinearityAbsDiff = ", mixed_linearity_abs_diff,
            ", reconstructedMatrixNorm = ", reconstructed_matrix_norm,
            ", reconstructedMatrixAntiHermitianViolation = ",
            reconstructed_matrix_anti_hermitian_violation,
            ", reconstructedMatrixTraceViolation = ",
            reconstructed_matrix_trace_violation,
            ", rawMatrixNorm = ", raw_matrix_norm,
            ", rawInvisibleComponentNorm = ",
            raw_invisible_component_norm,
            ", projectedRawMatrixNorm = ", projected_raw_matrix_norm,
            ", projectedRawMatrixAbsDiff = ",
            projected_raw_matrix_abs_diff,
            ", projectedRawMatrixAntiHermitianViolation = ",
            projected_raw_matrix_anti_hermitian_violation,
            ", projectedRawMatrixTraceViolation = ",
            projected_raw_matrix_trace_violation,
            ", actionMaxResidual = ", max_action_residual,
            ", forceMaxResidual = ", max_force_workspace_residual,
            ", actionImagRel = ", max_action_imag_relative));
    }

    rootLogger.info(
        "MDWF c_sw = 0 Wilson direction-independent contraction test passed "
        "with Ls = ", Ls,
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
        ", gramMinAbsPivot = ", reconstruction.minimum_absolute_pivot,
        ", basisNormRange = [", basis_minimum_norm, ", ",
        basis_maximum_norm, "]",
        ", maxBasisFiniteDifferenceAbsDiff = ",
        max_basis_finite_difference_abs_diff,
        ", maxBasisFiniteDifferenceRelDiff = ",
        max_basis_finite_difference_rel_diff,
        ", maxBasisReconstructionAbsDiff = ",
        max_basis_reconstruction_abs_diff,
        ", maxBasisRawMatrixAbsDiff = ",
        max_basis_raw_matrix_abs_diff,
        ", maxBasisProjectedMatrixAbsDiff = ",
        max_basis_projected_matrix_abs_diff,
        ", mixedFiniteDifference = ", mixed_finite_difference.derivative,
        ", mixedAnalytic = ", mixed_analytic_derivative,
        ", mixedReconstructed = ", mixed_reconstructed_derivative,
        ", mixedRawMatrix = ", mixed_raw_matrix_derivative,
        ", mixedProjectedRawMatrix = ",
        mixed_projected_matrix_derivative,
        ", mixedFiniteDifferenceAbsDiff = ",
        mixed_finite_difference_abs_diff,
        ", mixedFiniteDifferenceRelDiff = ",
        mixed_finite_difference_rel_diff,
        ", mixedReconstructionAbsDiff = ",
        mixed_reconstruction_abs_diff,
        ", mixedRawMatrixAbsDiff = ", mixed_raw_matrix_abs_diff,
        ", mixedProjectedRawMatrixAbsDiff = ",
        mixed_projected_matrix_abs_diff,
        ", mixedLinearityAbsDiff = ", mixed_linearity_abs_diff,
        ", reconstructedMatrixNorm = ", reconstructed_matrix_norm,
        ", rawMatrixNorm = ", raw_matrix_norm,
        ", rawInvisibleComponentNorm = ", raw_invisible_component_norm,
        ", projectedRawMatrixNorm = ", projected_raw_matrix_norm,
        ", projectedRawMatrixAbsDiff = ",
        projected_raw_matrix_abs_diff,
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

        runMDWFWilsonDirectionIndependentContractionCsw0Test<8>(comm_base);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
