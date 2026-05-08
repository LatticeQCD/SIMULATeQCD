/*
 * MDWF rational action comparison smoke test.
 *
 * This preserves the existing mdwfRationalNormalMdwfTest behavior by leaving
 * that target untouched.  Here we test only the action wrapper:
 *
 *     S = phi^\dagger R(M^\dagger M) phi
 *
 * against a reference built from repeated single-shift solves,
 *
 *     R(A) phi = c0 phi + sum_i numerator_i (A + shift_i)^(-1) phi.
 *
 * This is still pre-force and pre-RHMC: no pseudofermion RNG heatbath, no HMC
 * integration, no force terms, no HISQ, and no smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFCoupledCG.h"
#include "../experimental/mdwf/MDWFAdjointOperator.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"
#include "../experimental/mdwf/MDWFPseudofermionAction.h"
#include "../experimental/mdwf/MDWFRationalCoefficientAdapter.h"
#include "../experimental/mdwf/MDWFShiftedNormalOperator.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <stdexcept>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFRationalActionComparisonField {
    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t component = 0; component < 12; component++) {
            out.data[component] = COMPLEX(floatT)(
                static_cast<floatT>(0.5)
                + static_cast<floatT>(site.stack + 1)
                + static_cast<floatT>(0.001) * static_cast<floatT>(site.isite + component + 1),
                static_cast<floatT>(0.015) * static_cast<floatT>(component + 1)
                - static_cast<floatT>(0.0025) * static_cast<floatT>(site.stack));
        }
        return out;
    }
};

template<size_t Ls>
void runMDWFRationalActionComparisonTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    const double nonzeroCsw = 0.5;

    using ForwardOperator = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using AdjointOperator = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using ShiftedOperator = MDWFShiftedNormalOperator<NormalOperator, double>;
    using NormalAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, NormalOperator>;
    using ShiftedAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, ShiftedOperator>;
    using Spinor = typename NormalOperator::Spinor;

    MDWFExplicitRationalInput<double> actionInput{
        "action_comparison_coefficients",
        MDWFRationalCoefficientRole::Action,
        0.125,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    MDWFRationalCoefficients<double> coefficients = makeMDWFRationalCoefficients(actionInput);

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_rational_action_comparison_gauge");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260512);
    d_rand = h_rand;
    gauge.random(d_rand.state);

    Spinor field(commBase, "MDWF_rational_action_comparison_field");
    Spinor actionWorkspace(commBase, "MDWF_rational_action_comparison_workspace");
    Spinor reference(commBase, "MDWF_rational_action_comparison_reference");
    Spinor difference(commBase, "MDWF_rational_action_comparison_difference");
    Spinor shiftedSolution(commBase, "MDWF_rational_action_comparison_shifted_solution");
    Spinor shiftedApplied(commBase, "MDWF_rational_action_comparison_shifted_applied");
    Spinor residual(commBase, "MDWF_rational_action_comparison_residual");

    field.template iterateOverBulk<>(FillMDWFRationalActionComparisonField<double, All, HaloDepth, Ls>());
    field.updateAll();

    MDWFFifthDimCoefficients<double> fifthCoeff(1.0, -0.05, -0.05, 0.0, 0.0);
    ForwardOperator forward(gauge, fifthCoeff, 4.0, nonzeroCsw, "MDWF_rational_action_comparison_forward");
    AdjointOperator adjoint(gauge, fifthCoeff, 4.0, nonzeroCsw, "MDWF_rational_action_comparison_adjoint");
    NormalOperator normal(commBase, forward, adjoint, "MDWF_rational_action_comparison_normal");
    NormalAdapter normalAdapter(normal);

    MDWFRationalActionResult<double> actionResult
        = computeMDWFRationalAction<double, NormalAdapter>(
            normalAdapter, actionWorkspace, field, coefficients, 512, 1e-8, "MDWF_rational_action_comparison");

    reference = coefficients.constant * field;
    const double fieldNorm2 = normalAdapter.norm2(field);
    double maxActionResidue = 0.0;
    double maxSingleShiftResidue = 0.0;
    double maxRelativeResidual = 0.0;

    for (size_t term = 0; term < actionResult.rational_result.shifts.size(); term++) {
        maxActionResidue = std::max(maxActionResidue, actionResult.rational_result.shifts[term].residue);
    }

    for (size_t term = 0; term < coefficients.shift.size(); term++) {
        ShiftedOperator shifted(normal, coefficients.shift[term],
                                "MDWF_rational_action_comparison_shifted_" + std::to_string(term));
        ShiftedAdapter shiftedAdapter(shifted);
        MDWFCoupledCG<double, ShiftedAdapter> singleShiftCg;
        MDWFCoupledCGResult<double> singleShiftResult
            = singleShiftCg.invert(shiftedAdapter, shiftedSolution, field, 512, 1e-8, true);

        shifted.apply(shiftedApplied, shiftedSolution, true);
        residual = field;
        residual -= shiftedApplied;
        const double residualNorm2 = normalAdapter.norm2(residual);
        const double relativeResidual = std::sqrt(residualNorm2 / std::max(fieldNorm2, 1.0));

        reference.template axpyThisB<64>(coefficients.numerator[term], shiftedSolution);

        maxSingleShiftResidue = std::max(maxSingleShiftResidue, singleShiftResult.residue);
        maxRelativeResidual = std::max(maxRelativeResidual, relativeResidual);

        if (!singleShiftResult.converged
            || singleShiftResult.residue > 1e-8
            || relativeResidual > 1e-7) {
            throw std::runtime_error(stdLogger.fatal(
                "MDWF rational action comparison reference solve failed for term ", term,
                ": shift = ", coefficients.shift[term],
                ", converged = ", singleShiftResult.converged,
                ", iterations = ", singleShiftResult.iterations,
                ", residue = ", singleShiftResult.residue,
                ", relativeResidual = ", relativeResidual));
        }
    }
    reference.updateAll();

    difference = actionWorkspace;
    difference -= reference;

    const COMPLEX(double) referenceAction = normalAdapter.dotProduct5D(field, reference);
    const double referenceActionReal = real<double>(referenceAction);
    const double referenceActionImag = imag<double>(referenceAction);
    const double referenceNorm2 = normalAdapter.norm2(reference);
    const double differenceNorm2 = normalAdapter.norm2(difference);
    const double relativeOutputDifference = std::sqrt(differenceNorm2 / std::max(referenceNorm2, 1.0));
    const double actionScale = std::max(1.0, std::abs(referenceActionReal));
    const double actionRealRelativeDifference
        = std::abs(actionResult.action_real - referenceActionReal) / actionScale;
    const double actionImagRelativeDifference
        = std::abs(actionResult.action_imag - referenceActionImag) / actionScale;

    if (mdwfRationalCoefficientRoleName(actionInput.role) != "action"
        || forward.csw() != nonzeroCsw
        || adjoint.csw() != nonzeroCsw
        || !actionResult.rational_result.converged()
        || maxActionResidue > 1e-8
        || maxSingleShiftResidue > 1e-8
        || maxRelativeResidual > 1e-7
        || referenceActionReal <= 0.0
        || relativeOutputDifference > 1e-10
        || actionRealRelativeDifference > 1e-12
        || actionImagRelativeDifference > 1e-12) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF rational action comparison failed: c_sw = ", nonzeroCsw,
            ", rationalConverged = ", actionResult.rational_result.converged(),
            ", maxActionResidue = ", maxActionResidue,
            ", maxSingleShiftResidue = ", maxSingleShiftResidue,
            ", maxRelativeResidual = ", maxRelativeResidual,
            ", relativeOutputDifference = ", relativeOutputDifference,
            ", actionRealRelativeDifference = ", actionRealRelativeDifference,
            ", actionImagRelativeDifference = ", actionImagRelativeDifference,
            ", referenceActionReal = ", referenceActionReal,
            ", referenceActionImag = ", referenceActionImag));
    }

    rootLogger.info("MDWF rational action comparison passed with Ls = ", Ls,
                    ", c_sw = ", nonzeroCsw,
                    ", terms = ", coefficients.shift.size(),
                    ", actionReal = ", actionResult.action_real,
                    ", maxActionResidue = ", maxActionResidue,
                    ", maxSingleShiftResidue = ", maxSingleShiftResidue,
                    ", maxRelativeResidual = ", maxRelativeResidual,
                    ", relativeOutputDifference = ", relativeOutputDifference,
                    ", actionRealRelativeDifference = ", actionRealRelativeDifference);
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);

        LatticeParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfFifthDimTest.param", argc, argv);
        commBase.init(param.nodeDim());

        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);

        runMDWFRationalActionComparisonTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
