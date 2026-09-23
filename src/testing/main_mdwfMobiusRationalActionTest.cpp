/*
 * MDWF general-Mobius (RBC/UKQCD convention) rational action test.
 *
 * Evaluates S = phi^\dagger R(M^\dagger M) phi through the existing,
 * operator-generic computeMDWFRationalAction / MDWFRationalOperator /
 * MDWFCoupledMultiShiftCG path, with M / M^\dagger supplied by the
 * cluster-validated MDWFMobiusCloverLinearOperator /
 * MDWFMobiusCloverAdjointLinearOperator pair. No new solver, operator, or
 * action code is introduced.
 *
 * Checks, on a fixed nontrivial (random) gauge field, all at c_sw = 0.5:
 *
 *   1. Rational action versus a reference built from repeated single-shift
 *      solves with MDWFShiftedNormalOperator, for a well-conditioned control
 *      (M5 = -2, i.e. positive Wilson kernel mass; not a domain-wall choice)
 *      and physical-like parameters (M5 = 1.8, mf = 0.05), both at b5 = 1.5.
 *      This is the Mobius counterpart of mdwfRationalActionComparisonTest.
 *   2. b5 = 1 action regression: the Mobius rational action equals the
 *      existing Shamir clover rational action (MDWFLinearOperator /
 *      MDWFAdjointLinearOperator) at the control M5.
 *   3. Two-flavor heatbath identity: with phi = M^\dagger chi and
 *      R(x) = 1/x, S = phi^\dagger (M^\dagger M)^{-1} phi = chi^\dagger chi.
 *      This holds only if the supplied M^\dagger is the true adjoint of the
 *      inverted M, so it is independent of any other code path. Checked for
 *      the control and physical-like parameters at b5 = 1.5.
 *
 * This does not generate random pseudofermion noise, define determinant
 * powers or production rational approximations, touch RHMC/HMC or force
 * code, or validate gauge-dependent domain-wall physics.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFCoupledCG.h"
#include "../experimental/mdwf/MDWFMobiusMapping.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"
#include "../experimental/mdwf/MDWFPseudofermionAction.h"
#include "../experimental/mdwf/MDWFRationalCoefficientAdapter.h"
#include "../experimental/mdwf/MDWFShiftedNormalOperator.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFMobiusRationalActionField {
    floatT offset;

    explicit FillMDWFMobiusRationalActionField(floatT offset_in) : offset(offset_in) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t component = 0; component < 12; component++) {
            out.data[component] = COMPLEX(floatT)(
                offset
                + static_cast<floatT>(site.stack + 1)
                + static_cast<floatT>(0.001) * static_cast<floatT>(site.isite + component + 1),
                static_cast<floatT>(0.015) * static_cast<floatT>(component + 1)
                - static_cast<floatT>(0.0025) * static_cast<floatT>(site.stack));
        }
        return out;
    }
};

struct MDWFMobiusActionComparisonDiagnostics {
    double actionReal;
    double actionImagRel;
    bool multiShiftConverged;
    int maxMultiShiftIterations;
    double maxMultiShiftResidue;
    bool singleShiftConverged;
    double maxSingleShiftResidue;
    double maxSingleShiftRelativeResidual;
    double relativeOutputDifference;
    double actionRealRelativeDifference;
    double actionImagRelativeDifference;
};

struct MDWFMobiusHeatbathIdentityDiagnostics {
    double chiNorm2;
    double actionReal;
    double actionImagRel;
    double relativeDifference;
    bool converged;
    int iterations;
    double residue;
};

template<size_t HaloDepth, size_t Ls, class ForwardOperator, class AdjointOperator>
MDWFMobiusActionComparisonDiagnostics compareMDWFMobiusRationalAction(
    CommunicationBase &commBase,
    ForwardOperator &forward,
    AdjointOperator &adjoint,
    const MDWFRationalCoefficients<double> &coefficients,
    int maxIter,
    const std::string &name) {
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using ShiftedOperator = MDWFShiftedNormalOperator<NormalOperator, double>;
    using NormalAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, NormalOperator>;
    using ShiftedAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, ShiftedOperator>;
    using Spinor = typename NormalOperator::Spinor;

    Spinor field(commBase, name + "_field");
    Spinor actionWorkspace(commBase, name + "_workspace");
    Spinor reference(commBase, name + "_reference");
    Spinor difference(commBase, name + "_difference");
    Spinor shiftedSolution(commBase, name + "_shifted_solution");
    Spinor shiftedApplied(commBase, name + "_shifted_applied");
    Spinor residual(commBase, name + "_residual");

    field.template iterateOverBulk<>(FillMDWFMobiusRationalActionField<double, All, HaloDepth, Ls>(0.5));
    field.updateAll();

    NormalOperator normal(commBase, forward, adjoint, name + "_normal");
    NormalAdapter normalAdapter(normal);

    MDWFMobiusActionComparisonDiagnostics diag{};

    MDWFRationalActionResult<double> actionResult = computeMDWFRationalAction<double, NormalAdapter>(
        normalAdapter, actionWorkspace, field, coefficients, maxIter, 1e-8, name + "_action");

    diag.actionReal = actionResult.action_real;
    diag.actionImagRel = std::abs(actionResult.action_imag) / std::max(std::abs(actionResult.action_real), 1.0);
    diag.multiShiftConverged = actionResult.rational_result.converged();
    for (const auto &shift : actionResult.rational_result.shifts) {
        diag.maxMultiShiftIterations = std::max(diag.maxMultiShiftIterations, shift.iterations);
        diag.maxMultiShiftResidue = std::max(diag.maxMultiShiftResidue, shift.residue);
    }

    reference = coefficients.constant * field;
    const double fieldNorm2 = normalAdapter.norm2(field);
    diag.singleShiftConverged = true;

    for (size_t term = 0; term < coefficients.shift.size(); term++) {
        ShiftedOperator shifted(normal, coefficients.shift[term], name + "_shifted_" + std::to_string(term));
        ShiftedAdapter shiftedAdapter(shifted);
        MDWFCoupledCG<double, ShiftedAdapter> singleShiftCg;
        MDWFCoupledCGResult<double> singleShiftResult
            = singleShiftCg.invert(shiftedAdapter, shiftedSolution, field, maxIter, 1e-8, true);

        shifted.apply(shiftedApplied, shiftedSolution, true);
        residual = field;
        residual -= shiftedApplied;
        const double relativeResidual = std::sqrt(normalAdapter.norm2(residual) / std::max(fieldNorm2, 1.0));

        reference.template axpyThisB<64>(coefficients.numerator[term], shiftedSolution);

        diag.singleShiftConverged = diag.singleShiftConverged && singleShiftResult.converged;
        diag.maxSingleShiftResidue = std::max(diag.maxSingleShiftResidue, singleShiftResult.residue);
        diag.maxSingleShiftRelativeResidual = std::max(diag.maxSingleShiftRelativeResidual, relativeResidual);
    }
    reference.updateAll();

    difference = actionWorkspace;
    difference -= reference;

    const COMPLEX(double) referenceAction = normalAdapter.dotProduct5D(field, reference);
    const double referenceActionReal = real<double>(referenceAction);
    const double referenceActionImag = imag<double>(referenceAction);
    const double referenceNorm2 = normalAdapter.norm2(reference);
    const double differenceNorm2 = normalAdapter.norm2(difference);
    const double actionScale = std::max(1.0, std::abs(referenceActionReal));

    diag.relativeOutputDifference = std::sqrt(differenceNorm2 / std::max(referenceNorm2, 1.0));
    diag.actionRealRelativeDifference = std::abs(actionResult.action_real - referenceActionReal) / actionScale;
    diag.actionImagRelativeDifference = std::abs(actionResult.action_imag - referenceActionImag) / actionScale;

    return diag;
}

bool mdwfMobiusActionComparisonPassed(const MDWFMobiusActionComparisonDiagnostics &diag) {
    return std::isfinite(diag.actionReal)
           && diag.actionReal > 0.0
           && diag.actionImagRel <= 1e-10
           && diag.multiShiftConverged
           && diag.maxMultiShiftResidue <= 1e-8
           && diag.singleShiftConverged
           && diag.maxSingleShiftResidue <= 1e-8
           && diag.maxSingleShiftRelativeResidual <= 1e-7
           && diag.relativeOutputDifference <= 1e-10
           && diag.actionRealRelativeDifference <= 1e-12
           && diag.actionImagRelativeDifference <= 1e-12;
}

template<size_t HaloDepth, size_t Ls, class ForwardOperator, class AdjointOperator>
double computeMDWFMobiusRationalActionOnly(CommunicationBase &commBase,
                                           ForwardOperator &forward,
                                           AdjointOperator &adjoint,
                                           const MDWFRationalCoefficients<double> &coefficients,
                                           int maxIter,
                                           bool &converged,
                                           const std::string &name) {
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using NormalAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, NormalOperator>;
    using Spinor = typename NormalOperator::Spinor;

    Spinor field(commBase, name + "_field");
    Spinor actionWorkspace(commBase, name + "_workspace");
    field.template iterateOverBulk<>(FillMDWFMobiusRationalActionField<double, All, HaloDepth, Ls>(0.5));
    field.updateAll();

    NormalOperator normal(commBase, forward, adjoint, name + "_normal");
    NormalAdapter normalAdapter(normal);

    MDWFRationalActionResult<double> actionResult = computeMDWFRationalAction<double, NormalAdapter>(
        normalAdapter, actionWorkspace, field, coefficients, maxIter, 1e-8, name + "_action");
    converged = actionResult.rational_result.converged();
    return actionResult.action_real;
}

template<size_t HaloDepth, size_t Ls, class ForwardOperator, class AdjointOperator>
MDWFMobiusHeatbathIdentityDiagnostics checkMDWFMobiusHeatbathIdentity(
    CommunicationBase &commBase,
    ForwardOperator &forward,
    AdjointOperator &adjoint,
    int maxIter,
    double precision,
    const std::string &name) {
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using NormalAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, NormalOperator>;
    using Spinor = typename NormalOperator::Spinor;

    Spinor chi(commBase, name + "_chi");
    Spinor phi(commBase, name + "_phi");
    Spinor actionWorkspace(commBase, name + "_workspace");
    chi.template iterateOverBulk<>(FillMDWFMobiusRationalActionField<double, All, HaloDepth, Ls>(1.75));
    chi.updateAll();

    // phi = M^\dagger chi, the standard two-flavor pseudofermion heatbath.
    adjoint.apply(phi, chi, true);

    NormalOperator normal(commBase, forward, adjoint, name + "_normal");
    NormalAdapter normalAdapter(normal);

    MDWFExplicitRationalInput<double> inverseInput{
        name + "_inverse_coefficients",
        MDWFRationalCoefficientRole::Action,
        0.0,
        {1.0},
        {0.0}
    };
    const MDWFRationalCoefficients<double> inverse = makeMDWFRationalCoefficients(inverseInput);

    MDWFRationalActionResult<double> actionResult = computeMDWFRationalAction<double, NormalAdapter>(
        normalAdapter, actionWorkspace, phi, inverse, maxIter, precision, name + "_action");

    MDWFMobiusHeatbathIdentityDiagnostics diag{};
    diag.chiNorm2 = normalAdapter.norm2(chi);
    diag.actionReal = actionResult.action_real;
    diag.actionImagRel = std::abs(actionResult.action_imag) / std::max(std::abs(actionResult.action_real), 1.0);
    diag.relativeDifference = std::abs(actionResult.action_real - diag.chiNorm2) / std::max(diag.chiNorm2, 1.0);
    diag.converged = actionResult.rational_result.converged();
    diag.iterations = actionResult.rational_result.shifts.empty() ? 0 : actionResult.rational_result.shifts[0].iterations;
    diag.residue = actionResult.rational_result.shifts.empty() ? 0.0 : actionResult.rational_result.shifts[0].residue;
    return diag;
}

bool mdwfMobiusHeatbathIdentityPassed(const MDWFMobiusHeatbathIdentityDiagnostics &diag,
                                      double tolerance) {
    return std::isfinite(diag.actionReal)
           && diag.converged
           && diag.actionImagRel <= 1e-10
           && diag.relativeDifference <= tolerance;
}

void logMDWFMobiusActionComparison(const std::string &label, const MDWFMobiusActionComparisonDiagnostics &diag) {
    rootLogger.info("MDWF Mobius rational action comparison ", label,
                    ": actionReal = ", diag.actionReal,
                    ", actionImagRel = ", diag.actionImagRel,
                    ", multiShiftConverged = ", diag.multiShiftConverged,
                    ", maxMultiShiftIterations = ", diag.maxMultiShiftIterations,
                    ", maxMultiShiftResidue = ", diag.maxMultiShiftResidue,
                    ", maxSingleShiftResidue = ", diag.maxSingleShiftResidue,
                    ", maxSingleShiftRelativeResidual = ", diag.maxSingleShiftRelativeResidual,
                    ", relativeOutputDifference = ", diag.relativeOutputDifference,
                    ", actionRealRelativeDifference = ", diag.actionRealRelativeDifference,
                    ", actionImagRelativeDifference = ", diag.actionImagRelativeDifference);
}

void logMDWFMobiusHeatbathIdentity(const std::string &label, const MDWFMobiusHeatbathIdentityDiagnostics &diag) {
    rootLogger.info("MDWF Mobius two-flavor heatbath identity ", label,
                    ": chiNorm2 = ", diag.chiNorm2,
                    ", actionReal = ", diag.actionReal,
                    ", relativeDifference = ", diag.relativeDifference,
                    ", actionImagRel = ", diag.actionImagRel,
                    ", converged = ", diag.converged,
                    ", iterations = ", diag.iterations,
                    ", residue = ", diag.residue);
}

template<size_t Ls>
void runMDWFMobiusRationalActionTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using MobiusForward = MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using MobiusAdjoint = MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using ShamirForward = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using ShamirAdjoint = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;

    const double csw = 0.5;
    const double mf = 0.05;
    const double genericB5 = 1.5;
    const double controlM5 = -2.0;
    const double physicalM5 = 1.8;
    const int controlMaxIter = 2000;
    const int physicalMaxIter = 20000;
    const double identityPrecision = 1e-10;
    const double controlIdentityTolerance = 1e-8;
    const double physicalIdentityTolerance = 1e-6;

    MDWFExplicitRationalInput<double> actionInput{
        "mobius_action_comparison_coefficients",
        MDWFRationalCoefficientRole::Action,
        0.125,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    const MDWFRationalCoefficients<double> coefficients = makeMDWFRationalCoefficients(actionInput);

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_mobius_rational_action_gauge");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260923);
    d_rand = h_rand;
    gauge.random(d_rand.state);

    // --- Part 1: rational action vs repeated single-shift solves. ---
    MobiusForward controlForward(gauge, controlM5, mf, genericB5, csw, "MDWF_mobius_rational_action_control_forward");
    MobiusAdjoint controlAdjoint(gauge, controlM5, mf, genericB5, csw, "MDWF_mobius_rational_action_control_adjoint");
    const MDWFMobiusActionComparisonDiagnostics controlComparison = compareMDWFMobiusRationalAction<HaloDepth, Ls>(
        commBase, controlForward, controlAdjoint, coefficients, controlMaxIter,
        "MDWF_mobius_rational_action_control");
    logMDWFMobiusActionComparison("control (M5 = -2, b5 = 1.5)", controlComparison);

    MobiusForward physicalForward(gauge, physicalM5, mf, genericB5, csw,
                                  "MDWF_mobius_rational_action_physical_forward");
    MobiusAdjoint physicalAdjoint(gauge, physicalM5, mf, genericB5, csw,
                                  "MDWF_mobius_rational_action_physical_adjoint");
    const MDWFMobiusActionComparisonDiagnostics physicalComparison = compareMDWFMobiusRationalAction<HaloDepth, Ls>(
        commBase, physicalForward, physicalAdjoint, coefficients, physicalMaxIter,
        "MDWF_mobius_rational_action_physical");
    logMDWFMobiusActionComparison("physical-like (M5 = 1.8, mf = 0.05, b5 = 1.5)", physicalComparison);

    // --- Part 2: b5 = 1 Mobius vs Shamir clover rational action. ---
    MobiusForward b5OneForward(gauge, controlM5, mf, 1.0, csw, "MDWF_mobius_rational_action_b5_one_forward");
    MobiusAdjoint b5OneAdjoint(gauge, controlM5, mf, 1.0, csw, "MDWF_mobius_rational_action_b5_one_adjoint");
    bool b5OneConverged = false;
    const double b5OneAction = computeMDWFMobiusRationalActionOnly<HaloDepth, Ls>(
        commBase, b5OneForward, b5OneAdjoint, coefficients, controlMaxIter, b5OneConverged,
        "MDWF_mobius_rational_action_b5_one");

    const MDWFFifthDimCoefficients<double> shamirCoeff = mdwfShamirFifthDimCoefficients(mf);
    const double controlMass = mdwfShamirKernelMass(controlM5);
    ShamirForward shamirForward(gauge, shamirCoeff, controlMass, csw, "MDWF_mobius_rational_action_shamir_forward");
    ShamirAdjoint shamirAdjoint(gauge, shamirCoeff, controlMass, csw, "MDWF_mobius_rational_action_shamir_adjoint");
    bool shamirConverged = false;
    const double shamirAction = computeMDWFMobiusRationalActionOnly<HaloDepth, Ls>(
        commBase, shamirForward, shamirAdjoint, coefficients, controlMaxIter, shamirConverged,
        "MDWF_mobius_rational_action_shamir");

    const double b5OneActionRelDiff = std::abs(b5OneAction - shamirAction) / std::max(std::abs(shamirAction), 1.0);
    rootLogger.info("MDWF Mobius rational action b5 = 1 regression (M5 = -2): mobiusAction = ", b5OneAction,
                    ", shamirAction = ", shamirAction, ", relativeDifference = ", b5OneActionRelDiff,
                    ", mobiusConverged = ", b5OneConverged, ", shamirConverged = ", shamirConverged);

    // --- Part 3: two-flavor heatbath identity phi = M^\dagger chi => S = |chi|^2. ---
    const MDWFMobiusHeatbathIdentityDiagnostics controlIdentity = checkMDWFMobiusHeatbathIdentity<HaloDepth, Ls>(
        commBase, controlForward, controlAdjoint, controlMaxIter, identityPrecision,
        "MDWF_mobius_rational_action_control_identity");
    logMDWFMobiusHeatbathIdentity("control (M5 = -2, b5 = 1.5)", controlIdentity);

    const MDWFMobiusHeatbathIdentityDiagnostics physicalIdentity = checkMDWFMobiusHeatbathIdentity<HaloDepth, Ls>(
        commBase, physicalForward, physicalAdjoint, physicalMaxIter, identityPrecision,
        "MDWF_mobius_rational_action_physical_identity");
    logMDWFMobiusHeatbathIdentity("physical-like (M5 = 1.8, mf = 0.05, b5 = 1.5)", physicalIdentity);

    const bool controlComparisonPassed = mdwfMobiusActionComparisonPassed(controlComparison);
    const bool physicalComparisonPassed = mdwfMobiusActionComparisonPassed(physicalComparison);
    const bool b5OneRegressionPassed = b5OneConverged && shamirConverged && b5OneActionRelDiff <= 1e-12;
    const bool controlIdentityPassed = mdwfMobiusHeatbathIdentityPassed(controlIdentity, controlIdentityTolerance);
    const bool physicalIdentityPassed = mdwfMobiusHeatbathIdentityPassed(physicalIdentity, physicalIdentityTolerance);

    if (!controlComparisonPassed || !physicalComparisonPassed || !b5OneRegressionPassed
        || !controlIdentityPassed || !physicalIdentityPassed) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Mobius rational action test failed: control comparison passed = ", controlComparisonPassed,
            ", physical comparison passed = ", physicalComparisonPassed,
            ", b5 = 1 regression passed = ", b5OneRegressionPassed,
            ", control identity passed = ", controlIdentityPassed,
            ", physical identity passed = ", physicalIdentityPassed,
            " (see diagnostics above)"));
    }

    rootLogger.info("MDWF Mobius rational action test passed with Ls = ", Ls, ", c_sw = ", csw,
                    ", terms = ", coefficients.shift.size());
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

        runMDWFMobiusRationalActionTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
