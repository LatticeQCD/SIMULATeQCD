/*
 * MDWF general-Mobius (RBC/UKQCD convention) finite-difference action test.
 *
 * Uses the existing, action-evaluator-generic finite-difference harness to
 * perturb one gauge link, U -> exp(+-epsilon T) U, and evaluate
 *
 *     [S(U_+) - S(U_-)] / (2 epsilon),   S = phi^\dagger R(M^\dagger M) phi,
 *
 * with M / M^\dagger the cluster-validated MDWFMobiusCloverLinearOperator /
 * MDWFMobiusCloverAdjointLinearOperator pair. This is the Mobius counterpart
 * of mdwfFiniteDifferenceMdwfNonzeroCswTest and the target that a future
 * analytic Mobius force contraction will be compared against. No new
 * solver, operator, action, or harness code is introduced.
 *
 * Checks, on a fixed nontrivial (random) gauge field, all with the same probe
 * link and 3-term action coefficients as the Shamir test:
 *
 *   1. Control (M5 = -2, i.e. positive Wilson kernel mass; not a domain-wall
 *      choice), b5 = 1.5, c_sw = 0.5: finite, nonzero derivative, stable
 *      across epsilon = {1e-3, 3e-4, 1e-4} to relative 1e-3.
 *   2. b5 = 1 regression at the finest epsilon: the Mobius finite difference
 *      reproduces the Shamir clover finite difference exactly.
 *   3. Response checks at the finest epsilon: the derivative changes
 *      detectably between b5 = 1.5 and b5 = 1, and between c_sw = 0.5 and
 *      c_sw = 0.
 *   4. Physical-like (M5 = 1.8, mf = 0.05, b5 = 1.5, c_sw = 0.5): finite,
 *      nonzero derivative, stable across epsilon = {3e-4, 1e-4} to relative
 *      1e-3.
 *
 * This does not allocate or accumulate gauge force, update momenta, call
 * RHMC/HMC, touch HISQ, or use smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFFiniteDifferenceHarness.h"
#include "../experimental/mdwf/MDWFMobiusMapping.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"
#include "../experimental/mdwf/MDWFRationalCoefficientAdapter.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFMobiusFiniteDifferenceSource {
    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t component = 0; component < 12; component++) {
            out.data[component] = COMPLEX(floatT)(
                static_cast<floatT>(0.5)
                + static_cast<floatT>(site.stack + 1)
                + static_cast<floatT>(0.001) * static_cast<floatT>(site.isite + component + 1),
                static_cast<floatT>(0.017) * static_cast<floatT>(component + 1)
                - static_cast<floatT>(0.0025) * static_cast<floatT>(site.stack));
        }
        return out;
    }
};

template<size_t HaloDepth, size_t Ls>
class MDWFMobiusFiniteDifferenceActionEvaluator {
public:
    using ForwardOperator = MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using AdjointOperator = MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, NormalOperator>;
    using Spinor = typename NormalOperator::Spinor;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;

private:
    CommunicationBase &_commBase;
    Spinor &_field;
    MDWFRationalCoefficients<double> _coefficients;
    double _M5;
    double _mf;
    double _b5;
    double _csw;
    int _max_iter;
    double _precision;

public:
    MDWFMobiusFiniteDifferenceActionEvaluator(CommunicationBase &commBase,
                                              Spinor &field,
                                              const MDWFRationalCoefficients<double> &coefficients,
                                              double M5, double mf, double b5, double csw,
                                              int max_iter, double precision)
        : _commBase(commBase),
          _field(field),
          _coefficients(coefficients),
          _M5(M5),
          _mf(mf),
          _b5(b5),
          _csw(csw),
          _max_iter(max_iter),
          _precision(precision) {}

    MDWFFiniteDifferenceActionValue<double> operator()(Gauge &gauge) {
        ForwardOperator forward(gauge, _M5, _mf, _b5, _csw, "MDWF_mobius_fd_forward");
        AdjointOperator adjoint(gauge, _M5, _mf, _b5, _csw, "MDWF_mobius_fd_adjoint");
        NormalOperator normal(_commBase, forward, adjoint, "MDWF_mobius_fd_normal");
        Adapter adapter(normal);
        Spinor actionWorkspace(_commBase, "MDWF_mobius_fd_action_workspace");

        MDWFRationalActionResult<double> actionResult = computeMDWFRationalAction<double, Adapter>(
            adapter, actionWorkspace, _field, _coefficients, _max_iter, _precision, "MDWF_mobius_fd_action");

        return makeMDWFFiniteDifferenceActionValue(actionResult);
    }
};

template<size_t HaloDepth, size_t Ls>
class MDWFMobiusFiniteDifferenceShamirActionEvaluator {
public:
    using ForwardOperator = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using AdjointOperator = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, NormalOperator>;
    using Spinor = typename NormalOperator::Spinor;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;

private:
    CommunicationBase &_commBase;
    Spinor &_field;
    MDWFRationalCoefficients<double> _coefficients;
    MDWFFifthDimCoefficients<double> _fifth_coeff;
    double _mass;
    double _csw;
    int _max_iter;
    double _precision;

public:
    MDWFMobiusFiniteDifferenceShamirActionEvaluator(CommunicationBase &commBase,
                                                    Spinor &field,
                                                    const MDWFRationalCoefficients<double> &coefficients,
                                                    double M5, double mf, double csw,
                                                    int max_iter, double precision)
        : _commBase(commBase),
          _field(field),
          _coefficients(coefficients),
          _fifth_coeff(mdwfShamirFifthDimCoefficients(mf)),
          _mass(mdwfShamirKernelMass(M5)),
          _csw(csw),
          _max_iter(max_iter),
          _precision(precision) {}

    MDWFFiniteDifferenceActionValue<double> operator()(Gauge &gauge) {
        ForwardOperator forward(gauge, _fifth_coeff, _mass, _csw, "MDWF_mobius_fd_shamir_forward");
        AdjointOperator adjoint(gauge, _fifth_coeff, _mass, _csw, "MDWF_mobius_fd_shamir_adjoint");
        NormalOperator normal(_commBase, forward, adjoint, "MDWF_mobius_fd_shamir_normal");
        Adapter adapter(normal);
        Spinor actionWorkspace(_commBase, "MDWF_mobius_fd_shamir_action_workspace");

        MDWFRationalActionResult<double> actionResult = computeMDWFRationalAction<double, Adapter>(
            adapter, actionWorkspace, _field, _coefficients, _max_iter, _precision,
            "MDWF_mobius_fd_shamir_action");

        return makeMDWFFiniteDifferenceActionValue(actionResult);
    }
};

bool mdwfMobiusFiniteDifferenceResultValid(const MDWFFiniteDifferenceResult<double> &result) {
    return result.converged
           && std::isfinite(result.derivative)
           && result.max_shifted_residual <= 1e-8
           && result.action_imag_relative <= 1e-8
           && result.plus.action_real > 0.0
           && result.minus.action_real > 0.0;
}

void logMDWFMobiusFiniteDifference(const std::string &label, double epsilon,
                                   const MDWFFiniteDifferenceResult<double> &result) {
    rootLogger.info("MDWF Mobius finite difference ", label, " epsilon = ", epsilon,
                    ": derivative = ", result.derivative,
                    ", actionPlus = ", result.plus.action_real,
                    ", actionMinus = ", result.minus.action_real,
                    ", actionImagRelative = ", result.action_imag_relative,
                    ", maxShiftedResidual = ", result.max_shifted_residual,
                    ", converged = ", result.converged);
}

template<size_t Ls>
void runMDWFMobiusFiniteDifferenceTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;
    using MobiusEvaluator = MDWFMobiusFiniteDifferenceActionEvaluator<HaloDepth, Ls>;
    using ShamirEvaluator = MDWFMobiusFiniteDifferenceShamirActionEvaluator<HaloDepth, Ls>;

    const double csw = 0.5;
    const double mf = 0.05;
    const double genericB5 = 1.5;
    const double controlM5 = -2.0;
    const double physicalM5 = 1.8;
    const int controlMaxIter = 2000;
    const int physicalMaxIter = 20000;
    const double precision = 1e-8;
    const double stabilityTolerance = 1e-3;

    MDWFExplicitRationalInput<double> actionInput{
        "mobius_finite_difference_action_coefficients",
        MDWFRationalCoefficientRole::Action,
        0.125,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    const MDWFRationalCoefficients<double> coefficients = makeMDWFRationalCoefficients(actionInput);

    Gauge baseGauge(commBase, "MDWF_mobius_fd_base_gauge");
    Gauge gaugePlus(commBase, "MDWF_mobius_fd_gauge_plus");
    Gauge gaugeMinus(commBase, "MDWF_mobius_fd_gauge_minus");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260514);
    d_rand = h_rand;
    baseGauge.random(d_rand.state);
    baseGauge.updateAll();

    Spinor field(commBase, "MDWF_mobius_fd_field");
    field.template iterateOverBulk<>(FillMDWFMobiusFiniteDifferenceSource<double, All, HaloDepth, Ls>());
    field.updateAll();

    auto makeProbe = [](double epsilon) {
        return MDWFFiniteDifferenceProbe<double>{
            1, 2, 3, 0,
            1,
            0,
            epsilon,
            MDWFFiniteDifferenceMultiplicationSide::Left
        };
    };

    // --- Part 1: control epsilon sweep, b5 = 1.5, c_sw = 0.5. ---
    MobiusEvaluator controlEvaluator(commBase, field, coefficients, controlM5, mf, genericB5, csw,
                                     controlMaxIter, precision);
    const double controlEpsilons[3] = {1e-3, 3e-4, 1e-4};
    double controlDerivative[3] = {0.0, 0.0, 0.0};
    bool controlValid = true;
    for (size_t idx = 0; idx < 3; idx++) {
        const MDWFFiniteDifferenceResult<double> result = evaluateMDWFFiniteDifferenceAction(
            gaugePlus, gaugeMinus, baseGauge, makeProbe(controlEpsilons[idx]), controlEvaluator);
        logMDWFMobiusFiniteDifference("control (M5 = -2, b5 = 1.5)", controlEpsilons[idx], result);
        controlDerivative[idx] = result.derivative;
        controlValid = controlValid && mdwfMobiusFiniteDifferenceResultValid(result);
    }
    const double controlScale = std::max(1.0, std::abs(controlDerivative[2]));
    const double controlCoarseFineRelDiff = std::abs(controlDerivative[0] - controlDerivative[2]) / controlScale;
    const double controlMediumFineRelDiff = std::abs(controlDerivative[1] - controlDerivative[2]) / controlScale;

    // --- Part 2/3: finest-epsilon b5 = 1 regression and b5 / c_sw responses. ---
    const double finestEpsilon = controlEpsilons[2];

    MobiusEvaluator b5OneEvaluator(commBase, field, coefficients, controlM5, mf, 1.0, csw,
                                   controlMaxIter, precision);
    const MDWFFiniteDifferenceResult<double> b5OneResult = evaluateMDWFFiniteDifferenceAction(
        gaugePlus, gaugeMinus, baseGauge, makeProbe(finestEpsilon), b5OneEvaluator);
    logMDWFMobiusFiniteDifference("Mobius b5 = 1 (M5 = -2)", finestEpsilon, b5OneResult);

    ShamirEvaluator shamirEvaluator(commBase, field, coefficients, controlM5, mf, csw,
                                    controlMaxIter, precision);
    const MDWFFiniteDifferenceResult<double> shamirResult = evaluateMDWFFiniteDifferenceAction(
        gaugePlus, gaugeMinus, baseGauge, makeProbe(finestEpsilon), shamirEvaluator);
    logMDWFMobiusFiniteDifference("Shamir clover (M5 = -2)", finestEpsilon, shamirResult);

    MobiusEvaluator cswZeroEvaluator(commBase, field, coefficients, controlM5, mf, genericB5, 0.0,
                                     controlMaxIter, precision);
    const MDWFFiniteDifferenceResult<double> cswZeroResult = evaluateMDWFFiniteDifferenceAction(
        gaugePlus, gaugeMinus, baseGauge, makeProbe(finestEpsilon), cswZeroEvaluator);
    logMDWFMobiusFiniteDifference("control c_sw = 0 (M5 = -2, b5 = 1.5)", finestEpsilon, cswZeroResult);

    const double b5OneShamirDerivativeDiff = std::abs(b5OneResult.derivative - shamirResult.derivative);
    const double b5OneShamirActionDiff = std::max(std::abs(b5OneResult.plus.action_real - shamirResult.plus.action_real),
                                                  std::abs(b5OneResult.minus.action_real - shamirResult.minus.action_real));
    const double b5Response = std::abs(controlDerivative[2] - b5OneResult.derivative);
    const double cswResponse = std::abs(controlDerivative[2] - cswZeroResult.derivative);

    // --- Part 4: physical-like epsilon sweep. ---
    MobiusEvaluator physicalEvaluator(commBase, field, coefficients, physicalM5, mf, genericB5, csw,
                                      physicalMaxIter, precision);
    const double physicalEpsilons[2] = {3e-4, 1e-4};
    double physicalDerivative[2] = {0.0, 0.0};
    bool physicalValid = true;
    for (size_t idx = 0; idx < 2; idx++) {
        const MDWFFiniteDifferenceResult<double> result = evaluateMDWFFiniteDifferenceAction(
            gaugePlus, gaugeMinus, baseGauge, makeProbe(physicalEpsilons[idx]), physicalEvaluator);
        logMDWFMobiusFiniteDifference("physical-like (M5 = 1.8, mf = 0.05, b5 = 1.5)",
                                      physicalEpsilons[idx], result);
        physicalDerivative[idx] = result.derivative;
        physicalValid = physicalValid && mdwfMobiusFiniteDifferenceResultValid(result);
    }
    const double physicalScale = std::max(1.0, std::abs(physicalDerivative[1]));
    const double physicalMediumFineRelDiff = std::abs(physicalDerivative[0] - physicalDerivative[1]) / physicalScale;

    rootLogger.info("MDWF Mobius finite difference summary: controlDerivative_1e-4 = ", controlDerivative[2],
                    ", controlCoarseFineRelDiff = ", controlCoarseFineRelDiff,
                    ", controlMediumFineRelDiff = ", controlMediumFineRelDiff,
                    ", b5OneDerivative = ", b5OneResult.derivative,
                    ", shamirDerivative = ", shamirResult.derivative,
                    ", b5OneShamirDerivativeDiff = ", b5OneShamirDerivativeDiff,
                    ", b5OneShamirActionDiff = ", b5OneShamirActionDiff,
                    ", b5Response = ", b5Response,
                    ", cswZeroDerivative = ", cswZeroResult.derivative,
                    ", cswResponse = ", cswResponse,
                    ", physicalDerivative_1e-4 = ", physicalDerivative[1],
                    ", physicalMediumFineRelDiff = ", physicalMediumFineRelDiff);

    const bool controlPassed = controlValid
                               && std::abs(controlDerivative[2]) > 1e-10
                               && controlCoarseFineRelDiff <= stabilityTolerance
                               && controlMediumFineRelDiff <= stabilityTolerance;
    const bool regressionPassed = mdwfMobiusFiniteDifferenceResultValid(b5OneResult)
                                  && mdwfMobiusFiniteDifferenceResultValid(shamirResult)
                                  && b5OneShamirDerivativeDiff == 0.0
                                  && b5OneShamirActionDiff == 0.0;
    const bool responsePassed = mdwfMobiusFiniteDifferenceResultValid(cswZeroResult)
                                && b5Response > 1e-8
                                && cswResponse > 1e-8;
    const bool physicalPassed = physicalValid
                                && std::abs(physicalDerivative[1]) > 1e-10
                                && physicalMediumFineRelDiff <= stabilityTolerance;

    if (!controlPassed || !regressionPassed || !responsePassed || !physicalPassed) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Mobius finite-difference test failed: control passed = ", controlPassed,
            ", b5 = 1 regression passed = ", regressionPassed,
            ", response passed = ", responsePassed,
            ", physical-like passed = ", physicalPassed,
            " (see diagnostics above)"));
    }

    rootLogger.info("MDWF Mobius finite-difference test passed with Ls = ", Ls, ", c_sw = ", csw,
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

        runMDWFMobiusFiniteDifferenceTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
