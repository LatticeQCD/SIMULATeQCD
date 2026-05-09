/*
 * MDWF nonzero-c_sw finite-difference action smoke test.
 *
 * This uses the finite-difference harness with the actual MDWF normal-action
 * path and c_sw = 0.5.  It perturbs one gauge link, evaluates
 *
 *     [S(U_+) - S(U_-)] / (2 epsilon)
 *
 * through computeMDWFRationalAction, and checks that the result is finite and
 * stable across a small epsilon sweep.  It does not allocate or accumulate
 * gauge force, update momenta, call RHMC/HMC, touch HISQ, or use smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFAdjointOperator.h"
#include "../experimental/mdwf/MDWFFiniteDifferenceHarness.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"
#include "../experimental/mdwf/MDWFRationalCoefficientAdapter.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFFiniteDifferenceMdwfNonzeroCswSource {
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
class MDWFFiniteDifferenceMdwfNonzeroCswActionEvaluator {
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
    MDWFFiniteDifferenceMdwfNonzeroCswActionEvaluator(
        CommunicationBase &commBase,
        Spinor &field,
        const MDWFRationalCoefficients<double> &coefficients,
        MDWFFifthDimCoefficients<double> fifth_coeff,
        double mass,
        double csw,
        int max_iter,
        double precision)
        : _commBase(commBase),
          _field(field),
          _coefficients(coefficients),
          _fifth_coeff(fifth_coeff),
          _mass(mass),
          _csw(csw),
          _max_iter(max_iter),
          _precision(precision) {}

    MDWFFiniteDifferenceActionValue<double> operator()(Gauge &gauge) {
        ForwardOperator forward(gauge, _fifth_coeff, _mass, _csw,
                                "MDWF_finite_difference_nonzero_csw_forward");
        AdjointOperator adjoint(gauge, _fifth_coeff, _mass, _csw,
                                "MDWF_finite_difference_nonzero_csw_adjoint");
        NormalOperator normal(_commBase, forward, adjoint,
                              "MDWF_finite_difference_nonzero_csw_normal");
        Adapter adapter(normal);
        Spinor actionWorkspace(_commBase, "MDWF_finite_difference_nonzero_csw_action_workspace");

        MDWFRationalActionResult<double> actionResult
            = computeMDWFRationalAction<double, Adapter>(
                adapter, actionWorkspace, _field, _coefficients, _max_iter, _precision,
                "MDWF_finite_difference_nonzero_csw_action");

        return makeMDWFFiniteDifferenceActionValue(actionResult);
    }
};

template<size_t HaloDepth>
double mdwfFiniteDifferenceNonzeroCswPerturbedLinkDifference(
    Gaugefield<double, true, HaloDepth, R18> &gaugePlus,
    Gaugefield<double, true, HaloDepth, R18> &gaugeMinus,
    const MDWFFiniteDifferenceProbe<double> &probe,
    CommunicationBase &commBase) {

    typedef GIndexer<All, HaloDepth> GInd;

    Gaugefield<double, false, HaloDepth, R18> gaugePlusHost(
        commBase, "MDWF_fd_nonzero_csw_plus_host");
    Gaugefield<double, false, HaloDepth, R18> gaugeMinusHost(
        commBase, "MDWF_fd_nonzero_csw_minus_host");
    gaugePlusHost = gaugePlus;
    gaugeMinusHost = gaugeMinus;

    SU3Accessor<double, R18> plusAcc = gaugePlusHost.getAccessor();
    SU3Accessor<double, R18> minusAcc = gaugeMinusHost.getAccessor();
    const gSiteMu siteMu = GInd::getSiteMu(probe.x, probe.y, probe.z, probe.t, probe.mu);
    const SU3<double> linkDiff = plusAcc.getLink(siteMu) - minusAcc.getLink(siteMu);

    return infnorm(linkDiff);
}

template<size_t Ls>
void runMDWFFiniteDifferenceMdwfNonzeroCswTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    const double nonzeroCsw = 0.5;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;

    MDWFExplicitRationalInput<double> actionInput{
        "finite_difference_mdwf_nonzero_csw_action_coefficients",
        MDWFRationalCoefficientRole::Action,
        0.125,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    MDWFRationalCoefficients<double> coefficients = makeMDWFRationalCoefficients(actionInput);

    Gauge baseGauge(commBase, "MDWF_finite_difference_nonzero_csw_base_gauge");
    Gauge gaugePlus(commBase, "MDWF_finite_difference_nonzero_csw_gauge_plus");
    Gauge gaugeMinus(commBase, "MDWF_finite_difference_nonzero_csw_gauge_minus");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260514);
    d_rand = h_rand;
    baseGauge.random(d_rand.state);
    baseGauge.updateAll();

    Spinor field(commBase, "MDWF_finite_difference_nonzero_csw_field");
    field.template iterateOverBulk<>(
        FillMDWFFiniteDifferenceMdwfNonzeroCswSource<double, All, HaloDepth, Ls>());
    field.updateAll();

    MDWFFifthDimCoefficients<double> fifthCoeff(1.0, -0.05, -0.05, 0.0, 0.0);
    MDWFFiniteDifferenceMdwfNonzeroCswActionEvaluator<HaloDepth, Ls> nonzeroCswEvaluator(
        commBase, field, coefficients, fifthCoeff, 4.0, nonzeroCsw, 512, 1e-8);
    MDWFFiniteDifferenceMdwfNonzeroCswActionEvaluator<HaloDepth, Ls> csw0Evaluator(
        commBase, field, coefficients, fifthCoeff, 4.0, 0.0, 512, 1e-8);

    const double epsilons[3] = {1e-3, 3e-4, 1e-4};
    double derivative[3] = {0.0, 0.0, 0.0};
    double maxResidual = 0.0;
    double maxImagRel = 0.0;
    double minPerturbedLinkDiff = 1.0e300;
    MDWFFiniteDifferenceResult<double> finestNonzeroCswResult;
    MDWFFiniteDifferenceResult<double> finestCsw0Result;

    for (size_t idx = 0; idx < 3; idx++) {
        MDWFFiniteDifferenceProbe<double> probe{
            1, 2, 3, 0,
            1,
            0,
            epsilons[idx],
            MDWFFiniteDifferenceMultiplicationSide::Left
        };

        MDWFFiniteDifferenceResult<double> result = evaluateMDWFFiniteDifferenceAction(
            gaugePlus, gaugeMinus, baseGauge, probe, nonzeroCswEvaluator);
        const double perturbedLinkDiff = mdwfFiniteDifferenceNonzeroCswPerturbedLinkDifference(
            gaugePlus, gaugeMinus, probe, commBase);

        derivative[idx] = result.derivative;
        maxResidual = std::max(maxResidual, result.max_shifted_residual);
        maxImagRel = std::max(maxImagRel, result.action_imag_relative);
        minPerturbedLinkDiff = std::min(minPerturbedLinkDiff, perturbedLinkDiff);

        if (!result.converged
            || !std::isfinite(result.derivative)
            || result.max_shifted_residual > 1e-8
            || result.action_imag_relative > 1e-8
            || result.plus.action_real <= 0.0
            || result.minus.action_real <= 0.0
            || perturbedLinkDiff <= 0.0) {
            throw std::runtime_error(stdLogger.fatal(
                "MDWF nonzero-c_sw finite-difference action test failed for epsilon = ",
                epsilons[idx],
                ": c_sw = ", nonzeroCsw,
                ", converged = ", result.converged,
                ", derivative = ", result.derivative,
                ", actionPlus = ", result.plus.action_real,
                ", actionMinus = ", result.minus.action_real,
                ", actionImagRelative = ", result.action_imag_relative,
                ", maxShiftedResidual = ", result.max_shifted_residual,
                ", perturbedLinkDiff = ", perturbedLinkDiff));
        }

        if (idx == 2) {
            finestNonzeroCswResult = result;
            finestCsw0Result = evaluateMDWFFiniteDifferenceAction(
                gaugePlus, gaugeMinus, baseGauge, probe, csw0Evaluator);
            maxResidual = std::max(maxResidual, finestCsw0Result.max_shifted_residual);
            maxImagRel = std::max(maxImagRel, finestCsw0Result.action_imag_relative);
        }
    }

    const double derivativeScale = std::max(1.0, std::abs(derivative[2]));
    const double coarseFineRelDiff = std::abs(derivative[0] - derivative[2]) / derivativeScale;
    const double mediumFineRelDiff = std::abs(derivative[1] - derivative[2]) / derivativeScale;
    const double csw0Derivative = finestCsw0Result.derivative;
    const double derivativeCswResponse = std::abs(derivative[2] - csw0Derivative);
    const double actionCswResponse = std::abs(
        finestNonzeroCswResult.plus.action_real - finestCsw0Result.plus.action_real);
    const double actionScale = std::max(
        1.0, std::max(std::abs(finestNonzeroCswResult.plus.action_real),
                      std::abs(finestCsw0Result.plus.action_real)));
    const double actionCswResponseRel = actionCswResponse / actionScale;

    if (mdwfRationalCoefficientRoleName(actionInput.role) != "action"
        || std::abs(derivative[2]) <= 1e-10
        || !finestCsw0Result.converged
        || finestCsw0Result.max_shifted_residual > 1e-8
        || finestCsw0Result.action_imag_relative > 1e-8
        || !std::isfinite(csw0Derivative)
        || derivativeCswResponse <= 1e-8
        || actionCswResponseRel <= 1e-10
        || coarseFineRelDiff > 0.25
        || mediumFineRelDiff > 0.10) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF nonzero-c_sw finite-difference action stability test failed: c_sw = ",
            nonzeroCsw,
            ", derivative_1e-3 = ", derivative[0],
            ", derivative_3e-4 = ", derivative[1],
            ", derivative_1e-4 = ", derivative[2],
            ", derivative_csw0_1e-4 = ", csw0Derivative,
            ", derivativeCswResponse = ", derivativeCswResponse,
            ", actionCswResponseRel = ", actionCswResponseRel,
            ", coarseFineRelDiff = ", coarseFineRelDiff,
            ", mediumFineRelDiff = ", mediumFineRelDiff,
            ", maxResidual = ", maxResidual,
            ", maxImagRel = ", maxImagRel,
            ", minPerturbedLinkDiff = ", minPerturbedLinkDiff));
    }

    rootLogger.info("MDWF nonzero-c_sw finite-difference action test passed with Ls = ", Ls,
                    ", c_sw = ", nonzeroCsw,
                    ", derivative_1e-3 = ", derivative[0],
                    ", derivative_3e-4 = ", derivative[1],
                    ", derivative_1e-4 = ", derivative[2],
                    ", derivative_csw0_1e-4 = ", csw0Derivative,
                    ", derivativeCswResponse = ", derivativeCswResponse,
                    ", actionCswResponseRel = ", actionCswResponseRel,
                    ", maxResidual = ", maxResidual,
                    ", maxImagRel = ", maxImagRel,
                    ", minPerturbedLinkDiff = ", minPerturbedLinkDiff);
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

        runMDWFFiniteDifferenceMdwfNonzeroCswTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
