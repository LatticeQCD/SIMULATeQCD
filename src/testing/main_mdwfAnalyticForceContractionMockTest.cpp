/*
 * MDWF analytic force-contraction mock test.
 *
 * This validates the pre-force comparison boundary against the
 * finite-difference harness using a controlled single-link linear action,
 *
 *     S(U) = const + Re Tr[A U(x,mu)].
 *
 * The analytic directional derivative is contracted with the same generator,
 * site, direction, and left/right perturbation convention used by the
 * finite-difference harness.  This test does not accumulate gauge force,
 * update momenta, call RHMC/HMC, touch HISQ, or use smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFAnalyticForceContractionCheck.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

template<size_t HaloDepth>
SU3<double> mdwfAnalyticForceContractionMockGetLink(
    Gaugefield<double, true, HaloDepth, R18> &gauge,
    const MDWFFiniteDifferenceProbe<double> &probe,
    CommunicationBase &commBase,
    const std::string &name) {

    typedef GIndexer<All, HaloDepth> GInd;

    Gaugefield<double, false, HaloDepth, R18> gaugeHost(commBase, name + "_host");
    gaugeHost = gauge;

    SU3Accessor<double, R18> gaugeAcc = gaugeHost.getAccessor();
    const gSiteMu siteMu = GInd::getSiteMu(probe.x, probe.y, probe.z, probe.t, probe.mu);
    return gaugeAcc.getLink(siteMu);
}

template<size_t HaloDepth>
class MDWFSingleLinkLinearActionEvaluator {
public:
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;

private:
    CommunicationBase &_commBase;
    MDWFFiniteDifferenceProbe<double> _probe;
    SU3<double> _linear_action_matrix;
    double _action_offset;
    std::string _name;

public:
    MDWFSingleLinkLinearActionEvaluator(
        CommunicationBase &commBase,
        const MDWFFiniteDifferenceProbe<double> &probe,
        const SU3<double> &linear_action_matrix,
        double action_offset,
        const std::string &name)
        : _commBase(commBase),
          _probe(probe),
          _linear_action_matrix(linear_action_matrix),
          _action_offset(action_offset),
          _name(name) {}

    MDWFFiniteDifferenceActionValue<double> operator()(Gauge &gauge) {
        const SU3<double> link = mdwfAnalyticForceContractionMockGetLink<HaloDepth>(
            gauge, _probe, _commBase, _name);
        const double actionReal = _action_offset + real(tr_c(_linear_action_matrix, link));

        return {
            actionReal,
            0.0,
            0.0,
            true
        };
    }
};

template<size_t HaloDepth>
MDWFAnalyticForceContractionResult<double> runMDWFAnalyticForceContractionMockCase(
    Gaugefield<double, true, HaloDepth, R18> &baseGauge,
    Gaugefield<double, true, HaloDepth, R18> &gaugePlus,
    Gaugefield<double, true, HaloDepth, R18> &gaugeMinus,
    const MDWFFiniteDifferenceProbe<double> &probe,
    const SU3<double> &linearActionMatrix,
    double actionOffset,
    CommunicationBase &commBase,
    const std::string &name) {

    MDWFSingleLinkLinearActionEvaluator<HaloDepth> actionEvaluator(
        commBase, probe, linearActionMatrix, actionOffset, name);
    MDWFFiniteDifferenceResult<double> finiteDifference = evaluateMDWFFiniteDifferenceAction(
        gaugePlus, gaugeMinus, baseGauge, probe, actionEvaluator);

    const SU3<double> baseLink = mdwfAnalyticForceContractionMockGetLink<HaloDepth>(
        baseGauge, probe, commBase, name + "_base");
    const SU3<double> localActionDerivative = mdwfSingleLinkLinearActionDerivativeMatrix(
        baseLink, linearActionMatrix, probe.multiplication_side);
    const double analyticDerivative = mdwfContractSingleLinkActionDerivative(
        localActionDerivative, probe);
    MDWFAnalyticForceContractionResult<double> result = compareMDWFAnalyticForceContraction(
        finiteDifference, analyticDerivative, 1e-6, 1e-8);

    if (!result.passed
        || finiteDifference.max_shifted_residual != 0.0
        || finiteDifference.action_imag_relative != 0.0
        || finiteDifference.plus.action_real <= 0.0
        || finiteDifference.minus.action_real <= 0.0) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF analytic force-contraction mock case failed for ", name,
            ": finiteDifference = ", result.finite_difference_derivative,
            ", analytic = ", result.analytic_derivative,
            ", absDiff = ", result.absolute_difference,
            ", relDiff = ", result.relative_difference,
            ", maxShiftedResidual = ", finiteDifference.max_shifted_residual,
            ", actionImagRel = ", finiteDifference.action_imag_relative,
            ", actionPlus = ", finiteDifference.plus.action_real,
            ", actionMinus = ", finiteDifference.minus.action_real));
    }

    return result;
}

template<size_t Ls>
void runMDWFAnalyticForceContractionMockTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;

    Gauge baseGauge(commBase, "MDWF_analytic_force_contraction_mock_base_gauge");
    Gauge gaugePlus(commBase, "MDWF_analytic_force_contraction_mock_gauge_plus");
    Gauge gaugeMinus(commBase, "MDWF_analytic_force_contraction_mock_gauge_minus");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260515);
    d_rand = h_rand;
    baseGauge.random(d_rand.state);
    baseGauge.updateAll();

    const COMPLEX(double) zero(0.0, 0.0);
    const SU3<double> actionMatrixLeft(
        COMPLEX(double)(0.11, 0.03), COMPLEX(double)(-0.07, 0.02), zero,
        COMPLEX(double)(0.05, -0.04), COMPLEX(double)(-0.13, 0.01), COMPLEX(double)(0.09, 0.06),
        zero, COMPLEX(double)(-0.02, 0.08), COMPLEX(double)(0.04, -0.05));
    const SU3<double> actionMatrixRight(
        COMPLEX(double)(-0.06, 0.05), zero, COMPLEX(double)(0.12, -0.01),
        COMPLEX(double)(0.03, 0.07), COMPLEX(double)(0.08, -0.02), COMPLEX(double)(-0.04, 0.09),
        COMPLEX(double)(-0.11, -0.03), COMPLEX(double)(0.06, 0.04), COMPLEX(double)(0.02, 0.01));

    const MDWFFiniteDifferenceProbe<double> leftProbe{
        1, 2, 3, 0,
        1,
        0,
        1e-5,
        MDWFFiniteDifferenceMultiplicationSide::Left
    };
    const MDWFFiniteDifferenceProbe<double> rightProbe{
        2, 1, 3, 0,
        2,
        1,
        1e-5,
        MDWFFiniteDifferenceMultiplicationSide::Right
    };

    const MDWFAnalyticForceContractionResult<double> leftResult
        = runMDWFAnalyticForceContractionMockCase(
            baseGauge, gaugePlus, gaugeMinus, leftProbe, actionMatrixLeft, 10.0, commBase,
            "MDWF_analytic_force_contraction_left_mock");
    const MDWFAnalyticForceContractionResult<double> rightResult
        = runMDWFAnalyticForceContractionMockCase(
            baseGauge, gaugePlus, gaugeMinus, rightProbe, actionMatrixRight, 10.0, commBase,
            "MDWF_analytic_force_contraction_right_mock");

    rootLogger.info("MDWF analytic force-contraction mock test passed with Ls = ", Ls,
                    ", leftFiniteDifference = ", leftResult.finite_difference_derivative,
                    ", leftAnalytic = ", leftResult.analytic_derivative,
                    ", leftAbsDiff = ", leftResult.absolute_difference,
                    ", rightFiniteDifference = ", rightResult.finite_difference_derivative,
                    ", rightAnalytic = ", rightResult.analytic_derivative,
                    ", rightAbsDiff = ", rightResult.absolute_difference);
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

        runMDWFAnalyticForceContractionMockTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
