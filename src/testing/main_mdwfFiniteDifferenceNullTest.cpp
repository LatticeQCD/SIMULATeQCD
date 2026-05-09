/*
 * MDWF finite-difference null smoke test.
 *
 * This validates the finite-difference action harness before any force
 * accumulation exists.  The gauge link is genuinely perturbed, but the mock
 * normal operator is gauge independent, so
 *
 *     [S(U_+) - S(U_-)] / (2 epsilon) = 0
 *
 * up to numerical precision.  The action is still evaluated through
 * computeMDWFRationalAction.  This test does not allocate or accumulate gauge
 * force, update momenta, call RHMC/HMC, touch HISQ, or use smearing.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFFiniteDifferenceHarness.h"
#include "../experimental/mdwf/MDWFRationalCoefficientAdapter.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct MDWFFiniteDifferenceNullNormalApply {
    Vect12ArrayAcc<floatT> spinor_in;

    explicit MDWFFiniteDifferenceNullNormalApply(
        const MDWFSpinor<floatT, true, LatLayout, HaloDepth, Ls> &spinor_in_in)
        : spinor_in(spinor_in_in.getAccessor()) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        const floatT diagonal = static_cast<floatT>(1.0)
                                + static_cast<floatT>(0.1) * static_cast<floatT>(site.stack);
        return diagonal * spinor_in.getElement(site);
    }
};

template<class floatT, size_t HaloDepthSpin, size_t Ls>
class MDWFFiniteDifferenceNullNormalOperator {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;

    void apply(Spinor &spinor_out, const Spinor &spinor_in, bool update = false) {
        spinor_out.template iterateOverBulk<>(
            MDWFFiniteDifferenceNullNormalApply<floatT, All, HaloDepthSpin, Ls>(spinor_in));
        if (update) {
            spinor_out.updateAll();
        }
    }
};

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFFiniteDifferenceNullSource {
    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t component = 0; component < 12; component++) {
            out.data[component] = COMPLEX(floatT)(
                static_cast<floatT>(0.75)
                + static_cast<floatT>(site.stack + 1)
                + static_cast<floatT>(0.001) * static_cast<floatT>(site.isite + component + 1),
                static_cast<floatT>(0.01) * static_cast<floatT>(component + 1)
                - static_cast<floatT>(0.002) * static_cast<floatT>(site.stack));
        }
        return out;
    }
};

template<size_t HaloDepth, size_t Ls>
class MDWFFiniteDifferenceNullActionEvaluator {
public:
    using NormalOperator = MDWFFiniteDifferenceNullNormalOperator<double, HaloDepth, Ls>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, NormalOperator>;
    using Spinor = typename NormalOperator::Spinor;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;

private:
    CommunicationBase &_commBase;
    Spinor &_field;
    MDWFRationalCoefficients<double> _coefficients;
    int _max_iter;
    double _precision;

public:
    MDWFFiniteDifferenceNullActionEvaluator(CommunicationBase &commBase,
                                            Spinor &field,
                                            const MDWFRationalCoefficients<double> &coefficients,
                                            int max_iter,
                                            double precision)
        : _commBase(commBase),
          _field(field),
          _coefficients(coefficients),
          _max_iter(max_iter),
          _precision(precision) {}

    MDWFFiniteDifferenceActionValue<double> operator()(Gauge &) {
        NormalOperator normalOperator;
        Adapter adapter(normalOperator);
        Spinor actionWorkspace(_commBase, "MDWF_finite_difference_null_action_workspace");

        MDWFRationalActionResult<double> actionResult
            = computeMDWFRationalAction<double, Adapter>(
                adapter, actionWorkspace, _field, _coefficients, _max_iter, _precision,
                "MDWF_finite_difference_null_action");

        return makeMDWFFiniteDifferenceActionValue(actionResult);
    }
};

template<size_t HaloDepth>
double mdwfFiniteDifferencePerturbedLinkDifference(
    Gaugefield<double, true, HaloDepth, R18> &gaugePlus,
    Gaugefield<double, true, HaloDepth, R18> &gaugeMinus,
    const MDWFFiniteDifferenceProbe<double> &probe,
    CommunicationBase &commBase) {

    typedef GIndexer<All, HaloDepth> GInd;

    Gaugefield<double, false, HaloDepth, R18> gaugePlusHost(commBase, "MDWF_fd_plus_host");
    Gaugefield<double, false, HaloDepth, R18> gaugeMinusHost(commBase, "MDWF_fd_minus_host");
    gaugePlusHost = gaugePlus;
    gaugeMinusHost = gaugeMinus;

    SU3Accessor<double, R18> plusAcc = gaugePlusHost.getAccessor();
    SU3Accessor<double, R18> minusAcc = gaugeMinusHost.getAccessor();
    const gSiteMu siteMu = GInd::getSiteMu(probe.x, probe.y, probe.z, probe.t, probe.mu);
    const SU3<double> linkDiff = plusAcc.getLink(siteMu) - minusAcc.getLink(siteMu);

    return infnorm(linkDiff);
}

template<size_t Ls>
void runMDWFFiniteDifferenceNullTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;

    MDWFExplicitRationalInput<double> actionInput{
        "finite_difference_null_action_coefficients",
        MDWFRationalCoefficientRole::Action,
        0.125,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    MDWFRationalCoefficients<double> coefficients = makeMDWFRationalCoefficients(actionInput);

    Gauge baseGauge(commBase, "MDWF_finite_difference_null_base_gauge");
    Gauge gaugePlus(commBase, "MDWF_finite_difference_null_gauge_plus");
    Gauge gaugeMinus(commBase, "MDWF_finite_difference_null_gauge_minus");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260513);
    d_rand = h_rand;
    baseGauge.random(d_rand.state);
    baseGauge.updateAll();

    Spinor field(commBase, "MDWF_finite_difference_null_field");
    field.template iterateOverBulk<>(FillMDWFFiniteDifferenceNullSource<double, All, HaloDepth, Ls>());
    field.updateAll();

    MDWFFiniteDifferenceProbe<double> probe{
        1, 2, 3, 0,
        1,
        0,
        1e-4,
        MDWFFiniteDifferenceMultiplicationSide::Left
    };

    MDWFFiniteDifferenceNullActionEvaluator<HaloDepth, Ls> actionEvaluator(
        commBase, field, coefficients, 64, 1e-12);
    MDWFFiniteDifferenceResult<double> result = evaluateMDWFFiniteDifferenceAction(
        gaugePlus, gaugeMinus, baseGauge, probe, actionEvaluator);

    const double perturbedLinkDiff = mdwfFiniteDifferencePerturbedLinkDifference(
        gaugePlus, gaugeMinus, probe, commBase);

    if (mdwfRationalCoefficientRoleName(actionInput.role) != "action"
        || !result.converged
        || result.max_shifted_residual > 1e-12
        || result.action_imag_relative > 1e-12
        || std::abs(result.derivative) > 1e-8
        || std::abs(result.plus.action_real - result.minus.action_real) > 1e-10
        || perturbedLinkDiff <= 0.0) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF finite-difference null test failed: converged = ", result.converged,
            ", derivative = ", result.derivative,
            ", actionPlus = ", result.plus.action_real,
            ", actionMinus = ", result.minus.action_real,
            ", actionImagRelative = ", result.action_imag_relative,
            ", maxShiftedResidual = ", result.max_shifted_residual,
            ", perturbedLinkDiff = ", perturbedLinkDiff));
    }

    rootLogger.info("MDWF finite-difference null test passed with Ls = ", Ls,
                    ", epsilon = ", probe.epsilon,
                    ", derivative = ", result.derivative,
                    ", actionReal = ", result.plus.action_real,
                    ", maxShiftedResidual = ", result.max_shifted_residual,
                    ", perturbedLinkDiff = ", perturbedLinkDiff);
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

        runMDWFFiniteDifferenceNullTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
