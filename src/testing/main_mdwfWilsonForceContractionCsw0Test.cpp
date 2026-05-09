/*
 * MDWF c_sw = 0 Wilson-path force-contraction smoke test.
 *
 * This compares the finite-difference MDWF rational-action derivative against
 * a test-local scalar contraction of the unclovered Wilson hopping variation,
 *
 *     dS_i = -2 alpha_i Re[ eta_i^\dagger (dM/depsilon) chi_i ].
 *
 * The derivative helper mirrors the `DiracWilsonEvenOdd2` stencil used by the
 * current MDWF Wilson path.  It computes one scalar for one selected link and
 * generator.  It does not accumulate gauge force, update momenta, call
 * RHMC/HMC, touch HISQ, use smearing, or implement a production force kernel.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFAdjointOperator.h"
#include "../experimental/mdwf/MDWFAnalyticForceContractionCheck.h"
#include "../experimental/mdwf/MDWFFermionForceWorkspace.h"
#include "../experimental/mdwf/MDWFFiniteDifferenceHarness.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"
#include "../experimental/mdwf/MDWFRationalCoefficientAdapter.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFWilsonForceContractionCsw0Source {
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
class MDWFWilsonForceContractionCsw0ActionEvaluator {
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
    int _max_iter;
    double _precision;

public:
    MDWFWilsonForceContractionCsw0ActionEvaluator(
        CommunicationBase &commBase,
        Spinor &field,
        const MDWFRationalCoefficients<double> &coefficients,
        MDWFFifthDimCoefficients<double> fifth_coeff,
        double mass,
        int max_iter,
        double precision)
        : _commBase(commBase),
          _field(field),
          _coefficients(coefficients),
          _fifth_coeff(fifth_coeff),
          _mass(mass),
          _max_iter(max_iter),
          _precision(precision) {}

    MDWFFiniteDifferenceActionValue<double> operator()(Gauge &gauge) {
        const double csw = 0.0;

        ForwardOperator forward(gauge, _fifth_coeff, _mass, csw,
                                "MDWF_wilson_force_contraction_csw0_forward");
        AdjointOperator adjoint(gauge, _fifth_coeff, _mass, csw,
                                "MDWF_wilson_force_contraction_csw0_adjoint");
        NormalOperator normal(_commBase, forward, adjoint,
                              "MDWF_wilson_force_contraction_csw0_normal");
        Adapter adapter(normal);
        Spinor actionWorkspace(_commBase, "MDWF_wilson_force_contraction_csw0_action_workspace");

        MDWFRationalActionResult<double> actionResult
            = computeMDWFRationalAction<double, Adapter>(
                adapter, actionWorkspace, _field, _coefficients, _max_iter, _precision,
                "MDWF_wilson_force_contraction_csw0_action");

        return makeMDWFFiniteDifferenceActionValue(actionResult);
    }
};

template<class floatT>
ColorVect<floatT> mdwfWilsonForceGamma(uint8_t mu, const ColorVect<floatT> &spinor) {
    if (mu == 0) {
        return GammaXMultVec(spinor);
    }
    if (mu == 1) {
        return GammaYMultVec(spinor);
    }
    if (mu == 2) {
        return GammaZMultVec(spinor);
    }
    return GammaTMultVec(spinor);
}

template<class floatT>
COMPLEX(double) mdwfWilsonForceColorVectDot(const ColorVect<floatT> &left,
                                            const ColorVect<floatT> &right) {
    COMPLEX(double) result(0.0, 0.0);
    for (size_t spin = 0; spin < 4; spin++) {
        result += left[spin] * right[spin];
    }
    return result;
}

template<class floatT>
SU3<floatT> mdwfWilsonForceLinkDerivative(
    const SU3<floatT> &link,
    const MDWFFiniteDifferenceProbe<floatT> &probe) {

    const SU3<floatT> generator = mdwfFiniteDifferenceGenerator<floatT>(probe.generator_id);
    if (probe.multiplication_side == MDWFFiniteDifferenceMultiplicationSide::Right) {
        return link * generator;
    }
    return generator * link;
}

template<class floatT>
SU3<floatT> mdwfWilsonForceLinkDaggerDerivative(
    const SU3<floatT> &link,
    const MDWFFiniteDifferenceProbe<floatT> &probe) {

    const SU3<floatT> linkDagger = dagger(link);
    const SU3<floatT> generator = mdwfFiniteDifferenceGenerator<floatT>(probe.generator_id);
    if (probe.multiplication_side == MDWFFiniteDifferenceMultiplicationSide::Right) {
        return static_cast<floatT>(-1.0) * generator * linkDagger;
    }
    return static_cast<floatT>(-1.0) * linkDagger * generator;
}

template<size_t HaloDepth>
SU3<double> mdwfWilsonForceContractionGetLink(
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

template<class DeviceSpinor, size_t Ls>
double mdwfWilsonForceContractionTerm(
    const DeviceSpinor &chiDevice,
    const DeviceSpinor &etaDevice,
    Gaugefield<double, true, 2, R18> &gauge,
    const MDWFFiniteDifferenceProbe<double> &probe,
    CommunicationBase &commBase,
    const std::string &name) {

    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;

    MDWFSpinor<double, false, All, HaloDepth, Ls> chiHost(commBase, name + "_chi_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> etaHost(commBase, name + "_eta_host");
    chiHost = chiDevice;
    etaHost = etaDevice;

    Vect12ArrayAcc<double> chiAcc = chiHost.getAccessor();
    Vect12ArrayAcc<double> etaAcc = etaHost.getAccessor();

    const SU3<double> link = mdwfWilsonForceContractionGetLink<HaloDepth>(
        gauge, probe, commBase, name + "_gauge");
    const SU3<double> dLink = mdwfWilsonForceLinkDerivative(link, probe);
    const SU3<double> dLinkDagger = mdwfWilsonForceLinkDaggerDerivative(link, probe);

    const gSite linkSite = GInd::getSite(probe.x, probe.y, probe.z, probe.t);
    const gSite forwardOutputSite = GInd::site_up(linkSite, probe.mu);
    COMPLEX(double) contraction(0.0, 0.0);

    for (size_t stack = 0; stack < Ls; stack++) {
        const gSiteStack outputForward = GInd::getSiteStack(linkSite, stack);
        const gSiteStack inputForward = GInd::site_up(outputForward, probe.mu);
        const gSiteStack outputBackward = GInd::getSiteStack(forwardOutputSite, stack);
        const gSiteStack inputBackward = GInd::site_dn(outputBackward, probe.mu);

        Vect12<double> chiForwardVect = chiAcc.getElement(inputForward);
        Vect12<double> chiBackwardVect = chiAcc.getElement(inputBackward);
        Vect12<double> etaForwardVect = etaAcc.getElement(outputForward);
        Vect12<double> etaBackwardVect = etaAcc.getElement(outputBackward);

        const ColorVect<double> chiForward = convertVect12ToColorVect(chiForwardVect);
        const ColorVect<double> chiBackward = convertVect12ToColorVect(chiBackwardVect);
        const ColorVect<double> etaForward = convertVect12ToColorVect(etaForwardVect);
        const ColorVect<double> etaBackward = convertVect12ToColorVect(etaBackwardVect);

        const ColorVect<double> forwardHop = dLink * chiForward;
        const ColorVect<double> backwardHop = dLinkDagger * chiBackward;

        const ColorVect<double> dMForward = static_cast<double>(0.5)
                                            * (mdwfWilsonForceGamma(probe.mu, forwardHop)
                                               - forwardHop);
        const ColorVect<double> dMBackward = static_cast<double>(-0.5)
                                             * (backwardHop
                                                + mdwfWilsonForceGamma(probe.mu, backwardHop));

        contraction += mdwfWilsonForceColorVectDot(etaForward, dMForward);
        contraction += mdwfWilsonForceColorVectDot(etaBackward, dMBackward);
    }

    return real(contraction);
}

template<class Workspace, class Spinor, size_t Ls>
double mdwfWilsonForceContractionAnalyticDerivative(
    Workspace &workspace,
    Gaugefield<double, true, 2, R18> &gauge,
    const MDWFRationalCoefficients<double> &forceCoefficients,
    const MDWFFiniteDifferenceProbe<double> &probe,
    CommunicationBase &commBase) {

    double derivative = 0.0;
    for (size_t term = 0; term < workspace.size(); term++) {
        const double localContraction = mdwfWilsonForceContractionTerm<Spinor, Ls>(
            workspace.chi(term), workspace.eta(term), gauge, probe, commBase,
            "MDWF_wilson_force_contraction_term_" + std::to_string(term));
        derivative += -2.0 * forceCoefficients.numerator[term] * localContraction;
    }
    return derivative;
}

template<size_t Ls>
void runMDWFWilsonForceContractionCsw0Test(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    const double mass = 4.0;
    const double csw = 0.0;

    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using ForwardOperator = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using AdjointOperator = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using Workspace = MDWFFermionForceWorkspace<double, HaloDepth, HaloDepth, Ls,
                                                NormalOperator, ForwardOperator>;
    using Spinor = typename NormalOperator::Spinor;

    MDWFExplicitRationalInput<double> actionInput{
        "wilson_force_contraction_csw0_action_coefficients",
        MDWFRationalCoefficientRole::Action,
        0.125,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    MDWFExplicitRationalInput<double> forceInput{
        "wilson_force_contraction_csw0_force_coefficients",
        MDWFRationalCoefficientRole::Force,
        0.0,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    MDWFRationalCoefficients<double> actionCoefficients = makeMDWFRationalCoefficients(actionInput);
    MDWFRationalCoefficients<double> forceCoefficients = makeMDWFRationalCoefficients(forceInput);

    Gauge baseGauge(commBase, "MDWF_wilson_force_contraction_csw0_base_gauge");
    Gauge gaugePlus(commBase, "MDWF_wilson_force_contraction_csw0_gauge_plus");
    Gauge gaugeMinus(commBase, "MDWF_wilson_force_contraction_csw0_gauge_minus");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260514);
    d_rand = h_rand;
    baseGauge.random(d_rand.state);
    baseGauge.updateAll();

    Spinor field(commBase, "MDWF_wilson_force_contraction_csw0_field");
    field.template iterateOverBulk<>(
        FillMDWFWilsonForceContractionCsw0Source<double, All, HaloDepth, Ls>());
    field.updateAll();

    MDWFFifthDimCoefficients<double> fifthCoeff(1.0, -0.05, -0.05, 0.0, 0.0);
    MDWFFiniteDifferenceProbe<double> probe{
        1, 2, 3, 0,
        1,
        0,
        1e-4,
        MDWFFiniteDifferenceMultiplicationSide::Left
    };

    MDWFWilsonForceContractionCsw0ActionEvaluator<HaloDepth, Ls> actionEvaluator(
        commBase, field, actionCoefficients, fifthCoeff, mass, 512, 1e-8);
    MDWFFiniteDifferenceResult<double> finiteDifference = evaluateMDWFFiniteDifferenceAction(
        gaugePlus, gaugeMinus, baseGauge, probe, actionEvaluator);

    ForwardOperator forward(baseGauge, fifthCoeff, mass, csw,
                            "MDWF_wilson_force_contraction_csw0_forward");
    AdjointOperator adjoint(baseGauge, fifthCoeff, mass, csw,
                            "MDWF_wilson_force_contraction_csw0_adjoint");
    NormalOperator normal(commBase, forward, adjoint,
                          "MDWF_wilson_force_contraction_csw0_normal");
    Workspace workspace;
    workspace.prepare(normal, forward, field, forceCoefficients, 512, 1e-8,
                      "MDWF_wilson_force_contraction_csw0_workspace");

    double maxForceWorkspaceResidue = 0.0;
    for (const auto &info : workspace.shiftInfo()) {
        maxForceWorkspaceResidue = std::max(maxForceWorkspaceResidue, info.residue);
    }

    const double analyticDerivative
        = mdwfWilsonForceContractionAnalyticDerivative<Workspace, Spinor, Ls>(
            workspace, baseGauge, forceCoefficients, probe, commBase);
    MDWFAnalyticForceContractionResult<double> comparison
        = compareMDWFAnalyticForceContraction(finiteDifference, analyticDerivative, 5e-3, 5e-4);

    if (mdwfRationalCoefficientRoleName(actionInput.role) != "action"
        || mdwfRationalCoefficientRoleName(forceInput.role) != "force"
        || !finiteDifference.converged
        || !workspace.converged()
        || finiteDifference.max_shifted_residual > 1e-8
        || maxForceWorkspaceResidue > 1e-8
        || finiteDifference.action_imag_relative > 1e-8
        || !comparison.passed) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF c_sw = 0 Wilson force-contraction test failed: finiteDifference = ",
            comparison.finite_difference_derivative,
            ", analytic = ", comparison.analytic_derivative,
            ", absDiff = ", comparison.absolute_difference,
            ", relDiff = ", comparison.relative_difference,
            ", finiteDifferenceConverged = ", finiteDifference.converged,
            ", workspaceConverged = ", workspace.converged(),
            ", actionMaxResidual = ", finiteDifference.max_shifted_residual,
            ", forceMaxResidual = ", maxForceWorkspaceResidue,
            ", actionImagRel = ", finiteDifference.action_imag_relative));
    }

    rootLogger.info("MDWF c_sw = 0 Wilson force-contraction test passed with Ls = ", Ls,
                    ", finiteDifference = ", comparison.finite_difference_derivative,
                    ", analytic = ", comparison.analytic_derivative,
                    ", absDiff = ", comparison.absolute_difference,
                    ", relDiff = ", comparison.relative_difference,
                    ", actionMaxResidual = ", finiteDifference.max_shifted_residual,
                    ", forceMaxResidual = ", maxForceWorkspaceResidue);
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

        runMDWFWilsonForceContractionCsw0Test<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
