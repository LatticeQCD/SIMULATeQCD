/*
 * MDWF c_sw = 0 Wilson all-link random-direction contraction scaffold.
 *
 * This applies one deterministic anti-Hermitian traceless left perturbation to
 * every bulk gauge link, compares the centered action derivative with the
 * Wilson analytic contraction, and round-trips a direction-dependent test
 * accumulator through a gauge-field-like object.  It is single-rank only and
 * does not define a full force matrix, projection, momentum update, or HMC
 * sign.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFAdjointOperator.h"
#include "../experimental/mdwf/MDWFAllLinkRandomDirection.h"
#include "../experimental/mdwf/MDWFFermionForceWorkspace.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"
#include "../experimental/mdwf/MDWFRationalCoefficientAdapter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFWilsonAllLinkCsw0Source {
    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);

        for (size_t component = 0; component < 12; component++) {
            out.data[component] = COMPLEX(floatT)(
                static_cast<floatT>(0.5)
                + static_cast<floatT>(site.stack + 1)
                + static_cast<floatT>(0.001)
                  * static_cast<floatT>(site.isite + component + 1),
                static_cast<floatT>(0.017) * static_cast<floatT>(component + 1)
                - static_cast<floatT>(0.0025) * static_cast<floatT>(site.stack));
        }
        return out;
    }
};

template<size_t HaloDepth, size_t Ls>
class MDWFWilsonAllLinkCsw0ActionEvaluator {
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
    MDWFWilsonAllLinkCsw0ActionEvaluator(
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
                                "MDWF_wilson_all_link_csw0_forward");
        AdjointOperator adjoint(gauge, _fifth_coeff, _mass, csw,
                                "MDWF_wilson_all_link_csw0_adjoint");
        NormalOperator normal(_commBase, forward, adjoint,
                              "MDWF_wilson_all_link_csw0_normal");
        Adapter adapter(normal);
        Spinor actionWorkspace(_commBase, "MDWF_wilson_all_link_csw0_action_workspace");

        MDWFRationalActionResult<double> actionResult
            = computeMDWFRationalAction<double, Adapter>(
                adapter, actionWorkspace, _field, _coefficients, _max_iter, _precision,
                "MDWF_wilson_all_link_csw0_action");

        return makeMDWFFiniteDifferenceActionValue(actionResult);
    }
};

template<size_t HaloDepth, class ActionEvaluator>
MDWFFiniteDifferenceResult<double> evaluateMDWFWilsonAllLinkFiniteDifference(
    Gaugefield<double, true, HaloDepth, R18> &gauge_plus,
    Gaugefield<double, true, HaloDepth, R18> &gauge_minus,
    const Gaugefield<double, true, HaloDepth, R18> &base_gauge,
    double epsilon,
    ActionEvaluator &action_evaluator) {

    applyMDWFAllLinkPerturbation(gauge_plus, base_gauge, epsilon, 1);
    applyMDWFAllLinkPerturbation(gauge_minus, base_gauge, epsilon, -1);

    MDWFFiniteDifferenceActionValue<double> plus = action_evaluator(gauge_plus);
    MDWFFiniteDifferenceActionValue<double> minus = action_evaluator(gauge_minus);
    const double derivative = (plus.action_real - minus.action_real) / (2.0 * epsilon);
    const double plusScale = std::max(1.0, std::abs(plus.action_real));
    const double minusScale = std::max(1.0, std::abs(minus.action_real));

    return {
        plus,
        minus,
        derivative,
        std::max(std::abs(plus.action_imag) / plusScale,
                 std::abs(minus.action_imag) / minusScale),
        std::max(plus.max_shifted_residual, minus.max_shifted_residual),
        plus.converged && minus.converged
    };
}

template<class floatT>
ColorVect<floatT> mdwfWilsonAllLinkGamma(
    uint8_t mu,
    const ColorVect<floatT> &spinor) {

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
COMPLEX(double) mdwfWilsonAllLinkColorVectDot(
    const ColorVect<floatT> &left,
    const ColorVect<floatT> &right) {

    COMPLEX(double) result(0.0, 0.0);
    for (size_t spin = 0; spin < 4; spin++) {
        result += left[spin] * right[spin];
    }
    return result;
}

template<size_t HaloDepth, size_t Ls>
double mdwfWilsonAllLinkContractionTerm(
    Vect12ArrayAcc<double> chi_acc,
    Vect12ArrayAcc<double> eta_acc,
    SU3Accessor<double, R18> gauge_acc,
    const gSite &link_site,
    uint8_t mu,
    const SU3<double> &direction) {

    typedef GIndexer<All, HaloDepth> GInd;

    const SU3<double> link = gauge_acc.getLink(GInd::getSiteMu(link_site, mu));
    const SU3<double> dLink = direction * link;
    const SU3<double> dLinkDagger
        = static_cast<double>(-1.0) * dagger(link) * direction;
    const gSite forwardOutputSite = GInd::site_up(link_site, mu);
    COMPLEX(double) contraction(0.0, 0.0);

    for (size_t stack = 0; stack < Ls; stack++) {
        const gSiteStack outputForward = GInd::getSiteStack(link_site, stack);
        const gSiteStack inputForward = GInd::site_up(outputForward, mu);
        const gSiteStack outputBackward = GInd::getSiteStack(forwardOutputSite, stack);
        const gSiteStack inputBackward = GInd::site_dn(outputBackward, mu);

        Vect12<double> chiForwardVect = chi_acc.getElement(inputForward);
        Vect12<double> chiBackwardVect = chi_acc.getElement(inputBackward);
        Vect12<double> etaForwardVect = eta_acc.getElement(outputForward);
        Vect12<double> etaBackwardVect = eta_acc.getElement(outputBackward);

        const ColorVect<double> chiForward = convertVect12ToColorVect(
            chiForwardVect);
        const ColorVect<double> chiBackward = convertVect12ToColorVect(
            chiBackwardVect);
        const ColorVect<double> etaForward = convertVect12ToColorVect(
            etaForwardVect);
        const ColorVect<double> etaBackward = convertVect12ToColorVect(
            etaBackwardVect);

        const ColorVect<double> forwardHop = dLink * chiForward;
        const ColorVect<double> backwardHop = dLinkDagger * chiBackward;
        const ColorVect<double> dMForward
            = static_cast<double>(0.5)
              * (mdwfWilsonAllLinkGamma(mu, forwardHop) - forwardHop);
        const ColorVect<double> dMBackward
            = static_cast<double>(-0.5)
              * (backwardHop + mdwfWilsonAllLinkGamma(mu, backwardHop));

        contraction += mdwfWilsonAllLinkColorVectDot(etaForward, dMForward);
        contraction += mdwfWilsonAllLinkColorVectDot(etaBackward, dMBackward);
    }

    return real(contraction);
}

template<size_t Ls>
void runMDWFWilsonAllLinkRandomDirectionCsw0Test(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    const double mass = 4.0;
    const double csw = 0.0;
    typedef GIndexer<All, HaloDepth> GInd;

    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using ForwardOperator = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using AdjointOperator = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using Workspace = MDWFFermionForceWorkspace<double, HaloDepth, HaloDepth, Ls,
                                                NormalOperator, ForwardOperator>;
    using Spinor = typename NormalOperator::Spinor;
    using HostSpinor = MDWFSpinor<double, false, All, HaloDepth, Ls>;

    const LatticeData lat = GInd::getLatData();
    if (lat.vol4 != lat.globvol4) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Wilson all-link random-direction scaffold is single-rank only: "
            "local volume = ", lat.vol4, ", global volume = ", lat.globvol4));
    }

    MDWFExplicitRationalInput<double> actionInput{
        "wilson_all_link_csw0_action_coefficients",
        MDWFRationalCoefficientRole::Action,
        0.125,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    MDWFExplicitRationalInput<double> forceInput{
        "wilson_all_link_csw0_force_coefficients",
        MDWFRationalCoefficientRole::Force,
        0.0,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    MDWFRationalCoefficients<double> actionCoefficients
        = makeMDWFRationalCoefficients(actionInput);
    MDWFRationalCoefficients<double> forceCoefficients
        = makeMDWFRationalCoefficients(forceInput);

    Gauge baseGauge(commBase, "MDWF_wilson_all_link_csw0_base_gauge");
    Gauge gaugePlus(commBase, "MDWF_wilson_all_link_csw0_gauge_plus");
    Gauge gaugeMinus(commBase, "MDWF_wilson_all_link_csw0_gauge_minus");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260514);
    d_rand = h_rand;
    baseGauge.random(d_rand.state);
    baseGauge.updateAll();

    Spinor field(commBase, "MDWF_wilson_all_link_csw0_field");
    field.template iterateOverBulk<>(
        FillMDWFWilsonAllLinkCsw0Source<double, All, HaloDepth, Ls>());
    field.updateAll();
    Spinor forceField(commBase, "MDWF_wilson_all_link_csw0_force_field");
    forceField = field;
    forceField.updateAll();

    MDWFFifthDimCoefficients<double> fifthCoeff(1.0, -0.05, -0.05, 0.0, 0.0);
    MDWFWilsonAllLinkCsw0ActionEvaluator<HaloDepth, Ls> actionEvaluator(
        commBase, field, actionCoefficients, fifthCoeff, mass, 512, 1e-8);
    const std::array<double, 3> epsilons{{1e-3, 3e-4, 1e-4}};
    std::array<MDWFFiniteDifferenceResult<double>, 3> finiteDifferences;
    double maxActionResidual = 0.0;
    double maxActionImagRelative = 0.0;
    bool finiteDifferencesConverged = true;

    for (size_t epsilonIndex = 0; epsilonIndex < epsilons.size(); epsilonIndex++) {
        finiteDifferences[epsilonIndex]
            = evaluateMDWFWilsonAllLinkFiniteDifference(
                gaugePlus, gaugeMinus, baseGauge, epsilons[epsilonIndex],
                actionEvaluator);
        maxActionResidual = std::max(
            maxActionResidual,
            finiteDifferences[epsilonIndex].max_shifted_residual);
        maxActionImagRelative = std::max(
            maxActionImagRelative,
            finiteDifferences[epsilonIndex].action_imag_relative);
        finiteDifferencesConverged
            = finiteDifferencesConverged && finiteDifferences[epsilonIndex].converged;
    }

    ForwardOperator forward(baseGauge, fifthCoeff, mass, csw,
                            "MDWF_wilson_all_link_csw0_forward");
    AdjointOperator adjoint(baseGauge, fifthCoeff, mass, csw,
                            "MDWF_wilson_all_link_csw0_adjoint");
    NormalOperator normal(commBase, forward, adjoint,
                          "MDWF_wilson_all_link_csw0_normal");
    Workspace workspace;
    workspace.prepare(normal, forward, forceField, forceCoefficients, 512, 1e-8,
                      "MDWF_wilson_all_link_csw0_workspace");

    double maxForceWorkspaceResidual = 0.0;
    for (const auto &info : workspace.shiftInfo()) {
        maxForceWorkspaceResidual = std::max(
            maxForceWorkspaceResidual, info.residue);
    }

    HostGauge gaugeHost(commBase, "MDWF_wilson_all_link_csw0_gauge_host");
    gaugeHost = baseGauge;
    SU3Accessor<double, R18> gaugeAcc = gaugeHost.getAccessor();
    HostGauge finestGaugePlusHost(
        commBase, "MDWF_wilson_all_link_csw0_finest_plus_host");
    finestGaugePlusHost = gaugePlus;
    SU3Accessor<double, R18> finestGaugePlusAcc
        = finestGaugePlusHost.getAccessor();
    size_t perturbedLinkCount = 0;
    double minPerturbedLinkDifference = std::numeric_limits<double>::infinity();
    double maxPerturbedLinkDifference = 0.0;

    for (size_t siteIndex = 0; siteIndex < lat.vol4; siteIndex++) {
        const gSite site = GInd::getSite(siteIndex);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const gSiteMu siteMu = GInd::getSiteMu(site, mu);
            const double linkDifference = static_cast<double>(infnorm(
                finestGaugePlusAcc.getLink(siteMu) - gaugeAcc.getLink(siteMu)));
            if (linkDifference > 1e-12) {
                perturbedLinkCount++;
            }
            minPerturbedLinkDifference = std::min(
                minPerturbedLinkDifference, linkDifference);
            maxPerturbedLinkDifference = std::max(
                maxPerturbedLinkDifference, linkDifference);
        }
    }

    std::vector<double> linkDerivatives(lat.vol4 * 4, 0.0);

    for (size_t term = 0; term < workspace.size(); term++) {
        HostSpinor chiHost(
            commBase, "MDWF_wilson_all_link_csw0_chi_" + std::to_string(term));
        HostSpinor etaHost(
            commBase, "MDWF_wilson_all_link_csw0_eta_" + std::to_string(term));
        chiHost = workspace.chi(term);
        etaHost = workspace.eta(term);
        Vect12ArrayAcc<double> chiAcc = chiHost.getAccessor();
        Vect12ArrayAcc<double> etaAcc = etaHost.getAccessor();

        for (size_t siteIndex = 0; siteIndex < lat.vol4; siteIndex++) {
            const gSite site = GInd::getSite(siteIndex);
            for (uint8_t mu = 0; mu < 4; mu++) {
                const gSiteMu siteMu = GInd::getSiteMu(site, mu);
                const SU3<double> direction
                    = mdwfAllLinkDeterministicDirection<double, HaloDepth>(siteMu);
                const double contraction
                    = mdwfWilsonAllLinkContractionTerm<HaloDepth, Ls>(
                        chiAcc, etaAcc, gaugeAcc, site, mu, direction);
                linkDerivatives[siteIndex * 4 + mu]
                    += -2.0 * forceCoefficients.numerator[term] * contraction;
            }
        }
    }

    HostGauge accumulatorHost(
        commBase, "MDWF_wilson_all_link_csw0_accumulator_host");
    accumulatorHost.template iterateOverFullAllMu<>(
        MDWFAllLinkZeroMatrix<double>());
    SU3Accessor<double, R18> accumulatorHostAcc = accumulatorHost.getAccessor();

    double analyticDerivative = 0.0;
    double sumAbsLinkDerivative = 0.0;
    double minDirectionNorm = std::numeric_limits<double>::infinity();
    double maxDirectionNorm = 0.0;
    double maxAntiHermitianViolation = 0.0;
    double maxTraceViolation = 0.0;

    for (size_t siteIndex = 0; siteIndex < lat.vol4; siteIndex++) {
        const gSite site = GInd::getSite(siteIndex);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const gSiteMu siteMu = GInd::getSiteMu(site, mu);
            const SU3<double> direction
                = mdwfAllLinkDeterministicDirection<double, HaloDepth>(siteMu);
            const double directionNorm = -real(tr_c(direction, direction));
            const double derivative = linkDerivatives[siteIndex * 4 + mu];
            const SU3<double> accumulatorMatrix
                = mdwfDirectionalActionDerivativeMatrix(derivative, direction);

            accumulatorHostAcc.setLink(siteMu, accumulatorMatrix);
            analyticDerivative += derivative;
            sumAbsLinkDerivative += std::abs(derivative);
            minDirectionNorm = std::min(minDirectionNorm, directionNorm);
            maxDirectionNorm = std::max(maxDirectionNorm, directionNorm);
            maxAntiHermitianViolation = std::max(
                maxAntiHermitianViolation,
                static_cast<double>(infnorm(direction + dagger(direction))));
            maxTraceViolation = std::max(
                maxTraceViolation,
                static_cast<double>(abs(tr_c(direction))));
        }
    }

    Gauge accumulator(commBase, "MDWF_wilson_all_link_csw0_accumulator");
    accumulator = accumulatorHost;
    accumulator.updateAll();
    HostGauge accumulatorRoundTrip(
        commBase, "MDWF_wilson_all_link_csw0_accumulator_round_trip");
    accumulatorRoundTrip = accumulator;
    SU3Accessor<double, R18> accumulatorAcc = accumulatorRoundTrip.getAccessor();

    double accumulatedDerivative = 0.0;
    double maxStoredMatrixDifference = 0.0;
    size_t invalidLinkCount = 0;
    size_t inspectedLinkCount = 0;

    for (size_t siteIndex = 0; siteIndex < lat.vol4; siteIndex++) {
        const gSite site = GInd::getSite(siteIndex);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const gSiteMu siteMu = GInd::getSiteMu(site, mu);
            const SU3<double> direction
                = mdwfAllLinkDeterministicDirection<double, HaloDepth>(siteMu);
            const SU3<double> storedMatrix = accumulatorAcc.getLink(siteMu);
            const SU3<double> expectedMatrix
                = mdwfDirectionalActionDerivativeMatrix(
                    linkDerivatives[siteIndex * 4 + mu], direction);
            const double recoveredDerivative = real(tr_c(direction, storedMatrix));
            const double storedNorm = static_cast<double>(infnorm(storedMatrix));

            inspectedLinkCount++;
            accumulatedDerivative += recoveredDerivative;
            maxStoredMatrixDifference = std::max(
                maxStoredMatrixDifference,
                static_cast<double>(infnorm(storedMatrix - expectedMatrix)));
            if (!std::isfinite(recoveredDerivative)
                || !std::isfinite(storedNorm)) {
                invalidLinkCount++;
            }
        }
    }

    const double accumulatorAbsDiff = std::abs(
        accumulatedDerivative - analyticDerivative);
    const MDWFFiniteDifferenceResult<double> &finest = finiteDifferences.back();
    const double finiteDifferenceAbsDiff = std::abs(
        finest.derivative - analyticDerivative);
    const double finiteDifferenceScale = std::max(1.0, std::abs(finest.derivative));
    const double finiteDifferenceRelDiff
        = finiteDifferenceAbsDiff / finiteDifferenceScale;
    const double epsilonStabilityAbs = std::abs(
        finiteDifferences[1].derivative - finiteDifferences[2].derivative);
    const double epsilonStabilityScale = std::max(
        1.0, std::abs(finiteDifferences[2].derivative));
    const double epsilonStabilityRel = epsilonStabilityAbs / epsilonStabilityScale;
    const double finiteDifferenceTolerance
        = 5e-3 + 5e-4 * finiteDifferenceScale;
    const double epsilonStabilityTolerance
        = 5e-3 + 5e-4 * epsilonStabilityScale;
    const double accumulatorTolerance
        = 1e-10 * std::max(1.0, std::abs(analyticDerivative));
    const size_t expectedLinkCount = 4 * lat.vol4;

    if (mdwfRationalCoefficientRoleName(actionInput.role) != "action"
        || mdwfRationalCoefficientRoleName(forceInput.role) != "force"
        || !finiteDifferencesConverged
        || !workspace.converged()
        || maxActionResidual > 1e-8
        || maxForceWorkspaceResidual > 1e-8
        || maxActionImagRelative > 1e-8
        || inspectedLinkCount != expectedLinkCount
        || perturbedLinkCount != expectedLinkCount
        || invalidLinkCount != 0
        || sumAbsLinkDerivative <= 1e-12
        || std::abs(minDirectionNorm - 1.0) > 1e-12
        || std::abs(maxDirectionNorm - 1.0) > 1e-12
        || maxAntiHermitianViolation > 1e-12
        || maxTraceViolation > 1e-12
        || maxStoredMatrixDifference > 1e-12
        || accumulatorAbsDiff > accumulatorTolerance
        || finiteDifferenceAbsDiff > finiteDifferenceTolerance
        || epsilonStabilityAbs > epsilonStabilityTolerance
        || !std::isfinite(analyticDerivative)
        || !std::isfinite(accumulatedDerivative)) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF c_sw = 0 Wilson all-link random-direction test failed: "
            "Ls = ", Ls,
            ", c_sw = ", csw,
            ", inspectedLinks = ", inspectedLinkCount,
            ", expectedLinks = ", expectedLinkCount,
            ", perturbedLinks = ", perturbedLinkCount,
            ", invalidLinks = ", invalidLinkCount,
            ", perturbedLinkDifferenceRange = [",
            minPerturbedLinkDifference, ", ", maxPerturbedLinkDifference, "]",
            ", minDirectionNorm = ", minDirectionNorm,
            ", maxDirectionNorm = ", maxDirectionNorm,
            ", maxAntiHermitianViolation = ", maxAntiHermitianViolation,
            ", maxTraceViolation = ", maxTraceViolation,
            ", analytic = ", analyticDerivative,
            ", accumulated = ", accumulatedDerivative,
            ", finiteDifference = ", finest.derivative,
            ", accumulatorAbsDiff = ", accumulatorAbsDiff,
            ", finiteDifferenceAbsDiff = ", finiteDifferenceAbsDiff,
            ", finiteDifferenceRelDiff = ", finiteDifferenceRelDiff,
            ", epsilonStabilityAbs = ", epsilonStabilityAbs,
            ", epsilonStabilityRel = ", epsilonStabilityRel,
            ", maxStoredMatrixDifference = ", maxStoredMatrixDifference,
            ", sumAbsLinkDerivative = ", sumAbsLinkDerivative,
            ", actionMaxResidual = ", maxActionResidual,
            ", forceMaxResidual = ", maxForceWorkspaceResidual,
            ", actionImagRel = ", maxActionImagRelative));
    }

    for (size_t epsilonIndex = 0; epsilonIndex < epsilons.size(); epsilonIndex++) {
        rootLogger.info(
            "MDWF c_sw = 0 Wilson all-link finite difference with epsilon = ",
            epsilons[epsilonIndex],
            ", derivative = ", finiteDifferences[epsilonIndex].derivative,
            ", maxResidual = ",
            finiteDifferences[epsilonIndex].max_shifted_residual,
            ", actionImagRel = ",
            finiteDifferences[epsilonIndex].action_imag_relative);
    }

    rootLogger.info(
        "MDWF c_sw = 0 Wilson all-link random-direction test passed with Ls = ",
        Ls,
        ", links = ", inspectedLinkCount,
        ", perturbedLinks = ", perturbedLinkCount,
        ", perturbedLinkDifferenceRange = [",
        minPerturbedLinkDifference, ", ", maxPerturbedLinkDifference, "]",
        ", analytic = ", analyticDerivative,
        ", accumulated = ", accumulatedDerivative,
        ", finiteDifference = ", finest.derivative,
        ", accumulatorAbsDiff = ", accumulatorAbsDiff,
        ", finiteDifferenceAbsDiff = ", finiteDifferenceAbsDiff,
        ", finiteDifferenceRelDiff = ", finiteDifferenceRelDiff,
        ", epsilonStabilityAbs = ", epsilonStabilityAbs,
        ", epsilonStabilityRel = ", epsilonStabilityRel,
        ", directionNormRange = [", minDirectionNorm, ", ", maxDirectionNorm, "]",
        ", maxStoredMatrixDifference = ", maxStoredMatrixDifference,
        ", actionMaxResidual = ", maxActionResidual,
        ", forceMaxResidual = ", maxForceWorkspaceResidual);
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

        runMDWFWilsonAllLinkRandomDirectionCsw0Test<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
