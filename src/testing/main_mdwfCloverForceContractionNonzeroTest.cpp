/*
 * MDWF nonzero-c_sw clover-path force-contraction smoke test.
 *
 * This compares the finite-difference MDWF rational-action derivative against
 * a test-local scalar contraction of
 *
 *     dM/depsilon = dM_Wilson/depsilon + dM_clover/depsilon
 *
 * for the current Wilson/clover MDWF path.  It does not accumulate gauge
 * force, update momenta, call RHMC/HMC, touch HISQ, use smearing, or implement
 * a production force kernel.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFAdjointOperator.h"
#include "../experimental/mdwf/MDWFAnalyticForceContractionCheck.h"
#include "../experimental/mdwf/MDWFFermionForceWorkspace.h"
#include "../experimental/mdwf/MDWFFiniteDifferenceHarness.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"
#include "../experimental/mdwf/MDWFRationalCoefficientAdapter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFCloverForceContractionNonzeroSource {
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
class MDWFCloverForceContractionNonzeroActionEvaluator {
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
    MDWFCloverForceContractionNonzeroActionEvaluator(
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
                                "MDWF_clover_force_contraction_nonzero_forward");
        AdjointOperator adjoint(gauge, _fifth_coeff, _mass, _csw,
                                "MDWF_clover_force_contraction_nonzero_adjoint");
        NormalOperator normal(_commBase, forward, adjoint,
                              "MDWF_clover_force_contraction_nonzero_normal");
        Adapter adapter(normal);
        Spinor actionWorkspace(_commBase, "MDWF_clover_force_contraction_nonzero_action_workspace");

        MDWFRationalActionResult<double> actionResult
            = computeMDWFRationalAction<double, Adapter>(
                adapter, actionWorkspace, _field, _coefficients, _max_iter, _precision,
                "MDWF_clover_force_contraction_nonzero_action");

        return makeMDWFFiniteDifferenceActionValue(actionResult);
    }
};

template<class floatT>
ColorVect<floatT> mdwfCloverForceWilsonGamma(uint8_t mu, const ColorVect<floatT> &spinor) {
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
COMPLEX(double) mdwfCloverForceColorVectDot(const ColorVect<floatT> &left,
                                            const ColorVect<floatT> &right) {
    COMPLEX(double) result(0.0, 0.0);
    for (size_t spin = 0; spin < 4; spin++) {
        result += left[spin] * right[spin];
    }
    return result;
}

template<class floatT>
SU3<floatT> mdwfCloverForceLinkDerivative(
    const SU3<floatT> &link,
    const MDWFFiniteDifferenceProbe<floatT> &probe) {

    const SU3<floatT> generator = mdwfFiniteDifferenceGenerator<floatT>(probe.generator_id);
    if (probe.multiplication_side == MDWFFiniteDifferenceMultiplicationSide::Right) {
        return link * generator;
    }
    return generator * link;
}

template<class floatT>
SU3<floatT> mdwfCloverForceLinkDaggerDerivative(
    const SU3<floatT> &link,
    const MDWFFiniteDifferenceProbe<floatT> &probe) {

    const SU3<floatT> linkDagger = dagger(link);
    const SU3<floatT> generator = mdwfFiniteDifferenceGenerator<floatT>(probe.generator_id);
    if (probe.multiplication_side == MDWFFiniteDifferenceMultiplicationSide::Right) {
        return static_cast<floatT>(-1.0) * generator * linkDagger;
    }
    return static_cast<floatT>(-1.0) * linkDagger * generator;
}

template<class floatT>
bool mdwfCloverForceSameLink(const gSiteMu &siteMu,
                             const MDWFFiniteDifferenceProbe<floatT> &probe) {
    return siteMu.coord.x == probe.x
           && siteMu.coord.y == probe.y
           && siteMu.coord.z == probe.z
           && siteMu.coord.t == probe.t
           && siteMu.mu == probe.mu;
}

template<class floatT>
struct MDWFCloverForcePathFactor {
    gSiteMu site_mu;
    bool dagger_link;
};

template<class floatT>
SU3<floatT> mdwfCloverForcePathFactorValue(
    SU3Accessor<floatT, R18> gaugeAcc,
    const MDWFCloverForcePathFactor<floatT> &factor) {

    if (factor.dagger_link) {
        return gaugeAcc.getLinkDagger(factor.site_mu);
    }
    return gaugeAcc.getLink(factor.site_mu);
}

template<class floatT>
SU3<floatT> mdwfCloverForcePathFactorDerivative(
    SU3Accessor<floatT, R18> gaugeAcc,
    const MDWFCloverForcePathFactor<floatT> &factor,
    const MDWFFiniteDifferenceProbe<floatT> &probe) {

    const SU3<floatT> link = gaugeAcc.getLink(factor.site_mu);
    if (factor.dagger_link) {
        return mdwfCloverForceLinkDaggerDerivative(link, probe);
    }
    return mdwfCloverForceLinkDerivative(link, probe);
}

template<class floatT>
SU3<floatT> mdwfCloverForcePathDerivative(
    SU3Accessor<floatT, R18> gaugeAcc,
    const std::array<MDWFCloverForcePathFactor<floatT>, 4> &path,
    const MDWFFiniteDifferenceProbe<floatT> &probe) {

    SU3<floatT> derivative = su3_zero<floatT>();
    for (size_t active = 0; active < path.size(); active++) {
        if (!mdwfCloverForceSameLink(path[active].site_mu, probe)) {
            continue;
        }

        SU3<floatT> term = su3_one<floatT>();
        for (size_t factor = 0; factor < path.size(); factor++) {
            if (factor == active) {
                term *= mdwfCloverForcePathFactorDerivative(gaugeAcc, path[factor], probe);
            } else {
                term *= mdwfCloverForcePathFactorValue(gaugeAcc, path[factor]);
            }
        }
        derivative += term;
    }

    return derivative;
}

template<size_t HaloDepth>
SU3<double> mdwfCloverForcePlaqCloverDerivative(
    SU3Accessor<double, R18> gaugeAcc,
    gSite site,
    int mu,
    int nu,
    const MDWFFiniteDifferenceProbe<double> &probe) {

    typedef GIndexer<All, HaloDepth> GInd;

    const std::array<MDWFCloverForcePathFactor<double>, 4> pathP = {{
        {GInd::getSiteMu(site, mu), false},
        {GInd::getSiteMu(GInd::site_up(site, mu), nu), false},
        {GInd::getSiteMu(GInd::site_up(site, nu), mu), true},
        {GInd::getSiteMu(site, nu), true}
    }};
    const std::array<MDWFCloverForcePathFactor<double>, 4> pathQ = {{
        {GInd::getSiteMu(GInd::site_dn(site, nu), nu), true},
        {GInd::getSiteMu(GInd::site_dn(site, nu), mu), false},
        {GInd::getSiteMu(GInd::site_up_dn(site, mu, nu), nu), false},
        {GInd::getSiteMu(site, mu), true}
    }};
    const std::array<MDWFCloverForcePathFactor<double>, 4> pathR = {{
        {GInd::getSiteMu(GInd::site_dn(site, mu), mu), true},
        {GInd::getSiteMu(GInd::site_dn_dn(site, mu, nu), nu), true},
        {GInd::getSiteMu(GInd::site_dn_dn(site, mu, nu), mu), false},
        {GInd::getSiteMu(GInd::site_dn(site, nu), nu), false}
    }};
    const std::array<MDWFCloverForcePathFactor<double>, 4> pathS = {{
        {GInd::getSiteMu(site, nu), false},
        {GInd::getSiteMu(GInd::site_up_dn(site, nu, mu), mu), true},
        {GInd::getSiteMu(GInd::site_dn(site, mu), nu), true},
        {GInd::getSiteMu(GInd::site_dn(site, mu), mu), false}
    }};

    return mdwfCloverForcePathDerivative(gaugeAcc, pathP, probe)
           + mdwfCloverForcePathDerivative(gaugeAcc, pathQ, probe)
           + mdwfCloverForcePathDerivative(gaugeAcc, pathR, probe)
           + mdwfCloverForcePathDerivative(gaugeAcc, pathS, probe);
}

template<size_t HaloDepth>
SU3<double> mdwfCloverForceFieldStrengthDerivative(
    SU3Accessor<double, R18> gaugeAcc,
    gSite site,
    int mu,
    int nu,
    const MDWFFiniteDifferenceProbe<double> &probe) {

    const SU3<double> dQ = mdwfCloverForcePlaqCloverDerivative<HaloDepth>(
        gaugeAcc, site, mu, nu, probe);
    SU3<double> dF = (COMPLEX(double)(0.0, -1.0) / 8.0) * (dQ - dagger(dQ));
    const SU3<double> unity = su3_one<double>();
    dF = dF - (1.0 / 3.0) * tr_c(dF) * unity;
    return dF;
}

void mdwfCloverForceAddFmunuDerivativeToBlocks(
    const SU3<double> &dFmunu,
    int mu,
    int nu,
    Matrix6x6<double> &upper,
    Matrix6x6<double> &lower) {

    const COMPLEX(double) ii(0.0, 1.0);

    for (int colorRow = 0; colorRow < 3; colorRow++) {
        for (int colorColumn = 0; colorColumn < 3; colorColumn++) {
            const COMPLEX(double) value = dFmunu(colorRow, colorColumn);

            if (mu == 0 && nu == 1) {
                upper.val[colorRow][colorColumn] += value;
                upper.val[colorRow + 3][colorColumn + 3] -= value;
                lower.val[colorRow][colorColumn] += value;
                lower.val[colorRow + 3][colorColumn + 3] -= value;
            } else if (mu == 0 && nu == 2) {
                upper.val[colorRow][colorColumn + 3] += -ii * value;
                upper.val[colorRow + 3][colorColumn] += ii * value;
                lower.val[colorRow][colorColumn + 3] += -ii * value;
                lower.val[colorRow + 3][colorColumn] += ii * value;
            } else if (mu == 0 && nu == 3) {
                upper.val[colorRow][colorColumn + 3] += -value;
                upper.val[colorRow + 3][colorColumn] += -value;
                lower.val[colorRow][colorColumn + 3] += value;
                lower.val[colorRow + 3][colorColumn] += value;
            } else if (mu == 1 && nu == 2) {
                upper.val[colorRow][colorColumn + 3] += value;
                upper.val[colorRow + 3][colorColumn] += value;
                lower.val[colorRow][colorColumn + 3] += value;
                lower.val[colorRow + 3][colorColumn] += value;
            } else if (mu == 1 && nu == 3) {
                upper.val[colorRow][colorColumn + 3] += -ii * value;
                upper.val[colorRow + 3][colorColumn] += ii * value;
                lower.val[colorRow][colorColumn + 3] += ii * value;
                lower.val[colorRow + 3][colorColumn] += -ii * value;
            } else if (mu == 2 && nu == 3) {
                upper.val[colorRow][colorColumn] += -value;
                upper.val[colorRow + 3][colorColumn + 3] += value;
                lower.val[colorRow][colorColumn] += value;
                lower.val[colorRow + 3][colorColumn + 3] += -value;
            }
        }
    }
}

void mdwfCloverForceScaleMatrix(Matrix6x6<double> &matrix, double factor) {
    for (int row = 0; row < 6; row++) {
        for (int column = 0; column < 6; column++) {
            matrix.val[row][column] *= factor;
        }
    }
}

Vect12<double> mdwfCloverForceApplyCloverDerivative(
    Matrix6x6<double> &upper,
    Matrix6x6<double> &lower,
    const Vect12<double> &spinor) {

    Vect18<double> upperStored = upper.ConvertHermitianToVect18();
    Vect18<double> lowerStored = lower.ConvertHermitianToVect18();
    Matrix6x6<double> storedUpper(upperStored);
    Matrix6x6<double> storedLower(lowerStored);

    Vect12<double> out = storedUpper.MatrixXVect12UpDown(spinor, 0);
    out = storedLower.MatrixXVect12UpDown(out, 1);
    return out;
}

template<class DeviceSpinor, size_t Ls>
double mdwfCloverForceWilsonContractionTerm(
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
    Gaugefield<double, false, HaloDepth, R18> gaugeHost(commBase, name + "_gauge_host");
    chiHost = chiDevice;
    etaHost = etaDevice;
    gaugeHost = gauge;

    Vect12ArrayAcc<double> chiAcc = chiHost.getAccessor();
    Vect12ArrayAcc<double> etaAcc = etaHost.getAccessor();
    SU3Accessor<double, R18> gaugeAcc = gaugeHost.getAccessor();

    const gSiteMu siteMu = GInd::getSiteMu(probe.x, probe.y, probe.z, probe.t, probe.mu);
    const SU3<double> link = gaugeAcc.getLink(siteMu);
    const SU3<double> dLink = mdwfCloverForceLinkDerivative(link, probe);
    const SU3<double> dLinkDagger = mdwfCloverForceLinkDaggerDerivative(link, probe);

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

        const ColorVect<double> dMForward = 0.5
                                            * (mdwfCloverForceWilsonGamma(probe.mu, forwardHop)
                                               - forwardHop);
        const ColorVect<double> dMBackward = -0.5
                                             * (backwardHop
                                                + mdwfCloverForceWilsonGamma(probe.mu, backwardHop));

        contraction += mdwfCloverForceColorVectDot(etaForward, dMForward);
        contraction += mdwfCloverForceColorVectDot(etaBackward, dMBackward);
    }

    return real(contraction);
}

template<class DeviceSpinor, size_t Ls>
double mdwfCloverForceCloverContractionTerm(
    const DeviceSpinor &chiDevice,
    const DeviceSpinor &etaDevice,
    Gaugefield<double, true, 2, R18> &gauge,
    const MDWFFiniteDifferenceProbe<double> &probe,
    double csw,
    CommunicationBase &commBase,
    const std::string &name) {

    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;

    MDWFSpinor<double, false, All, HaloDepth, Ls> chiHost(commBase, name + "_chi_host");
    MDWFSpinor<double, false, All, HaloDepth, Ls> etaHost(commBase, name + "_eta_host");
    Gaugefield<double, false, HaloDepth, R18> gaugeHost(commBase, name + "_gauge_host");
    chiHost = chiDevice;
    etaHost = etaDevice;
    gaugeHost = gauge;

    Vect12ArrayAcc<double> chiAcc = chiHost.getAccessor();
    Vect12ArrayAcc<double> etaAcc = etaHost.getAccessor();
    SU3Accessor<double, R18> gaugeAcc = gaugeHost.getAccessor();

    COMPLEX(double) contraction(0.0, 0.0);
    const size_t volume = GInd::getLatData().vol4;

    for (size_t siteIndex = 0; siteIndex < volume; siteIndex++) {
        const gSite site = GInd::getSite(siteIndex);
        Matrix6x6<double> upper;
        Matrix6x6<double> lower;

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = mu + 1; nu < 4; nu++) {
                const SU3<double> dFmunu = mdwfCloverForceFieldStrengthDerivative<HaloDepth>(
                    gaugeAcc, site, mu, nu, probe);
                mdwfCloverForceAddFmunuDerivativeToBlocks(dFmunu, mu, nu, upper, lower);
            }
        }

        mdwfCloverForceScaleMatrix(upper, -0.5 * csw);
        mdwfCloverForceScaleMatrix(lower, -0.5 * csw);

        for (size_t stack = 0; stack < Ls; stack++) {
            const gSiteStack siteStack = GInd::getSiteStack(site, stack);
            const Vect12<double> dCloverChi = mdwfCloverForceApplyCloverDerivative(
                upper, lower, chiAcc.getElement(siteStack));
            contraction += etaAcc.getElement(siteStack) * dCloverChi;
        }
    }

    return real(contraction);
}

template<class Workspace, class Spinor, size_t Ls>
void mdwfCloverForceContractionAnalyticDerivative(
    Workspace &workspace,
    Gaugefield<double, true, 2, R18> &gauge,
    const MDWFRationalCoefficients<double> &forceCoefficients,
    const MDWFFiniteDifferenceProbe<double> &probe,
    double csw,
    CommunicationBase &commBase,
    double &wilsonDerivative,
    double &cloverDerivative) {

    wilsonDerivative = 0.0;
    cloverDerivative = 0.0;

    for (size_t term = 0; term < workspace.size(); term++) {
        const double wilsonContraction = mdwfCloverForceWilsonContractionTerm<Spinor, Ls>(
            workspace.chi(term), workspace.eta(term), gauge, probe, commBase,
            "MDWF_clover_force_wilson_term_" + std::to_string(term));
        const double cloverContraction = mdwfCloverForceCloverContractionTerm<Spinor, Ls>(
            workspace.chi(term), workspace.eta(term), gauge, probe, csw, commBase,
            "MDWF_clover_force_clover_term_" + std::to_string(term));

        wilsonDerivative += -2.0 * forceCoefficients.numerator[term] * wilsonContraction;
        cloverDerivative += -2.0 * forceCoefficients.numerator[term] * cloverContraction;
    }
}

template<size_t Ls>
void runMDWFCloverForceContractionNonzeroTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    const double mass = 4.0;
    const double csw = 0.5;

    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using ForwardOperator = MDWFLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using AdjointOperator = MDWFAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using NormalOperator = MDWFNormalOperator<ForwardOperator, AdjointOperator>;
    using Workspace = MDWFFermionForceWorkspace<double, HaloDepth, HaloDepth, Ls,
                                                NormalOperator, ForwardOperator>;
    using Spinor = typename NormalOperator::Spinor;

    MDWFExplicitRationalInput<double> actionInput{
        "clover_force_contraction_nonzero_action_coefficients",
        MDWFRationalCoefficientRole::Action,
        0.125,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    MDWFExplicitRationalInput<double> forceInput{
        "clover_force_contraction_nonzero_force_coefficients",
        MDWFRationalCoefficientRole::Force,
        0.0,
        {0.5, 0.25, 0.125},
        {0.0, 0.1, 0.3}
    };
    MDWFRationalCoefficients<double> actionCoefficients = makeMDWFRationalCoefficients(actionInput);
    MDWFRationalCoefficients<double> forceCoefficients = makeMDWFRationalCoefficients(forceInput);

    Gauge baseGauge(commBase, "MDWF_clover_force_contraction_nonzero_base_gauge");
    Gauge gaugePlus(commBase, "MDWF_clover_force_contraction_nonzero_gauge_plus");
    Gauge gaugeMinus(commBase, "MDWF_clover_force_contraction_nonzero_gauge_minus");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260514);
    d_rand = h_rand;
    baseGauge.random(d_rand.state);
    baseGauge.updateAll();

    Spinor field(commBase, "MDWF_clover_force_contraction_nonzero_field");
    field.template iterateOverBulk<>(
        FillMDWFCloverForceContractionNonzeroSource<double, All, HaloDepth, Ls>());
    field.updateAll();
    Spinor forceField(commBase, "MDWF_clover_force_contraction_nonzero_force_field");
    forceField = field;
    forceField.updateAll();

    MDWFFifthDimCoefficients<double> fifthCoeff(1.0, -0.05, -0.05, 0.0, 0.0);
    MDWFFiniteDifferenceProbe<double> probe{
        2, 2, 2, 2,
        1,
        0,
        1e-4,
        MDWFFiniteDifferenceMultiplicationSide::Left
    };

    MDWFCloverForceContractionNonzeroActionEvaluator<HaloDepth, Ls> actionEvaluator(
        commBase, field, actionCoefficients, fifthCoeff, mass, csw, 512, 1e-8);
    MDWFFiniteDifferenceResult<double> finiteDifference = evaluateMDWFFiniteDifferenceAction(
        gaugePlus, gaugeMinus, baseGauge, probe, actionEvaluator);

    ForwardOperator forward(baseGauge, fifthCoeff, mass, csw,
                            "MDWF_clover_force_contraction_nonzero_forward");
    AdjointOperator adjoint(baseGauge, fifthCoeff, mass, csw,
                            "MDWF_clover_force_contraction_nonzero_adjoint");
    NormalOperator normal(commBase, forward, adjoint,
                          "MDWF_clover_force_contraction_nonzero_normal");
    Workspace workspace;
    workspace.prepare(normal, forward, forceField, forceCoefficients, 512, 1e-8,
                      "MDWF_clover_force_contraction_nonzero_workspace");

    double maxForceWorkspaceResidue = 0.0;
    for (const auto &info : workspace.shiftInfo()) {
        maxForceWorkspaceResidue = std::max(maxForceWorkspaceResidue, info.residue);
    }

    double wilsonDerivative = 0.0;
    double cloverDerivative = 0.0;
    mdwfCloverForceContractionAnalyticDerivative<Workspace, Spinor, Ls>(
        workspace, baseGauge, forceCoefficients, probe, csw, commBase,
        wilsonDerivative, cloverDerivative);

    const double analyticDerivative = wilsonDerivative + cloverDerivative;
    MDWFAnalyticForceContractionResult<double> comparison
        = compareMDWFAnalyticForceContraction(finiteDifference, analyticDerivative, 5e-3, 5e-4);

    if (mdwfRationalCoefficientRoleName(actionInput.role) != "action"
        || mdwfRationalCoefficientRoleName(forceInput.role) != "force"
        || !finiteDifference.converged
        || !workspace.converged()
        || finiteDifference.max_shifted_residual > 1e-8
        || maxForceWorkspaceResidue > 1e-8
        || finiteDifference.action_imag_relative > 1e-8
        || !std::isfinite(wilsonDerivative)
        || !std::isfinite(cloverDerivative)
        || !comparison.passed) {
        const char *side = probe.multiplication_side
                           == MDWFFiniteDifferenceMultiplicationSide::Left ? "left" : "right";
        throw std::runtime_error(stdLogger.fatal(
            "MDWF nonzero-c_sw clover force-contraction test failed: c_sw = ", csw,
            ", mu = ", static_cast<int>(probe.mu),
            ", generator = ", probe.generator_id,
            ", side = ", side,
            ", finiteDifference = ", comparison.finite_difference_derivative,
            ", wilsonAnalytic = ", wilsonDerivative,
            ", cloverAnalytic = ", cloverDerivative,
            ", totalAnalytic = ", comparison.analytic_derivative,
            ", absDiff = ", comparison.absolute_difference,
            ", relDiff = ", comparison.relative_difference,
            ", finiteDifferenceConverged = ", finiteDifference.converged,
            ", workspaceConverged = ", workspace.converged(),
            ", actionMaxResidual = ", finiteDifference.max_shifted_residual,
            ", forceMaxResidual = ", maxForceWorkspaceResidue,
            ", actionImagRel = ", finiteDifference.action_imag_relative));
    }

    rootLogger.info("MDWF nonzero-c_sw clover force-contraction test passed with Ls = ", Ls,
                    ", c_sw = ", csw,
                    ", finiteDifference = ", comparison.finite_difference_derivative,
                    ", wilsonAnalytic = ", wilsonDerivative,
                    ", cloverAnalytic = ", cloverDerivative,
                    ", totalAnalytic = ", comparison.analytic_derivative,
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

        runMDWFCloverForceContractionNonzeroTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
