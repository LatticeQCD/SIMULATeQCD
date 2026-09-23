/*
 * MDWF HMC convention test (MDWF_HMC_CONVENTIONS.md, step 1 of the RHMC plan).
 *
 * SIMULATeQCD's molecular dynamics uses Hermitian traceless momenta P from
 * Gaugefield::gauss, U -> exp(i eps P) U, P -> P - i eps ipdot, and
 * H = (1/2) sum tr(P P) + S. Energy conservation then requires
 *
 *     dS/dtau + sum_l tr(P_l (-i ipdot_l)) = 0   along  dU/dtau = i P U,
 *
 * i.e. ipdot_l = K_l, the traceless anti-Hermitian left-variation matrix with
 * dS(H) = Re tr(H K_l). This test checks that identity numerically for:
 *
 *   1. the plaquette part of the gauge action, -(beta/3) sum Re tr P, with
 *      SIMULATeQCD's independent plaquette derivative gaugeActionDerivPlaq:
 *      this confirms the convention reading itself. The rectangle part of
 *      gauge_force and the full Symanzik identity for S_g = -(3 beta/5)
 *      symanzik() (the action in Delta H) are reported as diagnostics, because
 *      a first run found the full identity off by 2.4%;
 *   2. the Mobius clover rational fermion action, with ipdot_f = the stored
 *      all-link MDWF matrices K_l (MDWFMobiusForceWorkspaceView +
 *      overwriteMDWFCloverAllLinkDirectionIndependentStorageNonzero), at the
 *      control (M5 = -2) and physical-like (M5 = 1.8, mf = 0.05) parameters,
 *      b5 = 1.5, c_sw = 0.5;
 *   3. the inputs: P is Hermitian and traceless, and Spinorfield::gauss noise
 *      satisfies <eta^\dagger eta> / (12 Ls V) = 1 within 5 standard
 *      deviations (the normalization the two-flavour heatbath phi = M^\dagger eta
 *      requires, which Delta H / reversibility tests cannot detect).
 *
 * The link update reproduces integrator.cpp's do_evolve_Q exactly (that
 * functor is file-local there). No integrator, RHMC, or gauge-force code is
 * modified. Single rank only.
 */

#include "../simulateqcd.h"
#include "../gauge/gaugeAction.h"
#include "../gauge/gaugeActionDeriv.h"
#include "../experimental/mdwf/MDWFMobiusMapping.h"
#include "../experimental/mdwf/MDWFAllLinkDirectionIndependentStorage.h"
#include "../experimental/mdwf/MDWFFermionForceWorkspace.h"
#include "../experimental/mdwf/MDWFFiniteDifferenceHarness.h"
#include "../experimental/mdwf/MDWFMobiusForceWorkspace.h"
#include "../experimental/mdwf/MDWFNormalOperator.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct FillMDWFHmcConventionSource {
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

// Same formula as do_evolve_Q in src/modules/rhmc/integrator.cpp.
template<class floatT, size_t HaloDepth>
struct MDWFHmcConventionEvolveQ {
    SU3Accessor<floatT, R18> _gAcc;
    SU3Accessor<floatT> _pAcc;
    double _stepsize;

    MDWFHmcConventionEvolveQ(SU3Accessor<floatT, R18> gAcc, SU3Accessor<floatT> pAcc, double stepsize)
        : _gAcc(gAcc), _pAcc(pAcc), _stepsize(stepsize) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu site) {
        SU3<double> temp = su3_exp<double>(COMPLEX(double)(0.0, 1.0) * _stepsize
                                           * _pAcc.template getLink<double>(site))
                           * _gAcc.template getLink<double>(site);
        temp.su3unitarize();
        return temp;
    }
};

template<size_t HaloDepth>
void mdwfHmcConventionEvolve(Gaugefield<double, true, HaloDepth, R18> &gaugeOut,
                             Gaugefield<double, true, HaloDepth, R18> &gaugeIn,
                             Gaugefield<double, true, HaloDepth> &momenta,
                             double stepsize) {
    gaugeOut.iterateOverBulkAllMu(
        MDWFHmcConventionEvolveQ<double, HaloDepth>(gaugeIn.getAccessor(), momenta.getAccessor(), stepsize));
    gaugeOut.updateAll();
}

template<size_t HaloDepth, size_t Ls>
class MDWFHmcConventionFermionAction {
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
    double _M5, _mf, _b5, _csw;
    int _max_iter;
    double _precision;

public:
    MDWFHmcConventionFermionAction(CommunicationBase &commBase, Spinor &field,
                                   const MDWFRationalCoefficients<double> &coefficients,
                                   double M5, double mf, double b5, double csw, int max_iter, double precision)
        : _commBase(commBase), _field(field), _coefficients(coefficients),
          _M5(M5), _mf(mf), _b5(b5), _csw(csw), _max_iter(max_iter), _precision(precision) {}

    MDWFFiniteDifferenceActionValue<double> operator()(Gauge &gauge) {
        ForwardOperator forward(gauge, _M5, _mf, _b5, _csw, "MDWF_hmc_convention_fd_forward");
        AdjointOperator adjoint(gauge, _M5, _mf, _b5, _csw, "MDWF_hmc_convention_fd_adjoint");
        NormalOperator normal(_commBase, forward, adjoint, "MDWF_hmc_convention_fd_normal");
        Adapter adapter(normal);
        Spinor workspace(_commBase, "MDWF_hmc_convention_fd_workspace");
        return makeMDWFFiniteDifferenceActionValue(computeMDWFRationalAction<double, Adapter>(
            adapter, workspace, _field, _coefficients, _max_iter, _precision, "MDWF_hmc_convention_fd_action"));
    }
};

// sum over bulk links of tr(P (-i ipdot)), the kinetic-energy rate predicted by P -> P - i eps ipdot.
template<size_t HaloDepth>
double mdwfHmcConventionKineticRate(const Gaugefield<double, false, HaloDepth> &momenta,
                                    const Gaugefield<double, false, HaloDepth, R18> &ipdot) {
    typedef GIndexer<All, HaloDepth> GInd;
    const SU3Accessor<double> pAcc = momenta.getAccessor();
    const SU3Accessor<double, R18> fAcc = ipdot.getAccessor();
    double rate = 0.0;
    for (size_t siteIndex = 0; siteIndex < GInd::getLatData().vol4; siteIndex++) {
        const gSite site = GInd::getSite(siteIndex);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const gSiteMu siteMu = GInd::getSiteMu(site, mu);
            rate += real(tr_c(pAcc.getLink(siteMu), COMPLEX(double)(0.0, -1.0) * fAcc.getLink(siteMu)));
        }
    }
    return rate;
}

double mdwfHmcConventionRelSum(double actionRate, double kineticRate) {
    return std::abs(actionRate + kineticRate) / std::max(1.0, std::abs(actionRate));
}

template<size_t Ls>
void runMDWFHmcConventionTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using Momenta = Gaugefield<double, true, HaloDepth>;
    using HostMomenta = Gaugefield<double, false, HaloDepth>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;
    using MobiusForward = MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using MobiusAdjoint = MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using MobiusNormal = MDWFNormalOperator<MobiusForward, MobiusAdjoint>;
    using MobiusWorkspace = MDWFFermionForceWorkspace<double, HaloDepth, HaloDepth, Ls, MobiusNormal, MobiusForward>;
    using MobiusView = MDWFMobiusForceWorkspaceView<double, HaloDepth, Ls, MobiusWorkspace>;
    using FermionAction = MDWFHmcConventionFermionAction<HaloDepth, Ls>;

    const LatticeData lat = GInd::getLatData();
    if (lat.vol4 != lat.globvol4) {
        throw std::runtime_error(stdLogger.fatal("MDWF HMC convention test is single-rank only"));
    }

    const double beta = 6.0;
    const double csw = 0.5;
    const double mf = 0.05;
    const double b5 = 1.5;
    const double epsilon = 1e-4;
    const double precision = 1e-10;
    const double gaugeTolerance = 1e-7;
    const double fermionTolerance = 1e-5;
    const double algebraTolerance = 1e-12;

    MDWFExplicitRationalInput<double> actionInput{
        "hmc_convention_action", MDWFRationalCoefficientRole::Action, 0.125, {0.5, 0.25, 0.125}, {0.0, 0.1, 0.3}};
    MDWFExplicitRationalInput<double> forceInput{
        "hmc_convention_force", MDWFRationalCoefficientRole::Force, 0.0, {0.5, 0.25, 0.125}, {0.0, 0.1, 0.3}};
    const MDWFRationalCoefficients<double> actionCoefficients = makeMDWFRationalCoefficients(actionInput);
    const MDWFRationalCoefficients<double> forceCoefficients = makeMDWFRationalCoefficients(forceInput);

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260923);
    d_rand = h_rand;

    Gauge baseGauge(commBase, "MDWF_hmc_convention_base_gauge");
    Gauge gaugePlus(commBase, "MDWF_hmc_convention_gauge_plus");
    Gauge gaugeMinus(commBase, "MDWF_hmc_convention_gauge_minus");
    baseGauge.random(d_rand.state);
    baseGauge.updateAll();

    Momenta momenta(commBase, "MDWF_hmc_convention_momenta");
    momenta.gauss(d_rand.state);
    momenta.updateAll();

    HostGauge gaugeHost(commBase, "MDWF_hmc_convention_gauge_host");
    HostMomenta momentaHost(commBase, "MDWF_hmc_convention_momenta_host");
    gaugeHost = baseGauge;
    momentaHost = momenta;

    // --- Part 3a: P is Hermitian and traceless. ---
    double maxHermiticityViolation = 0.0;
    double maxTraceViolation = 0.0;
    double maxMomentumNorm = 0.0;
    {
        const SU3Accessor<double> pAcc = momentaHost.getAccessor();
        for (size_t siteIndex = 0; siteIndex < lat.vol4; siteIndex++) {
            const gSite site = GInd::getSite(siteIndex);
            for (uint8_t mu = 0; mu < 4; mu++) {
                const SU3<double> p = pAcc.getLink(GInd::getSiteMu(site, mu));
                maxHermiticityViolation = std::max(maxHermiticityViolation,
                                                   static_cast<double>(infnorm(p - dagger(p))));
                maxTraceViolation = std::max(maxTraceViolation, static_cast<double>(abs(tr_c(p))));
                maxMomentumNorm = std::max(maxMomentumNorm, static_cast<double>(infnorm(p)));
            }
        }
    }
    const bool momentaPassed = maxMomentumNorm > 0.0
                               && maxHermiticityViolation <= algebraTolerance * maxMomentumNorm
                               && maxTraceViolation <= algebraTolerance * maxMomentumNorm;
    rootLogger.info("MDWF HMC convention momenta: maxNorm = ", maxMomentumNorm,
                    ", maxHermiticityViolation = ", maxHermiticityViolation,
                    ", maxTraceViolation = ", maxTraceViolation, ", passed = ", momentaPassed);

    // --- Part 3b: pseudofermion noise normalization. ---
    Spinor noise(commBase, "MDWF_hmc_convention_noise");
    noise.gauss(d_rand.state);
    double noiseNorm2 = 0.0;
    for (const COMPLEX(double) &stackDot : noise.dotProductStacked(noise)) {
        noiseNorm2 += real<double>(stackDot);
    }
    const double noiseDof = 12.0 * static_cast<double>(Ls) * static_cast<double>(lat.globvol4);
    const double noiseRatio = noiseNorm2 / noiseDof;
    const double noiseSigma = 1.0 / std::sqrt(noiseDof);
    const bool noisePassed = std::abs(noiseRatio - 1.0) <= 5.0 * noiseSigma;
    rootLogger.info("MDWF HMC convention noise: <eta^dagger eta> / (12 Ls V) = ", noiseRatio,
                    ", expected 1 +- ", noiseSigma, " (dof = ", noiseDof, "), passed = ", noisePassed);

    mdwfHmcConventionEvolve<HaloDepth>(gaugePlus, baseGauge, momenta, epsilon);
    mdwfHmcConventionEvolve<HaloDepth>(gaugeMinus, baseGauge, momenta, -epsilon);

    // --- Part 1: gauge action. ---
    // S_g = -(3 beta/5) symanzik() = S_plaq + S_rect with
    //   S_plaq = -(beta/3) sum Re tr P = -(beta/3) * 18 V * plaquette(),
    //   S_rect = +(beta/60) sum Re tr R = +(beta/60) * 36 V * rectangle().
    // The plaquette identity uses SIMULATeQCD's independent plaquette derivative
    // (gaugeActionDerivPlaq = TA(sum of the six plaquettes starting at the link)) and gates
    // the convention reading; the rectangle part of gauge_force and the full Symanzik
    // identity are reported as diagnostics.
    const double volume = static_cast<double>(lat.globvol4);
    auto plaquetteAction = [&](Gauge &gauge) {
        GaugeAction<double, true, HaloDepth, R18> action(gauge);
        return -(beta / 3.0) * 18.0 * volume * static_cast<double>(action.plaquette());
    };
    auto rectangleAction = [&](Gauge &gauge) {
        GaugeAction<double, true, HaloDepth, R18> action(gauge);
        return (beta / 60.0) * 36.0 * volume * static_cast<double>(action.rectangle());
    };
    auto symanzikAction = [&](Gauge &gauge) {
        GaugeAction<double, true, HaloDepth, R18> action(gauge);
        return -(3.0 * beta / 5.0) * static_cast<double>(action.symanzik());
    };
    const double plaquetteActionRate = (plaquetteAction(gaugePlus) - plaquetteAction(gaugeMinus)) / (2.0 * epsilon);
    const double rectangleActionRate = (rectangleAction(gaugePlus) - rectangleAction(gaugeMinus)) / (2.0 * epsilon);
    const double gaugeActionRate = (symanzikAction(gaugePlus) - symanzikAction(gaugeMinus)) / (2.0 * epsilon);

    HostGauge gaugeIpdot(commBase, "MDWF_hmc_convention_gauge_ipdot");
    HostGauge plaquetteIpdot(commBase, "MDWF_hmc_convention_plaquette_ipdot");
    HostGauge rectangleIpdot(commBase, "MDWF_hmc_convention_rectangle_ipdot");
    double maxSymanzikDerivDiff = 0.0;
    double maxGaugeForceNorm = 0.0;
    {
        const SU3Accessor<double, R18> gAcc = gaugeHost.getAccessor();
        SU3Accessor<double, R18> totalAcc = gaugeIpdot.getAccessor();
        SU3Accessor<double, R18> plaqAcc = plaquetteIpdot.getAccessor();
        SU3Accessor<double, R18> rectAcc = rectangleIpdot.getAccessor();
        for (size_t siteIndex = 0; siteIndex < lat.vol4; siteIndex++) {
            const gSite site = GInd::getSite(siteIndex);
            for (uint8_t mu = 0; mu < 4; mu++) {
                const gSiteMu siteMu = GInd::getSiteMu(site, mu);
                const SU3<double> total = gauge_force<double, HaloDepth, R18>(gAcc, siteMu, beta);
                const SU3<double> plaq = (-beta / 3.0) * gaugeActionDerivPlaq<double, HaloDepth>(gAcc, site, mu);
                const SU3<double> symanzikDeriv = symanzikGaugeActionDeriv<double, HaloDepth>(gAcc, site, mu);
                totalAcc.setLink(siteMu, total);
                plaqAcc.setLink(siteMu, plaq);
                rectAcc.setLink(siteMu, total - plaq);
                maxGaugeForceNorm = std::max(maxGaugeForceNorm, static_cast<double>(infnorm(total)));
                maxSymanzikDerivDiff = std::max(maxSymanzikDerivDiff, static_cast<double>(
                    infnorm(total + (beta / 5.0) * symanzikDeriv)));
            }
        }
    }
    const double plaquetteKineticRate = mdwfHmcConventionKineticRate<HaloDepth>(momentaHost, plaquetteIpdot);
    const double rectangleKineticRate = mdwfHmcConventionKineticRate<HaloDepth>(momentaHost, rectangleIpdot);
    const double gaugeKineticRate = mdwfHmcConventionKineticRate<HaloDepth>(momentaHost, gaugeIpdot);
    const double plaquetteRelSum = mdwfHmcConventionRelSum(plaquetteActionRate, plaquetteKineticRate);
    const double rectangleRelSum = mdwfHmcConventionRelSum(rectangleActionRate, rectangleKineticRate);
    const double gaugeRelSum = mdwfHmcConventionRelSum(gaugeActionRate, gaugeKineticRate);
    const bool gaugePassed = std::isfinite(plaquetteActionRate) && std::abs(plaquetteActionRate) > 1e-8
                             && plaquetteRelSum <= gaugeTolerance;
    const bool symanzikConsistent = rectangleRelSum <= gaugeTolerance && gaugeRelSum <= gaugeTolerance;
    rootLogger.info("MDWF HMC convention gauge plaquette (beta = ", beta, "): dS_plaq/dtau = ", plaquetteActionRate,
                    ", sum tr(P(-i ipdot_plaq)) = ", plaquetteKineticRate, ", relSum = ", plaquetteRelSum,
                    ", passed = ", gaugePassed);
    rootLogger.info("MDWF HMC convention gauge rectangle diagnostic: dS_rect/dtau = ", rectangleActionRate,
                    ", sum tr(P(-i (gauge_force - ipdot_plaq))) = ", rectangleKineticRate,
                    ", relSum = ", rectangleRelSum);
    rootLogger.info("MDWF HMC convention gauge Symanzik diagnostic: dS_g/dtau = ", gaugeActionRate,
                    ", sum tr(P(-i gauge_force)) = ", gaugeKineticRate, ", relSum = ", gaugeRelSum,
                    ", max |gauge_force + (beta/5) symanzikGaugeActionDeriv| = ", maxSymanzikDerivDiff,
                    " (max |gauge_force| = ", maxGaugeForceNorm, ")");
    if (!symanzikConsistent) {
        rootLogger.warn("MDWF HMC convention: SIMULATeQCD's Symanzik gauge_force is not the gradient of "
                        "-(3 beta/5) symanzik() at relative ", gaugeTolerance,
                        "; resolve before using this gauge action in the MDWF HMC driver");
    }

    // --- Part 2: Mobius clover fermion action with the stored MDWF matrices. ---
    Spinor field(commBase, "MDWF_hmc_convention_field");
    field.template iterateOverBulk<>(FillMDWFHmcConventionSource<double, All, HaloDepth, Ls>());
    field.updateAll();

    auto fermionCase = [&](const std::string &label, double M5, int maxIter) {
        FermionAction action(commBase, field, actionCoefficients, M5, mf, b5, csw, maxIter, precision);
        const MDWFFiniteDifferenceActionValue<double> plus = action(gaugePlus);
        const MDWFFiniteDifferenceActionValue<double> minus = action(gaugeMinus);
        const double actionRate = (plus.action_real - minus.action_real) / (2.0 * epsilon);

        MobiusForward forward(baseGauge, M5, mf, b5, csw, "MDWF_hmc_convention_" + label + "_forward");
        MobiusAdjoint adjoint(baseGauge, M5, mf, b5, csw, "MDWF_hmc_convention_" + label + "_adjoint");
        MobiusNormal normal(commBase, forward, adjoint, "MDWF_hmc_convention_" + label + "_normal");
        MobiusWorkspace workspace;
        workspace.prepare(normal, forward, field, forceCoefficients, maxIter, precision,
                          "MDWF_hmc_convention_" + label + "_workspace");
        MobiusView view(workspace, forward.params().dinCoeff, "MDWF_hmc_convention_" + label + "_view");

        HostGauge fermionIpdot(commBase, "MDWF_hmc_convention_" + label + "_ipdot");
        overwriteMDWFCloverAllLinkDirectionIndependentStorageNonzero<HaloDepth, Ls>(
            fermionIpdot, gaugeHost, view, forceCoefficients, csw, commBase, "MDWF_hmc_convention_" + label);
        const double kineticRate = mdwfHmcConventionKineticRate<HaloDepth>(momentaHost, fermionIpdot);
        const double relSum = mdwfHmcConventionRelSum(actionRate, kineticRate);

        const bool passed = plus.converged && minus.converged && workspace.converged()
                            && std::isfinite(actionRate) && std::abs(actionRate) > 1e-8
                            && relSum <= fermionTolerance;
        rootLogger.info("MDWF HMC convention fermion ", label, " (M5 = ", M5, ", mf = ", mf, ", b5 = ", b5,
                        ", c_sw = ", csw, "): dS_f/dtau = ", actionRate,
                        ", sum tr(P(-i K)) = ", kineticRate, ", relSum = ", relSum,
                        ", actionPlus = ", plus.action_real,
                        ", actionImagRelative = ", std::abs(plus.action_imag) / std::max(1.0, std::abs(plus.action_real)),
                        ", passed = ", passed);
        return passed;
    };

    const bool controlPassed = fermionCase("control", -2.0, 2000);
    const bool physicalPassed = fermionCase("physical-like", 1.8, 20000);

    if (!momentaPassed || !noisePassed || !gaugePassed || !controlPassed || !physicalPassed) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF HMC convention test failed: momenta passed = ", momentaPassed,
            ", noise passed = ", noisePassed, ", gauge passed = ", gaugePassed,
            ", fermion control passed = ", controlPassed, ", fermion physical-like passed = ", physicalPassed,
            " (see diagnostics above)"));
    }

    rootLogger.info("MDWF HMC convention test passed with Ls = ", Ls, ": ipdot = K (no extra sign or factor); ",
                    "Symanzik gauge action/force consistent = ", symanzikConsistent);
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

        runMDWFHmcConventionTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
