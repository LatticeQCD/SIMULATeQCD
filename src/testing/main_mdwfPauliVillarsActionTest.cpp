/*
 * MDWF Pauli-Villars two-flavour action test (step 4 of the MDWF RHMC plan).
 *
 * Checks MDWFPauliVillarsTwoFlavorFermionAction (MDWFHmcFermionActions.h),
 *
 *   S = phi^\dagger M_1 (M_f^\dagger M_f)^{-1} M_1^\dagger phi,
 *   phi = M_1 (M_1^\dagger M_1)^{-1} M_f^\dagger eta,
 *
 * with Mobius clover M(m) at M5 = 1.8, b5 = 1.5, c_sw = 0.5, mf = 0.1, and
 * pv_mass = 1, on a random gauge field (6^4, Ls = 8):
 *
 *   1. Heatbath identity S = eta^\dagger eta (relative 1e-8).
 *   2. Force/energy identity along U -> exp(i eps P) U with Gaussian P:
 *      dS/dtau + sum tr(P (-i K)) = 0 (relative 1e-5), the check that
 *      established ipdot = K in mdwfMobiusHmcConventionTest, applied to the
 *      two-term Pauli-Villars force.
 *   3. Cancellation: at mf = pv_mass the two force terms cancel; the rms force
 *      must be below 1e-6 times the mf = 0.1 rms force.
 *   4. Diagnostic: rms of the Pauli-Villars force versus the bare two-flavour
 *      force MDWFBareTwoFlavorFermionAction on the same gauge field.
 *
 * Single rank; no HMC trajectory (see mdwfPauliVillarsHmcTrajectoryTest).
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFHmc.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

template<size_t HaloDepth>
void mdwfPvTestEvolve(Gaugefield<double, true, HaloDepth, R18> &gaugeOut,
                      Gaugefield<double, true, HaloDepth, R18> &gaugeIn,
                      Gaugefield<double, true, HaloDepth, R18> &momenta,
                      double stepsize) {
    gaugeOut.iterateOverBulkAllMu(MDWFHmcEvolveQ<HaloDepth>(gaugeIn.getAccessor(), momenta.getAccessor(), stepsize));
    gaugeOut.updateAll();
}

template<size_t HaloDepth>
double mdwfPvTestKineticRate(const Gaugefield<double, false, HaloDepth, R18> &momenta,
                             const Gaugefield<double, false, HaloDepth, R18> &ipdot) {
    typedef GIndexer<All, HaloDepth> GInd;
    const SU3Accessor<double, R18> pAcc = momenta.getAccessor();
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

template<size_t HaloDepth>
double mdwfPvTestForceRms(const Gaugefield<double, false, HaloDepth, R18> &ipdot) {
    typedef GIndexer<All, HaloDepth> GInd;
    const SU3Accessor<double, R18> acc = ipdot.getAccessor();
    double sum = 0.0;
    for (size_t siteIndex = 0; siteIndex < GInd::getLatData().vol4; siteIndex++) {
        const gSite site = GInd::getSite(siteIndex);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const SU3<double> k = acc.getLink(GInd::getSiteMu(site, mu));
            sum += -tr_d(k, k);
        }
    }
    return std::sqrt(sum / (4.0 * static_cast<double>(GInd::getLatData().vol4)));
}

template<class Spinor>
double mdwfPvTestNorm2(Spinor &spinor) {
    double sum = 0.0;
    for (const COMPLEX(double) &stackDot : spinor.dotProductStacked(spinor)) {
        sum += real<double>(stackDot);
    }
    return sum;
}

// Independent evaluation of S(U) = psi^\dagger (M_f^\dagger M_f)^{-1} psi, psi = M_1^\dagger phi, with freshly built
// operators on the given gauge field (no MDWFPauliVillarsTwoFlavorFermionAction involved).
template<size_t HaloDepth, size_t Ls>
double mdwfPvTestIndependentAction(CommunicationBase &commBase,
                                   Gaugefield<double, true, HaloDepth, R18> &gauge,
                                   MDWFSpinor<double, true, All, HaloDepth, Ls> &phi,
                                   const MDWFHmcParameters &param,
                                   const std::string &name) {
    using Forward = MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Adjoint = MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Normal = MDWFNormalOperator<Forward, Adjoint>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, Normal>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;

    Forward forwardF(gauge, param.M5, param.mf, param.b5, param.csw, name + "_forward_f");
    Adjoint adjointF(gauge, param.M5, param.mf, param.b5, param.csw, name + "_adjoint_f");
    Adjoint adjoint1(gauge, param.M5, param.pv_mass, param.b5, param.csw, name + "_adjoint_1");
    Normal normalF(commBase, forwardF, adjointF, name + "_normal_f");
    Adapter adapterF(normalF);
    Spinor psi(commBase, name + "_psi");
    Spinor workspace(commBase, name + "_workspace");
    adjoint1.apply(psi, phi, true);
    const MDWFRationalCoefficients<double> inverse
        = mdwfHmcInverseCoefficients(MDWFRationalCoefficientRole::Action, name + "_inverse");
    return computeMDWFRationalAction<double, Adapter>(adapterF, workspace, psi, inverse, param.max_iter,
                                                      param.precision, name + "_action").action_real;
}

template<size_t Ls>
void runMDWFPauliVillarsActionTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using PvAction = MDWFPauliVillarsTwoFlavorFermionAction<HaloDepth, Ls>;
    using BareAction = MDWFBareTwoFlavorFermionAction<HaloDepth, Ls>;

    MDWFHmcParameters param{};
    param.beta = 6.0;
    param.M5 = 1.8;
    param.mf = 0.1;
    param.b5 = 1.5;
    param.csw = 0.5;
    param.tau = 1.0;
    param.steps = 1;
    param.max_iter = 20000;
    param.precision = 1e-10;
    param.pv_mass = 1.0;
    const double epsilon = 1e-4;

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260924);
    d_rand = h_rand;

    Gauge gauge(commBase, "MDWF_pv_test_gauge");
    Gauge gaugePlus(commBase, "MDWF_pv_test_gauge_plus");
    Gauge gaugeMinus(commBase, "MDWF_pv_test_gauge_minus");
    Gauge momenta(commBase, "MDWF_pv_test_momenta");
    gauge.random(d_rand.state);
    gauge.updateAll();
    momenta.gauss(d_rand.state);
    momenta.updateAll();

    HostGauge gaugeHost(commBase, "MDWF_pv_test_gauge_host");
    HostGauge momentaHost(commBase, "MDWF_pv_test_momenta_host");
    HostGauge ipdotHost(commBase, "MDWF_pv_test_ipdot_host");
    gaugeHost = gauge;
    momentaHost = momenta;

    rootLogger.info("MDWF Pauli-Villars action test: M5 = ", param.M5, ", mf = ", param.mf, ", pv_mass = ",
                    param.pv_mass, ", b5 = ", param.b5, ", c_sw = ", param.csw, ", Ls = ", Ls,
                    ", solver precision = ", param.precision, ", random gauge");

    // --- Part 1: heatbath identity. ---
    PvAction pv(commBase, gauge, param);
    pv.heatbath(d_rand.state);
    const double action = pv.action();
    const double heatbathRelDiff = std::abs(action - pv.noiseNorm2()) / std::max(1.0, pv.noiseNorm2());
    const bool heatbathPassed = heatbathRelDiff <= 1e-8;
    rootLogger.info("MDWF Pauli-Villars action test heatbath: S = ", action, ", eta^dagger eta = ", pv.noiseNorm2(),
                    ", relDiff = ", heatbathRelDiff, ", passed = ", heatbathPassed);

    // --- Part 2: force/energy identity. ---
    pv.force(ipdotHost, gaugeHost);
    const double pvForceRms = mdwfPvTestForceRms<HaloDepth>(ipdotHost);
    const double kineticRate = mdwfPvTestKineticRate<HaloDepth>(momentaHost, ipdotHost);

    mdwfPvTestEvolve<HaloDepth>(gaugePlus, gauge, momenta, epsilon);
    mdwfPvTestEvolve<HaloDepth>(gaugeMinus, gauge, momenta, -epsilon);
    PvAction pvPlus(commBase, gaugePlus, param);
    PvAction pvMinus(commBase, gaugeMinus, param);
    pvPlus.phi() = pv.phi();
    pvMinus.phi() = pv.phi();
    pvPlus.phi().updateAll();
    pvMinus.phi().updateAll();
    const double actionPlus = pvPlus.action();
    const double actionMinus = pvMinus.action();
    const double actionRate = (actionPlus - actionMinus) / (2.0 * epsilon);

    // Diagnostics for the finite difference (a first run found dS/dtau = -1.09e10 against 432).
    const double actionBaseAgain = pv.action();
    const double independentBase = mdwfPvTestIndependentAction<HaloDepth, Ls>(commBase, gauge, pv.phi(), param,
                                                                               "MDWF_pv_test_indep_base");
    const double independentPlus = mdwfPvTestIndependentAction<HaloDepth, Ls>(commBase, gaugePlus, pv.phi(), param,
                                                                               "MDWF_pv_test_indep_plus");
    const double independentMinus = mdwfPvTestIndependentAction<HaloDepth, Ls>(commBase, gaugeMinus, pv.phi(), param,
                                                                                "MDWF_pv_test_indep_minus");
    rootLogger.info("MDWF Pauli-Villars action test diagnostics: |phi|^2 base/plus/minus = ", mdwfPvTestNorm2(pv.phi()),
                    " / ", mdwfPvTestNorm2(pvPlus.phi()), " / ", mdwfPvTestNorm2(pvMinus.phi()));
    rootLogger.info("MDWF Pauli-Villars action test diagnostics: class S(U) = ", actionBaseAgain,
                    ", S(U+) = ", actionPlus, ", S(U-) = ", actionMinus);
    rootLogger.info("MDWF Pauli-Villars action test diagnostics: independent S(U) = ", independentBase,
                    ", S(U+) = ", independentPlus, ", S(U-) = ", independentMinus,
                    ", independent dS/dtau = ", (independentPlus - independentMinus) / (2.0 * epsilon));
    const double identityRelSum = std::abs(actionRate + kineticRate) / std::max(1.0, std::abs(actionRate));
    const bool identityPassed = std::isfinite(actionRate) && std::abs(actionRate) > 1e-8 && identityRelSum <= 1e-5;
    rootLogger.info("MDWF Pauli-Villars action test identity: dS/dtau = ", actionRate,
                    ", sum tr(P(-i K)) = ", kineticRate, ", relSum = ", identityRelSum,
                    ", passed = ", identityPassed);

    // --- Part 3: cancellation at mf = pv_mass. ---
    MDWFHmcParameters equalParam = param;
    equalParam.mf = param.pv_mass;
    PvAction pvEqual(commBase, gauge, equalParam);
    pvEqual.heatbath(d_rand.state);
    rootLogger.info("MDWF Pauli-Villars action test diagnostics (mf = pv_mass): eta^dagger eta = ", pvEqual.noiseNorm2(),
                    ", |phi|^2 = ", mdwfPvTestNorm2(pvEqual.phi()), ", S = ", pvEqual.action());
    pvEqual.force(ipdotHost, gaugeHost);
    const double equalForceRms = mdwfPvTestForceRms<HaloDepth>(ipdotHost);
    const double cancellationRatio = equalForceRms / std::max(pvForceRms, 1e-300);
    const bool cancellationPassed = pvForceRms > 0.0 && cancellationRatio <= 1e-6;
    rootLogger.info("MDWF Pauli-Villars action test cancellation (mf = pv_mass = ", param.pv_mass,
                    "): rms force = ", equalForceRms, ", ratio to mf = ", param.mf, " rms force = ",
                    cancellationRatio, ", passed = ", cancellationPassed);

    // --- Part 4: diagnostic, Pauli-Villars versus bare two-flavour force. ---
    BareAction bare(commBase, gauge, param);
    bare.heatbath(d_rand.state);
    bare.force(ipdotHost, gaugeHost);
    const double bareForceRms = mdwfPvTestForceRms<HaloDepth>(ipdotHost);
    rootLogger.info("MDWF Pauli-Villars action test force strength: rms Pauli-Villars force = ", pvForceRms,
                    ", rms bare two-flavour force = ", bareForceRms, ", ratio = ", pvForceRms / bareForceRms);

    if (!heatbathPassed || !identityPassed || !cancellationPassed) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Pauli-Villars action test failed: heatbath passed = ", heatbathPassed,
            ", identity passed = ", identityPassed, ", cancellation passed = ", cancellationPassed,
            " (see diagnostics above)"));
    }
    rootLogger.info("MDWF Pauli-Villars action test passed with Ls = ", Ls);
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

        runMDWFPauliVillarsActionTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
