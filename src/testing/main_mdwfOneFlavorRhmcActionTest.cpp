/*
 * MDWF one-flavour Pauli-Villars RHMC action test (step 5 of the MDWF RHMC
 * plan: RHMC for the strange quark).
 *
 * Checks MDWFOneFlavorRhmcFermionAction (MDWFRhmcFermionActions.h),
 *
 *   S = phi^\dagger r_B^(1/4) r_A^(-1/2) r_B^(1/4) phi,   phi = r_B^(-1/4) r_A^(1/4) eta,
 *   A = M_s^\dagger M_s,  B = M_1^\dagger M_1,
 *
 * with Mobius clover M(m) at M5 = 1.8, b5 = 1.5, c_sw = 0.5, ms = 0.1,
 * pv_mass = 1 on a random gauge field (6^4, Ls = 8), solver precision 1e-11,
 * AlgRemez approximations with max relative error <= 1e-12 on intervals
 * measured here:
 *
 *   0. Lanczos (600 steps for A, 300 for B) gives the spectral range; the
 *      approximation intervals are [lambda_min / 2, 1.2 lambda_max].
 *   1. Operator-level approximation checks on a Gaussian vector v (relative
 *      norm differences):
 *        (r_B^(1/4))^4 v = B v                 (1e-8; fails if B's spectrum leaves the interval)
 *        r_B^(-1/4) r_B^(1/4) v = v            (1e-8; exact reciprocals, solver and wiring only)
 *        (r_A^(-1/2))^2 v = A^(-1) v (CG)      (1e-7; fails if A's spectrum leaves the interval)
 *        (r_A^(1/4))^2 r_A^(-1/2) v = v        (1e-7; heatbath versus action approximation of A)
 *      plus a diagnostic control: B^(1/4) of the same order on the too-narrow
 *      interval [4 lambda_min(B), ...] must be visibly worse.
 *   2. Heatbath identity S = eta^\dagger eta (relative 1e-6).
 *   3. Force/energy identity along U -> exp(i eps P) U with Gaussian P:
 *      dS/dtau + sum tr(P (-i K)) = 0 (relative 1e-5), with the force using
 *      the action approximations (force_error = 0), so it is the exact
 *      derivative of the action.
 *   4. Cancellation at ms = pv_mass: A = B, S = phi^\dagger phi up to the
 *      approximation error, so the rms force must be below 1e-6 times the
 *      ms = 0.1 rms force.
 *   5. Force approximations of lower accuracy (force_error = 1e-8): relative
 *      rms difference to the exact force below 1e-4, and the pole counts.
 *   6. Diagnostic: rms one-flavour force versus the Pauli-Villars two-flavour
 *      force at mf = ms on the same gauge field.
 *
 * Single rank; no HMC trajectory (see mdwfTwoPlusOneHmcTrajectoryTest).
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFRhmcFermionActions.h"
#include "../experimental/mdwf/MDWFSpectralBounds.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

template<size_t HaloDepth>
void mdwfRhmcTestEvolve(Gaugefield<double, true, HaloDepth, R18> &gaugeOut,
                        Gaugefield<double, true, HaloDepth, R18> &gaugeIn,
                        Gaugefield<double, true, HaloDepth, R18> &momenta,
                        double stepsize) {
    gaugeOut.iterateOverBulkAllMu(MDWFHmcEvolveQ<HaloDepth>(gaugeIn.getAccessor(), momenta.getAccessor(), stepsize));
    gaugeOut.updateAll();
}

template<size_t HaloDepth>
double mdwfRhmcTestKineticRate(const Gaugefield<double, false, HaloDepth, R18> &momenta,
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

// rms over bulk links of -tr(K K) for a (difference of) anti-Hermitian force(s); b = nullptr means K = a.
template<size_t HaloDepth>
double mdwfRhmcTestForceRms(const Gaugefield<double, false, HaloDepth, R18> &a,
                            const Gaugefield<double, false, HaloDepth, R18> *b = nullptr) {
    typedef GIndexer<All, HaloDepth> GInd;
    const SU3Accessor<double, R18> aAcc = a.getAccessor();
    double sum = 0.0;
    for (size_t siteIndex = 0; siteIndex < GInd::getLatData().vol4; siteIndex++) {
        const gSite site = GInd::getSite(siteIndex);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const gSiteMu siteMu = GInd::getSiteMu(site, mu);
            SU3<double> k = aAcc.getLink(siteMu);
            if (b != nullptr) {
                k -= b->getAccessor().getLink(siteMu);
            }
            sum += -tr_d(k, k);
        }
    }
    return std::sqrt(sum / (4.0 * static_cast<double>(GInd::getLatData().vol4)));
}

template<class Adapter, class Spinor>
double mdwfRhmcTestRelDiff(Adapter &adapter, Spinor &work, const Spinor &a, Spinor &b) {
    work = a;
    work.template axpyThisB<64>(-1.0, b);
    return std::sqrt(adapter.norm2(work) / adapter.norm2(b));
}

// Extreme Ritz values of M(mass)^\dagger M(mass) after `steps` Lanczos steps (and after steps / 2, for convergence).
template<size_t HaloDepth, size_t Ls>
MDWFLanczosCheckpoint mdwfRhmcTestBounds(CommunicationBase &commBase, Gaugefield<double, true, HaloDepth, R18> &gauge,
                                         const MDWFHmcParameters &param, double mass, int steps, uint4 *randState,
                                         const std::string &name) {
    using Forward = MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Adjoint = MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Normal = MDWFNormalOperator<Forward, Adjoint>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, Normal>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;

    Forward forward(gauge, param.M5, mass, param.b5, param.csw, name + "_forward");
    Adjoint adjoint(gauge, param.M5, mass, param.b5, param.csw, name + "_adjoint");
    Normal normal(commBase, forward, adjoint, name + "_normal");
    Adapter adapter(normal);
    Spinor start(commBase, name + "_start");
    start.gauss(randState);
    start.updateAll();
    const MDWFLanczosResult result = mdwfLanczosExtremes(adapter, start, steps, {steps / 2, steps}, name + "_lanczos");
    for (const MDWFLanczosCheckpoint &c : result.checkpoints) {
        rootLogger.info("MDWF one-flavour RHMC test Lanczos (mass = ", mass, "): steps = ", c.steps,
                        ", lambda_min = ", c.lambda_min, ", lambda_max = ", c.lambda_max);
    }
    return result.checkpoints.back();
}

void logMDWFRhmcApproximation(const std::string &label, const MDWFRemezApproximation &approx) {
    rootLogger.info("MDWF one-flavour RHMC test approximation ", label, ": x^(", approx.pnum, "/", approx.pden,
                    ") on [", approx.lambda_low, ", ", approx.lambda_high, "], order ", approx.order,
                    ", max relative error ", approx.max_relative_error);
}

template<size_t Ls>
void runMDWFOneFlavorRhmcActionTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;
    using Action = MDWFOneFlavorRhmcFermionAction<HaloDepth, Ls>;
    using PvAction = MDWFPauliVillarsTwoFlavorFermionAction<HaloDepth, Ls>;
    using CG = MDWFCoupledCG<double, typename Action::NormalAdapter>;

    MDWFHmcParameters param{};
    param.beta = 6.0;
    param.M5 = 1.8;
    param.mf = 0.1;
    param.b5 = 1.5;
    param.csw = 0.5;
    param.tau = 1.0;
    param.steps = 1;
    param.max_iter = 40000;
    param.precision = 1e-11;
    param.pv_mass = 1.0;
    param.rhmc.ms = 0.1;
    param.rhmc.action_error = 1e-12;
    param.rhmc.force_error = 0.0;
    param.rhmc.max_order = 30;
    param.rhmc.digits = 50;
    const double epsilon = 1e-4;

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260927);
    d_rand = h_rand;

    Gauge gauge(commBase, "MDWF_rhmc1_test_gauge");
    Gauge gaugePlus(commBase, "MDWF_rhmc1_test_gauge_plus");
    Gauge gaugeMinus(commBase, "MDWF_rhmc1_test_gauge_minus");
    Gauge momenta(commBase, "MDWF_rhmc1_test_momenta");
    gauge.random(d_rand.state);
    gauge.updateAll();
    momenta.gauss(d_rand.state);
    momenta.updateAll();

    HostGauge gaugeHost(commBase, "MDWF_rhmc1_test_gauge_host");
    HostGauge momentaHost(commBase, "MDWF_rhmc1_test_momenta_host");
    HostGauge ipdotHost(commBase, "MDWF_rhmc1_test_ipdot_host");
    HostGauge ipdotOtherHost(commBase, "MDWF_rhmc1_test_ipdot_other_host");
    gaugeHost = gauge;
    momentaHost = momenta;

    rootLogger.info("MDWF one-flavour RHMC test: M5 = ", param.M5, ", ms = ", param.rhmc.ms, ", pv_mass = ",
                    param.pv_mass, ", b5 = ", param.b5, ", c_sw = ", param.csw, ", Ls = ", Ls,
                    ", solver precision = ", param.precision, ", approximation target = ", param.rhmc.action_error,
                    ", random gauge");

    // --- Part 0: spectral ranges and approximation intervals. ---
    const MDWFLanczosCheckpoint boundsS = mdwfRhmcTestBounds<HaloDepth, Ls>(commBase, gauge, param, param.rhmc.ms, 600,
                                                                            d_rand.state, "MDWF_rhmc1_test_bounds_s");
    const MDWFLanczosCheckpoint boundsPv = mdwfRhmcTestBounds<HaloDepth, Ls>(commBase, gauge, param, param.pv_mass, 300,
                                                                             d_rand.state, "MDWF_rhmc1_test_bounds_pv");
    param.rhmc.lambda_low_s = 0.5 * boundsS.lambda_min;
    param.rhmc.lambda_high_s = 1.2 * boundsS.lambda_max;
    param.rhmc.lambda_low_pv = 0.5 * boundsPv.lambda_min;
    param.rhmc.lambda_high_pv = 1.2 * boundsPv.lambda_max;

    Action rhmc(commBase, gauge, param);
    logMDWFRhmcApproximation("A^(1/4) heatbath", rhmc.quarterS());
    logMDWFRhmcApproximation("A^(-1/2) action", rhmc.halfS());
    logMDWFRhmcApproximation("B^(+-1/4) action/heatbath", rhmc.quarterPv());
    rootLogger.info("MDWF one-flavour RHMC test approximations:\n", rhmc.describeApproximations());

    // --- Part 1: operator-level approximation checks. ---
    Spinor v(commBase, "MDWF_rhmc1_test_v");
    Spinor w1(commBase, "MDWF_rhmc1_test_w1");
    Spinor w2(commBase, "MDWF_rhmc1_test_w2");
    Spinor ref(commBase, "MDWF_rhmc1_test_ref");
    Spinor work(commBase, "MDWF_rhmc1_test_work");
    v.gauss(d_rand.state);
    v.updateAll();
    auto &adapterS = rhmc.adapterS();
    auto &adapterPv = rhmc.adapterPv();

    // (r_B^(1/4))^4 v versus B v.
    rhmc.applyApproximation(false, rhmc.actionPv(), w1, v);
    rhmc.applyApproximation(false, rhmc.actionPv(), w2, w1);
    rhmc.applyApproximation(false, rhmc.actionPv(), w1, w2);
    rhmc.applyApproximation(false, rhmc.actionPv(), w2, w1);
    adapterPv.apply(ref, v, true);
    const double fourthPowerDiff = mdwfRhmcTestRelDiff(adapterPv, work, w2, ref);

    // r_B^(-1/4) r_B^(1/4) v versus v.
    rhmc.applyApproximation(false, rhmc.actionPv(), w1, v);
    rhmc.applyApproximation(false, rhmc.heatbathPv(), w2, w1);
    const double reciprocalDiff = mdwfRhmcTestRelDiff(adapterPv, work, w2, v);

    // (r_A^(-1/2))^2 v versus A^(-1) v.
    rhmc.applyApproximation(true, rhmc.actionS(), w1, v);
    rhmc.applyApproximation(true, rhmc.actionS(), w2, w1);
    CG cg;
    const MDWFCoupledCGResult<double> cgResult = cg.invert(adapterS, ref, v, param.max_iter, param.precision, true);
    const double inverseDiff = mdwfRhmcTestRelDiff(adapterS, work, w2, ref);

    // (r_A^(1/4))^2 r_A^(-1/2) v versus v.
    rhmc.applyApproximation(true, rhmc.actionS(), w1, v);
    rhmc.applyApproximation(true, rhmc.heatbathS(), w2, w1);
    rhmc.applyApproximation(true, rhmc.heatbathS(), w1, w2);
    const double crossDiff = mdwfRhmcTestRelDiff(adapterS, work, w1, v);

    // Diagnostic control: B^(1/4) of the same order on a too-narrow interval.
    const MDWFRemezApproximation narrow = mdwfRemezPower(1, 4, 4.0 * boundsPv.lambda_min, param.rhmc.lambda_high_pv,
                                                         rhmc.quarterPv().order, 50);
    const MDWFRationalCoefficients<double> narrowPv = mdwfRhmcCoefficients(narrow.power);
    rhmc.applyApproximation(false, narrowPv, w1, v);
    rhmc.applyApproximation(false, narrowPv, w2, w1);
    rhmc.applyApproximation(false, narrowPv, w1, w2);
    rhmc.applyApproximation(false, narrowPv, w2, w1);
    adapterPv.apply(ref, v, true);
    const double narrowDiff = mdwfRhmcTestRelDiff(adapterPv, work, w2, ref);

    const bool approximationPassed = fourthPowerDiff <= 1e-8 && reciprocalDiff <= 1e-8 && cgResult.converged
                                     && inverseDiff <= 1e-7 && crossDiff <= 1e-7;
    rootLogger.info("MDWF one-flavour RHMC test approximations: |(r_B^(1/4))^4 v - B v| / |B v| = ", fourthPowerDiff,
                    ", |r_B^(-1/4) r_B^(1/4) v - v| / |v| = ", reciprocalDiff,
                    ", |(r_A^(-1/2))^2 v - A^(-1) v| / |A^(-1) v| = ", inverseDiff, " (CG ", cgResult.iterations,
                    " iterations), |(r_A^(1/4))^2 r_A^(-1/2) v - v| / |v| = ", crossDiff,
                    ", passed = ", approximationPassed);
    rootLogger.info("MDWF one-flavour RHMC test approximation control: B^(1/4) order ", narrow.order, " on [",
                    narrow.lambda_low, ", ", narrow.lambda_high, "] (error ", narrow.max_relative_error,
                    " there) gives |(r^4 v - B v| / |B v| = ", narrowDiff, " (versus ", fourthPowerDiff,
                    " on the measured interval)");

    // --- Part 2: heatbath identity. ---
    rhmc.heatbath(d_rand.state);
    const int heatbathIterations = rhmc.lastIterations();
    const double action = rhmc.action();
    const double heatbathRelDiff = std::abs(action - rhmc.noiseNorm2()) / std::max(1.0, rhmc.noiseNorm2());
    const bool heatbathPassed = heatbathRelDiff <= 1e-6;
    rootLogger.info("MDWF one-flavour RHMC test heatbath: S = ", action, ", eta^dagger eta = ", rhmc.noiseNorm2(),
                    ", relDiff = ", heatbathRelDiff, ", max multishift iterations heatbath / action = ",
                    heatbathIterations, " / ", rhmc.lastIterations(), ", passed = ", heatbathPassed);

    // --- Part 3: force/energy identity. ---
    rhmc.force(ipdotHost, gaugeHost);
    const double forceRms = mdwfRhmcTestForceRms<HaloDepth>(ipdotHost);
    const double kineticRate = mdwfRhmcTestKineticRate<HaloDepth>(momentaHost, ipdotHost);
    rootLogger.info("MDWF one-flavour RHMC test force: max multishift iterations = ", rhmc.lastIterations(),
                    ", rms force = ", forceRms);

    mdwfRhmcTestEvolve<HaloDepth>(gaugePlus, gauge, momenta, epsilon);
    mdwfRhmcTestEvolve<HaloDepth>(gaugeMinus, gauge, momenta, -epsilon);
    Action rhmcPlus(commBase, gaugePlus, param);
    Action rhmcMinus(commBase, gaugeMinus, param);
    rhmcPlus.phi() = rhmc.phi();
    rhmcMinus.phi() = rhmc.phi();
    rhmcPlus.phi().updateAll();
    rhmcMinus.phi().updateAll();
    const double actionPlus = rhmcPlus.action();
    const double actionMinus = rhmcMinus.action();
    const double actionRate = (actionPlus - actionMinus) / (2.0 * epsilon);
    const double identityRelSum = std::abs(actionRate + kineticRate) / std::max(1.0, std::abs(actionRate));
    const bool identityPassed = std::isfinite(actionRate) && std::abs(actionRate) > 1e-8 && identityRelSum <= 1e-5;
    rootLogger.info("MDWF one-flavour RHMC test identity: S(U+) = ", actionPlus, ", S(U-) = ", actionMinus,
                    ", dS/dtau = ", actionRate, ", sum tr(P(-i K)) = ", kineticRate, ", relSum = ", identityRelSum,
                    ", passed = ", identityPassed);

    // --- Part 4: cancellation at ms = pv_mass. ---
    MDWFHmcParameters equalParam = param;
    equalParam.rhmc.ms = param.pv_mass;
    Action rhmcEqual(commBase, gauge, equalParam);
    rhmcEqual.heatbath(d_rand.state);
    const double equalAction = rhmcEqual.action();
    rhmcEqual.force(ipdotOtherHost, gaugeHost);
    const double equalForceRms = mdwfRhmcTestForceRms<HaloDepth>(ipdotOtherHost);
    const double cancellationRatio = equalForceRms / std::max(forceRms, 1e-300);
    const bool cancellationPassed = forceRms > 0.0 && cancellationRatio <= 1e-6;
    rootLogger.info("MDWF one-flavour RHMC test cancellation (ms = pv_mass = ", param.pv_mass, "): S = ", equalAction,
                    ", eta^dagger eta = ", rhmcEqual.noiseNorm2(), ", rms force = ", equalForceRms,
                    ", ratio to ms = ", param.rhmc.ms, " rms force = ", cancellationRatio,
                    ", passed = ", cancellationPassed);

    // --- Part 5: lower-accuracy force approximations. ---
    MDWFHmcParameters splitParam = param;
    splitParam.rhmc.force_error = 1e-8;
    Action rhmcSplit(commBase, gauge, splitParam);
    rhmcSplit.phi() = rhmc.phi();
    rhmcSplit.phi().updateAll();
    rhmcSplit.force(ipdotOtherHost, gaugeHost);
    const double splitRelDiff = mdwfRhmcTestForceRms<HaloDepth>(ipdotOtherHost, &ipdotHost) / forceRms;
    const bool splitPassed = splitRelDiff <= 1e-4;
    logMDWFRhmcApproximation("A^(-1/2) force", rhmcSplit.halfSForce());
    logMDWFRhmcApproximation("B^(1/4) force", rhmcSplit.quarterPvForce());
    rootLogger.info("MDWF one-flavour RHMC test split force (force_error = ", splitParam.rhmc.force_error,
                    "): poles A ", rhmcSplit.forceS().shift.size(), " (action ", rhmc.forceS().shift.size(),
                    "), B ", rhmcSplit.forcePv().shift.size(), " (action ", rhmc.forcePv().shift.size(),
                    "), relative rms difference to the exact force = ", splitRelDiff, ", passed = ", splitPassed);

    // --- Part 6: diagnostic, one-flavour versus Pauli-Villars two-flavour force at mf = ms. ---
    MDWFHmcParameters pvParam = param;
    pvParam.mf = param.rhmc.ms;
    PvAction pv(commBase, gauge, pvParam);
    pv.heatbath(d_rand.state);
    pv.force(ipdotOtherHost, gaugeHost);
    const double pvForceRms = mdwfRhmcTestForceRms<HaloDepth>(ipdotOtherHost);
    rootLogger.info("MDWF one-flavour RHMC test force strength: rms one-flavour force = ", forceRms,
                    ", rms Pauli-Villars two-flavour force (mf = ", pvParam.mf, ") = ", pvForceRms,
                    ", ratio = ", forceRms / pvForceRms);

    if (!approximationPassed || !heatbathPassed || !identityPassed || !cancellationPassed || !splitPassed) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF one-flavour RHMC test failed: approximation passed = ", approximationPassed,
            ", heatbath passed = ", heatbathPassed, ", identity passed = ", identityPassed,
            ", cancellation passed = ", cancellationPassed, ", split force passed = ", splitPassed,
            " (see diagnostics above)"));
    }
    rootLogger.info("MDWF one-flavour RHMC test passed with Ls = ", Ls);
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

        runMDWFOneFlavorRhmcActionTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
