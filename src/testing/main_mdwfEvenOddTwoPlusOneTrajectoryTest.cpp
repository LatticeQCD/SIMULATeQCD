/*
 * Even/odd 2+1 flavour MDWF HMC trajectory test (EVEN_ODD_DESIGN.md stage E2b,
 * clover c_sw = 0.5; the c_sw = 0 version passed in job 34574588).
 *
 * MDWFEvenOddTwoPlusOneHmc (MDWFEvenOddFermionActions.h): Wilson gauge action,
 * even-site Pauli-Villars light pair (mf = 0.1) and even-site one-flavour RHMC
 * strange quark (ms = 0.2), pv_mass = 1, M5 = 1.8, b5 = 1.5, c_sw = 0.5, unit
 * start on 6^4, Ls = 8, beta = 6, tau = 0.2, solver precision 1e-10. Strange
 * intervals [lambda_min / 3, 1.5 lambda_max] of Mhat^+ Mhat on the start
 * configuration, errors 1e-12 (heatbath, action) and 1e-8 (force).
 *
 *   1. Heatbath identity per pseudofermion, pseudofermion part of the action
 *      (the log det M_oo part is not sampled by the heatbath) (1e-6).
 *   2. Reversibility over 8 steps (1e-8 links and momenta, 1e-6 in H).
 *   3. Delta H ratios between 3 and 6 for 8/16/32 steps.
 *   4. Cost: wall time of one 8-step trajectory from the same start with the
 *      even/odd HMC and with the unpreconditioned MDWFTwoPlusOneHmc at the
 *      same parameters (its intervals from Lanczos on M^+ M), reported.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFEvenOddFermionActions.h"
#include "../experimental/mdwf/MDWFSpectralBounds.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>

template<size_t HaloDepth>
double mdwfEo21MaxLinkDiff(CommunicationBase &commBase, Gaugefield<double, true, HaloDepth, R18> &a,
                           Gaugefield<double, true, HaloDepth, R18> &b) {
    typedef GIndexer<All, HaloDepth> GInd;
    Gaugefield<double, false, HaloDepth, R18> aHost(commBase, "MDWF_eo21_diff_a");
    Gaugefield<double, false, HaloDepth, R18> bHost(commBase, "MDWF_eo21_diff_b");
    aHost = a;
    bHost = b;
    const SU3Accessor<double, R18> aAcc = aHost.getAccessor();
    const SU3Accessor<double, R18> bAcc = bHost.getAccessor();
    double maxDiff = 0.0;
    for (size_t i = 0; i < GInd::getLatData().vol4; i++) {
        const gSite site = GInd::getSite(i);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const gSiteMu siteMu = GInd::getSiteMu(site, mu);
            maxDiff = std::max(maxDiff, static_cast<double>(infnorm(aAcc.getLink(siteMu) - bAcc.getLink(siteMu))));
        }
    }
    return maxDiff;
}

template<class Adapter, class Spinor>
MDWFLanczosCheckpoint mdwfEo21Lanczos(Adapter &adapter, Spinor &start, uint4 *randState, int steps,
                                      const std::string &name, const std::string &label) {
    start.gauss(randState);
    start.updateAll();
    const MDWFLanczosResult result = mdwfLanczosExtremes(adapter, start, steps, {steps / 2, steps}, name);
    const MDWFLanczosCheckpoint &c = result.checkpoints.back();
    rootLogger.info("MDWF even/odd 2+1 test Lanczos ", label, ": steps = ", c.steps, ", lambda_min = ", c.lambda_min,
                    ", lambda_max = ", c.lambda_max);
    return c;
}

template<size_t Ls>
void runMDWFEvenOddTwoPlusOneTrajectoryTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using EoHmc = MDWFEvenOddTwoPlusOneHmc<HaloDepth, Ls>;
    using FullHmc = MDWFTwoPlusOneHmc<HaloDepth, Ls>;
    using Types = MDWFEvenOddTypes<HaloDepth, Ls>;
    using Forward = MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Adjoint = MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Normal = MDWFNormalOperator<Forward, Adjoint>;
    using NormalAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, Normal>;

    MDWFHmcParameters param{};
    param.beta = 6.0;
    param.M5 = 1.8;
    param.mf = 0.1;
    param.b5 = 1.5;
    param.csw = 0.5;
    param.pv_mass = 1.0;
    param.tau = 0.2;
    param.steps = 8;
    param.max_iter = 20000;
    param.precision = 1e-10;
    param.rhmc.ms = 0.2;
    param.rhmc.action_error = 1e-12;
    param.rhmc.force_error = 1e-8;
    param.rhmc.max_order = 30;
    param.rhmc.digits = 50;

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20261004);
    d_rand = h_rand;

    Gauge gauge(commBase, "MDWF_eo21_gauge");
    gauge.one();
    gauge.updateAll();

    // Intervals of Mhat^+ Mhat (even/odd) and of M^+ M (unpreconditioned comparison).
    MDWFHmcParameters fullParam = param;
    {
        typename Types::EvenOdd eoS(gauge, param.M5, param.rhmc.ms, param.b5, param.csw, "MDWF_eo21_lzs_eo");
        typename Types::EvenOdd eo1(gauge, param.M5, param.pv_mass, param.b5, param.csw, "MDWF_eo21_lzp_eo");
        typename Types::NormalOp nS(eoS, commBase, "MDWF_eo21_lzs_nrm");
        typename Types::NormalOp n1(eo1, commBase, "MDWF_eo21_lzp_nrm");
        typename Types::Adapter aS(nS);
        typename Types::Adapter a1(n1);
        typename Types::SpinorE startE(commBase, "MDWF_eo21_lz_starte");
        eoS.refresh();
        eo1.refresh();
        const MDWFLanczosCheckpoint cS = mdwfEo21Lanczos(aS, startE, d_rand.state, 400, "MDWF_eo21_lzs",
                                                         "Mhat^+Mhat (ms)");
        const MDWFLanczosCheckpoint c1 = mdwfEo21Lanczos(a1, startE, d_rand.state, 200, "MDWF_eo21_lzp",
                                                         "Mhat^+Mhat (pv)");
        param.rhmc.lambda_low_s = cS.lambda_min / 3.0;
        param.rhmc.lambda_high_s = 1.5 * cS.lambda_max;
        param.rhmc.lambda_low_pv = c1.lambda_min / 3.0;
        param.rhmc.lambda_high_pv = 1.5 * c1.lambda_max;

        Forward fS(gauge, param.M5, param.rhmc.ms, param.b5, param.csw, "MDWF_eo21_fls_fwd");
        Adjoint dS(gauge, param.M5, param.rhmc.ms, param.b5, param.csw, "MDWF_eo21_fls_adj");
        Normal nfS(commBase, fS, dS, "MDWF_eo21_fls_nrm");
        NormalAdapter afS(nfS);
        Forward f1(gauge, param.M5, param.pv_mass, param.b5, param.csw, "MDWF_eo21_flp_fwd");
        Adjoint d1(gauge, param.M5, param.pv_mass, param.b5, param.csw, "MDWF_eo21_flp_adj");
        Normal nf1(commBase, f1, d1, "MDWF_eo21_flp_nrm");
        NormalAdapter af1(nf1);
        typename Types::SpinorAll startAll(commBase, "MDWF_eo21_lz_startall");
        const MDWFLanczosCheckpoint fcS = mdwfEo21Lanczos(afS, startAll, d_rand.state, 400, "MDWF_eo21_lzfs",
                                                          "M^+M (ms)");
        const MDWFLanczosCheckpoint fc1 = mdwfEo21Lanczos(af1, startAll, d_rand.state, 200, "MDWF_eo21_lzfp",
                                                          "M^+M (pv)");
        fullParam.rhmc.lambda_low_s = fcS.lambda_min / 3.0;
        fullParam.rhmc.lambda_high_s = 1.5 * fcS.lambda_max;
        fullParam.rhmc.lambda_low_pv = fc1.lambda_min / 3.0;
        fullParam.rhmc.lambda_high_pv = 1.5 * fc1.lambda_max;
    }

    EoHmc hmc(commBase, gauge, param, d_rand.state);
    auto &light = hmc.fermion().first();
    auto &strange = hmc.fermion().second();
    rootLogger.info("MDWF even/odd 2+1 test strange orders: A^(1/4) ", strange.quarterS().order, ", A^(-1/2) ",
                    strange.halfS().order, " (force ", strange.halfSForce().order, "), B^(1/4) ",
                    strange.quarterPv().order, " (force ", strange.quarterPvForce().order, ")");

    // --- 1. Heatbath identity. ---
    hmc.refreshMomenta();
    hmc.heatbath();
    light.action();
    strange.action();
    const double lightRel = std::abs(light.lastPseudofermionAction() - light.noiseNorm2())
                            / std::max(1.0, light.noiseNorm2());
    const double strangeRel = std::abs(strange.lastPseudofermionAction() - strange.noiseNorm2())
                              / std::max(1.0, strange.noiseNorm2());
    rootLogger.info("MDWF even/odd 2+1 test log det M_oo action parts: light ", light.lastDetAction(), ", strange ",
                    strange.lastDetAction());
    const bool heatbathPassed = lightRel <= 1e-6 && strangeRel <= 1e-6;
    rootLogger.info("MDWF even/odd 2+1 test heatbath: light relDiff = ", lightRel, ", strange relDiff = ", strangeRel,
                    ", passed = ", heatbathPassed);

    Gauge startGauge(commBase, "MDWF_eo21_start_gauge");
    Gauge startMomenta(commBase, "MDWF_eo21_start_momenta");
    startGauge = gauge;
    startMomenta = hmc.momenta();

    // --- 2. Reversibility. ---
    const MDWFHmcEnergy start = hmc.energy();
    auto t0 = std::chrono::steady_clock::now();
    hmc.integrate(param.steps);
    const double eoSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const MDWFHmcEnergy forward = hmc.energy();
    hmc.flipMomenta();
    hmc.integrate(param.steps);
    hmc.flipMomenta();
    const MDWFHmcEnergy back = hmc.energy();
    const double linkDiff = mdwfEo21MaxLinkDiff<HaloDepth>(commBase, gauge, startGauge);
    const double momDiff = mdwfEo21MaxLinkDiff<HaloDepth>(commBase, hmc.momenta(), startMomenta);
    const double energyDiff = std::abs(back.total() - start.total());
    const bool reversibilityPassed = linkDiff <= 1e-8 && momDiff <= 1e-8 && energyDiff <= 1e-6;
    rootLogger.info("MDWF even/odd 2+1 test reversibility (8 steps): forward Delta H = ", forward.total() - start.total(),
                    ", max link diff = ", linkDiff, ", max momentum diff = ", momDiff, ", |H_back - H_start| = ",
                    energyDiff, ", passed = ", reversibilityPassed);

    // --- 3. Delta H scaling. ---
    const std::array<int, 3> stepCounts{{8, 16, 32}};
    std::array<double, 3> deltaH{};
    for (size_t i = 0; i < stepCounts.size(); i++) {
        gauge = startGauge;
        gauge.updateAll();
        hmc.momenta() = startMomenta;
        hmc.momenta().updateAll();
        const MDWFHmcEnergy restored = hmc.energy();
        hmc.integrate(stepCounts[i]);
        deltaH[i] = hmc.energy().total() - restored.total();
        rootLogger.info("MDWF even/odd 2+1 test scaling: steps = ", stepCounts[i], ", Delta H = ", deltaH[i]);
    }
    const double r1 = std::abs(deltaH[0]) / std::max(std::abs(deltaH[1]), 1e-300);
    const double r2 = std::abs(deltaH[1]) / std::max(std::abs(deltaH[2]), 1e-300);
    const bool scalingPassed = r1 >= 3.0 && r1 <= 6.0 && r2 >= 3.0 && r2 <= 6.0;
    rootLogger.info("MDWF even/odd 2+1 test scaling: ratios ", r1, ", ", r2, " (leapfrog expects 4), passed = ",
                    scalingPassed);

    // --- 4. Cost versus the unpreconditioned HMC. ---
    gauge = startGauge;
    gauge.updateAll();
    FullHmc fullHmc(commBase, gauge, fullParam, d_rand.state);
    fullHmc.refreshMomenta();
    fullHmc.momenta() = startMomenta;
    fullHmc.momenta().updateAll();
    fullHmc.heatbath();
    const MDWFHmcEnergy fullStart = fullHmc.energy();
    t0 = std::chrono::steady_clock::now();
    fullHmc.integrate(param.steps);
    const double fullSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double fullDeltaH = fullHmc.energy().total() - fullStart.total();
    rootLogger.info("MDWF even/odd 2+1 test cost (8 steps, 9 fermion forces, same start): even/odd ", eoSeconds,
                    " s (Delta H ", forward.total() - start.total(), ") versus unpreconditioned ", fullSeconds,
                    " s (Delta H ", fullDeltaH, "), speedup ", fullSeconds / eoSeconds);

    if (!heatbathPassed || !reversibilityPassed || !scalingPassed) {
        throw std::runtime_error(stdLogger.fatal("MDWF even/odd 2+1 trajectory test failed: heatbath passed = ",
                                                 heatbathPassed, ", reversibility passed = ", reversibilityPassed,
                                                 ", scaling passed = ", scalingPassed));
    }
    rootLogger.info("MDWF even/odd 2+1 trajectory test passed with Ls = ", Ls);
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

        runMDWFEvenOddTwoPlusOneTrajectoryTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
