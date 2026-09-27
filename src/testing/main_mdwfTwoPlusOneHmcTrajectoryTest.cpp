/*
 * MDWF 2+1 flavour HMC trajectory test (step 5 of the MDWF RHMC plan).
 *
 * Exercises MDWFTwoPlusOneHmc (MDWFRhmcFermionActions.h): Wilson gauge action
 * plus the Mobius clover Pauli-Villars two-flavour light pseudofermion
 * (mf = 0.1) and the one-flavour Pauli-Villars RHMC strange pseudofermion
 * (ms = 0.2), pv_mass = 1, which together sample
 * det(M_l^\dagger M_l / M_1^\dagger M_1) det(M_s^\dagger M_s / M_1^\dagger M_1)^(1/2).
 * Unit start on 6^4, Ls = 8, M5 = 1.8, b5 = 1.5, c_sw = 0.5, beta = 6,
 * tau = 0.2, solver precision 1e-10. The strange-quark approximations use
 * intervals [lambda_min / 3, 1.5 lambda_max] from Lanczos on the start
 * configuration, max relative error 1e-12 (heatbath, action) and 1e-8 (force).
 *
 *   1. Heatbath identity: right after the heatbath, S_f = eta^\dagger eta for
 *      the light and the strange pseudofermion (relative 1e-6).
 *   2. Reversibility: integrate tau = 0.2 in 8 steps, flip the momenta,
 *      integrate back, flip again: gauge links and momenta return to their
 *      start (max element difference 1e-8) and H returns to its start
 *      (absolute 1e-6).
 *   3. Delta H scaling: from the identical start state, 8, 16, and 32
 *      leapfrog steps must give |Delta H| decreasing by a factor between 3
 *      and 6 per halving of eps, and restoring the start state must reproduce
 *      the start energy (relative 1e-9).
 *
 * No Metropolis statistics (<exp(-Delta H)> is a separate run), single rank.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFRhmcFermionActions.h"
#include "../experimental/mdwf/MDWFSpectralBounds.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

template<size_t HaloDepth>
double mdwfTwoPlusOneTestMaxLinkDiff(CommunicationBase &commBase,
                                     Gaugefield<double, true, HaloDepth, R18> &a,
                                     Gaugefield<double, true, HaloDepth, R18> &b) {
    typedef GIndexer<All, HaloDepth> GInd;
    Gaugefield<double, false, HaloDepth, R18> aHost(commBase, "MDWF_2p1_test_diff_a");
    Gaugefield<double, false, HaloDepth, R18> bHost(commBase, "MDWF_2p1_test_diff_b");
    aHost = a;
    bHost = b;
    const SU3Accessor<double, R18> aAcc = aHost.getAccessor();
    const SU3Accessor<double, R18> bAcc = bHost.getAccessor();
    double maxDiff = 0.0;
    for (size_t siteIndex = 0; siteIndex < GInd::getLatData().vol4; siteIndex++) {
        const gSite site = GInd::getSite(siteIndex);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const gSiteMu siteMu = GInd::getSiteMu(site, mu);
            maxDiff = std::max(maxDiff, static_cast<double>(infnorm(aAcc.getLink(siteMu) - bAcc.getLink(siteMu))));
        }
    }
    return maxDiff;
}

void logMDWFTwoPlusOneEnergy(const std::string &label, const MDWFHmcEnergy &energy) {
    rootLogger.info("MDWF 2+1 HMC trajectory test ", label, ": kinetic = ", energy.kinetic,
                    ", gauge = ", energy.gauge, ", fermion = ", energy.fermion, ", total = ", energy.total());
}

template<size_t HaloDepth, size_t Ls>
MDWFLanczosCheckpoint mdwfTwoPlusOneTestBounds(CommunicationBase &commBase,
                                               Gaugefield<double, true, HaloDepth, R18> &gauge,
                                               const MDWFHmcParameters &param, double mass, int steps,
                                               uint4 *randState, const std::string &name) {
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
        rootLogger.info("MDWF 2+1 HMC trajectory test Lanczos (mass = ", mass, "): steps = ", c.steps,
                        ", lambda_min = ", c.lambda_min, ", lambda_max = ", c.lambda_max,
                        result.breakdown ? " (breakdown: invariant subspace, exact)" : "");
    }
    return result.checkpoints.back();
}

template<size_t Ls>
void runMDWFTwoPlusOneHmcTrajectoryTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using Hmc = MDWFTwoPlusOneHmc<HaloDepth, Ls>;

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
    h_rand.make_rng_state(20260928);
    d_rand = h_rand;

    Gauge gauge(commBase, "MDWF_2p1_test_gauge");
    gauge.one();
    gauge.updateAll();

    const MDWFLanczosCheckpoint boundsS = mdwfTwoPlusOneTestBounds<HaloDepth, Ls>(
        commBase, gauge, param, param.rhmc.ms, 400, d_rand.state, "MDWF_2p1_test_bounds_s");
    const MDWFLanczosCheckpoint boundsPv = mdwfTwoPlusOneTestBounds<HaloDepth, Ls>(
        commBase, gauge, param, param.pv_mass, 200, d_rand.state, "MDWF_2p1_test_bounds_pv");
    param.rhmc.lambda_low_s = boundsS.lambda_min / 3.0;
    param.rhmc.lambda_high_s = 1.5 * boundsS.lambda_max;
    param.rhmc.lambda_low_pv = boundsPv.lambda_min / 3.0;
    param.rhmc.lambda_high_pv = 1.5 * boundsPv.lambda_max;

    rootLogger.info("MDWF 2+1 HMC trajectory test: Wilson beta = ", param.beta, ", M5 = ", param.M5,
                    ", mf = ", param.mf, ", ms = ", param.rhmc.ms, ", pv_mass = ", param.pv_mass,
                    ", b5 = ", param.b5, ", c_sw = ", param.csw, ", tau = ", param.tau, ", Ls = ", Ls,
                    ", solver precision = ", param.precision, ", strange intervals [", param.rhmc.lambda_low_s,
                    ", ", param.rhmc.lambda_high_s, "] and [", param.rhmc.lambda_low_pv, ", ",
                    param.rhmc.lambda_high_pv, "], unit gauge start");

    Hmc hmc(commBase, gauge, param, d_rand.state);
    auto &light = hmc.fermion().first();
    auto &strange = hmc.fermion().second();
    rootLogger.info("MDWF 2+1 HMC trajectory test strange approximations: A^(1/4) order ", strange.quarterS().order,
                    ", A^(-1/2) order ", strange.halfS().order, " (force ", strange.halfSForce().order,
                    "), B^(1/4) order ", strange.quarterPv().order, " (force ", strange.quarterPvForce().order, ")");

    // --- Part 1: heatbath identity, per pseudofermion. ---
    hmc.refreshMomenta();
    hmc.heatbath();
    const double lightAction = light.action();
    const double strangeAction = strange.action();
    const double lightRelDiff = std::abs(lightAction - light.noiseNorm2()) / std::max(1.0, light.noiseNorm2());
    const double strangeRelDiff = std::abs(strangeAction - strange.noiseNorm2()) / std::max(1.0, strange.noiseNorm2());
    const bool heatbathPassed = lightRelDiff <= 1e-6 && strangeRelDiff <= 1e-6;
    rootLogger.info("MDWF 2+1 HMC trajectory test heatbath: light S = ", lightAction, ", eta^dagger eta = ",
                    light.noiseNorm2(), ", relDiff = ", lightRelDiff, "; strange S = ", strangeAction,
                    ", eta^dagger eta = ", strange.noiseNorm2(), ", relDiff = ", strangeRelDiff,
                    ", passed = ", heatbathPassed);

    Gauge startGauge(commBase, "MDWF_2p1_test_start_gauge");
    Gauge startMomenta(commBase, "MDWF_2p1_test_start_momenta");
    startGauge = gauge;
    startMomenta = hmc.momenta();

    // --- Part 2: reversibility. ---
    const MDWFHmcEnergy start = hmc.energy();
    logMDWFTwoPlusOneEnergy("start", start);
    hmc.integrate(param.steps);
    const MDWFHmcEnergy forward = hmc.energy();
    logMDWFTwoPlusOneEnergy("after forward trajectory", forward);
    hmc.flipMomenta();
    hmc.integrate(param.steps);
    hmc.flipMomenta();
    const MDWFHmcEnergy back = hmc.energy();
    logMDWFTwoPlusOneEnergy("after reversed trajectory", back);

    const double reverseLinkDiff = mdwfTwoPlusOneTestMaxLinkDiff<HaloDepth>(commBase, gauge, startGauge);
    const double reverseMomentumDiff = mdwfTwoPlusOneTestMaxLinkDiff<HaloDepth>(commBase, hmc.momenta(), startMomenta);
    const double reverseEnergyDiff = std::abs(back.total() - start.total());
    const double forwardDeltaH = forward.total() - start.total();
    const bool reversibilityPassed = reverseLinkDiff <= 1e-8 && reverseMomentumDiff <= 1e-8
                                     && reverseEnergyDiff <= 1e-6;
    rootLogger.info("MDWF 2+1 HMC trajectory test reversibility (", param.steps, " steps): forward Delta H = ",
                    forwardDeltaH, ", max link diff = ", reverseLinkDiff,
                    ", max momentum diff = ", reverseMomentumDiff,
                    ", |H_back - H_start| = ", reverseEnergyDiff, ", passed = ", reversibilityPassed);
    rootLogger.info("MDWF 2+1 HMC trajectory test force strength (forward + reversed): max rms gauge force = ",
                    hmc.maxGaugeForceRms(), ", max rms fermion force (light + strange) = ", hmc.maxFermionForceRms());

    // --- Part 3: Delta H scaling from the identical start state. ---
    const std::array<int, 3> stepCounts{{8, 16, 32}};
    std::array<double, 3> deltaH{};
    double maxStartEnergyDiff = 0.0;
    for (size_t i = 0; i < stepCounts.size(); i++) {
        gauge = startGauge;
        gauge.updateAll();
        hmc.momenta() = startMomenta;
        hmc.momenta().updateAll();
        const MDWFHmcEnergy restored = hmc.energy();
        maxStartEnergyDiff = std::max(maxStartEnergyDiff,
                                      std::abs(restored.total() - start.total()) / std::max(1.0, std::abs(start.total())));
        hmc.integrate(stepCounts[i]);
        deltaH[i] = hmc.energy().total() - restored.total();
        rootLogger.info("MDWF 2+1 HMC trajectory test scaling: steps = ", stepCounts[i],
                        ", eps = ", param.tau / stepCounts[i], ", Delta H = ", deltaH[i]);
    }
    const double ratioCoarse = std::abs(deltaH[0]) / std::max(std::abs(deltaH[1]), 1e-300);
    const double ratioFine = std::abs(deltaH[1]) / std::max(std::abs(deltaH[2]), 1e-300);
    const bool scalingPassed = maxStartEnergyDiff <= 1e-9
                               && std::abs(deltaH[0]) > std::abs(deltaH[1])
                               && std::abs(deltaH[1]) > std::abs(deltaH[2])
                               && ratioCoarse >= 3.0 && ratioCoarse <= 6.0
                               && ratioFine >= 3.0 && ratioFine <= 6.0;
    rootLogger.info("MDWF 2+1 HMC trajectory test scaling: |dH(8)/dH(16)| = ", ratioCoarse,
                    ", |dH(16)/dH(32)| = ", ratioFine, " (leapfrog expects 4), restored start energy relDiff = ",
                    maxStartEnergyDiff, ", passed = ", scalingPassed);

    rootLogger.info("MDWF 2+1 HMC trajectory test: total force evaluations = ", hmc.forceEvaluations(),
                    ", max rms gauge force = ", hmc.maxGaugeForceRms(),
                    ", max rms fermion force = ", hmc.maxFermionForceRms());

    if (!heatbathPassed || !reversibilityPassed || !scalingPassed) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF 2+1 HMC trajectory test failed: heatbath passed = ", heatbathPassed,
            ", reversibility passed = ", reversibilityPassed, ", scaling passed = ", scalingPassed,
            " (see diagnostics above)"));
    }
    rootLogger.info("MDWF 2+1 HMC trajectory test passed with Ls = ", Ls);
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

        runMDWFTwoPlusOneHmcTrajectoryTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
