/*
 * MDWF Pauli-Villars HMC two-scale (Sexton-Weingarten) integrator test.
 *
 * Exercises MDWFHmcDriver::integrate(steps, gaugeSubsteps) with the Pauli-Villars
 * two-flavour action: fermion step eps = tau / steps, gauge step delta = eps /
 * gaugeSubsteps, as SWleapfrog in src/modules/rhmc/integrator.cpp. Seed,
 * parameters (Wilson beta = 6, M5 = 1.8, mf = 0.1, pv_mass = 1, b5 = 1.5,
 * c_sw = 0.5, tau = 0.2), unit start on 6^4 with Ls = 8, and draw order are
 * identical to mdwfPauliVillarsHmcTrajectoryTest, whose plain leapfrog gave
 * Delta H = 25.445, 6.3415, 1.58414 for 8, 16, 32 steps.
 *
 *   1. Reversibility with 8 fermion steps and 4 gauge substeps: links and
 *      momenta return within 1e-8 and H within 1e-6.
 *   2. Delta H scaling at 4 gauge substeps for 4, 8, 16 fermion steps: ratios
 *      between 3 and 6 per halving (both eps and delta halve, so O(eps^2)).
 *   3. Consistency: 1 gauge substep reproduces the plain leapfrog Delta H =
 *      25.445 at 8 steps (absolute 1e-3; the integrators are operation-by-
 *      operation identical there).
 *   4. Benefit: at 8 fermion steps, Delta H for 1, 2, 4, 8 gauge substeps is
 *      reported with the fermion-force and gauge-update counts; 4 substeps must
 *      give a smaller |Delta H| than 1 substep.
 *
 * Single rank; no Metropolis statistics.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFHmc.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

template<size_t HaloDepth>
double mdwfMultiscaleMaxLinkDiff(CommunicationBase &commBase,
                                 Gaugefield<double, true, HaloDepth, R18> &a,
                                 Gaugefield<double, true, HaloDepth, R18> &b) {
    typedef GIndexer<All, HaloDepth> GInd;
    Gaugefield<double, false, HaloDepth, R18> aHost(commBase, "MDWF_multiscale_diff_a");
    Gaugefield<double, false, HaloDepth, R18> bHost(commBase, "MDWF_multiscale_diff_b");
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

template<size_t Ls>
void runMDWFPauliVillarsMultiscaleTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using Hmc = MDWFPauliVillarsTwoFlavorHmc<HaloDepth, Ls>;

    MDWFHmcParameters param{};
    param.beta = 6.0;
    param.M5 = 1.8;
    param.mf = 0.1;
    param.b5 = 1.5;
    param.csw = 0.5;
    param.pv_mass = 1.0;
    param.tau = 0.2;
    param.steps = 8;
    param.gauge_substeps = 4;
    param.max_iter = 20000;
    param.precision = 1e-10;
    const double plainLeapfrogDeltaH8 = 25.445;

    rootLogger.info("MDWF multiscale test: Wilson beta = ", param.beta, ", M5 = ", param.M5, ", mf = ", param.mf,
                    ", pv_mass = ", param.pv_mass, ", b5 = ", param.b5, ", c_sw = ", param.csw, ", tau = ", param.tau,
                    ", Ls = ", Ls, ", solver precision = ", param.precision, ", unit gauge start");

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260923);
    d_rand = h_rand;

    Gauge gauge(commBase, "MDWF_multiscale_gauge");
    gauge.one();
    gauge.updateAll();

    Hmc hmc(commBase, gauge, param, d_rand.state);
    hmc.refreshMomenta();
    hmc.heatbath();

    Gauge startGauge(commBase, "MDWF_multiscale_start_gauge");
    Gauge startMomenta(commBase, "MDWF_multiscale_start_momenta");
    startGauge = gauge;
    startMomenta = hmc.momenta();
    const MDWFHmcEnergy start = hmc.energy();
    rootLogger.info("MDWF multiscale test start: kinetic = ", start.kinetic, ", gauge = ", start.gauge,
                    ", fermion = ", start.fermion, ", total = ", start.total());

    struct RunResult {
        double deltaH;
        double startEnergyRelDiff;
        int fermionForces;
        int gaugeUpdates;
    };
    auto runFromStart = [&](int steps, int substeps) {
        gauge = startGauge;
        gauge.updateAll();
        hmc.momenta() = startMomenta;
        hmc.momenta().updateAll();
        const MDWFHmcEnergy restored = hmc.energy();
        const int fermionBefore = hmc.forceEvaluations();
        const int gaugeBefore = hmc.gaugeUpdates();
        hmc.integrate(steps, substeps);
        const double deltaH = hmc.energy().total() - restored.total();
        const RunResult result{deltaH,
                               std::abs(restored.total() - start.total()) / std::max(1.0, std::abs(start.total())),
                               hmc.forceEvaluations() - fermionBefore,
                               hmc.gaugeUpdates() - gaugeBefore};
        rootLogger.info("MDWF multiscale test: fermion steps = ", steps, ", gauge substeps = ", substeps,
                        ", eps = ", param.tau / steps, ", delta = ", param.tau / (steps * substeps),
                        ", Delta H = ", result.deltaH, ", fermion forces = ", result.fermionForces,
                        ", gauge updates = ", result.gaugeUpdates);
        return result;
    };

    // --- Part 1: reversibility with 8 fermion steps and 4 gauge substeps. ---
    hmc.integrate(param.steps, param.gauge_substeps);
    const double forwardDeltaH = hmc.energy().total() - start.total();
    hmc.flipMomenta();
    hmc.integrate(param.steps, param.gauge_substeps);
    hmc.flipMomenta();
    const MDWFHmcEnergy back = hmc.energy();
    const double reverseLinkDiff = mdwfMultiscaleMaxLinkDiff<HaloDepth>(commBase, gauge, startGauge);
    const double reverseMomentumDiff = mdwfMultiscaleMaxLinkDiff<HaloDepth>(commBase, hmc.momenta(), startMomenta);
    const double reverseEnergyDiff = std::abs(back.total() - start.total());
    const bool reversibilityPassed = reverseLinkDiff <= 1e-8 && reverseMomentumDiff <= 1e-8
                                     && reverseEnergyDiff <= 1e-6;
    rootLogger.info("MDWF multiscale test reversibility (8 fermion steps, 4 gauge substeps): forward Delta H = ",
                    forwardDeltaH, ", max link diff = ", reverseLinkDiff, ", max momentum diff = ", reverseMomentumDiff,
                    ", |H_back - H_start| = ", reverseEnergyDiff, ", passed = ", reversibilityPassed);

    // --- Part 2: Delta H scaling at 4 gauge substeps. ---
    const std::array<int, 3> stepCounts{{4, 8, 16}};
    std::array<RunResult, 3> scaling{};
    double maxStartEnergyDiff = 0.0;
    for (size_t i = 0; i < stepCounts.size(); i++) {
        scaling[i] = runFromStart(stepCounts[i], 4);
        maxStartEnergyDiff = std::max(maxStartEnergyDiff, scaling[i].startEnergyRelDiff);
    }
    const double ratioCoarse = std::abs(scaling[0].deltaH) / std::max(std::abs(scaling[1].deltaH), 1e-300);
    const double ratioFine = std::abs(scaling[1].deltaH) / std::max(std::abs(scaling[2].deltaH), 1e-300);
    const bool scalingPassed = maxStartEnergyDiff <= 1e-9
                               && ratioCoarse >= 3.0 && ratioCoarse <= 6.0
                               && ratioFine >= 3.0 && ratioFine <= 6.0;
    rootLogger.info("MDWF multiscale test scaling (4 gauge substeps): |dH(4)/dH(8)| = ", ratioCoarse,
                    ", |dH(8)/dH(16)| = ", ratioFine, " (expects 4), restored start energy relDiff = ",
                    maxStartEnergyDiff, ", passed = ", scalingPassed);

    // --- Parts 3 and 4: gauge substep scan at 8 fermion steps. ---
    const RunResult sub1 = runFromStart(8, 1);
    const RunResult sub2 = runFromStart(8, 2);
    const RunResult &sub4 = scaling[1];
    const RunResult sub8 = runFromStart(8, 8);
    const double consistencyDiff = std::abs(sub1.deltaH - plainLeapfrogDeltaH8);
    const bool consistencyPassed = consistencyDiff <= 1e-3;
    const bool benefitPassed = std::abs(sub4.deltaH) < std::abs(sub1.deltaH);
    rootLogger.info("MDWF multiscale test consistency: 1 substep Delta H = ", sub1.deltaH,
                    " versus plain leapfrog ", plainLeapfrogDeltaH8, ", |diff| = ", consistencyDiff,
                    ", passed = ", consistencyPassed);
    rootLogger.info("MDWF multiscale test substep scan (8 fermion steps, 9 fermion forces each): Delta H = ",
                    sub1.deltaH, " (1), ", sub2.deltaH, " (2), ", sub4.deltaH, " (4), ", sub8.deltaH,
                    " (8); plain leapfrog with 32 steps (33 fermion forces) gave 1.58414; passed = ", benefitPassed);
    rootLogger.info("MDWF multiscale test: total fermion forces = ", hmc.forceEvaluations(),
                    ", total gauge updates = ", hmc.gaugeUpdates(),
                    ", max rms gauge force = ", hmc.maxGaugeForceRms(),
                    ", max rms fermion force = ", hmc.maxFermionForceRms());

    if (!reversibilityPassed || !scalingPassed || !consistencyPassed || !benefitPassed) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF multiscale test failed: reversibility passed = ", reversibilityPassed,
            ", scaling passed = ", scalingPassed, ", consistency passed = ", consistencyPassed,
            ", benefit passed = ", benefitPassed, " (see diagnostics above)"));
    }
    rootLogger.info("MDWF multiscale test passed with Ls = ", Ls);
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

        runMDWFPauliVillarsMultiscaleTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
