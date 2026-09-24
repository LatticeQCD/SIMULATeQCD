/*
 * MDWF two-flavour HMC driver test (step 2 of the MDWF RHMC plan).
 *
 * Exercises MDWFTwoFlavorHmc (MDWFHmc.h): Wilson gauge action plus the Mobius
 * clover two-flavour pseudofermion S_f = phi^\dagger (M^\dagger M)^{-1} phi,
 * leapfrog with U -> exp(i eps P) U, P -> P - i eps ipdot, ipdot_f = stored
 * all-link MDWF matrices. Starting from a unit gauge field on 6^4, Ls = 8:
 *
 *   1. Heatbath identity: right after phi = M^\dagger eta, S_f = eta^\dagger eta
 *      (relative 1e-8).
 *   2. Reversibility: integrate tau = 0.2 in 8 steps, flip the momenta,
 *      integrate back, flip again: gauge links and momenta return to their
 *      start (max element difference 1e-8) and H returns to its start
 *      (absolute 1e-6).
 *   3. Delta H scaling: from the identical start state, 8, 16, and 32
 *      leapfrog steps (eps = 0.025 ... 0.00625) must give |Delta H| decreasing
 *      by a factor between 3 and 6 per halving of eps (leapfrog Delta H is
 *      O(eps^2)), and restoring the start state must reproduce the start energy
 *      (relative 1e-9).
 *
 * A first run with tau = 0.5 and 4/8/16 steps showed leapfrog instability
 * above eps ~ 0.06 (Delta H ~ 1e6-1e8) and Delta H = 143 at eps = 0.05: the
 * bare 5D determinant (no Pauli-Villars factor yet) is a stiff force, so the
 * eps^2 regime needs much smaller steps. The driver's maximum rms gauge and
 * fermion forces are reported to quantify this.
 *
 * No Metropolis statistics yet (<exp(-Delta H)> is step 3), no Pauli-Villars
 * factor, single rank only.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFHmc.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

template<size_t HaloDepth>
double mdwfHmcTestMaxLinkDiff(CommunicationBase &commBase,
                              Gaugefield<double, true, HaloDepth, R18> &a,
                              Gaugefield<double, true, HaloDepth, R18> &b) {
    typedef GIndexer<All, HaloDepth> GInd;
    Gaugefield<double, false, HaloDepth, R18> aHost(commBase, "MDWF_hmc_test_diff_a");
    Gaugefield<double, false, HaloDepth, R18> bHost(commBase, "MDWF_hmc_test_diff_b");
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

void logMDWFHmcEnergy(const std::string &label, const MDWFHmcEnergy &energy) {
    rootLogger.info("MDWF HMC trajectory test ", label, ": kinetic = ", energy.kinetic,
                    ", gauge = ", energy.gauge, ", fermion = ", energy.fermion, ", total = ", energy.total());
}

template<size_t Ls>
void runMDWFHmcTrajectoryTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using Hmc = MDWFTwoFlavorHmc<HaloDepth, Ls>;

    MDWFHmcParameters param{};
    param.beta = 6.0;
    param.M5 = 1.8;
    param.mf = 0.1;
    param.b5 = 1.5;
    param.csw = 0.5;
    param.tau = 0.2;
    param.steps = 8;
    param.max_iter = 20000;
    param.precision = 1e-10;

    rootLogger.info("MDWF HMC trajectory test: Wilson beta = ", param.beta, ", M5 = ", param.M5, ", mf = ", param.mf,
                    ", b5 = ", param.b5, ", c_sw = ", param.csw, ", tau = ", param.tau, ", Ls = ", Ls,
                    ", solver precision = ", param.precision, ", unit gauge start");

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260923);
    d_rand = h_rand;

    Gauge gauge(commBase, "MDWF_hmc_test_gauge");
    gauge.one();
    gauge.updateAll();

    Hmc hmc(commBase, gauge, param, d_rand.state);

    // --- Part 1: heatbath identity. ---
    hmc.refreshMomenta();
    hmc.heatbath();
    const double startFermionAction = hmc.fermionAction();
    const double heatbathRelDiff = std::abs(startFermionAction - hmc.noiseNorm2())
                                   / std::max(1.0, hmc.noiseNorm2());
    const bool heatbathPassed = heatbathRelDiff <= 1e-8;
    rootLogger.info("MDWF HMC trajectory test heatbath: S_f = ", startFermionAction,
                    ", eta^dagger eta = ", hmc.noiseNorm2(), ", relDiff = ", heatbathRelDiff,
                    ", passed = ", heatbathPassed);

    Gauge startGauge(commBase, "MDWF_hmc_test_start_gauge");
    Gauge startMomenta(commBase, "MDWF_hmc_test_start_momenta");
    startGauge = gauge;
    startMomenta = hmc.momenta();

    // --- Part 2: reversibility. ---
    const MDWFHmcEnergy start = hmc.energy();
    logMDWFHmcEnergy("start", start);
    hmc.integrate(param.steps);
    const MDWFHmcEnergy forward = hmc.energy();
    logMDWFHmcEnergy("after forward trajectory", forward);
    hmc.flipMomenta();
    hmc.integrate(param.steps);
    hmc.flipMomenta();
    const MDWFHmcEnergy back = hmc.energy();
    logMDWFHmcEnergy("after reversed trajectory", back);

    const double reverseLinkDiff = mdwfHmcTestMaxLinkDiff<HaloDepth>(commBase, gauge, startGauge);
    const double reverseMomentumDiff = mdwfHmcTestMaxLinkDiff<HaloDepth>(commBase, hmc.momenta(), startMomenta);
    const double reverseEnergyDiff = std::abs(back.total() - start.total());
    const double forwardDeltaH = forward.total() - start.total();
    const bool reversibilityPassed = reverseLinkDiff <= 1e-8 && reverseMomentumDiff <= 1e-8
                                     && reverseEnergyDiff <= 1e-6;
    rootLogger.info("MDWF HMC trajectory test reversibility (", param.steps, " steps): forward Delta H = ",
                    forwardDeltaH, ", max link diff = ", reverseLinkDiff,
                    ", max momentum diff = ", reverseMomentumDiff,
                    ", |H_back - H_start| = ", reverseEnergyDiff, ", passed = ", reversibilityPassed);
    rootLogger.info("MDWF HMC trajectory test force strength (forward + reversed): max rms gauge force = ",
                    hmc.maxGaugeForceRms(), ", max rms fermion force = ", hmc.maxFermionForceRms());

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
        rootLogger.info("MDWF HMC trajectory test scaling: steps = ", stepCounts[i],
                        ", eps = ", param.tau / stepCounts[i], ", Delta H = ", deltaH[i]);
    }
    const double ratioCoarse = std::abs(deltaH[0]) / std::max(std::abs(deltaH[1]), 1e-300);
    const double ratioFine = std::abs(deltaH[1]) / std::max(std::abs(deltaH[2]), 1e-300);
    const bool scalingPassed = maxStartEnergyDiff <= 1e-9
                               && std::abs(deltaH[0]) > std::abs(deltaH[1])
                               && std::abs(deltaH[1]) > std::abs(deltaH[2])
                               && ratioCoarse >= 3.0 && ratioCoarse <= 6.0
                               && ratioFine >= 3.0 && ratioFine <= 6.0;
    rootLogger.info("MDWF HMC trajectory test scaling: |dH(8)/dH(16)| = ", ratioCoarse,
                    ", |dH(16)/dH(32)| = ", ratioFine, " (leapfrog expects 4), restored start energy relDiff = ",
                    maxStartEnergyDiff, ", passed = ", scalingPassed);

    rootLogger.info("MDWF HMC trajectory test: total force evaluations = ", hmc.forceEvaluations(),
                    ", max rms gauge force = ", hmc.maxGaugeForceRms(),
                    ", max rms fermion force = ", hmc.maxFermionForceRms());

    if (!heatbathPassed || !reversibilityPassed || !scalingPassed) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF HMC trajectory test failed: heatbath passed = ", heatbathPassed,
            ", reversibility passed = ", reversibilityPassed, ", scaling passed = ", scalingPassed,
            " (see diagnostics above)"));
    }
    rootLogger.info("MDWF HMC trajectory test passed with Ls = ", Ls);
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

        runMDWFHmcTrajectoryTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
