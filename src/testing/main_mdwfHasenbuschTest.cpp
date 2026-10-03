/*
 * Even/odd Hasenbusch ladder of the two-flavour light determinant ratio
 * (MDWFEvenOddHasenbuschTwoFlavorFermionAction, MDWFEvenOddFermionActions.h).
 *
 * 6^4, Ls = 8, M5 = 1.8, b5 = 1.5, mf = 0.1, pv_mass = 1, Wilson gauge action beta = 6, solver precision 1e-10.
 *
 *   A. No intermediate masses: the ladder is the single ratio. With the same random-number state it must give the
 *      Pauli-Villars action's action and force (c_sw = 0.5), to 1e-12 relative (the same arithmetic; exact equality
 *      expected).
 *   B. Ladder 0.1 < 0.3 < 0.6 < 1 (three factors), c_sw = 0.5, on a nontrivial configuration (two trajectories
 *      from the unit start):
 *      1. heatbath identity per factor (pseudofermion action = noise norm, 1e-6);
 *      2. the clover log det W terms telescope: sum over factors = 2 [log det W(mf) - log det W(pv)] of the single
 *         ratio (relative 1e-12);
 *      3. finite-difference force check of the whole ladder, dS/dt along exp(i t P) U (h = 1e-4) against
 *         i sum tr(P ipdot), relative 1e-5, and of each factor alone (forceFactor);
 *      4. Delta H ratios between 3 and 6 for 8/16/32 steps (tau = 0.2).
 *   C. Multi-level integrator (MDWFHmcDriver::integrateMultiLevel), same ladder:
 *      1. one level, all terms on it, gauge_substeps 1: the same trajectory as the two-scale integrator (Delta H to
 *         1e-8, final links to 1e-10; only the merged momentum kicks are split in halves);
 *      2. two levels, factor 0 (lightest) on level 0, factors 1 and 2 on level 1 with 2 substeps, one gauge step
 *         per level-1 step:
 *         force evaluations per trajectory n0 + 1 (level 0) and 2 n0 + 1 (level 1), reversibility over n0 = 4 steps
 *         (links and momenta 1e-8, |H_back - H_start| 1e-6), Delta H ratios between 3 and 6 for n0 = 4/8/16.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFEvenOddFermionActions.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>

template<size_t HaloDepth>
double mdwfHbMaxLinkDiff(CommunicationBase &commBase, Gaugefield<double, false, HaloDepth, R18> &a,
                         Gaugefield<double, false, HaloDepth, R18> &b) {
    typedef GIndexer<All, HaloDepth> GInd;
    const SU3Accessor<double, R18> aAcc = a.getAccessor();
    const SU3Accessor<double, R18> bAcc = b.getAccessor();
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

// i sum tr(P K) for momenta P and a host force K (dS/dt along exp(i t P) U, MDWF_HMC_CONVENTIONS.md).
template<size_t HaloDepth>
double mdwfHbAlongMomenta(CommunicationBase &commBase, Gaugefield<double, true, HaloDepth, R18> &momenta,
                          Gaugefield<double, false, HaloDepth, R18> &force) {
    typedef GIndexer<All, HaloDepth> GInd;
    Gaugefield<double, false, HaloDepth, R18> pHost(commBase, "MDWF_hb_pmom_host");
    pHost = momenta;
    const SU3Accessor<double, R18> pAcc = pHost.getAccessor();
    const SU3Accessor<double, R18> fAcc = force.getAccessor();
    COMPLEX(double) sum(0.0, 0.0);
    for (size_t i = 0; i < GInd::getLatData().vol4; i++) {
        const gSite site = GInd::getSite(i);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const gSiteMu siteMu = GInd::getSiteMu(site, mu);
            sum += tr_c(pAcc.getLink(siteMu), fAcc.getLink(siteMu));
        }
    }
    return real(COMPLEX(double)(0.0, 1.0) * sum);
}

template<size_t Ls>
void runMDWFHasenbuschTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using Ladder = MDWFEvenOddHasenbuschTwoFlavorFermionAction<HaloDepth, Ls>;
    using Pv = MDWFEvenOddPauliVillarsTwoFlavorFermionAction<HaloDepth, Ls>;
    using LadderHmc = MDWFEvenOddHasenbuschTwoFlavorHmc<HaloDepth, Ls>;

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

    Gauge gauge(commBase, "MDWF_hb_gauge");
    gauge.one();
    gauge.updateAll();

    // Nontrivial configuration: two always-accepted trajectories of the single-ratio HMC from the unit start.
    {
        grnd_state<false> h_rand;
        grnd_state<true> d_rand;
        h_rand.make_rng_state(20261003);
        d_rand = h_rand;
        MDWFEvenOddPauliVillarsTwoFlavorHmc<HaloDepth, Ls> warm(commBase, gauge, param, d_rand.state);
        std::mt19937_64 acceptRng(1UL);
        for (int traj = 0; traj < 2; traj++) {
            const MDWFHmcTrajectoryResult r = warm.trajectory(false, acceptRng);
            GaugeAction<double, true, HaloDepth, R18> action(gauge);
            rootLogger.info("MDWF Hasenbusch test start trajectory ", traj + 1, ": Delta H = ", r.delta_h,
                            ", plaquette = ", action.plaquette());
        }
    }
    HostGauge gaugeHost(commBase, "MDWF_hb_gauge_host");
    gaugeHost = gauge;

    // --- A. Ladder without intermediate masses = the single ratio, exactly. ---
    bool identityPassed = false;
    {
        grnd_state<false> h_rand;
        grnd_state<true> dA, dB;
        h_rand.make_rng_state(777);
        dA = h_rand;
        dB = h_rand;
        Ladder ladder(commBase, gauge, param, "MDWF_hb_lad1");
        Pv pv(commBase, gauge, param, "MDWF_hb_pv1");
        ladder.heatbath(dA.state);
        pv.heatbath(dB.state);
        const double sLadder = ladder.action();
        const double sPv = pv.action();
        HostGauge fLadder(commBase, "MDWF_hb_flad1");
        HostGauge fPv(commBase, "MDWF_hb_fpv1");
        ladder.force(fLadder, gaugeHost);
        pv.force(fPv, gaugeHost);
        const double forceDiff = mdwfHbMaxLinkDiff<HaloDepth>(commBase, fLadder, fPv);
        identityPassed = ladder.factorCount() == 1 && std::abs(sLadder - sPv) <= 1e-12 * std::abs(sPv)
                         && forceDiff <= 1e-12;
        rootLogger.info("MDWF Hasenbusch test A (no intermediate masses): factors = ", ladder.factorCount(),
                        ", action ladder ", sLadder, " single ratio ", sPv, " (difference ", sLadder - sPv,
                        "), max force difference ", forceDiff, ", passed = ", identityPassed);
    }

    // --- B. Three-factor ladder. ---
    param.hasenbusch_masses = {0.3, 0.6};
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20261004);
    d_rand = h_rand;
    LadderHmc hmc(commBase, gauge, param, d_rand.state);
    Ladder &ladder = hmc.fermion();
    rootLogger.info("MDWF Hasenbusch test B: ", ladder.factorCount(), " factors, masses ", ladder.masses()[0], " ",
                    ladder.masses()[1], " ", ladder.masses()[2], " ", ladder.masses()[3]);

    // 1. Heatbath identity per factor.
    hmc.refreshMomenta();
    hmc.heatbath();
    bool heatbathPassed = ladder.factorCount() == 3;
    for (size_t i = 0; i < ladder.factorCount(); i++) {
        auto &f = ladder.factor(i);
        f.action();
        const double rel = std::abs(f.lastPseudofermionAction() - f.noiseNorm2()) / std::max(1.0, f.noiseNorm2());
        heatbathPassed = heatbathPassed && rel <= 1e-6;
        rootLogger.info("MDWF Hasenbusch test B1 factor ", i, " (", f.massNumerator(), " / ", f.massDenominator(),
                        "): pseudofermion action ", f.lastPseudofermionAction(), ", noise ", f.noiseNorm2(),
                        ", relative difference ", rel);
    }

    // 2. Clover log det W terms telescope.
    ladder.action();
    Pv single(commBase, gauge, param, "MDWF_hb_pvtel");
    {
        grnd_state<false> hr;
        grnd_state<true> dr;
        hr.make_rng_state(5);
        dr = hr;
        single.heatbath(dr.state);
    }
    single.action();
    const double detLadder = ladder.lastDetAction();
    const double detSingle = single.lastDetAction();
    const double detRel = std::abs(detLadder - detSingle) / std::max(std::abs(detSingle), 1e-300);
    const bool telescopePassed = detRel <= 1e-12 && detSingle != 0.0;
    rootLogger.info("MDWF Hasenbusch test B2 clover det terms: ladder sum ", detLadder, ", single ratio ", detSingle,
                    ", relative difference ", detRel, ", passed = ", telescopePassed);

    // 3. Finite-difference force, whole ladder and each factor.
    const double h = 1e-4;
    Gauge saved(commBase, "MDWF_hb_saved");
    auto numerical = [&](const std::function<double()> &action) {
        saved = gauge;
        hmc.evolveQ(h);
        const double plus = action();
        gauge = saved;
        gauge.updateAll();
        hmc.evolveQ(-h);
        const double minus = action();
        gauge = saved;
        gauge.updateAll();
        return (plus - minus) / (2.0 * h);
    };
    bool forcePassed = true;
    {
        const double analytic = hmc.fermionForceAlongMomenta();
        const double num = numerical([&]() { return hmc.fermionAction(); });
        const double rel = std::abs(num - analytic) / std::max(std::abs(analytic), 1e-300);
        forcePassed = forcePassed && rel <= 1e-5;
        rootLogger.info("MDWF Hasenbusch test B3 whole ladder: dS/dt numerical ", num, ", from the force ", analytic,
                        ", relative difference ", rel);
    }
    HostGauge fPart(commBase, "MDWF_hb_fpart");
    for (size_t i = 0; i < ladder.factorCount(); i++) {
        gaugeHost = gauge;
        ladder.forceFactor(i, fPart, gaugeHost);
        const double analytic = mdwfHbAlongMomenta<HaloDepth>(commBase, hmc.momenta(), fPart);
        const double num = numerical([&]() { return ladder.factor(i).action(); });
        const double rel = std::abs(num - analytic) / std::max(std::abs(analytic), 1e-300);
        forcePassed = forcePassed && rel <= 1e-5;
        rootLogger.info("MDWF Hasenbusch test B3 factor ", i, ": dS/dt numerical ", num, ", from the force ", analytic,
                        ", relative difference ", rel);
    }
    rootLogger.info("MDWF Hasenbusch test B3 passed = ", forcePassed);

    // 4. Delta H scaling.
    hmc.refreshMomenta();
    hmc.heatbath();
    Gauge startGauge(commBase, "MDWF_hb_start_gauge");
    Gauge startMomenta(commBase, "MDWF_hb_start_momenta");
    startGauge = gauge;
    startMomenta = hmc.momenta();
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
        rootLogger.info("MDWF Hasenbusch test B4 scaling: steps = ", stepCounts[i], ", Delta H = ", deltaH[i]);
    }
    const double r1 = std::abs(deltaH[0]) / std::max(std::abs(deltaH[1]), 1e-300);
    const double r2 = std::abs(deltaH[1]) / std::max(std::abs(deltaH[2]), 1e-300);
    const bool scalingPassed = r1 >= 3.0 && r1 <= 6.0 && r2 >= 3.0 && r2 <= 6.0;
    rootLogger.info("MDWF Hasenbusch test B4 scaling: ratios ", r1, ", ", r2, " (leapfrog expects 4), passed = ",
                    scalingPassed);

    // --- C. Multi-level integrator. ---
    HostGauge linksA(commBase, "MDWF_hb_links_a");
    HostGauge linksB(commBase, "MDWF_hb_links_b");
    auto restoreStart = [&]() {
        gauge = startGauge;
        gauge.updateAll();
        hmc.momenta() = startMomenta;
        hmc.momenta().updateAll();
    };

    // C1. One level = the two-scale integrator.
    restoreStart();
    hmc.setIntegratorLevels({}, {});
    const MDWFHmcEnergy c1Start = hmc.energy();
    hmc.integrate(8);
    const double dhOld = hmc.energy().total() - c1Start.total();
    linksA = gauge;
    restoreStart();
    hmc.setIntegratorLevels({0, 0, 0}, {});
    hmc.integrate(8);
    const double dhMulti = hmc.energy().total() - c1Start.total();
    linksB = gauge;
    const double c1Links = mdwfHbMaxLinkDiff<HaloDepth>(commBase, linksA, linksB);
    const bool c1Passed = std::abs(dhMulti - dhOld) <= 1e-8 && c1Links <= 1e-10;
    rootLogger.info("MDWF Hasenbusch test C1 one level: Delta H two-scale ", dhOld, ", multi-level ", dhMulti,
                    " (difference ", dhMulti - dhOld, "), max final link difference ", c1Links, ", passed = ", c1Passed);

    // C2. Two fermion levels.
    hmc.setIntegratorLevels({0, 1, 1}, {2});
    restoreStart();
    const std::vector<int> evalBefore = hmc.termForceEvaluations();
    const MDWFHmcEnergy c2Start = hmc.energy();
    hmc.integrate(4);
    const MDWFHmcEnergy c2Forward = hmc.energy();
    const std::vector<int> evalAfter = hmc.termForceEvaluations();
    const int n0 = 4;
    const bool countsPassed = evalAfter.size() == 3 && evalAfter[0] - evalBefore[0] == n0 + 1
                              && evalAfter[1] - evalBefore[1] == 2 * n0 + 1 && evalAfter[2] - evalBefore[2] == 2 * n0 + 1;
    rootLogger.info("MDWF Hasenbusch test C2 force evaluations per trajectory: ", evalAfter[0] - evalBefore[0], ", ",
                    evalAfter[1] - evalBefore[1], ", ", evalAfter[2] - evalBefore[2], " (expected ", n0 + 1, ", ",
                    2 * n0 + 1, ", ", 2 * n0 + 1, "), passed = ", countsPassed);
    hmc.flipMomenta();
    hmc.integrate(4);
    hmc.flipMomenta();
    const MDWFHmcEnergy c2Back = hmc.energy();
    linksA = gauge;
    linksB = startGauge;
    const double c2Links = mdwfHbMaxLinkDiff<HaloDepth>(commBase, linksA, linksB);
    linksA = hmc.momenta();
    linksB = startMomenta;
    const double c2Mom = mdwfHbMaxLinkDiff<HaloDepth>(commBase, linksA, linksB);
    const double c2Energy = std::abs(c2Back.total() - c2Start.total());
    const bool reversibilityPassed = c2Links <= 1e-8 && c2Mom <= 1e-8 && c2Energy <= 1e-6;
    rootLogger.info("MDWF Hasenbusch test C2 reversibility: forward Delta H ", c2Forward.total() - c2Start.total(),
                    ", max link diff ", c2Links, ", max momentum diff ", c2Mom, ", |H_back - H_start| ", c2Energy,
                    ", passed = ", reversibilityPassed);
    const std::array<int, 3> levelSteps{{4, 8, 16}};
    std::array<double, 3> dh2{};
    for (size_t i = 0; i < levelSteps.size(); i++) {
        restoreStart();
        const MDWFHmcEnergy e0 = hmc.energy();
        hmc.integrate(levelSteps[i]);
        dh2[i] = hmc.energy().total() - e0.total();
        rootLogger.info("MDWF Hasenbusch test C2 scaling: level-0 steps = ", levelSteps[i], ", Delta H = ", dh2[i]);
    }
    const double q1 = std::abs(dh2[0]) / std::max(std::abs(dh2[1]), 1e-300);
    const double q2 = std::abs(dh2[1]) / std::max(std::abs(dh2[2]), 1e-300);
    const bool c2ScalingPassed = q1 >= 3.0 && q1 <= 6.0 && q2 >= 3.0 && q2 <= 6.0;
    rootLogger.info("MDWF Hasenbusch test C2 scaling: ratios ", q1, ", ", q2, ", passed = ", c2ScalingPassed);
    const bool multiLevelPassed = c1Passed && countsPassed && reversibilityPassed && c2ScalingPassed;

    if (!identityPassed || !heatbathPassed || !telescopePassed || !forcePassed || !scalingPassed || !multiLevelPassed) {
        throw std::runtime_error(stdLogger.fatal("MDWF Hasenbusch test failed: A ", identityPassed, ", B1 ",
                                                 heatbathPassed, ", B2 ", telescopePassed, ", B3 ", forcePassed,
                                                 ", B4 ", scalingPassed, ", C ", multiLevelPassed));
    }
    rootLogger.info("MDWF Hasenbusch test passed with Ls = ", Ls);
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

        runMDWFHasenbuschTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
