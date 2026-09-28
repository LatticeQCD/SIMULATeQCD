/*
 * Symanzik gauge force test for the MDWF HMC driver (resolves the open
 * "SIMULATeQCD rectangle term" item of TODO.md).
 *
 * mdwfMobiusHmcConventionTest found SIMULATeQCD's tree-level Symanzik force
 * gauge_force (gaugeActionDeriv.h), evaluated through a HOST accessor, off by
 * 2.4% from the gradient of S_g = -(3 beta/5) symanzik(), entirely in the
 * rectangle term. Diagnosis: GIndexer::site_up_2dn(s, mu, nu)
 * (src/base/indexer/bulkIndexer.h) is site_move<1, -2>(s, mu, nu) = s + mu - 2 nu
 * on the GPU, but its host fallback is site_up_dn_dn(s, mu, mu, nu) = s - nu.
 * gauge_force uses it for one of the six rectangle staples, the 1x2 staple
 * below the link, U_nu^\dagger(s+mu-nu) U_nu^\dagger(s+mu-2nu) U_mu^\dagger(s-2nu) U_nu(s-2nu) U_nu(s-nu).
 * SIMULATeQCD's HMC evaluates gauge_force on the device, where it is correct.
 *
 * On a random 6^4 gauge field with Gaussian momenta (seed and draws of
 * mdwfMobiusHmcConventionTest), along U -> exp(i eps P) U, eps = 1e-4:
 *
 *   1. Device gauge_force (MDWFHmcSymanzikGaugeForce, as in MDWFHmcDriver):
 *      dS_g/dtau + sum tr(P (-i K)) = 0 for the full Symanzik action (relative
 *      1e-7), for its rectangle part K_sym - K_Wilson (1e-6), and for the
 *      Wilson plaquette force (1e-7).
 *   2. Root cause: the host gauge_force differs from the device one; adding to
 *      it TA(r_1 U_mu(s) (1/12) (correct - host) staple) with r_1 = beta/5, for
 *      the one staple above, reproduces the device force to 1e-12 relative, and
 *      the host rectangle rate is reported (the convention test's -22.28 versus
 *      -40.49).
 *   3. HMC wiring: MDWFPauliVillarsTwoFlavorHmc with symanzik_gauge (unit start,
 *      tau = 0.2, mf = 0.1, pv_mass = 1): reversibility (1e-8 links and
 *      momenta, 1e-6 in H) and Delta H ratios between 3 and 6 for 8/16/32 steps.
 *
 * Single rank.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFHmc.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

template<size_t HaloDepth>
double mdwfSymTestKineticRate(const Gaugefield<double, false, HaloDepth> &momenta,
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

double mdwfSymTestRelSum(double actionRate, double kineticRate) {
    return std::abs(actionRate + kineticRate) / std::max(1.0, std::abs(actionRate));
}

template<size_t HaloDepth>
double mdwfSymTestMaxLinkDiff(CommunicationBase &commBase, Gaugefield<double, true, HaloDepth, R18> &a,
                              Gaugefield<double, true, HaloDepth, R18> &b) {
    typedef GIndexer<All, HaloDepth> GInd;
    Gaugefield<double, false, HaloDepth, R18> aHost(commBase, "MDWF_sym_test_diff_a");
    Gaugefield<double, false, HaloDepth, R18> bHost(commBase, "MDWF_sym_test_diff_b");
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

void logMDWFSymEnergy(const std::string &label, const MDWFHmcEnergy &energy) {
    rootLogger.info("MDWF Symanzik test HMC ", label, ": kinetic = ", energy.kinetic, ", gauge = ", energy.gauge,
                    ", fermion = ", energy.fermion, ", total = ", energy.total());
}

void runMDWFSymanzikGaugeForceTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    const size_t Ls = 8;
    typedef GIndexer<All, HaloDepth> GInd;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using Momenta = Gaugefield<double, true, HaloDepth>;
    using HostMomenta = Gaugefield<double, false, HaloDepth>;

    const LatticeData lat = GInd::getLatData();
    if (lat.vol4 != lat.globvol4) {
        throw std::runtime_error(stdLogger.fatal("MDWF Symanzik gauge force test is single-rank only"));
    }
    const double beta = 6.0;
    const double epsilon = 1e-4;
    const double volume = static_cast<double>(lat.globvol4);

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20260923);
    d_rand = h_rand;

    Gauge gauge(commBase, "MDWF_sym_test_gauge");
    Gauge gaugePlus(commBase, "MDWF_sym_test_gauge_plus");
    Gauge gaugeMinus(commBase, "MDWF_sym_test_gauge_minus");
    gauge.random(d_rand.state);
    gauge.updateAll();
    Momenta momenta(commBase, "MDWF_sym_test_momenta");
    momenta.gauss(d_rand.state);
    momenta.updateAll();
    HostMomenta momentaHost(commBase, "MDWF_sym_test_momenta_host");
    momentaHost = momenta;

    // --- Part 1: device forces versus finite differences of the actions. ---
    gaugePlus.iterateOverBulkAllMu(MDWFHmcEvolveQ<HaloDepth>(gauge.getAccessor(), momenta.getAccessor(), epsilon));
    gaugePlus.updateAll();
    gaugeMinus.iterateOverBulkAllMu(MDWFHmcEvolveQ<HaloDepth>(gauge.getAccessor(), momenta.getAccessor(), -epsilon));
    gaugeMinus.updateAll();

    auto symanzikAction = [&](Gauge &g) {
        GaugeAction<double, true, HaloDepth, R18> action(g);
        return -(3.0 * beta / 5.0) * static_cast<double>(action.symanzik());
    };
    auto plaquetteAction = [&](Gauge &g) {
        GaugeAction<double, true, HaloDepth, R18> action(g);
        return -(beta / 3.0) * 18.0 * volume * static_cast<double>(action.plaquette());
    };
    auto rectangleAction = [&](Gauge &g) {
        GaugeAction<double, true, HaloDepth, R18> action(g);
        return (beta / 60.0) * 36.0 * volume * static_cast<double>(action.rectangle());
    };
    const double symanzikRate = (symanzikAction(gaugePlus) - symanzikAction(gaugeMinus)) / (2.0 * epsilon);
    const double plaquetteRate = (plaquetteAction(gaugePlus) - plaquetteAction(gaugeMinus)) / (2.0 * epsilon);
    const double rectangleRate = (rectangleAction(gaugePlus) - rectangleAction(gaugeMinus)) / (2.0 * epsilon);

    Gauge ipdotDevice(commBase, "MDWF_sym_test_ipdot_symanzik");
    Gauge ipdotWilsonDevice(commBase, "MDWF_sym_test_ipdot_wilson");
    ipdotDevice.iterateOverBulkAllMu(MDWFHmcSymanzikGaugeForce<HaloDepth>(gauge.getAccessor(), beta));
    ipdotWilsonDevice.iterateOverBulkAllMu(MDWFHmcWilsonGaugeForce<HaloDepth>(gauge.getAccessor(), beta));
    HostGauge symHost(commBase, "MDWF_sym_test_k_sym_dev");
    HostGauge wilsonHost(commBase, "MDWF_sym_test_k_wilson_dev");
    HostGauge rectHost(commBase, "MDWF_sym_test_k_rect_dev");
    symHost = ipdotDevice;
    wilsonHost = ipdotWilsonDevice;
    {
        SU3Accessor<double, R18> rAcc = rectHost.getAccessor();
        const SU3Accessor<double, R18> sAcc = symHost.getAccessor();
        const SU3Accessor<double, R18> wAcc = wilsonHost.getAccessor();
        for (size_t siteIndex = 0; siteIndex < lat.vol4; siteIndex++) {
            const gSite site = GInd::getSite(siteIndex);
            for (uint8_t mu = 0; mu < 4; mu++) {
                const gSiteMu siteMu = GInd::getSiteMu(site, mu);
                rAcc.setLink(siteMu, sAcc.getLink(siteMu) - wAcc.getLink(siteMu));
            }
        }
    }
    const double symanzikKinetic = mdwfSymTestKineticRate<HaloDepth>(momentaHost, symHost);
    const double wilsonKinetic = mdwfSymTestKineticRate<HaloDepth>(momentaHost, wilsonHost);
    const double rectangleKinetic = mdwfSymTestKineticRate<HaloDepth>(momentaHost, rectHost);
    const double symanzikRelSum = mdwfSymTestRelSum(symanzikRate, symanzikKinetic);
    const double wilsonRelSum = mdwfSymTestRelSum(plaquetteRate, wilsonKinetic);
    const double rectangleRelSum = mdwfSymTestRelSum(rectangleRate, rectangleKinetic);
    const bool devicePassed = std::abs(symanzikRate) > 1e-8 && symanzikRelSum <= 1e-7 && wilsonRelSum <= 1e-7
                              && rectangleRelSum <= 1e-6;
    rootLogger.info("MDWF Symanzik test device force: Symanzik dS_g/dtau = ", symanzikRate,
                    ", sum tr(P(-i K)) = ", symanzikKinetic, ", relSum = ", symanzikRelSum);
    rootLogger.info("MDWF Symanzik test device force: rectangle dS_rect/dtau = ", rectangleRate,
                    ", sum tr(P(-i (K_sym - K_Wilson))) = ", rectangleKinetic, ", relSum = ", rectangleRelSum,
                    "; plaquette dS_plaq/dtau = ", plaquetteRate, ", Wilson sum = ", wilsonKinetic,
                    ", relSum = ", wilsonRelSum, ", passed = ", devicePassed);

    // --- Part 2: root cause, host gauge_force and the one corrupted staple. ---
    HostGauge gaugeHost(commBase, "MDWF_sym_test_gauge_host");
    HostGauge hostForce(commBase, "MDWF_sym_test_k_sym_host");
    HostGauge fixedForce(commBase, "MDWF_sym_test_k_sym_fixed");
    gaugeHost = gauge;
    double maxHostDiff = 0.0;
    double maxFixedDiff = 0.0;
    double maxForce = 0.0;
    {
        const SU3Accessor<double, R18> g = gaugeHost.getAccessor();
        const SU3Accessor<double, R18> dAcc = symHost.getAccessor();
        SU3Accessor<double, R18> hAcc = hostForce.getAccessor();
        SU3Accessor<double, R18> fAcc = fixedForce.getAccessor();
        const double r1 = beta / 5.0;
        const double c2 = 1.0 / 12.0;
        for (size_t siteIndex = 0; siteIndex < lat.vol4; siteIndex++) {
            const gSite s = GInd::getSite(siteIndex);
            for (int mu = 0; mu < 4; mu++) {
                const gSiteMu siteMu = GInd::getSiteMu(s, mu);
                const SU3<double> host = gauge_force<double, HaloDepth, R18>(g, siteMu, beta);
                SU3<double> correction = su3_zero<double>();
                for (int nuAux = 1; nuAux < 4; nuAux++) {
                    const int nu = (mu + nuAux) % 4;
                    const gSite upMuDnNu = GInd::site_dn(GInd::site_up(s, mu), nu);   // s + mu - nu
                    const gSite upMu2DnNu = GInd::site_dn(upMuDnNu, nu);               // s + mu - 2 nu
                    const gSite dnNu = GInd::site_dn(s, nu);                            // s - nu
                    const gSite dn2Nu = GInd::site_dn(dnNu, nu);                        // s - 2 nu
                    const SU3<double> head = g.getLinkDagger(GInd::getSiteMu(upMuDnNu, nu));
                    const SU3<double> tail = g.getLinkDagger(GInd::getSiteMu(dn2Nu, mu))
                                             * g.getLink(GInd::getSiteMu(dn2Nu, nu))
                                             * g.getLink(GInd::getSiteMu(dnNu, nu));
                    const SU3<double> right = head * g.getLinkDagger(GInd::getSiteMu(upMu2DnNu, nu)) * tail;
                    const SU3<double> wrong = head * g.getLinkDagger(GInd::getSiteMu(dnNu, nu)) * tail;
                    correction += c2 * (right - wrong);
                }
                SU3<double> fix = r1 * g.getLink(siteMu) * correction;
                fix.TA();
                const SU3<double> device = dAcc.getLink(siteMu);
                hAcc.setLink(siteMu, host);
                fAcc.setLink(siteMu, host + fix);
                maxForce = std::max(maxForce, static_cast<double>(infnorm(device)));
                maxHostDiff = std::max(maxHostDiff, static_cast<double>(infnorm(host - device)));
                maxFixedDiff = std::max(maxFixedDiff, static_cast<double>(infnorm(host + fix - device)));
            }
        }
    }
    const double hostKinetic = mdwfSymTestKineticRate<HaloDepth>(momentaHost, hostForce);
    const double hostRectangleKinetic = hostKinetic - wilsonKinetic;
    const bool rootCausePassed = maxHostDiff > 1e-6 * maxForce && maxFixedDiff <= 1e-12 * maxForce;
    rootLogger.info("MDWF Symanzik test host force: max |K_host - K_device| = ", maxHostDiff, " (max |K| = ", maxForce,
                    "), host Symanzik sum = ", hostKinetic, " (relSum ", mdwfSymTestRelSum(symanzikRate, hostKinetic),
                    "), host rectangle sum = ", hostRectangleKinetic, " versus dS_rect/dtau = ", rectangleRate);
    rootLogger.info("MDWF Symanzik test root cause: host force + TA(r_1 U_mu (1/12)(correct - host) staple at "
                    "site_up_2dn) versus device force, max diff = ", maxFixedDiff, ", passed = ", rootCausePassed);

    // --- Part 3: HMC driver with the Symanzik gauge action. ---
    using Hmc = MDWFPauliVillarsTwoFlavorHmc<HaloDepth, Ls>;
    MDWFHmcParameters param{};
    param.beta = beta;
    param.M5 = 1.8;
    param.mf = 0.1;
    param.b5 = 1.5;
    param.csw = 0.5;
    param.pv_mass = 1.0;
    param.tau = 0.2;
    param.steps = 8;
    param.max_iter = 20000;
    param.precision = 1e-10;
    param.symanzik_gauge = true;

    Gauge hmcGauge(commBase, "MDWF_sym_test_hmc_gauge");
    hmcGauge.one();
    hmcGauge.updateAll();
    Hmc hmc(commBase, hmcGauge, param, d_rand.state);
    hmc.refreshMomenta();
    hmc.heatbath();
    Gauge startGauge(commBase, "MDWF_sym_test_start_gauge");
    Gauge startMomenta(commBase, "MDWF_sym_test_start_momenta");
    startGauge = hmcGauge;
    startMomenta = hmc.momenta();

    rootLogger.info("MDWF Symanzik test HMC: unit-gauge S_g = ", hmc.gaugeAction(), " (expected -(3 beta/5)(5/3 * 18 V "
                    "- 1/12 * 36 V) = ", -(3.0 * beta / 5.0) * (5.0 / 3.0 * 18.0 * volume - 36.0 * volume / 12.0) / 3.0,
                    ")");
    const MDWFHmcEnergy start = hmc.energy();
    logMDWFSymEnergy("start", start);
    hmc.integrate(param.steps);
    const MDWFHmcEnergy forward = hmc.energy();
    hmc.flipMomenta();
    hmc.integrate(param.steps);
    hmc.flipMomenta();
    const MDWFHmcEnergy back = hmc.energy();
    logMDWFSymEnergy("after reversed trajectory", back);
    const double linkDiff = mdwfSymTestMaxLinkDiff<HaloDepth>(commBase, hmcGauge, startGauge);
    const double momentumDiff = mdwfSymTestMaxLinkDiff<HaloDepth>(commBase, hmc.momenta(), startMomenta);
    const double energyDiff = std::abs(back.total() - start.total());
    const bool reversibilityPassed = linkDiff <= 1e-8 && momentumDiff <= 1e-8 && energyDiff <= 1e-6;
    rootLogger.info("MDWF Symanzik test HMC reversibility (", param.steps, " steps): forward Delta H = ",
                    forward.total() - start.total(), ", max link diff = ", linkDiff, ", max momentum diff = ",
                    momentumDiff, ", |H_back - H_start| = ", energyDiff, ", passed = ", reversibilityPassed);

    const std::array<int, 3> stepCounts{{8, 16, 32}};
    std::array<double, 3> deltaH{};
    for (size_t i = 0; i < stepCounts.size(); i++) {
        hmcGauge = startGauge;
        hmcGauge.updateAll();
        hmc.momenta() = startMomenta;
        hmc.momenta().updateAll();
        const MDWFHmcEnergy restored = hmc.energy();
        hmc.integrate(stepCounts[i]);
        deltaH[i] = hmc.energy().total() - restored.total();
        rootLogger.info("MDWF Symanzik test HMC scaling: steps = ", stepCounts[i], ", eps = ",
                        param.tau / stepCounts[i], ", Delta H = ", deltaH[i]);
    }
    const double ratioCoarse = std::abs(deltaH[0]) / std::max(std::abs(deltaH[1]), 1e-300);
    const double ratioFine = std::abs(deltaH[1]) / std::max(std::abs(deltaH[2]), 1e-300);
    const bool scalingPassed = ratioCoarse >= 3.0 && ratioCoarse <= 6.0 && ratioFine >= 3.0 && ratioFine <= 6.0;
    rootLogger.info("MDWF Symanzik test HMC scaling: |dH(8)/dH(16)| = ", ratioCoarse, ", |dH(16)/dH(32)| = ",
                    ratioFine, " (leapfrog expects 4), max rms gauge force = ", hmc.maxGaugeForceRms(),
                    ", passed = ", scalingPassed);

    if (!devicePassed || !rootCausePassed || !reversibilityPassed || !scalingPassed) {
        throw std::runtime_error(stdLogger.fatal(
            "MDWF Symanzik gauge force test failed: device passed = ", devicePassed, ", root cause passed = ",
            rootCausePassed, ", reversibility passed = ", reversibilityPassed, ", scaling passed = ", scalingPassed,
            " (see diagnostics above)"));
    }
    rootLogger.info("MDWF Symanzik gauge force test passed");
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

        runMDWFSymanzikGaugeForceTest(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
