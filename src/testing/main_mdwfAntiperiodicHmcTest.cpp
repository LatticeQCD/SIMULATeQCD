/*
 * Antiperiodic temporal fermion boundary conditions in the MDWF HMC
 * (MDWFHmcParameters::antiperiodic_t, MDWFFermionBoundary.h, MDWFHmc.h).
 *
 * MDWFEvenOddTwoPlusOneHmc (the production path: even-site Pauli-Villars light
 * pair mf = 0.1, even-site one-flavour RHMC strange ms = 0.2, log det M_oo term
 * for c_sw = 0.5), M5 = 1.8, b5 = 1.5, pv_mass = 1, Wilson gauge action
 * beta = 6, 6^4, Ls = 8, solver precision 1e-10.
 *
 *   1. Boundary links: on the unit field the fermion gauge field has exactly
 *      vol3 links equal to -1, the temporal links on t = Lt - 1, and all
 *      others equal to +1.
 *   2. Finite-difference force check on a nontrivial configuration (two
 *      trajectories from the unit start): dS_f/dt along U(t) = exp(i t P) U,
 *      central difference with h = 1e-4, against i sum tr(P ipdot_f)
 *      (MDWF_HMC_CONVENTIONS.md), (a) with Gaussian P on all links and (b)
 *      with P only on the temporal links of the last time slice, the links
 *      whose sign the boundary condition flips (a sign error in the force
 *      there would flip the sign of (b)). Relative difference <= 1e-5.
 *   3. Delta H ratios between 3 and 6 for 8/16/32 steps (tau = 0.2).
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFEvenOddFermionActions.h"
#include "../experimental/mdwf/MDWFSpectralBounds.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

// Keeps the link on the temporal links of the last time slice and sets every other link to zero.
template<size_t HaloDepth>
struct MDWFApKeepBoundaryTimeLinks {
    SU3Accessor<double> _acc;
    int _ltLast;

    MDWFApKeepBoundaryTimeLinks(Gaugefield<double, true, HaloDepth, R18> &field, int ltLast)
        : _acc(field.getAccessor()), _ltLast(ltLast) {}

    __host__ __device__ SU3<double> operator()(gSiteMu siteMu) {
        if (siteMu.mu == 3 && static_cast<int>(siteMu.coord.t) == _ltLast) {
            return _acc.getLink(siteMu);
        }
        return su3_zero<double>();
    }
};

template<class Adapter, class Spinor>
MDWFLanczosCheckpoint mdwfApLanczos(Adapter &adapter, Spinor &start, uint4 *randState, int steps,
                                    const std::string &name, const std::string &label) {
    start.gauss(randState);
    start.updateAll();
    const MDWFLanczosResult result = mdwfLanczosExtremes(adapter, start, steps, {steps / 2, steps}, name);
    const MDWFLanczosCheckpoint &c = result.checkpoints.back();
    rootLogger.info("MDWF antiperiodic HMC test Lanczos ", label, ": steps = ", c.steps, ", lambda_min = ",
                    c.lambda_min, ", lambda_max = ", c.lambda_max);
    return c;
}

// Central difference of S_f along exp(i t P) U at fixed pseudofermions; the gauge field is restored.
template<class Hmc, class Gauge>
double mdwfApNumericalDerivative(Hmc &hmc, Gauge &gauge, Gauge &saved, double h) {
    saved = gauge;
    hmc.evolveQ(h);
    const double plus = hmc.fermionAction();
    gauge = saved;
    gauge.updateAll();
    hmc.evolveQ(-h);
    const double minus = hmc.fermionAction();
    gauge = saved;
    gauge.updateAll();
    return (plus - minus) / (2.0 * h);
}

template<size_t Ls>
void runMDWFAntiperiodicHmcTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using EoHmc = MDWFEvenOddTwoPlusOneHmc<HaloDepth, Ls>;
    using Types = MDWFEvenOddTypes<HaloDepth, Ls>;

    const LatticeData lat = GInd::getLatData();
    const int ltLast = static_cast<int>(lat.lt) - 1;

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
    param.antiperiodic_t = true;
    param.rhmc.ms = 0.2;
    param.rhmc.action_error = 1e-12;
    param.rhmc.force_error = 1e-8;
    param.rhmc.max_order = 30;
    param.rhmc.digits = 50;

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20261008);
    d_rand = h_rand;

    Gauge gauge(commBase, "MDWF_ap_gauge");
    gauge.one();
    gauge.updateAll();

    // Strange-quark intervals of Mhat^+ Mhat with the antiperiodic boundary condition.
    {
        MDWFFermionGauge<HaloDepth> bcGauge(commBase, gauge, true, "MDWF_ap_lz_links");
        typename Types::EvenOdd eoS(bcGauge.get(), param.M5, param.rhmc.ms, param.b5, param.csw, "MDWF_ap_lzs_eo");
        typename Types::EvenOdd eo1(bcGauge.get(), param.M5, param.pv_mass, param.b5, param.csw, "MDWF_ap_lzp_eo");
        typename Types::NormalOp nS(eoS, commBase, "MDWF_ap_lzs_nrm");
        typename Types::NormalOp n1(eo1, commBase, "MDWF_ap_lzp_nrm");
        typename Types::Adapter aS(nS);
        typename Types::Adapter a1(n1);
        typename Types::SpinorE startE(commBase, "MDWF_ap_lz_starte");
        eoS.refresh();
        eo1.refresh();
        const MDWFLanczosCheckpoint cS = mdwfApLanczos(aS, startE, d_rand.state, 400, "MDWF_ap_lzs", "Mhat^+Mhat (ms)");
        const MDWFLanczosCheckpoint c1 = mdwfApLanczos(a1, startE, d_rand.state, 200, "MDWF_ap_lzp", "Mhat^+Mhat (pv)");
        param.rhmc.lambda_low_s = cS.lambda_min / 3.0;
        param.rhmc.lambda_high_s = 1.5 * cS.lambda_max;
        param.rhmc.lambda_low_pv = c1.lambda_min / 3.0;
        param.rhmc.lambda_high_pv = 1.5 * c1.lambda_max;
    }

    EoHmc hmc(commBase, gauge, param, d_rand.state);

    // --- 1. Boundary links on the unit field. ---
    HostGauge fermionHost(commBase, "MDWF_ap_flinks_host");
    fermionHost = hmc.fermionGauge();
    const SU3Accessor<double, R18> fAcc = fermionHost.getAccessor();
    size_t minusLinks = 0, wrongLinks = 0;
    for (size_t i = 0; i < lat.vol4; i++) {
        const gSite site = GInd::getSite(i);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const SU3<double> link = fAcc.getLink(GInd::getSiteMu(site, mu));
            const bool boundary = mu == 3 && static_cast<int>(site.coord.t) == ltLast;
            const SU3<double> expected = boundary ? static_cast<double>(-1.0) * su3_one<double>() : su3_one<double>();
            if (boundary) {
                minusLinks++;
            }
            if (static_cast<double>(infnorm(link - expected)) > 1e-14) {
                wrongLinks++;
            }
        }
    }
    const bool linksPassed = minusLinks == lat.vol3 && wrongLinks == 0;
    rootLogger.info("MDWF antiperiodic HMC test boundary links: ", minusLinks, " links -1 (expected ", lat.vol3,
                    "), ", wrongLinks, " links differ from the expected +-1, passed = ", linksPassed);

    // --- Nontrivial configuration: two trajectories from the unit start, always kept. ---
    std::mt19937_64 acceptRng(20261008UL);
    for (int traj = 0; traj < 2; traj++) {
        const MDWFHmcTrajectoryResult r = hmc.trajectory(false, acceptRng);
        GaugeAction<double, true, HaloDepth, R18> action(gauge);
        rootLogger.info("MDWF antiperiodic HMC test start trajectory ", traj + 1, ": Delta H = ", r.delta_h,
                        ", plaquette = ", action.plaquette());
    }

    // --- 2. Finite-difference force check. ---
    Gauge saved(commBase, "MDWF_ap_fd_saved");
    hmc.refreshMomenta();
    hmc.heatbath();
    const double h = 1e-4;
    const double analyticAll = hmc.fermionForceAlongMomenta();
    const double numericAll = mdwfApNumericalDerivative(hmc, gauge, saved, h);
    const double relAll = std::abs(numericAll - analyticAll) / std::max(std::abs(analyticAll), 1e-300);

    hmc.momenta().iterateOverBulkAllMu(MDWFApKeepBoundaryTimeLinks<HaloDepth>(hmc.momenta(), ltLast));
    hmc.momenta().updateAll();
    const double analyticBoundary = hmc.fermionForceAlongMomenta();
    const double numericBoundary = mdwfApNumericalDerivative(hmc, gauge, saved, h);
    const double relBoundary = std::abs(numericBoundary - analyticBoundary)
                               / std::max(std::abs(analyticBoundary), 1e-300);
    const bool fdPassed = relAll <= 1e-5 && relBoundary <= 1e-5 && std::abs(analyticBoundary) > 0.0;
    rootLogger.info("MDWF antiperiodic HMC test force, P on all links: dS_f/dt numerical ", numericAll,
                    ", from the force ", analyticAll, ", relative difference ", relAll);
    rootLogger.info("MDWF antiperiodic HMC test force, P on the boundary temporal links only: dS_f/dt numerical ",
                    numericBoundary, ", from the force ", analyticBoundary, ", relative difference ", relBoundary,
                    " (a sign error on these links would give ", -analyticBoundary, "), passed = ", fdPassed);

    // --- 3. Delta H scaling. ---
    hmc.refreshMomenta();
    hmc.heatbath();
    Gauge startGauge(commBase, "MDWF_ap_start_gauge");
    Gauge startMomenta(commBase, "MDWF_ap_start_momenta");
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
        rootLogger.info("MDWF antiperiodic HMC test scaling: steps = ", stepCounts[i], ", Delta H = ", deltaH[i]);
    }
    const double r1 = std::abs(deltaH[0]) / std::max(std::abs(deltaH[1]), 1e-300);
    const double r2 = std::abs(deltaH[1]) / std::max(std::abs(deltaH[2]), 1e-300);
    const bool scalingPassed = r1 >= 3.0 && r1 <= 6.0 && r2 >= 3.0 && r2 <= 6.0;
    rootLogger.info("MDWF antiperiodic HMC test scaling: ratios ", r1, ", ", r2, " (leapfrog expects 4), passed = ",
                    scalingPassed);

    if (!linksPassed || !fdPassed || !scalingPassed) {
        throw std::runtime_error(stdLogger.fatal("MDWF antiperiodic HMC test failed: links passed = ", linksPassed,
                                                 ", force passed = ", fdPassed, ", scaling passed = ", scalingPassed));
    }
    rootLogger.info("MDWF antiperiodic HMC test passed with Ls = ", Ls);
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

        runMDWFAntiperiodicHmcTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
