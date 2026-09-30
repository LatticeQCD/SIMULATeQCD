/*
 * Even-site MDWF pseudofermion action test (EVEN_ODD_DESIGN.md stage E2a,
 * c_sw = 0), MDWFEvenOddFermionActions.h.
 *
 * Random gauge field, 6^4, Ls = 8, M5 = 1.8, b5 = 1.5, c_sw = 0, pv_mass = 1:
 *
 *   A. MDWFEvenOddPauliVillarsTwoFlavorFermionAction, mf = 0.1:
 *      heatbath identity S = eta^+ eta (1e-10); force/energy identity
 *      dS/dtau + sum tr(P(-i K)) = 0 along U -> exp(i eps P) U, eps = 1e-4
 *      (1e-5); cancellation at mf = pv_mass (ratio 1e-6); force wall time and
 *      CG iterations versus MDWFPauliVillarsTwoFlavorFermionAction (diagnostic).
 *   B. MDWFEvenOddOneFlavorRhmcFermionAction, ms = 0.1, intervals
 *      [lambda_min / 2, 1.2 lambda_max] of Mhat^+ Mhat from Lanczos
 *      (600 / 300 steps), approximation error 1e-12: heatbath identity (1e-6);
 *      energy identity with force_error = 0 (1e-5); cancellation at
 *      ms = pv_mass (1e-6); force with force_error = 1e-8 versus the exact
 *      force (1e-4).
 *
 * The energy identity is the decisive check of the full-lattice form of the
 * even-site force (Re[u^+ dMhat v] = Re[U^+ dM V]).
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFEvenOddFermionActions.h"
#include "../experimental/mdwf/MDWFSpectralBounds.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

template<size_t HaloDepth>
void mdwfEoActEvolve(Gaugefield<double, true, HaloDepth, R18> &out, Gaugefield<double, true, HaloDepth, R18> &in,
                     Gaugefield<double, true, HaloDepth, R18> &momenta, double eps) {
    out.iterateOverBulkAllMu(MDWFHmcEvolveQ<HaloDepth>(in.getAccessor(), momenta.getAccessor(), eps));
    out.updateAll();
}

template<size_t HaloDepth>
double mdwfEoActKineticRate(const Gaugefield<double, false, HaloDepth, R18> &momenta,
                            const Gaugefield<double, false, HaloDepth, R18> &ipdot) {
    typedef GIndexer<All, HaloDepth> GInd;
    const SU3Accessor<double, R18> pAcc = momenta.getAccessor();
    const SU3Accessor<double, R18> fAcc = ipdot.getAccessor();
    double rate = 0.0;
    for (size_t i = 0; i < GInd::getLatData().vol4; i++) {
        const gSite site = GInd::getSite(i);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const gSiteMu siteMu = GInd::getSiteMu(site, mu);
            rate += real(tr_c(pAcc.getLink(siteMu), COMPLEX(double)(0.0, -1.0) * fAcc.getLink(siteMu)));
        }
    }
    return rate;
}

template<size_t HaloDepth>
double mdwfEoActForceRms(const Gaugefield<double, false, HaloDepth, R18> &a,
                         const Gaugefield<double, false, HaloDepth, R18> *b = nullptr) {
    typedef GIndexer<All, HaloDepth> GInd;
    const SU3Accessor<double, R18> aAcc = a.getAccessor();
    double sum = 0.0;
    for (size_t i = 0; i < GInd::getLatData().vol4; i++) {
        const gSite site = GInd::getSite(i);
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

double mdwfEoActSeconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// Extreme Ritz values of Mhat(mass)^+ Mhat(mass) on the even sites.
template<size_t HaloDepth, size_t Ls>
MDWFLanczosCheckpoint mdwfEoActSchurBounds(CommunicationBase &commBase, Gaugefield<double, true, HaloDepth, R18> &gauge,
                                           const MDWFHmcParameters &param, double mass, int steps, uint4 *randState,
                                           const std::string &name) {
    using Types = MDWFEvenOddTypes<HaloDepth, Ls>;
    typename Types::EvenOdd eo(gauge, param.M5, mass, param.b5, param.csw, name + "_eo");
    typename Types::NormalOp normal(eo, commBase, name + "_nrm");
    typename Types::Adapter adapter(normal);
    typename Types::SpinorE start(commBase, name + "_start");
    eo.refresh();
    start.gauss(randState);
    start.updateAll();
    const MDWFLanczosResult result = mdwfLanczosExtremes(adapter, start, steps, {steps / 2, steps}, name + "_lz");
    for (const MDWFLanczosCheckpoint &c : result.checkpoints) {
        rootLogger.info("MDWF even/odd action test Lanczos Mhat^+Mhat (mass = ", mass, "): steps = ", c.steps,
                        ", lambda_min = ", c.lambda_min, ", lambda_max = ", c.lambda_max);
    }
    return result.checkpoints.back();
}

template<size_t Ls>
void runMDWFEvenOddActionTest(CommunicationBase &commBase) {
    const size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using EoPv = MDWFEvenOddPauliVillarsTwoFlavorFermionAction<HaloDepth, Ls>;
    using EoRhmc = MDWFEvenOddOneFlavorRhmcFermionAction<HaloDepth, Ls>;
    using FullPv = MDWFPauliVillarsTwoFlavorFermionAction<HaloDepth, Ls>;

    MDWFHmcParameters param{};
    param.beta = 6.0;
    param.M5 = 1.8;
    param.mf = 0.1;
    param.b5 = 1.5;
    param.csw = 0.0;
    param.tau = 1.0;
    param.steps = 1;
    param.max_iter = 20000;
    param.precision = 1e-11;
    param.pv_mass = 1.0;
    param.rhmc.ms = 0.1;
    param.rhmc.action_error = 1e-12;
    param.rhmc.force_error = 0.0;
    param.rhmc.max_order = 30;
    param.rhmc.digits = 50;
    const double eps = 1e-4;

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20261003);
    d_rand = h_rand;

    Gauge gauge(commBase, "MDWF_eoact_gauge");
    Gauge gaugePlus(commBase, "MDWF_eoact_gplus");
    Gauge gaugeMinus(commBase, "MDWF_eoact_gminus");
    Gauge momenta(commBase, "MDWF_eoact_mom");
    gauge.random(d_rand.state);
    gauge.updateAll();
    momenta.gauss(d_rand.state);
    momenta.updateAll();
    mdwfEoActEvolve<HaloDepth>(gaugePlus, gauge, momenta, eps);
    mdwfEoActEvolve<HaloDepth>(gaugeMinus, gauge, momenta, -eps);

    HostGauge gaugeHost(commBase, "MDWF_eoact_ghost");
    HostGauge momentaHost(commBase, "MDWF_eoact_momhost");
    HostGauge ipdotHost(commBase, "MDWF_eoact_ipdot");
    HostGauge ipdotOtherHost(commBase, "MDWF_eoact_kother");
    gaugeHost = gauge;
    momentaHost = momenta;

    rootLogger.info("MDWF even/odd action test: M5 = ", param.M5, ", b5 = ", param.b5, ", c_sw = ", param.csw,
                    ", mf = ", param.mf, ", ms = ", param.rhmc.ms, ", pv_mass = ", param.pv_mass, ", Ls = ", Ls,
                    ", solver precision = ", param.precision, ", random gauge");

    // --- A. Even-site Pauli-Villars two-flavour action. ---
    EoPv pv(commBase, gauge, param);
    pv.heatbath(d_rand.state);
    const double pvAction = pv.action();
    const double pvHeatbath = std::abs(pvAction - pv.noiseNorm2()) / std::max(1.0, pv.noiseNorm2());
    auto start = std::chrono::steady_clock::now();
    pv.force(ipdotHost, gaugeHost);
    const double eoPvForceSeconds = mdwfEoActSeconds(start);
    const int eoPvIterations = pv.lastIterations();
    const double pvForceRms = mdwfEoActForceRms<HaloDepth>(ipdotHost);
    const double pvKinetic = mdwfEoActKineticRate<HaloDepth>(momentaHost, ipdotHost);
    EoPv pvPlus(commBase, gaugePlus, param);
    EoPv pvMinus(commBase, gaugeMinus, param);
    pvPlus.phi() = pv.phi();
    pvMinus.phi() = pv.phi();
    const double pvRate = (pvPlus.action() - pvMinus.action()) / (2.0 * eps);
    const double pvRelSum = std::abs(pvRate + pvKinetic) / std::max(1.0, std::abs(pvRate));

    MDWFHmcParameters equalParam = param;
    equalParam.mf = param.pv_mass;
    EoPv pvEqual(commBase, gauge, equalParam);
    pvEqual.heatbath(d_rand.state);
    pvEqual.force(ipdotOtherHost, gaugeHost);
    const double pvCancellation = mdwfEoActForceRms<HaloDepth>(ipdotOtherHost) / std::max(pvForceRms, 1e-300);

    FullPv fullPv(commBase, gauge, param);
    fullPv.heatbath(d_rand.state);
    start = std::chrono::steady_clock::now();
    fullPv.force(ipdotOtherHost, gaugeHost);
    const double fullPvForceSeconds = mdwfEoActSeconds(start);
    const double fullPvForceRms = mdwfEoActForceRms<HaloDepth>(ipdotOtherHost);

    const bool pvPassed = pvHeatbath <= 1e-10 && std::abs(pvRate) > 1e-8 && pvRelSum <= 1e-5 && pvCancellation <= 1e-6;
    rootLogger.info("MDWF even/odd action test PV two-flavour: heatbath S = ", pvAction, ", eta^+ eta = ",
                    pv.noiseNorm2(), ", relDiff = ", pvHeatbath, "; dS/dtau = ", pvRate, ", sum tr(P(-i K)) = ",
                    pvKinetic, ", relSum = ", pvRelSum, "; cancellation (mf = pv_mass) = ", pvCancellation,
                    ", passed = ", pvPassed);
    rootLogger.info("MDWF even/odd action test PV two-flavour force cost: even/odd ", eoPvForceSeconds, " s (CG ",
                    eoPvIterations, " iterations) versus full ", fullPvForceSeconds, " s, speedup ",
                    fullPvForceSeconds / eoPvForceSeconds, "; rms force even/odd ", pvForceRms, ", full ",
                    fullPvForceRms, " (different pseudofermions)");

    // --- B. Even-site one-flavour RHMC action. ---
    const MDWFLanczosCheckpoint bS = mdwfEoActSchurBounds<HaloDepth, Ls>(commBase, gauge, param, param.rhmc.ms, 600,
                                                                         d_rand.state, "MDWF_eoact_bs");
    const MDWFLanczosCheckpoint bPv = mdwfEoActSchurBounds<HaloDepth, Ls>(commBase, gauge, param, param.pv_mass, 300,
                                                                          d_rand.state, "MDWF_eoact_bpv");
    param.rhmc.lambda_low_s = 0.5 * bS.lambda_min;
    param.rhmc.lambda_high_s = 1.2 * bS.lambda_max;
    param.rhmc.lambda_low_pv = 0.5 * bPv.lambda_min;
    param.rhmc.lambda_high_pv = 1.2 * bPv.lambda_max;

    EoRhmc rhmc(commBase, gauge, param);
    rootLogger.info("MDWF even/odd action test RHMC orders: A^(1/4) ", rhmc.quarterS().order, " (",
                    rhmc.quarterS().max_relative_error, "), A^(-1/2) ", rhmc.halfS().order, " (",
                    rhmc.halfS().max_relative_error, "), B^(1/4) ", rhmc.quarterPv().order, " (",
                    rhmc.quarterPv().max_relative_error, ")");
    rhmc.heatbath(d_rand.state);
    const int hbIterations = rhmc.lastIterations();
    const double rAction = rhmc.action();
    const double rHeatbath = std::abs(rAction - rhmc.noiseNorm2()) / std::max(1.0, rhmc.noiseNorm2());
    start = std::chrono::steady_clock::now();
    rhmc.force(ipdotHost, gaugeHost);
    const double rForceSeconds = mdwfEoActSeconds(start);
    const int rForceIterations = rhmc.lastIterations();
    const double rForceRms = mdwfEoActForceRms<HaloDepth>(ipdotHost);
    const double rKinetic = mdwfEoActKineticRate<HaloDepth>(momentaHost, ipdotHost);
    EoRhmc rPlus(commBase, gaugePlus, param);
    EoRhmc rMinus(commBase, gaugeMinus, param);
    rPlus.phi() = rhmc.phi();
    rMinus.phi() = rhmc.phi();
    const double rRate = (rPlus.action() - rMinus.action()) / (2.0 * eps);
    const double rRelSum = std::abs(rRate + rKinetic) / std::max(1.0, std::abs(rRate));

    MDWFHmcParameters rEqual = param;
    rEqual.rhmc.ms = param.pv_mass;
    EoRhmc rhmcEqual(commBase, gauge, rEqual);
    rhmcEqual.heatbath(d_rand.state);
    rhmcEqual.force(ipdotOtherHost, gaugeHost);
    const double rCancellation = mdwfEoActForceRms<HaloDepth>(ipdotOtherHost) / std::max(rForceRms, 1e-300);

    MDWFHmcParameters rSplit = param;
    rSplit.rhmc.force_error = 1e-8;
    EoRhmc rhmcSplit(commBase, gauge, rSplit);
    rhmcSplit.phi() = rhmc.phi();
    rhmcSplit.force(ipdotOtherHost, gaugeHost);
    const double rSplitDiff = mdwfEoActForceRms<HaloDepth>(ipdotOtherHost, &ipdotHost) / rForceRms;

    const bool rPassed = rHeatbath <= 1e-6 && std::abs(rRate) > 1e-8 && rRelSum <= 1e-5 && rCancellation <= 1e-6
                         && rSplitDiff <= 1e-4;
    rootLogger.info("MDWF even/odd action test one-flavour RHMC: heatbath S = ", rAction, ", eta^+ eta = ",
                    rhmc.noiseNorm2(), ", relDiff = ", rHeatbath, " (multishift ", hbIterations,
                    " iterations); dS/dtau = ", rRate, ", sum tr(P(-i K)) = ", rKinetic, ", relSum = ", rRelSum,
                    "; cancellation (ms = pv_mass) = ", rCancellation, "; split force (1e-8) relative diff = ",
                    rSplitDiff, ", passed = ", rPassed);
    rootLogger.info("MDWF even/odd action test one-flavour RHMC force: ", rForceSeconds, " s, max multishift ",
                    rForceIterations, " iterations, rms force ", rForceRms, " (ratio to even/odd PV ",
                    rForceRms / pvForceRms, ")");

    if (!pvPassed || !rPassed) {
        throw std::runtime_error(stdLogger.fatal("MDWF even/odd action test failed: PV passed = ", pvPassed,
                                                 ", RHMC passed = ", rPassed, " (see diagnostics above)"));
    }
    rootLogger.info("MDWF even/odd action test passed with Ls = ", Ls);
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

        runMDWFEvenOddActionTest<8>(commBase);
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
