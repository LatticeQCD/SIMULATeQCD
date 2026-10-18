/*
 * DSDR determinant ratio (MDWFDSDRFermionActions.h) and the plaquette + rectangle (Iwasaki) gauge action
 * (MDWFHmcRectangleGaugeForce, MDWFHmc.h).
 *
 * mode = unit (lattice from the parameter file, e.g. 8^4; nontrivial configuration from three always-accepted
 * trajectories of the DSDR-only HMC from the unit start; twists, M5, gauge c1, fermion BCs from the parameter file):
 *
 *   A. Operator M(mu) = D_W(-M5) + i mu gamma5 (mu = second twist):
 *      1. even/odd blocks against the full operator: with x_o = -M_oo^-1 M_oe x_e, M (x_e, x_o) = (Mhat x_e, 0);
 *      2. adjoints: <y, Mhat x> = <Mhat^+ y, x> and <y, M x> = <M^+ y, x>;
 *      3. gamma5-hermiticity of D_W: |M x|^2 = |D_W x|^2 + mu^2 |x|^2 (the cross term vanishes);
 *      all to 1e-12 relative.
 *   B. DSDR ladder (MDWFDSDRHmc, tau = 0.2):
 *      1. heatbath identity per factor (action = noise norm, 1e-8);
 *      2. a factor with equal twists: action = |phi|^2 and force / force of the lightest factor below 1e-6;
 *      3. finite-difference force, dS/dt along exp(i t P) U (h = 1e-4) against i sum tr(P ipdot), whole ladder and
 *         each factor, relative 1e-5;
 *      4. Delta H ratios between 3 and 6 for 8/16/32 steps (tau = 0.2).
 *   C. Gauge action:
 *      1. plaquette + rectangle at (3 beta/5, c1 = -1/12) = SIMULATeQCD's Symanzik action -(3 beta/5) symanzik() at
 *         beta (action relative 1e-12, force 1e-12 absolute per link);
 *      2. finite-difference force of the rectangle action at the parameter-file c1, relative 1e-6.
 *
 * mode = gpt (cross-check against GPT, gpt_dsdr_check.py, on the configuration Gaugefile; lattice = its size): prints
 * gauge-invariant numbers that GPT computes with g.qcd.fermion.wilson_twisted_mass(mass = -M5, mu, boundary_phases =
 * [1, 1, 1, -1 if antiperiodic_t]), g.qcd.pseudofermion.action.two_flavor_ratio_evenodd_schur and
 * g.qcd.gauge.action.iwasaki / improved_with_rectangle(beta, c1):
 *
 *   G1. plaquette and rectangle means, the rectangle gauge action in GPT's normalization
 *       beta V [c0 (1 - P) 6 + c1 (1 - R) 12];
 *   G2. the gauge force on the four links at source_site and its rms;
 *   G3. per twist, T(mu) = sum_a Re e_a^+ (M^+ M)^-1 e_a at source_site (12 spin-colour unit vectors; independent of
 *       the spin basis and of the sign of mu);
 *   G4. per DSDR factor, sum over the 12 point sources phi = e_a at source_site (an even site) of the factor's
 *       action, and the summed force on the four links at source_site and its rms.
 *
 * mode = gptmdwf: G1, G2 and G6 (light Mobius ratio factors vs gpt_mdwf_check.py), see runMDWFLightGptCheck.
 *
 * mode = exact: exact-comparison suite tests/exact8 (HMC part, GPT: tests/exact8/gpt/gpt_exact_hmc.py), see
 *   runMDWFExactHmc. exact_timing = 1: same trajectories without the per-step diagnostics, timed (tests/exact32 timing).
 *
 * Usage (from the build's testing directory): ./mdwfDSDRTest [param file] [key=value ...]
 * Parameters: parameter/tests/mdwfDSDRTest.param. Single rank, except mode = exact (any Nodes layout).
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFDSDRFermionActions.h"
#include "../experimental/mdwf/MDWFEofaFermionAction.h"
#include "../experimental/mdwf/MDWFEvenOddFermionActions.h"
#include "../experimental/mdwf/MDWFHashFields.h"
#include "../experimental/mdwf/MDWFHmc.h"
#include "../experimental/mdwf/MDWFRhmcFermionActions.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class MDWFDSDRTestParameters : public LatticeParameters {
public:
    Parameter<std::string> mode;
    Parameter<double> M5;
    DynamicParameter<double> dsdr_twists;
    Parameter<int> antiperiodic_t;
    Parameter<double> gauge_c1;
    Parameter<double> precision;
    Parameter<int> max_iter;
    DynamicParameter<int> source_site;
    // mode = gptmdwf (G6): Mobius operator and light ratio factors
    Parameter<int> Ls;
    Parameter<double> b5;
    DynamicParameter<double> mdwf_pairs;      // m_light m_heavy [m_light m_heavy ...]
    DynamicParameter<int> mdwf_s_sources;     // fifth-dimension slices of the point sources (default 0 and Ls - 1)
    Parameter<int> force_halos;
    // mode = eofa (unit tests E1-E5) / gpteofa (G7): EOFA masses m1 < m2 and heatbath approximation
    Parameter<double> eofa_m1;
    Parameter<double> eofa_m2;
    Parameter<double> eofa_lambda_low;
    Parameter<double> eofa_lambda_high;
    Parameter<double> eofa_error;
    Parameter<int> eofa_zolotarev;      // > 0: GPT's Zolotarev approximation of this order instead of Remez (eofa_error)
    Parameter<double> md_tau;           // mode = gptmd (T7): MD time (GPT tau / sqrt 2), steps, gauge steps, hash seed
    Parameter<int> md_steps;
    Parameter<int> md_gauge_steps;
    Parameter<std::string> md_seed;
    // mode = exact (tests/exact8): light Hasenbusch ladder mf < hasenbusch_masses < eofa_m2, number of trajectories
    Parameter<double> mf;
    DynamicParameter<double> hasenbusch_masses;
    Parameter<int> exact_traj;
    Parameter<int> light_form;          // 0 = SIMULATeQCD form, 1 = GPT form of the light ratio factors
    Parameter<int> exact_timing;        // mode = exact: 1 = timed trajectories, no per-step diagnostics
    Parameter<int> eofa_schur;          // mode = exact: EOFA solve form, 0 = Mhat (even sites), 2 = GPT's eo2_ne form

    MDWFDSDRTestParameters() {
        addDefault(mode, "mode", std::string("unit"));
        addDefault(M5, "M5", 1.8);
        addOptional(dsdr_twists, "dsdr_twists");
        addDefault(antiperiodic_t, "antiperiodic_t", 1);
        addDefault(gauge_c1, "gauge_c1", -0.331);
        addDefault(precision, "precision", 1e-12);
        addDefault(max_iter, "max_iter", 100000);
        addOptional(source_site, "source_site");
        addDefault(Ls, "Ls", 16);
        addDefault(b5, "b5", 1.9);
        addOptional(mdwf_pairs, "mdwf_pairs");
        addOptional(mdwf_s_sources, "mdwf_s_sources");
        addDefault(force_halos, "force_halos", 1);
        addDefault(eofa_m1, "eofa_m1", 0.04122);
        addDefault(eofa_m2, "eofa_m2", 1.0);
        addDefault(eofa_lambda_low, "eofa_lambda_low", 1.0);
        addDefault(eofa_lambda_high, "eofa_lambda_high", 12.0);
        addDefault(eofa_error, "eofa_error", 1e-10);
        addDefault(eofa_zolotarev, "eofa_zolotarev", 0);
        addDefault(md_tau, "md_tau", 0.70710678118654752);
        addDefault(md_steps, "md_steps", 5);
        addDefault(md_gauge_steps, "md_gauge_steps", 4);
        addDefault(md_seed, "md_seed", std::string("parity_md"));
        addDefault(mf, "mf", 0.01);
        addOptional(hasenbusch_masses, "hasenbusch_masses");
        addDefault(exact_traj, "exact_traj", 10);
        addDefault(light_form, "light_form", 0);
        addDefault(exact_timing, "exact_timing", 0);
        addDefault(eofa_schur, "eofa_schur", 0);
    }
};

template<size_t HaloDepth>
using DSDRHostGauge = Gaugefield<double, false, HaloDepth, R18>;
template<size_t HaloDepth>
using DSDRGauge = Gaugefield<double, true, HaloDepth, R18>;

template<size_t HaloDepth>
double mdwfDSDRMaxLinkDiff(DSDRHostGauge<HaloDepth> &a, DSDRHostGauge<HaloDepth> &b) {
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

// sqrt of the mean over links of -tr(K K) (the driver's forceRms).
template<size_t HaloDepth>
double mdwfDSDRForceRms(DSDRHostGauge<HaloDepth> &f) {
    typedef GIndexer<All, HaloDepth> GInd;
    const SU3Accessor<double, R18> acc = f.getAccessor();
    double sum = 0.0;
    for (size_t i = 0; i < GInd::getLatData().vol4; i++) {
        const gSite site = GInd::getSite(i);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const SU3<double> k = acc.getLink(GInd::getSiteMu(site, mu));
            sum += -tr_d(k, k);
        }
    }
    return std::sqrt(sum / (4.0 * static_cast<double>(GInd::getLatData().vol4)));
}

// i sum tr(P K): dS/dt along exp(i t P) U predicted by the force K.
template<size_t HaloDepth>
double mdwfDSDRAlongMomenta(CommunicationBase &commBase, DSDRGauge<HaloDepth> &momenta, DSDRHostGauge<HaloDepth> &force) {
    typedef GIndexer<All, HaloDepth> GInd;
    DSDRHostGauge<HaloDepth> pHost(commBase, "MDWF_dsdr_pmom_host");
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

template<size_t HaloDepth>
void mdwfDSDRZero(DSDRHostGauge<HaloDepth> &f, CommunicationBase &commBase) {
    DSDRGauge<HaloDepth> zero(commBase, "MDWF_dsdr_zero_dev");
    zero.iterateOverBulkAllMu(MDWFDevZeroLinks<HaloDepth>());
    f = zero;
}

// The four link matrices at x of a host force field, row by row (re, im).
template<size_t HaloDepth>
void mdwfDSDRPrintLinks(DSDRHostGauge<HaloDepth> &f, sitexyzt x, const std::string &label) {
    typedef GIndexer<All, HaloDepth> GInd;
    const SU3Accessor<double, R18> acc = f.getAccessor();
    const gSite site = GInd::getSite(x.x, x.y, x.z, x.t);
    for (uint8_t mu = 0; mu < 4; mu++) {
        const SU3<double> k = acc.getLink(GInd::getSiteMu(site, mu));
        std::ostringstream s;
        s << std::setprecision(10);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                const COMPLEX(double) z = k(i, j);
                s << " (" << z.cREAL << "," << z.cIMAG << ")";
            }
            if (i < 2) {
                s << " |";
            }
        }
        rootLogger.info(label, " link mu = ", static_cast<int>(mu), ":", s.str());
    }
}

template<size_t HaloDepth>
bool runMDWFDSDRUnitTest(CommunicationBase &commBase, MDWFDSDRTestParameters &tp, const std::vector<double> &twists) {
    using Gauge = DSDRGauge<HaloDepth>;
    using HostGauge = DSDRHostGauge<HaloDepth>;
    using Hmc = MDWFDSDRHmc<HaloDepth, 8>;
    using Ladder = MDWFDSDRFermionAction<HaloDepth>;
    using Factor = MDWFDSDRRatioFermionAction<HaloDepth>;
    using EvenOdd = MDWFTwistedWilsonEvenOdd<HaloDepth>;
    using SpinorE = typename EvenOdd::SpinorE;
    using SpinorO = typename EvenOdd::SpinorO;
    using SpinorAll = typename EvenOdd::SpinorAll;

    MDWFHmcParameters param{};
    param.beta = 2.2;
    param.M5 = tp.M5();
    param.tau = 0.2;
    param.steps = 8;
    param.gauge_substeps = 4;
    param.max_iter = tp.max_iter();
    param.precision = tp.precision();
    param.antiperiodic_t = tp.antiperiodic_t() != 0;
    param.rectangle_gauge = true;
    param.gauge_c1 = tp.gauge_c1();
    param.dsdr_twists = twists;

    Gauge gauge(commBase, "MDWF_dsdr_gauge");
    gauge.one();
    gauge.updateAll();
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20261005);
    d_rand = h_rand;
    Hmc hmc(commBase, gauge, param, d_rand.state);
    Ladder &ladder = hmc.fermion();
    std::mt19937_64 acceptRng(1UL);
    for (int traj = 0; traj < 3; traj++) {
        const MDWFHmcTrajectoryResult r = hmc.trajectory(false, acceptRng);
        GaugeAction<double, true, HaloDepth, R18> action(gauge);
        rootLogger.info("MDWF DSDR test start trajectory ", traj + 1, ": Delta H = ", r.delta_h, ", plaquette = ",
                        action.plaquette());
    }
    rootLogger.info("MDWF DSDR test: twists ", ladder.describe(), " (", ladder.factorCount(), " factors), M5 = ",
                    param.M5, ", Iwasaki-type gauge c1 = ", param.gauge_c1, ", beta = ", param.beta,
                    ", fermion BCs in time: ", param.antiperiodic_t ? "antiperiodic" : "periodic");

    // --- A. Operator. ---
    Gauge &fgauge = hmc.fermionGauge();
    const double mu = twists.size() > 1 ? twists[1] : twists[0];
    EvenOdd eo(fgauge, param.M5, mu, "MDWF_dsdr_teo");
    EvenOdd eo0(fgauge, param.M5, 0.0, "MDWF_dsdr_teo0");
    SpinorE xe(commBase, "MDWF_dsdr_xe"), ye(commBase, "MDWF_dsdr_ye"), te(commBase, "MDWF_dsdr_te"),
        se(commBase, "MDWF_dsdr_se");
    SpinorO xo(commBase, "MDWF_dsdr_xo"), yo(commBase, "MDWF_dsdr_yo"), to(commBase, "MDWF_dsdr_to");
    SpinorAll xa(commBase, "MDWF_dsdr_xa"), ya(commBase, "MDWF_dsdr_ya"), ta(commBase, "MDWF_dsdr_ta"),
        sa(commBase, "MDWF_dsdr_sa");
    auto dot = [](auto &a, auto &b) {
        const std::vector<COMPLEX(double)> d = a.dotProductStacked(b);
        return d.at(0);
    };

    // 1. Even/odd blocks against the full operator.
    xe.gauss(d_rand.state);
    xe.updateAll();
    eo.Moe(to, xe);
    eo.MooInv(xo, to);
    xo *= COMPLEX(double)(-1.0, 0.0);
    EvenOdd::merge(xa, xe, xo);
    eo.applyFull(ya, xa, false);
    EvenOdd::split(ye, yo, ya);
    eo.schur(se, xe, false);
    te = ye;
    te -= se;
    const double a1Even = std::sqrt(real<double>(dot(te, te)) / real<double>(dot(se, se)));
    const double a1Odd = std::sqrt(real<double>(dot(yo, yo)) / real<double>(dot(se, se)));
    const bool a1Passed = a1Even <= 1e-12 && a1Odd <= 1e-12;
    rootLogger.info("MDWF DSDR test A1 (mu = ", mu, "): |even part of M(x_e, x_o) - Mhat x_e| / |Mhat x_e| = ", a1Even,
                    ", |odd part| / |Mhat x_e| = ", a1Odd, ", passed = ", a1Passed);

    // 2. Adjoints.
    ye.gauss(d_rand.state);
    ye.updateAll();
    eo.schur(se, xe, false);
    eo.schur(te, ye, true);
    const COMPLEX(double) lhsE = dot(ye, se);
    const COMPLEX(double) rhsE = dot(te, xe);
    const double a2Schur = abs(lhsE - rhsE) / abs(lhsE);
    xa.gauss(d_rand.state);
    xa.updateAll();
    ya.gauss(d_rand.state);
    ya.updateAll();
    eo.applyFull(sa, xa, false);
    eo.applyFull(ta, ya, true);
    const COMPLEX(double) lhsA = dot(ya, sa);
    const COMPLEX(double) rhsA = dot(ta, xa);
    const double a2Full = abs(lhsA - rhsA) / abs(lhsA);
    const bool a2Passed = a2Schur <= 1e-12 && a2Full <= 1e-12;
    rootLogger.info("MDWF DSDR test A2 adjoints: Schur relative difference ", a2Schur, ", full ", a2Full,
                    ", passed = ", a2Passed);

    // 3. |M x|^2 = |D_W x|^2 + mu^2 |x|^2.
    eo.applyFull(sa, xa, false);
    eo0.applyFull(ta, xa, false);
    const double mx2 = real<double>(dot(sa, sa));
    const double dx2 = real<double>(dot(ta, ta));
    const double x2 = real<double>(dot(xa, xa));
    const double a3 = std::abs(mx2 - dx2 - mu * mu * x2) / mx2;
    const bool a3Passed = a3 <= 1e-12;
    rootLogger.info("MDWF DSDR test A3 gamma5-hermiticity: |M x|^2 = ", mx2, ", |D_W x|^2 + mu^2 |x|^2 = ",
                    dx2 + mu * mu * x2, ", relative difference ", a3, ", passed = ", a3Passed);

    // --- B. DSDR ladder. ---
    hmc.refreshMomenta();
    hmc.heatbath();
    bool b1Passed = true;
    for (size_t i = 0; i < ladder.factorCount(); i++) {
        Factor &f = ladder.factor(i);
        f.action();
        const double rel = std::abs(f.lastPseudofermionAction() - f.noiseNorm2()) / std::max(1.0, f.noiseNorm2());
        b1Passed = b1Passed && rel <= 1e-8;
        rootLogger.info("MDWF DSDR test B1 factor ", i, " (", f.muNumerator(), " / ", f.muDenominator(),
                        "): action ", f.lastPseudofermionAction(), ", noise ", f.noiseNorm2(), ", relative difference ",
                        rel, ", CG iterations ", f.lastIterations());
    }

    HostGauge gaugeHost(commBase, "MDWF_dsdr_gauge_host");
    HostGauge fA(commBase, "MDWF_dsdr_fa");
    HostGauge fB(commBase, "MDWF_dsdr_fb");
    gaugeHost = hmc.fermionGauge();
    bool b2Passed = false;
    {
        Factor eq(commBase, fgauge, param, twists.back(), twists.back(), "MDWF_dsdr_eq");
        eq.heatbath(d_rand.state);
        const double s = eq.action();
        const double phi2 = real<double>(dot(eq.phi(), eq.phi()));
        eq.force(fA, gaugeHost);
        ladder.forceFactor(0, fB, gaugeHost);
        const double ratio = mdwfDSDRForceRms<HaloDepth>(fA) / mdwfDSDRForceRms<HaloDepth>(fB);
        const double rel = std::abs(s - phi2) / phi2;
        b2Passed = rel <= 1e-10 && ratio <= 1e-6;
        rootLogger.info("MDWF DSDR test B2 equal twists ", twists.back(), ": action ", s, ", |phi|^2 ", phi2,
                        " (relative difference ", rel, "), force rms / lightest factor's ", ratio, ", passed = ",
                        b2Passed);
    }

    const double h = 1e-4;
    Gauge saved(commBase, "MDWF_dsdr_saved");
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
    bool b3Passed = true;
    {
        const double analytic = hmc.fermionForceAlongMomenta();
        const double num = numerical([&]() { return hmc.fermionAction(); });
        const double rel = std::abs(num - analytic) / std::max(std::abs(analytic), 1e-300);
        b3Passed = b3Passed && rel <= 1e-5;
        rootLogger.info("MDWF DSDR test B3 whole ladder: dS/dt numerical ", num, ", from the force ", analytic,
                        ", relative difference ", rel);
    }
    for (size_t i = 0; i < ladder.factorCount(); i++) {
        gaugeHost = hmc.fermionGauge();
        ladder.forceFactor(i, fA, gaugeHost);
        const double analytic = mdwfDSDRAlongMomenta<HaloDepth>(commBase, hmc.momenta(), fA);
        const double num = numerical([&]() {
            hmc.fermionGauge();   // re-derives the fermion links from the moved gauge field
            return ladder.factor(i).action();
        });
        const double rel = std::abs(num - analytic) / std::max(std::abs(analytic), 1e-300);
        b3Passed = b3Passed && rel <= 1e-5;
        rootLogger.info("MDWF DSDR test B3 factor ", i, ": dS/dt numerical ", num, ", from the force ", analytic,
                        ", relative difference ", rel, ", force rms ", mdwfDSDRForceRms<HaloDepth>(fA));
    }
    rootLogger.info("MDWF DSDR test B3 passed = ", b3Passed);

    hmc.refreshMomenta();
    hmc.heatbath();
    Gauge startGauge(commBase, "MDWF_dsdr_start_gauge");
    Gauge startMomenta(commBase, "MDWF_dsdr_start_momenta");
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
        hmc.integrate(stepCounts[i], 1);
        deltaH[i] = hmc.energy().total() - restored.total();
        rootLogger.info("MDWF DSDR test B4 scaling: steps = ", stepCounts[i], ", Delta H = ", deltaH[i]);
    }
    const double r1 = std::abs(deltaH[0]) / std::max(std::abs(deltaH[1]), 1e-300);
    const double r2 = std::abs(deltaH[1]) / std::max(std::abs(deltaH[2]), 1e-300);
    const bool b4Passed = r1 >= 3.0 && r1 <= 6.0 && r2 >= 3.0 && r2 <= 6.0;
    rootLogger.info("MDWF DSDR test B4 scaling: ratios ", r1, ", ", r2, " (leapfrog expects 4), passed = ", b4Passed);

    // --- C. Gauge action. ---
    gauge = startGauge;
    gauge.updateAll();
    bool c1Passed = false;
    {
        const double betaS = 3.0;
        GaugeAction<double, true, HaloDepth, R18> action(gauge);
        const double sSym = -(3.0 * betaS / 5.0) * static_cast<double>(action.symanzik());
        const double sRect = mdwfRectangleGaugeAction<HaloDepth>(gauge, 3.0 * betaS / 5.0, -1.0 / 12.0);
        Gauge fSymDev(commBase, "MDWF_dsdr_fsym");
        Gauge fRectDev(commBase, "MDWF_dsdr_frect");
        fSymDev.iterateOverBulkAllMu(MDWFHmcSymanzikGaugeForce<HaloDepth>(gauge.getAccessor(), betaS));
        fRectDev.iterateOverBulkAllMu(MDWFHmcRectangleGaugeForce<HaloDepth>(
            gauge.getAccessor(), 3.0 * betaS / 5.0, 1.0 + 8.0 / 12.0, -1.0 / 12.0));
        fA = fSymDev;
        fB = fRectDev;
        const double forceDiff = mdwfDSDRMaxLinkDiff<HaloDepth>(fA, fB);
        const double rel = std::abs(sSym - sRect) / std::abs(sSym);
        c1Passed = rel <= 1e-12 && forceDiff <= 1e-12;
        rootLogger.info("MDWF DSDR test C1 rectangle (3 beta/5, c1 = -1/12) vs Symanzik (beta = ", betaS, "): action ",
                        sRect, " vs ", sSym, " (relative difference ", rel, "), max force difference ", forceDiff,
                        " (force rms ", mdwfDSDRForceRms<HaloDepth>(fA), "), passed = ", c1Passed);
    }
    bool c2Passed = false;
    {
        hmc.refreshMomenta();
        Gauge fDev(commBase, "MDWF_dsdr_fgauge");
        hmc.gaugeForce(fDev);
        fA = fDev;
        const double analytic = mdwfDSDRAlongMomenta<HaloDepth>(commBase, hmc.momenta(), fA);
        const double num = numerical([&]() { return hmc.gaugeAction(); });
        const double rel = std::abs(num - analytic) / std::max(std::abs(analytic), 1e-300);
        c2Passed = rel <= 1e-6;
        rootLogger.info("MDWF DSDR test C2 rectangle gauge force (c1 = ", param.gauge_c1, "): dS/dt numerical ", num,
                        ", from the force ", analytic, ", relative difference ", rel, ", passed = ", c2Passed);
    }

    const bool passed = a1Passed && a2Passed && a3Passed && b1Passed && b2Passed && b3Passed && b4Passed && c1Passed
                        && c2Passed;
    rootLogger.info("MDWF DSDR test summary: A1 ", a1Passed, ", A2 ", a2Passed, ", A3 ", a3Passed, ", B1 ", b1Passed,
                    ", B2 ", b2Passed, ", B3 ", b3Passed, ", B4 ", b4Passed, ", C1 ", c1Passed, ", C2 ", c2Passed);
    return passed;
}

// G1 (plaquette, rectangle, gauge action in GPT's normalization) and G2 (gauge force at x0 and its rms).
template<size_t HaloDepth>
void mdwfGptGaugeChecks(CommunicationBase &commBase, DSDRGauge<HaloDepth> &gauge, const MDWFHmcParameters &param,
                        sitexyzt x0) {
    using Gauge = DSDRGauge<HaloDepth>;
    using HostGauge = DSDRHostGauge<HaloDepth>;
    typedef GIndexer<All, HaloDepth> GInd;
    GaugeAction<double, true, HaloDepth, R18> action(gauge);
    const double P = static_cast<double>(action.plaquette());
    const double R = static_cast<double>(action.rectangle());
    const double c1 = param.gauge_c1;
    const double c0 = 1.0 - 8.0 * c1;
    const double V = static_cast<double>(GInd::getLatData().globvol4);
    const double sGpt = param.beta * V * (c0 * (1.0 - P) * 6.0 + c1 * (1.0 - R) * 12.0);
    const double sOurs = mdwfRectangleGaugeAction<HaloDepth>(gauge, param.beta, c1);
    rootLogger.info(std::setprecision(14), "MDWF DSDR GPT check G1: plaquette ", P, ", rectangle ", R,
                    ", gauge action (GPT normalization) ", sGpt, ", ours ", sOurs, ", ours + beta V (6 c0 + 12 c1) ",
                    sOurs + param.beta * V * (6.0 * c0 + 12.0 * c1));
    Gauge fDev(commBase, "MDWF_dsdrg_fgauge");
    fDev.iterateOverBulkAllMu(MDWFHmcRectangleGaugeForce<HaloDepth>(gauge.getAccessor(), param.beta, c0, c1));
    HostGauge fHost(commBase, "MDWF_dsdrg_fgauge_host");
    fHost = fDev;
    rootLogger.info(std::setprecision(14), "MDWF DSDR GPT check G2: gauge force rms ",
                    mdwfDSDRForceRms<HaloDepth>(fHost));
    mdwfDSDRPrintLinks<HaloDepth>(fHost, x0, "MDWF DSDR GPT check G2 gauge force (ipdot = TA)");
}

template<size_t HaloDepth>
void runMDWFDSDRGptCheck(CommunicationBase &commBase, MDWFDSDRTestParameters &tp, const std::vector<double> &twists) {
    using Gauge = DSDRGauge<HaloDepth>;
    using HostGauge = DSDRHostGauge<HaloDepth>;
    using EvenOdd = MDWFTwistedWilsonEvenOdd<HaloDepth>;
    using SpinorAll = typename EvenOdd::SpinorAll;
    using Factor = MDWFDSDRRatioFermionAction<HaloDepth>;
    using FullOp = MDWFTwistedFullNormalOperator<HaloDepth>;
    using FullAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, 1, FullOp>;
    typedef GIndexer<All, HaloDepth> GInd;

    if (!tp.GaugefileName.isSet()) {
        throw std::runtime_error(stdLogger.fatal("MDWF DSDR test mode = gpt needs Gaugefile"));
    }
    sitexyzt x0(0, 0, 0, 0);
    if (tp.source_site.isSet()) {
        const std::vector<int> s = tp.source_site.get();
        if (s.size() != 4) {
            throw std::runtime_error(stdLogger.fatal("source_site needs four coordinates"));
        }
        x0 = sitexyzt(s[0], s[1], s[2], s[3]);
    }
    if ((x0.x + x0.y + x0.z + x0.t) % 2 != 0) {
        throw std::runtime_error(stdLogger.fatal("source_site must be an even site (x + y + z + t even)"));
    }

    MDWFHmcParameters param{};
    param.beta = tp.beta.isSet() ? tp.beta() : 1.725;
    param.M5 = tp.M5();
    param.max_iter = tp.max_iter();
    param.precision = tp.precision();
    param.antiperiodic_t = tp.antiperiodic_t() != 0;
    param.rectangle_gauge = true;
    param.gauge_c1 = tp.gauge_c1();
    param.dsdr_twists = twists;

    Gauge gauge(commBase, "MDWF_dsdrg_gauge");
    rootLogger.info("MDWF DSDR GPT check: reading ", tp.GaugefileName());
    gauge.readconf_nersc(tp.GaugefileName());
    gauge.updateAll();
    MDWFFermionGauge<HaloDepth> fermionGauge(commBase, gauge, param.antiperiodic_t, "MDWF_dsdrg_bc");
    Gauge &fgauge = fermionGauge.get();
    HostGauge gaugeHost(commBase, "MDWF_dsdrg_ghost");
    gaugeHost = fgauge;
    rootLogger.info("MDWF DSDR GPT check: M5 = ", param.M5, ", beta = ", param.beta, ", c1 = ", param.gauge_c1,
                    ", fermion BCs in time: ", param.antiperiodic_t ? "antiperiodic" : "periodic", ", source site (",
                    x0.x, ",", x0.y, ",", x0.z, ",", x0.t, "), CG precision ", param.precision);

    // G1, G2.
    mdwfGptGaugeChecks<HaloDepth>(commBase, gauge, param, x0);

    // G3. Local traces of (M^+ M)^-1.
    for (const double mu : twists) {
        EvenOdd eo(fgauge, param.M5, mu, "MDWF_dsdrg_eo");
        FullOp normal(eo, commBase, "MDWF_dsdrg_nrm");
        FullAdapter adapter(normal);
        SpinorAll src(commBase, "MDWF_dsdrg_src");
        SpinorAll sol(commBase, "MDWF_dsdrg_sol");
        MDWFCoupledCG<double, FullAdapter> cg;
        double trace = 0.0;
        int iterations = 0;
        for (int a = 0; a < 12; a++) {
            src.template iterateOverBulk<BLOCKSIZE>(MDWFPointSourceField<double, All, HaloDepth>(x0, a));
            src.updateAll();
            const MDWFCoupledCGResult<double> r = cg.invert(adapter, sol, src, param.max_iter, param.precision, true);
            if (!r.converged) {
                throw std::runtime_error(stdLogger.fatal("MDWF DSDR GPT check: trace solve did not converge"));
            }
            iterations = std::max(iterations, r.iterations);
            trace += real<double>(adapter.dotProduct5D(src, sol));
        }
        rootLogger.info(std::setprecision(14), "MDWF DSDR GPT check G3: mu = ", mu,
                        ": T = sum_a e_a^+ (M^+ M)^-1 e_a = ", trace, " (CG iterations up to ", iterations, ")");
    }

    // G4. Point-source factor actions and forces.
    HostGauge fPart(commBase, "MDWF_dsdrg_fpart");
    HostGauge fSum(commBase, "MDWF_dsdrg_fsum");
    for (size_t i = 0; i + 1 < twists.size(); i++) {
        Factor factor(commBase, fgauge, param, twists[i], twists[i + 1], "MDWF_dsdrg_f" + std::to_string(i));
        mdwfDSDRZero<HaloDepth>(fSum, commBase);
        double sSum = 0.0;
        for (int a = 0; a < 12; a++) {
            factor.phi().template iterateOverBulk<BLOCKSIZE>(MDWFPointSourceField<double, Even, HaloDepth>(x0, a));
            factor.phi().updateAll();
            sSum += factor.action();
            factor.force(fPart, gaugeHost);
            mdwfHmcAddHostForce<HaloDepth>(fSum, fPart);
        }
        const std::string label = "MDWF DSDR GPT check G4 factor " + std::to_string(twists[i]) + " / "
                                  + std::to_string(twists[i + 1]);
        rootLogger.info(std::setprecision(14), label, ": sum_a S(phi = e_a) = ", sSum, ", summed force rms ",
                        mdwfDSDRForceRms<HaloDepth>(fSum));
        mdwfDSDRPrintLinks<HaloDepth>(fSum, x0, label + " summed force (ipdot = TA)");
    }
}

// Point source e_a at (x, s) of an even-site 5D MDWF field.
template<class floatT, size_t HaloDepth, size_t Ls>
struct MDWFPointSource5D {
    sitexyzt _x;
    size_t _s;
    int _component;

    MDWFPointSource5D(sitexyzt x, size_t s, int component) : _x(x), _s(s), _component(component) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);
        const sitexyzt c = site.coord;
        if (site.stack == _s && c.x == _x.x && c.y == _x.y && c.z == _x.z && c.t == _x.t) {
            out.data[_component] = COMPLEX(floatT)(1.0, 0.0);
        }
        return out;
    }
};

/*
 * G6: light (Mobius MDWF) two-flavour ratio factors against GPT (gpt_mdwf_check.py). Per pair (m_l, m_h), A = Mhat(m_l),
 * B = Mhat(m_h) (even-site Schur complements), summed over the point sources phi = e_a at (source_site, s), s in
 * mdwf_s_sources (default 0 and Ls - 1, a set closed under s -> Ls - 1 - s), a = 12 spin-colour components:
 *   S_SQ  = phi^+ B (A^+ A)^-1 B^+ phi   (MDWFEvenOddTwoFlavorRatioFermionAction, the production light factor),
 *   S_GPT = phi^+ B^+ (A A^+)^-1 B phi   (GPT two_flavor_ratio_evenodd_schur([M(m_l), M(m_h)]), = |A^-1 B phi|^2),
 * and the summed force of S_SQ (rms, four links at source_site). Both forms sample det(A^+A) / det(B^+B), but differ
 * point by point, so each code prints both actions (GPT: also the gradient of the S_SQ form).
 */
template<size_t HaloDepth, size_t Ls>
void runMDWFLightGptCheck(CommunicationBase &commBase, MDWFDSDRTestParameters &tp) {
    using Gauge = DSDRGauge<HaloDepth>;
    using HostGauge = DSDRHostGauge<HaloDepth>;
    using Types = MDWFEvenOddTypes<HaloDepth, Ls>;
    using EvenOdd = typename Types::EvenOdd;
    using Spinor = typename Types::SpinorE;
    using Factor = MDWFEvenOddTwoFlavorRatioFermionAction<HaloDepth, Ls>;
    using CG = MDWFCoupledCG<double, typename Types::Adapter>;

    if (!tp.GaugefileName.isSet() || !tp.mdwf_pairs.isSet()) {
        throw std::runtime_error(stdLogger.fatal("MDWF light GPT check needs Gaugefile and mdwf_pairs"));
    }
    const std::vector<double> pairs = tp.mdwf_pairs.get();
    if (pairs.empty() || pairs.size() % 2 != 0) {
        throw std::runtime_error(stdLogger.fatal("mdwf_pairs needs pairs m_light m_heavy"));
    }
    std::vector<size_t> sSources{0, Ls - 1};
    if (tp.mdwf_s_sources.isSet()) {
        sSources.clear();
        for (int s : tp.mdwf_s_sources.get()) {
            sSources.push_back(static_cast<size_t>(s));
        }
    }
    sitexyzt x0(0, 0, 0, 0);
    if (tp.source_site.isSet()) {
        const std::vector<int> c = tp.source_site.get();
        x0 = sitexyzt(c[0], c[1], c[2], c[3]);
    }
    if ((x0.x + x0.y + x0.z + x0.t) % 2 != 0) {
        throw std::runtime_error(stdLogger.fatal("source_site must be an even site (x + y + z + t even)"));
    }

    MDWFHmcParameters param{};
    param.beta = tp.beta.isSet() ? tp.beta() : 1.829;
    param.M5 = tp.M5();
    param.b5 = tp.b5();
    param.csw = 0.0;
    param.max_iter = tp.max_iter();
    param.precision = tp.precision();
    param.antiperiodic_t = tp.antiperiodic_t() != 0;
    param.rectangle_gauge = true;
    param.gauge_c1 = tp.gauge_c1();

    Gauge gauge(commBase, "MDWF_lg_gauge");
    rootLogger.info("MDWF light GPT check: reading ", tp.GaugefileName());
    gauge.readconf_nersc(tp.GaugefileName());
    gauge.updateAll();
    MDWFFermionGauge<HaloDepth> fermionGauge(commBase, gauge, param.antiperiodic_t, "MDWF_lg_bc");
    Gauge &fgauge = fermionGauge.get();
    HostGauge gaugeHost(commBase, "MDWF_lg_ghost");
    gaugeHost = fgauge;
    std::ostringstream sl;
    for (size_t s : sSources) {
        sl << " " << s;
    }
    rootLogger.info("MDWF light GPT check: Ls = ", Ls, ", M5 = ", param.M5, ", b5 = ", param.b5, ", c5 = ",
                    param.b5 - 1.0, ", fermion BCs in time: ", param.antiperiodic_t ? "antiperiodic" : "periodic",
                    ", source site (", x0.x, ",", x0.y, ",", x0.z, ",", x0.t, "), slices", sl.str(),
                    ", CG precision ", param.precision);

    mdwfGptGaugeChecks<HaloDepth>(commBase, gauge, param, x0);

    HostGauge fPart(commBase, "MDWF_lg_fpart");
    HostGauge fSum(commBase, "MDWF_lg_fsum");
    for (size_t p = 0; p < pairs.size(); p += 2) {
        const double ml = pairs[p];
        const double mh = pairs[p + 1];
        Factor factor(commBase, fgauge, param, ml, mh, "MDWF_lg_f" + std::to_string(p / 2));
        EvenOdd eoA(fgauge, param.M5, ml, param.b5, 0.0, "MDWF_lg_eoa");
        EvenOdd eoB(fgauge, param.M5, mh, param.b5, 0.0, "MDWF_lg_eob");
        eoA.refresh();
        eoB.refresh();
        typename Types::NormalOp normalA(eoA, commBase, "MDWF_lg_nrma");
        typename Types::Adapter adapterA(normalA);
        Spinor t1(commBase, "MDWF_lg_t1"), t2(commBase, "MDWF_lg_t2"), x(commBase, "MDWF_lg_x");
        mdwfDSDRZero<HaloDepth>(fSum, commBase);
        double sSQ = 0.0;
        double sGPT = 0.0;
        int iterations = 0;
        for (size_t s : sSources) {
            for (int a = 0; a < 12; a++) {
                factor.phi().template iterateOverBulk<BLOCKSIZE>(MDWFPointSource5D<double, HaloDepth, Ls>(x0, s, a));
                factor.phi().updateAll();
                sSQ += factor.action();
                factor.force(fPart, gaugeHost);
                mdwfHmcAddHostForce<HaloDepth>(fSum, fPart);
                // S_GPT = |x|^2, x = (A^+ A)^-1 A^+ B phi = A^-1 B phi.
                eoB.schur(t1, factor.phi(), false);
                eoA.schur(t2, t1, true);
                CG cg;
                const MDWFCoupledCGResult<double> r = cg.invert(adapterA, x, t2, param.max_iter, param.precision, true);
                if (!r.converged) {
                    throw std::runtime_error(stdLogger.fatal("MDWF light GPT check: S_GPT solve did not converge"));
                }
                iterations = std::max(iterations, r.iterations);
                sGPT += adapterA.norm2(x);
            }
        }
        std::ostringstream label;
        label << std::setprecision(10) << "MDWF light GPT check G6 factor " << ml << " / " << mh;
        rootLogger.info(std::setprecision(14), label.str(), ": sum S_SQ = ", sSQ, ", sum S_GPT = ", sGPT,
                        ", summed S_SQ force rms ", mdwfDSDRForceRms<HaloDepth>(fSum), " (CG iterations up to ",
                        std::max(iterations, factor.lastIterations()), ")");
        mdwfDSDRPrintLinks<HaloDepth>(fSum, x0, label.str() + " summed S_SQ force (ipdot = TA)");
    }
}

// Common MDWFHmcParameters for the EOFA modes.
inline MDWFHmcParameters mdwfEofaTestParameters(MDWFDSDRTestParameters &tp) {
    MDWFHmcParameters param{};
    param.beta = tp.beta.isSet() ? tp.beta() : 1.8;
    param.M5 = tp.M5();
    param.b5 = tp.b5();
    param.csw = 0.0;
    param.mf = tp.eofa_m1();
    param.pv_mass = tp.eofa_m2();
    param.rhmc.ms = tp.eofa_m1();
    param.max_iter = tp.max_iter();
    param.precision = tp.precision();
    param.antiperiodic_t = tp.antiperiodic_t() != 0;
    param.rectangle_gauge = true;
    param.gauge_c1 = tp.gauge_c1();
    param.tau = 0.2;
    param.steps = 4;
    param.gauge_substeps = 1;
    param.eofa.lambda_low = tp.eofa_lambda_low();
    param.eofa.lambda_high = tp.eofa_lambda_high();
    param.eofa.error = tp.eofa_error();
    param.eofa.zolotarev_order = tp.eofa_zolotarev();
    return param;
}

/*
 * mode = eofa: EOFA action (MDWFEofaFermionAction.h) unit tests on the configuration Gaugefile (or a random one):
 *   E1. two-mass operator M(m_+, m_-), m_+ != m_-: M_oo M_oo^-1 = 1 (and dagger), and the even/odd full solves
 *       M x = b, M^+ y = b checked with the block form of M (relative 1e-9);
 *   E2. K hermitian: <a, K b> = <K a, b> (relative 1e-9);
 *   E3. the force identity g5 R D_-^+ M^-+ R g5 I_u P phi = M^-1 D_- I_u P phi for both chiralities (1e-8);
 *   E4. heatbath: S(phi) = |eta|^2 (relative 1e-8, with the Remez error eofa_error);
 *   E5. finite-difference force: dS/dt along exp(i t P) U against i sum tr(P ipdot) (relative 1e-5).
 */
template<size_t HaloDepth, size_t Ls>
bool runMDWFEofaUnitTest(CommunicationBase &commBase, MDWFDSDRTestParameters &tp) {
    using Gauge = DSDRGauge<HaloDepth>;
    using Action = MDWFEofaRatioFermionAction<HaloDepth, Ls>;
    using Hmc = MDWFHmcDriver<HaloDepth, Ls, Action>;
    using Solver = typename Action::Solver;
    using EvenOdd = typename Solver::EvenOdd;
    using SpinorAll = typename EvenOdd::SpinorAll;
    using SpinorE = typename EvenOdd::SpinorE;
    using SpinorO = typename EvenOdd::SpinorO;
    using Spinor4 = typename Action::Spinor;

    const MDWFHmcParameters param = mdwfEofaTestParameters(tp);
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20261006);
    d_rand = h_rand;
    Gauge gauge(commBase, "MDWF_eofat_gauge");
    if (tp.GaugefileName.isSet()) {
        gauge.readconf_nersc(tp.GaugefileName());
    } else {
        gauge.random(d_rand.state);
    }
    gauge.updateAll();
    Hmc hmc(commBase, gauge, param, d_rand.state);
    Gauge &fgauge = hmc.fermionGauge();
    Action &action = hmc.fermion();
    rootLogger.info("MDWF EOFA test: Ls = ", Ls, ", M5 = ", param.M5, ", b5 = ", param.b5, ", m1 = ", param.rhmc.ms,
                    ", m2 = ", param.pv_mass, ", heatbath approximation order ", action.inverseSqrt().order,
                    " on [", param.eofa.lambda_low, ", ", param.eofa.lambda_high, "] (error ",
                    action.inverseSqrt().max_relative_error, ")");
    auto rel = [](double a, double b) { return std::abs(a - b) / std::max(std::abs(b), 1e-300); };
    auto norm5 = [](auto &v) {
        double n = 0.0;
        for (const auto &d : v.dotProductStacked(v)) {
            n += real<double>(d);
        }
        return std::sqrt(n);
    };

    // E0: fused Schur operator (c_sw = 0) against the block path, double and float, one and two boundary masses.
    bool e0Passed = true;
    {
        using EvenOddF = MDWFMobiusCloverEvenOdd<float, HaloDepth, HaloDepth, Ls>;
        using SpinorEF = typename EvenOddF::SpinorE;
        Gaugefield<float, true, HaloDepth, R18> gaugeF(commBase, "MDWF_eofat_gaugef");
        gaugeF.convert_precision(fgauge);
        gaugeF.updateAll();
        const bool saved = mdwfFusedSchur();
        double errD = 0.0, errF = 0.0;
        const double masses[2][2] = {{0.1, 0.1}, {0.1, 0.7}};
        for (const auto &m : masses) {
            EvenOdd eo(fgauge, param.M5, m[0], param.b5, 0.0, "MDWF_eofat_fe");
            eo.setMasses(m[0], m[1]);
            eo.refresh();
            EvenOddF eoF(gaugeF, static_cast<float>(param.M5), 0.1f, static_cast<float>(param.b5), 0.0f,
                         "MDWF_eofat_ff");
            eoF.refreshFrom(eo);
            SpinorAll all(commBase, "MDWF_eofat_fall");
            SpinorE v(commBase, "MDWF_eofat_fv"), r1(commBase, "MDWF_eofat_fr1"), r2(commBase, "MDWF_eofat_fr2");
            SpinorO junk(commBase, "MDWF_eofat_fjunk");
            SpinorEF vF(commBase, "MDWF_eofat_fvf"), r1F(commBase, "MDWF_eofat_fr1f"), r2F(commBase, "MDWF_eofat_fr2f");
            all.gauss(d_rand.state);
            EvenOdd::split(v, junk, all);
            vF.convert_precision(v);
            for (int dagger = 0; dagger < 2; dagger++) {
                mdwfFusedSchur() = false;
                eo.schur(r1, v, dagger);
                eoF.schur(r1F, vF, dagger);
                mdwfFusedSchur() = true;
                eo.schur(r2, v, dagger);
                eoF.schur(r2F, vF, dagger);
                r2.template axpyThisB<64>(-1.0, r1);
                errD = std::max(errD, norm5(r2) / norm5(r1));
                r2F.template axpyThisB<64>(-1.0f, r1F);
                errF = std::max(errF, norm5(r2F) / norm5(r1F));
            }
        }
        mdwfFusedSchur() = saved;
        e0Passed = errD <= 1e-13 && errF <= 1e-5;
        rootLogger.info("MDWF EOFA test E0 fused Schur operator vs block path (Mhat and Mhat^+, m_+ = m_- and "
                        "m_+ != m_-): max relative difference double ", errD, ", float ", errF, ", passed = ",
                        e0Passed);
    }

    // E1
    bool e1Passed = true;
    {
        const double mPlus = 0.1, mMinus = 0.7;
        Solver solver(fgauge, param.M5, mPlus, param.b5, 0.0, "MDWF_eofat_sol");
        EvenOdd &eo = solver.evenOdd();
        eo.setMasses(mPlus, mMinus);
        eo.refresh();
        SpinorAll b(commBase, "MDWF_eofat_b"), x(commBase, "MDWF_eofat_x"), mx(commBase, "MDWF_eofat_mx");
        SpinorE xe(commBase, "MDWF_eofat_xe"), ae(commBase, "MDWF_eofat_ae"), be(commBase, "MDWF_eofat_be");
        SpinorO xo(commBase, "MDWF_eofat_xo"), ao(commBase, "MDWF_eofat_ao"), bo(commBase, "MDWF_eofat_bo");
        b.gauss(d_rand.state);
        b.updateAll();
        double invErr = 0.0;
        EvenOdd::split(xe, xo, b);
        for (int dagger = 0; dagger < 2; dagger++) {
            eo.Moo(ao, xo, dagger);
            eo.MooInv(bo, ao, dagger);
            bo.template axpyThisB<64>(-1.0, xo);
            invErr = std::max(invErr, norm5(bo) / norm5(xo));
        }
        double solveErr = 0.0;
        for (int dagger = 0; dagger < 2; dagger++) {
            if (dagger) {
                solver.solveDagger(x, b, param.max_iter, 1e-12);
            } else {
                solver.solve(x, b, param.max_iter, 1e-12);
            }
            EvenOdd::split(xe, xo, x);
            if (!dagger) {
                eo.Mee(ae, xe);
                eo.Meo(be, xo);
                ae += be;
                eo.Moe(ao, xe);
                eo.Moo(bo, xo);
                ao += bo;
            } else {
                eo.Mee(ae, xe, true);
                eo.MoeDagger(be, xo);
                ae += be;
                eo.MeoDagger(ao, xe);
                eo.Moo(bo, xo, true);
                ao += bo;
            }
            EvenOdd::merge(mx, ae, ao);
            mx.template axpyThisB<64>(-1.0, b);
            solveErr = std::max(solveErr, norm5(mx) / norm5(b));
        }
        e1Passed = invErr <= 1e-12 && solveErr <= 1e-9;
        rootLogger.info("MDWF EOFA test E1 two-mass operator (m_+ = ", mPlus, ", m_- = ", mMinus,
                        "): max |M_oo^-1 M_oo v - v| / |v| = ", invErr, ", max |M x - b| / |b| (x = M^-1 b, M^-+ b) = ",
                        solveErr, ", passed = ", e1Passed);
    }

    // E6: the full-operator mixed-precision solver (EOFA heatbath / action path) with m_+ != m_-, fast and original
    // mixed CG, against the double solver.
    bool e6Passed = true;
    {
        using MixedSolver = typename Action::MixedSolver;
        const double mPlus = 0.1, mMinus = 0.7;
        Solver solver(fgauge, param.M5, mPlus, param.b5, 0.0, "MDWF_eofat_e6d");
        MixedSolver mixed(fgauge, param.M5, mPlus, param.b5, 0.0, "MDWF_eofat_e6m", 0.1);
        solver.evenOdd().setMasses(mPlus, mMinus);
        mixed.evenOdd().setMasses(mPlus, mMinus);
        SpinorAll b(commBase, "MDWF_eofat_e6b"), xd(commBase, "MDWF_eofat_e6xd"), xm(commBase, "MDWF_eofat_e6xm");
        b.gauss(d_rand.state);
        b.updateAll();
        const bool savedFast = mdwfFastMixedCG();
        double err[2] = {0.0, 0.0};
        for (int fast = 0; fast < 2; fast++) {
            mdwfFastMixedCG() = fast != 0;
            for (int dagger = 0; dagger < 2; dagger++) {
                if (dagger) {
                    solver.solveDagger(xd, b, param.max_iter, 1e-10);
                    mixed.solveDagger(xm, b, param.max_iter, 1e-10);
                } else {
                    solver.solve(xd, b, param.max_iter, 1e-10);
                    mixed.solve(xm, b, param.max_iter, 1e-10);
                }
                xm.template axpyThisB<64>(-1.0, xd);
                err[fast] = std::max(err[fast], norm5(xm) / norm5(xd));
            }
        }
        mdwfFastMixedCG() = savedFast;
        e6Passed = err[0] <= 1e-7 && err[1] <= 1e-7;
        rootLogger.info("MDWF EOFA test E6 mixed full-operator solves (m_+ = ", mPlus, ", m_- = ", mMinus,
                        "): max |x_mixed - x_double| / |x_double|, original CG ", err[0], ", fast CG ", err[1],
                        ", passed = ", e6Passed);
    }

    // E2
    bool e2Passed = false;
    {
        Spinor4 a(commBase, "MDWF_eofat_4a"), c(commBase, "MDWF_eofat_4c"), ka(commBase, "MDWF_eofat_4ka"),
            kc(commBase, "MDWF_eofat_4kc");
        a.gauss(d_rand.state);
        c.gauss(d_rand.state);
        action.applyKForTest(ka, a);
        action.applyKForTest(kc, c);
        const COMPLEX(double) left = a.dotProductStacked(kc)[0];
        const COMPLEX(double) right = ka.dotProductStacked(c)[0];
        const double diff = abs(left - right) / abs(left);
        const double aka = real<double>(a.dotProductStacked(ka)[0]) / real<double>(a.dotProductStacked(a)[0]);
        e2Passed = diff <= 1e-9;
        rootLogger.info("MDWF EOFA test E2 hermiticity: <a, K c> = ", left, ", <K a, c> = ", right,
                        ", relative difference ", diff, "; <a, K a> / <a, a> = ", aka, ", passed = ", e2Passed);
    }

    // E3, E4 (heatbath fills phi)
    hmc.refreshMomenta();
    hmc.heatbath();
    const double e4Err = action.lastHeatbathError();
    const bool e4Passed = e4Err <= 1e-8;
    rootLogger.info("MDWF EOFA test E4 heatbath: |S(phi) / |eta|^2 - 1| = ", e4Err, ", passed = ", e4Passed);
    const double e3Plus = action.forceIdentityCheck(true);
    const double e3Minus = action.forceIdentityCheck(false);
    const bool e3Passed = e3Plus <= 1e-8 && e3Minus <= 1e-8;
    rootLogger.info("MDWF EOFA test E3 force identity x = g5 R D_-^+ y: relative difference P_+ ", e3Plus, ", P_- ",
                    e3Minus, ", passed = ", e3Passed);

    // E5
    bool e5Passed = false;
    {
        const double h = 1e-4;
        Gauge saved(commBase, "MDWF_eofat_saved");
        const double analytic = hmc.fermionForceAlongMomenta();
        saved = gauge;
        hmc.evolveQ(h);
        const double plus = hmc.fermionAction();
        gauge = saved;
        gauge.updateAll();
        hmc.evolveQ(-h);
        const double minus = hmc.fermionAction();
        gauge = saved;
        gauge.updateAll();
        const double num = (plus - minus) / (2.0 * h);
        e5Passed = rel(num, analytic) <= 1e-5;
        rootLogger.info("MDWF EOFA test E5 finite-difference force: dS/dt numerical ", num, ", from the force ",
                        analytic, ", relative difference ", rel(num, analytic), ", passed = ", e5Passed);
    }
    const bool passed = e0Passed && e6Passed && e1Passed && e2Passed && e3Passed && e4Passed && e5Passed;
    rootLogger.info("MDWF EOFA test summary: E0 ", e0Passed, ", E6 ", e6Passed, ", E1 ", e1Passed, ", E2 ", e2Passed, ", E3 ", e3Passed, ", E4 ", e4Passed,
                    ", E5 ", e5Passed);
    return passed;
}

/*
 * mode = gpteofa (G7, against gpt_eofa_check.py): G1, G2 and, for exact_one_flavor_ratio(m1, m2) with the 4D point
 * sources phi = e_a at source_site (12 spin-colour components), sum_a S(phi = e_a) and the summed force.
 */
template<size_t HaloDepth, size_t Ls>
void runMDWFEofaGptCheck(CommunicationBase &commBase, MDWFDSDRTestParameters &tp) {
    using Gauge = DSDRGauge<HaloDepth>;
    using HostGauge = DSDRHostGauge<HaloDepth>;
    using Action = MDWFEofaRatioFermionAction<HaloDepth, Ls>;
    if (!tp.GaugefileName.isSet()) {
        throw std::runtime_error(stdLogger.fatal("MDWF EOFA GPT check needs Gaugefile"));
    }
    sitexyzt x0(0, 0, 0, 0);
    if (tp.source_site.isSet()) {
        const std::vector<int> c = tp.source_site.get();
        x0 = sitexyzt(c[0], c[1], c[2], c[3]);
    }
    MDWFHmcParameters param = mdwfEofaTestParameters(tp);
    param.beta = tp.beta.isSet() ? tp.beta() : 1.829;
    Gauge gauge(commBase, "MDWF_eofag_gauge");
    gauge.readconf_nersc(tp.GaugefileName());
    gauge.updateAll();
    MDWFFermionGauge<HaloDepth> fermionGauge(commBase, gauge, param.antiperiodic_t, "MDWF_eofag_bc");
    Gauge &fgauge = fermionGauge.get();
    HostGauge gaugeHost(commBase, "MDWF_eofag_ghost");
    gaugeHost = fgauge;
    rootLogger.info("MDWF EOFA GPT check: Ls = ", Ls, ", M5 = ", param.M5, ", b5 = ", param.b5, ", m1 = ",
                    param.rhmc.ms, ", m2 = ", param.pv_mass, ", source site (", x0.x, ",", x0.y, ",", x0.z, ",", x0.t,
                    "), CG precision ", param.precision);
    mdwfGptGaugeChecks<HaloDepth>(commBase, gauge, param, x0);

    Action action(commBase, fgauge, param, "MDWF_eofag_act");
    HostGauge fPart(commBase, "MDWF_eofag_fpart");
    HostGauge fSum(commBase, "MDWF_eofag_fsum");
    mdwfDSDRZero<HaloDepth>(fSum, commBase);
    double sSum = 0.0;
    for (int a = 0; a < 12; a++) {
        action.phi().template iterateOverBulk<BLOCKSIZE>(MDWFPointSourceField<double, All, HaloDepth>(x0, a));
        action.phi().updateAll();
        sSum += action.action();
        action.force(fPart, gaugeHost);
        mdwfHmcAddHostForce<HaloDepth>(fSum, fPart);
    }
    std::ostringstream label;
    label << std::setprecision(10) << "MDWF EOFA GPT check G7 eofa " << param.rhmc.ms << " / " << param.pv_mass;
    rootLogger.info(std::setprecision(14), label.str(), ": sum_a S(phi = e_a) = ", sSum, ", summed force rms ",
                    mdwfDSDRForceRms<HaloDepth>(fSum));
    mdwfDSDRPrintLinks<HaloDepth>(fSum, x0, label.str() + " summed force (ipdot = TA)");
}

/*
 * mode = gptmd (regression test T7, tests/regression/gpt/gpt_md_parity.py): ONE molecular-dynamics trajectory without
 * accept/reject, number by number against GPT. Start configuration Gaugefile; momenta and heatbath noise from the
 * hash fields of MDWFHashFields.h (identical in GPT): momenta seed md_seed vector 1, DSDR factor i noise eta (even
 * sites) vector 10 + i, EOFA noise vector 20, eta scaled by 1/sqrt 2; phi from each action's own heatbath
 * (equilibrium pseudofermions, S = |eta|^2, dH O(1) as in production). Action terms: Iwasaki gauge + DSDR (dsdr_twists) + EOFA (eofa_m1 / eofa_m2);
 * integrator OMF2 force-gradient (md_scheme 2) with md_steps fermion steps x md_gauge_steps gauge steps over md_tau
 * (= GPT tau / sqrt 2, GPT momenta pi = P / sqrt 2). Solves at `precision` (double). Printed: H and its parts before and
 * after, dH, plaquettes, and the links at the origin after the trajectory.
 */
template<size_t HaloDepth, size_t Ls>
void runMDWFMdGptCheck(CommunicationBase &commBase, MDWFDSDRTestParameters &tp, const std::vector<double> &twists) {
    using Gauge = DSDRGauge<HaloDepth>;
    using HostGauge = DSDRHostGauge<HaloDepth>;
    using Action = MDWFSumFermionAction<HaloDepth, MDWFDSDRFermionAction<HaloDepth>, MDWFEofaRatioFermionAction<HaloDepth, Ls>>;
    using Hmc = MDWFHmcDriver<HaloDepth, Ls, Action>;
    if (!tp.GaugefileName.isSet()) {
        throw std::runtime_error(stdLogger.fatal("mode = gptmd needs Gaugefile"));
    }
    MDWFHmcParameters param = mdwfEofaTestParameters(tp);
    param.beta = tp.beta.isSet() ? tp.beta() : 1.829;
    param.dsdr_twists = twists;
    param.tau = tp.md_tau();
    param.steps = tp.md_steps();
    param.gauge_substeps = tp.md_gauge_steps();
    param.md_scheme = 2;
    Gauge gauge(commBase, "MDWF_md_gauge");
    gauge.readconf_nersc(tp.GaugefileName());
    gauge.updateAll();
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20261007);
    d_rand = h_rand;
    Hmc hmc(commBase, gauge, param, d_rand.state);

    const uint64_t seed = mdwfSeedFromString(tp.md_seed());
    hmc.momenta().iterateOverBulkAllMu(MDWFHashMomentum<HaloDepth>(seed, 1));
    hmc.momenta().updateAll();
    auto &dsdr = hmc.fermion().first();
    auto &eofa = hmc.fermion().second();
    // Heatbath with hash noise eta (<|eta|^2> = 1 per complex component, as GPT's rng.cnormal(sigma = 1/sqrt 2)):
    // DSDR phi = Mhat_b^-1 Mhat_f eta, EOFA phi = r(K) eta, so that S(phi) = |eta|^2 (equilibrium pseudofermions).
    const double etaScale = 1.0 / std::sqrt(2.0);
    std::ostringstream hb;
    hb << std::setprecision(16) << "T7 heatbath:";
    for (size_t i = 0; i < dsdr.factorCount(); i++) {
        dsdr.factor(i).noise().template iterateOverBulk<BLOCKSIZE>(MDWFHashSpinor<Even, HaloDepth>(seed, 10 + i, etaScale));
        dsdr.factor(i).heatbathFromNoise();
        hb << (i ? ", " : " ") << "eta2_dsdr" << i << " = " << dsdr.factor(i).noiseNorm2() << ", phi2_dsdr" << i
           << " = " << dsdr.factor(i).phiNorm2();
    }
    eofa.noise().template iterateOverBulk<BLOCKSIZE>(MDWFHashSpinor<All, HaloDepth>(seed, 20, etaScale));
    eofa.heatbathFromNoise();
    hb << ", eta2_eofa = " << eofa.noiseNorm2() << ", phi2_eofa = " << eofa.phiNorm2();
    rootLogger.info(hb.str());
    rootLogger.info("T7 EOFA heatbath approximation: order ", eofa.inverseSqrt().order, " on [", param.eofa.lambda_low,
                    ", ", param.eofa.lambda_high, "], error ", param.eofa.error);
    rootLogger.info("T7 setup: tau ", param.tau, ", OMF2 force-gradient ", param.steps, " x ", param.gauge_substeps,
                    ", beta ", param.beta, ", c1 ", param.gauge_c1, ", DSDR M5 ", param.M5, ", EOFA ", param.rhmc.ms, " / ",
                    param.pv_mass, ", Ls ", Ls, ", b5 ", param.b5, ", CG precision ", param.precision, ", seed '",
                    tp.md_seed(), "'");

    auto report = [&](const char *when) {
        const MDWFHmcEnergy e = hmc.energy();
        std::ostringstream line;
        line << std::setprecision(16) << "T7 " << when << ": K = " << e.kinetic << ", S_gauge = " << e.gauge;
        for (size_t i = 0; i < dsdr.factorCount(); i++) {
            line << ", S_dsdr" << i << " = " << dsdr.factor(i).action();
        }
        line << ", S_eofa = " << eofa.action() << ", H = " << e.total();
        GaugeAction<double, true, HaloDepth, R18> ga(gauge);
        line << ", plaquette = " << static_cast<double>(ga.plaquette());
        rootLogger.info(line.str());
        return e.total();
    };
    const double h0 = report("before");
    hmc.integrate(param.steps);
    const double h1 = report("after");
    rootLogger.info(std::setprecision(16), "T7 dH = ", h1 - h0, ", fermion force evaluations ", hmc.forceEvaluations());
    HostGauge host(commBase, "MDWF_md_host");
    host = gauge;
    const SU3Accessor<double, R18> acc = host.getAccessor();
    typedef GIndexer<All, HaloDepth> GInd;
    for (int mu = 0; mu < 4; mu++) {
        const SU3<double> u = acc.getLink(GInd::getSiteMu(GInd::getSite(0, 0, 0, 0), mu));
        std::ostringstream line;
        line << std::setprecision(16) << "T7 link after mu = " << mu << ":";
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                line << " (" << real(u(r, c)) << "," << imag(u(r, c)) << ")";
            }
        }
        rootLogger.info(line.str());
    }
}

/*
 * mode = exact (exact-comparison suite tests/exact8; GPT: tests/exact8/gpt/gpt_exact_hmc.py): exact_traj HMC
 * trajectories of the full production action (Iwasaki gauge + light Hasenbusch ladder in SIMULATeQCD's form
 * S = phi^+ B (A^+ A)^-1 B^+ phi + EOFA strange + DSDR) from Gaugefile, with every random number from the hash fields of
 * MDWFHashFields.h (identical in GPT). Trajectory n (1, 2, ...), B = 1000 n: momenta vector B + 1, DSDR factor i noise
 * B + 10 + i (4D even), EOFA noise B + 20 (4D), light factor i noise B + 100 + 32 i + s (5D even, slice s), all noise
 * scaled by 1/sqrt 2 (GPT's cnormal(sigma = 2^-1/2)); Metropolis uniform u = hash uniform(seed, 999, n),
 * accept if dH <= 0 or u < exp(-dH). Integrator OMF2 force-gradient md_steps x md_gauge_steps over md_tau (= GPT tau /
 * sqrt 2), one fermion level as GPT; the trajectory is integrated one outer step at a time (integrateTime) and at every
 * step boundary k = 0 ... md_steps all numbers are printed IN GPT CONVENTIONS:
 *   MD time t_GPT = sqrt(2) t; S_gauge + beta V (6 c0 + 12 c1) (GPT keeps the constant); forces F_GPT = (i/2) ipdot
 *   (ipdot = -2i F_GPT, test G2): per term sum_links norm2(F_GPT) and the four links at the origin.
 * Lines start with "X8 " (compare.py of tests/exact8 parses them). Solves at `precision`, double, zero initial guess.
 */
template<size_t HaloDepth, size_t Ls>
void runMDWFExactHmc(CommunicationBase &commBase, MDWFDSDRTestParameters &tp, const std::vector<double> &twists) {
    using Gauge = DSDRGauge<HaloDepth>;
    using HostGauge = DSDRHostGauge<HaloDepth>;
    using Light = MDWFEvenOddHasenbuschTwoFlavorFermionAction<HaloDepth, Ls>;
    using Eofa = MDWFEofaRatioFermionAction<HaloDepth, Ls>;
    using Dsdr = MDWFDSDRFermionAction<HaloDepth>;
    using Action = MDWFSumFermionAction<HaloDepth, MDWFSumFermionAction<HaloDepth, Light, Eofa>, Dsdr>;
    using Hmc = MDWFHmcDriver<HaloDepth, Ls, Action>;
    typedef GIndexer<All, HaloDepth> GInd;
    if (!tp.GaugefileName.isSet()) {
        throw std::runtime_error(stdLogger.fatal("mode = exact needs Gaugefile"));
    }
    MDWFHmcParameters param = mdwfEofaTestParameters(tp);
    param.beta = tp.beta.isSet() ? tp.beta() : 1.829;
    param.mf = tp.mf();
    if (tp.hasenbusch_masses.isSet()) {
        param.hasenbusch_masses = tp.hasenbusch_masses.get();
    }
    param.pv_mass = tp.eofa_m2();
    param.rhmc.ms = tp.eofa_m1();
    param.dsdr_twists = twists;
    param.tau = tp.md_tau();
    param.steps = tp.md_steps();
    param.gauge_substeps = tp.md_gauge_steps();
    param.md_scheme = 2;
    param.mixed_precision = false;
    param.chrono_size = 0;
    param.light_form = tp.light_form();
    param.eofa_schur = tp.eofa_schur();
    // exact_timing: no S(phi) recomputation after the EOFA heatbath (2 extra solves; GPT's timed heatbath has none)
    param.eofa.check_heatbath = tp.exact_timing() == 0;

    Gauge gauge(commBase, "MDWF_x8_gauge");
    gauge.readconf_nersc(tp.GaugefileName());
    gauge.updateAll();
    Gauge saved(commBase, "MDWF_x8_saved");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20261016);
    d_rand = h_rand;
    Hmc hmc(commBase, gauge, param, d_rand.state);
    Light &light = hmc.fermion().first().first();
    Eofa &eofa = hmc.fermion().first().second();
    Dsdr &dsdr = hmc.fermion().second();

    std::vector<std::string> labels;
    for (size_t i = 0; i < light.factorCount(); i++) {
        labels.push_back("light" + std::to_string(i));
    }
    labels.push_back("eofa");
    for (size_t i = 0; i < dsdr.factorCount(); i++) {
        labels.push_back("dsdr" + std::to_string(i));
    }
    if (hmc.forceTermCount() != labels.size()) {
        throw std::runtime_error(stdLogger.fatal("mode = exact: ", hmc.forceTermCount(), " force terms, expected ",
                                                 labels.size()));
    }
    const double volume = static_cast<double>(GInd::getLatData().globvol4);
    const double c0 = 1.0 - 8.0 * param.gauge_c1;
    const double gaugeConstant = param.beta * volume * (6.0 * c0 + 12.0 * param.gauge_c1);
    const double tGpt = std::sqrt(2.0);
    const uint64_t seed = mdwfSeedFromString(tp.md_seed());
    {
        std::ostringstream line;
        line << std::setprecision(16) << "X8 setup: config " << tp.GaugefileName() << ", Ls " << Ls << ", beta "
             << param.beta << ", c1 " << param.gauge_c1 << ", M5 " << param.M5 << ", b5 " << param.b5 << ", c5 "
             << param.b5 - 1.0 << ", light form " << (param.light_form == 1 ? "gpt" : "sq") << ", light ladder " << param.mf;
        for (const double m : param.hasenbusch_masses) {
            line << " " << m;
        }
        line << " " << param.pv_mass << ", EOFA " << param.rhmc.ms << " / " << param.pv_mass << ", DSDR twists";
        for (const double m : twists) {
            line << " " << m;
        }
        line << ", OMF2 force-gradient " << param.steps << " x " << param.gauge_substeps << ", tau_GPT "
             << tGpt * param.tau << ", CG precision " << param.precision << ", seed '" << tp.md_seed() << "', terms";
        for (const auto &l : labels) {
            line << " " << l;
        }
        rootLogger.info(line.str());
        rootLogger.info("X8 EOFA heatbath approximation: ", param.eofa.zolotarev_order > 0 ? "Zolotarev " : "", "order ",
                        eofa.inverseSqrt().order, " on [", param.eofa.lambda_low, ", ", param.eofa.lambda_high,
                        "], error ", eofa.inverseSqrt().max_relative_error);
    }

    HostGauge forceHost(commBase, "MDWF_x8_force_host");
    Gauge forceDevice(commBase, "MDWF_x8_force_dev");
    // sum_links norm2(F_GPT) = sum_links -tr(K K) / 4 and the four links F_GPT = (i/2) K at the origin.
    auto printForce = [&](const std::string &tag, const std::string &label) {
        const SU3Accessor<double, R18> acc = forceHost.getAccessor();
        double sum = 0.0;
        for (size_t i = 0; i < GInd::getLatData().vol4; i++) {
            const gSite site = GInd::getSite(i);
            for (uint8_t mu = 0; mu < 4; mu++) {
                const SU3<double> k = acc.getLink(GInd::getSiteMu(site, mu));
                sum += -tr_d(k, k);
            }
        }
        sum = commBase.reduce(sum) / 4.0;
        std::ostringstream line;
        line << std::setprecision(16) << tag << " force " << label << ": norm2 = " << sum;
        for (int mu = 0; mu < 4; mu++) {
            const SU3<double> f = COMPLEX(double)(0.0, 0.5) * acc.getLink(GInd::getSiteMu(GInd::getSite(0, 0, 0, 0), mu));
            line << " | mu " << mu << ":";
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) {
                    line << " (" << real(f(r, c)) << "," << imag(f(r, c)) << ")";
                }
            }
        }
        rootLogger.info(line.str());
    };
    auto plaquette = [&]() {
        GaugeAction<double, true, HaloDepth, R18> ga(gauge);
        return static_cast<double>(ga.plaquette());
    };
    // All numbers at one step boundary; returns H.
    auto report = [&](const std::string &tag, double t) {
        const double K = hmc.kineticEnergy();
        const double Sg = hmc.gaugeAction() + gaugeConstant;
        std::vector<double> S;
        for (size_t i = 0; i < light.factorCount(); i++) {
            S.push_back(light.factor(i).action());
        }
        S.push_back(eofa.action());
        for (size_t i = 0; i < dsdr.factorCount(); i++) {
            S.push_back(dsdr.factor(i).action());
        }
        double H = K + Sg;
        std::ostringstream line;
        line << std::setprecision(16) << tag << " t_gpt = " << tGpt * t << ": K = " << K << ", S_gauge = " << Sg;
        for (size_t i = 0; i < S.size(); i++) {
            line << ", S_" << labels[i] << " = " << S[i];
            H += S[i];
        }
        line << ", H = " << H << ", plaquette = " << plaquette();
        rootLogger.info(line.str());
        hmc.gaugeForce(forceDevice);
        forceHost = forceDevice;
        printForce(tag, "gauge");
        for (size_t i = 0; i < labels.size(); i++) {
            hmc.termForce(i, forceHost);
            printForce(tag, labels[i]);
        }
        return H;
    };

    const double etaScale = 1.0 / std::sqrt(2.0);
    const double dt = param.tau / static_cast<double>(param.steps);
    const bool timing = tp.exact_timing() != 0;
    // exact_timing: H from the actions only (no forces, no plaquette), the trajectory in one integrator call (forces
    // at the step boundaries reused, as in production), and the time and CG iterations of every phase.
    auto energy = [&]() {
        double H = hmc.kineticEnergy() + hmc.gaugeAction() + gaugeConstant;
        for (size_t i = 0; i < light.factorCount(); i++) {
            H += light.factor(i).action();
        }
        H += eofa.action();
        for (size_t i = 0; i < dsdr.factorCount(); i++) {
            H += dsdr.factor(i).action();
        }
        return H;
    };
    auto clock = [&]() {
        gpuDeviceSynchronize();
        commBase.globalBarrier();
        return std::chrono::steady_clock::now();
    };
    auto seconds = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    if (timing) {
        rootLogger.info("X8T setup: ranks ", commBase.getNumberProcesses(), ", Nodes ", tp.nodeDim()[0], " ",
                        tp.nodeDim()[1], " ", tp.nodeDim()[2], " ", tp.nodeDim()[3], ", EOFA solve form ",
                        param.eofa_schur == 2 ? "Schur two (GPT eo2_ne)" : "Mhat",
                        ", timed trajectories without per-step diagnostics");
    }
    int accepted = 0;
    for (int n = 1; n <= tp.exact_traj(); n++) {
        const uint64_t B = 1000 * static_cast<uint64_t>(n);
        const auto tStart = clock();
        const long long cgStart = mdwfCoupledCGIterationCount();
        saved = gauge;
        // The heatbath and action() calls below bypass the driver, so sync its antiperiodic link copy here: after a
        // rejection it still holds the rejected end links (MDWFHmcDriver::trajectory syncs in heatbath()).
        hmc.fermionGauge();
        hmc.momenta().iterateOverBulkAllMu(MDWFHashMomentum<HaloDepth>(seed, B + 1));
        hmc.momenta().updateAll();
        std::ostringstream hb;
        hb << std::setprecision(16) << "X8 traj " << n << " heatbath:";
        for (size_t i = 0; i < light.factorCount(); i++) {
            auto &f = light.factor(i);
            f.noise().template iterateOverBulk<BLOCKSIZE>(MDWFHashSpinor<Even, HaloDepth>(seed, B + 100 + 32 * i, etaScale));
            f.heatbathFromNoise();
            hb << " eta2_light" << i << " = " << f.noiseNorm2() << ", phi2_light" << i << " = " << f.phiNorm2() << ",";
        }
        eofa.noise().template iterateOverBulk<BLOCKSIZE>(MDWFHashSpinor<All, HaloDepth>(seed, B + 20, etaScale));
        eofa.heatbathFromNoise();
        hb << " eta2_eofa = " << eofa.noiseNorm2() << ", phi2_eofa = " << eofa.phiNorm2();
        for (size_t i = 0; i < dsdr.factorCount(); i++) {
            dsdr.factor(i).noise().template iterateOverBulk<BLOCKSIZE>(MDWFHashSpinor<Even, HaloDepth>(seed, B + 10 + i, etaScale));
            dsdr.factor(i).heatbathFromNoise();
            hb << ", eta2_dsdr" << i << " = " << dsdr.factor(i).noiseNorm2() << ", phi2_dsdr" << i << " = "
               << dsdr.factor(i).phiNorm2();
        }
        rootLogger.info(hb.str());

        double h0 = 0.0, h1 = 0.0;
        if (timing) {
            const auto tHeatbath = clock();
            const long long cgHeatbath = mdwfCoupledCGIterationCount();
            h0 = energy();
            const auto tEnergy0 = clock();
            const long long cgEnergy0 = mdwfCoupledCGIterationCount();
            const int forcesBefore = hmc.forceEvaluations();
            const std::vector<long long> termCgBefore = hmc.termForceCgIterations();
            hmc.integrateTime(param.tau, param.steps);
            const auto tMd = clock();
            const long long cgMd = mdwfCoupledCGIterationCount();
            const int forceEvals = hmc.forceEvaluations() - forcesBefore;
            h1 = energy();
            const auto tEnergy1 = clock();
            const long long cgEnergy1 = mdwfCoupledCGIterationCount();
            rootLogger.info(std::setprecision(6), "X8T traj ", n, " seconds: heatbath ", seconds(tStart, tHeatbath),
                            ", energy0 ", seconds(tHeatbath, tEnergy0), ", md ", seconds(tEnergy0, tMd), ", energy1 ",
                            seconds(tMd, tEnergy1), ", total ", seconds(tStart, tEnergy1),
                            " | cg_iterations: heatbath ", cgHeatbath - cgStart, ", energy0 ", cgEnergy0 - cgHeatbath,
                            ", md ", cgMd - cgEnergy0, ", energy1 ", cgEnergy1 - cgMd,
                            " | fermion_force_evaluations ", forceEvals / static_cast<int>(labels.size()),
                            " (term evaluations ", forceEvals, ")");
            std::ostringstream terms;
            terms << "X8T traj " << n << " md cg_iterations per term:";
            const std::vector<long long> &termCg = hmc.termForceCgIterations();
            for (size_t i = 0; i < labels.size(); i++) {
                terms << " " << labels[i] << " " << termCg[i] - (i < termCgBefore.size() ? termCgBefore[i] : 0);
            }
            rootLogger.info(terms.str());
        }
        for (int k = 0; k <= param.steps && !timing; k++) {
            std::ostringstream tag;
            tag << "X8 traj " << n << " step " << k;
            const double h = report(tag.str(), k * dt);
            if (k == 0) {
                h0 = h;
            }
            if (k == param.steps) {
                h1 = h;
            } else {
                hmc.integrateTime(dt, 1);
            }
        }
        const double dH = h1 - h0;
        const double u = mdwfHashUniform(seed, 999, static_cast<uint64_t>(n));
        const bool accept = dH <= 0.0 || u < std::exp(-dH);
        if (!accept) {
            gauge = saved;
            gauge.updateAll();
        }
        accepted += accept ? 1 : 0;
        rootLogger.info(std::setprecision(16), "X8 traj ", n, " end: dH = ", dH, ", exp(-dH) = ", std::exp(-dH),
                        ", u = ", u, ", accepted = ", accept ? 1 : 0, ", plaquette = ", plaquette());
    }
    rootLogger.info("X8 done: ", accepted, " of ", tp.exact_traj(), " accepted");
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);

        MDWFDSDRTestParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfDSDRTest.param", argc, argv);
        commBase.forceHalos(param.force_halos() != 0);
        commBase.init(param.nodeDim());

        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);

        std::vector<double> twists{0.02, 0.07, 0.18, 0.5};
        if (param.dsdr_twists.isSet()) {
            twists = param.dsdr_twists.get();
        }

        if (param.mode() == "unit") {
            if (!runMDWFDSDRUnitTest<HaloDepth>(commBase, param, twists)) {
                throw std::runtime_error(stdLogger.fatal("MDWF DSDR test failed"));
            }
            rootLogger.info("MDWF DSDR test passed");
        } else if (param.mode() == "gpt") {
            runMDWFDSDRGptCheck<HaloDepth>(commBase, param, twists);
            rootLogger.info("MDWF DSDR GPT check done");
        } else if (param.mode() == "gptmdwf") {
            if (param.Ls() == 16) {
                runMDWFLightGptCheck<HaloDepth, 16>(commBase, param);
            } else if (param.Ls() == 8) {
                runMDWFLightGptCheck<HaloDepth, 8>(commBase, param);
            } else {
                throw std::runtime_error(stdLogger.fatal("mode = gptmdwf supports Ls = 8 or 16"));
            }
            rootLogger.info("MDWF light GPT check done");
        } else if (param.mode() == "eofa" || param.mode() == "gpteofa") {
            const bool unit = param.mode() == "eofa";
            bool ok = true;
            if (param.Ls() == 16) {
                if (unit) {
                    ok = runMDWFEofaUnitTest<HaloDepth, 16>(commBase, param);
                } else {
                    runMDWFEofaGptCheck<HaloDepth, 16>(commBase, param);
                }
            } else if (param.Ls() == 8) {
                if (unit) {
                    ok = runMDWFEofaUnitTest<HaloDepth, 8>(commBase, param);
                } else {
                    runMDWFEofaGptCheck<HaloDepth, 8>(commBase, param);
                }
            } else {
                throw std::runtime_error(stdLogger.fatal("EOFA modes support Ls = 8 or 16"));
            }
            if (!ok) {
                throw std::runtime_error(stdLogger.fatal("MDWF EOFA test failed"));
            }
            rootLogger.info(unit ? "MDWF EOFA test passed" : "MDWF EOFA GPT check done");
        } else if (param.mode() == "gptmd") {
            if (param.Ls() == 16) {
                runMDWFMdGptCheck<HaloDepth, 16>(commBase, param, twists);
            } else if (param.Ls() == 8) {
                runMDWFMdGptCheck<HaloDepth, 8>(commBase, param, twists);
            } else {
                throw std::runtime_error(stdLogger.fatal("mode = gptmd supports Ls = 8 or 16"));
            }
            rootLogger.info("MDWF MD GPT check done");
        } else if (param.mode() == "exact") {
            if (param.Ls() == 16) {
                runMDWFExactHmc<HaloDepth, 16>(commBase, param, twists);
            } else if (param.Ls() == 8) {
                runMDWFExactHmc<HaloDepth, 8>(commBase, param, twists);
            } else {
                throw std::runtime_error(stdLogger.fatal("mode = exact supports Ls = 8 or 16"));
            }
        } else {
            throw std::runtime_error(stdLogger.fatal("mode must be unit, gpt, gptmdwf, eofa, gpteofa, gptmd or exact"));
        }
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
