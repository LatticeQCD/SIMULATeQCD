/*
 * Test of the smeared kernel links of the MDWF operator (MDWFKernelSmearing.h), fat7_su3 with
 * two steps as used in the m_res measurements:
 *
 *   A  unit field stays unit:   W = 1 for U = 1 (fat7 coefficients sum to 1, projection of 1 is 1), 1e-14
 *   B  SU(3) links:             random U: max |W^+ W - 1| and max |det W - 1| <= 1e-11 (fat7_su3), and the
 *                               fat7_u3 links are unitary as well
 *   C  steps iterate:           two steps equal one step applied to the result of one step, 1e-14
 *   D  gauge covariance:        smearing commutes with a random gauge transformation g:
 *                               W[g U g^+](x, mu) = g(x) W[U](x, mu) g(x + mu)^+, 1e-8 (a real defect gives
 *                               1e-2 ... 1; rounding in the closed-form (V^+V)^(-1/2) on the maximally rough
 *                               field reaches about 1e-11, first run 1.06e-11)
 *
 * D is the strongest check: any missing, extra or misplaced staple path, a wrong orientation, or a projection
 * that is not U(3)-covariant breaks it. The random field is maximally rough, so it also exercises the U(3)
 * projection far from the identity. Usage: ./mdwfKernelSmearingTest (parameter/tests/mdwfFifthDimTest.param,
 * 6^4, single rank).
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFKernelSmearing.h"

#include <algorithm>
#include <stdexcept>
#include <string>

// U'(x, mu) = g(x) U(x, mu) g(x + mu)^+, with g(x) taken from the mu = 0 links of a second random field.
template<size_t HaloDepth>
struct MDWFTestGaugeTransform {
    SU3Accessor<double, R18> _u, _g;

    MDWFTestGaugeTransform(Gaugefield<double, true, HaloDepth, R18> &u, Gaugefield<double, true, HaloDepth, R18> &g)
        : _u(u.getAccessor()), _g(g.getAccessor()) {}

    __host__ __device__ SU3<double> operator()(gSiteMu siteMu) {
        typedef GIndexer<All, HaloDepth> GInd;
        const gSite site = static_cast<gSite>(siteMu);
        const gSite up = GInd::site_up(site, siteMu.mu);
        return _g.getLink(GInd::getSiteMu(site, 0)) * _u.getLink(siteMu)
               * _g.getLinkDagger(GInd::getSiteMu(up, 0));
    }
};

// max over links of infnorm(A - B) (host, single rank).
template<size_t HaloDepth>
double mdwfMaxLinkDifference(Gaugefield<double, true, HaloDepth, R18> &a, Gaugefield<double, true, HaloDepth, R18> &b,
                             const std::string &name) {
    typedef GIndexer<All, HaloDepth> GInd;
    Gaugefield<double, false, HaloDepth, R18> ha(a.getComm(), name + "_a"), hb(b.getComm(), name + "_b");
    ha = a;
    hb = b;
    const SU3Accessor<double, R18> accA = ha.getAccessor(), accB = hb.getAccessor();
    double maxDiff = 0.0;
    for (size_t i = 0; i < GInd::getLatData().vol4; i++) {
        const gSite site = GInd::getSite(i);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const SU3<double> d = accA.getLink(GInd::getSiteMu(site, mu)) - accB.getLink(GInd::getSiteMu(site, mu));
            maxDiff = std::max(maxDiff, static_cast<double>(infnorm(d)));
        }
    }
    return maxDiff;
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);
        LatticeParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfFifthDimTest.param", argc, argv);
        commBase.init(param.nodeDim());
        const size_t HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);
        if (GIndexer<All, HaloDepth>::getLatData().vol4 != GIndexer<All, HaloDepth>::getLatData().globvol4) {
            throw std::runtime_error(stdLogger.fatal("MDWF kernel smearing test is single-rank"));
        }
        using Gauge = Gaugefield<double, true, HaloDepth, R18>;
        const MDWFKernelSmearingType su3 = MDWFKernelSmearingType::Fat7SU3;
        bool passed = true;

        // A: unit field.
        Gauge unit(commBase, "MDWF_smtest_unit"), unitSmeared(commBase, "MDWF_smtest_unit_sm");
        unit.one();
        unit.updateAll();
        mdwfSmearKernelLinks<HaloDepth>(unitSmeared, unit, su3, 2, "MDWF_smtest_sA");
        const double diffA = mdwfMaxLinkDifference<HaloDepth>(unitSmeared, unit, "MDWF_smtest_hA");
        const bool passedA = diffA <= 1e-14;
        rootLogger.info("MDWF kernel smearing test A (unit field stays unit): max |W - 1| = ", diffA,
                        ", passed = ", passedA);
        passed = passed && passedA;

        // Random fields: U, and g for the gauge transformation.
        grnd_state<false> h_rand;
        grnd_state<true> d_rand;
        h_rand.make_rng_state(20261004);
        d_rand = h_rand;
        Gauge u(commBase, "MDWF_smtest_u"), g(commBase, "MDWF_smtest_g");
        u.random(d_rand.state);
        u.updateAll();
        g.random(d_rand.state);
        g.updateAll();

        // B: SU(3) (and U(3)) links.
        Gauge w(commBase, "MDWF_smtest_w"), wU3(commBase, "MDWF_smtest_wu3");
        mdwfSmearKernelLinks<HaloDepth>(w, u, su3, 2, "MDWF_smtest_sB");
        mdwfSmearKernelLinks<HaloDepth>(wU3, u, MDWFKernelSmearingType::Fat7U3, 2, "MDWF_smtest_sBu3");
        const MDWFKernelLinkReport rep = mdwfKernelLinkReport<HaloDepth>(w, "MDWF_smtest_rB");
        const MDWFKernelLinkReport repU3 = mdwfKernelLinkReport<HaloDepth>(wU3, "MDWF_smtest_rBu3");
        // Production fields (smooth) reach 4.5e-13; the maximally rough random field gets one decade of room.
        const bool passedB = rep.maxUnitarityViolation <= 1e-11 && rep.maxDetDeviation <= 1e-11
                             && repU3.maxUnitarityViolation <= 1e-11;
        rootLogger.info("MDWF kernel smearing test B (random field, two steps): fat7_su3 max |W^+ W - 1| = ",
                        rep.maxUnitarityViolation, ", max |det W - 1| = ", rep.maxDetDeviation,
                        "; fat7_u3 max |W^+ W - 1| = ", repU3.maxUnitarityViolation, ", max |arg det| = ",
                        repU3.maxDetPhase, "; passed = ", passedB);
        passed = passed && passedB;

        // C: two steps = one step applied twice.
        Gauge once(commBase, "MDWF_smtest_once"), twice(commBase, "MDWF_smtest_twice");
        mdwfSmearKernelLinks<HaloDepth>(once, u, su3, 1, "MDWF_smtest_sC1");
        mdwfSmearKernelLinks<HaloDepth>(twice, once, su3, 1, "MDWF_smtest_sC2");
        const double diffC = mdwfMaxLinkDifference<HaloDepth>(twice, w, "MDWF_smtest_hC");
        const bool passedC = diffC <= 1e-14;
        rootLogger.info("MDWF kernel smearing test C (steps iterate): max |W_2 - W_1(W_1)| = ", diffC,
                        ", passed = ", passedC);
        passed = passed && passedC;

        // D: gauge covariance.
        Gauge uT(commBase, "MDWF_smtest_ut"), wOfUT(commBase, "MDWF_smtest_w_ut"), wT(commBase, "MDWF_smtest_wt");
        uT.iterateOverBulkAllMu(MDWFTestGaugeTransform<HaloDepth>(u, g));
        uT.updateAll();
        mdwfSmearKernelLinks<HaloDepth>(wOfUT, uT, su3, 2, "MDWF_smtest_sD");
        wT.iterateOverBulkAllMu(MDWFTestGaugeTransform<HaloDepth>(w, g));
        wT.updateAll();
        const double diffD = mdwfMaxLinkDifference<HaloDepth>(wOfUT, wT, "MDWF_smtest_hD");
        // Rounding of the closed-form U(3) projection on ill-conditioned V^+V (rough random field) reaches ~1e-11;
        // a missing or misplaced staple, a wrong orientation or a non-covariant projection gives 1e-2 ... 1.
        const bool passedD = diffD <= 1e-8;
        rootLogger.info("MDWF kernel smearing test D (gauge covariance): max |W[gUg^+] - g W[U] g^+| = ", diffD,
                        ", passed = ", passedD);
        passed = passed && passedD;

        if (!passed) {
            throw std::runtime_error(stdLogger.fatal("MDWF kernel smearing test failed"));
        }
        rootLogger.info("MDWF kernel smearing test passed (fat7_su3, two steps)");
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
