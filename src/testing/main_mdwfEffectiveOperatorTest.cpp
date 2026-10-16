/*
 * Tests of the 4D effective MDWF Dirac operator (MDWFEffectiveOperator.h) and of the thick-restart Lanczos eigensolver
 * (MDWFThickRestartLanczos.h), Ls = 8, 6^4, single rank:
 *
 *   A  free field, constant spinor (zero momentum, periodic): D_W = -M5, M_std = kappa - Hop with
 *      kappa = (1 - b5 M5) / (1 + c5 M5), so the wall-to-wall leakage is delta = kappa^Ls and
 *        D(m) v = [m + (1 - m) delta / (1 + delta)] v,   gwDefect(v) = 1 - eps^2 = 4 delta / (1 + delta)^2,
 *      for Shamir (b5 = 1) and Mobius (b5 = 2), m = 0 and 0.1, M5 = 1.8 and 1.0; 1e-9 relative.
 *      Fixes the orientation of P (light mode on the walls) and the sign of the boundary mass.
 *   B  eps is Hermitian (gamma5-Hermiticity of D): <u|eps v> = conj(<v|eps u>) for Gaussian u, v on a smoothed random
 *      gauge field (fat7 x4 of a random field), Shamir and Mobius b5 = 1.5, c_sw = 0 and 1; 1e-8 relative.
 *   C  affine mass dependence: D(m) v = m v + (1 - m) D(0) v on the same field; 1e-8 relative.
 *   D  eigensolver: the four lowest eigenpairs of H_W^2 on the smoothed field from thick-restart Lanczos without (six
 *      modes) and with a Chebyshev filter (lower edge at the sixth eigenvalue) agree with each other (1e-8 relative), their residuals |A y - theta y| are below
 *      1e-6 lambda_max, the vectors are orthonormal (1e-10), and the lowest lies at or below the plain-Lanczos
 *      Ritz value (mdwfLanczosExtremes, an upper bound that converges slowly for isolated small eigenvalues).
 *
 * Usage: ./mdwfEffectiveOperatorTest (parameter/tests/mdwfFifthDimTest.param).
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFCoupledSolverAdapter.h"
#include "../experimental/mdwf/MDWFEffectiveOperator.h"
#include "../experimental/mdwf/MDWFKernelSmearing.h"
#include "../experimental/mdwf/MDWFSpectralBounds.h"
#include "../experimental/mdwf/MDWFThickRestartLanczos.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

constexpr size_t TestHalo = 2;
constexpr size_t TestLs = 8;
using TestSpinor = MDWFSpinor<double, true, All, TestHalo, 1>;
using TestGauge = Gaugefield<double, true, TestHalo, R18>;
using TestEffective = MDWFEffectiveDiracOperator<TestHalo, TestLs>;

// The same 12 components on every site.
struct MDWFConstantSpinor {
    __host__ __device__ Vect12<double> operator()(gSiteStack) {
        Vect12<double> out(0.0);
        for (int k = 0; k < 12; k++) {
            out.data[k] = COMPLEX(double)(0.3 + 0.1 * k, 0.05 * k - 0.4);
        }
        return out;
    }
};

double mdwfTestNorm(TestSpinor &v) {
    return std::sqrt(real<double>(mdwfDot4(v, v)));
}

// |a - b| / |reference|
double mdwfTestRelativeDifference(TestSpinor &a, TestSpinor &b, TestSpinor &reference, TestSpinor &work) {
    work = a;
    work.template axpyThisB<64>(-1.0, b);
    return mdwfTestNorm(work) / mdwfTestNorm(reference);
}

bool testFreeField(CommunicationBase &commBase) {
    rootLogger.info("MDWF effective operator test: start A (unit gauge field)");
    TestGauge unit(commBase, "MDWF_eot_unit");
    unit.one();
    unit.updateAll();
    rootLogger.info("MDWF effective operator test: unit field set");
    TestSpinor v(commBase, "MDWF_eot_v"), dv(commBase, "MDWF_eot_dv"), expected(commBase, "MDWF_eot_exp"),
            work(commBase, "MDWF_eot_work"), ev(commBase, "MDWF_eot_ev");
    v.template iterateOverBulk<BLOCKSIZE>(MDWFConstantSpinor());
    v.updateAll();
    rootLogger.info("MDWF effective operator test: constant spinor set");
    const double vNorm2 = real<double>(mdwfDot4(v, v));
    rootLogger.info("MDWF effective operator test: |v|^2 = ", vNorm2);
    bool passed = true;
    for (const double b5 : {1.0, 2.0}) {
        for (const double M5 : {1.8, 1.0}) {
            for (const double m : {0.0, 0.1}) {
                const double c5 = b5 - 1.0;
                const double kappa = (1.0 - b5 * M5) / (1.0 + c5 * M5);
                const double delta = std::pow(kappa, static_cast<double>(TestLs));
                const double factor = m + (1.0 - m) * delta / (1.0 + delta);
                const double defect = 4.0 * delta / ((1.0 + delta) * (1.0 + delta));
                TestEffective d(unit, M5, m, b5, 0.0, 10000, 1e-12, "MDWF_eot_free");
                rootLogger.info("MDWF effective operator test: operator built (b5 = ", b5, ", M5 = ", M5, ", m = ", m, ")");
                d.applyD(dv, v);
                rootLogger.info("MDWF effective operator test: D v done (", d.iterations(), " CG iterations)");
                expected = v;
                expected *= COMPLEX(double)(factor, 0.0);
                const double diff = mdwfTestRelativeDifference(dv, expected, v, work);   // factor can be 0
                d.applyEpsilon(ev, v);
                const double gw = 1.0 - real<double>(mdwfDot4(ev, ev)) / vNorm2;
                const bool ok = diff <= 1e-9 && std::abs(gw - defect) <= 1e-9 * std::max(defect, 1e-3);
                rootLogger.info("MDWF effective operator test A (free field, zero momentum): b5 = ", b5, ", M5 = ", M5,
                                ", m = ", m, ": expected factor ", factor, ", |D v - factor v| / |v| = ", diff,
                                "; gwDefect ", gw, " expected ", defect, "; passed = ", ok);
                passed = passed && ok;
            }
        }
    }
    return passed;
}

bool testHermiticityAndMass(TestGauge &smooth, CommunicationBase &commBase) {
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(20261014);
    d_rand = h_rand;
    TestSpinor u(commBase, "MDWF_eot_u"), v(commBase, "MDWF_eot_v2"), eu(commBase, "MDWF_eot_eu"),
            ev(commBase, "MDWF_eot_ev2"), dm(commBase, "MDWF_eot_dm"), d0(commBase, "MDWF_eot_d0"),
            work(commBase, "MDWF_eot_work2");
    u.gauss(d_rand.state);
    v.gauss(d_rand.state);
    bool passed = true;
    for (const double b5 : {1.0, 1.5}) {
        for (const double csw : {0.0, 1.0}) {
            const double M5 = 1.4, m = 0.05;
            TestEffective d(smooth, M5, m, b5, csw, 10000, 1e-12, "MDWF_eot_h");
            TestEffective dZero(smooth, M5, 0.0, b5, csw, 10000, 1e-12, "MDWF_eot_h0");
            d.applyEpsilon(eu, u);
            d.applyEpsilon(ev, v);
            const COMPLEX(double) left = mdwfDot4(u, ev), right = mdwfDot4(v, eu);
            const double scale = std::sqrt(real<double>(mdwfDot4(u, u)) * real<double>(mdwfDot4(v, v)));
            const double herm = std::sqrt((real(left) - real(right)) * (real(left) - real(right))
                                          + (imag(left) + imag(right)) * (imag(left) + imag(right))) / scale;
            const bool okB = herm <= 1e-8;
            // C: D(m) v = m v + (1 - m) D(0) v
            d.applyD(dm, v);
            dZero.applyD(d0, v);
            d0 *= COMPLEX(double)(1.0 - m, 0.0);
            d0.template axpyThisB<64>(m, v);
            const double affine = mdwfTestRelativeDifference(dm, d0, dm, work);
            const bool okC = affine <= 1e-8;
            rootLogger.info("MDWF effective operator test B/C (smoothed random field): b5 = ", b5, ", c_sw = ", csw,
                            ": |<u|eps v> - conj<v|eps u>| / (|u||v|) = ", herm, " (passed = ", okB,
                            "), |D(m) v - m v - (1 - m) D(0) v| / |D(m) v| = ", affine, " (passed = ", okC,
                            "); PV solves ", d.solves(), ", ", d.iterations(), " CG iterations");
            passed = passed && okB && okC;
        }
    }
    return passed;
}

bool testEigensolver(TestGauge &smooth, CommunicationBase &commBase) {
    using Kernel = MDWFHermitianWilsonSquaredOperator<TestHalo>;
    Kernel a(smooth, 1.4, 0.0, "MDWF_eot_k");
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(42);
    d_rand = h_rand;
    TestSpinor start(commBase, "MDWF_eot_start"), ay(commBase, "MDWF_eot_ay");
    start.gauss(d_rand.state);

    MDWFCoupledSolverAdapter<double, TestHalo, TestHalo, 1, Kernel> adapter(a);
    const MDWFLanczosResult plain = mdwfLanczosExtremes(adapter, start, 600, {600}, "MDWF_eot_lz");
    const double lambdaMin = plain.checkpoints.back().lambda_min, lambdaMax = plain.checkpoints.back().lambda_max;

    // Run 0: six lowest modes without filter. Run 1: four lowest with a Chebyshev filter whose lower edge is the
    // sixth eigenvalue from run 0 (so the wanted four and the fifth lie below it).
    const MDWFThickRestartParameters params[2] = {{6, 30, 16, 1e-10, 2000}, {4, 24, 12, 1e-10, 2000}};
    std::vector<std::vector<double>> rayleigh(2);
    bool passed = true;
    for (int run = 0; run < 2; run++) {
        const MDWFThickRestartParameters &p = params[run];
        const int degree = run == 0 ? 0 : 10;
        const double lo = run == 0 ? 0.0 : rayleigh[0][5];
        MDWFChebyshevOperator<TestSpinor, Kernel> op(a, lo, 1.05 * lambdaMax, degree, commBase,
                                                     "MDWF_eot_cheb" + std::to_string(run));
        std::vector<std::unique_ptr<TestSpinor>> vecs;
        const MDWFThickRestartResult r = mdwfThickRestartLanczos(op, start, p, vecs,
                                                                 "MDWF_eot_tr" + std::to_string(run));
        double maxResidual = 0.0, maxOrtho = 0.0;
        for (int l = 0; l < p.nev; l++) {
            vecs[l]->updateAll();
            a.apply(ay, *vecs[l]);
            const double theta = real<double>(mdwfDot4(*vecs[l], ay));
            rayleigh[run].push_back(theta);
            ay.template axpyThisB<64>(-theta, *vecs[l]);
            maxResidual = std::max(maxResidual, mdwfTestNorm(ay));
            for (int l2 = 0; l2 <= l; l2++) {
                const COMPLEX(double) c = mdwfDot4(*vecs[l2], *vecs[l]);
                const double target = l2 == l ? 1.0 : 0.0;
                maxOrtho = std::max(maxOrtho, std::abs(real(c) - target) + std::abs(imag(c)));
            }
        }
        const bool ok = r.converged && maxResidual <= 1e-6 * lambdaMax && maxOrtho <= 1e-10;
        rootLogger.info("MDWF eigensolver test D: Chebyshev degree ", degree, ": converged = ", r.converged, " after ",
                        r.restarts, " restarts, ", r.applications, " operator calls; lowest Rayleigh quotients ",
                        rayleigh[run][0], " ", rayleigh[run][1], " ", rayleigh[run][2], " ", rayleigh[run][3],
                        "; max |A y - theta y| = ", maxResidual, ", max orthonormality defect ", maxOrtho,
                        "; passed = ", ok);
        passed = passed && ok;
    }
    double agree = 0.0;
    for (int l = 0; l < 4; l++) {
        agree = std::max(agree, std::abs(rayleigh[0][l] - rayleigh[1][l]) / rayleigh[0][l]);
    }
    // Plain Lanczos without reorthogonalization gives an upper bound on lambda_min that converges slowly for an
    // isolated small eigenvalue (first run: 2.6e-5 after 600 steps against the true 1.27e-6, residual 3e-9), so it
    // is only a bound here: the thick-restart value must lie at or below it.
    const bool below = rayleigh[0][0] <= lambdaMin * (1.0 + 1e-10);
    const bool ok = agree <= 1e-8 && below;
    rootLogger.info("MDWF eigensolver test D: plain and filtered agree to ", agree, " (relative); lowest ",
                    rayleigh[0][0], " at or below the plain-Lanczos upper bound ", lambdaMin, ": ", below,
                    "; passed = ", ok);
    return passed && ok;
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);
        LatticeParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfFifthDimTest.param", argc, argv);
        commBase.init(param.nodeDim());
        initIndexer(TestHalo, param, commBase);
        if (GIndexer<All, TestHalo>::getLatData().vol4 != GIndexer<All, TestHalo>::getLatData().globvol4) {
            throw std::runtime_error(stdLogger.fatal("MDWF effective operator test is single-rank"));
        }

        bool passed = testFreeField(commBase);

        // Smoothed random field: four fat7 steps of a random SU(3) field.
        grnd_state<false> h_rand;
        grnd_state<true> d_rand;
        h_rand.make_rng_state(20261013);
        d_rand = h_rand;
        TestGauge rough(commBase, "MDWF_eot_rough"), smooth(commBase, "MDWF_eot_smooth");
        rough.random(d_rand.state);
        rough.updateAll();
        mdwfSmearKernelLinks<TestHalo>(smooth, rough, MDWFKernelSmearingType::Fat7SU3, 4, "MDWF_eot_smear");
        smooth.updateAll();

        passed = testHermiticityAndMass(smooth, commBase) && passed;
        passed = testEigensolver(smooth, commBase) && passed;
        if (!passed) {
            throw std::runtime_error(stdLogger.fatal("MDWF effective operator test failed"));
        }
        rootLogger.info("MDWF effective operator test passed");
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
