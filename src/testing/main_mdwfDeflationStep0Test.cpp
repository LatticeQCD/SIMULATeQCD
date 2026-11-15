/*
 * Unit test of the deflation Step 0 tools (MDWFFieldDump.h, MDWFDeflation.h) on a small lattice. Any node layout
 * (the job runs it on 4 GPUs with Nodes = 1 2 2 1, as production). Every check prints PASS or FAIL; the program
 * returns 0 only if all pass and then prints "mdwfDeflationStep0Test: ALL PASS".
 *
 *   D1  gauge dump round trip (links): bit-exact
 *   D2  gauge dump round trip (momenta-like, not unitary): bit-exact
 *   D3  even 5D spinor dump round trip: bit-exact
 *   D4  compact vector <-> field: bit-exact; eigenvector-set file round trip: bit-exact
 *   H1  host Hermitian eigensolver, inverse and singular values
 *   O1  operator Hermitian and positive
 *   E1  K lowest eigenpairs: converged, residuals, orthonormality
 *   C1  plain CG: true residual
 *   C2  deflated CG, exact subspace: true residual, fewer iterations than plain CG
 *   C3  deflated CG, arbitrary (noise) subspace: still the solution (true residual)
 *   C4  deflated initial guess (Galerkin and spectral) + plain CG: true residual
 *   T1  tracking on a nearby gauge field: Ritz values bounded below by the exact eigenvalues (min-max) for T1, T2, T3
 *       and T2, T3 not above T1 (their spans contain the stale one); mean cos^2 with the exact subspace reported
 *   R1  refresh snapshots (mode refresh, 2026-11-13): n = 0 is the input vector itself (CG and Chebyshev), bit-exact
 *   R2  CG snapshots at n = 1 3 8 20 from ONE run = separate mdwfDeflCG runs with maxIter = n, precision 0
 *   R3  Chebyshev snapshots at degrees 1 2 7 16 from ONE run = MDWFChebyshevOperator (the eigensolver's filter) of
 *       each degree on its own (times -1)
 *   R4  Chebyshev filter values: for u = sum_k c_k v_k (exact eigenvectors of E1), snapshot n has <v_k|y_n> = c_k
 *       T_n(L(lambda_k)) and y_n = sum_k c_k T_n(L(lambda_k)) v_k, n = 4 16 64 (both to 1e-6 of |y_n|)
 *
 * Usage: ./mdwfDeflationStep0Test <param file> [key=value ...]   (parameter/tests/mdwfDeflationStep0Test.param);
 * dump files go to measurements_dir (must exist).
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFDeflation.h"

#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class MDWFDeflationStep0TestParameters : public LatticeParameters {
public:
    Parameter<int> ls;
    Parameter<double> M5;
    Parameter<double> b5;
    Parameter<double> mf;
    Parameter<int> antiperiodic_t;
    Parameter<int> light_form;
    Parameter<int> force_halos;
    Parameter<int> k;
    Parameter<double> roughness;
    Parameter<double> md_step;
    Parameter<int> seed;

    MDWFDeflationStep0TestParameters() {
        addDefault(ls, "Ls", 8);
        addDefault(M5, "M5", 1.8);
        addDefault(b5, "b5", 1.5);
        addDefault(mf, "mf", 0.02);
        addDefault(antiperiodic_t, "antiperiodic_t", 1);
        addDefault(light_form, "light_form", 1);
        addDefault(force_halos, "force_halos", 0);
        addDefault(k, "k", 8);
        addDefault(roughness, "roughness", 0.3);
        addDefault(md_step, "md_step", 0.02);
        addDefault(seed, "seed", 20261110);
    }
};

// U <- exp(i eps P) U (as the HMC's Q update).
template<size_t HaloDepth>
struct MDWFStep0TestMove {
    SU3Accessor<double, R18> _gAcc;
    SU3Accessor<double> _pAcc;
    double _eps;

    MDWFStep0TestMove(SU3Accessor<double, R18> gAcc, SU3Accessor<double> pAcc, double eps)
        : _gAcc(gAcc), _pAcc(pAcc), _eps(eps) {}

    __host__ __device__ SU3<double> operator()(gSiteMu site) {
        SU3<double> temp = su3_exp<double>(COMPLEX(double)(0.0, 1.0) * _eps * _pAcc.getLink(site)) * _gAcc.getLink(site);
        temp.su3unitarize();
        return temp;
    }
};

static int mdwfStep0TestFailures = 0;

template<class... Args>
void mdwfStep0TestCheck(bool ok, const Args &...what) {
    if (ok) {
        rootLogger.info("PASS  ", what...);
    } else {
        rootLogger.error("FAIL  ", what...);
        mdwfStep0TestFailures++;
    }
}

// Largest |a - b| over the bulk links of two device gauge-like fields, all ranks.
template<size_t HaloDepth>
double mdwfStep0TestLinkDiff(CommunicationBase &comm, Gaugefield<double, true, HaloDepth, R18> &a,
                             Gaugefield<double, true, HaloDepth, R18> &b) {
    typedef GIndexer<All, HaloDepth> GInd;
    Gaugefield<double, false, HaloDepth, R18> ha(comm, "MDWF_s0t_diff_ha"), hb(comm, "MDWF_s0t_diff_hb");
    ha = a;
    hb = b;
    const SU3Accessor<double, R18> accA = ha.getAccessor(), accB = hb.getAccessor();
    double diff = 0.0;
    for (size_t i = 0; i < GInd::getLatData().vol4; i++) {
        const gSite site = GInd::getSite(i);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const gSiteMu siteMu = GInd::getSiteMu(site, mu);
            diff = std::max(diff, static_cast<double>(infnorm(accA.getLink(siteMu) - accB.getLink(siteMu))));
        }
    }
    return comm.reduceMax(diff);
}

// Number of doubles that differ (bitwise) between two compact vectors, all ranks.
template<size_t HaloDepth, size_t Ls>
double mdwfStep0TestVecDiff(CommunicationBase &comm, const MDWFCompactVec<HaloDepth, Ls> &a,
                            const MDWFCompactVec<HaloDepth, Ls> &b) {
    std::vector<double> ha, hb;
    a.toHost(ha);
    b.toHost(hb);
    double count = 0.0;
    for (size_t i = 0; i < ha.size(); i++) {
        if (std::memcmp(&ha[i], &hb[i], sizeof(double)) != 0) {
            count += 1.0;
        }
    }
    return comm.reduce(count);
}

template<size_t Ls>
void runMDWFDeflationStep0Test(CommunicationBase &commBase, MDWFDeflationStep0TestParameters &param) {
    constexpr size_t HaloDepth = 2;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using Vec = MDWFCompactVec<HaloDepth, Ls>;
    using Set = MDWFDeflSet<HaloDepth, Ls>;
    using Op = MDWFDeflOperator<HaloDepth, Ls>;
    using Field = typename Op::Field;
    using Subspace = MDWFDeflSubspace<HaloDepth, Ls>;

    const std::string dir = param.measurements_dir();
    const int K = param.k();
    const uint64_t seed = static_cast<uint64_t>(param.seed());
    const bool mmdag = param.light_form() == 1;

    grnd_state<false> h_rand;
    grnd_state<true> d_rand;
    h_rand.make_rng_state(param.seed());
    d_rand = h_rand;

    // Gauge fields: U = exp(i roughness P) 1, U' = exp(i md_step P') U.
    Gauge gauge(commBase, "MDWF_s0t_gauge"), momenta(commBase, "MDWF_s0t_momenta"), moved(commBase, "MDWF_s0t_moved");
    Gauge readBack(commBase, "MDWF_s0t_readback");
    gauge.one();
    momenta.gauss(d_rand.state);
    gauge.iterateOverBulkAllMu(MDWFStep0TestMove<HaloDepth>(gauge.getAccessor(), momenta.getAccessor(), param.roughness()));
    gauge.updateAll();
    momenta.gauss(d_rand.state);
    momenta.updateAll();
    moved.iterateOverBulkAllMu(MDWFStep0TestMove<HaloDepth>(gauge.getAccessor(), momenta.getAccessor(), param.md_step()));
    moved.updateAll();

    // ----------------------------------------------------------------------------------------------- D1, D2, D3
    mdwfDumpGauge<HaloDepth>(commBase, gauge, dir + "/s0t_U", "MDWF_s0t_stage_w");
    commBase.globalBarrier();
    readBack.one();
    mdwfReadGauge<HaloDepth>(commBase, readBack, dir + "/s0t_U", "MDWF_s0t_stage_r");
    mdwfStep0TestCheck(mdwfStep0TestLinkDiff<HaloDepth>(commBase, gauge, readBack) == 0.0,
                       "D1 gauge dump round trip (links) bit-exact");
    mdwfStep0TestCheck(mdwfStep0TestLinkDiff<HaloDepth>(commBase, gauge, moved) > 0.0,
                       "D1 control: different fields are seen as different");

    mdwfDumpGauge<HaloDepth>(commBase, momenta, dir + "/s0t_P", "MDWF_s0t_stage_w");
    commBase.globalBarrier();
    readBack.one();
    mdwfReadGauge<HaloDepth>(commBase, readBack, dir + "/s0t_P", "MDWF_s0t_stage_r");
    mdwfStep0TestCheck(mdwfStep0TestLinkDiff<HaloDepth>(commBase, momenta, readBack) == 0.0,
                       "D2 gauge dump round trip (momenta) bit-exact");

    Vec noise(commBase), copy(commBase);
    noise.noise(seed + 1);
    {
        Field field(commBase, "MDWF_s0t_field_a"), field2(commBase, "MDWF_s0t_field_b");
        noise.unpack(field);
        copy.pack(field);
        mdwfStep0TestCheck(mdwfStep0TestVecDiff(commBase, noise, copy) == 0.0, "D4 compact vector <-> field bit-exact");
        mdwfDumpEvenSpinor<HaloDepth, Ls>(commBase, field, dir + "/s0t_S", "MDWF_s0t_stage_sw");
        commBase.globalBarrier();
        copy.zero();
        copy.unpack(field2);
        mdwfReadEvenSpinor<HaloDepth, Ls>(commBase, field2, dir + "/s0t_S", "MDWF_s0t_stage_sr");
        copy.pack(field2);
        mdwfStep0TestCheck(mdwfStep0TestVecDiff(commBase, noise, copy) == 0.0,
                           "D3 even spinor dump round trip bit-exact");
        mdwfStep0TestCheck(std::abs(noise.norm2() / (24.0 * static_cast<double>(GIndexer<All, HaloDepth>::getLatData().globvol4)
                                                     / 2.0 * static_cast<double>(Ls)) - 1.0) < 0.05,
                           "D4 noise vector: mean |component|^2 = 2 within 5% (", noise.norm2(), ")");
    }

    // ------------------------------------------------------------------------------------------------------- H1
    {
        const int n = 12;
        std::vector<MDWFDeflC> h(n * n);
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                const double re = std::sin(1.0 + 3.7 * i + 1.3 * j), im = i == j ? 0.0 : std::cos(2.0 + 0.9 * i * j + j);
                h[i * n + j] = MDWFDeflC(re, im);
                h[j * n + i] = std::conj(h[i * n + j]);
            }
        }
        std::vector<double> ev;
        std::vector<MDWFDeflC> q;
        mdwfHermitianEigen(h, n, ev, q);
        double residual = 0.0, unitarity = 0.0;
        for (int l = 0; l < n; l++) {
            for (int i = 0; i < n; i++) {
                MDWFDeflC z(0.0, 0.0);
                for (int j = 0; j < n; j++) {
                    z += h[i * n + j] * q[j * n + l];
                }
                residual = std::max(residual, std::abs(z - ev[l] * q[i * n + l]));
            }
            for (int m = 0; m < n; m++) {
                MDWFDeflC z(0.0, 0.0);
                for (int i = 0; i < n; i++) {
                    z += std::conj(q[i * n + l]) * q[i * n + m];
                }
                unitarity = std::max(unitarity, std::abs(z - (l == m ? 1.0 : 0.0)));
            }
        }
        bool ascending = true;
        for (int l = 1; l < n; l++) {
            ascending = ascending && ev[l] >= ev[l - 1];
        }
        mdwfStep0TestCheck(residual < 1e-12 && unitarity < 1e-12 && ascending,
                           "H1 Hermitian eigensolver: residual ", residual, ", unitarity ", unitarity);
        // G = H^2 + 1 is positive definite; G^-1 G = 1; singular values of H = |eigenvalues|.
        std::vector<MDWFDeflC> g(n * n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                MDWFDeflC z(i == j ? 1.0 : 0.0, 0.0);
                for (int l = 0; l < n; l++) {
                    z += h[i * n + l] * h[l * n + j];
                }
                g[i * n + j] = z;
            }
        }
        const std::vector<MDWFDeflC> inv = mdwfHermitianInverse(g, n);
        double identity = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                MDWFDeflC z(0.0, 0.0);
                for (int l = 0; l < n; l++) {
                    z += inv[i * n + l] * g[l * n + j];
                }
                identity = std::max(identity, std::abs(z - (i == j ? 1.0 : 0.0)));
            }
        }
        std::vector<double> absEv(n);
        for (int l = 0; l < n; l++) {
            absEv[l] = std::abs(ev[l]);
        }
        std::sort(absEv.begin(), absEv.end(), [](double a, double b) { return a > b; });
        const std::vector<double> sigma = mdwfSingularValues(h, n, n);
        double svd = 0.0;
        for (int l = 0; l < n; l++) {
            svd = std::max(svd, std::abs(sigma[l] - absEv[l]));
        }
        mdwfStep0TestCheck(identity < 1e-10 && svd < 1e-7, "H1 inverse: |G^-1 G - 1| ", identity,
                           ", singular values: ", svd);
    }

    // ------------------------------------------------------------------------------------------------------- O1
    Gauge fermionGauge(commBase, "MDWF_s0t_fgauge"), fermionGaugeMoved(commBase, "MDWF_s0t_fgmoved");
    mdwfFermionGaugeField<HaloDepth>(fermionGauge, gauge, param.antiperiodic_t() != 0);
    mdwfFermionGaugeField<HaloDepth>(fermionGaugeMoved, moved, param.antiperiodic_t() != 0);
    Op op(fermionGauge, param.M5(), param.mf(), param.b5(), mmdag, "MDWF_s0t_opa");
    Op opMoved(fermionGaugeMoved, param.M5(), param.mf(), param.b5(), mmdag, "MDWF_s0t_opb");
    op.refresh();
    opMoved.refresh();
    Vec x(commBase), y(commBase), ax(commBase), ay(commBase);
    x.noise(seed + 2);
    y.noise(seed + 3);
    op.apply(ax, x);
    op.apply(ay, y);
    {
        const MDWFDeflC xay = mdwfDeflDot(x, ay), axy = mdwfDeflDot(ax, y), xax = mdwfDeflDot(x, ax);
        mdwfStep0TestCheck(std::abs(xay - axy) < 1e-11 * std::abs(xay) + 1e-9 * std::abs(xax) * 1e-3
                           && xax.real() > 0.0 && std::abs(xax.imag()) < 1e-11 * xax.real(),
                           "O1 operator Hermitian (<x|Ay> - <Ax|y> = ", std::abs(xay - axy), " of ", std::abs(xay),
                           ") and positive (<x|Ax> = ", xax.real(), ")");
    }

    // ------------------------------------------------------------------------------------------------------- E1
    Set v, vMoved;
    MDWFDeflEigenParameters ep;
    ep.nev = K;
    const MDWFDeflEigenResult eig = mdwfDeflLowModes(op, v, ep, seed + 4, commBase);
    {
        double worst = 0.0, ortho = 0.0;
        for (int k = 0; k < K; k++) {
            worst = std::max(worst, eig.residual[k] / eig.lambda[k]);
        }
        const std::vector<MDWFDeflC> g = mdwfDeflGram(mdwfDeflPointers(v, K), mdwfDeflPointers(v, K));
        for (int a = 0; a < K; a++) {
            for (int b = 0; b < K; b++) {
                ortho = std::max(ortho, std::abs(g[a * K + b] - (a == b ? 1.0 : 0.0)));
            }
        }
        bool ascending = eig.lambda[0] > 0.0;
        for (int k = 1; k < K; k++) {
            ascending = ascending && eig.lambda[k] >= eig.lambda[k - 1];
        }
        mdwfStep0TestCheck(eig.converged && worst < 1e-6 && ortho < 1e-10 && ascending, "E1 ", K, " lowest eigenpairs: "
                           "converged ", eig.converged, ", lambda ", eig.lambda.front(), " ... ", eig.lambda.back(),
                           " (lambda_max bound ", eig.hi, "), worst residual / lambda ", worst, ", orthonormality ",
                           ortho, ", attempts ", eig.attempts, ", applications ", eig.applications);
        // One more eigenvalue must not lie below the K-th: a missed mode would show here.
        Set more;
        MDWFDeflEigenParameters ep2 = ep;
        ep2.nev = K + 4;
        const MDWFDeflEigenResult eig2 = mdwfDeflLowModes(op, more, ep2, seed + 40, commBase);
        double shift = 0.0;
        for (int k = 0; k < K; k++) {
            shift = std::max(shift, std::abs(eig2.lambda[k] - eig.lambda[k]) / eig.lambda[k]);
        }
        mdwfStep0TestCheck(shift < 1e-8, "E1 the same ", K, " eigenvalues from a run with nev = ", K + 4,
                           " and another start vector: largest relative difference ", shift);
        // File round trip.
        mdwfDeflWriteSet<HaloDepth, Ls>(commBase, v, K, eig.lambda, dir + "/s0t_set");
        Set back;
        std::vector<double> lambdaBack;
        mdwfDeflReadSet<HaloDepth, Ls>(commBase, back, K, lambdaBack, dir + "/s0t_set");
        double differing = 0.0;
        for (int k = 0; k < K; k++) {
            differing += mdwfStep0TestVecDiff(commBase, *v[k], *back[k]);
            differing += lambdaBack[k] == eig.lambda[k] ? 0.0 : 1.0;
        }
        mdwfStep0TestCheck(differing == 0.0, "D4 eigenvector set file round trip bit-exact");
    }

    // ------------------------------------------------------------------------------------- C1, C2, C3, C4
    const double precision = 1e-8;
    const int maxIter = 100000;
    Vec b(commBase), sol(commBase);
    b.noise(seed + 5);
    const MDWFDeflCGResult none = mdwfDeflCG<Op, HaloDepth, Ls>(op, sol, b, nullptr, MDWFDeflNone, precision, maxIter,
                                                               commBase);
    mdwfStep0TestCheck(none.converged && none.trueResidual < 2e-8, "C1 plain CG: ", none.iterations,
                       " iterations, true residual ", none.trueResidual);
    Subspace sub;
    sub.setup(op, mdwfDeflPointers(v, K), commBase);
    const MDWFDeflCGResult exact = mdwfDeflCG(op, sol, b, &sub, MDWFDeflProject, precision, maxIter, commBase);
    mdwfStep0TestCheck(exact.converged && exact.trueResidual < 2e-8 && exact.iterations < none.iterations,
                       "C2 deflated CG, exact subspace: ", exact.iterations, " iterations (plain ", none.iterations,
                       "), true residual ", exact.trueResidual);
    const MDWFDeflCGResult init = mdwfDeflCG(op, sol, b, &sub, MDWFDeflInit, precision, maxIter, commBase);
    mdwfStep0TestCheck(init.converged && init.trueResidual < 2e-8 && init.iterations <= none.iterations,
                       "C4 deflated initial guess (Galerkin) + CG: ", init.iterations, " iterations, true residual ",
                       init.trueResidual);
    {
        Subspace spectral;
        spectral.setupSpectral(mdwfDeflPointers(v, K), eig.lambda);
        const MDWFDeflCGResult r = mdwfDeflCG(op, sol, b, &spectral, MDWFDeflInit, precision, maxIter, commBase);
        mdwfStep0TestCheck(r.converged && r.trueResidual < 2e-8 && std::abs(r.iterations - init.iterations) <= 5,
                           "C4 deflated initial guess (spectral, exact pairs) + CG: ", r.iterations,
                           " iterations, true residual ", r.trueResidual);
    }
    {
        Set junk;
        mdwfDeflAllocate(junk, 4, commBase);
        for (int k = 0; k < 4; k++) {
            junk[k]->noise(seed + 100 + k);
        }
        mdwfDeflOrthonormalize(junk, 4);
        Subspace bad;
        bad.setup(op, mdwfDeflPointers(junk, 4), commBase);
        const MDWFDeflCGResult r = mdwfDeflCG(op, sol, b, &bad, MDWFDeflProject, precision, maxIter, commBase);
        mdwfStep0TestCheck(r.converged && r.trueResidual < 2e-8, "C3 deflated CG, noise subspace: ", r.iterations,
                           " iterations, true residual ", r.trueResidual);
    }

    // ------------------------------------------------------------------------------------------------------- T1
    {
        const MDWFDeflEigenResult eigMoved = mdwfDeflLowModes(opMoved, vMoved, ep, seed + 6, commBase);
        const std::vector<const Vec *> pExact = mdwfDeflPointers(vMoved, K);
        Set t1, t2, t3, xs, zs;
        Vec work(commBase);
        for (Set *s : {&t1, &t2, &t3, &xs, &zs}) {
            mdwfDeflAllocate(*s, K, commBase);
        }
        for (int k = 0; k < K; k++) {
            *t2[k] = *v[k];
            *t3[k] = *v[k];
        }
        const MDWFDeflRitz r1 = mdwfDeflRayleighRitz(opMoved, mdwfDeflPointers(v, K), K, t1, work, true);
        const MDWFDeflRitz r2 = mdwfDeflTrackStep(opMoved, &op, t2, K, xs, zs, work, true);
        const MDWFDeflRitz r3 = mdwfDeflTrackStep(opMoved, static_cast<Op *>(nullptr), t3, K, xs, zs, work, true);
        bool bounds = true;
        for (int k = 0; k < K; k++) {
            const double tol = 1e-9 * eigMoved.lambda[K - 1];
            bounds = bounds && r1.theta[k] >= eigMoved.lambda[k] - tol && r2.theta[k] >= eigMoved.lambda[k] - tol
                     && r3.theta[k] >= eigMoved.lambda[k] - tol && r2.theta[k] <= r1.theta[k] + tol
                     && r3.theta[k] <= r1.theta[k] + tol;
        }
        auto capture = [&](const Set &s) {
            const std::vector<double> sigma = mdwfDeflPrincipalCosines(pExact, mdwfDeflPointers(s, K));
            double c = 0.0;
            for (const double sg : sigma) {
                c += sg * sg / K;
            }
            return c;
        };
        const double c0 = capture(v), c1 = capture(t1), c2 = capture(t2), c3 = capture(t3);
        mdwfStep0TestCheck(eigMoved.converged && bounds && std::abs(c0 - c1) < 1e-9,
                           "T1 tracking to a field moved by md_step: Ritz bounds hold; mean cos^2 with the exact "
                           "subspace: T0 = T1 ", c0, ", T2 ", c2, ", T3 ", c3, "; largest Ritz value T1 ", r1.theta.back(),
                           ", T2 ", r2.theta.back(), ", T3 ", r3.theta.back(), ", exact ", eigMoved.lambda.back());
        Subspace subMoved;
        Vec solMoved(commBase);
        const MDWFDeflCGResult plain = mdwfDeflCG<Op, HaloDepth, Ls>(opMoved, solMoved, b, nullptr, MDWFDeflNone,
                                                                    precision, maxIter, commBase);
        subMoved.setup(opMoved, pExact, commBase);
        const MDWFDeflCGResult ex = mdwfDeflCG(opMoved, solMoved, b, &subMoved, MDWFDeflProject, precision, maxIter, commBase);
        subMoved.setup(opMoved, mdwfDeflPointers(v, K), commBase);
        const MDWFDeflCGResult st = mdwfDeflCG(opMoved, solMoved, b, &subMoved, MDWFDeflProject, precision, maxIter, commBase);
        subMoved.setup(opMoved, mdwfDeflPointers(t2, K), commBase);
        const MDWFDeflCGResult tr = mdwfDeflCG(opMoved, solMoved, b, &subMoved, MDWFDeflProject, precision, maxIter, commBase);
        mdwfStep0TestCheck(plain.converged && ex.converged && st.converged && tr.converged && ex.trueResidual < 2e-8
                           && st.trueResidual < 2e-8 && tr.trueResidual < 2e-8,
                           "T1 CG on the moved field: plain ", plain.iterations, ", exact ", ex.iterations,
                           ", stale span (T1) ", st.iterations, ", T2 ", tr.iterations, " iterations");
    }

    // --------------------------------------------------------------------------------------- R1, R2, R3, R4
    {
        typedef std::map<int, std::unique_ptr<Vec>> Snapshots;
        auto keep = [&commBase](Snapshots &into) {
            return MDWFDeflSnapshotCallback<HaloDepth, Ls>([&into, &commBase](int n, const Vec &x, double) {
                into[n].reset(new Vec(commBase));
                *into[n] = x;
            });
        };
        auto relDiff = [](const Vec &a, const Vec &b, Vec &work) {
            work = a;
            work.axpyThisB(-1.0, b);
            return std::sqrt(work.norm2() / b.norm2());
        };
        Vec w(commBase), ref(commBase), work(commBase);
        w.noise(seed + 7);
        const double lo = eig.lambda[K - 1], hi = eig.hi;
        Snapshots cg, cheb;
        const long cgApps = mdwfDeflCGSnapshots(op, w, {0, 1, 3, 8, 20}, keep(cg), commBase);
        const long chebApps = mdwfDeflChebyshevSnapshots(op, w, lo, hi, {0, 1, 2, 7, 16}, keep(cheb), commBase);
        mdwfStep0TestCheck(mdwfStep0TestVecDiff(commBase, *cg[0], w) == 0.0
                           && mdwfStep0TestVecDiff(commBase, *cheb[0], w) == 0.0 && cgApps == 20 && chebApps == 16,
                           "R1 refresh snapshot n = 0 is the input (CG and Chebyshev, bit-exact); applications ",
                           cgApps, " and ", chebApps);
        double worst = 0.0, bits = 0.0;
        for (const int n : {1, 3, 8, 20}) {
            mdwfDeflCG<Op, HaloDepth, Ls>(op, ref, w, nullptr, MDWFDeflNone, 0.0, n, commBase);
            worst = std::max(worst, relDiff(*cg[n], ref, work));
            bits += mdwfStep0TestVecDiff(commBase, *cg[n], ref);
        }
        mdwfStep0TestCheck(worst < 1e-12, "R2 nested CG snapshots n = 1 3 8 20 = separate fixed-n CG runs: largest "
                           "relative difference ", worst, " (", bits, " differing doubles)");
        worst = 0.0;
        bits = 0.0;
        for (const int n : {1, 2, 7, 16}) {
            MDWFChebyshevOperator<Vec, Op> filter(op, lo, hi, n, commBase, "MDWF_s0t_cheb");
            filter.apply(ref, w);
            ref *= COMPLEX(double)(-1.0, 0.0);
            worst = std::max(worst, relDiff(*cheb[n], ref, work));
            bits += mdwfStep0TestVecDiff(commBase, *cheb[n], ref);
        }
        mdwfStep0TestCheck(worst < 1e-12, "R3 nested Chebyshev snapshots, degrees 1 2 7 16 = MDWFChebyshevOperator of "
                           "each degree: largest relative difference ", worst, " (", bits, " differing doubles)");
        // R4: u = sum_k c_k v_k, c_k = 1 / (k + 1).
        std::vector<MDWFDeflC> c(K);
        for (int k = 0; k < K; k++) {
            c[k] = MDWFDeflC(1.0 / (k + 1.0), 0.0);
        }
        Vec u(commBase);
        u.combine(mdwfDeflPointers(v, K), c, true);
        Snapshots filtered;
        mdwfDeflChebyshevSnapshots(op, u, lo, hi, {4, 16, 64}, keep(filtered), commBase);
        double worstValue = 0.0, worstVector = 0.0, tMax = 0.0;
        for (const int n : {4, 16, 64}) {
            std::vector<MDWFDeflC> expected(K);
            for (int k = 0; k < K; k++) {
                const double l = (hi + lo - 2.0 * eig.lambda[k]) / (hi - lo);   // T_n(l) by its recurrence
                double t0 = 1.0, t1 = l;
                for (int d = 2; d <= n; d++) {
                    const double t2 = 2.0 * l * t1 - t0;
                    t0 = t1;
                    t1 = t2;
                }
                expected[k] = c[k] * t1;
                tMax = std::max(tMax, std::abs(t1));
                const MDWFDeflC got = mdwfDeflDot(*v[k], *filtered[n]);
                worstValue = std::max(worstValue, std::abs(got - expected[k]) / std::sqrt(filtered[n]->norm2()));
            }
            ref.combine(mdwfDeflPointers(v, K), expected, true);
            worstVector = std::max(worstVector, relDiff(*filtered[n], ref, work));
        }
        mdwfStep0TestCheck(worstValue < 1e-6 && worstVector < 1e-6, "R4 Chebyshev filter on exact eigenvectors, "
                           "degrees 4 16 64 on [", lo, ", ", hi, "]: largest |<v_k|y_n> - c_k T_n(L(lambda_k))| / |y_n| ",
                           worstValue, ", |y_n - sum_k c_k T_n v_k| / |y_n| ", worstVector, " (largest T_n ", tMax, ")");
    }

    if (mdwfStep0TestFailures == 0) {
        rootLogger.info("mdwfDeflationStep0Test: ALL PASS");
    } else {
        throw std::runtime_error(stdLogger.fatal("mdwfDeflationStep0Test: ", mdwfStep0TestFailures, " check(s) FAILED"));
    }
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);
        MDWFDeflationStep0TestParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfDeflationStep0Test.param", argc, argv);
        commBase.forceHalos(param.force_halos() != 0);
        commBase.init(param.nodeDim());
        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);
        if (param.ls() == 8) {
            runMDWFDeflationStep0Test<8>(commBase, param);
        } else if (param.ls() == 24) {
            runMDWFDeflationStep0Test<24>(commBase, param);
        } else {
            throw std::runtime_error(stdLogger.fatal("mdwfDeflationStep0Test supports Ls = 8 or 24, got ", param.ls()));
        }
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
