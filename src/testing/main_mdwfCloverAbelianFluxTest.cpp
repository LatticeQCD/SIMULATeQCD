/*
 * Analytic check of the sign and normalization of the clover term in the MDWF Wilson kernel.
 *
 * The kernel D_K = D_W(-M5) - (c_sw / 2) sum_{mu < nu} sigma_{mu nu} F_{mu nu} (applyMDWFCloverWilsonSlice, the
 * clover blocks of preCalcFmunu in DWilson.h) is meant to be the Sheikholeslami-Wohlert / Luescher-Sint-Sommer-
 * Weisz operator: with the hopping convention used here, -1/2 [(1 - gamma_mu) U_mu(x) psi(x + mu)
 * + (1 + gamma_mu) U_mu(x - mu)^dagger psi(x - mu)], the LSSW clover term c_sw (i/4) sum_{mu nu} sigma_{mu nu}
 * Fhat_{mu nu}, Fhat = (Q - Q^dagger) / 8 with Q the counter-clockwise four-leaf sum and sigma_{mu nu} =
 * (i/2) [gamma_mu, gamma_nu], equals -(c_sw/2) sum_{mu<nu} sigma_{mu nu} F_{mu nu} with the Hermitian
 * F = -(i/8)(Q - Q^dagger) of FieldStrengthTensor (whose colour trace is removed).
 *
 * Test field: a constant abelian flux in one plane (p, q), p < q, embedded diagonally in SU(3) with phases
 * phi_c = 2 pi n_c / (L_p L_q), n = (1, 1, -2):
 *   U_q(x) = diag(exp(i phi_c x_p)),  U_p(x) = diag(exp(-i phi_c L_p x_q)) on x_p = L_p - 1 (1 elsewhere),
 *   all other links 1.
 * Every plaquette U_p(x) U_q(x+p) U_p(x+q)^dagger U_q(x)^dagger is diag(exp(i phi_c)), so all four leaves agree
 * and F_pq = diag(sin phi_c) - (1/3) sum_c sin phi_c exactly, on every site; all other F vanish. Then
 *   D_K(c_sw) psi - D_K(0) psi = -(c_sw / 2) sigma_pq F_pq psi
 * for any psi (the hopping and mass terms cancel). sigma_pq is built here from the gamma matrices of
 * fullSpinor.h (GammaX/Y/Z/TMultVec, the matrices the hopping term uses for directions 0..3), independently of
 * the hand-written blocks in preCalcFmunu, and compared site by site for all six planes and two c_sw.
 *
 * Usage: ./mdwfCloverAbelianFluxTest (parameter/tests/mdwfFifthDimTest.param, 6^4, single rank).
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFWilsonSlice.h"

#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>

using cplx = std::complex<double>;
using Mat4 = std::array<std::array<cplx, 4>, 4>;

// Constant abelian flux in the plane (p, q), colour phases phi_c = 2 pi n_c / (L_p L_q), n = (1, 1, -2).
template<size_t HaloDepth>
struct MDWFAbelianFluxLinks {
    int _p, _q, _lp;
    double _phi[3];

    MDWFAbelianFluxLinks(int p, int q, int lp, int lq) : _p(p), _q(q), _lp(lp) {
        const int n[3] = {1, 1, -2};
        for (int c = 0; c < 3; c++) {
            _phi[c] = 2.0 * M_PI * n[c] / static_cast<double>(lp * lq);
        }
    }

    __host__ __device__ static int coordinate(const sitexyzt &x, int mu) {
        return mu == 0 ? static_cast<int>(x.x) : mu == 1 ? static_cast<int>(x.y)
                                              : mu == 2 ? static_cast<int>(x.z) : static_cast<int>(x.t);
    }

    __host__ __device__ SU3<double> operator()(gSiteMu siteMu) {
        SU3<double> link = su3_one<double>();
        double angle[3] = {0.0, 0.0, 0.0};
        const int xp = coordinate(siteMu.coord, _p), xq = coordinate(siteMu.coord, _q);
        if (siteMu.mu == _q) {
            for (int c = 0; c < 3; c++) {
                angle[c] = _phi[c] * xp;
            }
        } else if (siteMu.mu == _p && xp == _lp - 1) {
            for (int c = 0; c < 3; c++) {
                angle[c] = -_phi[c] * _lp * xq;
            }
        }
        for (int c = 0; c < 3; c++) {
            link(c, c) = COMPLEX(double)(cos(angle[c]), sin(angle[c]));
        }
        return link;
    }
};

// Deterministic, everywhere nonzero test spinor (Ls = 1).
template<size_t HaloDepth>
struct MDWFFluxTestSpinor {
    __host__ __device__ Vect12<double> operator()(gSiteStack site) {
        Vect12<double> out(0.0);
        for (int k = 0; k < 12; k++) {
            const double a = 0.37 * static_cast<double>(site.isite % 97) + 1.3 * k + 0.1;
            out.data[k] = COMPLEX(double)(cos(a) + 0.2 * k, sin(1.7 * a) - 0.1);
        }
        return out;
    }
};

// The matrices of GammaXMultVec, GammaYMultVec, GammaZMultVec, GammaTMultVec (fullSpinor.h): res = G * v.
Mat4 mdwfFluxTestGamma(int mu) {
    const cplx i(0.0, 1.0);
    Mat4 g{};
    if (mu == 0) {
        g[0][3] = i;   g[1][2] = i;   g[2][1] = -i;  g[3][0] = -i;
    } else if (mu == 1) {
        g[0][3] = -1;  g[1][2] = 1;   g[2][1] = 1;   g[3][0] = -1;
    } else if (mu == 2) {
        g[0][2] = i;   g[1][3] = -i;  g[2][0] = -i;  g[3][1] = i;
    } else {
        g[0][2] = 1;   g[1][3] = 1;   g[2][0] = 1;   g[3][1] = 1;
    }
    return g;
}

// sigma_{mu nu} = (i/2) [gamma_mu, gamma_nu].
Mat4 mdwfFluxTestSigma(int mu, int nu) {
    const Mat4 a = mdwfFluxTestGamma(mu), b = mdwfFluxTestGamma(nu);
    Mat4 s{};
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            cplx v = 0.0;
            for (int k = 0; k < 4; k++) {
                v += a[r][k] * b[k][c] - b[r][k] * a[k][c];
            }
            s[r][c] = cplx(0.0, 0.5) * v;
        }
    }
    return s;
}

template<size_t HaloDepth>
bool runPlane(CommunicationBase &commBase, int p, int q, double csw) {
    typedef GIndexer<All, HaloDepth> GInd;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, 1>;
    using HostSpinor = MDWFSpinor<double, false, All, HaloDepth, 1>;
    using Clover = Spinorfield<double, true, All, HaloDepth, 18, 1>;
    const LatticeData lat = GInd::getLatData();
    const int extent[4] = {static_cast<int>(lat.lx), static_cast<int>(lat.ly), static_cast<int>(lat.lz),
                           static_cast<int>(lat.lt)};
    const double mass = 4.0 - 1.4;   // D_W(-M5) at M5 = 1.4; the mass term cancels in the difference

    Gaugefield<double, true, HaloDepth, R18> gauge(commBase, "MDWF_flux_gauge");
    gauge.iterateOverBulkAllMu(MDWFAbelianFluxLinks<HaloDepth>(p, q, extent[p], extent[q]));
    gauge.updateAll();

    Spinor in(commBase, "MDWF_flux_in"), withClover(commBase, "MDWF_flux_csw"), without(commBase, "MDWF_flux_0"),
            tmp(commBase, "MDWF_flux_tmp");
    Clover fUp(commBase, "MDWF_flux_fup"), fLo(commBase, "MDWF_flux_flo"), fiUp(commBase, "MDWF_flux_fiup"),
            fiLo(commBase, "MDWF_flux_filo");
    in.template iterateOverBulk<BLOCKSIZE>(MDWFFluxTestSpinor<HaloDepth>());
    in.updateAll();
    applyMDWFCloverWilsonSlice<double, HaloDepth, HaloDepth, 1>(withClover, gauge, tmp, fUp, fLo, fiUp, fiLo, in,
                                                                mass, csw, true);
    applyMDWFCloverWilsonSlice<double, HaloDepth, HaloDepth, 1>(without, gauge, tmp, fUp, fLo, fiUp, fiLo, in,
                                                                mass, 0.0, true);

    HostSpinor hIn(commBase, "MDWF_flux_hin"), hWith(commBase, "MDWF_flux_hcsw"), hWithout(commBase, "MDWF_flux_h0");
    hIn = in;
    hWith = withClover;
    hWithout = without;
    Vect12ArrayAcc<double> aIn = hIn.getAccessor(), aWith = hWith.getAccessor(), aWithout = hWithout.getAccessor();

    // Expected F_pq per colour: sin(phi_c) minus the colour average (FieldStrengthTensor removes the trace).
    const int n[3] = {1, 1, -2};
    double f[3], mean = 0.0;
    for (int c = 0; c < 3; c++) {
        f[c] = std::sin(2.0 * M_PI * n[c] / static_cast<double>(extent[p] * extent[q]));
        mean += f[c] / 3.0;
    }
    for (double &fc : f) {
        fc -= mean;
    }
    const Mat4 sigma = mdwfFluxTestSigma(p, q);

    double maxDiff = 0.0, maxExpected = 0.0;
    for (size_t i = 0; i < lat.vol4; i++) {
        const gSiteStack site = GInd::getSiteStack(GInd::getSite(i), 0);
        const Vect12<double> v = aIn.getElement(site);
        const Vect12<double> w1 = aWith.getElement(site), w0 = aWithout.getElement(site);
        for (int s = 0; s < 4; s++) {
            for (int c = 0; c < 3; c++) {
                cplx expected = 0.0;
                for (int t = 0; t < 4; t++) {
                    const COMPLEX(double) vt = v.data[3 * t + c];
                    expected += sigma[s][t] * cplx(real(vt), imag(vt));
                }
                expected *= -0.5 * csw * f[c];
                const COMPLEX(double) got = w1.data[3 * s + c] - w0.data[3 * s + c];
                maxDiff = std::max(maxDiff, std::abs(cplx(real(got), imag(got)) - expected));
                maxExpected = std::max(maxExpected, std::abs(expected));
            }
        }
    }
    const bool passed = maxExpected > 1e-3 && maxDiff <= 1e-12 * maxExpected;
    rootLogger.info("MDWF clover abelian flux: plane (", "xyzt"[p], ",", "xyzt"[q], "), c_sw = ", csw,
                    ": max |clover term| = ", maxExpected, ", max |D_K(c_sw) - D_K(0) - expected| = ", maxDiff,
                    ", passed = ", passed);
    return passed;
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
        if (GIndexer<All, HaloDepth>::getLatData().vol4 != GIndexer<All, HaloDepth>::getLatData().globvol4) {
            throw std::runtime_error(stdLogger.fatal("MDWF clover abelian flux test is single-rank"));
        }

        bool passed = true;
        for (const double csw : {0.5, 1.0}) {
            for (int p = 0; p < 4; p++) {
                for (int q = p + 1; q < 4; q++) {
                    passed = runPlane<HaloDepth>(commBase, p, q, csw) && passed;
                }
            }
        }
        if (!passed) {
            throw std::runtime_error(stdLogger.fatal("MDWF clover abelian flux test failed"));
        }
        rootLogger.info("MDWF clover abelian flux test passed: sign and normalization of the clover term agree with "
                        "-(c_sw/2) sigma_{mu nu} F_{mu nu} in all six planes");
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
