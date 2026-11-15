/*
 * Low modes of the MDWF 4D propagator and low-mode averaging (LMA) of the connected screening correlators, plus
 * low-mode-exact disconnected loops. Port of GPT's measure_lowmode_split_mdwf.py (DSDRMDWFruns/Measurements), with
 * the same operator, the same algebra and the production conventions of MDWFMeasurements.h. Branch mdwf-lma,
 * 2026-11-06; plan and validation ladder: DSDRMDWFruns/SQCDruns/LMA_CV_PLAN.md.
 *
 * Operator. S = Export M(m)^-1 D_- Import is the raw 4D propagator of production (GPT's fermion.propagator) and
 * A = S g5 is Hermitian. H = A^-1 = g5 S^-1 has eigenpairs H psi_k = mu_k psi_k, hence (complete basis)
 *     S = sum_k psi_k psi_k^+ g5 / mu_k,    S_L = the same sum over the N modes of smallest |mu|.
 * The modes come from thick-restart Lanczos (MDWFThickRestartLanczos.h) on Op = -A^2 (smallest eigenvalues of Op =
 * largest a^2 = smallest |mu|; 2 solves per application), then Rayleigh-Ritz with A in span{v_i, A v_i}, which
 * separates the +-mu pairs that share an eigenspace of A^2.
 *
 * Spin densities per slice (the only GPU work after the modes):  D_kl[s][s'](n) = sum_{x_mu = n} sum_c
 * conj(psi_k[s, c]) psi_l[s', c], so that B^M_kl(n) = sum_{x_mu = n} psi_k^+ M psi_l = sum_{ss'} M[s][s'] D_kl[s][s'].
 *
 * Connected channel G, production convention C_G(n) = sum_{x_mu = n} Tr[S(x,x0)^+ L S(x,x0) R], L = g5 G, R = G^+ g5
 * (MDWFMesonChannelDensity). With S_L(x, x0) = sum_k psi_k(x) phi_k^+, phi_k = g5 psi_k(x0) / mu_k:
 *     point LL     C^LL_pt(n)   = sum_kl B^L_kl(n) b_lk,           b_lk = phi_l^+ R phi_k
 *     all-to-all   C^LL_a2a(dn) = (1/V4) sum_kl sum_n0 B^L_kl(n0 + dn) B^{g5 R g5}_lk(n0) / (mu_k mu_l)
 * (the point LL averaged over every source position x0 at fixed separation).  LMA: C = C_pt - C^LL_pt + C^LL_a2a.
 *
 * Loops (production L_raw = eta^+ S eta, L5_raw = eta^+ g5 S eta per slice), Gamma = 1 or g5:
 *     exact low part  tr[Gamma S_L](n)        = sum_k B^{g5 Gamma}_kk(n) / mu_k
 *     stochastic low  eta^+ Gamma S_L eta (n) = sum_k (sum_{x_mu = n} eta^+ Gamma psi_k) c_k / mu_k, c_k = psi_k^+ g5 eta
 *     LMA loop        L' = L - stochastic low + exact low        (same expectation, less noise)
 *
 * Self-tests (lma_selftest = 1): the point LL is also built explicitly as S_L columns in the store and contracted with
 * the production kernel MDWFMesonChannelDensity; the stochastic low part of the first light noise vector is also
 * computed from S_L eta directly. Both must agree with the formulas to rounding (LMA_CV_PLAN.md, test L2).
 *
 * Single rank (as the rest of mdwfMeasure).
 */

#pragma once

#include "MDWFMeasurements.h"
#include "MDWFThickRestartLanczos.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------------------------------------------
// Functors
// ---------------------------------------------------------------------------------------------------------------

// Store column `column` <- (g5) in4; the other stacks are copied unchanged.
template<size_t HaloDepth, size_t Ls>
struct MDWFStoreColumnFrom4D {
    Vect12ArrayAcc<double> _store, _in;
    size_t _column;
    bool _gamma5;

    MDWFStoreColumnFrom4D(const MDWFSpinor<double, true, All, HaloDepth, Ls> &store,
                          const MDWFSpinor<double, true, All, HaloDepth, 1> &in, size_t column, bool gamma5)
        : _store(store.getAccessor()), _in(in.getAccessor()), _column(column), _gamma5(gamma5) {}

    __host__ __device__ Vect12<double> operator()(gSiteStack site) {
        typedef GIndexer<All, HaloDepth> GInd;
        if (site.stack != _column) {
            return _store.getElement(site);
        }
        Vect12<double> v = _in.getElement(GInd::getSiteStack(site, 0));
        if (_gamma5) {
            for (int k = 6; k < 12; k++) {
                v.data[k] = -v.data[k];
            }
        }
        return v;
    }
};

// 4D field <- store column.
template<size_t HaloDepth, size_t Ls>
struct MDWF4DFromStoreColumn {
    Vect12ArrayAcc<double> _store;
    size_t _column;

    MDWF4DFromStoreColumn(const MDWFSpinor<double, true, All, HaloDepth, Ls> &store, size_t column)
        : _store(store.getAccessor()), _column(column) {}

    __host__ __device__ Vect12<double> operator()(gSiteStack site) {
        typedef GIndexer<All, HaloDepth> GInd;
        return _store.getElement(GInd::getSiteStack(site, _column));
    }
};

// Store column `column` <- (column, or 0 if zero) + coef * mode; the other stacks unchanged.
template<size_t HaloDepth, size_t Ls>
struct MDWFColumnAxpy {
    Vect12ArrayAcc<double> _store, _mode;
    size_t _column;
    COMPLEX(double) _coef;
    bool _zero;

    MDWFColumnAxpy(const MDWFSpinor<double, true, All, HaloDepth, Ls> &store,
                   const MDWFSpinor<double, true, All, HaloDepth, 1> &mode, size_t column, COMPLEX(double) coef,
                   bool zero)
        : _store(store.getAccessor()), _mode(mode.getAccessor()), _column(column), _coef(coef), _zero(zero) {}

    __host__ __device__ Vect12<double> operator()(gSiteStack site) {
        typedef GIndexer<All, HaloDepth> GInd;
        Vect12<double> v = _store.getElement(site);
        if (site.stack != _column) {
            return v;
        }
        const Vect12<double> p = _mode.getElement(GInd::getSiteStack(site, 0));
        for (int k = 0; k < 12; k++) {
            v.data[k] = (_zero ? COMPLEX(double)(0.0, 0.0) : v.data[k]) + _coef * p.data[k];
        }
        return v;
    }
};

// Real or imaginary part of sum_c conj(a[3 s + c]) b[3 sp + c] at a site (one entry of the spin density matrix).
template<size_t HaloDepth>
struct MDWFPairSpinDensity {
    Vect12ArrayAcc<double> _a, _b;
    int _s, _sp;
    bool _imag;

    MDWFPairSpinDensity(const MDWFSpinor<double, true, All, HaloDepth, 1> &a,
                        const MDWFSpinor<double, true, All, HaloDepth, 1> &b, int s, int sp, bool imag)
        : _a(a.getAccessor()), _b(b.getAccessor()), _s(s), _sp(sp), _imag(imag) {}

    __host__ __device__ double operator()(gSite site) {
        typedef GIndexer<All, HaloDepth> GInd;
        const Vect12<double> va = _a.getElement(GInd::getSiteStack(site, 0));
        const Vect12<double> vb = _b.getElement(GInd::getSiteStack(site, 0));
        COMPLEX(double) z(0.0, 0.0);
        for (int c = 0; c < 3; c++) {
            z += conj(va.data[3 * _s + c]) * vb.data[3 * _sp + c];
        }
        return _imag ? imag(z) : real(z);
    }
};

// Real or imaginary part of component k of a 4D field at the global site x0 (0 elsewhere): a global sum picks it.
template<size_t HaloDepth>
struct MDWFSiteComponent {
    Vect12ArrayAcc<double> _a;
    int _x0[4];
    int _k;
    bool _imag;

    MDWFSiteComponent(const MDWFSpinor<double, true, All, HaloDepth, 1> &a, const std::array<int, 4> &x0, int k,
                      bool imag)
        : _a(a.getAccessor()), _k(k), _imag(imag) {
        for (int i = 0; i < 4; i++) {
            _x0[i] = x0[i];
        }
    }

    __host__ __device__ double operator()(gSite site) {
        typedef GIndexer<All, HaloDepth> GInd;
        LatticeData lat = GInd::getLatData();
        const sitexyzt g = lat.globalPos(site.coord);
        if (g.x != _x0[0] || g.y != _x0[1] || g.z != _x0[2] || g.t != _x0[3]) {
            return 0.0;
        }
        const COMPLEX(double) v = _a.getElement(GInd::getSiteStack(site, 0)).data[_k];
        return _imag ? imag(v) : real(v);
    }
};

// Real or imaginary part of sum_k conj(store_col[k]) (g5 mode)[k]: eta^+ Gamma psi with eta in a store column.
template<size_t HaloDepth, size_t Ls>
struct MDWFColumnModeDensity {
    Vect12ArrayAcc<double> _store, _mode;
    size_t _column;
    bool _gamma5, _imag;

    MDWFColumnModeDensity(const MDWFSpinor<double, true, All, HaloDepth, Ls> &store,
                          const MDWFSpinor<double, true, All, HaloDepth, 1> &mode, size_t column, bool gamma5,
                          bool imag)
        : _store(store.getAccessor()), _mode(mode.getAccessor()), _column(column), _gamma5(gamma5), _imag(imag) {}

    __host__ __device__ double operator()(gSite site) {
        typedef GIndexer<All, HaloDepth> GInd;
        const Vect12<double> e = _store.getElement(GInd::getSiteStack(site, _column));
        const Vect12<double> p = _mode.getElement(GInd::getSiteStack(site, 0));
        COMPLEX(double) z(0.0, 0.0);
        for (int k = 0; k < 12; k++) {
            const double sign = (_gamma5 && k >= 6) ? -1.0 : 1.0;
            z += sign * conj(e.data[k]) * p.data[k];
        }
        return _imag ? imag(z) : real(z);
    }
};

// ---------------------------------------------------------------------------------------------------------------
// Low modes
// ---------------------------------------------------------------------------------------------------------------

struct MDWFLowModeInfo {
    double mu = 0.0;          // eigenvalue of H = g5 S^-1
    double chirality = 0.0;   // <psi|g5|psi>
    double lambda = 0.0;      // |mu| sqrt(1 - chi^2)
    double mresMode = 0.0;    // mu chi - m (GPT's m_res_mode)
    double residual = 0.0;    // |A psi - psi/mu| / |1/mu|
};

struct MDWFLowModeParameters {
    int nModes = 16;
    double precision = 1e-10;   // solver precision inside the eigensolver and the checks
    double tol = 1e-8;          // Lanczos tolerance (relative to the largest Ritz value of -A^2)
    int nBasis = 0;             // 0: 2 nModes + 8
    int nKeep = 0;              // 0: nModes + 4
    int maxRestarts = 50;
    int maxIter = 100000;
};

/*
 * Holds the N lowest modes of H for one configuration and the light-quark measurement object used to apply S.
 */
template<size_t HaloDepth, size_t Ls, class Solver>
class MDWFLowModes {
public:
    using Spinor4 = MDWFSpinor<double, true, All, HaloDepth, 1>;
    using Store = MDWFSpinor<double, true, All, HaloDepth, Ls>;
    using Meas = MDWFResidualMassMeasurement<HaloDepth, Ls, Solver>;

private:
    Meas &_meas;
    Store &_store;
    MDWFLowModeParameters _p;
    double _m;
    long _solves = 0;
    long _iterations = 0;
    LatticeContainer<true, double> _reduction;

public:
    std::vector<std::unique_ptr<Spinor4>> psi;   // modes, ordered by |mu| ascending
    std::vector<MDWFLowModeInfo> info;

    MDWFLowModes(Meas &meas, Store &store, double mass, const MDWFLowModeParameters &p)
        : _meas(meas), _store(store), _p(p), _m(mass), _reduction(store.getComm()) {
        typedef GIndexer<All, HaloDepth> GInd;
        _reduction.adjustSize(GInd::getLatData().vol4);
        static_assert(Ls >= 4, "the store needs four stacks");
        if (_p.nBasis <= 0) {
            _p.nBasis = 2 * _p.nModes + 8;
        }
        if (_p.nKeep <= 0) {
            _p.nKeep = _p.nModes + 4;
        }
    }

    long solves() const { return _solves; }
    long iterations() const { return _iterations; }
    int size() const { return static_cast<int>(psi.size()); }
    LatticeContainer<true, double> &reduction() { return _reduction; }

    // out = A in = S g5 in (one solve; uses store columns 0 and 1).
    void applyA(Spinor4 &out, const Spinor4 &in) {
        _store.template iterateOverBulk<BLOCKSIZE>(MDWFStoreColumnFrom4D<HaloDepth, Ls>(_store, in, 0, true));
        _meas.source5D().template iterateOverBulk<BLOCKSIZE>(MDWFColumnSource<HaloDepth, Ls>(_store, 0));
        _meas.solveSource(_p.maxIter, _p.precision);
        _solves++;
        _iterations += _meas.solver().lastIterations();
        _store.template iterateOverBulk<BLOCKSIZE>(MDWFExportColumn<HaloDepth, Ls>(_store, _meas.solution5D(), 1));
        out.template iterateOverBulk<BLOCKSIZE>(MDWF4DFromStoreColumn<HaloDepth, Ls>(_store, 1));
    }

    // Op = -A^2 for the eigensolver.
    struct MinusASquared {
        MDWFLowModes &lm;
        Spinor4 tmp;
        explicit MinusASquared(MDWFLowModes &l, CommunicationBase &comm) : lm(l), tmp(comm, "MDWF_lma_op_tmp") {}
        void apply(Spinor4 &out, const Spinor4 &in) {
            lm.applyA(tmp, in);
            lm.applyA(out, tmp);
            out *= COMPLEX(double)(-1.0, 0.0);
        }
    };

    void compute(uint64_t seed) {
        CommunicationBase &comm = _store.getComm();
        // Start vector: Z2 hash noise (deterministic per configuration through the seed).
        Spinor4 start(comm, "MDWF_lma_start");
        _store.template iterateOverBulk<BLOCKSIZE>(MDWFNoiseColumn<HaloDepth, Ls>(_store, 2, seed, 0));
        start.template iterateOverBulk<BLOCKSIZE>(MDWF4DFromStoreColumn<HaloDepth, Ls>(_store, 2));

        MDWFThickRestartParameters tp{_p.nModes, _p.nBasis, _p.nKeep, _p.tol, _p.maxRestarts};
        std::vector<std::unique_ptr<Spinor4>> basis;
        MinusASquared op(*this, comm);
        const MDWFThickRestartResult tr = mdwfThickRestartLanczos<Spinor4>(op, start, tp, basis, "MDWF_lma_trl");
        rootLogger.info("MDWF LMA: Lanczos on -A^2: ", tr.applications, " applications, ", tr.restarts,
                        " restarts, converged ", tr.converged, ", ", _solves, " solves so far");
        if (!tr.converged) {
            rootLogger.warn("MDWF LMA: Lanczos not converged; the Rayleigh-Ritz residuals below are the real test");
        }
        rayleighRitz(basis);
    }

private:
    // Hermitian n x n eigenproblem through the real symmetric 2n x 2n embedding [[Re, -Im], [Im, Re]].
    static void hermitianEigen(const std::vector<COMPLEX(double)> &h, int n, std::vector<double> &evals,
                               std::vector<std::vector<COMPLEX(double)>> &evecs) {
        const int N2 = 2 * n;
        std::vector<double> a(static_cast<size_t>(N2) * N2, 0.0);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                const COMPLEX(double) z = 0.5 * (h[i * n + j] + conj(h[j * n + i]));
                a[i * N2 + j] = real(z);
                a[(i + n) * N2 + (j + n)] = real(z);
                a[i * N2 + (j + n)] = -imag(z);
                a[(i + n) * N2 + j] = imag(z);
            }
        }
        std::vector<double> ev, v;
        mdwfSymmetricEigen(a, N2, ev, v);
        evals.clear();
        evecs.clear();
        // Every complex eigenpair appears twice (z and i z); keep a vector only if it is new (Gram-Schmidt).
        for (int l = 0; l < N2 && static_cast<int>(evals.size()) < n; l++) {
            std::vector<COMPLEX(double)> z(n);
            for (int i = 0; i < n; i++) {
                z[i] = COMPLEX(double)(v[i * N2 + l], v[(i + n) * N2 + l]);
            }
            for (int pass = 0; pass < 2; pass++) {
                for (const auto &q : evecs) {
                    COMPLEX(double) c(0.0, 0.0);
                    for (int i = 0; i < n; i++) {
                        c += conj(q[i]) * z[i];
                    }
                    for (int i = 0; i < n; i++) {
                        z[i] -= c * q[i];
                    }
                }
            }
            double nrm = 0.0;
            for (int i = 0; i < n; i++) {
                nrm += real(conj(z[i]) * z[i]);
            }
            if (nrm > 0.25) {
                const double s = 1.0 / std::sqrt(nrm);
                for (int i = 0; i < n; i++) {
                    z[i] *= s;
                }
                evals.push_back(ev[l]);
                evecs.push_back(z);
            }
        }
        if (static_cast<int>(evals.size()) != n) {
            throw std::runtime_error(stdLogger.fatal("MDWF LMA: Hermitian eigensolver found ", evals.size(), " of ",
                                                     n, " eigenvectors"));
        }
    }

    void rayleighRitz(std::vector<std::unique_ptr<Spinor4>> &ritz) {
        CommunicationBase &comm = _store.getComm();
        const int N = _p.nModes;
        std::vector<std::unique_ptr<Spinor4>> cand;
        for (int i = 0; i < N; i++) {
            cand.emplace_back(new Spinor4(comm, "MDWF_lma_c" + std::to_string(i)));
            *cand.back() = *ritz[i];
        }
        for (int i = 0; i < N; i++) {
            cand.emplace_back(new Spinor4(comm, "MDWF_lma_c" + std::to_string(N + i)));
            applyA(*cand.back(), *ritz[i]);
        }
        ritz.clear();   // free the Lanczos basis
        // Orthonormal basis of span{v_i, A v_i}.
        std::vector<std::unique_ptr<Spinor4>> b;
        for (auto &c : cand) {
            const double n0 = std::sqrt(real<double>(mdwfDot4(*c, *c)));
            for (int pass = 0; pass < 2; pass++) {
                for (auto &q : b) {
                    const COMPLEX(double) z = mdwfDot4(*q, *c);
                    c->axpyThisB(COMPLEX(double)(-real(z), -imag(z)), *q);
                }
            }
            const double n1 = std::sqrt(real<double>(mdwfDot4(*c, *c)));
            if (n1 > 1e-10 * n0) {
                *c *= COMPLEX(double)(1.0 / n1, 0.0);
                b.push_back(std::move(c));
            }
        }
        cand.clear();
        const int n = static_cast<int>(b.size());
        std::vector<std::unique_ptr<Spinor4>> Ab;
        for (int j = 0; j < n; j++) {
            Ab.emplace_back(new Spinor4(comm, "MDWF_lma_ab" + std::to_string(j)));
            applyA(*Ab.back(), *b[j]);
        }
        std::vector<COMPLEX(double)> h(static_cast<size_t>(n) * n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                h[i * n + j] = mdwfDot4(*b[i], *Ab[j]);
            }
        }
        std::vector<double> a;
        std::vector<std::vector<COMPLEX(double)>> U;
        hermitianEigen(h, n, a, U);
        std::vector<int> order(n);
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&](int x, int y) { return std::abs(a[x]) > std::abs(a[y]); });

        psi.clear();
        info.clear();
        Spinor4 apsi(comm, "MDWF_lma_apsi"), g5psi(comm, "MDWF_lma_g5psi");
        for (int r = 0; r < N && r < n; r++) {
            const int k = order[r];
            psi.emplace_back(new Spinor4(comm, "MDWF_lma_psi" + std::to_string(r)));
            Spinor4 &p = *psi.back();
            p = *b[0];
            p *= U[k][0];
            apsi = *Ab[0];
            apsi *= U[k][0];
            for (int i = 1; i < n; i++) {
                p.axpyThisB(U[k][i], *b[i]);
                apsi.axpyThisB(U[k][i], *Ab[i]);
            }
            const double nrm = std::sqrt(real<double>(mdwfDot4(p, p)));
            p *= COMPLEX(double)(1.0 / nrm, 0.0);
            apsi *= COMPLEX(double)(1.0 / nrm, 0.0);
            MDWFLowModeInfo in;
            in.mu = 1.0 / a[k];
            apsi.template axpyThisB<64>(-a[k], p);
            in.residual = std::sqrt(real<double>(mdwfDot4(apsi, apsi))) / std::abs(a[k]);
            applyMDWFGamma5<double, true, All, HaloDepth, 1>(g5psi, p, false);
            in.chirality = real<double>(mdwfDot4(p, g5psi));
            in.lambda = std::abs(in.mu) * std::sqrt(std::max(0.0, 1.0 - in.chirality * in.chirality));
            in.mresMode = in.mu * in.chirality - _m;
            info.push_back(in);
        }
        for (size_t r = 0; r < info.size(); r++) {
            rootLogger.info("MDWF LMA: mode ", r, ": mu = ", info[r].mu, ", chirality = ", info[r].chirality,
                            ", lambda = ", info[r].lambda, ", m_res_mode = ", info[r].mresMode, ", residual = ",
                            info[r].residual);
        }
    }

public:
    // Spin-density slices D_kl[16][n] along mu for all pairs (k <= l computed, the rest by Hermiticity).
    // Returns d[(k * N + l) * 16 + 4 s + sp][n].
    std::vector<std::vector<COMPLEX(double)>> spinDensities(int mu, bool diagonalOnly) {
        const int N = size();
        std::vector<std::vector<COMPLEX(double)>> d(static_cast<size_t>(N) * N * 16);
        for (int k = 0; k < N; k++) {
            for (int l = (diagonalOnly ? k : k); l < (diagonalOnly ? k + 1 : N); l++) {
                for (int s = 0; s < 4; s++) {
                    for (int sp = 0; sp < 4; sp++) {
                        const std::vector<double> re = mdwfSliceSums<HaloDepth>(
                            _reduction, MDWFPairSpinDensity<HaloDepth>(*psi[k], *psi[l], s, sp, false), mu);
                        const std::vector<double> im = mdwfSliceSums<HaloDepth>(
                            _reduction, MDWFPairSpinDensity<HaloDepth>(*psi[k], *psi[l], s, sp, true), mu);
                        auto &kl = d[(static_cast<size_t>(k) * N + l) * 16 + 4 * s + sp];
                        kl.resize(re.size());
                        for (size_t n = 0; n < re.size(); n++) {
                            kl[n] = COMPLEX(double)(re[n], im[n]);
                        }
                        if (l != k) {   // D_lk[sp][s] = conj(D_kl[s][sp])
                            auto &lk = d[(static_cast<size_t>(l) * N + k) * 16 + 4 * sp + s];
                            lk.resize(re.size());
                            for (size_t n = 0; n < re.size(); n++) {
                                lk[n] = COMPLEX(double)(re[n], -im[n]);
                            }
                        }
                    }
                }
            }
        }
        return d;
    }

    // psi_k at the global site x0 as 12 complex numbers.
    std::vector<std::array<COMPLEX(double), 12>> valuesAt(const std::array<int, 4> &x0) {
        std::vector<std::array<COMPLEX(double), 12>> out(size());
        for (int k = 0; k < size(); k++) {
            for (int c = 0; c < 12; c++) {
                double part[2];
                for (int im = 0; im < 2; im++) {
                    const std::vector<double> s = mdwfSliceSums<HaloDepth>(
                        _reduction, MDWFSiteComponent<HaloDepth>(*psi[k], x0, c, im == 1), 3);
                    part[im] = 0.0;
                    for (double v : s) {
                        part[im] += v;
                    }
                }
                out[k][c] = COMPLEX(double)(part[0], part[1]);
            }
        }
        return out;
    }
};

// B^M_kl(n) = sum_{ss'} M[s][s'] D_kl[s][s'](n).
inline std::vector<COMPLEX(double)> mdwfLmaBilinear(const std::vector<std::vector<COMPLEX(double)>> &d, int N, int k,
                                                    int l, const MDWFSpinMatrix &M) {
    const size_t base = (static_cast<size_t>(k) * N + l) * 16;
    const size_t len = d[base].size();
    std::vector<COMPLEX(double)> b(len, COMPLEX(double)(0.0, 0.0));
    for (int e = 0; e < 16; e++) {
        if (real(M[e]) == 0.0 && imag(M[e]) == 0.0) {
            continue;
        }
        for (size_t n = 0; n < len; n++) {
            b[n] += M[e] * d[base + e][n];
        }
    }
    return b;
}

// phi^+ M chi for spin-colour vectors (M acts on spin).
inline COMPLEX(double) mdwfLmaSiteBilinear(const std::array<COMPLEX(double), 12> &phi, const MDWFSpinMatrix &M,
                                           const std::array<COMPLEX(double), 12> &chi) {
    COMPLEX(double) z(0.0, 0.0);
    for (int s = 0; s < 4; s++) {
        for (int sp = 0; sp < 4; sp++) {
            if (real(M[4 * s + sp]) == 0.0 && imag(M[4 * s + sp]) == 0.0) {
                continue;
            }
            for (int c = 0; c < 3; c++) {
                z += conj(phi[3 * s + c]) * M[4 * s + sp] * chi[3 * sp + c];
            }
        }
    }
    return z;
}

struct MDWFLmaChannelResult {
    std::vector<int> nSub;                                  // mode counts (e.g. N/4, N/2, N)
    std::vector<std::vector<double>> pointLL, a2aLL;        // [sub][n]
    double maxImag = 0.0;                                   // largest |imaginary part| relative to max |real|
};

/*
 * Point and all-to-all LL correlators of channel G along the direction of the density d (mdwfLmaBilinear layout),
 * for the first nSub modes. V4 = global volume.
 */
inline MDWFLmaChannelResult mdwfLmaChannel(const std::vector<std::vector<COMPLEX(double)>> &d, int N,
                                           const std::vector<double> &mu,
                                           const std::vector<std::array<COMPLEX(double), 12>> &psiAtSource,
                                           const std::string &channel, double V4, const std::vector<int> &nSub) {
    const MDWFSpinMatrix g5 = mdwfGammaMatrix("5");
    const MDWFSpinMatrix G = mdwfGammaChannel(channel);
    const MDWFSpinMatrix L = mdwfSpinProduct(g5, G);
    const MDWFSpinMatrix R = mdwfSpinProduct(mdwfSpinDagger(G), g5);
    const MDWFSpinMatrix Rs = mdwfSpinProduct(g5, mdwfSpinDagger(G));   // g5 R g5
    // phi_k = g5 psi_k(x0) / mu_k
    std::vector<std::array<COMPLEX(double), 12>> phi(N);
    for (int k = 0; k < N; k++) {
        for (int c = 0; c < 12; c++) {
            phi[k][c] = psiAtSource[k][c] * ((c >= 6 ? -1.0 : 1.0) / mu[k]);
        }
    }
    const size_t Ln = d[0].size();
    MDWFLmaChannelResult res;
    res.nSub = nSub;
    std::vector<std::vector<COMPLEX(double)>> BL(static_cast<size_t>(N) * N), BR(static_cast<size_t>(N) * N);
    for (int k = 0; k < N; k++) {
        for (int l = 0; l < N; l++) {
            BL[k * N + l] = mdwfLmaBilinear(d, N, k, l, L);
            BR[k * N + l] = mdwfLmaBilinear(d, N, k, l, Rs);
        }
    }
    double maxRe = 0.0, maxIm = 0.0;
    for (int ns : nSub) {
        std::vector<COMPLEX(double)> pt(Ln, COMPLEX(double)(0.0, 0.0)), aa(Ln, COMPLEX(double)(0.0, 0.0));
        for (int k = 0; k < ns; k++) {
            for (int l = 0; l < ns; l++) {
                const COMPLEX(double) blk = mdwfLmaSiteBilinear(phi[l], R, phi[k]);
                const auto &bl = BL[k * N + l];
                const auto &br = BR[l * N + k];
                const double w = 1.0 / (mu[k] * mu[l] * V4);
                for (size_t n = 0; n < Ln; n++) {
                    pt[n] += bl[n] * blk;
                }
                for (size_t dn = 0; dn < Ln; dn++) {
                    COMPLEX(double) s(0.0, 0.0);
                    for (size_t n0 = 0; n0 < Ln; n0++) {
                        s += bl[(n0 + dn) % Ln] * br[n0];
                    }
                    aa[dn] += s * w;
                }
            }
        }
        std::vector<double> ptr(Ln), aar(Ln);
        for (size_t n = 0; n < Ln; n++) {
            ptr[n] = real(pt[n]);
            aar[n] = real(aa[n]);
            maxRe = std::max({maxRe, std::abs(ptr[n]), std::abs(aar[n])});
            maxIm = std::max({maxIm, std::abs(imag(pt[n])), std::abs(imag(aa[n]))});
        }
        res.pointLL.push_back(ptr);
        res.a2aLL.push_back(aar);
    }
    res.maxImag = maxIm / std::max(maxRe, 1e-300);
    return res;
}

/*
 * Self-test L2: the point LL of the given channels, built explicitly. For each colour the four spin columns of
 * S_L(x, x0) = sum_k psi_k(x) phi_k^+ go into store stacks 0..3 and are contracted with the production kernel.
 */
template<size_t HaloDepth, size_t Ls, class Solver>
std::vector<std::vector<double>> mdwfLmaExplicitPointLL(MDWFLowModes<HaloDepth, Ls, Solver> &lm,
                                                        MDWFSpinor<double, true, All, HaloDepth, Ls> &store,
                                                        const std::vector<std::array<COMPLEX(double), 12>> &psiAtSource,
                                                        const std::vector<std::string> &channels, int mu) {
    const int N = lm.size();
    const MDWFSpinMatrix g5 = mdwfGammaMatrix("5");
    std::vector<std::vector<double>> out(channels.size());
    for (int colour = 0; colour < 3; colour++) {
        for (int spin = 0; spin < 4; spin++) {
            const int a = 3 * spin + colour;
            for (int k = 0; k < N; k++) {
                // column (spin, colour) of S_L(x, x0): sum_k psi_k(x) conj(phi_k[a]), phi_k = g5 psi_k(x0) / mu_k
                const COMPLEX(double) phi = psiAtSource[k][a] * ((a >= 6 ? -1.0 : 1.0) / lm.info[k].mu);
                store.template iterateOverBulk<BLOCKSIZE>(
                    MDWFColumnAxpy<HaloDepth, Ls>(store, *lm.psi[k], spin, conj(phi), k == 0));
            }
        }
        for (size_t ch = 0; ch < channels.size(); ch++) {
            const MDWFSpinMatrix G = mdwfGammaChannel(channels[ch]);
            const std::vector<double> s = mdwfSliceSums<HaloDepth>(
                lm.reduction(), MDWFMesonChannelDensity<HaloDepth, Ls>(store, mdwfSpinProduct(g5, G),
                                                                       mdwfSpinProduct(mdwfSpinDagger(G), g5)), mu);
            out[ch].resize(s.size(), 0.0);
            for (size_t n = 0; n < s.size(); n++) {
                out[ch][n] += s[n];
            }
        }
    }
    return out;
}

/*
 * Low-mode parts of the loops of one noise vector (eta in store column `etaColumn`), along mu, for Gamma = 1 (g5 =
 * false) and Gamma = g5: lowStoch[Gamma][n] = sum_k (sum_{x_mu = n} eta^+ Gamma psi_k) c_k / mu_k, c_k = psi_k^+ g5 eta.
 * c is computed on the first call (cOut empty) and reused for the other directions.
 */
template<size_t HaloDepth, size_t Ls, class Solver>
std::array<std::vector<COMPLEX(double)>, 2> mdwfLmaNoiseLowPart(MDWFLowModes<HaloDepth, Ls, Solver> &lm,
                                                                MDWFSpinor<double, true, All, HaloDepth, Ls> &store,
                                                                size_t etaColumn, int mu,
                                                                std::vector<COMPLEX(double)> &cOut) {
    const int N = lm.size();
    std::array<std::vector<COMPLEX(double)>, 2> low;
    std::vector<std::array<std::vector<COMPLEX(double)>, 2>> s(N);
    for (int k = 0; k < N; k++) {
        for (int g = 0; g < 2; g++) {
            const std::vector<double> re = mdwfSliceSums<HaloDepth>(
                lm.reduction(), MDWFColumnModeDensity<HaloDepth, Ls>(store, *lm.psi[k], etaColumn, g == 1, false), mu);
            const std::vector<double> im = mdwfSliceSums<HaloDepth>(
                lm.reduction(), MDWFColumnModeDensity<HaloDepth, Ls>(store, *lm.psi[k], etaColumn, g == 1, true), mu);
            s[k][g].resize(re.size());
            for (size_t n = 0; n < re.size(); n++) {
                s[k][g][n] = COMPLEX(double)(re[n], im[n]);
            }
        }
    }
    if (cOut.empty()) {
        cOut.resize(N);
        for (int k = 0; k < N; k++) {   // c_k = psi_k^+ g5 eta = conj(eta^+ g5 psi_k)
            COMPLEX(double) t(0.0, 0.0);
            for (const auto &z : s[k][1]) {
                t += z;
            }
            cOut[k] = conj(t);
        }
    }
    const size_t Ln = s[0][0].size();
    for (int g = 0; g < 2; g++) {
        low[g].assign(Ln, COMPLEX(double)(0.0, 0.0));
        for (int k = 0; k < N; k++) {
            const COMPLEX(double) w = cOut[k] / lm.info[k].mu;
            for (size_t n = 0; n < Ln; n++) {
                low[g][n] += s[k][g][n] * w;
            }
        }
    }
    return low;
}

// Exact low-mode loops along mu: [Gamma = 1, Gamma = g5][n] = sum_k B^{g5 Gamma}_kk(n) / mu_k (diagonal densities).
inline std::array<std::vector<COMPLEX(double)>, 2> mdwfLmaExactLoops(const std::vector<std::vector<COMPLEX(double)>> &d,
                                                                     int N, const std::vector<double> &mu) {
    const MDWFSpinMatrix g5 = mdwfGammaMatrix("5"), one = mdwfSpinIdentity();
    std::array<std::vector<COMPLEX(double)>, 2> e;
    for (int g = 0; g < 2; g++) {
        for (int k = 0; k < N; k++) {
            const std::vector<COMPLEX(double)> b = mdwfLmaBilinear(d, N, k, k, g == 0 ? g5 : one);
            if (e[g].empty()) {
                e[g].assign(b.size(), COMPLEX(double)(0.0, 0.0));
            }
            for (size_t n = 0; n < b.size(); n++) {
                e[g][n] += b[n] / mu[k];
            }
        }
    }
    return e;
}
