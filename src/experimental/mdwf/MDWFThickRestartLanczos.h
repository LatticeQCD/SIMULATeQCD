/*
 * Low eigenmodes (eigenvalues and eigenvectors) of a Hermitian operator A on 4D Wilson spinors (MDWFSpinor with
 * Ls = 1), by thick-restart Lanczos (K. Wu, H. Simon, SIAM J. Matrix Anal. Appl. 22 (2000) 602) with optional
 * Chebyshev filtering.
 *
 *   mdwfThickRestartLanczos   the nev smallest eigenvalues of a Hermitian operator Op (Op.apply(out, in)), with full
 *                             reorthogonalization (two classical Gram-Schmidt passes) and thick restarts: after nBasis
 *                             Lanczos vectors the nKeep smallest Ritz vectors and the last Lanczos vector are kept and
 *                             the projected matrix becomes arrowhead + tridiagonal. Stops when the nev smallest Ritz
 *                             pairs have residual |Op y - theta y| <= tol * max_l |theta_l| (estimated from the last
 *                             Lanczos coefficient), or after maxRestarts restarts.
 *   MDWFChebyshevOperator     Op = -T_d(L(A)), L(A) = (hi + lo - 2 A) / (hi - lo): eigenvalues of A in [lo, hi] are
 *                             mapped into [-1, 1] (|T_d| <= 1), those below lo to L > 1 where T_d = cosh(d acosh L) is
 *                             large and positive for every d, so the smallest eigenvalues of Op are the smallest of A
 *                             (same eigenvectors, same order) and are well separated. hi must be at
 *                             least lambda_max(A); lo sits above the wanted part of the spectrum. degree 0 gives Op = A.
 *   mdwfSymmetricEigen        dense real symmetric eigensolver (cyclic Jacobi) for the projected matrix.
 *
 * Final eigenvalues of A itself are obtained by the caller as Rayleigh quotients <y|A|y> of the returned vectors.
 * Memory: nBasis + 1 + nKeep 4D vectors. Single rank or multi rank as far as dotProduct goes.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// <a|b> of two unstacked spinor fields. Spinorfield::dotProduct passes elems as NStacks to
// LatticeContainer::iterateOverBulkStacked (cudaErrorInvalidConfiguration at launch); dotProductStacked is correct.
template<class Spinor>
COMPLEX(double) mdwfDot4(Spinor &a, const Spinor &b) {
    return a.dotProductStacked(b)[0];
}

// Eigenvalues (ascending) and eigenvectors of the real symmetric n x n matrix a (row-major). evecs is row-major with
// the eigenvectors in its columns: evecs[i * n + l] = component i of eigenvector l.
inline void mdwfSymmetricEigen(std::vector<double> a, int n, std::vector<double> &evals, std::vector<double> &evecs) {
    std::vector<double> v(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; i++) {
        v[i * n + i] = 1.0;
    }
    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0.0, total = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                total += a[i * n + j] * a[i * n + j];
                if (i != j) {
                    off += a[i * n + j] * a[i * n + j];
                }
            }
        }
        if (off <= 1e-30 * total) {
            break;
        }
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                const double apq = a[p * n + q];
                if (apq == 0.0) {
                    continue;
                }
                const double theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                const double t = std::abs(theta) > 1e150 ? 0.5 / theta   // theta^2 would overflow
                                 : (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < n; k++) {   // columns p, q: A J
                    const double akp = a[k * n + p], akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; k++) {   // rows p, q: J^T A
                    const double apk = a[p * n + k], aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; k++) {   // V J
                    const double vkp = v[k * n + p], vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int x, int y) { return a[x * n + x] < a[y * n + y]; });
    evals.assign(n, 0.0);
    evecs.assign(static_cast<size_t>(n) * n, 0.0);
    for (int l = 0; l < n; l++) {
        evals[l] = a[order[l] * n + order[l]];
        for (int i = 0; i < n; i++) {
            evecs[i * n + l] = v[i * n + order[l]];
        }
    }
}

struct MDWFThickRestartParameters {
    int nev;           // wanted eigenpairs (smallest)
    int nBasis;        // Lanczos vectors before a restart (> nKeep)
    int nKeep;         // Ritz vectors kept at a restart (>= nev)
    double tol;        // residual tolerance relative to the largest Ritz value in magnitude
    int maxRestarts;
};

struct MDWFThickRestartResult {
    std::vector<double> ritz;       // nev smallest Ritz values of Op, ascending
    std::vector<double> residual;   // their estimated residuals |Op y - theta y|
    int restarts = 0;
    int applications = 0;           // applications of Op
    bool converged = false;
};

// basis is (re)allocated; on return basis[0 .. nev-1] hold the Ritz vectors of the nev smallest Ritz values.
template<class Spinor, class Op>
MDWFThickRestartResult mdwfThickRestartLanczos(Op &op, const Spinor &start, const MDWFThickRestartParameters &p,
                                               std::vector<std::unique_ptr<Spinor>> &basis, const std::string &name) {
    const int m = p.nBasis;
    if (!(p.nev > 0 && p.nKeep >= p.nev && p.nKeep < m - 1)) {
        throw std::runtime_error(stdLogger.fatal("thick-restart Lanczos needs 0 < nev <= nKeep < nBasis - 1, got nev = ",
                                                 p.nev, ", nKeep = ", p.nKeep, ", nBasis = ", m));
    }
    basis.clear();
    for (int i = 0; i <= m; i++) {
        basis.emplace_back(new Spinor(start.getComm(), name + "_v" + std::to_string(i)));
    }
    std::vector<std::unique_ptr<Spinor>> rotated;
    for (int i = 0; i < p.nKeep; i++) {
        rotated.emplace_back(new Spinor(start.getComm(), name + "_r" + std::to_string(i)));
    }
    Spinor w(start.getComm(), name + "_w");

    MDWFThickRestartResult result;
    *basis[0] = start;
    const double startNorm = std::sqrt(real<double>(mdwfDot4(*basis[0], *basis[0])));
    *basis[0] *= COMPLEX(double)(1.0 / startNorm, 0.0);

    int k = 0;                                  // kept Ritz vectors at the start of this cycle
    std::vector<double> theta, s;               // kept Ritz values and their couplings to basis[k]
    std::vector<double> alpha(m, 0.0), beta(m, 0.0);
    std::vector<double> evals, y;

    // basis[0 .. count-1] <- basis[0 .. m-1] Y[:, 0 .. count-1]
    auto rotate = [&](int count) {
        for (int i = 0; i < count; i++) {
            *rotated[i] = *basis[0];
            *rotated[i] *= COMPLEX(double)(y[0 * m + i], 0.0);
            for (int j = 1; j < m; j++) {
                rotated[i]->template axpyThisB<64>(y[j * m + i], *basis[j]);
            }
        }
        for (int i = 0; i < count; i++) {
            *basis[i] = *rotated[i];
        }
    };

    for (int restart = 0; ; restart++) {
        for (int j = k; j < m; j++) {
            basis[j]->updateAll();
            op.apply(w, *basis[j]);
            result.applications++;
            const double a = real<double>(mdwfDot4(*basis[j], w));
            w.template axpyThisB<64>(-a, *basis[j]);
            if (j == k) {
                for (int i = 0; i < k; i++) {
                    w.template axpyThisB<64>(-s[i], *basis[i]);
                }
            } else {
                w.template axpyThisB<64>(-beta[j - 1], *basis[j - 1]);
            }
            for (int pass = 0; pass < 2; pass++) {   // full reorthogonalization; coefficients are rounding only
                for (int i = 0; i <= j; i++) {
                    const COMPLEX(double) c = mdwfDot4(*basis[i], w);
                    w.axpyThisB(COMPLEX(double)(-real(c), -imag(c)), *basis[i]);
                }
            }
            const double b = std::sqrt(real<double>(mdwfDot4(w, w)));
            if (!(b > 1e-300)) {
                throw std::runtime_error(stdLogger.fatal("thick-restart Lanczos breakdown at step ", j,
                                                         " (invariant subspace); use another start vector"));
            }
            alpha[j] = a;
            beta[j] = b;
            *basis[j + 1] = w;
            *basis[j + 1] *= COMPLEX(double)(1.0 / b, 0.0);
        }

        std::vector<double> t(static_cast<size_t>(m) * m, 0.0);
        for (int i = 0; i < k; i++) {
            t[i * m + i] = theta[i];
            t[i * m + k] = s[i];
            t[k * m + i] = s[i];
        }
        for (int j = k; j < m; j++) {
            t[j * m + j] = alpha[j];
            if (j + 1 < m) {
                t[j * m + j + 1] = beta[j];
                t[(j + 1) * m + j] = beta[j];
            }
        }
        mdwfSymmetricEigen(t, m, evals, y);
        const double last = beta[m - 1];
        double scale = 0.0;
        for (double e : evals) {
            scale = std::max(scale, std::abs(e));
        }
        int nconv = 0;
        while (nconv < p.nev && std::abs(last * y[(m - 1) * m + nconv]) <= p.tol * scale) {
            nconv++;
        }
        rootLogger.info("thick-restart Lanczos ", name, ": cycle ", restart, ", ", result.applications,
                        " applications, ", nconv, " of ", p.nev, " converged; Ritz values ", evals[0], " ... ",
                        evals[p.nev - 1], ", residual of the last wanted one ",
                        std::abs(last * y[(m - 1) * m + p.nev - 1]), " (tolerance ", p.tol * scale, ")");
        if (nconv >= p.nev || restart >= p.maxRestarts) {
            result.converged = nconv >= p.nev;
            result.restarts = restart;
            result.ritz.assign(evals.begin(), evals.begin() + p.nev);
            result.residual.resize(p.nev);
            for (int l = 0; l < p.nev; l++) {
                result.residual[l] = std::abs(last * y[(m - 1) * m + l]);
            }
            rotate(p.nev);
            return result;
        }
        // Thick restart: keep the nKeep smallest Ritz pairs and the last Lanczos vector.
        const int kk = p.nKeep;
        rotate(kk);
        *basis[kk] = *basis[m];
        theta.assign(evals.begin(), evals.begin() + kk);
        s.resize(kk);
        for (int i = 0; i < kk; i++) {
            s[i] = last * y[(m - 1) * m + i];
        }
        k = kk;
    }
}

// Op = -T_d((hi + lo - 2 A) / (hi - lo)), or Op = A for degree 0.
template<class Spinor, class OpA>
class MDWFChebyshevOperator {
    OpA &_a;
    double _scale, _shift;
    int _degree;
    Spinor _y0, _y1, _y2, _t;

public:
    MDWFChebyshevOperator(OpA &a, double lo, double hi, int degree, CommunicationBase &comm, const std::string &name)
        : _a(a), _scale(-2.0 / (hi - lo)), _shift((hi + lo) / (hi - lo)), _degree(degree),
          _y0(comm, name + "_y0"), _y1(comm, name + "_y1"), _y2(comm, name + "_y2"), _t(comm, name + "_t") {
        if (degree > 0 && !(hi > lo)) {
            throw std::runtime_error(stdLogger.fatal("Chebyshev filter needs hi > lo, got lo = ", lo, ", hi = ", hi));
        }
    }

    int applicationsPerCall() const {
        return _degree == 0 ? 1 : _degree;
    }

    void apply(Spinor &out, const Spinor &in) {
        if (_degree == 0) {
            _a.apply(out, in);
            return;
        }
        Spinor *p0 = &_y0, *p1 = &_y1, *p2 = &_y2;
        *p0 = in;
        p0->updateAll();
        _a.apply(_t, *p0);                       // y1 = L y0
        *p1 = _t;
        *p1 *= COMPLEX(double)(_scale, 0.0);
        p1->template axpyThisB<64>(_shift, *p0);
        for (int d = 2; d <= _degree; d++) {    // y2 = 2 L y1 - y0
            p1->updateAll();
            _a.apply(_t, *p1);
            *p2 = _t;
            *p2 *= COMPLEX(double)(2.0 * _scale, 0.0);
            p2->template axpyThisB<64>(2.0 * _shift, *p1);
            p2->template axpyThisB<64>(-1.0, *p0);
            Spinor *old = p0;
            p0 = p1;
            p1 = p2;
            p2 = old;
        }
        out = *p1;
        out *= COMPLEX(double)(-1.0, 0.0);
    }
};
