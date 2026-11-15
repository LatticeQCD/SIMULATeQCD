/*
 * Deflation study for the light MDWF force solves (Step 0 of DSDRMDWFruns/SQCDruns/deflation/DESIGN.md): tools to
 * measure how fast the low-mode subspace of the operator of a force solve moves along the molecular dynamics and how
 * much of it simple tracking recovers. Measurement code only; nothing here is used by the HMC.
 *
 * Operator. Hasenbusch factor t of the even/odd ladder (MDWFEvenOddTwoFlavorRatioFermionAction) solves, in the force,
 *     light_form = 1 (GPT form, production):  (Mhat Mhat^+) chi = Mhat(m_t+1) phi,   Mhat = Mhat(m_t),
 *     light_form = 0:                         (Mhat^+ Mhat) chi = Mhat(m_t+1)^+ phi,
 * on the even sites, m_0 = mf, m_1, ... = hasenbusch_masses. MDWFDeflOperator is that operator A (Hermitian, positive
 * definite). Mhat Mhat^+ and Mhat^+ Mhat share the eigenvalues, not the eigenvectors.
 *
 * Vectors. A 5D even-site vector with halo (HaloDepth 2, 4 ranks 32 x 16 x 16 x 8) takes 2.6 times its bulk, so K = 64
 * of them would not fit next to a Lanczos basis. MDWFCompactVec stores the bulk only (sizeh x Ls x 12 complex doubles
 * per rank: 151 MB per rank at 32^3 x 8, Ls = 24) and has the vector operations the thick-restart Lanczos
 * (MDWFThickRestartLanczos.h, unchanged) and the solvers need; the operator unpacks into a halo field, applies A and
 * packs the result.
 *
 * Contents
 *   mdwfHermitianEigen, mdwfHermitianInverse, mdwfSingularValues   host dense algebra (complex Jacobi)
 *   MDWFCompactVec, mdwfDot4 (5D), mdwfDeflGram                   vectors
 *   MDWFDeflOperator                                              A on compact vectors
 *   mdwfDeflLambdaMax, mdwfDeflLowModes                           lowest K eigenpairs: Chebyshev-filtered thick-restart
 *                                                                 Lanczos, then Rayleigh-Ritz with A itself
 *   mdwfDeflRayleighRitz, mdwfDeflTrackStep                       Ritz pairs of A in a span, tracking variants T2 / T3
 *   MDWFDeflSubspace, mdwfDeflCG                                  deflated CG (Saad, Yeung, Erhel, Guyomarc'h,
 *                                                                 SIAM J. Sci. Comput. 21 (2000) 1909)
 *   mdwfDeflWriteSet, mdwfDeflReadSet                             eigenvector sets on disk (bulk, one file per rank)
 *   mdwfDeflCGSnapshots, mdwfDeflChebyshevSnapshots              refresh of a stale vector by a fixed number of
 *                                                                 applications of A (inverse iteration by CG, Chebyshev
 *                                                                 low-pass filter), with the iterate at several counts
 */

#pragma once

#include "MDWFSpinor.h"

#include "MDWFCoupledSolverAdapter.h"
#include "MDWFFermionBoundary.h"
#include "MDWFFieldDump.h"
#include "MDWFMobiusEvenOdd.h"
#include "MDWFThickRestartLanczos.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------------------------------------------
// Host dense algebra
// ---------------------------------------------------------------------------------------------------------------

typedef std::complex<double> MDWFDeflC;

// Eigenvalues (ascending) and eigenvectors of the Hermitian n x n matrix h (row-major; only its Hermitian part is
// used). evecs[i * n + l] = component i of eigenvector l. Cyclic complex Jacobi.
inline void mdwfHermitianEigen(std::vector<MDWFDeflC> h, int n, std::vector<double> &evals,
                               std::vector<MDWFDeflC> &evecs) {
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            const MDWFDeflC z = 0.5 * (h[i * n + j] + std::conj(h[j * n + i]));
            h[i * n + j] = z;
            h[j * n + i] = std::conj(z);
        }
    }
    std::vector<MDWFDeflC> v(static_cast<size_t>(n) * n, MDWFDeflC(0.0, 0.0));
    for (int i = 0; i < n; i++) {
        v[i * n + i] = 1.0;
    }
    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0.0, total = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                total += std::norm(h[i * n + j]);
                if (i != j) {
                    off += std::norm(h[i * n + j]);
                }
            }
        }
        if (off <= 1e-30 * total) {
            break;
        }
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                const double apq = std::abs(h[p * n + q]);
                if (apq == 0.0) {
                    continue;
                }
                // J = diag(1, e^{-i phi}) R, R = [[c, s], [-s, c]] in (p, q), e^{i phi} = h_pq / |h_pq|.
                const MDWFDeflC phase = h[p * n + q] / apq;
                const double theta = (std::real(h[q * n + q]) - std::real(h[p * n + p])) / (2.0 * apq);
                const double t = std::abs(theta) > 1e150 ? 0.5 / theta
                                 : (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                const MDWFDeflC jpp(c, 0.0), jpq(s, 0.0), jqp = -s * std::conj(phase), jqq = c * std::conj(phase);
                for (int k = 0; k < n; k++) {   // columns: H J
                    const MDWFDeflC hkp = h[k * n + p], hkq = h[k * n + q];
                    h[k * n + p] = hkp * jpp + hkq * jqp;
                    h[k * n + q] = hkp * jpq + hkq * jqq;
                }
                for (int k = 0; k < n; k++) {   // rows: J^+ H
                    const MDWFDeflC hpk = h[p * n + k], hqk = h[q * n + k];
                    h[p * n + k] = std::conj(jpp) * hpk + std::conj(jqp) * hqk;
                    h[q * n + k] = std::conj(jpq) * hpk + std::conj(jqq) * hqk;
                }
                for (int k = 0; k < n; k++) {   // V J
                    const MDWFDeflC vkp = v[k * n + p], vkq = v[k * n + q];
                    v[k * n + p] = vkp * jpp + vkq * jqp;
                    v[k * n + q] = vkp * jpq + vkq * jqq;
                }
            }
        }
    }
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(),
              [&](int x, int y) { return std::real(h[x * n + x]) < std::real(h[y * n + y]); });
    evals.assign(n, 0.0);
    evecs.assign(static_cast<size_t>(n) * n, MDWFDeflC(0.0, 0.0));
    for (int l = 0; l < n; l++) {
        evals[l] = std::real(h[order[l] * n + order[l]]);
        for (int i = 0; i < n; i++) {
            evecs[i * n + l] = v[i * n + order[l]];
        }
    }
}

// Inverse of a Hermitian positive definite matrix (row-major); throws if an eigenvalue is not positive.
inline std::vector<MDWFDeflC> mdwfHermitianInverse(const std::vector<MDWFDeflC> &g, int n) {
    std::vector<double> d;
    std::vector<MDWFDeflC> q;
    mdwfHermitianEigen(g, n, d, q);
    if (n > 0 && !(d[0] > 0.0)) {
        throw std::runtime_error(stdLogger.fatal("mdwfHermitianInverse: matrix not positive definite, lowest "
                                                 "eigenvalue ", d[0]));
    }
    std::vector<MDWFDeflC> inv(static_cast<size_t>(n) * n, MDWFDeflC(0.0, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            MDWFDeflC z(0.0, 0.0);
            for (int l = 0; l < n; l++) {
                z += q[i * n + l] * std::conj(q[j * n + l]) / d[l];
            }
            inv[i * n + j] = z;
        }
    }
    return inv;
}

// Singular values (descending) of the rows x cols matrix m (row-major), from the eigenvalues of m^+ m.
inline std::vector<double> mdwfSingularValues(const std::vector<MDWFDeflC> &m, int rows, int cols) {
    std::vector<MDWFDeflC> mm(static_cast<size_t>(cols) * cols, MDWFDeflC(0.0, 0.0));
    for (int a = 0; a < cols; a++) {
        for (int b = 0; b < cols; b++) {
            MDWFDeflC z(0.0, 0.0);
            for (int r = 0; r < rows; r++) {
                z += std::conj(m[r * cols + a]) * m[r * cols + b];
            }
            mm[a * cols + b] = z;
        }
    }
    std::vector<double> ev;
    std::vector<MDWFDeflC> q;
    mdwfHermitianEigen(mm, cols, ev, q);
    std::vector<double> sigma(cols);
    for (int a = 0; a < cols; a++) {
        sigma[a] = std::sqrt(std::max(ev[cols - 1 - a], 0.0));
    }
    return sigma;
}

// ---------------------------------------------------------------------------------------------------------------
// Compact vectors: the bulk of a 5D even-site spinor, 24 doubles (12 complex) per (even site, s), index
// i = isite + sizeh * s as GIndexer<Even>::getSiteStack.
// ---------------------------------------------------------------------------------------------------------------

struct MDWFFlatIndex {
    __device__ __host__ size_t operator()(const dim3 &blockDim, const uint3 &blockIdx, const uint3 &threadIdx) {
        return static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    }
};

struct MDWFCvScale {   // a *= s
    double *_a;
    double _sr, _si;
    __host__ __device__ void operator()(size_t i) {
        double *a = _a + 24 * i;
        for (int c = 0; c < 12; c++) {
            const double re = a[2 * c], im = a[2 * c + 1];
            a[2 * c] = _sr * re - _si * im;
            a[2 * c + 1] = _sr * im + _si * re;
        }
    }
};

struct MDWFCvAxpy {   // a += s x
    double *_a;
    const double *_x;
    double _sr, _si;
    __host__ __device__ void operator()(size_t i) {
        double *a = _a + 24 * i;
        const double *x = _x + 24 * i;
        for (int c = 0; c < 12; c++) {
            const double re = x[2 * c], im = x[2 * c + 1];
            a[2 * c] += _sr * re - _si * im;
            a[2 * c + 1] += _sr * im + _si * re;
        }
    }
};

struct MDWFCvXpay {   // a = x + s a (s real)
    double *_a;
    const double *_x;
    double _s;
    __host__ __device__ void operator()(size_t i) {
        double *a = _a + 24 * i;
        const double *x = _x + 24 * i;
        for (int c = 0; c < 24; c++) {
            a[c] = x[c] + _s * a[c];
        }
    }
};

// a (+)= sum_{k < n} s_k x_k, n <= 8 (several vectors per pass: block updates of the deflated CG and rotations).
struct MDWFCvAxpyN {
    double *_a;
    const double *_x[8];
    double _sr[8], _si[8];
    int _n;
    bool _assign;
    __host__ __device__ void operator()(size_t i) {
        double *a = _a + 24 * i;
        for (int c = 0; c < 12; c++) {
            double re = _assign ? 0.0 : a[2 * c], im = _assign ? 0.0 : a[2 * c + 1];
            for (int k = 0; k < _n; k++) {
                const double xr = _x[k][24 * i + 2 * c], xi = _x[k][24 * i + 2 * c + 1];
                re += _sr[k] * xr - _si[k] * xi;
                im += _sr[k] * xi + _si[k] * xr;
            }
            a[2 * c] = re;
            a[2 * c + 1] = im;
        }
    }
};

struct MDWFCvDot {   // out[i] = sum_c conj(a_c) b_c
    const double *_a, *_b;
    COMPLEX(double) *_out;
    __host__ __device__ void operator()(size_t i) {
        const double *a = _a + 24 * i, *b = _b + 24 * i;
        double re = 0.0, im = 0.0;
        for (int c = 0; c < 12; c++) {
            re += a[2 * c] * b[2 * c] + a[2 * c + 1] * b[2 * c + 1];
            im += a[2 * c] * b[2 * c + 1] - a[2 * c + 1] * b[2 * c];
        }
        _out[i] = COMPLEX(double)(re, im);
    }
};

template<size_t HaloDepth, size_t Ls>
struct MDWFCvPack {   // compact <- field (bulk)
    double *_a;
    Vect12ArrayAcc<double> _f;
    size_t _sizeh;
    __host__ __device__ void operator()(size_t i) {
        typedef GIndexer<Even, HaloDepth> GInd;
        const Vect12<double> v = _f.getElement(GInd::getSiteStack(i % _sizeh, i / _sizeh));
        double *a = _a + 24 * i;
        for (int c = 0; c < 12; c++) {
            a[2 * c] = v.data[c].cREAL;
            a[2 * c + 1] = v.data[c].cIMAG;
        }
    }
};

template<size_t HaloDepth, size_t Ls>
struct MDWFCvUnpack {   // field (bulk) <- compact
    const double *_a;
    Vect12ArrayAcc<double> _f;
    size_t _sizeh;
    __host__ __device__ void operator()(size_t i) {
        typedef GIndexer<Even, HaloDepth> GInd;
        const double *a = _a + 24 * i;
        Vect12<double> v;
        for (int c = 0; c < 12; c++) {
            v.data[c] = COMPLEX(double)(a[2 * c], a[2 * c + 1]);
        }
        _f.setElement(GInd::getSiteStack(i % _sizeh, i / _sizeh), v);
    }
};

__host__ __device__ inline uint64_t mdwfDeflSplitMix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Complex Gaussian noise from a hash of (seed, global site, s, component): the same vector for any node layout.
template<size_t HaloDepth, size_t Ls>
struct MDWFCvNoise {
    double *_a;
    size_t _sizeh;
    uint64_t _seed;
    __host__ __device__ void operator()(size_t i) {
        typedef GIndexer<Even, HaloDepth> GInd;
        const gSite site = GInd::getSite(i % _sizeh);
        LatticeData lat = GInd::getLatData();
        const sitexyzt g = lat.globalPos(site.coord);
        const uint64_t lex = static_cast<uint64_t>(g.x)
                             + lat.globLX * (static_cast<uint64_t>(g.y)
                                             + lat.globLY * (static_cast<uint64_t>(g.z)
                                                             + lat.globLZ * static_cast<uint64_t>(g.t)));
        const uint64_t base = (lex * Ls + static_cast<uint64_t>(i / _sizeh)) * 12;
        double *a = _a + 24 * i;
        for (int c = 0; c < 12; c++) {
            const uint64_t h1 = mdwfDeflSplitMix64(_seed ^ mdwfDeflSplitMix64(2 * (base + c)));
            const uint64_t h2 = mdwfDeflSplitMix64(_seed ^ mdwfDeflSplitMix64(2 * (base + c) + 1));
            const double u1 = static_cast<double>(h1 >> 11) * (1.0 / 9007199254740992.0);
            const double u2 = static_cast<double>(h2 >> 11) * (1.0 / 9007199254740992.0);
            const double r = sqrt(-2.0 * log(1.0 - u1));
            a[2 * c] = r * cos(2.0 * M_PI * u2);
            a[2 * c + 1] = r * sin(2.0 * M_PI * u2);
        }
    }
};

inline std::string mdwfDeflUniqueName(const char *what) {
    // Names must not be substrings of one another (MemoryManagement::getSmartName, see mdwfHmcInstancePrefix).
    static unsigned long counter = 0;
    return std::string("MDWFdefl") + what + std::to_string(counter++) + "x";
}

template<size_t HaloDepth, size_t Ls>
class MDWFCompactVec {
public:
    using Field = MDWFSpinor<double, true, Even, HaloDepth, Ls>;

private:
    CommunicationBase &_comm;
    gMemoryPtr<true> _mem;
    size_t _sizeh;
    size_t _n;   // site-stacks: sizeh * Ls
    mutable std::shared_ptr<LatticeContainer<true, COMPLEX(double)>> _reduction;   // shared memory, made on first use

    template<class Functor>
    void run(Functor f) const {
        iterateFunctorNoReturn<true>(f, MDWFFlatIndex(), _n);
    }

public:
    // The name is ignored (the constructor signature of the fields the Lanczos code creates).
    explicit MDWFCompactVec(CommunicationBase &comm, const std::string & = std::string())
        : _comm(comm), _mem(MemoryManagement::getMemAt<true>(mdwfDeflUniqueName("cv"))),
          _sizeh(GIndexer<Even, HaloDepth>::getLatData().sizeh), _n(_sizeh * Ls) {
        _mem->template adjustSize<double>(24 * _n);
    }

    MDWFCompactVec(const MDWFCompactVec &) = delete;

    CommunicationBase &getComm() const {
        return _comm;
    }

    size_t siteStacks() const {
        return _n;
    }

    size_t bytes() const {
        return 24 * _n * sizeof(double);
    }

    double *ptr() const {
        return _mem->template getPointer<double>();
    }

    MDWFCompactVec &operator=(const MDWFCompactVec &rhs) {
        if (this != &rhs) {
            const gpuError_t err = gpuMemcpy(ptr(), rhs.ptr(), bytes(), gpuMemcpyDeviceToDevice);
            if (err) {
                GpuError("MDWFCompactVec: copy", err);
            }
        }
        return *this;
    }

    void operator*=(const COMPLEX(double) &s) {
        run(MDWFCvScale{ptr(), s.cREAL, s.cIMAG});
    }

    template<size_t BlockSize = 64>
    void axpyThisB(const double &s, const MDWFCompactVec &x) {
        run(MDWFCvAxpy{ptr(), x.ptr(), s, 0.0});
    }

    template<size_t BlockSize = 64>
    void axpyThisB(const COMPLEX(double) &s, const MDWFCompactVec &x) {
        run(MDWFCvAxpy{ptr(), x.ptr(), s.cREAL, s.cIMAG});
    }

    void axpy(const MDWFDeflC &s, const MDWFCompactVec &x) {
        run(MDWFCvAxpy{ptr(), x.ptr(), s.real(), s.imag()});
    }

    // this = x + s this
    void xpay(double s, const MDWFCompactVec &x) {
        run(MDWFCvXpay{ptr(), x.ptr(), s});
    }

    // this (+)= sum_k coef[k] x[k]
    void combine(const std::vector<const MDWFCompactVec *> &x, const std::vector<MDWFDeflC> &coef, bool assign) {
        if (x.size() != coef.size()) {
            throw std::runtime_error(stdLogger.fatal("MDWFCompactVec::combine: size mismatch"));
        }
        if (x.empty() && assign) {
            zero();
        }
        for (size_t first = 0; first < x.size(); first += 8) {
            MDWFCvAxpyN f;
            f._a = ptr();
            f._n = static_cast<int>(std::min<size_t>(8, x.size() - first));
            f._assign = assign && first == 0;
            for (int k = 0; k < 8; k++) {
                const size_t j = first + std::min(k, f._n - 1);
                f._x[k] = x[j]->ptr();
                f._sr[k] = k < f._n ? coef[j].real() : 0.0;
                f._si[k] = k < f._n ? coef[j].imag() : 0.0;
            }
            run(f);
        }
    }

    void zero() {
        run(MDWFCvScale{ptr(), 0.0, 0.0});
    }

    void updateAll() {}   // no halo

    void pack(const Field &field) {
        run(MDWFCvPack<HaloDepth, Ls>{ptr(), field.getAccessor(), _sizeh});
    }

    void unpack(Field &field) const {
        run(MDWFCvUnpack<HaloDepth, Ls>{ptr(), field.getAccessor(), _sizeh});
    }

    void noise(uint64_t seed) {
        run(MDWFCvNoise<HaloDepth, Ls>{ptr(), _sizeh, seed});
    }

    // <this|b>, all ranks.
    COMPLEX(double) dot(const MDWFCompactVec &b) const {
        if (!_reduction) {
            _reduction = std::make_shared<LatticeContainer<true, COMPLEX(double)>>(_comm);
        }
        _reduction->adjustSize(_n);
        run(MDWFCvDot{ptr(), b.ptr(), _reduction->get_ContainerArrayPtr()->template getPointer<COMPLEX(double)>()});
        COMPLEX(double) result = 0.0;
        _reduction->reduce(result, _n);
        return result;
    }

    double norm2() const {
        return real<double>(dot(*this));
    }

    // Host copies of the bulk (bit-exact), for the files.
    void toHost(std::vector<double> &host) const {
        host.resize(24 * _n);
        const gpuError_t err = gpuMemcpy(host.data(), ptr(), bytes(), gpuMemcpyDeviceToHost);
        if (err) {
            GpuError("MDWFCompactVec: copy to host", err);
        }
    }

    void fromHost(const std::vector<double> &host) {
        if (host.size() != 24 * _n) {
            throw std::runtime_error(stdLogger.fatal("MDWFCompactVec::fromHost: size mismatch"));
        }
        const gpuError_t err = gpuMemcpy(ptr(), host.data(), bytes(), gpuMemcpyHostToDevice);
        if (err) {
            GpuError("MDWFCompactVec: copy from host", err);
        }
    }
};

// The 5D inner product for the thick-restart Lanczos (more specialized than its 4D template).
template<size_t HaloDepth, size_t Ls>
COMPLEX(double) mdwfDot4(MDWFCompactVec<HaloDepth, Ls> &a, const MDWFCompactVec<HaloDepth, Ls> &b) {
    return a.dot(b);
}

template<size_t HaloDepth, size_t Ls>
inline MDWFDeflC mdwfDeflDot(const MDWFCompactVec<HaloDepth, Ls> &a, const MDWFCompactVec<HaloDepth, Ls> &b) {
    const COMPLEX(double) z = a.dot(b);
    return MDWFDeflC(real<double>(z), imag<double>(z));
}

template<size_t HaloDepth, size_t Ls>
using MDWFDeflSet = std::vector<std::unique_ptr<MDWFCompactVec<HaloDepth, Ls>>>;

template<size_t HaloDepth, size_t Ls>
void mdwfDeflAllocate(MDWFDeflSet<HaloDepth, Ls> &set, int count, CommunicationBase &comm) {
    set.clear();
    for (int k = 0; k < count; k++) {
        set.emplace_back(new MDWFCompactVec<HaloDepth, Ls>(comm));
    }
}

template<size_t HaloDepth, size_t Ls>
std::vector<const MDWFCompactVec<HaloDepth, Ls> *> mdwfDeflPointers(const MDWFDeflSet<HaloDepth, Ls> &set, int count) {
    std::vector<const MDWFCompactVec<HaloDepth, Ls> *> p;
    for (int k = 0; k < count; k++) {
        p.push_back(set[k].get());
    }
    return p;
}

// g[a * nb + b] = <A_a|B_b> for the first na / nb vectors.
template<size_t HaloDepth, size_t Ls>
std::vector<MDWFDeflC> mdwfDeflGram(const std::vector<const MDWFCompactVec<HaloDepth, Ls> *> &a,
                                    const std::vector<const MDWFCompactVec<HaloDepth, Ls> *> &b) {
    std::vector<MDWFDeflC> g(a.size() * b.size());
    for (size_t i = 0; i < a.size(); i++) {
        for (size_t j = 0; j < b.size(); j++) {
            g[i * b.size() + j] = mdwfDeflDot(*a[i], *b[j]);
        }
    }
    return g;
}

// Cosines of the principal angles between span(a) and span(b) (both orthonormal), descending.
template<size_t HaloDepth, size_t Ls>
std::vector<double> mdwfDeflPrincipalCosines(const std::vector<const MDWFCompactVec<HaloDepth, Ls> *> &a,
                                             const std::vector<const MDWFCompactVec<HaloDepth, Ls> *> &b) {
    return mdwfSingularValues(mdwfDeflGram(a, b), static_cast<int>(a.size()), static_cast<int>(b.size()));
}

// ---------------------------------------------------------------------------------------------------------------
// Operator
// ---------------------------------------------------------------------------------------------------------------

template<size_t HaloDepth, size_t Ls>
class MDWFDeflOperator {
public:
    using EvenOdd = MDWFMobiusCloverEvenOdd<double, HaloDepth, HaloDepth, Ls>;
    using Normal = MDWFMobiusSchurNormalOperator<EvenOdd>;
    using Field = typename EvenOdd::SpinorE;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using Vec = MDWFCompactVec<HaloDepth, Ls>;

private:
    EvenOdd _eo;
    Normal _normal;
    Field _in, _out;   // shared by all operators (they hold nothing between calls)
    long _applications = 0;

public:
    // fermionGauge: the links with the fermion boundary conditions (MDWFFermionBoundary.h). mmdag: Mhat Mhat^+.
    MDWFDeflOperator(Gauge &fermionGauge, double M5, double mass, double b5, bool mmdag, const std::string &name)
        : _eo(fermionGauge, M5, mass, b5, 0.0, name + "_eo"),
          _normal(_eo, fermionGauge.getComm(), name + "_nrm"),
          _in(fermionGauge.getComm(), EvenOdd::sharedEvenName("defl_op_in")),
          _out(fermionGauge.getComm(), EvenOdd::sharedEvenName("defl_op_out")) {
        _normal.setMMdag(mmdag);
    }

    // After every change of the gauge field.
    void refresh() {
        _eo.refresh();
    }

    void apply(Vec &out, const Vec &in) {
        in.unpack(_in);
        _normal.apply(_out, _in, false);
        out.pack(_out);
        _applications++;
    }

    long applications() const {
        return _applications;
    }

    EvenOdd &evenOdd() {
        return _eo;
    }

    Field &fieldIn() {
        return _in;
    }
};

// Largest Ritz value of A after `steps` Lanczos steps (no reorthogonalization): a lower bound of lambda_max.
template<class Op, class Vec>
double mdwfDeflLambdaMax(Op &op, const Vec &start, int steps) {
    Vec previous(start.getComm()), current(start.getComm()), work(start.getComm());
    current = start;
    current *= COMPLEX(double)(1.0 / std::sqrt(current.norm2()), 0.0);
    std::vector<double> alpha, beta;
    for (int j = 0; j < steps; j++) {
        op.apply(work, current);
        const double a = real<double>(current.dot(work));
        work.axpyThisB(-a, current);
        if (j > 0) {
            work.axpyThisB(-beta.back(), previous);
        }
        alpha.push_back(a);
        const double b = std::sqrt(work.norm2());
        if (!(b > 1e-14 * std::abs(a))) {
            break;
        }
        beta.push_back(b);
        previous = current;
        current = work;
        current *= COMPLEX(double)(1.0 / b, 0.0);
    }
    const int m = static_cast<int>(alpha.size());
    std::vector<double> t(static_cast<size_t>(m) * m, 0.0), ev, y;
    for (int j = 0; j < m; j++) {
        t[j * m + j] = alpha[j];
        if (j + 1 < m) {
            t[j * m + j + 1] = beta[j];
            t[(j + 1) * m + j] = beta[j];
        }
    }
    mdwfSymmetricEigen(t, m, ev, y);
    return ev.back();
}

// ---------------------------------------------------------------------------------------------------------------
// Rayleigh-Ritz
// ---------------------------------------------------------------------------------------------------------------

struct MDWFDeflRitz {
    std::vector<double> theta;      // Ritz values, ascending
    std::vector<double> residual;   // |A y - theta y| (if requested)
    int rank = 0;                   // numerical rank of the span
};

// x_k -= sum_j v_j <v_j|x_k> (twice), then x_k normalized; v orthonormal. Returns the norms after the projection.
template<size_t HaloDepth, size_t Ls>
std::vector<double> mdwfDeflProjectOut(MDWFDeflSet<HaloDepth, Ls> &x, int nx,
                                       const std::vector<const MDWFCompactVec<HaloDepth, Ls> *> &v) {
    std::vector<double> norms(nx, 0.0);
    std::vector<MDWFDeflC> coef(v.size());
    for (int k = 0; k < nx; k++) {
        for (int pass = 0; pass < 2; pass++) {
            for (size_t j = 0; j < v.size(); j++) {
                coef[j] = -mdwfDeflDot(*v[j], *x[k]);
            }
            x[k]->combine(v, coef, false);
        }
        norms[k] = std::sqrt(x[k]->norm2());
        if (norms[k] > 0.0) {
            *x[k] *= COMPLEX(double)(1.0 / norms[k], 0.0);
        }
    }
    return norms;
}

// Modified Gram-Schmidt of the first n vectors (removes the rounding of a rotation).
template<size_t HaloDepth, size_t Ls>
void mdwfDeflOrthonormalize(MDWFDeflSet<HaloDepth, Ls> &v, int n) {
    for (int k = 0; k < n; k++) {
        for (int j = 0; j < k; j++) {
            v[k]->axpy(-mdwfDeflDot(*v[j], *v[k]), *v[j]);
        }
        *v[k] *= COMPLEX(double)(1.0 / std::sqrt(v[k]->norm2()), 0.0);
    }
}

/*
 * The nOut lowest Ritz pairs of A in span(w) (w normalized, possibly nearly dependent: directions of the overlap
 * matrix below rankTol times its largest eigenvalue are dropped). out[0 .. nOut-1] <- orthonormal Ritz vectors (out
 * must not alias w). work: one scratch vector. withResiduals costs nOut more applications.
 */
template<class Op, size_t HaloDepth, size_t Ls>
MDWFDeflRitz mdwfDeflRayleighRitz(Op &op, const std::vector<const MDWFCompactVec<HaloDepth, Ls> *> &w, int nOut,
                                  MDWFDeflSet<HaloDepth, Ls> &out, MDWFCompactVec<HaloDepth, Ls> &work,
                                  bool withResiduals, double rankTol = 1e-12) {
    const int m = static_cast<int>(w.size());
    std::vector<MDWFDeflC> s(static_cast<size_t>(m) * m), h(static_cast<size_t>(m) * m);
    for (int k = 0; k < m; k++) {
        op.apply(work, *w[k]);
        for (int l = 0; l <= k; l++) {
            const MDWFDeflC hlk = mdwfDeflDot(*w[l], work);
            h[l * m + k] = hlk;
            h[k * m + l] = std::conj(hlk);
            const MDWFDeflC slk = mdwfDeflDot(*w[l], *w[k]);
            s[l * m + k] = slk;
            s[k * m + l] = std::conj(slk);
        }
    }
    // Canonical orthogonalization: S = Q D Q^+, T = Q_r D_r^(-1/2).
    std::vector<double> d;
    std::vector<MDWFDeflC> q;
    mdwfHermitianEigen(s, m, d, q);
    std::vector<int> keep;
    for (int l = 0; l < m; l++) {
        if (d[l] > rankTol * d[m - 1]) {
            keep.push_back(l);
        }
    }
    const int r = static_cast<int>(keep.size());
    if (r < nOut) {
        throw std::runtime_error(stdLogger.fatal("mdwfDeflRayleighRitz: span of rank ", r, " for ", nOut, " vectors"));
    }
    std::vector<MDWFDeflC> t(static_cast<size_t>(m) * r);
    for (int i = 0; i < m; i++) {
        for (int a = 0; a < r; a++) {
            t[i * r + a] = q[i * m + keep[a]] / std::sqrt(d[keep[a]]);
        }
    }
    std::vector<MDWFDeflC> ht(static_cast<size_t>(m) * r, MDWFDeflC(0.0, 0.0)), hp(static_cast<size_t>(r) * r,
                                                                                    MDWFDeflC(0.0, 0.0));
    for (int i = 0; i < m; i++) {
        for (int a = 0; a < r; a++) {
            MDWFDeflC z(0.0, 0.0);
            for (int j = 0; j < m; j++) {
                z += h[i * m + j] * t[j * r + a];
            }
            ht[i * r + a] = z;
        }
    }
    for (int a = 0; a < r; a++) {
        for (int b = 0; b < r; b++) {
            MDWFDeflC z(0.0, 0.0);
            for (int i = 0; i < m; i++) {
                z += std::conj(t[i * r + a]) * ht[i * r + b];
            }
            hp[a * r + b] = z;
        }
    }
    std::vector<double> theta;
    std::vector<MDWFDeflC> y;
    mdwfHermitianEigen(hp, r, theta, y);

    MDWFDeflRitz result;
    result.rank = r;
    result.theta.assign(theta.begin(), theta.begin() + nOut);
    std::vector<MDWFDeflC> coef(m);
    for (int k = 0; k < nOut; k++) {
        for (int i = 0; i < m; i++) {
            MDWFDeflC z(0.0, 0.0);
            for (int a = 0; a < r; a++) {
                z += t[i * r + a] * y[a * r + k];
            }
            coef[i] = z;
        }
        out[k]->combine(w, coef, true);
    }
    mdwfDeflOrthonormalize(out, nOut);
    if (withResiduals) {
        result.residual.resize(nOut);
        for (int k = 0; k < nOut; k++) {
            op.apply(work, *out[k]);
            result.theta[k] = real<double>(out[k]->dot(work));
            work.axpyThisB(-result.theta[k], *out[k]);
            result.residual[k] = std::sqrt(work.norm2());
        }
    }
    return result;
}

/*
 * One tracking step of a K-dimensional subspace v (orthonormal) to the operator a:
 *   aPrev != nullptr   T2: lowest K Ritz vectors of a in span{v, (a - aPrev) v}   (3K applications)
 *   aPrev == nullptr   T3: lowest K Ritz vectors of a in span{v, a v}             (2K applications)
 * v is replaced by the Ritz vectors. x, z: scratch sets of K vectors (their contents are destroyed).
 */
template<class Op, size_t HaloDepth, size_t Ls>
MDWFDeflRitz mdwfDeflTrackStep(Op &a, Op *aPrev, MDWFDeflSet<HaloDepth, Ls> &v, int K, MDWFDeflSet<HaloDepth, Ls> &x,
                               MDWFDeflSet<HaloDepth, Ls> &z, MDWFCompactVec<HaloDepth, Ls> &work,
                               bool withResiduals) {
    for (int k = 0; k < K; k++) {
        if (aPrev != nullptr) {
            aPrev->apply(*x[k], *v[k]);
            a.apply(work, *v[k]);
            *x[k] *= COMPLEX(double)(-1.0, 0.0);
            x[k]->axpyThisB(1.0, work);
        } else {
            a.apply(*x[k], *v[k]);
        }
    }
    std::vector<const MDWFCompactVec<HaloDepth, Ls> *> w = mdwfDeflPointers(v, K);
    const std::vector<double> norms = mdwfDeflProjectOut(x, K, w);
    for (int k = 0; k < K; k++) {
        if (norms[k] > 0.0) {
            w.push_back(x[k].get());
        }
    }
    const MDWFDeflRitz ritz = mdwfDeflRayleighRitz(a, w, K, z, work, withResiduals);
    v.swap(z);
    return ritz;
}

// ---------------------------------------------------------------------------------------------------------------
// Lowest eigenpairs
// ---------------------------------------------------------------------------------------------------------------

struct MDWFDeflEigenParameters {
    int nev = 16;
    int nBasis = 0;           // 0: nev + max(24, nev / 2) + 16
    int nKeep = 0;            // 0: nev + max(8, nev / 4)
    double tol = 1e-11;       // Lanczos tolerance, relative to the largest Ritz value of the filtered operator
    int maxRestarts = 60;
    double strength = 4.0;    // Chebyshev amplification of lambda = 0: T_d = cosh(strength)
    double lo = 0.0;          // filter lower edge (above the wanted eigenvalues); <= 0: found automatically
    double hi = 0.0;          // filter upper edge (>= lambda_max); <= 0: 1.15 x Lanczos estimate
    int hiSteps = 60;
    int maxDegree = 6000;
    int scoutRestarts = 2;    // restarts of an automatic attempt before the edge is moved
    int maxAttempts = 8;
};

struct MDWFDeflEigenResult {
    std::vector<double> lambda;     // ascending
    std::vector<double> residual;   // |A v - lambda v|
    double lo = 0.0, hi = 0.0;
    int degree = 0;
    int attempts = 0;
    long applications = 0;          // applications of A, everything included
    bool converged = false;
    double seconds = 0.0;
};

/*
 * The nev lowest eigenpairs of A into out[0 .. nev-1]: thick-restart Lanczos on -T_d(L(A)) (MDWFChebyshevOperator),
 * then Rayleigh-Ritz with A itself. The filter edge lo must lie above lambda_nev; too high an edge leaves the wanted
 * modes in a cluster (slow), too low an edge hides some of them. With p.lo <= 0 the edge starts at 1e-3 hi and is
 * moved to 2.5 times the largest Rayleigh quotient of the current Ritz vectors (an upper bound of lambda_nev) until
 * the Lanczos converges with all wanted Rayleigh quotients below 0.8 lo.
 */
template<class Op, size_t HaloDepth, size_t Ls>
MDWFDeflEigenResult mdwfDeflLowModes(Op &op, MDWFDeflSet<HaloDepth, Ls> &out, MDWFDeflEigenParameters p,
                                     uint64_t seed, CommunicationBase &comm) {
    using Vec = MDWFCompactVec<HaloDepth, Ls>;
    const auto t0 = std::chrono::steady_clock::now();
    const long applicationsBefore = op.applications();
    if (p.nBasis <= 0) {
        p.nBasis = p.nev + std::max(24, p.nev / 2) + 16;
    }
    if (p.nKeep <= 0) {
        p.nKeep = p.nev + std::max(8, p.nev / 4);
    }
    MDWFDeflEigenResult result;
    Vec start(comm), work(comm);
    start.noise(seed);
    result.hi = p.hi > 0.0 ? p.hi : 1.15 * mdwfDeflLambdaMax(op, start, p.hiSteps);
    const bool autoEdge = !(p.lo > 0.0);
    double lo = autoEdge ? 1e-3 * result.hi : p.lo;

    MDWFDeflSet<HaloDepth, Ls> basis;
    std::vector<double> rayleigh(p.nev);
    for (int attempt = 1; ; attempt++) {
        const double arg = (result.hi + lo) / (result.hi - lo);
        const int degree = std::min(p.maxDegree, std::max(8, static_cast<int>(std::ceil(p.strength / std::acosh(arg)))));
        const bool last = attempt >= p.maxAttempts;
        const MDWFThickRestartParameters tp{p.nev, p.nBasis, p.nKeep, p.tol,
                                            (autoEdge && !last) ? p.scoutRestarts : p.maxRestarts};
        rootLogger.info("MDWF deflation eigensolver: attempt ", attempt, ", Chebyshev interval [", lo, ", ", result.hi,
                        "], degree ", degree, ", nev ", p.nev, ", nbasis ", p.nBasis, ", nkeep ", p.nKeep,
                        ", max restarts ", tp.maxRestarts);
        MDWFThickRestartResult tr;
        {
            MDWFChebyshevOperator<Vec, Op> filtered(op, lo, result.hi, degree, comm, "MDWF_defl_cheb");
            tr = mdwfThickRestartLanczos<Vec>(filtered, start, tp, basis, "MDWF_defl_trl");
        }
        basis.resize(p.nev);   // frees the rest of the Lanczos basis
        double largest = 0.0;
        for (int k = 0; k < p.nev; k++) {
            op.apply(work, *basis[k]);
            rayleigh[k] = real<double>(basis[k]->dot(work)) / basis[k]->norm2();
            largest = std::max(largest, rayleigh[k]);
        }
        result.lo = lo;
        result.degree = degree;
        result.attempts = attempt;
        result.converged = tr.converged;
        rootLogger.info("MDWF deflation eigensolver: attempt ", attempt, " converged ", tr.converged, " after ",
                        tr.restarts, " restarts, largest Rayleigh quotient ", largest, " (edge ", lo, ")");
        const bool edgeOk = largest < 0.8 * lo;
        if ((tr.converged && edgeOk) || last) {
            if (!(tr.converged && edgeOk)) {
                rootLogger.warn("MDWF deflation eigensolver: NOT converged within ", p.maxAttempts, " attempts; the "
                                "residuals below are the real test");
            }
            break;
        }
        if (!autoEdge) {
            // A given edge: only one retry with a moved edge is useful if the edge was too low.
            if (tr.converged || largest >= 0.8 * lo) {
                lo = 2.5 * largest;
            } else {
                p.maxRestarts *= 2;
            }
        } else if (tr.converged || 2.5 * largest < 0.9 * lo || largest >= 0.8 * lo) {
            lo = 2.5 * largest;          // converged with the edge too low, or the edge far above the wanted modes
        } else {
            p.scoutRestarts = p.maxRestarts;   // the edge is right: only more restarts help
        }
        // Next start vector: the sum of the current Ritz vectors (rich in the wanted directions).
        start = *basis[0];
        for (int k = 1; k < p.nev; k++) {
            start.axpyThisB(1.0, *basis[k]);
        }
    }
    // Rayleigh-Ritz with A in the span of the Lanczos vectors: eigenvalues ascending, true residuals.
    mdwfDeflAllocate(out, p.nev, comm);
    const MDWFDeflRitz ritz = mdwfDeflRayleighRitz(op, mdwfDeflPointers(basis, p.nev), p.nev, out, work, true);
    result.lambda = ritz.theta;
    result.residual = ritz.residual;
    result.applications = op.applications() - applicationsBefore;
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return result;
}

// ---------------------------------------------------------------------------------------------------------------
// Deflated CG
// ---------------------------------------------------------------------------------------------------------------

/*
 * A deflation subspace span(v_0 ... v_{K-1}) (any basis) with its Galerkin data for one operator: av_k = A v_k and
 * ginv = (V^+ A V)^-1. setup() costs K applications; setupSpectral() takes the vectors as eigenvectors with the given
 * eigenvalues instead (no applications, av is not available: initial guess only).
 */
template<size_t HaloDepth, size_t Ls>
struct MDWFDeflSubspace {
    using Vec = MDWFCompactVec<HaloDepth, Ls>;
    std::vector<const Vec *> v;
    MDWFDeflSet<HaloDepth, Ls> av;
    std::vector<MDWFDeflC> ginv;
    bool galerkin = false;

    int size() const {
        return static_cast<int>(v.size());
    }

    template<class Op>
    void setup(Op &op, const std::vector<const Vec *> &vectors, CommunicationBase &comm) {
        v = vectors;
        const int k = size();
        if (static_cast<int>(av.size()) < k) {
            mdwfDeflAllocate(av, k, comm);
        }
        std::vector<MDWFDeflC> g(static_cast<size_t>(k) * k);
        for (int b = 0; b < k; b++) {
            op.apply(*av[b], *v[b]);
            for (int a = 0; a <= b; a++) {
                const MDWFDeflC z = mdwfDeflDot(*v[a], *av[b]);
                g[a * k + b] = z;
                g[b * k + a] = std::conj(z);
            }
        }
        ginv = mdwfHermitianInverse(g, k);
        galerkin = true;
    }

    void setupSpectral(const std::vector<const Vec *> &vectors, const std::vector<double> &lambda) {
        v = vectors;
        const int k = size();
        ginv.assign(static_cast<size_t>(k) * k, MDWFDeflC(0.0, 0.0));
        for (int a = 0; a < k; a++) {
            ginv[a * k + a] = 1.0 / lambda[a];
        }
        galerkin = false;
    }

    // coef = ginv (U^+ r), U = v or av.
    std::vector<MDWFDeflC> coefficients(const Vec &r, bool withAv) const {
        const int k = size();
        std::vector<MDWFDeflC> h(k), c(k);
        for (int a = 0; a < k; a++) {
            h[a] = mdwfDeflDot(withAv ? *av[a] : *v[a], r);
        }
        for (int a = 0; a < k; a++) {
            MDWFDeflC z(0.0, 0.0);
            for (int b = 0; b < k; b++) {
                z += ginv[a * k + b] * h[b];
            }
            c[a] = z;
        }
        return c;
    }
};

struct MDWFDeflCGResult {
    int iterations = 0;          // CG iterations = applications of A in the loop
    double residual = 0.0;       // recursive |r| / |b| at the end
    double trueResidual = 0.0;   // |b - A x| / |b|
    double initialResidual = 0.0;   // |b - A x0| / |b| after the deflated initial guess
    bool converged = false;
    double seconds = 0.0;
};

enum MDWFDeflCGMode {
    MDWFDeflNone = 0,      // plain CG from zero
    MDWFDeflInit = 1,      // deflated initial guess x0 = V ginv V^+ b, then plain CG
    MDWFDeflProject = 2    // deflated CG: the search directions are kept A-orthogonal to V in every iteration
};

/*
 * Solves A x = b to |r| <= precision |b| (recursive residual) in double precision. Deflated CG of Saad et al.:
 *   x0 = V ginv V^+ b,  r0 = b - A x0,  p0 = r0 - V ginv (AV)^+ r0,
 *   x += a p,  r -= a A p,  p = r + beta p - V ginv (AV)^+ r.
 * It converges for any subspace (a bad one costs iterations, not correctness). MDWFDeflProject needs a Galerkin
 * subspace (setup()).
 */
template<class Op, size_t HaloDepth, size_t Ls>
MDWFDeflCGResult mdwfDeflCG(Op &op, MDWFCompactVec<HaloDepth, Ls> &x, const MDWFCompactVec<HaloDepth, Ls> &b,
                            const MDWFDeflSubspace<HaloDepth, Ls> *subspace, MDWFDeflCGMode mode, double precision,
                            int maxIter, CommunicationBase &comm) {
    using Vec = MDWFCompactVec<HaloDepth, Ls>;
    const auto t0 = std::chrono::steady_clock::now();
    if (mode != MDWFDeflNone && subspace == nullptr) {
        throw std::runtime_error(stdLogger.fatal("mdwfDeflCG: deflation without a subspace"));
    }
    if (mode == MDWFDeflProject && !subspace->galerkin) {
        throw std::runtime_error(stdLogger.fatal("mdwfDeflCG: MDWFDeflProject needs MDWFDeflSubspace::setup()"));
    }
    Vec r(comm), p(comm), ap(comm);
    MDWFDeflCGResult result;
    const double bb = b.norm2();
    if (!(bb > 0.0)) {
        throw std::runtime_error(stdLogger.fatal("mdwfDeflCG: zero right-hand side"));
    }
    const double target = precision * precision * bb;

    r = b;
    if (mode == MDWFDeflNone) {
        x.zero();
    } else {
        x.combine(subspace->v, subspace->coefficients(b, false), true);
        op.apply(ap, x);
        r.axpyThisB(-1.0, ap);
    }
    double rr = r.norm2();
    result.initialResidual = std::sqrt(rr / bb);
    p = r;
    if (mode == MDWFDeflProject) {
        std::vector<MDWFDeflC> mu = subspace->coefficients(r, true);
        for (auto &z : mu) {
            z = -z;
        }
        p.combine(subspace->v, mu, false);
    }
    while (rr > target && result.iterations < maxIter) {
        op.apply(ap, p);
        const double pap = real<double>(p.dot(ap));
        if (!std::isfinite(pap) || !(pap > 0.0)) {
            throw std::runtime_error(stdLogger.fatal("mdwfDeflCG: p^+ A p = ", pap, " at iteration ",
                                                     result.iterations));
        }
        const double alpha = rr / pap;
        x.axpyThisB(alpha, p);
        r.axpyThisB(-alpha, ap);
        const double rrNew = r.norm2();
        result.iterations++;
        const double beta = rrNew / rr;
        rr = rrNew;
        if (rr <= target) {
            break;
        }
        p.xpay(beta, r);
        if (mode == MDWFDeflProject) {
            std::vector<MDWFDeflC> mu = subspace->coefficients(r, true);
            for (auto &z : mu) {
                z = -z;
            }
            p.combine(subspace->v, mu, false);
        }
    }
    result.residual = std::sqrt(rr / bb);
    result.converged = rr <= target;
    op.apply(ap, x);
    r = b;
    r.axpyThisB(-1.0, ap);
    result.trueResidual = std::sqrt(r.norm2() / bb);
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return result;
}

// ---------------------------------------------------------------------------------------------------------------
// Eigenvector sets on disk: <base>.r<rank>, header + count eigenvalues + count vectors (bulk doubles, bit-exact).
// The reader needs the node layout of the writer.
// ---------------------------------------------------------------------------------------------------------------

struct MDWFDeflSetHeader {
    char magic[8];        // "MDWFEVS1"
    int64_t count;
    int64_t doublesPerVector;
    int64_t globL[4];
    int64_t localL[4];
    int64_t gPos[4];
    int64_t nranks;
    int64_t rank;
    int64_t stacks;
};

template<size_t HaloDepth, size_t Ls>
MDWFDeflSetHeader mdwfDeflSetHeader(CommunicationBase &comm, int64_t count) {
    const LatticeData lat = GIndexer<All, HaloDepth>::getLatData();
    MDWFDeflSetHeader h{};
    std::memcpy(h.magic, "MDWFEVS1", 8);
    h.count = count;
    h.doublesPerVector = 24 * static_cast<int64_t>(lat.sizeh) * static_cast<int64_t>(Ls);
    h.globL[0] = lat.globLX; h.globL[1] = lat.globLY; h.globL[2] = lat.globLZ; h.globL[3] = lat.globLT;
    h.localL[0] = lat.lx; h.localL[1] = lat.ly; h.localL[2] = lat.lz; h.localL[3] = lat.lt;
    h.gPos[0] = lat.gPosX; h.gPos[1] = lat.gPosY; h.gPos[2] = lat.gPosZ; h.gPos[3] = lat.gPosT;
    h.nranks = comm.getNumberProcesses();
    h.rank = comm.MyRank();
    h.stacks = Ls;
    return h;
}

template<size_t HaloDepth, size_t Ls>
void mdwfDeflWriteSet(CommunicationBase &comm, const MDWFDeflSet<HaloDepth, Ls> &set, int count,
                      const std::vector<double> &lambda, const std::string &base) {
    const std::string file = mdwfFieldDumpRankFile(base, comm.MyRank());
    const std::string part = file + ".part";
    std::FILE *out = std::fopen(part.c_str(), "wb");
    if (out == nullptr) {
        throw std::runtime_error(stdLogger.fatal("MDWF deflation set: cannot open ", part));
    }
    const MDWFDeflSetHeader h = mdwfDeflSetHeader<HaloDepth, Ls>(comm, count);
    bool ok = std::fwrite(&h, sizeof(h), 1, out) == 1;
    ok = ok && std::fwrite(lambda.data(), sizeof(double), count, out) == static_cast<size_t>(count);
    std::vector<double> host;
    for (int k = 0; k < count && ok; k++) {
        set[k]->toHost(host);
        ok = std::fwrite(host.data(), sizeof(double), host.size(), out) == host.size();
    }
    ok = (std::fclose(out) == 0) && ok;
    if (!ok || std::rename(part.c_str(), file.c_str()) != 0) {
        throw std::runtime_error(stdLogger.fatal("MDWF deflation set: write error on ", file));
    }
    comm.globalBarrier();
}

// Reads the first count vectors (count <= the number in the file) into set[0 .. count-1] (allocated here).
template<size_t HaloDepth, size_t Ls>
void mdwfDeflReadSet(CommunicationBase &comm, MDWFDeflSet<HaloDepth, Ls> &set, int count, std::vector<double> &lambda,
                     const std::string &base) {
    const std::string file = mdwfFieldDumpRankFile(base, comm.MyRank());
    std::FILE *in = std::fopen(file.c_str(), "rb");
    if (in == nullptr) {
        throw std::runtime_error(stdLogger.fatal("MDWF deflation set: cannot open ", file));
    }
    MDWFDeflSetHeader h{};
    const MDWFDeflSetHeader mine = mdwfDeflSetHeader<HaloDepth, Ls>(comm, count);
    bool ok = std::fread(&h, sizeof(h), 1, in) == 1;
    ok = ok && std::memcmp(h.magic, mine.magic, 8) == 0 && h.count >= count
         && h.doublesPerVector == mine.doublesPerVector && h.nranks == mine.nranks && h.rank == mine.rank
         && h.stacks == mine.stacks;
    for (int d = 0; d < 4 && ok; d++) {
        ok = h.globL[d] == mine.globL[d] && h.localL[d] == mine.localL[d] && h.gPos[d] == mine.gPos[d];
    }
    if (!ok) {
        std::fclose(in);
        throw std::runtime_error(stdLogger.fatal("MDWF deflation set ", file, ": header mismatch (", h.count,
                                                 " vectors for ", count, " wanted; lattice, node layout and Ls must "
                                                 "be those of the writer)"));
    }
    std::vector<double> all(h.count);
    ok = std::fread(all.data(), sizeof(double), h.count, in) == static_cast<size_t>(h.count);
    lambda.assign(all.begin(), all.begin() + count);
    if (static_cast<int>(set.size()) < count) {
        mdwfDeflAllocate(set, count, comm);
    }
    std::vector<double> host(h.doublesPerVector);
    for (int k = 0; k < count && ok; k++) {
        ok = std::fread(host.data(), sizeof(double), host.size(), in) == host.size();
        if (ok) {
            set[k]->fromHost(host);
        }
    }
    std::fclose(in);
    if (!ok) {
        throw std::runtime_error(stdLogger.fatal("MDWF deflation set: short read on ", file));
    }
}

// All eigenvalue slots of a set file (rank 0's file of the writer; every rank's file holds the same values).
inline std::vector<double> mdwfDeflReadSetLambdas(const std::string &base) {
    const std::string file = mdwfFieldDumpRankFile(base, 0);
    std::FILE *in = std::fopen(file.c_str(), "rb");
    if (in == nullptr) {
        throw std::runtime_error(stdLogger.fatal("MDWF deflation set: cannot open ", file));
    }
    MDWFDeflSetHeader h{};
    bool ok = std::fread(&h, sizeof(h), 1, in) == 1 && std::memcmp(h.magic, "MDWFEVS1", 8) == 0 && h.count > 0;
    std::vector<double> lambda(ok ? h.count : 0);
    ok = ok && std::fread(lambda.data(), sizeof(double), h.count, in) == static_cast<size_t>(h.count);
    std::fclose(in);
    if (!ok) {
        throw std::runtime_error(stdLogger.fatal("MDWF deflation set ", file, ": cannot read the eigenvalues"));
    }
    return lambda;
}

// ---------------------------------------------------------------------------------------------------------------
// Refresh of a stale vector: a fixed number n of applications of the NEW operator A, the iterate kept at every n of
// an ascending list (one run gives all of them, since both recurrences below are nested in n).
//   CG         x_n = n iterations of plain CG on A x = b from x_0 = 0, no convergence stop: inverse iteration with
//              an inexact solve. The operations are those of mdwfDeflCG(MDWFDeflNone) with precision 0, so x_n equals
//              mdwfDeflCG(..., maxIter = n) bit for bit. n = 0 gives b itself (x_0 = 0 is no direction).
//   Chebyshev  y_n = T_n(L(A)) v, L(A) = (hi + lo - 2 A) / (hi - lo), by y_0 = v, y_1 = L y_0, y_k = 2 L y_{k-1} -
//              y_{k-2}: the recurrence of MDWFChebyshevOperator (which returns -y_d for its degree d), with fixed lo, hi
//              for all k, so y_k is the degree-k filter for every k. Eigenvalues of A in [lo, hi] get |T_n| <= 1, those
//              below lo cosh(n acosh L(lambda)) > 1. hi must be >= lambda_max(A). If |y_k| exceeds 1e100, y_k and
//              y_{k-1} are scaled by the same factor (the recurrence is linear, so later y_k only change by that
//              factor); the snapshots are meant to be normalised by the caller.
// The callback gets (n, iterate, seconds since the start without the time spent in the callbacks). n = 0 in the
// list passes the input vector itself. Both return the applications of A (= the largest n).
// ---------------------------------------------------------------------------------------------------------------

template<size_t HaloDepth, size_t Ls>
using MDWFDeflSnapshotCallback = std::function<void(int, const MDWFCompactVec<HaloDepth, Ls> &, double)>;

inline void mdwfDeflCheckSnapshotList(const std::vector<int> &n) {
    for (size_t i = 0; i < n.size(); i++) {
        if (n[i] < 0 || (i > 0 && n[i] <= n[i - 1])) {
            throw std::runtime_error(stdLogger.fatal("MDWF refresh: snapshot list must be ascending, >= 0"));
        }
    }
}

template<class Op, size_t HaloDepth, size_t Ls>
long mdwfDeflCGSnapshots(Op &op, const MDWFCompactVec<HaloDepth, Ls> &b, const std::vector<int> &snapshots,
                         const MDWFDeflSnapshotCallback<HaloDepth, Ls> &callback, CommunicationBase &comm) {
    using Vec = MDWFCompactVec<HaloDepth, Ls>;
    mdwfDeflCheckSnapshotList(snapshots);
    const auto t0 = std::chrono::steady_clock::now();
    double paused = 0.0;
    size_t next = 0;
    auto emit = [&](int n, const Vec &v) {
        const auto c0 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(c0 - t0).count() - paused;
        callback(n, v, seconds);
        paused += std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
        next++;
    };
    if (next < snapshots.size() && snapshots[next] == 0) {
        emit(0, b);
    }
    if (next == snapshots.size()) {
        return 0;
    }
    Vec x(comm), r(comm), p(comm), ap(comm);
    r = b;
    x.zero();
    double rr = r.norm2();
    if (!(rr > 0.0)) {
        throw std::runtime_error(stdLogger.fatal("mdwfDeflCGSnapshots: zero right-hand side"));
    }
    p = r;
    int it = 0;
    while (next < snapshots.size()) {
        if (rr == 0.0) {   // exact solution reached (b in a tiny invariant subspace): x stays as it is
            emit(snapshots[next], x);
            continue;
        }
        op.apply(ap, p);
        const double pap = real<double>(p.dot(ap));
        if (!std::isfinite(pap) || !(pap > 0.0)) {
            throw std::runtime_error(stdLogger.fatal("mdwfDeflCGSnapshots: p^+ A p = ", pap, " at iteration ", it));
        }
        const double alpha = rr / pap;
        x.axpyThisB(alpha, p);
        r.axpyThisB(-alpha, ap);
        const double rrNew = r.norm2();
        it++;
        const double beta = rrNew / rr;
        rr = rrNew;
        if (snapshots[next] == it) {
            emit(it, x);
        }
        if (next == snapshots.size()) {
            break;
        }
        p.xpay(beta, r);
    }
    return it;
}

template<class Op, size_t HaloDepth, size_t Ls>
long mdwfDeflChebyshevSnapshots(Op &op, const MDWFCompactVec<HaloDepth, Ls> &v, double lo, double hi,
                                const std::vector<int> &snapshots,
                                const MDWFDeflSnapshotCallback<HaloDepth, Ls> &callback, CommunicationBase &comm) {
    using Vec = MDWFCompactVec<HaloDepth, Ls>;
    mdwfDeflCheckSnapshotList(snapshots);
    if (!(hi > lo)) {
        throw std::runtime_error(stdLogger.fatal("mdwfDeflChebyshevSnapshots: needs hi > lo, got lo = ", lo, ", hi = ",
                                                 hi));
    }
    // As the MDWFChebyshevOperator constructor.
    const double scale = -2.0 / (hi - lo), shift = (hi + lo) / (hi - lo);
    const auto t0 = std::chrono::steady_clock::now();
    double paused = 0.0;
    size_t next = 0;
    auto emit = [&](int n, const Vec &y) {
        const auto c0 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(c0 - t0).count() - paused;
        callback(n, y, seconds);
        paused += std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
        next++;
    };
    if (next < snapshots.size() && snapshots[next] == 0) {
        emit(0, v);
    }
    if (next == snapshots.size()) {
        return 0;
    }
    Vec y0(comm), y1(comm), y2(comm), t(comm);
    Vec *p0 = &y0, *p1 = &y1, *p2 = &y2;
    *p0 = v;
    op.apply(t, *p0);                            // y1 = L y0
    *p1 = t;
    *p1 *= COMPLEX(double)(scale, 0.0);
    p1->axpyThisB(shift, *p0);
    int degree = 1;
    if (snapshots[next] == 1) {
        emit(1, *p1);
    }
    while (next < snapshots.size()) {           // y2 = 2 L y1 - y0
        op.apply(t, *p1);
        *p2 = t;
        *p2 *= COMPLEX(double)(2.0 * scale, 0.0);
        p2->axpyThisB(2.0 * shift, *p1);
        p2->axpyThisB(-1.0, *p0);
        Vec *old = p0;
        p0 = p1;
        p1 = p2;
        p2 = old;
        degree++;
        if (degree % 16 == 0) {
            const double n2 = p1->norm2();
            if (!std::isfinite(n2)) {
                throw std::runtime_error(stdLogger.fatal("mdwfDeflChebyshevSnapshots: overflow at degree ", degree,
                                                         " (hi below lambda_max?)"));
            }
            if (n2 > 1e200) {
                const COMPLEX(double) s(1.0 / std::sqrt(n2), 0.0);
                *p1 *= s;
                *p0 *= s;
            }
        }
        if (snapshots[next] == degree) {
            emit(degree, *p1);
        }
    }
    return degree;
}
