/*
 * Two-grid preconditioner for the MDWF Schur normal operator A (Stage 0 of DSDRMDWFruns/SQCDruns/multigrid/DESIGN.md):
 * an offline study tool, single rank. Nothing here is used by the HMC or the production measurements.
 *
 * Coarse space (HDCG type, Boyle / Yamaguchi arXiv:2103.05034). The even sites of the lattice are cut into 4D blocks
 * b of extent block[mu] (each >= 4 where there is more than one block in that direction, so that A, which couples
 * even sites up to taxicab distance 4, reaches only the nearest blocks), each block keeps the full fifth dimension,
 * and each block is split by chirality chi (P_+ = components 0..5, P_- = 6..11 of the chiral basis). ns fine vectors
 * W_k are orthonormalised separately on every (b, chi): the coarse vector c has the components c[(b, chi, k)],
 *     P c   = sum_k c[(b(x), chi, k)] W_k(x)          (prolongation)
 *     R v   = P^+ v,  (R v)[(b, chi, k)] = <W_k|v>_(b, chi)   (restriction)
 * and the coarse operator A_c = P^+ A P is assembled by probing: the blocks are coloured with period 4 (or one colour
 * per block in directions with fewer than 4 blocks) so that no block sees two blocks of the same colour; one
 * application of A per (colour, chi, k) gives the A_c columns of all blocks of that colour.
 *
 * Coarse solve: dense Cholesky of A_c (cuSOLVER; only built with MDWF_TWOGRID_CUSOLVER, the study target) - the
 * coarse dimension is limited to ~33k (17 GB).
 *
 * Preconditioner (symmetric two-grid cycle, fixed linear SPD operator):
 *     x = S r;  x += P A_c^-1 R (r - A x);  x += S (r - A x)
 * with S = degree-d Chebyshev approximation of A^-1 on [lo, hi] (d - 1 applications of A); 2 d applications of A per
 * cycle. Outer solver: flexible CG (Notay 2000, truncation 1) with the production stopping rule
 * |r|^2 <= precision^2 max(|b|^2, 1) and a true-residual check at the end. Plain CG with the same rule is the baseline.
 * All in double precision: Stage 0 counts applications of A, it does not tune precision.
 */

#pragma once

#include "MDWFDeflation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MDWF_TWOGRID_CUSOLVER
#include <cusolverDn.h>
#endif

// ---------------------------------------------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------------------------------------------

template<size_t HaloDepth>
struct MDWFTwoGridSiteCoord {   // global coordinates of even site i
    int *_out;
    __host__ __device__ void operator()(size_t i) {
        typedef GIndexer<Even, HaloDepth> GInd;
        const gSite site = GInd::getSite(i);
        LatticeData lat = GInd::getLatData();
        const sitexyzt g = lat.globalPos(site.coord);
        _out[4 * i] = g.x;
        _out[4 * i + 1] = g.y;
        _out[4 * i + 2] = g.z;
        _out[4 * i + 3] = g.t;
    }
};

#define MDWF_TWOGRID_THREADS 128

#ifdef __GPUCC__
// out[(bc * ns + k)] = <W_k|v> over block b = bc / 2, chirality chi = bc % 2 (all s). Grid (ns, 2 nBlocks).
__global__ void __launch_bounds__(MDWF_TWOGRID_THREADS)
    mdwfTwoGridRestrictKernel(const double *const *W, const double *v, const int *blockSites, int sitesPerBlock,
                              size_t sizeh, int Ls, int ns, double *out) {
    const int k = blockIdx.x;
    const int bc = blockIdx.y;
    const int b = bc >> 1, chi = bc & 1;
    const double *w = W[k];
    const int total = sitesPerBlock * Ls * 6;
    double re = 0.0, im = 0.0;
    for (int m = threadIdx.x; m < total; m += blockDim.x) {
        const int c = m % 6;
        const int rest = m / 6;
        const int s = rest % Ls;
        const int j = rest / Ls;
        const size_t i = static_cast<size_t>(blockSites[b * sitesPerBlock + j]) + sizeh * static_cast<size_t>(s);
        const size_t off = 24 * i + 2 * (6 * chi + c);
        const double ar = w[off], ai = w[off + 1], br = v[off], bi = v[off + 1];
        re += ar * br + ai * bi;
        im += ar * bi - ai * br;
    }
    __shared__ double sre[MDWF_TWOGRID_THREADS], sim[MDWF_TWOGRID_THREADS];
    sre[threadIdx.x] = re;
    sim[threadIdx.x] = im;
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) {
        if (static_cast<int>(threadIdx.x) < h) {
            sre[threadIdx.x] += sre[threadIdx.x + h];
            sim[threadIdx.x] += sim[threadIdx.x + h];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const size_t o = static_cast<size_t>(bc) * ns + k;
        out[2 * o] = sre[0];
        out[2 * o + 1] = sim[0];
    }
}

// out (+)= P c: thread t = (site-stack i = t / 2, chirality t % 2).
__global__ void __launch_bounds__(MDWF_TWOGRID_THREADS)
    mdwfTwoGridProlongKernel(const double *const *W, const double *c, const int *blockOfSite, size_t sizeh, size_t n,
                             int ns, double *out, bool accumulate) {
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= 2 * n) {
        return;
    }
    const size_t i = t >> 1;
    const int chi = static_cast<int>(t & 1);
    const int b = blockOfSite[i % sizeh];
    const double *cc = c + 2 * (static_cast<size_t>(b * 2 + chi) * ns);
    double acc[12];
    double *o = out + 24 * i + 12 * chi;
    for (int q = 0; q < 12; q++) {
        acc[q] = accumulate ? o[q] : 0.0;
    }
    for (int k = 0; k < ns; k++) {
        const double cr = cc[2 * k], ci = cc[2 * k + 1];
        const double *w = W[k] + 24 * i + 12 * chi;
        for (int q = 0; q < 6; q++) {
            const double wr = w[2 * q], wi = w[2 * q + 1];
            acc[2 * q] += cr * wr - ci * wi;
            acc[2 * q + 1] += cr * wi + ci * wr;
        }
    }
    for (int q = 0; q < 12; q++) {
        o[q] = acc[q];
    }
}
#endif

inline void mdwfTwoGridCheck(gpuError_t err, const char *what) {
    if (err) {
        GpuError(what, err);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Dense coarse solver
// ---------------------------------------------------------------------------------------------------------------

class MDWFTwoGridDense {
    int _n = 0;
#ifdef MDWF_TWOGRID_CUSOLVER
    cusolverDnHandle_t _handle = nullptr;
    cuDoubleComplex *_a = nullptr, *_work = nullptr;
    int *_info = nullptr;
    int _lwork = 0;

    static void check(cusolverStatus_t s, const char *what) {
        if (s != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error(stdLogger.fatal("MDWF two-grid: cuSOLVER ", what, " failed, status ",
                                                     static_cast<int>(s)));
        }
    }

    void release() {
        if (_a) { gpuFree(_a); _a = nullptr; }
        if (_work) { gpuFree(_work); _work = nullptr; }
        if (_info) { gpuFree(_info); _info = nullptr; }
    }
#endif

public:
    MDWFTwoGridDense() {
#ifdef MDWF_TWOGRID_CUSOLVER
        check(cusolverDnCreate(&_handle), "create");
#endif
    }

    ~MDWFTwoGridDense() {
#ifdef MDWF_TWOGRID_CUSOLVER
        release();
        if (_handle) {
            cusolverDnDestroy(_handle);
        }
#endif
    }

    MDWFTwoGridDense(const MDWFTwoGridDense &) = delete;

    // a: column-major n x n Hermitian positive definite matrix (host). Returns seconds.
    double factor(const std::vector<std::complex<double>> &a, int n) {
#ifdef MDWF_TWOGRID_CUSOLVER
        const auto t0 = std::chrono::steady_clock::now();
        release();
        _n = n;
        const size_t bytes = sizeof(cuDoubleComplex) * static_cast<size_t>(n) * n;
        mdwfTwoGridCheck(gpuMalloc((void **) &_a, bytes), "MDWF two-grid: coarse matrix allocation");
        mdwfTwoGridCheck(gpuMalloc((void **) &_info, sizeof(int)), "MDWF two-grid: info allocation");
        mdwfTwoGridCheck(gpuMemcpy(_a, a.data(), bytes, gpuMemcpyHostToDevice), "MDWF two-grid: coarse matrix copy");
        check(cusolverDnZpotrf_bufferSize(_handle, CUBLAS_FILL_MODE_LOWER, n, _a, n, &_lwork), "potrf buffer");
        mdwfTwoGridCheck(gpuMalloc((void **) &_work, sizeof(cuDoubleComplex) * std::max(_lwork, 1)),
                         "MDWF two-grid: workspace allocation");
        check(cusolverDnZpotrf(_handle, CUBLAS_FILL_MODE_LOWER, n, _a, n, _work, _lwork, _info), "potrf");
        int info = 0;
        mdwfTwoGridCheck(gpuMemcpy(&info, _info, sizeof(int), gpuMemcpyDeviceToHost), "MDWF two-grid: info copy");
        if (info != 0) {
            throw std::runtime_error(stdLogger.fatal("MDWF two-grid: Cholesky of A_c failed, info = ", info,
                                                     " (A_c not positive definite)"));
        }
        gpuDeviceSynchronize();
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
#else
        (void)a; (void)n;
        throw std::runtime_error("MDWF two-grid: dense coarse solver needs MDWF_TWOGRID_CUSOLVER (mdwfTwoGridStudy)");
#endif
    }

    // In place: x <- A_c^-1 x (device vector of n complex doubles).
    void solve(double *x) {
#ifdef MDWF_TWOGRID_CUSOLVER
        check(cusolverDnZpotrs(_handle, CUBLAS_FILL_MODE_LOWER, _n, 1, _a, _n, reinterpret_cast<cuDoubleComplex *>(x),
                               _n, _info),
              "potrs");
#else
        (void)x;
        throw std::runtime_error("MDWF two-grid: dense coarse solver needs MDWF_TWOGRID_CUSOLVER");
#endif
    }
};

// ---------------------------------------------------------------------------------------------------------------
// Block layout and coarse space
// ---------------------------------------------------------------------------------------------------------------

struct MDWFTwoGridStats {
    long setupApplications = 0;    // near-null vector generation
    long probeApplications = 0;    // assembly of A_c
    double setupSeconds = 0.0, orthoSeconds = 0.0, probeSeconds = 0.0, factorSeconds = 0.0;
    double hermiticity = 0.0;      // max |A_c - A_c^+| / max |A_c|
    double orthoDefect = 0.0;      // max |G - 1| per block after orthonormalisation
};

template<size_t HaloDepth, size_t Ls>
class MDWFTwoGrid {
public:
    using Vec = MDWFCompactVec<HaloDepth, Ls>;
    using Set = MDWFDeflSet<HaloDepth, Ls>;
    using Op = MDWFDeflOperator<HaloDepth, Ls>;
    using C = std::complex<double>;

private:
    CommunicationBase &_comm;
    int _block[4];
    int _nb[4];
    int _nBlocks = 0, _sitesPerBlock = 0, _ns = 0;
    size_t _sizeh;
    std::vector<int> _blockOfSite;       // host copy
    std::vector<int> _coarseOfBlock;     // block -> (bx, by, bz, bt), 4 per block
    gMemoryPtr<true> _dBlockOfSite, _dBlockSites, _dWptr, _dC1, _dC2;
    Set _W;
    MDWFTwoGridDense _dense;
    std::unique_ptr<Vec> _t1, _t2, _t3, _t4, _t5;   // work vectors (_t2.._t4: smoother, _t1, _t5: cycle)
    double _lo = 0.0, _hi = 0.0;
    int _degree = 1;
    // profile (study): seconds per part of the cycle, with a device synchronisation around each part
    bool _profile = false;
    double _tSmooth = 0.0, _tApply = 0.0, _tCoarse = 0.0, _tVector = 0.0;
    std::chrono::steady_clock::time_point _tMark;

    void mark() {
        if (_profile) {
            gpuDeviceSynchronize();
            _tMark = std::chrono::steady_clock::now();
        }
    }

    void lap(double &acc) {
        if (_profile) {
            gpuDeviceSynchronize();
            const auto now = std::chrono::steady_clock::now();
            acc += std::chrono::duration<double>(now - _tMark).count();
            _tMark = now;
        }
    }

    static std::string uname(const char *w) {
        return mdwfDeflUniqueName(w);
    }

    int blockIndex(int bx, int by, int bz, int bt) const {
        return bx + _nb[0] * (by + _nb[1] * (bz + _nb[2] * bt));
    }

    void uploadPointers() {
        std::vector<const double *> p(_ns);
        for (int k = 0; k < _ns; k++) {
            p[k] = _W[k]->ptr();
        }
        _dWptr->template adjustSize<const double *>(std::max(_ns, 1));
        mdwfTwoGridCheck(gpuMemcpy(_dWptr->template getPointer<const double *>(), p.data(), sizeof(double *) * _ns,
                                   gpuMemcpyHostToDevice),
                         "MDWF two-grid: basis pointer copy");
    }

    // c (device, coarse) <- R v using basis pointers Wp
    void restrictWith(const double *const *Wp, int ns, double *c, const Vec &v) const {
#ifdef __GPUCC__
        dim3 grid(static_cast<unsigned>(ns), static_cast<unsigned>(2 * _nBlocks));
        mdwfTwoGridRestrictKernel<<<grid, MDWF_TWOGRID_THREADS>>>(
            Wp, v.ptr(), _dBlockSites->template getPointer<int>(), _sitesPerBlock, _sizeh, static_cast<int>(Ls), ns, c);
        mdwfTwoGridCheck(gpuGetLastError(), "MDWF two-grid: restrict kernel");
#else
        (void)Wp; (void)ns; (void)c; (void)v;
        throw std::runtime_error("MDWF two-grid: GPU build only");
#endif
    }

    void prolongWith(const double *const *Wp, int ns, Vec &out, const double *c, bool accumulate) const {
#ifdef __GPUCC__
        const size_t n = out.siteStacks();
        const unsigned blocks = static_cast<unsigned>((2 * n + MDWF_TWOGRID_THREADS - 1) / MDWF_TWOGRID_THREADS);
        mdwfTwoGridProlongKernel<<<blocks, MDWF_TWOGRID_THREADS>>>(
            Wp, c, _dBlockOfSite->template getPointer<int>(), _sizeh, n, ns, out.ptr(), accumulate);
        mdwfTwoGridCheck(gpuGetLastError(), "MDWF two-grid: prolong kernel");
#else
        (void)Wp; (void)ns; (void)out; (void)c; (void)accumulate;
        throw std::runtime_error("MDWF two-grid: GPU build only");
#endif
    }

    const double *const *wptr() const {
        return _dWptr->template getPointer<const double *>();
    }

public:
    // block: extent per direction (x, y, z, t); must divide the (single-rank) lattice.
    MDWFTwoGrid(CommunicationBase &comm, const std::vector<int> &block)
        : _comm(comm), _sizeh(GIndexer<Even, HaloDepth>::getLatData().sizeh),
          _dBlockOfSite(MemoryManagement::getMemAt<true>(uname("tgbos"))),
          _dBlockSites(MemoryManagement::getMemAt<true>(uname("tgbs"))),
          _dWptr(MemoryManagement::getMemAt<true>(uname("tgwp"))),
          _dC1(MemoryManagement::getMemAt<true>(uname("tgc1"))),
          _dC2(MemoryManagement::getMemAt<true>(uname("tgc2"))) {
        if (comm.getNumberProcesses() != 1) {
            throw std::runtime_error(stdLogger.fatal("MDWF two-grid: single rank only (Stage 0)"));
        }
        if (block.size() != 4) {
            throw std::runtime_error(stdLogger.fatal("MDWF two-grid: block needs 4 extents"));
        }
        const LatticeData lat = GIndexer<All, HaloDepth>::getLatData();
        const int L[4] = {static_cast<int>(lat.globLX), static_cast<int>(lat.globLY), static_cast<int>(lat.globLZ),
                          static_cast<int>(lat.globLT)};
        _nBlocks = 1;
        for (int mu = 0; mu < 4; mu++) {
            _block[mu] = block[mu];
            if (_block[mu] < 1 || L[mu] % _block[mu] != 0) {
                throw std::runtime_error(stdLogger.fatal("MDWF two-grid: block extent ", _block[mu],
                                                         " does not divide L = ", L[mu], " in direction ", mu));
            }
            _nb[mu] = L[mu] / _block[mu];
            if (_nb[mu] > 1 && _block[mu] < 4) {
                throw std::runtime_error(stdLogger.fatal("MDWF two-grid: block extent ", _block[mu], " < 4 in direction ",
                                                         mu, ": A couples even sites up to distance 4"));
            }
            _nBlocks *= _nb[mu];
        }
        if (2 * _nBlocks > 65535) {
            throw std::runtime_error(stdLogger.fatal("MDWF two-grid: too many blocks (", _nBlocks, ")"));
        }
        // coordinates of the even sites
        gMemoryPtr<true> dCoord = MemoryManagement::getMemAt<true>(uname("tgcoord"));
        dCoord->template adjustSize<int>(4 * _sizeh);
        iterateFunctorNoReturn<true>(MDWFTwoGridSiteCoord<HaloDepth>{dCoord->template getPointer<int>()},
                                     MDWFFlatIndex(), _sizeh);
        std::vector<int> coord(4 * _sizeh);
        mdwfTwoGridCheck(gpuMemcpy(coord.data(), dCoord->template getPointer<int>(), sizeof(int) * 4 * _sizeh,
                                   gpuMemcpyDeviceToHost),
                         "MDWF two-grid: coordinate copy");
        _blockOfSite.resize(_sizeh);
        std::vector<int> count(_nBlocks, 0);
        for (size_t i = 0; i < _sizeh; i++) {
            const int *x = &coord[4 * i];
            const int b = blockIndex(x[0] / _block[0], x[1] / _block[1], x[2] / _block[2], x[3] / _block[3]);
            _blockOfSite[i] = b;
            count[b]++;
        }
        _sitesPerBlock = count[0];
        for (int b = 0; b < _nBlocks; b++) {
            if (count[b] != _sitesPerBlock) {
                throw std::runtime_error(stdLogger.fatal("MDWF two-grid: blocks hold different numbers of even sites"));
            }
        }
        std::vector<int> sites(static_cast<size_t>(_nBlocks) * _sitesPerBlock);
        std::fill(count.begin(), count.end(), 0);
        for (size_t i = 0; i < _sizeh; i++) {
            const int b = _blockOfSite[i];
            sites[static_cast<size_t>(b) * _sitesPerBlock + count[b]++] = static_cast<int>(i);
        }
        _dBlockOfSite->template adjustSize<int>(_sizeh);
        _dBlockSites->template adjustSize<int>(sites.size());
        mdwfTwoGridCheck(gpuMemcpy(_dBlockOfSite->template getPointer<int>(), _blockOfSite.data(), sizeof(int) * _sizeh,
                                   gpuMemcpyHostToDevice),
                         "MDWF two-grid: block map copy");
        mdwfTwoGridCheck(gpuMemcpy(_dBlockSites->template getPointer<int>(), sites.data(), sizeof(int) * sites.size(),
                                   gpuMemcpyHostToDevice),
                         "MDWF two-grid: block site list copy");
        _coarseOfBlock.resize(4 * _nBlocks);
        for (int bt = 0; bt < _nb[3]; bt++)
            for (int bz = 0; bz < _nb[2]; bz++)
                for (int by = 0; by < _nb[1]; by++)
                    for (int bx = 0; bx < _nb[0]; bx++) {
                        const int b = blockIndex(bx, by, bz, bt);
                        _coarseOfBlock[4 * b] = bx;
                        _coarseOfBlock[4 * b + 1] = by;
                        _coarseOfBlock[4 * b + 2] = bz;
                        _coarseOfBlock[4 * b + 3] = bt;
                    }
        rootLogger.info("MDWF two-grid: blocks ", _block[0], "x", _block[1], "x", _block[2], "x", _block[3], " x Ls ",
                        Ls, ", ", _nBlocks, " blocks (", _nb[0], "x", _nb[1], "x", _nb[2], "x", _nb[3], "), ",
                        _sitesPerBlock, " even sites per block");
    }

    int nBlocks() const { return _nBlocks; }
    int ns() const { return _ns; }
    int coarseDim() const { return 2 * _nBlocks * _ns; }
    int blockDim() const { return _sitesPerBlock * static_cast<int>(Ls) * 6; }   // per chirality

    // Device coarse work vectors (2 x coarseDim doubles each).
    double *c1() { return _dC1->template getPointer<double>(); }
    double *c2() { return _dC2->template getPointer<double>(); }

    // Takes the raw near-null vectors (moved in) and orthonormalises them on every (block, chirality), twice
    // (Cholesky QR2). Afterwards R P = 1.
    void setBasis(Set &raw, int ns, MDWFTwoGridStats &stats) {
        const auto t0 = std::chrono::steady_clock::now();
        if (ns < 1 || ns > static_cast<int>(raw.size())) {
            throw std::runtime_error(stdLogger.fatal("MDWF two-grid: ns = ", ns, " but ", raw.size(), " raw vectors"));
        }
        if (ns > blockDim()) {
            throw std::runtime_error(stdLogger.fatal("MDWF two-grid: ns = ", ns, " exceeds the block dimension ",
                                                     blockDim()));
        }
        _W.clear();
        for (int k = 0; k < ns; k++) {
            _W.emplace_back(std::move(raw[k]));
        }
        raw.erase(raw.begin(), raw.begin() + ns);
        _ns = ns;
        const size_t nc = static_cast<size_t>(coarseDim());
        _dC1->template adjustSize<double>(2 * nc);
        _dC2->template adjustSize<double>(2 * nc);
        uploadPointers();
        for (int pass = 0; pass < 2; pass++) {
            orthonormalise(stats, pass == 1);
        }
        stats.orthoSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (!_t1) {
            _t1.reset(new Vec(_comm));
            _t2.reset(new Vec(_comm));
            _t3.reset(new Vec(_comm));
            _t4.reset(new Vec(_comm));
            _t5.reset(new Vec(_comm));
        }
    }

private:
    // Gram matrices G_(b,chi)[k][l] = <W_k|W_l>_(b,chi); W <- W L^-+ per (b, chi).
    void orthonormalise(MDWFTwoGridStats &stats, bool measureDefect) {
        const int ns = _ns;
        const size_t nc = static_cast<size_t>(coarseDim());
        const int nbc = 2 * _nBlocks;
        std::vector<C> gram(static_cast<size_t>(nbc) * ns * ns);
        std::vector<C> col(nc);
        for (int l = 0; l < ns; l++) {
            restrictWith(wptr(), ns, c1(), *_W[l]);
            mdwfTwoGridCheck(gpuMemcpy(col.data(), c1(), sizeof(C) * nc, gpuMemcpyDeviceToHost),
                             "MDWF two-grid: Gram copy");
            for (int bc = 0; bc < nbc; bc++) {
                for (int k = 0; k < ns; k++) {
                    gram[(static_cast<size_t>(bc) * ns + k) * ns + l] = col[static_cast<size_t>(bc) * ns + k];
                }
            }
        }
        double defect = 0.0;
        // coefficient vectors: coef[k][(bc, l)] = T_bc[k][l] = conj((L^-1)_{k l})
        std::vector<std::vector<C>> coef(ns, std::vector<C>(nc, C(0.0, 0.0)));
        int fallback = 0;
        for (int bc = 0; bc < nbc; bc++) {
            std::vector<C> g(gram.begin() + static_cast<size_t>(bc) * ns * ns,
                             gram.begin() + static_cast<size_t>(bc + 1) * ns * ns);
            for (int k = 0; k < ns; k++) {
                for (int l = 0; l < ns; l++) {
                    defect = std::max(defect, std::abs(g[k * ns + l] - (k == l ? C(1.0, 0.0) : C(0.0, 0.0))));
                }
            }
            // Cholesky G = L L^+ (lower), with a tiny shift if G is numerically singular
            std::vector<C> Lm(static_cast<size_t>(ns) * ns, C(0.0, 0.0));
            double shift = 0.0;
            for (int attempt = 0; attempt < 6; attempt++) {
                bool ok = true;
                std::fill(Lm.begin(), Lm.end(), C(0.0, 0.0));
                for (int j = 0; j < ns && ok; j++) {
                    double d = std::real(g[j * ns + j]) + shift;
                    for (int q = 0; q < j; q++) {
                        d -= std::norm(Lm[j * ns + q]);
                    }
                    if (!(d > 0.0)) {
                        ok = false;
                        break;
                    }
                    Lm[j * ns + j] = std::sqrt(d);
                    for (int i = j + 1; i < ns; i++) {
                        C s = g[i * ns + j];
                        for (int q = 0; q < j; q++) {
                            s -= Lm[i * ns + q] * std::conj(Lm[j * ns + q]);
                        }
                        Lm[i * ns + j] = s / Lm[j * ns + j];
                    }
                }
                if (ok) {
                    break;
                }
                shift = (shift == 0.0) ? 1e-14 * std::real(g[0]) + 1e-300 : shift * 100.0;
                fallback++;
                if (attempt == 5) {
                    throw std::runtime_error(stdLogger.fatal("MDWF two-grid: Gram matrix of block ", bc / 2,
                                                             " chirality ", bc % 2, " not positive definite"));
                }
            }
            // Linv = L^-1 (lower triangular)
            std::vector<C> Linv(static_cast<size_t>(ns) * ns, C(0.0, 0.0));
            for (int j = 0; j < ns; j++) {
                Linv[j * ns + j] = 1.0 / Lm[j * ns + j];
                for (int i = j + 1; i < ns; i++) {
                    C s(0.0, 0.0);
                    for (int q = j; q < i; q++) {
                        s -= Lm[i * ns + q] * Linv[q * ns + j];
                    }
                    Linv[i * ns + j] = s / Lm[i * ns + i];
                }
            }
            for (int k = 0; k < ns; k++) {
                for (int l = 0; l <= k; l++) {
                    coef[k][static_cast<size_t>(bc) * ns + l] = std::conj(Linv[k * ns + l]);
                }
            }
        }
        if (measureDefect) {
            stats.orthoDefect = defect;
        }
        if (fallback > 0) {
            rootLogger.info("MDWF two-grid: ", fallback, " shifted Cholesky factorisations of block Gram matrices");
        }
        // W'_k = P(coef_k) built from the old W into new vectors
        Set fresh;
        for (int k = 0; k < ns; k++) {
            fresh.emplace_back(new Vec(_comm));
            mdwfTwoGridCheck(gpuMemcpy(c1(), coef[k].data(), sizeof(C) * nc, gpuMemcpyHostToDevice),
                             "MDWF two-grid: coefficient copy");
            prolongWith(wptr(), ns, *fresh[k], c1(), false);
        }
        gpuDeviceSynchronize();
        _W.swap(fresh);
        uploadPointers();
    }

public:
    // c <- R v (device coarse vector)
    void restrict_(double *c, const Vec &v) const {
        restrictWith(wptr(), _ns, c, v);
    }

    // out (+)= P c
    void prolong(Vec &out, const double *c, bool accumulate) const {
        prolongWith(wptr(), _ns, out, c, accumulate);
    }

    // A_c = R A P by probing; dense factorisation. Applications of A counted in stats.probeApplications.
    // keepMatrix: host copy of A_c returned in *matrix (tests).
    void buildCoarse(Op &op, MDWFTwoGridStats &stats, std::vector<C> *matrix = nullptr) {
        const auto t0 = std::chrono::steady_clock::now();
        const long a0 = op.applications();
        const int n = coarseDim();
        int ncol[4], ncolors = 1;
        for (int mu = 0; mu < 4; mu++) {
            ncol[mu] = (_nb[mu] >= 4 && _nb[mu] % 4 == 0) ? 4 : _nb[mu];
            ncolors *= ncol[mu];
        }
        auto colorOf = [&](int b) {
            const int *x = &_coarseOfBlock[4 * b];
            return (x[0] % ncol[0]) + ncol[0] * ((x[1] % ncol[1]) + ncol[1] * ((x[2] % ncol[2]) + ncol[2] * (x[3] % ncol[3])));
        };
        // source[b' * ncolors + col] = the block of colour col within reach (offsets -1, 0, 1) of b', or -1
        std::vector<int> source(static_cast<size_t>(_nBlocks) * ncolors, -1);
        for (int bp = 0; bp < _nBlocks; bp++) {
            const int *x = &_coarseOfBlock[4 * bp];
            for (int o = 0; o < 81; o++) {
                int y[4], q = o;
                for (int mu = 0; mu < 4; mu++) {
                    y[mu] = ((x[mu] + (q % 3) - 1) % _nb[mu] + _nb[mu]) % _nb[mu];
                    q /= 3;
                }
                const int b = blockIndex(y[0], y[1], y[2], y[3]);
                int &slot = source[static_cast<size_t>(bp) * ncolors + colorOf(b)];
                if (slot >= 0 && slot != b) {
                    throw std::runtime_error(stdLogger.fatal("MDWF two-grid: probing colours not separated"));
                }
                slot = b;
            }
        }
        std::vector<C> a(static_cast<size_t>(n) * n, C(0.0, 0.0));
        std::vector<C> e(n), r(n);
        Vec &v = *_t1, &av = *_t2;
        for (int col = 0; col < ncolors; col++) {
            for (int chi = 0; chi < 2; chi++) {
                for (int k = 0; k < _ns; k++) {
                    std::fill(e.begin(), e.end(), C(0.0, 0.0));
                    for (int b = 0; b < _nBlocks; b++) {
                        if (colorOf(b) == col) {
                            e[static_cast<size_t>(b * 2 + chi) * _ns + k] = 1.0;
                        }
                    }
                    mdwfTwoGridCheck(gpuMemcpy(c1(), e.data(), sizeof(C) * n, gpuMemcpyHostToDevice),
                                     "MDWF two-grid: probe copy");
                    prolong(v, c1(), false);
                    op.apply(av, v);
                    restrict_(c1(), av);
                    mdwfTwoGridCheck(gpuMemcpy(r.data(), c1(), sizeof(C) * n, gpuMemcpyDeviceToHost),
                                     "MDWF two-grid: probe result copy");
                    for (int bp = 0; bp < _nBlocks; bp++) {
                        const int b = source[static_cast<size_t>(bp) * ncolors + col];
                        if (b < 0) {
                            continue;
                        }
                        const size_t J = static_cast<size_t>(b * 2 + chi) * _ns + k;
                        for (int q = 0; q < 2 * _ns; q++) {
                            const size_t I = static_cast<size_t>(bp * 2) * _ns + q;
                            a[I + J * n] = r[I];
                        }
                    }
                }
            }
        }
        stats.probeApplications += op.applications() - a0;
        double amax = 0.0, dev = 0.0;
        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) {
                const C x = a[i + static_cast<size_t>(j) * n], y = a[j + static_cast<size_t>(i) * n];
                amax = std::max(amax, std::abs(x));
                dev = std::max(dev, std::abs(x - std::conj(y)));
            }
        }
        stats.hermiticity = amax > 0.0 ? dev / amax : 0.0;
        for (int j = 0; j < n; j++) {   // exact Hermitian part
            for (int i = 0; i < j; i++) {
                const C h = 0.5 * (a[i + static_cast<size_t>(j) * n] + std::conj(a[j + static_cast<size_t>(i) * n]));
                a[i + static_cast<size_t>(j) * n] = h;
                a[j + static_cast<size_t>(i) * n] = std::conj(h);
            }
            a[j + static_cast<size_t>(j) * n] = C(std::real(a[j + static_cast<size_t>(j) * n]), 0.0);
        }
        stats.probeSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        rootLogger.info("MDWF two-grid: A_c (dimension ", n, ") from ", ncolors * 2 * _ns, " probes (", ncolors,
                        " colours), Hermiticity defect ", stats.hermiticity);
        stats.factorSeconds += _dense.factor(a, n);
        if (matrix != nullptr) {
            matrix->swap(a);
        }
    }

    // Factorises a given A_c (host, column-major, coarseDim()^2), e.g. one kept from buildCoarse.
    void setCoarse(const std::vector<C> &a) {
        if (a.size() != static_cast<size_t>(coarseDim()) * coarseDim()) {
            throw std::runtime_error(stdLogger.fatal("MDWF two-grid: setCoarse: wrong matrix size"));
        }
        _dense.factor(a, coarseDim());
    }

    void setSmoother(double lo, double hi, int degree) {
        if (!(hi > lo && lo > 0.0) || degree < 1) {
            throw std::runtime_error(stdLogger.fatal("MDWF two-grid: smoother needs 0 < lo < hi and degree >= 1"));
        }
        _lo = lo;
        _hi = hi;
        _degree = degree;
    }

    // x = S r: degree-d Chebyshev iteration for A x = r from x = 0 on [lo, hi] (d - 1 applications of A).
    void smooth(Op &op, Vec &x, const Vec &r) {
        Vec &res = *_t3, &dv = *_t4, &ad = *_t2;
        const double theta = 0.5 * (_hi + _lo), delta = 0.5 * (_hi - _lo), sigma = theta / delta;
        double rho = 1.0 / sigma;
        res = r;
        dv = r;
        dv *= COMPLEX(double)(1.0 / theta, 0.0);
        x = dv;
        for (int k = 1; k < _degree; k++) {
            op.apply(ad, dv);
            res.axpy(MDWFDeflC(-1.0, 0.0), ad);
            const double rhoNew = 1.0 / (2.0 * sigma - rho);
            dv *= COMPLEX(double)(rhoNew * rho, 0.0);
            dv.axpy(MDWFDeflC(2.0 * rhoNew / delta, 0.0), res);
            x.axpy(MDWFDeflC(1.0, 0.0), dv);
            rho = rhoNew;
        }
    }

    // Profile of precondition(): smoother, residual applications of A, coarse correction (restrict + solve +
    // prolong), vector updates. Study only (adds device synchronisations).
    void setProfile(bool on) { _profile = on; }
    void resetProfile() { _tSmooth = _tApply = _tCoarse = _tVector = 0.0; }
    double profSmooth() const { return _tSmooth; }
    double profApply() const { return _tApply; }
    double profCoarse() const { return _tCoarse; }
    double profVector() const { return _tVector; }

    // z = M r (symmetric two-grid cycle; 2 d applications of A).
    void precondition(Op &op, Vec &z, const Vec &r) {
        Vec &t = *_t1, &y = *_t2;
        mark();
        smooth(op, z, r);
        lap(_tSmooth);
        op.apply(y, z);
        lap(_tApply);
        t = r;
        t.axpy(MDWFDeflC(-1.0, 0.0), y);
        lap(_tVector);
        restrict_(c1(), t);
        _dense.solve(c1());
        prolong(z, c1(), true);
        lap(_tCoarse);
        op.apply(y, z);
        lap(_tApply);
        t = r;
        t.axpy(MDWFDeflC(-1.0, 0.0), y);
        lap(_tVector);
        // post-smoothing (smooth works in _t2.._t4; t is _t1, the result goes to _t5)
        Vec &post = *_t5;
        smooth(op, post, t);
        lap(_tSmooth);
        z.axpy(MDWFDeflC(1.0, 0.0), post);
        lap(_tVector);
    }

    // Coarse correction only: z = P A_c^-1 R r (no application of A).
    void coarseCorrection(Vec &z, const Vec &r) {
        restrict_(c1(), r);
        _dense.solve(c1());
        prolong(z, c1(), false);
    }

    const Vec &basis(int k) const {
        return *_W[k];
    }
};

// ---------------------------------------------------------------------------------------------------------------
// Krylov solvers with the production stopping rule
// ---------------------------------------------------------------------------------------------------------------

struct MDWFTwoGridSolveResult {
    int iterations = 0;          // outer iterations
    long applications = 0;       // applications of A (fine matvecs), everything included
    double seconds = 0.0;
    double trueResidual = 0.0;   // |b - A x| / max(|b|, 1)
    bool converged = false;
};

// Plain CG (x = 0 start).
template<class Op, class Vec>
MDWFTwoGridSolveResult mdwfTwoGridPlainCG(Op &op, Vec &x, const Vec &b, double precision, int maxIter,
                                          CommunicationBase &comm) {
    const auto t0 = std::chrono::steady_clock::now();
    const long a0 = op.applications();
    Vec r(comm), p(comm), ap(comm);
    const double bnorm = b.norm2();
    const double target = precision * precision * std::max(bnorm, 1.0);
    x.zero();
    r = b;
    p = b;
    double rr = bnorm;
    MDWFTwoGridSolveResult res;
    for (int it = 0; it < maxIter; it++) {
        op.apply(ap, p);
        const double pap = real<double>(p.dot(ap));
        const double alpha = rr / pap;
        x.axpy(MDWFDeflC(alpha, 0.0), p);
        r.axpy(MDWFDeflC(-alpha, 0.0), ap);
        const double rrNew = r.norm2();
        res.iterations = it + 1;
        if (rrNew <= target) {
            op.apply(ap, x);
            r = b;
            r.axpy(MDWFDeflC(-1.0, 0.0), ap);
            const double t = r.norm2();
            if (t <= target) {
                res.converged = true;
                break;
            }
            p = r;          // restart from the true residual
            rr = t;
            continue;
        }
        p.xpay(rrNew / rr, r);
        rr = rrNew;
    }
    op.apply(ap, x);
    r = b;
    r.axpy(MDWFDeflC(-1.0, 0.0), ap);
    res.trueResidual = std::sqrt(r.norm2() / std::max(bnorm, 1.0));
    res.applications = op.applications() - a0;
    res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return res;
}

// Flexible CG (truncation 1) preconditioned by the two-grid cycle.
template<size_t HaloDepth, size_t Ls>
MDWFTwoGridSolveResult mdwfTwoGridFCG(MDWFTwoGrid<HaloDepth, Ls> &tg, typename MDWFTwoGrid<HaloDepth, Ls>::Op &op,
                                      MDWFCompactVec<HaloDepth, Ls> &x, const MDWFCompactVec<HaloDepth, Ls> &b,
                                      double precision, int maxIter, CommunicationBase &comm) {
    using Vec = MDWFCompactVec<HaloDepth, Ls>;
    const auto t0 = std::chrono::steady_clock::now();
    const long a0 = op.applications();
    Vec r(comm), z(comm), p(comm), q(comm);
    const double bnorm = b.norm2();
    const double target = precision * precision * std::max(bnorm, 1.0);
    x.zero();
    r = b;
    tg.precondition(op, z, r);
    p = z;
    MDWFTwoGridSolveResult res;
    for (int it = 0; it < maxIter; it++) {
        op.apply(q, p);
        const double pq = real<double>(p.dot(q));
        const COMPLEX(double) pr = p.dot(r);
        const MDWFDeflC alpha(real<double>(pr) / pq, imag<double>(pr) / pq);
        x.axpy(alpha, p);
        r.axpy(-alpha, q);
        res.iterations = it + 1;
        if (r.norm2() <= target) {
            op.apply(q, x);
            r = b;
            r.axpy(MDWFDeflC(-1.0, 0.0), q);
            if (r.norm2() <= target) {
                res.converged = true;
                break;
            }
            tg.precondition(op, z, r);   // restart from the true residual
            p = z;
            continue;
        }
        tg.precondition(op, z, r);
        const COMPLEX(double) qz = q.dot(z);
        const COMPLEX(double) beta(-real<double>(qz) / pq, -imag<double>(qz) / pq);
        p *= beta;
        p.axpy(MDWFDeflC(1.0, 0.0), z);
    }
    op.apply(q, x);
    r = b;
    r.axpy(MDWFDeflC(-1.0, 0.0), q);
    res.trueResidual = std::sqrt(r.norm2() / std::max(bnorm, 1.0));
    res.applications = op.applications() - a0;
    res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return res;
}
