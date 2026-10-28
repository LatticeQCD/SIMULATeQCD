/*
 * cpuSpinorKernels.h
 *
 * Fused vector kernels of the multi-shift CG for the CPU backend. Each replaces a sequence of
 * spinor kernels and reductions by one OpenMP pass over the flat colour arrays, without per-site
 * index arithmetic:
 *   shiftedRealDot:    s = shift p - s, then Re(p.s)              (operator=, realdotProduct)
 *   axpyNorm:          r += a s, then |r|^2                        (axpyThisB, realdotProduct)
 *   multiShiftUpdate:  x_j += alpha_j p_j, p_j = beta_j p_j + z_j r, and p0 = p_0 on the bulk
 *                      (MultiShiftUpdateNoReturn, copyFromStackToStackDevice)
 * and, for the stacked CG (invert_new, double), realDotStacked, axpyNormStacked and
 * cgUpdateStacked (x_j += alpha_j p_j, p_j = r_j + beta_j p_j).
 * The results are bit-identical to the unfused kernels: the per-site arithmetic is the same in the
 * same order, and sums over bulk sites are added in the same fixed 4096-site chunks as cpuSum
 * (latticeContainer.cpp) before the MPI reduction. SIMULATEQCD_CPU_FUSED=0 switches them off.
 */
#pragma once

#ifdef USE_CPU
#include "spinorfield.h"
#include "../base/wrapper/cpuProfile.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <string_view>
#include <type_traits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace cpuBlas {

inline bool fusedEnabled() {
    static const bool on = [] {
        const char *value = std::getenv("SIMULATEQCD_CPU_FUSED");
        return !value || std::string_view(value) != "0";
    }();
    return on;
}

//! The sites of one layout as runs of consecutive full indices, at most piece sites long: the bulk
//! runs in bulk-index order (bulk index, full index, length), then the halo sites (full, length).
struct Segments {
    static constexpr size_t piece = 256;
    size_t key[6] = {};
    size_t nBulk = 0, nFull = 0;
    std::vector<size_t> bulk, full, len;
    std::vector<size_t> haloFull, haloLen;
    std::vector<double> values;  // per bulk site, for the chunked sums
};

template<Layout LatLayout, size_t HaloDepth>
Segments &segments() {
    typedef GIndexer<LatLayout, HaloDepth> GInd;
    static Segments seg;
    const LatticeData lat = GInd::getLatData();
    const size_t key[6] = {lat.lx, lat.ly, lat.lz, lat.lt, lat.vol4Full, lat.sizehFull};
    if (std::equal(key, key + 6, seg.key) && seg.nBulk) return seg;

    seg = Segments();
    std::copy(key, key + 6, seg.key);
    seg.nBulk = LatLayout == All ? lat.vol4 : lat.sizeh;
    seg.nFull = LatLayout == All ? lat.vol4Full : lat.sizehFull;
    std::vector<char> isBulk(seg.nFull, 0);
    for (size_t i = 0; i < seg.nBulk; ++i) {
        const size_t f = GInd::getSite(i).isiteFull;
        if (f >= seg.nFull) throw std::runtime_error(stdLogger.fatal("cpuBlas::segments: full index out of range"));
        isBulk[f] = 1;
        if (!seg.len.empty() && seg.full.back() + seg.len.back() == f && seg.len.back() < Segments::piece) {
            seg.len.back()++;
        } else {
            seg.bulk.push_back(i);
            seg.full.push_back(f);
            seg.len.push_back(1);
        }
    }
    for (size_t f = 0; f < seg.nFull; ++f) {
        if (isBulk[f]) continue;
        if (!seg.haloLen.empty() && seg.haloFull.back() + seg.haloLen.back() == f
            && seg.haloLen.back() < Segments::piece) {
            seg.haloLen.back()++;
        } else {
            seg.haloFull.push_back(f);
            seg.haloLen.push_back(1);
        }
    }
    seg.values.resize(seg.nBulk);
    return seg;
}

//! This thread's share [begin, end) of n runs, contiguous as in a static schedule. Kernels over
//! several stacks loop over the stacks outside the runs, so every array is streamed over all runs
//! of the thread (with per-run streams the mixed multi-shift update was slower than unfused, job 3839433).
inline void threadRange(size_t n, size_t &begin, size_t &end) {
#ifdef _OPENMP
    const size_t t = omp_get_thread_num(), nt = omp_get_num_threads();
#else
    const size_t t = 0, nt = 1;
#endif
    begin = n * t / nt;
    end = n * (t + 1) / nt;
}

//! kernel(stack, full, n, values) on every bulk run of every stack < N writes the n site values;
//! returns the sum per stack, added as in cpuSum.
template<size_t N, class Kernel>
std::array<double, N> bulkSums(Segments &seg, Kernel kernel) {
    constexpr size_t chunk = 4096;
    const size_t nChunks = (seg.nBulk + chunk - 1) / chunk;
    if (seg.values.size() < N * seg.nBulk) seg.values.resize(N * seg.nBulk);
    std::vector<double> partial(N * nChunks);
    double *values = seg.values.data();
#pragma omp parallel
    {
        size_t begin, end;
        threadRange(seg.bulk.size(), begin, end);
        for (size_t j = 0; j < N; ++j) {
            for (size_t p = begin; p < end; ++p) kernel(j, seg.full[p], seg.len[p], values + j * seg.nBulk + seg.bulk[p]);
        }
#pragma omp barrier
#pragma omp for schedule(static)
        for (size_t jc = 0; jc < N * nChunks; ++jc) {
            const size_t j = jc / nChunks, c = jc % nChunks;
            const size_t end = (c + 1) * chunk < seg.nBulk ? (c + 1) * chunk : seg.nBulk;
            const double *v = values + j * seg.nBulk;
            double sum = 0.0;
            for (size_t i = c * chunk; i < end; ++i) sum += v[i];
            partial[jc] = sum;
        }
    }
    std::array<double, N> sums;
    for (size_t j = 0; j < N; ++j) {
        double sum = 0.0;
        for (size_t c = 0; c < nChunks; ++c) sum += partial[j * nChunks + c];
        sums[j] = sum;
    }
    return sums;
}

//! kernel(full, n, values) on every bulk run writes the n site values; returns their sum.
template<class Kernel>
double bulkSum(Segments &seg, Kernel kernel) {
    return bulkSums<1>(seg, [&](size_t, size_t f0, size_t n, double *values) { kernel(f0, n, values); })[0];
}

//! The colour arrays of a spinor as interleaved reals.
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
void colours(const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &spinor, floatT *c[3]) {
    const Vect3arrayAcc<floatT> acc = spinor.getAccessor();
    for (int k = 0; k < 3; ++k) c[k] = reinterpret_cast<floatT *>(acc.getEntryPointer(k));
}

//! s = shift * p - s on the bulk, then Re(p.s) (as s = shift * p - s; p.realdotProduct(s)).
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth>
double shiftedRealDot(Spinorfield<floatT, onDevice, LatLayout, HaloDepth, 1> &s,
                      const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, 1> &p, const floatT shift) {
    cpuProfile::Scope profile(std::is_same_v<floatT, float> ? "fused shiftedRealDot<float>" : "fused shiftedRealDot<double>");
    floatT *S[3], *P[3];
    colours(s, S);
    colours(p, P);
    const double local = bulkSum(segments<LatLayout, HaloDepth>(), [&](size_t f0, size_t n, double *values) {
        for (size_t i = 2 * f0; i < 2 * (f0 + n); ++i) {
            for (int k = 0; k < 3; ++k) S[k][i] = shift * P[k][i] - S[k][i];
        }
        for (size_t i = 0; i < n; ++i) {
            const size_t re = 2 * (f0 + i), im = re + 1;
            double v = (double)P[0][re] * (double)S[0][re] + (double)P[0][im] * (double)S[0][im];
            v += (double)P[1][re] * (double)S[1][re] + (double)P[1][im] * (double)S[1][im];
            v += (double)P[2][re] * (double)S[2][re] + (double)P[2][im] * (double)S[2][im];
            values[i] = v;
        }
    });
    return s.getComm().reduce(local);
}

//! r += a * s on the bulk, then |r|^2 (as r.axpyThisB(a, s); r.realdotProduct(r)).
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth>
double axpyNorm(Spinorfield<floatT, onDevice, LatLayout, HaloDepth, 1> &r, const floatT a,
                const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, 1> &s) {
    cpuProfile::Scope profile(std::is_same_v<floatT, float> ? "fused axpyNorm<float>" : "fused axpyNorm<double>");
    floatT *R[3], *S[3];
    colours(r, R);
    colours(s, S);
    const double local = bulkSum(segments<LatLayout, HaloDepth>(), [&](size_t f0, size_t n, double *values) {
        for (size_t i = 2 * f0; i < 2 * (f0 + n); ++i) {
            for (int k = 0; k < 3; ++k) R[k][i] = R[k][i] + a * S[k][i];
        }
        for (size_t i = 0; i < n; ++i) {
            const size_t re = 2 * (f0 + i), im = re + 1;
            double v = (double)R[0][re] * (double)R[0][re] + (double)R[0][im] * (double)R[0][im];
            v += (double)R[1][re] * (double)R[1][re] + (double)R[1][im] * (double)R[1][im];
            v += (double)R[2][re] * (double)R[2][re] + (double)R[2][im] * (double)R[2][im];
            values[i] = v;
        }
    });
    return r.getComm().reduce(local);
}

//! For the stacks j < maxStack on all sites: x_j += alpha_j p_j (in floatX), p_j = beta_j p_j + z_j r;
//! then p0 = p_0 on the bulk. As MultiShiftUpdateNoReturn / MixedMultiShiftUpdateNoReturn followed
//! by p0.copyFromStackToStackDevice<NStacks, 0, 0>(p).
template<class floatX, class floatS, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
void multiShiftUpdate(Spinorfield<floatX, onDevice, LatLayout, HaloDepth, NStacks> &x,
                      Spinorfield<floatS, onDevice, LatLayout, HaloDepth, NStacks> &p,
                      const Spinorfield<floatS, onDevice, LatLayout, HaloDepth, 1> &r,
                      SimpleArray<floatX, NStacks> alpha, SimpleArray<floatS, NStacks> z,
                      SimpleArray<floatS, NStacks> beta, const size_t maxStack,
                      Spinorfield<floatS, onDevice, LatLayout, HaloDepth, 1> &p0) {
    cpuProfile::Scope profile(std::is_same_v<floatX, floatS> ? "fused multiShiftUpdate" : "fused multiShiftUpdate (mixed)");
    Segments &seg = segments<LatLayout, HaloDepth>();
    floatX *X[3];
    floatS *P[3], *R[3], *P0[3];
    colours(x, X);
    colours(p, P);
    colours(r, R);
    colours(p0, P0);
    const size_t stride = 2 * seg.nFull;
    const size_t nBulkRuns = seg.bulk.size(), nRuns = nBulkRuns + seg.haloFull.size();
#pragma omp parallel
    {
        size_t begin, end;
        threadRange(nRuns, begin, end);
        for (size_t j = 0; j < maxStack; ++j) {
            const floatX a = alpha[j];
            const floatS b = beta[j], c = z[j];
            for (int k = 0; k < 3; ++k) {
                floatX *Xj = X[k] + j * stride;
                floatS *Pj = P[k] + j * stride;
                const floatS *Rk = R[k];
                floatS *P0k = P0[k];
                for (size_t run = begin; run < end; ++run) {
                    const bool bulk = run < nBulkRuns;
                    const size_t f0 = bulk ? seg.full[run] : seg.haloFull[run - nBulkRuns];
                    const size_t n = bulk ? seg.len[run] : seg.haloLen[run - nBulkRuns];
                    if (j == 0 && bulk) {
                        for (size_t i = 2 * f0; i < 2 * (f0 + n); ++i) {
                            const floatS pOld = Pj[i];
                            Xj[i] = Xj[i] + a * static_cast<floatX>(pOld);
                            const floatS pNew = pOld * b + Rk[i] * c;
                            Pj[i] = pNew;
                            P0k[i] = pNew;
                        }
                    } else {
                        for (size_t i = 2 * f0; i < 2 * (f0 + n); ++i) {
                            const floatS pOld = Pj[i];
                            Xj[i] = Xj[i] + a * static_cast<floatX>(pOld);
                            Pj[i] = pOld * b + Rk[i] * c;
                        }
                    }
                }
            }
        }
    }
}

// The stacked CG (ConjugateGradient::invert_new, one solve per stack), double precision.

//! Re(p_j.s_j) on the bulk for every stack (as real(p.dotProductStacked(s))).
template<bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
SimpleArray<double, NStacks> realDotStacked(const Spinorfield<double, onDevice, LatLayout, HaloDepth, NStacks> &p,
                                            const Spinorfield<double, onDevice, LatLayout, HaloDepth, NStacks> &s) {
    cpuProfile::Scope profile("fused realDotStacked");
    Segments &seg = segments<LatLayout, HaloDepth>();
    double *P[3], *S[3];
    colours(p, P);
    colours(s, S);
    const size_t stride = 2 * seg.nFull;
    const std::array<double, NStacks> local = bulkSums<NStacks>(seg, [&](size_t j, size_t f0, size_t n, double *values) {
        const double *Pj[3] = {P[0] + j * stride, P[1] + j * stride, P[2] + j * stride};
        const double *Sj[3] = {S[0] + j * stride, S[1] + j * stride, S[2] + j * stride};
        for (size_t i = 0; i < n; ++i) {
            const size_t re = 2 * (f0 + i), im = re + 1;
            double v = Pj[0][re] * Sj[0][re] + Pj[0][im] * Sj[0][im];
            v += Pj[1][re] * Sj[1][re] + Pj[1][im] * Sj[1][im];
            v += Pj[2][re] * Sj[2][re] + Pj[2][im] * Sj[2][im];
            values[i] = v;
        }
    });
    SimpleArray<double, NStacks> result;
    for (size_t j = 0; j < NStacks; ++j) result[j] = local[j];
    p.getComm().reduce(&result[0], NStacks);
    return result;
}

//! r_j += b_j s_j on the bulk, then |r_j|^2 for every stack (as r.axpyThisLoopd(b, s, NStacks);
//! real(r.dotProductStacked(r))).
template<bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
SimpleArray<double, NStacks> axpyNormStacked(Spinorfield<double, onDevice, LatLayout, HaloDepth, NStacks> &r,
                                             SimpleArray<double, NStacks> b,
                                             const Spinorfield<double, onDevice, LatLayout, HaloDepth, NStacks> &s) {
    cpuProfile::Scope profile("fused axpyNormStacked");
    Segments &seg = segments<LatLayout, HaloDepth>();
    double *R[3], *S[3];
    colours(r, R);
    colours(s, S);
    const size_t stride = 2 * seg.nFull;
    const std::array<double, NStacks> local = bulkSums<NStacks>(seg, [&](size_t j, size_t f0, size_t n, double *values) {
        double *Rj[3] = {R[0] + j * stride, R[1] + j * stride, R[2] + j * stride};
        const double *Sj[3] = {S[0] + j * stride, S[1] + j * stride, S[2] + j * stride};
        const double bj = b[j];
        for (size_t i = 2 * f0; i < 2 * (f0 + n); ++i) {
            for (int k = 0; k < 3; ++k) Rj[k][i] = Rj[k][i] + bj * Sj[k][i];
        }
        for (size_t i = 0; i < n; ++i) {
            const size_t re = 2 * (f0 + i), im = re + 1;
            double v = Rj[0][re] * Rj[0][re] + Rj[0][im] * Rj[0][im];
            v += Rj[1][re] * Rj[1][re] + Rj[1][im] * Rj[1][im];
            v += Rj[2][re] * Rj[2][re] + Rj[2][im] * Rj[2][im];
            values[i] = v;
        }
    });
    SimpleArray<double, NStacks> result;
    for (size_t j = 0; j < NStacks; ++j) result[j] = local[j];
    r.getComm().reduce(&result[0], NStacks);
    return result;
}

//! x_j += alpha_j p_j, then p_j = r_j + beta_j p_j on the bulk for every stack (as
//! x.axpyThisLoopd(alpha, p, NStacks); p.xpayThisBd(beta, r)).
template<bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
void cgUpdateStacked(Spinorfield<double, onDevice, LatLayout, HaloDepth, NStacks> &x,
                     Spinorfield<double, onDevice, LatLayout, HaloDepth, NStacks> &p,
                     const Spinorfield<double, onDevice, LatLayout, HaloDepth, NStacks> &r,
                     SimpleArray<double, NStacks> alpha, SimpleArray<double, NStacks> beta) {
    cpuProfile::Scope profile("fused cgUpdateStacked");
    Segments &seg = segments<LatLayout, HaloDepth>();
    double *X[3], *P[3], *R[3];
    colours(x, X);
    colours(p, P);
    colours(r, R);
    const size_t stride = 2 * seg.nFull;
#pragma omp parallel
    {
        size_t begin, end;
        threadRange(seg.bulk.size(), begin, end);
        for (size_t j = 0; j < NStacks; ++j) {
            const double a = alpha[j], b = beta[j];
            for (int k = 0; k < 3; ++k) {
                double *Xj = X[k] + j * stride, *Pj = P[k] + j * stride;
                const double *Rj = R[k] + j * stride;
                for (size_t run = begin; run < end; ++run) {
                    const size_t f0 = seg.full[run], n = seg.len[run];
                    for (size_t i = 2 * f0; i < 2 * (f0 + n); ++i) {
                        const double pOld = Pj[i];
                        Xj[i] = Xj[i] + a * pOld;
                        Pj[i] = Rj[i] + b * pOld;
                    }
                }
            }
        }
    }
}

} // namespace cpuBlas
#endif
