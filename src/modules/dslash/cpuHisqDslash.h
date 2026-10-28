/*
 * cpuHisqDslash.h
 *
 * AVX-512 HISQ Dslash for the CPU backend (one or NS stacked right-hand sides). One SIMD lane per output
 * site: 8 sites per register in double, 16 in float, and every complex number is held as
 * a vector of real parts and a vector of imaginary parts, so the SU(3) arithmetic is plain
 * FMAs. The 16 spinor and 16 link indices of every site come from a table built once per
 * lattice geometry with the same indexer calls as HisqDslashFunctor (see dslash.cpp), so
 * halos and periodic wrap-around need no special cases. Where the W lanes of a block address
 * consecutive complex numbers (stride 1: spinors of neighbouring sites) or every second one
 * (stride 2: links of one parity in the full-lattice gauge arrays), operands are fetched by
 * plain loads plus permutes; elsewhere (row wrap-around, the padded last block) by gathers.
 *
 * Term k = 4*mu + {0: fat forward, 1: fat backward (dagger), 2: Naik forward,
 * 3: Naik backward (dagger)}, as in HisqDslashFunctor. With NS stacks, every link is loaded
 * once and applied to all stacks (stack s sits stackStride reals after stack 0, so the table
 * of stack 0 serves all of them); each stack gets the same operation order as one RHS.
 * The table also splits the blocks into interior (only bulk sites of the input are read) and
 * boundary, so that on several ranks M^dagger M can compute the interior while the input halos
 * are exchanged (SIMULATEQCD_CPU_OVERLAP=0 switches this off).
 * Set SIMULATEQCD_CPU_SIMD=0 to fall back to the scalar functors.
 */
#pragma once

#if defined(USE_CPU) && defined(__AVX512F__)
#define CPU_HISQ_DSLASH_SIMD 1

#include <immintrin.h>
#include <cstdint>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <string_view>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace cpuSimd {

//! Switch for the vectorised kernels: on unless SIMULATEQCD_CPU_SIMD=0; tests can flip it.
inline bool &enabledFlag() {
    static bool on = [] {
        const char *value = std::getenv("SIMULATEQCD_CPU_SIMD");
        return !value || std::string_view(value) != "0";
    }();
    return on;
}
inline bool enabled() { return enabledFlag(); }

//! Overlap of halo exchanges with the interior sites of the Dslash in M^dagger M on several
//! ranks: on unless SIMULATEQCD_CPU_OVERLAP=0.
inline bool overlapEnabled() {
    static const bool on = [] {
        const char *value = std::getenv("SIMULATEQCD_CPU_OVERLAP");
        return !value || std::string_view(value) != "0";
    }();
    return on;
}

//! Site and link indices of one parity, in blocks of W sites. Every index is stored
//! doubled (it addresses the real part in an array of interleaved complex numbers).
//! The last block is padded with the indices of its last site.
struct DslashTable {
    size_t lx = 0, ly = 0, lz = 0, lt = 0;  // full local extents the table was built for
    size_t nSites = 0, nBlocks = 0;
    int width = 0;
    std::vector<int32_t> out;     // [block][lane]
    std::vector<int32_t> spinor;  // [block][term][lane]
    std::vector<int32_t> link;    // [block][term][lane]
    std::vector<uint8_t> stride;  // [block][33]: spinor terms, link terms, out; 0 = gather
    // Blocks that read only bulk sites of the input (interior) and the others (boundary), so
    // the interior can be computed while the input halos are exchanged.
    std::vector<uint32_t> interior, boundary;
};

//! 1 or 2 if the doubled indices idx[0..W) advance by 2 or 4 (complex stride 1 or 2), else 0.
inline uint8_t strideOf(const int32_t *idx, int W) {
    for (int stride = 1; stride <= 2; ++stride) {
        bool regular = true;
        for (int l = 1; l < W && regular; ++l) regular = idx[l] == idx[0] + 2 * stride * l;
        if (regular) return static_cast<uint8_t>(stride);
    }
    return 0;
}

struct VecD {
    using real = double;
    using V = __m512d;
    using I = __m256i;
    using M = __mmask8;
    static constexpr int W = 8;
    static V zero() { return _mm512_setzero_pd(); }
    static V set1(double a) { return _mm512_set1_pd(a); }
    static I index(const int32_t *p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)); }
    static I plusOne(I i) { return _mm256_add_epi32(i, _mm256_set1_epi32(1)); }
    static V gather(const double *base, I i) { return _mm512_i32gather_pd(i, base, 8); }
    static void scatter(double *base, M m, I i, V v) { _mm512_mask_i32scatter_pd(base, m, i, v, 8); }
    static V mul(V a, V b) { return _mm512_mul_pd(a, b); }
    static V add(V a, V b) { return _mm512_add_pd(a, b); }
    static V sub(V a, V b) { return _mm512_sub_pd(a, b); }
    static V fmadd(V a, V b, V c) { return _mm512_fmadd_pd(a, b, c); }   // a*b + c
    static V fnmadd(V a, V b, V c) { return _mm512_fnmadd_pd(a, b, c); } // c - a*b
    static V fmsub(V a, V b, V c) { return _mm512_fmsub_pd(a, b, c); }   // a*b - c
    static M mask(size_t n) { return static_cast<M>(n >= W ? 0xFFu : (1u << n) - 1u); }
    //! W consecutive interleaved complex numbers at p -> real and imaginary vectors.
    static void load1(const double *p, V &re, V &im) {
        const V lo = _mm512_loadu_pd(p), hi = _mm512_loadu_pd(p + 8);
        re = _mm512_permutex2var_pd(lo, _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), hi);
        im = _mm512_permutex2var_pd(lo, _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15), hi);
    }
    //! Every second of 2W interleaved complex numbers at p.
    static void load2(const double *p, V &re, V &im) {
        const __m512i pick = _mm512_setr_epi64(0, 4, 8, 12, 1, 5, 9, 13);
        const V a = _mm512_permutex2var_pd(_mm512_loadu_pd(p), pick, _mm512_loadu_pd(p + 8));
        const V b = _mm512_permutex2var_pd(_mm512_loadu_pd(p + 16), pick, _mm512_loadu_pd(p + 24));
        re = _mm512_shuffle_f64x2(a, b, 0x44);
        im = _mm512_shuffle_f64x2(a, b, 0xEE);
    }
    static void store1(double *p, V re, V im) {
        _mm512_storeu_pd(p, _mm512_permutex2var_pd(re, _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11), im));
        _mm512_storeu_pd(p + 8, _mm512_permutex2var_pd(re, _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15), im));
    }
};

struct VecF {
    using real = float;
    using V = __m512;
    using I = __m512i;
    using M = __mmask16;
    static constexpr int W = 16;
    static V zero() { return _mm512_setzero_ps(); }
    static V set1(float a) { return _mm512_set1_ps(a); }
    static I index(const int32_t *p) { return _mm512_loadu_si512(p); }
    static I plusOne(I i) { return _mm512_add_epi32(i, _mm512_set1_epi32(1)); }
    static V gather(const float *base, I i) { return _mm512_i32gather_ps(i, base, 4); }
    static void scatter(float *base, M m, I i, V v) { _mm512_mask_i32scatter_ps(base, m, i, v, 4); }
    static V mul(V a, V b) { return _mm512_mul_ps(a, b); }
    static V add(V a, V b) { return _mm512_add_ps(a, b); }
    static V sub(V a, V b) { return _mm512_sub_ps(a, b); }
    static V fmadd(V a, V b, V c) { return _mm512_fmadd_ps(a, b, c); }
    static V fnmadd(V a, V b, V c) { return _mm512_fnmadd_ps(a, b, c); }
    static V fmsub(V a, V b, V c) { return _mm512_fmsub_ps(a, b, c); }
    static M mask(size_t n) { return static_cast<M>(n >= W ? 0xFFFFu : (1u << n) - 1u); }
    static void load1(const float *p, V &re, V &im) {
        const V lo = _mm512_loadu_ps(p), hi = _mm512_loadu_ps(p + 16);
        re = _mm512_permutex2var_ps(lo, _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30), hi);
        im = _mm512_permutex2var_ps(lo, _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31), hi);
    }
    static void load2(const float *p, V &re, V &im) {
        const __m512i pick = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29);
        const V a = _mm512_permutex2var_ps(_mm512_loadu_ps(p), pick, _mm512_loadu_ps(p + 16));
        const V b = _mm512_permutex2var_ps(_mm512_loadu_ps(p + 32), pick, _mm512_loadu_ps(p + 48));
        re = _mm512_shuffle_f32x4(a, b, 0x44);
        im = _mm512_shuffle_f32x4(a, b, 0xEE);
    }
    static void store1(float *p, V re, V im) {
        _mm512_storeu_ps(p, _mm512_permutex2var_ps(re, _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23), im));
        _mm512_storeu_ps(p + 16, _mm512_permutex2var_ps(re, _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31), im));
    }
};

template<class S> struct Cplx { typename S::V re, im; };

//! Where W complex numbers of one entry array live: doubled lane indices, their stride class.
template<class S>
struct Where {
    const int32_t *idx;
    uint8_t stride;
};

template<class S>
inline Cplx<S> load(const typename S::real *base, const Where<S> &w) {
    Cplx<S> z;
    if (w.stride == 1) {
        S::load1(base + w.idx[0], z.re, z.im);
    } else if (w.stride == 2) {
        S::load2(base + w.idx[0], z.re, z.im);
    } else {
        const typename S::I re = S::index(w.idx);
        z.re = S::gather(base, re);
        z.im = S::gather(base, S::plusOne(re));
    }
    return z;
}

//! a*b
template<class S>
inline Cplx<S> cmul(const Cplx<S> &a, const Cplx<S> &b) {
    return {S::fmsub(a.re, b.re, S::mul(a.im, b.im)), S::fmadd(a.re, b.im, S::mul(a.im, b.re))};
}

//! conj(a*b - c*d)
template<class S>
inline Cplx<S> conjCross(const Cplx<S> &a, const Cplx<S> &b, const Cplx<S> &c, const Cplx<S> &d) {
    const Cplx<S> ab = cmul<S>(a, b), cd = cmul<S>(c, d);
    return {S::sub(ab.re, cd.re), S::sub(cd.im, ab.im)};
}

//! acc += coef * (U v) for row-major U.
template<class S>
inline void addMatVec(Cplx<S> acc[3], typename S::V coef, const Cplx<S> U[9], const Cplx<S> v[3]) {
    for (int i = 0; i < 3; ++i) {
        typename S::V re = S::mul(U[3 * i].re, v[0].re), im = S::mul(U[3 * i].re, v[0].im);
        re = S::fnmadd(U[3 * i].im, v[0].im, re);
        im = S::fmadd(U[3 * i].im, v[0].re, im);
        for (int j = 1; j < 3; ++j) {
            re = S::fmadd(U[3 * i + j].re, v[j].re, re);
            re = S::fnmadd(U[3 * i + j].im, v[j].im, re);
            im = S::fmadd(U[3 * i + j].re, v[j].im, im);
            im = S::fmadd(U[3 * i + j].im, v[j].re, im);
        }
        acc[i].re = S::fmadd(coef, re, acc[i].re);
        acc[i].im = S::fmadd(coef, im, acc[i].im);
    }
}

//! acc += coef * (U^dagger v) for row-major U: (U^dagger v)_i = sum_j conj(U_ji) v_j.
template<class S>
inline void addMatDaggerVec(Cplx<S> acc[3], typename S::V coef, const Cplx<S> U[9], const Cplx<S> v[3]) {
    for (int i = 0; i < 3; ++i) {
        typename S::V re = S::mul(U[i].re, v[0].re), im = S::mul(U[i].re, v[0].im);
        re = S::fmadd(U[i].im, v[0].im, re);
        im = S::fnmadd(U[i].im, v[0].re, im);
        for (int j = 1; j < 3; ++j) {
            re = S::fmadd(U[3 * j + i].re, v[j].re, re);
            re = S::fmadd(U[3 * j + i].im, v[j].im, re);
            im = S::fmadd(U[3 * j + i].re, v[j].im, im);
            im = S::fnmadd(U[3 * j + i].im, v[j].re, im);
        }
        acc[i].re = S::fmadd(coef, re, acc[i].re);
        acc[i].im = S::fmadd(coef, im, acc[i].im);
    }
}

//! R18 link: all nine entries stored.
template<class S>
inline void loadFat(Cplx<S> U[9], const typename S::real *const fat[9], const Where<S> &w) {
    for (int e = 0; e < 9; ++e) U[e] = load<S>(fat[e], w);
}

//! U3R14 link: rows 0 and 1 stored, entry 6 holds the U(1) phase; row 2 is
//! conj(row0 x row1) * phase, as in SU3::u3reconstruct.
template<class S>
inline void loadNaik(Cplx<S> U[9], const typename S::real *const naik[7], const Where<S> &w) {
    for (int e = 0; e < 6; ++e) U[e] = load<S>(naik[e], w);
    const Cplx<S> phase = load<S>(naik[6], w);
    U[6] = cmul<S>(conjCross<S>(U[1], U[5], U[2], U[4]), phase);
    U[7] = cmul<S>(conjCross<S>(U[2], U[3], U[0], U[5]), phase);
    U[8] = cmul<S>(conjCross<S>(U[0], U[4], U[1], U[3]), phase);
}

//! out = alpha * Dslash(in) + beta * aux (aux may be null), on the sites of table, for NS
//! stacks that lie stackStride reals apart in out, in and aux. With blocks, only on the nList
//! listed blocks; then progress (if given) is called by thread 0, which computes no blocks, to
//! let MPI move a halo exchange that runs at the same time.
template<class S, size_t NS = 1>
void hisqDslash(const DslashTable &table, typename S::real *const out[3],
                const typename S::real *const in[3], const typename S::real *const fat[9],
                const typename S::real *const naik[7], typename S::real c1000, typename S::real c3000,
                typename S::real alpha, const typename S::real *const aux[3], typename S::real beta,
                const size_t stackStride = 0, const uint32_t *blocks = nullptr, size_t nList = 0,
                void (*progress)() = nullptr) {
    using V = typename S::V;
    const V cFat = S::set1(c1000), cNaik = S::set1(c3000);
    const V minusFat = S::set1(-c1000), minusNaik = S::set1(-c3000);
    const V a = S::set1(alpha), b = S::set1(beta);
    const int W = S::W;
    const size_t n = blocks ? nList : table.nBlocks;
    // With progress on several threads, thread 0 only polls MPI while the others share the
    // blocks: a thread that computes and polls would hold up the whole loop. Otherwise the
    // blocks are split as by schedule(static).
    std::atomic<int> workersDone{0};
#pragma omp parallel
    {
#ifdef _OPENMP
        const size_t nt = omp_get_num_threads(), t = omp_get_thread_num();
#else
        const size_t nt = 1, t = 0;
#endif
        const bool poll = progress && nt > 1;
        if (poll && t == 0) {
            while (workersDone.load(std::memory_order_acquire) < static_cast<int>(nt - 1)) progress();
        } else {
            const size_t nw = poll ? nt - 1 : nt, w = poll ? t - 1 : t;
            const size_t q = n / nw, r = n % nw;
            const size_t begin = w * q + std::min(w, r), end = begin + q + (w < r ? 1 : 0);
            for (size_t j = begin; j < end; ++j) {
                if (progress && !poll && j % 16 == 0) progress();
                const size_t block = blocks ? blocks[j] : j;
                const int32_t *spinor = &table.spinor[block * 16 * W];
                const int32_t *link = &table.link[block * 16 * W];
                const uint8_t *stride = &table.stride[block * 33];
                auto sp = [&](int k) { return Where<S>{spinor + k * W, stride[k]}; };
                auto ln = [&](int k) { return Where<S>{link + k * W, stride[16 + k]}; };
                Cplx<S> acc[NS][3];
                for (size_t s = 0; s < NS; ++s)
                    for (int c = 0; c < 3; ++c) acc[s][c] = {S::zero(), S::zero()};
                Cplx<S> U[9], v[3];
                // v = in[stack s] at the neighbours of term k
                auto neighbour = [&](size_t s, int k) {
                    for (int c = 0; c < 3; ++c) v[c] = load<S>(in[c] + s * stackStride, sp(k));
                };
                for (int mu = 0; mu < 4; ++mu) {
                    const int k = 4 * mu;
                    loadFat<S>(U, fat, ln(k + 0));
                    for (size_t s = 0; s < NS; ++s) { neighbour(s, k + 0); addMatVec<S>(acc[s], cFat, U, v); }

                    loadFat<S>(U, fat, ln(k + 1));
                    for (size_t s = 0; s < NS; ++s) { neighbour(s, k + 1); addMatDaggerVec<S>(acc[s], minusFat, U, v); }

                    loadNaik<S>(U, naik, ln(k + 2));
                    for (size_t s = 0; s < NS; ++s) { neighbour(s, k + 2); addMatVec<S>(acc[s], cNaik, U, v); }

                    loadNaik<S>(U, naik, ln(k + 3));
                    for (size_t s = 0; s < NS; ++s) { neighbour(s, k + 3); addMatDaggerVec<S>(acc[s], minusNaik, U, v); }
                }
                const Where<S> where{&table.out[block * W], stride[32]};
                const typename S::I outRe = S::index(where.idx);
                const typename S::I outIm = S::plusOne(outRe);
                const typename S::M lanes = S::mask(table.nSites - block * W);
                for (size_t s = 0; s < NS; ++s) {
                    for (int c = 0; c < 3; ++c) {
                        typename S::real *const o = out[c] + s * stackStride;
                        V re = S::mul(a, acc[s][c].re), im = S::mul(a, acc[s][c].im);
                        if (aux) {
                            const Cplx<S> x = load<S>(aux[c] + s * stackStride, where);
                            re = S::fmadd(b, x.re, re);
                            im = S::fmadd(b, x.im, im);
                        }
                        if (where.stride == 1) {
                            S::store1(o + where.idx[0], re, im);
                        } else {
                            S::scatter(o, lanes, outRe, re);
                            S::scatter(o, lanes, outIm, im);
                        }
                    }
                }
            }
            if (poll) workersDone.fetch_add(1, std::memory_order_release);
        }
    }
}

} // namespace cpuSimd

#endif
