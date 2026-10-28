/*
 * simdLane.h
 *
 * Lane<T>: W = 64 / sizeof(T) values of T in one AVX-512 register (8 double, 16 float), with the
 * arithmetic of T element by element. GPUcomplex<Lane<T>> and SU3<Lane<T>> then hold one complex
 * number or SU3 matrix per lane, and the SU3 code runs on W sites at once with the same operations
 * in the same order as for one site: without FMA contraction the lanes are bit-identical to the
 * scalar results. The sites of the lanes are a cpuSimd::LaneSite (laneSite.h); loadLanes() and
 * storeLanes() move the W links of an R18 field between memory and an SU3<Lane<T>>.
 */
#pragma once

#include "../indexer/laneSite.h"

#ifdef CPU_LANE_SIMD
#include "su3.h"
#include "../../modules/dslash/cpuHisqDslash.h"
#include <algorithm>
#include <type_traits>

namespace cpuSimd {

template<class T>
struct Lane {
    typedef T V __attribute__((vector_size(64)));
    static constexpr int W = 64 / sizeof(T);
    V v;

    Lane() = default;
    Lane(T a) : v(V{} + a) {}

    static Lane of(V x) {
        Lane r;
        r.v = x;
        return r;
    }

    friend Lane operator+(Lane a, Lane b) { return of(a.v + b.v); }
    friend Lane operator-(Lane a, Lane b) { return of(a.v - b.v); }
    friend Lane operator*(Lane a, Lane b) { return of(a.v * b.v); }
    friend Lane operator/(Lane a, Lane b) { return of(a.v / b.v); }
    friend Lane operator-(Lane a) { return of(-a.v); }
    Lane &operator+=(Lane b) { v += b.v; return *this; }
    Lane &operator-=(Lane b) { v -= b.v; return *this; }
    Lane &operator*=(Lane b) { v *= b.v; return *this; }
    Lane &operator/=(Lane b) { v /= b.v; return *this; }
};

//! Real and imaginary parts, the storage of GPUcomplex<Lane<T>> (as double2 for GPUcomplex<double>).
template<class T>
struct Lane2 {
    Lane<T> x, y;
};

template<class T> using LaneOps = std::conditional_t<std::is_same_v<T, double>, VecD, VecF>;

} // namespace cpuSimd

template<class T> class Selector<cpuSimd::Lane<T>> {
public:
    using Type = cpuSimd::Lane2<T>;
};

// su3_zero<floatT>() builds the matrix from int literals, which do not convert to
// GPUcomplex<Lane<T>> in one step.
template<>
inline SU3<cpuSimd::Lane<double>> su3_zero<cpuSimd::Lane<double>>() {
    const GPUcomplex<cpuSimd::Lane<double>> z(0.0, 0.0);
    return SU3<cpuSimd::Lane<double>>(z, z, z, z, z, z, z, z, z);
}
template<>
inline SU3<cpuSimd::Lane<float>> su3_zero<cpuSimd::Lane<float>>() {
    const GPUcomplex<cpuSimd::Lane<float>> z(0.0f, 0.0f);
    return SU3<cpuSimd::Lane<float>>(z, z, z, z, z, z, z, z, z);
}

//! Scalar coefficients times lane matrices (floatT(c) * SU3<floatT> in the scalar code).
template<class T>
inline SU3<cpuSimd::Lane<T>> operator*(const T &x, const SU3<cpuSimd::Lane<T>> &y) {
    return cpuSimd::Lane<T>(x) * y;
}
template<class T>
inline SU3<cpuSimd::Lane<T>> operator*(const SU3<cpuSimd::Lane<T>> &x, const T &y) {
    return x * cpuSimd::Lane<T>(y);
}

namespace cpuSimd {

//! Doubled indices (real part in interleaved complex storage) of the W links in direction mu at
//! the lanes of s; lanes beyond s.n repeat the last valid one. Returns true if they are
//! consecutive (lane l at first + 2l), in which case only first is set.
template<int W>
inline bool laneIndices(const LaneSiteMu &s, int64_t &first, int32_t lanes[W]) {
    const LaneGeometry &g = *s.g;
    int c[4] = {s.c[0], s.c[1], s.c[2], s.c[3]};
    if (s.n == W && c[0] >= 0 && c[0] + 2 * (W - 1) < g.lFull[0]) {
        first = 2 * g.index(c, s.mu);
        return true;
    }
    const int x0 = c[0];
    for (int l = 0; l < W; ++l) {
        c[0] = wrapLaneCoord(x0 + 2 * std::min(l, s.n - 1), g.lFull[0]);
        lanes[l] = static_cast<int32_t>(2 * g.index(c, s.mu));
    }
    return false;
}

//! The W links at s of an R18 field (entry e at entries[e]), or their daggers.
template<class T>
inline SU3<Lane<T>> loadLanes(const T *const entries[9], const LaneSiteMu &s, bool dagger) {
    using S = LaneOps<T>;
    using V = typename Lane<T>::V;
    constexpr int W = S::W;
    int64_t first = 0;
    alignas(64) int32_t lanes[W];
    GPUcomplex<Lane<T>> e[9];
    if (laneIndices<W>(s, first, lanes)) {
        for (int k = 0; k < 9; ++k) {
            typename S::V re, im;
            S::load1(entries[k] + first, re, im);
            e[k] = GPUcomplex<Lane<T>>(Lane<T>::of((V)re), Lane<T>::of((V)im));
        }
    } else {
        const typename S::I re = S::index(lanes), im = S::plusOne(re);
        for (int k = 0; k < 9; ++k) {
            e[k] = GPUcomplex<Lane<T>>(Lane<T>::of((V)S::gather(entries[k], re)),
                                       Lane<T>::of((V)S::gather(entries[k], im)));
        }
    }
    if (!dagger) return SU3<Lane<T>>(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8]);
    return SU3<Lane<T>>(conj(e[0]), conj(e[3]), conj(e[6]),
                        conj(e[1]), conj(e[4]), conj(e[7]),
                        conj(e[2]), conj(e[5]), conj(e[8]));
}

//! Writes the n valid lanes of m to the links at s of an R18 field.
template<class T>
inline void storeLanes(T *const entries[9], const LaneSiteMu &s, SU3<Lane<T>> m) {
    using S = LaneOps<T>;
    using SV = typename S::V;
    constexpr int W = S::W;
    int64_t first = 0;
    alignas(64) int32_t lanes[W];
    if (laneIndices<W>(s, first, lanes)) {
        for (int k = 0; k < 9; ++k) {
            const GPUcomplex<Lane<T>> z = m(k / 3, k % 3);
            S::store1(entries[k] + first, (SV)z.cREAL.v, (SV)z.cIMAG.v);
        }
    } else {
        const typename S::I re = S::index(lanes), im = S::plusOne(re);
        const typename S::M mask = S::mask(s.n);
        for (int k = 0; k < 9; ++k) {
            const GPUcomplex<Lane<T>> z = m(k / 3, k % 3);
            S::scatter(entries[k], mask, re, (SV)z.cREAL.v);
            S::scatter(entries[k], mask, im, (SV)z.cIMAG.v);
        }
    }
}

} // namespace cpuSimd
#endif
