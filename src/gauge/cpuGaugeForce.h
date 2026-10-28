/*
 * cpuGaugeForce.h
 *
 * AVX-512 Symanzik gauge force for the CPU backend: the staple sum of gauge_force() in
 * gaugeActionDeriv.h, term by term, with one SIMD lane per site (8 sites per register in
 * double, 16 in float; real and imaginary parts in separate vectors, as in cpuHisqDslash.h).
 * A block is W bulk sites of one parity in one x-row, so for every neighbour the W links lie
 * at consecutive indices of the full-lattice gauge arrays, unless the row wraps around in x
 * (unsplit x direction, blocks at the row ends); those operands, and every operand of a
 * partial block (lx/2 not a multiple of W), are gathered. Indices are computed per block from
 * the full coordinates with the formula of GIndexer::coordMuToIndexMu_Full and the periodic
 * moves of GIndexer::site_move, so no index table is needed (it would hold 94 entries per site
 * and direction). The four directions of a block are done together, so neighbouring links are
 * reused from cache.
 */
#pragma once

#include "../modules/dslash/cpuHisqDslash.h"

#ifdef CPU_HISQ_DSLASH_SIMD
#include "gaugefield.h"
#include <algorithm>
#include <limits>
#include <type_traits>

namespace cpuSimd {

//! Extents of the local lattice: bulk, halo depth and full (bulk + halos) per direction.
struct ForceGeometry {
    int l[4], h[4], lFull[4];
    int64_t vol1Full, vol2Full, vol3Full, vol4Full, sizehFull;

    //! Index of the link in direction mu at full coordinates c, as coordMuToIndexMu_Full.
    int64_t index(const int c[4], int mu) const {
        return ((c[0] + c[1] * vol1Full + c[2] * vol2Full + c[3] * vol3Full) >> 1)
               + sizehFull * ((c[0] + c[1] + c[2] + c[3]) & 1) + mu * vol4Full;
    }
};

inline int wrapCoord(int x, int extent) { return x < 0 ? x + extent : (x >= extent ? x - extent : x); }

//! c = a * b (c must not alias a or b).
template<class S>
inline void matMul(Cplx<S> c[9], const Cplx<S> a[9], const Cplx<S> b[9]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            typename S::V re = S::mul(a[3 * i].re, b[j].re), im = S::mul(a[3 * i].re, b[j].im);
            re = S::fnmadd(a[3 * i].im, b[j].im, re);
            im = S::fmadd(a[3 * i].im, b[j].re, im);
            for (int k = 1; k < 3; ++k) {
                re = S::fmadd(a[3 * i + k].re, b[3 * k + j].re, re);
                re = S::fnmadd(a[3 * i + k].im, b[3 * k + j].im, re);
                im = S::fmadd(a[3 * i + k].re, b[3 * k + j].im, im);
                im = S::fmadd(a[3 * i + k].im, b[3 * k + j].re, im);
            }
            c[3 * i + j] = {re, im};
        }
    }
}

//! a = a * b
template<class S>
inline void matMulAssign(Cplx<S> a[9], const Cplx<S> b[9]) {
    Cplx<S> c[9];
    matMul<S>(c, a, b);
    for (int e = 0; e < 9; ++e) a[e] = c[e];
}

//! y += coef * x
template<class S>
inline void matAxpy(Cplx<S> y[9], typename S::V coef, const Cplx<S> x[9]) {
    for (int e = 0; e < 9; ++e) y[e] = {S::fmadd(coef, x[e].re, y[e].re), S::fmadd(coef, x[e].im, y[e].im)};
}

//! y += x
template<class S>
inline void matAdd(Cplx<S> y[9], const Cplx<S> x[9]) {
    for (int e = 0; e < 9; ++e) y[e] = {S::add(y[e].re, x[e].re), S::add(y[e].im, x[e].im)};
}

//! W sites of one parity in one x-row: full coordinates of lane 0, n valid lanes (lane l
//! sits at x = c[0] + 2l).
struct ForceBlock {
    int c[4];
    int n;
};

//! U = link in direction dir at (site + a mu^ + b nu^) for every lane, or its dagger.
template<class S>
inline void loadForceLink(Cplx<S> U[9], const typename S::real *const gauge[9], const ForceGeometry &g,
                          const ForceBlock &b, int mu, int a, int nu, int bnu, int dir, bool dagger) {
    constexpr int W = S::W;
    int d[4] = {0, 0, 0, 0};
    d[mu] += a;
    d[nu] += bnu;
    int c[4];
    for (int i = 1; i < 4; ++i) c[i] = wrapCoord(b.c[i] + d[i], g.lFull[i]);
    c[0] = b.c[0] + d[0];
    if (b.n == W && c[0] >= 0 && c[0] + 2 * (W - 1) < g.lFull[0]) {
        // No wrap-around: lane l is the link l entries after lane 0 (same parity, x + 2l).
        const int64_t first = 2 * g.index(c, dir);
        for (int e = 0; e < 9; ++e) S::load1(gauge[e] + first, U[e].re, U[e].im);
    } else {
        alignas(64) int32_t lanes[W];
        const int x0 = c[0];
        for (int l = 0; l < W; ++l) {
            c[0] = wrapCoord(x0 + 2 * std::min(l, b.n - 1), g.lFull[0]);
            lanes[l] = static_cast<int32_t>(2 * g.index(c, dir));
        }
        const Where<S> where{lanes, 0};
        for (int e = 0; e < 9; ++e) U[e] = load<S>(gauge[e], where);
    }
    if (dagger) {
        const typename S::V zero = S::zero();
        for (int i = 0; i < 3; ++i) {
            U[4 * i].im = S::sub(zero, U[4 * i].im);
            for (int j = i + 1; j < 3; ++j) {
                const Cplx<S> upper = U[3 * i + j];
                U[3 * i + j] = {U[3 * j + i].re, S::sub(zero, U[3 * j + i].im)};
                U[3 * j + i] = {upper.re, S::sub(zero, upper.im)};
            }
        }
    }
}

//! force = TA(r_1 U_mu(x) * staple) on every bulk link, with r_1 = beta/3 * 3/5, as
//! gauge_force(). Entry arrays of R18 fields; gauge must have up-to-date halos.
template<class S>
void gaugeForce(const ForceGeometry &g, typename S::real *const force[9], const typename S::real *const gauge[9],
                const typename S::real r1) {
    using V = typename S::V;
    constexpr int W = S::W;
    const V c1 = S::set1(-5.0 / 3.0), c2 = S::set1(1.0 / 12.0), r = S::set1(r1);
    const V half = S::set1(0.5), twoThirds = S::set1(0.6666666666666666), third = S::set1(0.3333333333333333);
    const V zero = S::zero();
    const int perRow = g.l[0] / 2;
    const int blocksPerRow = (perRow + W - 1) / W;
    const size_t rows = static_cast<size_t>(g.l[1]) * g.l[2] * g.l[3];
    const size_t nBlocks = rows * 2 * blocksPerRow;
    // Block order: row, parity, position in the row, so both parities of a row run together.
#pragma omp parallel for schedule(static)
    for (size_t block = 0; block < nBlocks; ++block) {
        const int j = static_cast<int>(block % blocksPerRow);
        const int parity = static_cast<int>((block / blocksPerRow) % 2);
        const size_t row = block / (2 * blocksPerRow);
        ForceBlock b;
        b.c[1] = static_cast<int>(row % g.l[1]) + g.h[1];
        b.c[2] = static_cast<int>((row / g.l[1]) % g.l[2]) + g.h[2];
        b.c[3] = static_cast<int>(row / (static_cast<size_t>(g.l[1]) * g.l[2])) + g.h[3];
        b.c[0] = g.h[0] + ((parity - b.c[1] - b.c[2] - b.c[3] - g.h[0]) & 1) + 2 * W * j;
        b.n = std::min(W, perRow - W * j);

        for (int mu = 0; mu < 4; ++mu) {
            Cplx<S> staple[9], m0[9], m1[9], A[9], B[9];
            for (int e = 0; e < 9; ++e) staple[e] = {zero, zero};
            for (int nu_aux = 1; nu_aux < 4; ++nu_aux) {
                const int nu = (mu + nu_aux) % 4;
                // L(a, b, dir): link in direction dir at site + a mu^ + b nu^.
                auto L = [&](Cplx<S> U[9], int a, int bnu, int dir, bool dagger) {
                    loadForceLink<S>(U, gauge, g, b, mu, a, nu, bnu, dir, dagger);
                };
                L(m0, 1, -1, nu, true);
                L(A, 1, -2, nu, true);  matMulAssign<S>(m0, A);
                L(A, 0, -2, mu, true);  matMulAssign<S>(m0, A);
                L(A, 0, -2, nu, false); matMulAssign<S>(m0, A);
                L(A, 0, -1, nu, false); matMulAssign<S>(m0, A);
                matAxpy<S>(staple, c2, m0);

                L(m0, 1, 0, nu, false);
                L(A, 1, 1, nu, false);  matMulAssign<S>(m0, A);
                L(A, 0, 2, mu, true);   matMulAssign<S>(m0, A);
                L(A, 0, 1, nu, true);   matMulAssign<S>(m0, A);
                L(A, 0, 0, nu, true);   matMulAssign<S>(m0, A);
                matAxpy<S>(staple, c2, m0);

                L(m1, 2, 0, nu, false);
                L(A, 1, 1, mu, true);   matMulAssign<S>(m1, A);
                L(A, 0, 1, mu, true);   matMulAssign<S>(m1, A);
                L(A, 0, 0, nu, true);   matMulAssign<S>(m1, A);
                L(m0, 2, -1, nu, true);
                L(A, 1, -1, mu, true);  matMulAssign<S>(m0, A);
                L(A, 0, -1, mu, true);  matMulAssign<S>(m0, A);
                L(A, 0, -1, nu, false); matMulAssign<S>(m0, A);
                matAdd<S>(m0, m1);
                L(A, 1, 0, mu, false);  matMul<S>(m1, A, m0);
                matAxpy<S>(staple, c2, m1);

                L(m0, -1, 1, mu, true);
                L(A, -1, 0, nu, true);  matMulAssign<S>(m0, A);
                L(A, -1, 0, mu, false); matMulAssign<S>(m0, A);
                L(m1, 1, 0, nu, false);
                L(A, 0, 1, mu, true);   matMulAssign<S>(m1, A);
                L(A, 0, 0, nu, true);
                for (int e = 0; e < 9; ++e)
                    B[e] = {S::fmadd(c2, m0[e].re, S::mul(c1, A[e].re)), S::fmadd(c2, m0[e].im, S::mul(c1, A[e].im))};
                matMul<S>(A, m1, B);
                matAdd<S>(staple, A);

                L(m0, -1, -1, mu, true);
                L(A, -1, -1, nu, false); matMulAssign<S>(m0, A);
                L(A, -1, 0, mu, false);  matMulAssign<S>(m0, A);
                L(m1, 1, -1, nu, true);
                L(A, 0, -1, mu, true);   matMulAssign<S>(m1, A);
                L(A, 0, -1, nu, false);
                for (int e = 0; e < 9; ++e)
                    B[e] = {S::fmadd(c2, m0[e].re, S::mul(c1, A[e].re)), S::fmadd(c2, m0[e].im, S::mul(c1, A[e].im))};
                matMul<S>(A, m1, B);
                matAdd<S>(staple, A);
            }
            loadForceLink<S>(A, gauge, g, b, mu, 0, (mu + 1) % 4, 0, mu, false);
            for (int e = 0; e < 9; ++e) A[e] = {S::mul(r, A[e].re), S::mul(r, A[e].im)};
            matMul<S>(m0, A, staple);

            // Traceless anti-hermitian part, as SU3::TA().
            Cplx<S> T[9];
            for (int i = 0; i < 3; ++i) {
                const V otherIm = S::add(m0[4 * ((i + 1) % 3)].im, m0[4 * ((i + 2) % 3)].im);
                T[4 * i] = {zero, S::fmsub(twoThirds, m0[4 * i].im, S::mul(third, otherIm))};
                for (int k = i + 1; k < 3; ++k) {
                    const Cplx<S> &x = m0[3 * i + k], &y = m0[3 * k + i];
                    T[3 * i + k] = {S::mul(half, S::sub(x.re, y.re)), S::mul(half, S::add(x.im, y.im))};
                    T[3 * k + i] = {S::mul(half, S::sub(y.re, x.re)), S::mul(half, S::add(y.im, x.im))};
                }
            }

            int c[4] = {b.c[0], b.c[1], b.c[2], b.c[3]};
            if (b.n == W) {
                const int64_t first = 2 * g.index(c, mu);
                for (int e = 0; e < 9; ++e) S::store1(force[e] + first, T[e].re, T[e].im);
            } else {
                alignas(64) int32_t lanes[W];
                for (int l = 0; l < W; ++l) {
                    c[0] = b.c[0] + 2 * std::min(l, b.n - 1);
                    lanes[l] = static_cast<int32_t>(2 * g.index(c, mu));
                }
                const typename S::I re = S::index(lanes), im = S::plusOne(re);
                const typename S::M mask = S::mask(b.n);
                for (int e = 0; e < 9; ++e) {
                    S::scatter(force[e], mask, re, T[e].re);
                    S::scatter(force[e], mask, im, T[e].im);
                }
            }
        }
    }
}

} // namespace cpuSimd

//! ipdot = gauge_force(gauge, beta) on all bulk links with the AVX-512 kernel. Returns false
//! (and does nothing) where it does not apply: SIMULATEQCD_CPU_SIMD=0, odd lx, or indices
//! beyond 32 bits.
template<class floatT, size_t HaloDepth, bool onDevice>
bool cpuGaugeForce(Gaugefield<floatT, onDevice, HaloDepth, R18> &ipdot,
                   Gaugefield<floatT, onDevice, HaloDepth, R18> &gauge, const floatT beta) {
    if constexpr (!(std::is_same_v<floatT, double> || std::is_same_v<floatT, float>)) {
        return false;
    } else {
        if (!cpuSimd::enabled()) return false;
        using S = std::conditional_t<std::is_same_v<floatT, double>, cpuSimd::VecD, cpuSimd::VecF>;
        const LatticeData lat = GIndexer<All, HaloDepth>::getLatData();
        cpuSimd::ForceGeometry g;
        const size_t l[4] = {lat.lx, lat.ly, lat.lz, lat.lt};
        const size_t lFull[4] = {lat.lxFull, lat.lyFull, lat.lzFull, lat.ltFull};
        for (int i = 0; i < 4; ++i) {
            g.l[i] = static_cast<int>(l[i]);
            g.lFull[i] = static_cast<int>(lFull[i]);
            g.h[i] = static_cast<int>(lat.HaloDepth[i]);
        }
        g.vol1Full = lat.vol1Full;
        g.vol2Full = lat.vol2Full;
        g.vol3Full = lat.vol3Full;
        g.vol4Full = lat.vol4Full;
        g.sizehFull = lat.sizehFull;
        if (g.l[0] % 2 != 0 || 8 * g.vol4Full > std::numeric_limits<int32_t>::max()) return false;

        cpuProfile::Scope profile(std::is_same_v<floatT, double> ? "cpuSimd::gaugeForce<double>"
                                                                 : "cpuSimd::gaugeForce<float>");
        const auto gaugeAcc = gauge.getAccessor();
        const auto forceAcc = ipdot.getAccessor();
        floatT *force[9];
        const floatT *links[9];
        for (int e = 0; e < 9; ++e) {
            force[e] = reinterpret_cast<floatT *>(forceAcc.getEntryPointer(e));
            links[e] = reinterpret_cast<const floatT *>(gaugeAcc.getEntryPointer(e));
        }
        // r_1 as in gauge_force(): the MILC convention adds the factor 3/5.
        const floatT r1 = static_cast<floatT>(static_cast<floatT>(beta / 3.0) * 3.0 / 5.0);
        cpuSimd::gaugeForce<S>(g, force, links, r1);
        return true;
    }
}
#endif
