/*
 * laneIterate.h
 *
 * Site-vectorised iterateOverBulkAllMu for the CPU backend: a functor whose operator() also takes
 * a cpuSimd::LaneSiteMu (see laneSite.h, simdLane.h) is called once per block of W bulk sites of
 * one parity in one x-row and direction, and its SU3<Lane<floatT>> result is stored to the W links.
 * Blocks are ordered row, parity, position in the row, with the four directions of a block
 * together, as in cpuGaugeForce.h.
 */
#pragma once

#include "../base/math/su3Accessor.h"

#ifdef CPU_LANE_SIMD
#include "../base/wrapper/cpuProfile.h"
#include <limits>
#include <cstdlib>
#include <string>
#include <string_view>
#include <type_traits>

namespace cpuSimd {

//! Switch for the lane path alone: on unless SIMULATEQCD_CPU_LANES=0 (SIMULATEQCD_CPU_SIMD=0 also turns it off).
inline bool lanesEnabled() {
    static const bool on = [] {
        const char *value = std::getenv("SIMULATEQCD_CPU_LANES");
        return !value || std::string_view(value) != "0";
    }();
    return on && enabled();
}

template<class Functor, class floatT>
inline constexpr bool laneFunctor = std::is_invocable_r_v<SU3<Lane<floatT>>, Functor &, const LaneSiteMu &>;

//! out = op on every bulk link. Returns false (and does nothing) where it does not apply:
//! lanes switched off, odd lx, or indices beyond 32 bits.
template<size_t HaloDepth, class floatT, class Functor>
bool iterateLanesBulkAllMu(SU3Accessor<floatT, R18> out, Functor op) {
    if (!lanesEnabled()) return false;
    const LatticeData lat = GIndexer<All, HaloDepth>::getLatData();
    LaneGeometry g;
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

    static const std::string name = "lanes " + std::string(type_name<Functor>());
    cpuProfile::Scope profile(name);
    constexpr int W = Lane<floatT>::W;
    const int perRow = g.l[0] / 2;
    const int blocksPerRow = (perRow + W - 1) / W;
    const size_t rows = static_cast<size_t>(g.l[1]) * g.l[2] * g.l[3];
    const size_t nBlocks = rows * 2 * blocksPerRow;
    // Each thread works on its own copy of the functor, as each GPU thread gets its own kernel arguments.
#pragma omp parallel firstprivate(op)
    {
#pragma omp for schedule(static)
        for (size_t block = 0; block < nBlocks; ++block) {
            const int j = static_cast<int>(block % blocksPerRow);
            const int parity = static_cast<int>((block / blocksPerRow) % 2);
            const size_t row = block / (2 * blocksPerRow);
            LaneSiteMu s;
            s.g = &g;
            s.c[1] = static_cast<int>(row % g.l[1]) + g.h[1];
            s.c[2] = static_cast<int>((row / g.l[1]) % g.l[2]) + g.h[2];
            s.c[3] = static_cast<int>(row / (static_cast<size_t>(g.l[1]) * g.l[2])) + g.h[3];
            s.c[0] = g.h[0] + ((parity - s.c[1] - s.c[2] - s.c[3] - g.h[0]) & 1) + 2 * W * j;
            s.n = std::min(W, perRow - W * j);
            for (int mu = 0; mu < 4; ++mu) {
                s.mu = mu;
                out.setLink(s, op(s));
            }
        }
    }
    return true;
}

} // namespace cpuSimd
#endif
