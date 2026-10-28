/*
 * laneSite.h
 *
 * Site types for the site-vectorised CPU kernels (AVX-512): a LaneSite is a block of W bulk
 * sites of one parity in one x-row, lane l at x = c[0] + 2l, so a functor written for gSite can
 * run on W sites at once with SU3<cpuSimd::Lane<floatT>> in place of SU3<floatT>. GIndexer moves
 * a LaneSite like a gSite (periodic on the full lattice); SU3Accessor loads its W links with one
 * stride-1 load per entry, or gathers them where the row wraps around in x or the block is
 * partial. siteReal_t<floatT, Site> is the real type a functor computes in at that site type:
 * floatT for gSite and gSiteMu, Lane<floatT> for the lane types.
 */
#pragma once

#include <cstdint>

#if defined(USE_CPU) && defined(__AVX512F__)
#define CPU_LANE_SIMD 1

namespace cpuSimd {

template<class T> struct Lane;

//! Extents of the local lattice: bulk, halo depth and full (bulk + halos) per direction.
struct LaneGeometry {
    int l[4], h[4], lFull[4];
    int64_t vol1Full, vol2Full, vol3Full, vol4Full, sizehFull;

    //! Index of the link in direction mu at full coordinates c, as coordMuToIndexMu_Full.
    int64_t index(const int c[4], int mu) const {
        return ((c[0] + c[1] * vol1Full + c[2] * vol2Full + c[3] * vol3Full) >> 1)
               + sizehFull * ((c[0] + c[1] + c[2] + c[3]) & 1) + mu * vol4Full;
    }
};

inline int wrapLaneCoord(int x, int extent) { return x < 0 ? x + extent : (x >= extent ? x - extent : x); }

//! n valid lanes; full coordinates of lane 0. c[1..3] are kept periodic, c[0] is wrapped per
//! lane when the links are loaded.
struct LaneSite {
    const LaneGeometry *g;
    int c[4];
    int n;

    LaneSite moved(int mu, int steps) const {
        LaneSite s = *this;
        if (mu == 0) {
            s.c[0] += steps;
        } else {
            s.c[mu] = wrapLaneCoord(s.c[mu] + steps, g->lFull[mu]);
        }
        return s;
    }
};

struct LaneSiteMu : LaneSite {
    int mu;
};

} // namespace cpuSimd
#endif

//! True for the lane site types; functors that read other than R18 fields exclude them.
template<class Site> inline constexpr bool isLaneSite = false;
#ifdef CPU_LANE_SIMD
template<> inline constexpr bool isLaneSite<cpuSimd::LaneSite> = true;
template<> inline constexpr bool isLaneSite<cpuSimd::LaneSiteMu> = true;
#endif

template<class floatT, class Site> struct SiteReal { using type = floatT; };
#ifdef CPU_LANE_SIMD
template<class floatT> struct SiteReal<floatT, cpuSimd::LaneSite> { using type = cpuSimd::Lane<floatT>; };
template<class floatT> struct SiteReal<floatT, cpuSimd::LaneSiteMu> { using type = cpuSimd::Lane<floatT>; };
#endif
template<class floatT, class Site> using siteReal_t = typename SiteReal<floatT, Site>::type;
