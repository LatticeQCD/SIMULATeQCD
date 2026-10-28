/*
 * cpuHaloCopy.h
 *
 * Halo pack (ExtractInnerHalo) and unpack (InjectOuterHalo) for the CPU backend as plain copies.
 * The functor path computes the halo and lattice coordinates of every site on every call; at
 * 80^3x8 that ran at ~20 GB/s (0.24 ms per 4.9 MB spinor halo, job 3846763). Here the mapping
 * between halo buffer and field is computed once per field type and geometry, with the same index
 * functors, and stored as runs of consecutive face sites (see Runs).
 * Every copy moves the same values to the same places as the functors, so results are
 * bit-identical. SIMULATEQCD_CPU_HALO=0 switches back to the functors.
 */
#pragma once

#ifdef USE_CPU
#include "calcGSiteHalo_dynamic.h"
#include "../indexer/haloIndexer.h"
#include "../wrapper/cpuProfile.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace cpuHalo {

inline bool enabled() {
    static const bool on = [] {
        const char *value = std::getenv("SIMULATEQCD_CPU_HALO");
        return !value || std::string_view(value) != "0";
    }();
    return on;
}

//! Copies between the halo buffer and the field, per run of len halo sites: buffer element
//! dst + k*ElemCount + mu of entry e (entry arrays stride apart) <-> field element
//! src + k + mu*muStride of entry array e, for k < len and mu < ElemCount. Within a run the
//! sites are consecutive in the field (face sites are stored row by row), and the ElemCount
//! links/stacks of a site are muStride apart for every site (else valid = false).
struct Runs {
    static constexpr size_t piece = 512;
    size_t key[7] = {};
    bool built = false, valid = false;
    size_t muStride = 0;
    std::vector<size_t> src, dst, stride, len;
};

//! The runs for packing (inner = true) or unpacking (inner = false) the halos of a field with
//! accessor type Accessor, cached per geometry. Built with the index functors of the
//! ExtractInnerHalo / InjectOuterHalo kernels.
template<class floatT, class Accessor, size_t EntryCount, size_t ElemCount, Layout LatLayout, size_t HaloDepth, bool inner>
Runs &runs(const Accessor &acc) {
    typedef GIndexer<LatLayout, HaloDepth> GInd;
    typedef HaloIndexer<LatLayout, HaloDepth> HInd;
    static Runs r;
    const LatticeData lat = GInd::getLatData();
    const size_t haloSites = HInd::getBufferSize();
    const size_t key[7] = {lat.lx, lat.ly, lat.lz, lat.lt, lat.vol4Full, lat.sizehFull, haloSites};
    if (r.built && std::equal(key, key + 7, r.key)) return r;

    cpuProfile::Scope profile("halo copy table");
    r = Runs();
    std::copy(key, key + 7, r.key);
    r.muStride = ElemCount > 1 ? acc.template getIndexComm<LatLayout, HaloDepth>(0, 1)
                                 - acc.template getIndexComm<LatLayout, HaloDepth>(0, 0) : 0;
    std::vector<HaloSite> sites(haloSites);
    std::vector<size_t> src(haloSites);
    bool regular = true;
    const dim3 blockDim(1, 1, 1);
#pragma omp parallel for schedule(static) reduction(&&:regular)
    for (size_t i = 0; i < haloSites; ++i) {
        const uint3 blockIdx{static_cast<unsigned>(i), 0, 0}, threadIdx{0, 0, 0};
        if constexpr (inner) {
            sites[i] = CalcInnerHaloIndexComm<floatT, LatLayout, HaloDepth>()(blockDim, blockIdx, threadIdx);
        } else {
            sites[i] = CalcOuterHaloIndexComm<floatT, LatLayout, HaloDepth>()(blockDim, blockIdx, threadIdx);
        }
        src[i] = acc.template getIndexComm<LatLayout, HaloDepth>(sites[i].LatticeIndex, 0);
        for (size_t mu = 1; mu < ElemCount; ++mu) {
            regular = regular && acc.template getIndexComm<LatLayout, HaloDepth>(sites[i].LatticeIndex, mu)
                                 == src[i] + mu * r.muStride;
        }
    }
    r.built = true;
    r.valid = regular;
    if (!regular) return r;
    for (size_t i = 0; i < haloSites; ++i) {
        const HaloSite &site = sites[i];
        const size_t dst = HInd::get_SubHaloOffset(site.HalNumber) * EntryCount * ElemCount
                           + site.LocHalIndex * ElemCount;
        if (i > 0 && site.HalNumber == sites[i - 1].HalNumber && r.len.back() < Runs::piece
            && r.src.back() + r.len.back() == src[i] && r.dst.back() + r.len.back() * ElemCount == dst) {
            r.len.back()++;
        } else {
            r.src.push_back(src[i]);
            r.dst.push_back(dst);
            r.stride.push_back(HInd::get_SubHaloSize(site.HalNumber) * ElemCount);
            r.len.push_back(1);
        }
    }
    return r;
}

//! Pack (inner = true: buffer <- field) or unpack (inner = false: field <- buffer). Returns
//! false (and copies nothing) if the field layout does not fit Runs.
template<class floatT, class Accessor, size_t EntryCount, size_t ElemCount, Layout LatLayout, size_t HaloDepth, bool inner>
bool copy(const Accessor &acc, COMPLEX(floatT) *buffer, std::string_view name) {
    Runs &r = runs<floatT, Accessor, EntryCount, ElemCount, LatLayout, HaloDepth, inner>(acc);
    if (!r.valid) return false;
    cpuProfile::Scope profile(name);
    COMPLEX(floatT) *field[EntryCount];
    for (size_t e = 0; e < EntryCount; ++e) field[e] = acc.getEntryPointer(e);
    const size_t n = r.len.size(), muStride = r.muStride;
#pragma omp parallel for schedule(static)
    for (size_t p = 0; p < n; ++p) {
        const size_t src = r.src[p], dst = r.dst[p], stride = r.stride[p], len = r.len[p];
        for (size_t e = 0; e < EntryCount; ++e) {
            COMPLEX(floatT) *f = field[e] + src;
            COMPLEX(floatT) *b = buffer + dst + e * stride;
            if constexpr (ElemCount == 1) {
                if constexpr (inner) std::memcpy(static_cast<void *>(b), f, len * sizeof(COMPLEX(floatT)));
                else std::memcpy(static_cast<void *>(f), b, len * sizeof(COMPLEX(floatT)));
            } else {
                // Site by site, so the buffer is written (or read) contiguously.
                for (size_t k = 0; k < len; ++k) {
                    for (size_t mu = 0; mu < ElemCount; ++mu) {
                        if constexpr (inner) b[k * ElemCount + mu] = f[k + mu * muStride];
                        else f[k + mu * muStride] = b[k * ElemCount + mu];
                    }
                }
            }
        }
    }
    return true;
}

//! A halo exchange that was only started (updateAll with COMM_START). All fields share the halo
//! buffers (which LatticeContainer also uses for reductions), so at most one may be running:
//! SiteComm finishes it before any other exchange starts. Between start and finish nothing may
//! reduce over a LatticeContainer either; that is not checked.
struct PendingExchange {
    virtual void finishPendingExchange() = 0;
    static PendingExchange *&current() {
        static PendingExchange *pending = nullptr;
        return pending;
    }
    static void finishCurrent() {
        if (current()) current()->finishPendingExchange();
    }
};

} // namespace cpuHalo
#endif
