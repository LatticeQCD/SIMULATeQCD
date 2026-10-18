/*
 * MDWF halo exchange for 5D spinor fields on several ranks (GPU-aware MPI).
 *
 * Every MDWF spinor kernel (Wilson hop, fifth-dimension coupling, force gathers) reads at most the nearest neighbour
 * x +- mu, so of the depth-HaloDepth halo only the innermost layer (local coordinate -1 and l_mu in each split direction
 * mu) is ever read. SIMULATeQCD's generic updateAll exchanges all HaloDepth layers with generic pack / unpack kernels
 * (one thread per halo site looping over the Ls stacks; nsys at 32^3 x 8, Ls 16, 4 GPUs: pack 93 us + unpack 39 us per
 * segment, twice the time of the whole Dirac operator). This exchange
 *   - packs only the boundary layer (local mu = 0 and l_mu - 1, and only the field's parity for even/odd fields),
 *     one thread per (face site, stack), into a component-major buffer (coalesced reads and writes),
 *   - sends it with MPI_Isend / MPI_Irecv on device buffers (GPU-aware MPI) to the neighbour ranks,
 *   - unpacks into the first halo layer (local mu = -1 and l_mu) at the same memory positions as SIMULATeQCD's own
 *     inject (GIndexer::getSiteFull).
 * The outer halo layers (depth >= 2) of spinor fields are left stale; no MDWF spinor kernel reads them.
 *
 * Used by mdwfUpdateHalo (MDWFSpinor.h) for device spinor fields when several ranks and GPU-aware MPI are in use;
 * environment MDWF_FAST_HALO=0 falls back to SIMULATeQCD's updateAll(COMM_BOTH | Hyperplane).
 * Parity: even/odd fields store sites by the parity of their LOCAL coordinates; local extents are even, so the halo
 * site -1 (l_mu) has the parity of the neighbour's l_mu - 1 (0), and both sides enumerate the face identically.
 */

#pragma once

#include "../../spinor/spinorfield.h"
#include "MDWFFloatSafeWilson.h"   // chiral-basis gamma matrices (mdwfGamma*Mult), ColorVect

#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

/*
 * Half-spinor (spin-projected) halos for the overlapped Wilson hop (mdwfOverlapHop), MDWF_HALF_HALO=0 disables.
 * The hop (MDWFFusedHop) reads its halo only as (1 - gamma_mu) g psi(x + mu) (halo layer l_mu) and
 * (1 + gamma_mu) g psi(x - mu) (halo layer -1), g = gamma5 for G5In hops, else 1. In the chiral basis gamma_mu =
 * [[0, A], [B, 0]] (2x2 spin blocks, AB = 1), so w = (1 +- gamma_mu) g psi / 2 is fixed by its upper two spins u
 * (6 complex numbers instead of 12): w = (u, 0) +- gamma_mu (u, 0). The sender packs u, the receiver stores g w in the
 * halo; the hop then computes (1 -+ gamma_mu) g (g w) = (1 -+ gamma_mu) g psi, the same as with the full spinor.
 * Afterwards the halo of the exchanged field holds g w, not psi: valid only for that one hop (the fused Schur operator
 * never reads its input's halo otherwise; every other halo reader exchanges first with mdwfUpdateHalo).
 */
enum MDWFHaloMode : int { MDWFHaloFull = 0, MDWFHaloHalf = 1, MDWFHaloHalfG5 = 2 };

inline bool mdwfHalfHaloEnabled() {
    static const bool on = [] {
        const char *env = std::getenv("MDWF_HALF_HALO");
        return env == nullptr || std::string(env) != "0";
    }();
    return on;
}

// Number of floatT per (face site, stack) in the buffer.
inline int mdwfHaloComponents(int mode) { return mode == MDWFHaloFull ? 24 : 12; }

template<class floatT>
__host__ __device__ inline ColorVect<floatT> mdwfGammaMuMult(const ColorVect<floatT> &v, int mu) {
    switch (mu) {
        case 0: return mdwfGammaXMult(v);
        case 1: return mdwfGammaYMult(v);
        case 2: return mdwfGammaZMult(v);
        default: return mdwfGammaTMult(v);
    }
}

inline bool mdwfFastHaloEnabled() {
    static const bool on = [] {
        const char *env = std::getenv("MDWF_FAST_HALO");
        return env == nullptr || std::string(env) != "0";
    }();
    return on;
}

// Face geometry of direction mu: the three other directions (da fastest), their local extents, and the number of face
// sites of the field's layout (half the face for even/odd fields).
struct MDWFHaloFace {
    int mu;
    int da, db, dc;
    int na, nb, nc;
    int lmu;
    int nFace;
};

template<Layout LatLayout, size_t HaloDepth>
__device__ __host__ inline sitexyzt mdwfHaloFaceCoord(const MDWFHaloFace &f, int j, int q) {
    int a, b, c;
    if (LatLayout == All) {
        a = j % f.na;
        b = (j / f.na) % f.nb;
        c = j / (f.na * f.nb);
    } else {
        const int parity = (LatLayout == Odd) ? 1 : 0;
        const int half = f.na / 2;
        const int k = j % half;
        b = (j / half) % f.nb;
        c = j / (half * f.nb);
        a = 2 * k + ((parity + b + c + q) & 1);
    }
    sitexyzt coord(0, 0, 0, 0);
    coord[f.mu] = q;
    coord[f.da] = a;
    coord[f.db] = b;
    coord[f.dc] = c;
    return coord;
}

// Pack boundary layer mu = q (bulk coordinate) of all Ls stacks: buffer[(comp * Ls + s) * nFace + j], comp = 2 * i + re/im.
// Mode = MDWFHaloFull: the 12 spinor components. Half modes: the upper two spins (components 0..5) of
// (1 + sign gamma_mu) g psi / 2, g = gamma5 for MDWFHaloHalfG5 (see mdwfHalfHaloEnabled).
template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls, int Mode>
__global__ void mdwfHaloPackKernel(VectArrayAcc<floatT, 12> acc, floatT *buffer, MDWFHaloFace f, int q, int sign) {
    typedef GIndexer<LatLayout, HaloDepth> GInd;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t n = static_cast<size_t>(f.nFace) * Ls;
    if (idx >= n) {
        return;
    }
    const int j = static_cast<int>(idx % f.nFace);
    const int s = static_cast<int>(idx / f.nFace);
    const sitexyzt coord = mdwfHaloFaceCoord<LatLayout, HaloDepth>(f, j, q);
    const gSiteStack site = GInd::getSiteStack(GInd::getSite(coord.x, coord.y, coord.z, coord.t), s);
    Vect<floatT, 12> v = acc.getElement(site);
    constexpr int nComplex = Mode == MDWFHaloFull ? 12 : 6;
    if (Mode != MDWFHaloFull) {
        ColorVect<floatT> c = convertVect12ToColorVect(v);
        if (Mode == MDWFHaloHalfG5) {
            c = mdwfGamma5Mult(c);
        }
        ColorVect<floatT> w = static_cast<floatT>(0.5) * (c + static_cast<floatT>(sign) * mdwfGammaMuMult(c, f.mu));
        v = convertColorVectToVect12(w);
    }
#pragma unroll
    for (int i = 0; i < nComplex; i++) {
        buffer[(static_cast<size_t>(2 * i) * Ls + s) * f.nFace + j] = v(i).cREAL;
        buffer[(static_cast<size_t>(2 * i + 1) * Ls + s) * f.nFace + j] = v(i).cIMAG;
    }
}

// Unpack into halo layer mu = q (bulk coordinate -1 or l_mu). Half modes: w = (u, 0) + sign gamma_mu (u, 0) from the
// upper two spins u, stored as g w.
template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls, int Mode>
__global__ void mdwfHaloUnpackKernel(VectArrayAcc<floatT, 12> acc, const floatT *buffer, MDWFHaloFace f, int q, int sign) {
    typedef GIndexer<LatLayout, HaloDepth> GInd;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t n = static_cast<size_t>(f.nFace) * Ls;
    if (idx >= n) {
        return;
    }
    const int j = static_cast<int>(idx % f.nFace);
    const int s = static_cast<int>(idx / f.nFace);
    const sitexyzt coord = mdwfHaloFaceCoord<LatLayout, HaloDepth>(f, j, q);
    const gSiteStack site = GInd::getSiteStack(GInd::getSiteFull(GInd::coordToFullCoord(coord)), s);
    constexpr int nComplex = Mode == MDWFHaloFull ? 12 : 6;
    Vect<floatT, 12> v;
#pragma unroll
    for (int i = 0; i < 12; i++) {
        v(i) = i < nComplex ? GPUcomplex<floatT>(buffer[(static_cast<size_t>(2 * i) * Ls + s) * f.nFace + j],
                                                 buffer[(static_cast<size_t>(2 * i + 1) * Ls + s) * f.nFace + j])
                            : GPUcomplex<floatT>(0, 0);
    }
    if (Mode != MDWFHaloFull) {
        ColorVect<floatT> u = convertVect12ToColorVect(v);
        ColorVect<floatT> w = u + static_cast<floatT>(sign) * mdwfGammaMuMult(u, f.mu);
        if (Mode == MDWFHaloHalfG5) {
            w = mdwfGamma5Mult(w);
        }
        v = convertColorVectToVect12(w);
    }
    acc.setElement(site, v);
}

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
void mdwfHaloPackLaunch(int mode, unsigned grid, int block, gpuStream_t stream, VectArrayAcc<floatT, 12> acc, floatT *buffer,
                        MDWFHaloFace f, int q, int sign) {
    if (mode == MDWFHaloHalf) {
        mdwfHaloPackKernel<floatT, LatLayout, HaloDepth, Ls, MDWFHaloHalf><<<grid, block, 0, stream>>>(acc, buffer, f, q, sign);
    } else if (mode == MDWFHaloHalfG5) {
        mdwfHaloPackKernel<floatT, LatLayout, HaloDepth, Ls, MDWFHaloHalfG5><<<grid, block, 0, stream>>>(acc, buffer, f, q, sign);
    } else {
        mdwfHaloPackKernel<floatT, LatLayout, HaloDepth, Ls, MDWFHaloFull><<<grid, block, 0, stream>>>(acc, buffer, f, q, sign);
    }
}

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
void mdwfHaloUnpackLaunch(int mode, unsigned grid, int block, gpuStream_t stream, VectArrayAcc<floatT, 12> acc,
                          const floatT *buffer, MDWFHaloFace f, int q, int sign) {
    if (mode == MDWFHaloHalf) {
        mdwfHaloUnpackKernel<floatT, LatLayout, HaloDepth, Ls, MDWFHaloHalf><<<grid, block, 0, stream>>>(acc, buffer, f, q, sign);
    } else if (mode == MDWFHaloHalfG5) {
        mdwfHaloUnpackKernel<floatT, LatLayout, HaloDepth, Ls, MDWFHaloHalfG5><<<grid, block, 0, stream>>>(acc, buffer, f, q, sign);
    } else {
        mdwfHaloUnpackKernel<floatT, LatLayout, HaloDepth, Ls, MDWFHaloFull><<<grid, block, 0, stream>>>(acc, buffer, f, q, sign);
    }
}

// Device buffers shared by all exchanges (grown on demand, never freed).
inline void *mdwfHaloDeviceBuffer(int slot, size_t bytes) {
    static std::vector<void *> ptr(8, nullptr);
    static std::vector<size_t> size(8, 0);
    if (bytes > size.at(slot)) {
        if (ptr[slot] != nullptr) {
            gpuFree(ptr[slot]);
        }
        gpuError_t err = gpuMalloc(&ptr[slot], bytes);
        if (err != gpuSuccess) {
            GpuError("MDWFHaloExchange: gpuMalloc", err);
        }
        size[slot] = bytes;
    }
    return ptr[slot];
}

// One exchange in flight: what finish() needs.
struct MDWFHaloPending {
    std::vector<MDWFHaloFace> faces;
    std::vector<size_t> offset;
    std::vector<MPI_Request> requests;
    void *base = nullptr;
    bool active = false;
    int mode = MDWFHaloFull;
};

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
MDWFHaloPending mdwfFastHaloStart(Spinorfield<floatT, true, LatLayout, HaloDepth, 12, Ls> &field, gpuStream_t stream = 0,
                                  int mode = MDWFHaloFull) {
    typedef GIndexer<LatLayout, HaloDepth> GInd;
    CommunicationBase &comm = field.getComm();
    const LatticeData lat = GInd::getLatData();
    const LatticeDimensions nodes = comm.nodes();
    const LatticeDimensions me = comm.mycoords();
    const int local[4] = {static_cast<int>(lat.lx), static_cast<int>(lat.ly), static_cast<int>(lat.lz),
                          static_cast<int>(lat.lt)};
    MPI_Comm cart = comm.getCart_comm();
    MPI_Datatype type = sizeof(floatT) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT;

    MDWFHaloPending pending;
    pending.mode = mode;
    const size_t comps = static_cast<size_t>(mdwfHaloComponents(mode));
    for (int mu = 0; mu < 4; mu++) {
        if (nodes[mu] == 1) {
            continue;
        }
        MDWFHaloFace f;
        f.mu = mu;
        int other[3];
        int k = 0;
        for (int nu = 0; nu < 4; nu++) {
            if (nu != mu) {
                other[k++] = nu;
            }
        }
        f.da = other[0];
        f.db = other[1];
        f.dc = other[2];
        f.na = local[f.da];
        f.nb = local[f.db];
        f.nc = local[f.dc];
        f.lmu = local[mu];
        f.nFace = f.na * f.nb * f.nc / (LatLayout == All ? 1 : 2);
        pending.faces.push_back(f);
    }
    if (pending.faces.empty()) {
        return pending;
    }

    // Buffers: per face 4 slices (send low, send high, recv low, recv high) of comps * Ls * nFace floats.
    pending.offset.resize(pending.faces.size());
    size_t total = 0;
    for (size_t i = 0; i < pending.faces.size(); i++) {
        pending.offset[i] = total;
        total += 4 * comps * Ls * pending.faces[i].nFace;
    }
    floatT *base = static_cast<floatT *>(mdwfHaloDeviceBuffer(sizeof(floatT) == sizeof(double) ? 0 : 1,
                                                                total * sizeof(floatT)));
    pending.base = base;
    auto acc = field.getAccessor();
    const int block = 256;

    // Pack on `stream` (default: the legacy default stream, ordered after the kernels that wrote the field). The caller
    // of a non-default stream orders it after those kernels. The previous exchange's unpack ran on the same stream, so
    // the receive buffers are free once this synchronisation returns.
    for (size_t i = 0; i < pending.faces.size(); i++) {
        const MDWFHaloFace &f = pending.faces[i];
        const size_t slice = comps * Ls * f.nFace;
        const unsigned grid = static_cast<unsigned>((static_cast<size_t>(f.nFace) * Ls + block - 1) / block);
        // Layer 0 goes to the low neighbour's halo l_mu (read as (1 - gamma_mu)), layer l_mu - 1 to the high
        // neighbour's halo -1 (read as (1 + gamma_mu)).
        mdwfHaloPackLaunch<floatT, LatLayout, HaloDepth, Ls>(mode, grid, block, stream, acc, base + pending.offset[i], f, 0, -1);
        mdwfHaloPackLaunch<floatT, LatLayout, HaloDepth, Ls>(mode, grid, block, stream, acc, base + pending.offset[i] + slice,
                                                             f, f.lmu - 1, +1);
    }
    gpuError_t err = gpuStreamSynchronize(stream);
    if (err != gpuSuccess) {
        GpuError("MDWFHaloExchange: pack", err);
    }

    pending.requests.reserve(4 * pending.faces.size());
    for (size_t i = 0; i < pending.faces.size(); i++) {
        const MDWFHaloFace &f = pending.faces[i];
        const size_t slice = comps * Ls * f.nFace;
        LatticeDimensions low = me;
        LatticeDimensions high = me;
        low[f.mu] = (me[f.mu] - 1 + nodes[f.mu]) % nodes[f.mu];
        high[f.mu] = (me[f.mu] + 1) % nodes[f.mu];
        const int rankLow = comm.getRank(low);
        const int rankHigh = comm.getRank(high);
        const int count = static_cast<int>(slice);
        // Tags: 2 mu + 0 = data travelling downwards (my mu = 0 layer -> low neighbour's halo l_mu),
        //       2 mu + 1 = data travelling upwards (my mu = l_mu - 1 layer -> high neighbour's halo -1).
        MPI_Request r;
        MPI_Irecv(base + pending.offset[i] + 2 * slice, count, type, rankLow, 2 * f.mu + 1, cart, &r);   // -> halo -1
        pending.requests.push_back(r);
        MPI_Irecv(base + pending.offset[i] + 3 * slice, count, type, rankHigh, 2 * f.mu + 0, cart, &r);  // -> halo l_mu
        pending.requests.push_back(r);
        MPI_Isend(base + pending.offset[i], count, type, rankLow, 2 * f.mu + 0, cart, &r);
        pending.requests.push_back(r);
        MPI_Isend(base + pending.offset[i] + slice, count, type, rankHigh, 2 * f.mu + 1, cart, &r);
        pending.requests.push_back(r);
    }
    pending.active = true;
    return pending;
}

// Wait for the transfers and unpack (on `stream`, default: legacy default stream).
template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
void mdwfFastHaloFinish(Spinorfield<floatT, true, LatLayout, HaloDepth, 12, Ls> &field, MDWFHaloPending &pending,
                        gpuStream_t stream = 0) {
    if (!pending.active) {
        return;
    }
    MPI_Waitall(static_cast<int>(pending.requests.size()), pending.requests.data(), MPI_STATUSES_IGNORE);
    floatT *base = static_cast<floatT *>(pending.base);
    auto acc = field.getAccessor();
    const int block = 256;
    const size_t comps = static_cast<size_t>(mdwfHaloComponents(pending.mode));
    for (size_t i = 0; i < pending.faces.size(); i++) {
        const MDWFHaloFace &f = pending.faces[i];
        const size_t slice = comps * Ls * f.nFace;
        const unsigned grid = static_cast<unsigned>((static_cast<size_t>(f.nFace) * Ls + block - 1) / block);
        mdwfHaloUnpackLaunch<floatT, LatLayout, HaloDepth, Ls>(pending.mode, grid, block, stream, acc,
                                                               base + pending.offset[i] + 2 * slice, f, -1, +1);
        mdwfHaloUnpackLaunch<floatT, LatLayout, HaloDepth, Ls>(pending.mode, grid, block, stream, acc,
                                                               base + pending.offset[i] + 3 * slice, f, f.lmu, -1);
    }
    gpuError_t err = gpuGetLastError();
    if (err != gpuSuccess) {
        GpuError("MDWFHaloExchange: unpack launch", err);
    }
    pending.active = false;
}

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
void mdwfFastHaloExchange(Spinorfield<floatT, true, LatLayout, HaloDepth, 12, Ls> &field) {
    MDWFHaloPending pending = mdwfFastHaloStart(field);
    mdwfFastHaloFinish(field, pending);
}

// MDWF_OVERLAP_PRIO=0: overlapped hop packs on the default stream and launches the interior after the pack (job 2228876).
inline bool mdwfOverlapPriorityEnabled() {
    static const bool on = [] {
        const char *env = std::getenv("MDWF_OVERLAP_PRIO");
        return env == nullptr || std::string(env) != "0";
    }();
    return on;
}

inline bool mdwfOverlapEnabled() {
    static const bool on = [] {
        const char *env = std::getenv("MDWF_OVERLAP");
        return env == nullptr || std::string(env) != "0";
    }();
    return on;
}

// Bulk site indices (GIndexer<LatLayout, HaloDepth> isite) whose nearest neighbours in all split directions are local
// (interior) or not (boundary), built once per layout on the host.
template<Layout LatLayout, size_t HaloDepth>
struct MDWFSiteLists {
    unsigned *interior = nullptr;
    unsigned *boundary = nullptr;
    size_t nInterior = 0;
    size_t nBoundary = 0;

    static MDWFSiteLists &get(const LatticeDimensions &nodes) {
        static MDWFSiteLists lists;
        if (lists.interior == nullptr && lists.boundary == nullptr) {
            lists.build(nodes);
        }
        return lists;
    }

    void build(const LatticeDimensions &nodes) {
        typedef GIndexer<LatLayout, HaloDepth> GInd;
        const LatticeData lat = GInd::getLatData();
        const int local[4] = {static_cast<int>(lat.lx), static_cast<int>(lat.ly), static_cast<int>(lat.lz),
                              static_cast<int>(lat.lt)};
        const size_t n = LatLayout == All ? lat.vol4 : lat.sizeh;
        std::vector<unsigned> in, bd;
        for (size_t i = 0; i < n; i++) {
            const gSite site = GInd::getSite(i);
            bool boundary = false;
            for (int mu = 0; mu < 4; mu++) {
                const int x = sitexyzt(site.coord)[mu];
                if (nodes[mu] > 1 && (x == 0 || x == local[mu] - 1)) {
                    boundary = true;
                }
            }
            (boundary ? bd : in).push_back(static_cast<unsigned>(i));
        }
        nInterior = in.size();
        nBoundary = bd.size();
        gpuMalloc(reinterpret_cast<void **>(&interior), std::max<size_t>(1, nInterior) * sizeof(unsigned));
        gpuMalloc(reinterpret_cast<void **>(&boundary), std::max<size_t>(1, nBoundary) * sizeof(unsigned));
        gpuMemcpy(interior, in.data(), nInterior * sizeof(unsigned), gpuMemcpyHostToDevice);
        gpuMemcpy(boundary, bd.data(), nBoundary * sizeof(unsigned), gpuMemcpyHostToDevice);
        rootLogger.info("MDWF overlap: layout ", static_cast<int>(LatLayout), ", interior sites ", nInterior,
                        ", boundary sites ", nBoundary);
    }
};

// Thread (x, stack) -> the x-th site of a site list, at stack threadIdx.y (as CalcGSiteStack).
template<Layout LatLayout, size_t HaloDepth>
struct MDWFCalcListSiteStack {
    const unsigned *list;
    __device__ __host__ gSiteStack operator()(const dim3 &blockDim, const uint3 &blockIdx, const uint3 &threadIdx) {
        typedef GIndexer<LatLayout, HaloDepth> GInd;
        const size_t i = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
        return GInd::getSiteStack(GInd::getSite(static_cast<size_t>(list[i])), threadIdx.y);
    }
};

#ifdef USE_HIP
#define MDWF_STREAM_CREATE_NONBLOCKING(s) hipStreamCreateWithFlags(s, hipStreamNonBlocking)
#define MDWF_STREAM_CREATE_HIGHEST_PRIORITY(s)                                                                         \
    do {                                                                                                               \
        int lo_, hi_;                                                                                                  \
        hipDeviceGetStreamPriorityRange(&lo_, &hi_);                                                                   \
        hipStreamCreateWithPriority(s, hipStreamNonBlocking, hi_);                                                     \
    } while (0)
#else
#define MDWF_STREAM_CREATE_NONBLOCKING(s) cudaStreamCreateWithFlags(s, cudaStreamNonBlocking)
#define MDWF_STREAM_CREATE_HIGHEST_PRIORITY(s)                                                                         \
    do {                                                                                                               \
        int lo_, hi_;                                                                                                  \
        cudaDeviceGetStreamPriorityRange(&lo_, &hi_);                                                                  \
        cudaStreamCreateWithPriority(s, cudaStreamNonBlocking, hi_);                                                   \
    } while (0)
#endif

// Streams / events of the overlapped hop: stream (interior and boundary sites), comm (pack / unpack, highest priority,
// so its kernels get SMs ahead of the queued interior blocks).
struct MDWFOverlapStreams {
    gpuStream_t stream;
    gpuStream_t comm;
    gpuEvent_t ready;
    gpuEvent_t unpacked;
    gpuEvent_t done;
    static MDWFOverlapStreams &get() {
        static MDWFOverlapStreams s = [] {
            MDWFOverlapStreams t;
            MDWF_STREAM_CREATE_NONBLOCKING(&t.stream);
            MDWF_STREAM_CREATE_HIGHEST_PRIORITY(&t.comm);
            gpuEventCreateWithFlags(&t.ready, gpuEventDisableTiming);
            gpuEventCreateWithFlags(&t.unpacked, gpuEventDisableTiming);
            gpuEventCreateWithFlags(&t.done, gpuEventDisableTiming);
            return t;
        }();
        return s;
    }
};

/*
 * out = op over all bulk sites of out (op reads in at x and x +- mu), with in's halo exchanged by the MDWF exchange and
 * the interior sites computed while the halo data is in flight:
 *   pack in, post MPI | interior sites on a non-blocking stream | MPI wait, unpack (default stream)
 *   | boundary sites on the same stream after the unpack | default stream waits for them.
 * Falls back to "halo update, then op over all sites" on one rank, without GPU-aware MPI, or with MDWF_OVERLAP=0.
 */
// (defined in MDWFSpinor.h after mdwfUpdateHalo, which it uses as fallback)
