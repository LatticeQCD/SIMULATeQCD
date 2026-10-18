/*
 * MDWF 5D spinor scaffold.
 *
 * This header gives the MDWF code path a named 5D spinor representation
 * without introducing a new storage layout.  The existing Spinorfield stack
 * dimension is interpreted as the physical fifth dimension Ls only through
 * these MDWF aliases.
 */

#pragma once

#include "../../spinor/spinorfield.h"
#include "MDWFHaloExchange.h"

#include <cstdlib>
#include <string>
#include <type_traits>

template<size_t Ls>
struct MDWFSpinorLayout {
    static_assert(Ls > 0, "MDWF fifth dimension Ls must be positive");

    // Wilson spinors use 12 complex components per 4D site: 4 spin x 3 color.
    static constexpr size_t spin_color_components = 12;

    // In MDWF code, this is the physical fifth dimension, not independent RHS.
    static constexpr size_t fifth_dimension = Ls;
};

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t Ls>
using MDWFSpinor = Spinorfield<floatT,
                               onDevice,
                               LatLayout,
                               HaloDepth,
                               MDWFSpinorLayout<Ls>::spin_color_components,
                               MDWFSpinorLayout<Ls>::fifth_dimension>;

template<class floatT, bool onDevice, size_t HaloDepth, size_t Ls>
using MDWFSpinorAll = SpinorfieldAll<floatT,
                                     onDevice,
                                     HaloDepth,
                                     MDWFSpinorLayout<Ls>::spin_color_components,
                                     MDWFSpinorLayout<Ls>::fifth_dimension>;

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct MDWFSpinorTypes {
    using Field = MDWFSpinor<floatT, onDevice, LatLayout, HaloDepth, Ls>;
    using EvenOddField = MDWFSpinorAll<floatT, onDevice, HaloDepth, Ls>;

    static constexpr size_t spin_color_components = MDWFSpinorLayout<Ls>::spin_color_components;
    static constexpr size_t fifth_dimension = MDWFSpinorLayout<Ls>::fifth_dimension;
};

/*
 * Name of a scratch spinor shared by every MDWF object that asks for the same role: names starting with SHARED_ map
 * to one buffer in MemoryManagement (src/base/memoryManagement.h). For work fields that hold nothing between calls
 * (operator temporaries, solver/force scratch), so that k Hasenbusch factors, the strange RHMC and their operators do
 * not each own a private copy (at 32^3 x 8, Ls = 16 an even-site field is 0.86 GB). The name carries precision,
 * layout, halo depth and Ls, so one buffer never serves two field sizes. Two fields used in the same call must have
 * different roles.
 */
template<class floatT, Layout LatLayout, size_t HaloDepth, size_t Ls>
std::string mdwfSharedSpinorName(const std::string &role) {
    return std::string("SHARED_MDWF_") + (std::is_same<floatT, double>::value ? "d" : "f") + "_L"
           + std::to_string(static_cast<int>(LatLayout)) + "_h" + std::to_string(HaloDepth) + "_s" + std::to_string(Ls)
           + "_" + role;
}

// Template parameters of a device spinor field type.
template<class Spinor>
struct MDWFSpinorTraits;

template<class floatT_, Layout L, size_t H, size_t E, size_t N>
struct MDWFSpinorTraits<Spinorfield<floatT_, true, L, H, E, N>> {
    using floatT = floatT_;
    static constexpr Layout layout = L;
    static constexpr size_t halo = H;
    static constexpr size_t elems = E;
    static constexpr size_t stacks = N;
};

/*
 * Halo update used by all MDWF code. Spinor fields exchange only the hyperplane (face) segments: every MDWF spinor kernel
 * (Wilson hop, fifth-dimension coupling, force gathers chi(x) / eta(x + mu)) reads at most one 4D direction outside the
 * local volume, so the plane / stripe / corner segments (needed only for diagonal neighbours x + mu + nu) are never read.
 * With two or more split directions this removes the edge and corner messages. Any other field (gauge links, momenta,
 * forces: plaquette / rectangle kernels do read diagonal neighbours) keeps the full update. Environment
 * MDWF_HALO_ALLTYPES=1 restores the full update for spinors too (A/B check).
 */
inline bool mdwfSpinorHaloAllTypes() {
    static const bool all = [] {
        const char *env = std::getenv("MDWF_HALO_ALLTYPES");
        return env != nullptr && std::string(env) == "1";
    }();
    return all;
}

template<class Field>
inline void mdwfUpdateHalo(Field &field) {
    field.updateAll();
}

template<class floatT, bool onDevice, Layout L, size_t H, size_t E, size_t N>
inline void mdwfUpdateHalo(Spinorfield<floatT, onDevice, L, H, E, N> &field) {
    if (mdwfSpinorHaloAllTypes()) {
        field.updateAll();
    } else {
        field.updateAll(COMM_BOTH | Hyperplane);
    }
}

// Device Wilson-type spinors (12 components) on several ranks with GPU-aware MPI: the MDWF exchange of the innermost
// halo layer only (MDWFHaloExchange.h); MDWF_FAST_HALO=0 or MDWF_HALO_ALLTYPES=1 fall back to SIMULATeQCD's updateAll.
template<class floatT, Layout L, size_t H, size_t N>
inline void mdwfUpdateHalo(Spinorfield<floatT, true, L, H, 12, N> &field) {
    CommunicationBase &comm = field.getComm();
    if (mdwfSpinorHaloAllTypes()) {
        field.updateAll();
    } else if (mdwfFastHaloEnabled() && comm.getNumberProcesses() > 1 && comm.gpuAwareMPIAvail() && !comm.useGpuP2P()) {
        mdwfFastHaloExchange(field);
    } else {
        field.updateAll(COMM_BOTH | Hyperplane);
    }
}

template<unsigned BlockSize, class floatT, Layout LOut, Layout LIn, size_t HaloDepth, size_t Ls, class Functor>
void mdwfOverlapHop(Spinorfield<floatT, true, LOut, HaloDepth, 12, Ls> &out,
                    Spinorfield<floatT, true, LIn, HaloDepth, 12, Ls> &in, Functor op, int halfMode = MDWFHaloFull) {
    CommunicationBase &comm = in.getComm();
    // halfMode (MDWFHaloHalf / MDWFHaloHalfG5): op is a Wilson hop that reads the halo only spin-projected
    // (MDWFFusedHop, G5In -> HalfG5); MDWF_HALF_HALO=0 sends full spinors.
    const int haloMode = mdwfHalfHaloEnabled() ? halfMode : static_cast<int>(MDWFHaloFull);
    if (!(mdwfOverlapEnabled() && mdwfFastHaloEnabled() && !mdwfSpinorHaloAllTypes() && comm.getNumberProcesses() > 1
          && comm.gpuAwareMPIAvail() && !comm.useGpuP2P())) {
        mdwfUpdateHalo(in);
        out.template iterateOverBulk<BlockSize>(op);
        return;
    }
    MDWFSiteLists<LOut, HaloDepth> &lists = MDWFSiteLists<LOut, HaloDepth>::get(comm.nodes());
    MDWFOverlapStreams &st = MDWFOverlapStreams::get();
    WriteAtReadStack writeAtRead;

    if (mdwfOverlapPriorityEnabled()) {
        // Both streams start after the default-stream work that wrote in (and last read out). The interior is queued
        // first, so the GPU never idles while the host waits for the pack and MPI; the pack / unpack run on the
        // highest-priority stream and take SMs ahead of the queued interior blocks. (Same priority, job 2229615: the
        // interior starved the pack, the MPI transfer started only after it, 4 GPUs 2.13 -> 2.44 ms per CG iteration.)
        // The interior reads only bulk sites of in, the unpack writes only its halo: no conflict.
        gpuEventRecord(st.ready, 0);
        gpuStreamWaitEvent(st.comm, st.ready, 0);
        gpuStreamWaitEvent(st.stream, st.ready, 0);
        if (lists.nInterior > 0) {
            out.template iterateFunctor<BlockSize>(op, MDWFCalcListSiteStack<LOut, HaloDepth>{lists.interior},
                                                   writeAtRead, lists.nInterior, Ls, 1, st.stream);
        }
        MDWFHaloPending pending = mdwfFastHaloStart(in, st.comm, haloMode);   // syncs only the comm stream (the pack)
        mdwfFastHaloFinish(in, pending, st.comm);                    // MPI wait, unpack on the comm stream
        gpuEventRecord(st.unpacked, st.comm);
        gpuStreamWaitEvent(st.stream, st.unpacked, 0);
        if (lists.nBoundary > 0) {
            out.template iterateFunctor<BlockSize>(op, MDWFCalcListSiteStack<LOut, HaloDepth>{lists.boundary},
                                                   writeAtRead, lists.nBoundary, Ls, 1, st.stream);
        }
        gpuEventRecord(st.done, st.stream);
        gpuStreamWaitEvent(0, st.done, 0);
        return;
    }

    // Pack first: launching the interior before the pack (job 2229615) starves the pack kernels of SMs, the MPI transfer
    // then starts only after the interior and the overlap is lost (4 GPUs 2.13 -> 2.44 ms per CG iteration).
    MDWFHaloPending pending = mdwfFastHaloStart(in, 0, haloMode);   // ends with a sync of the default stream: in and out are final
    if (lists.nInterior > 0) {
        out.template iterateFunctor<BlockSize>(op, MDWFCalcListSiteStack<LOut, HaloDepth>{lists.interior}, writeAtRead,
                                               lists.nInterior, Ls, 1, st.stream);
    }
    mdwfFastHaloFinish(in, pending);                    // MPI wait, unpack on the default stream
    gpuEventRecord(st.unpacked, 0);
    gpuStreamWaitEvent(st.stream, st.unpacked, 0);
    if (lists.nBoundary > 0) {
        out.template iterateFunctor<BlockSize>(op, MDWFCalcListSiteStack<LOut, HaloDepth>{lists.boundary}, writeAtRead,
                                               lists.nBoundary, Ls, 1, st.stream);
    }
    gpuEventRecord(st.done, st.stream);
    gpuStreamWaitEvent(0, st.done, 0);                 // later default-stream work sees the complete out
}
