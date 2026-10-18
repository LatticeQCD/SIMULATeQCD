/*
 * Fermion boundary conditions in time for MDWF (shared by the HMC driver,
 * MDWFHmc.h, and the residual-mass measurement, MDWFResidualMass.h).
 *
 * Antiperiodic temporal boundary conditions for the fermions are imposed by
 * giving the Dirac operator a copy of the gauge field whose temporal links on
 * the last time slice carry a factor -1 (the hop from t = Lt - 1 to t = 0
 * picks up the sign). The gauge action keeps the thin field; the clover term
 * is unchanged, since every plaquette crossing the boundary contains two such
 * links.
 *
 * Force: any fermion force computed as TA(U' dS/dU') with ALL links (the
 * differentiated one and those in paths and spinor transport) taken from the
 * copy U' equals the force TA(U dS/dU) with respect to the thin links U,
 * because U' = z U with z = +-1 per link gives dS/dU = z dS/dU' and z^2 = 1.
 * The MDWF force storage (MDWFDeviceForceStorage.h and the host paths) reads
 * only the gauge field it is given, so the HMC passes it the copy.
 *
 * Single rank.
 */

#pragma once

#include "MDWFSpinor.h"

#include "../../gauge/gaugefield.h"

#include <stdexcept>
#include <string>

// Copy of the gauge field with U_t(x) -> -U_t(x) on the last GLOBAL time slice (antiperiodic fermion BCs in time; with
// the lattice split in t only the rank holding t = globLT - 1 flips, as staggeredPhases.h).
template<size_t HaloDepth>
struct MDWFAntiperiodicTimeLinks {
    SU3Accessor<double, R18> _gauge;
    int _globalTLast;

    MDWFAntiperiodicTimeLinks(Gaugefield<double, true, HaloDepth, R18> &gauge, int globalTLast)
        : _gauge(gauge.getAccessor()), _globalTLast(globalTLast) {}

    __host__ __device__ SU3<double> operator()(gSiteMu siteMu) {
        typedef GIndexer<All, HaloDepth> GInd;
        const SU3<double> link = _gauge.getLink(siteMu);
        LatticeData lat = GInd::getLatData();
        if (siteMu.mu == 3 && static_cast<int>(lat.globalPos(siteMu.coord).t) == _globalTLast) {
            return static_cast<double>(-1.0) * link;
        }
        return link;
    }
};

template<size_t HaloDepth>
void mdwfFermionGaugeField(Gaugefield<double, true, HaloDepth, R18> &fermionGauge,
                           Gaugefield<double, true, HaloDepth, R18> &gauge, bool antiperiodicTime) {
    typedef GIndexer<All, HaloDepth> GInd;
    const LatticeData lat = GInd::getLatData();
    if (antiperiodicTime) {
        fermionGauge.iterateOverBulkAllMu(MDWFAntiperiodicTimeLinks<HaloDepth>(gauge, static_cast<int>(lat.globLT) - 1));
    } else {
        fermionGauge = gauge;
    }
    mdwfUpdateHalo(fermionGauge);
}

// Gauge field seen by the fermions: the thin field itself (periodic) or a -U_t copy on the last time slice
// (antiperiodic), refreshed from the thin field by sync().
template<size_t HaloDepth>
class MDWFFermionGauge {
public:
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;

private:
    Gauge &_thin;
    Gauge _copy;
    bool _antiperiodic;

public:
    MDWFFermionGauge(CommunicationBase &commBase, Gauge &thin, bool antiperiodicTime, const std::string &name)
        : _thin(thin), _copy(commBase, name), _antiperiodic(antiperiodicTime) {
        sync();
    }

    // Recompute the copy from the thin field (no-op for periodic BCs, where get() is the thin field).
    void sync() {
        if (_antiperiodic) {
            mdwfFermionGaugeField<HaloDepth>(_copy, _thin, true);
        }
    }

    Gauge &get() {
        return _antiperiodic ? _copy : _thin;
    }

    bool antiperiodic() const {
        return _antiperiodic;
    }
};
