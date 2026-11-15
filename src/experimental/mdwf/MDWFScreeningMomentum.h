/*
 * Screening correlators at non-zero TRANSVERSE momentum (2026-11-12), for the isolated-state test of the sigma
 * (and pi, delta, eta) screening masses: a single screening state obeys E(p_perp)^2 = M^2 + p_perp^2 (rotation
 * symmetry among x, y, z), a two-particle state does not (finite-volume levels of e.g. pi pi at total p_perp).
 *
 * Connected part (point source x0, the production column store, no extra solves): per site and source colour the
 * complex density of channel G
 *     G(x) = tr[S(x,x0)^+ L S(x,x0) R],  L = g5 G, R = G^+ g5     (the same contraction as MDWFMesonChannelDensity,
 *                                                                 which keeps only the real part)
 * is accumulated (colour summed) and projected along a slicing direction mu with a momentum n transverse to mu:
 *     C_G(mu, n, r) = sum_{x : x_mu = x0_mu + r mod N_mu} G(x) exp(i p.(x - x0)),   p = 2 pi n / L, n_mu = 0.
 *   At n = 0 the real part equals the screening_spatial rows (checked in the driver).
 *
 * Disconnected part (noise loops, from the store after mdwfMeasureCondensateVector: column 0 = eta, 1 = S eta):
 *     L_G(mu, n, x_mu) = sum_{x with this x_mu} exp(i p.x) eta^+(x) G (S eta)(x),   G = 1 (L_raw) or g5 (L5_raw)
 *   (global coordinates, no source). The isoscalar disconnected correlator at momentum p is then built from noise
 *   pairs i != j as  sum_z L_i(n, z) L_j(-n, z + dz)  (both n and -n are written); at n != 0 there is no vacuum
 *   subtraction. At n = 0 the loops equal the existing L_raw / L5_raw slices (checked in the driver).
 *
 * Momenta: every integer vector n with n_mu = 0, n_t = 0 and 0 < |n|^2 <= n2max (spatial slicing directions only).
 * Single rank (as MDWFMeasurements.h).
 */

#pragma once

#include "MDWFPionDynamics.h"   // MDWFMonomial, mdwfMonomialApply, mdwfInner, mdwfPionAccIndex, MDWFPionProjection

#include <algorithm>
#include <array>
#include <string>
#include <utility>
#include <vector>

static const int MDWF_SM_MAX_TERMS = 8;

// Transverse momenta of slicing direction mu (0, 1, 2): n_mu = 0, ordered by |n|^2, then lexicographically.
inline std::vector<std::array<int, 3>> mdwfTransverseMomenta(int mu, int n2max, bool includeZero) {
    std::vector<std::array<int, 3>> out;
    if (mu < 0 || mu > 2) {
        throw std::runtime_error(stdLogger.fatal("MDWF screening momentum: only spatial slicing directions, got ", mu));
    }
    int nmax = 0;
    while ((nmax + 1) * (nmax + 1) <= n2max) {
        nmax++;
    }
    for (int n2 = includeZero ? 0 : 1; n2 <= n2max; n2++) {
        for (int a = -nmax; a <= nmax; a++) {
            for (int b = -nmax; b <= nmax; b++) {
                for (int c = -nmax; c <= nmax; c++) {
                    const std::array<int, 3> n = {a, b, c};
                    if (n[mu] != 0 || a * a + b * b + c * c != n2) {
                        continue;
                    }
                    out.push_back(n);
                }
            }
        }
    }
    return out;
}

template<size_t HaloDepth>
struct MDWFScreeningMomentumZero {
    LatticeContainerAccessor _acc;
    int _n;

    MDWFScreeningMomentumZero(LatticeContainerAccessor acc, int n) : _acc(acc), _n(n) {}

    __host__ __device__ void operator()(gSite site) {
        for (int ch = 0; ch < _n; ch++) {
            _acc.setElement<COMPLEX(double)>(mdwfPionAccIndex<HaloDepth>(site, ch), COMPLEX(double)(0.0, 0.0));
        }
    }
};

// Adds the complex density of up to MDWF_SM_MAX_TERMS channels (one source colour, store stacks 0..3 = the four
// spin columns) to the accumulator, channel _base + t.
template<size_t HaloDepth, size_t Ls>
struct MDWFScreeningMomentumKernel {
    Vect12ArrayAcc<double> _store;
    LatticeContainerAccessor _acc;
    MDWFMonomial _l[MDWF_SM_MAX_TERMS], _r[MDWF_SM_MAX_TERMS];
    int _n, _base;

    MDWFScreeningMomentumKernel(const MDWFSpinor<double, true, All, HaloDepth, Ls> &store, LatticeContainerAccessor acc,
                                const std::vector<std::pair<MDWFMonomial, MDWFMonomial>> &lr, int base, int n)
        : _store(store.getAccessor()), _acc(acc), _n(n), _base(base) {
        if (n > MDWF_SM_MAX_TERMS) {
            throw std::runtime_error(stdLogger.fatal("MDWF screening momentum: too many terms per kernel"));
        }
        for (int t = 0; t < n; t++) {
            _l[t] = lr[base + t].first;
            _r[t] = lr[base + t].second;
        }
    }

    __host__ __device__ void operator()(gSite site) {
        typedef GIndexer<All, HaloDepth> GInd;
        Vect12<double> v[4];
        for (int s = 0; s < 4; s++) {
            v[s] = _store.getElement(GInd::getSiteStack(site, s));
        }
        for (int t = 0; t < _n; t++) {
            // sum_{j, j'} R_{j j'} <v_j', L v_j>, R monomial: R_{j, col[j]} = val[j]
            COMPLEX(double) val(0.0, 0.0);
            for (int j = 0; j < 4; j++) {
                val += _r[t].val[j] * mdwfInner(v[_r[t].col[j]], mdwfMonomialApply(_l[t], v[j]));
            }
            const size_t idx = mdwfPionAccIndex<HaloDepth>(site, _base + t);
            _acc.setElement<COMPLEX(double)>(idx, _acc.getElement<COMPLEX(double)>(idx) + val);
        }
    }
};

// Connected momentum-projected screening correlators from the production point-source column store.
template<size_t HaloDepth, size_t Ls>
class MDWFScreeningMomentum {
    LatticeContainer<true, COMPLEX(double)> &_acc, &_red;
    std::vector<std::pair<MDWFMonomial, MDWFMonomial>> _lr;

public:
    MDWFScreeningMomentum(LatticeContainer<true, COMPLEX(double)> &acc, LatticeContainer<true, COMPLEX(double)> &red,
                          const std::vector<std::string> &channels)
        : _acc(acc), _red(red) {
        typedef GIndexer<All, HaloDepth> GInd;
        const LatticeData lat = GInd::getLatData();
        const MDWFSpinMatrix g5 = mdwfGammaMatrix("5");
        for (const std::string &channel : channels) {
            const MDWFSpinMatrix g = mdwfGammaChannel(channel);
            _lr.emplace_back(mdwfMonomial(mdwfSpinProduct(g5, g)),
                             mdwfMonomial(mdwfSpinProduct(mdwfSpinDagger(g), g5)));
        }
        _acc.adjustSize(_lr.size() * lat.vol4);
        _red.adjustSize(lat.vol4);
        CalcGSite<All, HaloDepth> calcGSite;
        iterateFunctorNoReturn<true, BLOCKSIZE>(
            MDWFScreeningMomentumZero<HaloDepth>(_acc.getAccessor(), static_cast<int>(_lr.size())), calcGSite,
            lat.vol4);
    }

    // After the four spin columns of one source colour are in store stacks 0..3.
    void addColour(const MDWFSpinor<double, true, All, HaloDepth, Ls> &store) {
        typedef GIndexer<All, HaloDepth> GInd;
        const LatticeData lat = GInd::getLatData();
        CalcGSite<All, HaloDepth> calcGSite;
        const int nch = static_cast<int>(_lr.size());
        for (int base = 0; base < nch; base += MDWF_SM_MAX_TERMS) {
            const int n = std::min(MDWF_SM_MAX_TERMS, nch - base);
            iterateFunctorNoReturn<true, BLOCKSIZE>(
                MDWFScreeningMomentumKernel<HaloDepth, Ls>(store, _acc.getAccessor(), _lr, base, n), calcGSite,
                lat.vol4);
        }
    }

    // C(mu, n, r), r = 0 .. N_mu - 1 relative to the source.
    std::vector<COMPLEX(double)> project(int ch, int mu, const std::array<int, 3> &n, const std::array<int, 4> &x0) {
        typedef GIndexer<All, HaloDepth> GInd;
        const LatticeData lat = GInd::getLatData();
        const std::array<size_t, 4> ext = mdwfLatticeExtents(lat);
        const int N = static_cast<int>(ext[mu]);
        CalcGSite<All, HaloDepth> calcGSite;
        std::vector<COMPLEX(double)> slices;
        _red.template iterateFunctor<BLOCKSIZE>(MDWFPionProjection<HaloDepth>(_acc.getAccessor(), ch, n, x0),
                                                calcGSite, MDWFWriteAtSlices<HaloDepth>(mu), lat.vol4);
        _red.reduceStackedLocal(slices, N, lat.vol4 / N);
        std::vector<COMPLEX(double)> out(N);
        for (int r = 0; r < N; r++) {
            out[r] = slices[(x0[mu] + r) % N];
        }
        return out;
    }
};

// exp(i p.x) <eta, G S eta>(x) of two store columns (G = 1 or g5), global coordinates.
template<size_t HaloDepth, size_t Ls>
struct MDWFLoopMomentumDensity {
    Vect12ArrayAcc<double> _store;
    size_t _a, _b;
    bool _gamma5;
    double _p[3];

    MDWFLoopMomentumDensity(const MDWFSpinor<double, true, All, HaloDepth, Ls> &store, size_t a, size_t b,
                            bool gamma5, const std::array<int, 3> &n)
        : _store(store.getAccessor()), _a(a), _b(b), _gamma5(gamma5) {
        typedef GIndexer<All, HaloDepth> GInd;
        const LatticeData lat = GInd::getLatData();
        const double L[3] = {static_cast<double>(lat.globLX), static_cast<double>(lat.globLY),
                             static_cast<double>(lat.globLZ)};
        for (int i = 0; i < 3; i++) {
            _p[i] = 2.0 * M_PI * n[i] / L[i];
        }
    }

    __host__ __device__ COMPLEX(double) operator()(gSite site) {
        typedef GIndexer<All, HaloDepth> GInd;
        LatticeData lat = GInd::getLatData();   // globalPos is not const
        const sitexyzt g = lat.globalPos(site.coord);
        const Vect12<double> va = _store.getElement(GInd::getSiteStack(site, _a));
        const Vect12<double> vb = _store.getElement(GInd::getSiteStack(site, _b));
        COMPLEX(double) s(0.0, 0.0);
        for (int k = 0; k < 12; k++) {
            const double sign = (_gamma5 && k >= 6) ? -1.0 : 1.0;   // as MDWFColumnInnerDensity
            s += sign * conj(va.data[k]) * vb.data[k];
        }
        const double phi = _p[0] * static_cast<int>(g.x) + _p[1] * static_cast<int>(g.y)
                           + _p[2] * static_cast<int>(g.z);
        return s * COMPLEX(double)(cos(phi), sin(phi));
    }
};

// L(mu, n, x_mu), x_mu = 0 .. N_mu - 1 (global coordinate, as mdwfColumnInnerSlices).
template<size_t HaloDepth, size_t Ls>
std::vector<COMPLEX(double)> mdwfLoopMomentumSlices(LatticeContainer<true, COMPLEX(double)> &red,
                                                    const MDWFSpinor<double, true, All, HaloDepth, Ls> &store,
                                                    size_t a, size_t b, bool gamma5, int mu,
                                                    const std::array<int, 3> &n) {
    typedef GIndexer<All, HaloDepth> GInd;
    const LatticeData lat = GInd::getLatData();
    const std::array<size_t, 4> ext = mdwfLatticeExtents(lat);
    const int N = static_cast<int>(ext[mu]);
    CalcGSite<All, HaloDepth> calcGSite;
    red.adjustSize(lat.vol4);
    std::vector<COMPLEX(double)> slices;
    red.template iterateFunctor<BLOCKSIZE>(MDWFLoopMomentumDensity<HaloDepth, Ls>(store, a, b, gamma5, n), calcGSite,
                                           MDWFWriteAtSlices<HaloDepth>(mu), lat.vol4);
    red.reduceStackedLocal(slices, N, lat.vol4 / N);
    slices.resize(N);
    return slices;
}
