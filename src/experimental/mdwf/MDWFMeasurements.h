/*
 * Connected screening correlators and stochastic condensate traces for Mobius domain-wall fermions, in the
 * conventions of the GPT production measurement scripts (DSDRMDWFruns/Measurements: measure_mres_screening_mdwf.py,
 * chiral-condensate_4d_clean_light_strange.py), so that both codes give the same numbers configuration by
 * configuration (parity suite: LatticeSimulateMDWF/parity).
 *
 *   4D propagator  S = Export M^-1 D_- Import          (MDWFResidualMass.h; GPT fermion.propagator(solver), which
 *                                                       equals bulk_propagator_to_propagator * bulk_propagator)
 *   screening      C_G(n) = sum_{x, x_mu = n} Tr[G S(x, x0) G^+ g5 S(x, x0)^+ g5]
 *                         = sum_{x, x_mu = n} sum_{k, b} conj(S_kb) (g5 G S G^+ g5)_kb,   point source at x0
 *   condensate     I_id = eta^+ eta,  I_raw = eta^+ S eta,  S2_raw = eta^+ S S eta,  P5_raw = eta^+ g5 S eta
 *
 * Gamma matrices: the chiral basis of MDWFFloatSafeWilson.h, which is Grid's/GPT's component by component, so the
 * channel names I, X, Y, Z, T, 5 and products such as "5.X" = g5 gX (left to right, as GPT's gamma_product) are the
 * same matrices, and a noise vector with the same components gives the same traces in both codes.
 *
 * Noise: Z2, every complex component +1 or -1 (imaginary part 0, as GPT's rng.zn(n = 2)), from a counter-based hash
 * of (seed, vector index, global site, component): mdwfHashZ2 below. parity/gpt/mdwf_parity_common.py builds the
 * identical vector in GPT. The seed string (default "<label>_noise") is mapped to 64 bits by FNV-1a.
 *
 * Point-source columns: for each colour c the four spin columns (s, c) are solved, exported to 4D and kept in stacks
 * 0..3 of one 5D-shaped field, then all channels are contracted for that colour (the source-side matrix G^+ g5 only
 * mixes spins of the same colour). The same solves give C_PP and C_J5q (m_res) as MDWFResidualMassMeasurement.
 *
 * Single rank.
 */

#pragma once

#include "MDWFResidualMass.h"

#include <array>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------------------------------------------
// Noise
// ---------------------------------------------------------------------------------------------------------------

__host__ __device__ inline uint64_t mdwfSplitMix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Component k (= 3 spin + colour) of noise vector vec at the global lexicographic site x + Lx (y + Ly (z + Lz t)).
__host__ __device__ inline double mdwfHashZ2(uint64_t seed, uint64_t vec, uint64_t site, uint64_t k) {
    const uint64_t h = mdwfSplitMix64(seed ^ mdwfSplitMix64((vec << 40) ^ (site * 12 + k)));
    return (h >> 63) ? -1.0 : 1.0;
}

inline uint64_t mdwfSeedFromString(const std::string &s) {
    uint64_t h = 0xcbf29ce484222325ULL;   // FNV-1a 64
    for (const unsigned char ch : s) {
        h ^= ch;
        h *= 0x100000001b3ULL;
    }
    return h;
}

// ---------------------------------------------------------------------------------------------------------------
// Spin matrices (4 x 4, row-major) and gamma channels
// ---------------------------------------------------------------------------------------------------------------

using MDWFSpinMatrix = std::array<COMPLEX(double), 16>;

inline MDWFSpinMatrix mdwfSpinIdentity() {
    MDWFSpinMatrix m;
    m.fill(COMPLEX(double)(0.0, 0.0));
    for (int i = 0; i < 4; i++) {
        m[5 * i] = COMPLEX(double)(1.0, 0.0);
    }
    return m;
}

// Grid's chiral basis, identical to mdwfGamma{X,Y,Z,T,5}Mult (MDWFFloatSafeWilson.h).
inline MDWFSpinMatrix mdwfGammaMatrix(const std::string &name) {
    const COMPLEX(double) one(1.0, 0.0), i(0.0, 1.0);
    MDWFSpinMatrix m;
    m.fill(COMPLEX(double)(0.0, 0.0));
    auto set = [&](int r, int c, COMPLEX(double) v) { m[4 * r + c] = v; };
    if (name == "I") {
        return mdwfSpinIdentity();
    } else if (name == "X") {
        set(0, 3, i), set(1, 2, i), set(2, 1, -1.0 * i), set(3, 0, -1.0 * i);
    } else if (name == "Y") {
        set(0, 3, -1.0 * one), set(1, 2, one), set(2, 1, one), set(3, 0, -1.0 * one);
    } else if (name == "Z") {
        set(0, 2, i), set(1, 3, -1.0 * i), set(2, 0, -1.0 * i), set(3, 1, i);
    } else if (name == "T") {
        set(0, 2, one), set(1, 3, one), set(2, 0, one), set(3, 1, one);
    } else if (name == "5") {
        set(0, 0, one), set(1, 1, one), set(2, 2, -1.0 * one), set(3, 3, -1.0 * one);
    } else {
        throw std::runtime_error(stdLogger.fatal("MDWF measurements: unknown gamma matrix '", name, "'"));
    }
    return m;
}

inline MDWFSpinMatrix mdwfSpinProduct(const MDWFSpinMatrix &a, const MDWFSpinMatrix &b) {
    MDWFSpinMatrix m;
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            COMPLEX(double) s(0.0, 0.0);
            for (int k = 0; k < 4; k++) {
                s += a[4 * r + k] * b[4 * k + c];
            }
            m[4 * r + c] = s;
        }
    }
    return m;
}

inline MDWFSpinMatrix mdwfSpinDagger(const MDWFSpinMatrix &a) {
    MDWFSpinMatrix m;
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            m[4 * r + c] = conj(a[4 * c + r]);
        }
    }
    return m;
}

// "5.X.T" -> g5 gX gT (GPT's gamma_product).
inline MDWFSpinMatrix mdwfGammaChannel(const std::string &channel) {
    std::istringstream in(channel);
    std::string token;
    MDWFSpinMatrix m = mdwfSpinIdentity();
    int n = 0;
    while (std::getline(in, token, '.')) {
        if (token.empty()) {
            continue;
        }
        m = mdwfSpinProduct(m, mdwfGammaMatrix(token));
        n++;
    }
    if (n == 0) {
        throw std::runtime_error(stdLogger.fatal("MDWF measurements: empty gamma channel '", channel, "'"));
    }
    return m;
}

// The channel list of GPT's old MesonCorr.py (measure_mres_screening_mdwf.py old_meson_channels): 22 channels.
inline std::vector<std::string> mdwfOldMesonChannels() {
    const std::vector<std::string> single = {"I", "5", "X", "Y", "Z", "T"};
    std::vector<std::string> channels = single, tmp;
    for (size_t mu = 1; mu < single.size(); mu++) {
        for (size_t nu = mu + 1; nu < single.size(); nu++) {
            channels.push_back(single[mu] + "." + single[nu]);
            if (single[mu] != "5") {
                tmp.push_back("5." + single[mu] + "." + single[nu]);
            }
        }
    }
    channels.insert(channels.end(), tmp.begin(), tmp.end());
    return channels;
}

// "old" | "minimal" | "u1a" | comma-separated list, as GPT's parse_channels.
inline std::vector<std::string> mdwfParseChannels(const std::string &raw) {
    if (raw == "old" || raw == "all" || raw == "mesoncorr") {
        return mdwfOldMesonChannels();
    }
    if (raw == "u1a" || raw == "ua1" || raw == "minimal_u1a") {
        return {"5", "I"};
    }
    if (raw == "minimal" || raw == "single") {
        return {"I", "5", "X", "Y", "Z", "T"};
    }
    std::vector<std::string> channels;
    std::istringstream in(raw);
    std::string token;
    while (std::getline(in, token, ',')) {
        if (!token.empty()) {
            mdwfGammaChannel(token);   // validates
            channels.push_back(token);
        }
    }
    if (channels.empty()) {
        throw std::runtime_error(stdLogger.fatal("MDWF measurements: empty channel list '", raw, "'"));
    }
    return channels;
}

// ---------------------------------------------------------------------------------------------------------------
// Kernels on the column store (a 5D-shaped field whose stacks hold 4D spin-colour vectors)
// ---------------------------------------------------------------------------------------------------------------

// Rewrites stack `column` of the store with the 4D export P_+ psi(Ls-1) + P_- psi(0) of a 5D solution; the other
// stacks are copied unchanged.
template<size_t HaloDepth, size_t Ls>
struct MDWFExportColumn {
    Vect12ArrayAcc<double> _store, _psi;
    size_t _column;

    MDWFExportColumn(const MDWFSpinor<double, true, All, HaloDepth, Ls> &store,
                     const MDWFSpinor<double, true, All, HaloDepth, Ls> &psi, size_t column)
        : _store(store.getAccessor()), _psi(psi.getAccessor()), _column(column) {}

    __host__ __device__ Vect12<double> operator()(gSiteStack site) {
        typedef GIndexer<All, HaloDepth> GInd;
        if (site.stack != _column) {
            return _store.getElement(site);
        }
        return mdwfProjectPlus(_psi.getElement(GInd::getSiteStack(site, Ls - 1)))
               + mdwfProjectMinus(_psi.getElement(GInd::getSiteStack(site, 0)));
    }
};

// Rewrites stack `column` of the store with Z2 hash noise vector `vec`; the other stacks are copied unchanged.
template<size_t HaloDepth, size_t Ls>
struct MDWFNoiseColumn {
    Vect12ArrayAcc<double> _store;
    size_t _column;
    uint64_t _seed, _vec;

    MDWFNoiseColumn(const MDWFSpinor<double, true, All, HaloDepth, Ls> &store, size_t column, uint64_t seed,
                    uint64_t vec)
        : _store(store.getAccessor()), _column(column), _seed(seed), _vec(vec) {}

    __host__ __device__ Vect12<double> operator()(gSiteStack site) {
        typedef GIndexer<All, HaloDepth> GInd;
        if (site.stack != _column) {
            return _store.getElement(site);
        }
        LatticeData lat = GInd::getLatData();   // globalPos is not const
        const sitexyzt g = lat.globalPos(site.coord);
        const uint64_t lex = static_cast<uint64_t>(g.x)
                             + lat.globLX * (static_cast<uint64_t>(g.y)
                                             + lat.globLY * (static_cast<uint64_t>(g.z)
                                                             + lat.globLZ * static_cast<uint64_t>(g.t)));
        Vect12<double> out(0.0);
        for (int k = 0; k < 12; k++) {
            out.data[k] = COMPLEX(double)(mdwfHashZ2(_seed, _vec, lex, k), 0.0);
        }
        return out;
    }
};

// Unprojected 5D source from stack `column` of the store: slice 0 holds the upper (P_+) components, slice Ls-1 the
// lower (P_-) ones, as MDWFPhysicalPointSource; MDWFResidualMassMeasurement::solveSource applies D_-.
template<size_t HaloDepth, size_t Ls>
struct MDWFColumnSource {
    Vect12ArrayAcc<double> _store;
    size_t _column;

    MDWFColumnSource(const MDWFSpinor<double, true, All, HaloDepth, Ls> &store, size_t column)
        : _store(store.getAccessor()), _column(column) {}

    __host__ __device__ Vect12<double> operator()(gSiteStack site) {
        typedef GIndexer<All, HaloDepth> GInd;
        const Vect12<double> v = _store.getElement(GInd::getSiteStack(site, _column));
        if (site.stack == 0) {
            return mdwfProjectPlus(v);
        }
        if (site.stack == Ls - 1) {
            return mdwfProjectMinus(v);
        }
        return Vect12<double>(0.0);
    }
};

// Real or imaginary part of the local inner product <a, b> (or <a, g5 b>) of two store columns.
template<size_t HaloDepth, size_t Ls>
struct MDWFColumnInnerDensity {
    Vect12ArrayAcc<double> _store;
    size_t _a, _b;
    bool _gamma5, _imag;

    MDWFColumnInnerDensity(const MDWFSpinor<double, true, All, HaloDepth, Ls> &store, size_t a, size_t b,
                           bool gamma5, bool imag)
        : _store(store.getAccessor()), _a(a), _b(b), _gamma5(gamma5), _imag(imag) {}

    __host__ __device__ double operator()(gSite site) {
        typedef GIndexer<All, HaloDepth> GInd;
        const Vect12<double> va = _store.getElement(GInd::getSiteStack(site, _a));
        const Vect12<double> vb = _store.getElement(GInd::getSiteStack(site, _b));
        COMPLEX(double) s(0.0, 0.0);
        for (int k = 0; k < 12; k++) {
            const double sign = (_gamma5 && k >= 6) ? -1.0 : 1.0;
            s += sign * conj(va.data[k]) * vb.data[k];
        }
        return _imag ? imag(s) : real(s);
    }
};

// sum_{s, k} conj(v_s[k]) (L w_s)[k], w_s = sum_s' R[s'][s] v_s', v_s = stack s (spin column s, one colour):
// the colour-c part of sum_{k, b} conj(S_kb) (L S R)_kb with L = g5 G, R = G^+ g5.
template<size_t HaloDepth, size_t Ls>
struct MDWFMesonChannelDensity {
    Vect12ArrayAcc<double> _store;
    COMPLEX(double) _l[16], _r[16];

    MDWFMesonChannelDensity(const MDWFSpinor<double, true, All, HaloDepth, Ls> &store, const MDWFSpinMatrix &l,
                            const MDWFSpinMatrix &r)
        : _store(store.getAccessor()) {
        for (int i = 0; i < 16; i++) {
            _l[i] = l[i];
            _r[i] = r[i];
        }
    }

    __host__ __device__ double operator()(gSite site) {
        typedef GIndexer<All, HaloDepth> GInd;
        Vect12<double> v[4];
        for (int s = 0; s < 4; s++) {
            v[s] = _store.getElement(GInd::getSiteStack(site, s));
        }
        double sum = 0.0;
        for (int s = 0; s < 4; s++) {
            COMPLEX(double) w[12];
            for (int k = 0; k < 12; k++) {
                w[k] = COMPLEX(double)(0.0, 0.0);
                for (int sp = 0; sp < 4; sp++) {
                    w[k] += _r[4 * sp + s] * v[sp].data[k];
                }
            }
            for (int i = 0; i < 4; i++) {
                for (int c = 0; c < 3; c++) {
                    COMPLEX(double) u(0.0, 0.0);
                    for (int j = 0; j < 4; j++) {
                        u += _l[4 * i + j] * w[3 * j + c];
                    }
                    sum += real(conj(v[s].data[3 * i + c]) * u);
                }
            }
        }
        return sum;
    }
};

// ---------------------------------------------------------------------------------------------------------------
// Measurements
// ---------------------------------------------------------------------------------------------------------------

// Per-slice sums of a 4D density along direction mu (global coordinate, not shifted by the source).
template<size_t HaloDepth, class Density>
std::vector<double> mdwfSliceSums(LatticeContainer<true, double> &reduction, Density density, int mu) {
    typedef GIndexer<All, HaloDepth> GInd;
    const LatticeData lat = GInd::getLatData();
    const std::array<size_t, 4> extents = mdwfLatticeExtents(lat);
    CalcGSite<All, HaloDepth> calcGSite;
    std::vector<double> slices;
    reduction.template iterateFunctor<BLOCKSIZE>(density, calcGSite, MDWFWriteAtSlices<HaloDepth>(mu), lat.vol4);
    reduction.reduceStackedLocal(slices, extents[mu], lat.vol4 / extents[mu]);
    slices.resize(extents[mu]);
    return slices;
}

struct MDWFPointMeasurementResult {
    std::array<std::vector<double>, 4> pp, j5q;                 // C_PP, C_J5q along mu
    std::vector<std::array<std::vector<double>, 4>> channels;   // C_G along mu, per channel
    int iterations;
    double maxResidue;
    double seconds;
};

/*
 * Point source at x0: 12 solves (colour-major: for each colour the four spin columns), C_PP and C_J5q (m_res) and
 * the connected screening correlators of the given channels along all four directions.
 */
template<size_t HaloDepth, size_t Ls, class Solver>
MDWFPointMeasurementResult mdwfMeasurePointSource(MDWFResidualMassMeasurement<HaloDepth, Ls, Solver> &meas,
                                                  MDWFSpinor<double, true, All, HaloDepth, Ls> &store,
                                                  const std::array<int, 4> &x0,
                                                  const std::vector<std::string> &channels, int maxIter,
                                                  double precision) {
    static_assert(Ls >= 4, "the column store needs four stacks");
    typedef GIndexer<All, HaloDepth> GInd;
    const LatticeData lat = GInd::getLatData();
    MDWFPointMeasurementResult result;
    mdwfResizeDirectionalCorrelators(result.pp, lat);
    mdwfResizeDirectionalCorrelators(result.j5q, lat);
    result.channels.resize(channels.size());
    for (auto &c : result.channels) {
        mdwfResizeDirectionalCorrelators(c, lat);
    }
    result.iterations = 0;
    result.maxResidue = 0.0;
    result.seconds = 0.0;

    std::vector<std::pair<MDWFSpinMatrix, MDWFSpinMatrix>> lr;
    const MDWFSpinMatrix g5 = mdwfGammaMatrix("5");
    for (const std::string &channel : channels) {
        const MDWFSpinMatrix g = mdwfGammaChannel(channel);
        lr.emplace_back(mdwfSpinProduct(g5, g), mdwfSpinProduct(mdwfSpinDagger(g), g5));
    }

    LatticeContainer<true, double> reduction(store.getComm());
    reduction.adjustSize(lat.vol4);
    for (int colour = 0; colour < 3; colour++) {
        for (int spin = 0; spin < 4; spin++) {
            const int a = 3 * spin + colour;
            const MDWFResidualMassComponent c = meas.measureComponent(a, x0[0], x0[1], x0[2], x0[3], maxIter,
                                                                      precision);
            result.iterations += c.iterations;
            result.maxResidue = std::max(result.maxResidue, c.residue);
            result.seconds += c.seconds;
            for (int mu = 0; mu < 4; mu++) {
                for (size_t n = 0; n < c.pp[mu].size(); n++) {
                    result.pp[mu][n] += c.pp[mu][n];
                    result.j5q[mu][n] += c.j5q[mu][n];
                }
            }
            store.template iterateOverBulk<BLOCKSIZE>(MDWFExportColumn<HaloDepth, Ls>(store, meas.solution5D(), spin));
        }
        for (size_t ch = 0; ch < channels.size(); ch++) {
            for (int mu = 0; mu < 4; mu++) {
                const std::vector<double> s = mdwfSliceSums<HaloDepth>(
                    reduction, MDWFMesonChannelDensity<HaloDepth, Ls>(store, lr[ch].first, lr[ch].second), mu);
                for (size_t n = 0; n < s.size(); n++) {
                    result.channels[ch][mu][n] += s[n];
                }
            }
        }
    }
    return result;
}

struct MDWFCondensateVector {
    COMPLEX(double) id, raw, s2, p5;   // eta^+ eta, eta^+ S eta, eta^+ S S eta, eta^+ g5 S eta
    int iterations;
    double seconds;
};

template<size_t HaloDepth, size_t Ls>
COMPLEX(double) mdwfColumnInner(LatticeContainer<true, double> &reduction,
                                const MDWFSpinor<double, true, All, HaloDepth, Ls> &store, size_t a, size_t b,
                                bool gamma5) {
    double part[2];
    for (int im = 0; im < 2; im++) {
        const std::vector<double> s = mdwfSliceSums<HaloDepth>(
            reduction, MDWFColumnInnerDensity<HaloDepth, Ls>(store, a, b, gamma5, im == 1), 3);
        part[im] = 0.0;
        for (const double v : s) {
            part[im] += v;
        }
    }
    return COMPLEX(double)(part[0], part[1]);
}

// Slice-resolved <a, b> (or <a, g5 b>) of two store columns along mu: the disconnected loops L_raw / L5_raw of GPT's
// measure_condensate_slices_mdwf.py (sum over the sites with x_mu fixed), accumulated in double precision.
template<size_t HaloDepth, size_t Ls>
std::vector<COMPLEX(double)> mdwfColumnInnerSlices(LatticeContainer<true, double> &reduction,
                                                   const MDWFSpinor<double, true, All, HaloDepth, Ls> &store, size_t a,
                                                   size_t b, bool gamma5, int mu) {
    const std::vector<double> re = mdwfSliceSums<HaloDepth>(
        reduction, MDWFColumnInnerDensity<HaloDepth, Ls>(store, a, b, gamma5, false), mu);
    const std::vector<double> im = mdwfSliceSums<HaloDepth>(
        reduction, MDWFColumnInnerDensity<HaloDepth, Ls>(store, a, b, gamma5, true), mu);
    std::vector<COMPLEX(double)> s(re.size());
    for (size_t n = 0; n < re.size(); n++) {
        s[n] = COMPLEX(double)(re[n], im[n]);
    }
    return s;
}

/*
 * One noise vector: eta (Z2 hash, stack 0), S eta (stack 1) and, if connected, S S eta (stack 2). The noise is
 * written only when writeNoise (the first flavour of a vector), so all flavours see the same eta.
 */
template<size_t HaloDepth, size_t Ls, class Solver>
MDWFCondensateVector mdwfMeasureCondensateVector(MDWFResidualMassMeasurement<HaloDepth, Ls, Solver> &meas,
                                                 MDWFSpinor<double, true, All, HaloDepth, Ls> &store, uint64_t seed,
                                                 uint64_t vec, bool writeNoise, bool connected, int maxIter,
                                                 double precision) {
    static_assert(Ls >= 3, "the column store needs three stacks");
    typedef GIndexer<All, HaloDepth> GInd;
    const LatticeData lat = GInd::getLatData();
    LatticeContainer<true, double> reduction(store.getComm());
    reduction.adjustSize(lat.vol4);
    MDWFCondensateVector r{};
    if (writeNoise) {
        store.template iterateOverBulk<BLOCKSIZE>(MDWFNoiseColumn<HaloDepth, Ls>(store, 0, seed, vec));
    }
    r.id = mdwfColumnInner<HaloDepth, Ls>(reduction, store, 0, 0, false);
    for (size_t column = 1; column <= (connected ? 2u : 1u); column++) {
        meas.source5D().template iterateOverBulk<BLOCKSIZE>(MDWFColumnSource<HaloDepth, Ls>(store, column - 1));
        r.seconds += meas.solveSource(maxIter, precision);
        r.iterations += meas.solver().lastIterations();
        store.template iterateOverBulk<BLOCKSIZE>(MDWFExportColumn<HaloDepth, Ls>(store, meas.solution5D(), column));
    }
    r.raw = mdwfColumnInner<HaloDepth, Ls>(reduction, store, 0, 1, false);
    r.p5 = mdwfColumnInner<HaloDepth, Ls>(reduction, store, 0, 1, true);
    if (connected) {
        r.s2 = mdwfColumnInner<HaloDepth, Ls>(reduction, store, 0, 2, false);
    }
    return r;   // store column 0 = eta, column 1 = S eta (for mdwfColumnInnerSlices)
}
