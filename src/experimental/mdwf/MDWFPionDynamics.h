/*
 * Temporal pion / charge correlators with local, midpoint (J5q) and Mobius CONSERVED currents, a port of the GPT
 * measurement DSDRMDWFruns/Measurements/measure_charge_spectral_mdwf.py (= measure_pion_dynamics_mdwf.py plus the
 * rho / a1 channels), with the same output file (one long TSV per configuration):
 *
 *   label conf mass Ls M5 b5 c5 src_x src_y src_z src_t direction n2 n_vec channel coord re im
 *
 * One point source x0, 12 solves (colour-major, the four spin columns of one colour kept as full 5D solutions).
 *
 * Local channels (sink Gsnk at x, source Gsrc at x0), S = 4D propagator (Export M^-1 D_- Import, as GPT's
 * bulk_propagator_to_propagator * bulk_propagator):
 *     G(x) = tr[Gsnk S(x,x0) Gsrc g5 S(x,x0)^+ g5] = sum_c sum_{j,j'} (Gsrc g5)_{j j'} <S_j', g5 Gsnk S_j>(x)
 *     PP g5/g5, A0A0 A0/A0^+, A0P A0/g5, PA0 g5/A0^+, V0V0 gT/gT^+, SS 1/1, ViVi and AiAi (average over i = x, y, z)
 *   with A0 = gT g5, Ai = gi g5.
 * Midpoint: J5qP = sum_a |P_+ psi_a(Ls/2-1) + P_- psi_a(Ls/2)|^2 (GPT tr[J5q J5q^+]).
 *   J5qA0 (2026-11-03, not in GPT): J5q midpoint sink, local A0^+ source, with Sm = the midpoint field
 *     J5qA0 = tr[g5 Sm A0^+ g5 Sm^+ g5] = sum_{j,j'} (A0^+ g5)_{j j'} <Sm_j', Sm_j>   (J5qP is the same with g5 for A0^+)
 *   It completes the axial Ward identity for the axial-charge correlator (exact on the lattice, tau != 0):
 *     A0cA0(tau) - A0cA0(tau-1) = 2 [m PA0(tau) + J5qA0(tau)]
 *   so the tau dependence of G_A (pion curvature) comes from the precise PA0 plus the residual-mass term.
 *
 * Conserved currents: GPT mobius.conserved_current (a port of Grid's CayleyFermion5D::ContractConservedCurrent with
 * source terms), L = R = the 5D propagator. Written out, GPT's s loop is
 *     Qx(k)  = psi(k) for 0 <= k < Ls,   Qx(-1) = -m psi(Ls-1) + eta,   Qx(Ls) = -m psi(0) + eta   (eta = point source)
 *     Y(a)   = b Qx(a) + c [P_+ Qx(a-1) + P_- Qx(a+1)]
 *     B_jj'  = sum_s w_s (-1/(2(b+c))) { <Y_j(Ls-1-s)(x), g5 (1 - g_mu) U_mu(x) Y_j'(s)(x+mu)>
 *                                      - <U_mu(x) Y_j(Ls-1-s)(x+mu), g5 (1 + g_mu) Y_j'(s)(x)> }
 *     C(x)   = g5 B (source spin space),   GPT tr[C X] = tr[B X g5],
 *   w_s = sign (vector +1, axial -1) for s < Ls/2, +1 otherwise; colour c of the source summed. U carries the fermion
 *   boundary phases (fermionGauge), as GPT's covariant_shift. The current lives on the link x -> x + mu.
 *     A0cP  axial mu = t, X = g5          A0cA0 axial mu = t, X = A0^+       V0cV0 vector mu = t, X = gT^+
 *     VicVi vector mu = i, X = gi^+ (1/3 sum)                                AicAi axial mu = i, X = Ai^+ (1/3 sum)
 *
 * Projection: temporal correlators sum_x f(x) exp(i p.(x - x0)) per time slice, p = 2 pi n / N (spatial), averaged
 * over each orbit |n|^2 = 0..n2max, written at tau = (t - t0) mod Nt; spatial ones (PP, SS, A0A0) along z at p = 0,
 * at r = (z - z0) mod Nz.
 *
 * Transverse / longitudinal split (2026-11-05, not in GPT): the full spatial tensor G_ij (sink i, source j^+) of
 * V_i V_j, A_i A_j (local-local) and V^c_i V_j, A^c_i A_j (conserved sink, local source) is accumulated from the same
 * solves. Per momentum n, with n^ = n/|n|:
 *     G_L = n^_i n^_j G_ij,   G_T = (sum_i G_ii - G_L) / 2,   averaged over the orbit;  n = 0: G_T = G_L = sum_i G_ii / 3
 * written as ViVi_T, ViVi_L, AiAi_T, AiAi_L, VicVi_T, VicVi_L, AicAi_T, AicAi_L (temporal_t, every |n|^2).
 * The conserved sink V^c_i sits on the link x -> x + i, so its projection carries exp(i p.(x + i/2 - x0)) (half-link
 * phase). The old averaged channels ViVi ... AicAi are unchanged (no half-link phase, as GPT), so
 *     (G_L + 2 G_T)/3 = ViVi, AiAi exactly;  = VicVi, AicAi at n = 0 only.
 * For |n|^2 <= 4 the lattice momentum direction (sin(p_i/2)) equals n^, so the split is the lattice T/L split.
 * At p != 0 the A_i L channel contains the pion.
 *
 * Screening direction (2026-11-07, not in GPT): spatial_z rows (p = 0, r = (z - z0) mod Nz, sum over x, y, t) of
 *     PP, SS, A0A0 (as before) and J5qP, AzP, AzcP, V0V0, V0cV0, A0cA0, and the diagonals of the four spatial tensors
 *     VxVx VyVy VzVz, AxAx AyAy AzAz (local-local), VxcVx VycVy VzcVz, AxcAx AycAy AzcAz (conserved sink, local source)
 *   AzP  = local Az = gZ g5 sink, g5 source;  AzcP = conserved axial current in direction z (link z -> z+1), g5 source.
 *   Both use the same solves (no extra cost beyond contractions). Exact lattice identities along z (r != 0):
 *     AzcP(r) - AzcP(r-1) = 2 [m PP(r) + J5qP(r)]          (spatial PCAC, logged as "z-PCAC")
 *     VzcVz(r) = VzcVz(r-1)                                (vector current conservation along z, logged)
 *   so AzcP / AzP gives Z_A and VicVi / ViVi (i = x, y, t transverse to z) gives Z_V from the screening direction.
 *   The conserved Vz/Az sink sits on the link z -> z+1, written at r = z - z0 of the link start.
 *
 * Single rank (as MDWFMeasurements.h).
 */

#pragma once

#include "MDWFMeasurements.h"

#include <cmath>
#include <map>
#include <string>
#include <vector>

// Spin matrix with one non-zero entry per row: (M v)_i = val[i] v_{col[i]}. All gamma products are of this form.
struct MDWFMonomial {
    int col[4];
    COMPLEX(double) val[4];
};

inline MDWFMonomial mdwfMonomial(const MDWFSpinMatrix &m) {
    MDWFMonomial g;
    for (int i = 0; i < 4; i++) {
        int n = 0;
        g.col[i] = 0;
        g.val[i] = COMPLEX(double)(0.0, 0.0);
        for (int j = 0; j < 4; j++) {
            if (abs(m[4 * i + j]) > 1e-14) {
                g.col[i] = j;
                g.val[i] = m[4 * i + j];
                n++;
            }
        }
        if (n != 1) {
            throw std::runtime_error(stdLogger.fatal("MDWF pion dynamics: spin matrix is not monomial"));
        }
    }
    return g;
}

__host__ __device__ inline Vect12<double> mdwfMonomialApply(const MDWFMonomial &g, const Vect12<double> &v) {
    Vect12<double> out(0.0);
    for (int i = 0; i < 4; i++) {
        for (int a = 0; a < 3; a++) {
            out.data[3 * i + a] = g.val[i] * v.data[3 * g.col[i] + a];
        }
    }
    return out;
}

__host__ __device__ inline Vect12<double> mdwfLinkApply(SU3<double> u, const Vect12<double> &v) {
    Vect12<double> out(0.0);
    for (int i = 0; i < 4; i++) {
        for (int a = 0; a < 3; a++) {
            COMPLEX(double) s(0.0, 0.0);
            for (int b = 0; b < 3; b++) {
                s += u(a, b) * v.data[3 * i + b];
            }
            out.data[3 * i + a] = s;
        }
    }
    return out;
}

__host__ __device__ inline COMPLEX(double) mdwfInner(const Vect12<double> &a, const Vect12<double> &b) {
    COMPLEX(double) s(0.0, 0.0);
    for (int k = 0; k < 12; k++) {
        s += conj(a.data[k]) * b.data[k];
    }
    return s;
}

// tr[B R] for a 4 x 4 B (row-major) and a monomial R: sum_j R_{j, col[j]} B_{col[j], j}.
__host__ __device__ inline COMPLEX(double) mdwfTraceMonomial(const COMPLEX(double) *B, const MDWFMonomial &r) {
    COMPLEX(double) s(0.0, 0.0);
    for (int j = 0; j < 4; j++) {
        s += r.val[j] * B[4 * r.col[j] + j];
    }
    return s;
}

// Channels, in GPT's order of temporal_channels; SS (spatial only) last.
enum MDWFPionChannel {
    PD_PP = 0, PD_A0A0, PD_A0P, PD_PA0, PD_V0V0, PD_J5qP, PD_A0cP, PD_A0cA0, PD_V0cV0, PD_ViVi, PD_AiAi, PD_VicVi,
    PD_AicAi, PD_J5qA0, PD_AzP, PD_AzcP, PD_SS, PD_NCHANNELS
};

// Spatial tensor accumulators after the output channels: type 0 V_i V_j, 1 A_i A_j, 2 V^c_i V_j, 3 A^c_i A_j.
static const int PD_TENSOR_BASE = PD_NCHANNELS;
static const int PD_NACC = PD_NCHANNELS + 36;

__host__ __device__ inline int mdwfPionTensorIndex(int type, int i, int j) {
    return PD_TENSOR_BASE + 9 * type + 3 * i + j;
}

inline const char *mdwfPionChannelName(int ch) {
    static const char *names[PD_NCHANNELS] = {"PP", "A0A0", "A0P", "PA0", "V0V0", "J5qP", "A0cP", "A0cA0", "V0cV0",
                                              "ViVi", "AiAi", "VicVi", "AicAi", "J5qA0", "AzP", "AzcP",
                                              "SS"};
    return names[ch];
}

// Index of a site in the accumulator: channel-major, then lexicographic x + Lx (y + Ly (z + Lz t)).
template<size_t HaloDepth>
__host__ __device__ inline size_t mdwfPionAccIndex(const gSite &site, int ch) {
    typedef GIndexer<All, HaloDepth> GInd;
    const LatticeData lat = GInd::getLatData();
    const sitexyzt c = site.coord;
    return static_cast<size_t>(ch) * lat.vol4 + c.x + lat.lx * (c.y + lat.ly * (c.z + lat.lz * c.t));
}

template<size_t HaloDepth>
struct MDWFPionZero {
    LatticeContainerAccessor _acc;

    explicit MDWFPionZero(LatticeContainerAccessor acc) : _acc(acc) {}

    __host__ __device__ void operator()(gSite site) {
        for (int ch = 0; ch < PD_NACC; ch++) {
            _acc.setElement<COMPLEX(double)>(mdwfPionAccIndex<HaloDepth>(site, ch), COMPLEX(double)(0.0, 0.0));
        }
    }
};

// The four spin columns of one source colour (full 5D solutions).
template<size_t HaloDepth, size_t Ls>
struct MDWFPionColumns {
    Vect12ArrayAcc<double> _q0, _q1, _q2, _q3;

    MDWFPionColumns(const MDWFSpinor<double, true, All, HaloDepth, Ls> &q0,
                    const MDWFSpinor<double, true, All, HaloDepth, Ls> &q1,
                    const MDWFSpinor<double, true, All, HaloDepth, Ls> &q2,
                    const MDWFSpinor<double, true, All, HaloDepth, Ls> &q3)
        : _q0(q0.getAccessor()), _q1(q1.getAccessor()), _q2(q2.getAccessor()), _q3(q3.getAccessor()) {}

    __host__ __device__ Vect12<double> get(int j, const gSite &site, size_t s) {
        typedef GIndexer<All, HaloDepth> GInd;
        const gSiteStack ss = GInd::getSiteStack(site, s);
        switch (j) {
            case 0:
                return _q0.getElement(ss);
            case 1:
                return _q1.getElement(ss);
            case 2:
                return _q2.getElement(ss);
            default:
                return _q3.getElement(ss);
        }
    }
};

struct MDWFPionLocalTerm {
    MDWFMonomial l, r;   // L = g5 Gsnk, R = Gsrc g5
    double weight;
    int channel;
};

static const int MDWF_PION_MAX_LOCAL = 12;

// Local channels and J5qP of source colour c, added to the accumulator.
template<size_t HaloDepth, size_t Ls>
struct MDWFPionLocalKernel {
    MDWFPionColumns<HaloDepth, Ls> _q;
    LatticeContainerAccessor _acc;
    MDWFPionLocalTerm _terms[MDWF_PION_MAX_LOCAL];
    int _nterms;
    MDWFMonomial _rJ5qA0;   // A0^+ g5
    bool _j5q;              // also accumulate J5qP and J5qA0 (false for the extra tensor passes)

    MDWFPionLocalKernel(const MDWFPionColumns<HaloDepth, Ls> &q, LatticeContainerAccessor acc,
                        const std::vector<MDWFPionLocalTerm> &terms, const MDWFMonomial &rJ5qA0, bool j5q = true)
        : _q(q), _acc(acc), _nterms(static_cast<int>(terms.size())), _rJ5qA0(rJ5qA0), _j5q(j5q) {
        if (terms.size() > static_cast<size_t>(MDWF_PION_MAX_LOCAL)) {
            throw std::runtime_error(stdLogger.fatal("MDWF pion dynamics: too many local terms"));
        }
        for (int t = 0; t < _nterms; t++) {
            _terms[t] = terms[t];
        }
    }

    __host__ __device__ void operator()(gSite site) {
        Vect12<double> S[4], P[4];
        double j5q = 0.0;
        for (int j = 0; j < 4; j++) {
            S[j] = mdwfProjectMinus(_q.get(j, site, 0)) + mdwfProjectPlus(_q.get(j, site, Ls - 1));
            P[j] = mdwfProjectPlus(_q.get(j, site, Ls / 2 - 1)) + mdwfProjectMinus(_q.get(j, site, Ls / 2));
            j5q += real(mdwfInner(P[j], P[j]));
        }
        if (_j5q) {   // J5qA0 = sum_j R_{j, col[j]} <P_col[j], P_j>,  R = A0^+ g5
            COMPLEX(double) v(0.0, 0.0);
            for (int j = 0; j < 4; j++) {
                v += _rJ5qA0.val[j] * mdwfInner(P[_rJ5qA0.col[j]], P[j]);
            }
            const size_t idx = mdwfPionAccIndex<HaloDepth>(site, PD_J5qA0);
            _acc.setElement<COMPLEX(double)>(idx, _acc.getElement<COMPLEX(double)>(idx) + v);
        }
        for (int t = 0; t < _nterms; t++) {
            COMPLEX(double) M[16];
            for (int j = 0; j < 4; j++) {
                const Vect12<double> ls = mdwfMonomialApply(_terms[t].l, S[j]);
                for (int jp = 0; jp < 4; jp++) {
                    M[4 * jp + j] = mdwfInner(S[jp], ls);   // M_{j' j} = <S_j', L S_j>
                }
            }
            // tr[R M] = sum_{j, j'} R_{j j'} M_{j' j}; R monomial: R_{j, col[j]}
            COMPLEX(double) v(0.0, 0.0);
            for (int j = 0; j < 4; j++) {
                v += _terms[t].r.val[j] * M[4 * _terms[t].r.col[j] + j];
            }
            const size_t idx = mdwfPionAccIndex<HaloDepth>(site, _terms[t].channel);
            _acc.setElement<COMPLEX(double)>(idx, _acc.getElement<COMPLEX(double)>(idx) + _terms[t].weight * v);
        }
        if (_j5q) {
            const size_t idx = mdwfPionAccIndex<HaloDepth>(site, PD_J5qP);
            _acc.setElement<COMPLEX(double)>(idx, _acc.getElement<COMPLEX(double)>(idx) + COMPLEX(double)(j5q, 0.0));
        }
    }
};

struct MDWFPionConservedTerm {
    MDWFMonomial r;   // X g5 (GPT tr[C X] = tr[B X g5])
    double weight;
    int channel;
    int axial;
};

static const int MDWF_PION_MAX_CONSERVED = 9;   // mu = z: VicVi, AicAi + 3 V^c_i V_j + 3 A^c_i A_j + AzcP

// Conserved vector / axial currents in direction mu for source colour c, added to the accumulator.
template<size_t HaloDepth, size_t Ls>
struct MDWFPionConservedKernel {
    MDWFPionColumns<HaloDepth, Ls> _q;
    SU3Accessor<double> _gauge;
    LatticeContainerAccessor _acc;
    MDWFMonomial _gmu, _g5;
    MDWFPionConservedTerm _terms[MDWF_PION_MAX_CONSERVED];
    int _nterms, _mu, _colour;
    int _x0[4];
    double _mass, _b, _c;

    MDWFPionConservedKernel(const MDWFPionColumns<HaloDepth, Ls> &q, Gaugefield<double, true, HaloDepth, R18> &gauge,
                            LatticeContainerAccessor acc, int mu, int colour, const std::array<int, 4> &x0,
                            double mass, double b, double c, const std::vector<MDWFPionConservedTerm> &terms)
        : _q(q), _gauge(gauge.getAccessor()), _acc(acc), _nterms(static_cast<int>(terms.size())), _mu(mu),
          _colour(colour), _mass(mass), _b(b), _c(c) {
        if (terms.size() > static_cast<size_t>(MDWF_PION_MAX_CONSERVED)) {
            throw std::runtime_error(stdLogger.fatal("MDWF pion dynamics: too many conserved terms"));
        }
        static const char *gname[4] = {"X", "Y", "Z", "T"};
        _gmu = mdwfMonomial(mdwfGammaMatrix(gname[mu]));
        _g5 = mdwfMonomial(mdwfGammaMatrix("5"));
        for (int t = 0; t < _nterms; t++) {
            _terms[t] = terms[t];
        }
        for (int i = 0; i < 4; i++) {
            _x0[i] = x0[i];
        }
    }

    __host__ __device__ bool atSource(const gSite &site) {
        typedef GIndexer<All, HaloDepth> GInd;
        LatticeData lat = GInd::getLatData();
        const sitexyzt g = lat.globalPos(site.coord);
        return static_cast<int>(g.x) == _x0[0] && static_cast<int>(g.y) == _x0[1] && static_cast<int>(g.z) == _x0[2]
               && static_cast<int>(g.t) == _x0[3];
    }

    // Qx(k) of column j: the 5D solution for 0 <= k < Ls, the mass/source-extended slices at k = -1 and k = Ls.
    __host__ __device__ Vect12<double> qx(int j, const gSite &site, int k, bool src) {
        if (k >= 0 && k < static_cast<int>(Ls)) {
            return _q.get(j, site, static_cast<size_t>(k));
        }
        Vect12<double> v = _q.get(j, site, k < 0 ? Ls - 1 : 0);
        for (int i = 0; i < 12; i++) {
            v.data[i] = -_mass * v.data[i];
        }
        if (src) {
            v.data[3 * j + _colour] += COMPLEX(double)(1.0, 0.0);
        }
        return v;
    }

    // Y(a) = b Qx(a) + c [P_+ Qx(a-1) + P_- Qx(a+1)]
    __host__ __device__ Vect12<double> y(int j, const gSite &site, int a, bool src) {
        const Vect12<double> q = qx(j, site, a, src);
        const Vect12<double> qm = qx(j, site, a - 1, src);
        const Vect12<double> qp = qx(j, site, a + 1, src);
        Vect12<double> out(0.0);
        for (int i = 0; i < 6; i++) {
            out.data[i] = _b * q.data[i] + _c * qm.data[i];
        }
        for (int i = 6; i < 12; i++) {
            out.data[i] = _b * q.data[i] + _c * qp.data[i];
        }
        return out;
    }

    __host__ __device__ void operator()(gSite site) {
        typedef GIndexer<All, HaloDepth> GInd;
        const gSite nb = GInd::site_up(site, _mu);
        const SU3<double> u = _gauge.getLink(GInd::getSiteMu(site, _mu));
        const bool srcX = atSource(site), srcN = atSource(nb);
        const double bpc = -0.5 / (_b + _c);
        COMPLEX(double) lo[16], hi[16];
        for (int i = 0; i < 16; i++) {
            lo[i] = COMPLEX(double)(0.0, 0.0);
            hi[i] = COMPLEX(double)(0.0, 0.0);
        }
        for (int s = 0; s < static_cast<int>(Ls); s++) {
            const int sr = static_cast<int>(Ls) - 1 - s;
            Vect12<double> yxr[4], unr[4], a1[4], a2[4];
            for (int j = 0; j < 4; j++) {
                yxr[j] = y(j, site, sr, srcX);
                unr[j] = mdwfLinkApply(u, y(j, nb, sr, srcN));
                const Vect12<double> uns = mdwfLinkApply(u, y(j, nb, s, srcN));
                const Vect12<double> yxs = y(j, site, s, srcX);
                const Vect12<double> guns = mdwfMonomialApply(_gmu, uns);
                const Vect12<double> gyxs = mdwfMonomialApply(_gmu, yxs);
                Vect12<double> m1(0.0), p1(0.0);
                for (int i = 0; i < 12; i++) {
                    m1.data[i] = uns.data[i] - guns.data[i];
                    p1.data[i] = yxs.data[i] + gyxs.data[i];
                }
                a1[j] = mdwfMonomialApply(_g5, m1);   // g5 (1 - g_mu) U Y(s)(x+mu)
                a2[j] = mdwfMonomialApply(_g5, p1);   // g5 (1 + g_mu) Y(s)(x)
            }
            COMPLEX(double) *B = (s < static_cast<int>(Ls) / 2) ? lo : hi;
            for (int j2 = 0; j2 < 4; j2++) {
                for (int jp = 0; jp < 4; jp++) {
                    B[4 * j2 + jp] += bpc * (mdwfInner(yxr[j2], a1[jp]) - mdwfInner(unr[j2], a2[jp]));
                }
            }
        }
        for (int t = 0; t < _nterms; t++) {
            COMPLEX(double) B[16];
            for (int i = 0; i < 16; i++) {
                B[i] = _terms[t].axial ? hi[i] - lo[i] : hi[i] + lo[i];
            }
            const COMPLEX(double) v = mdwfTraceMonomial(B, _terms[t].r);
            const size_t idx = mdwfPionAccIndex<HaloDepth>(site, _terms[t].channel);
            _acc.setElement<COMPLEX(double)>(idx, _acc.getElement<COMPLEX(double)>(idx) + _terms[t].weight * v);
        }
    }
};

// f_ch(x) exp(i p.(x - x0)) for the slice reduction.
template<size_t HaloDepth>
struct MDWFPionProjection {
    LatticeContainerAccessor _acc;
    int _ch;
    double _p[3];
    int _x0[3];
    double _shift;   // sink position offset in units of the momentum: phase p.(x - x0) + _shift (half-link phase)

    MDWFPionProjection(LatticeContainerAccessor acc, int ch, const std::array<int, 3> &n,
                       const std::array<int, 4> &x0, int halfLinkDir = -1)
        : _acc(acc), _ch(ch), _shift(0.0) {
        typedef GIndexer<All, HaloDepth> GInd;
        const LatticeData lat = GInd::getLatData();
        const double L[3] = {static_cast<double>(lat.globLX), static_cast<double>(lat.globLY),
                             static_cast<double>(lat.globLZ)};
        for (int i = 0; i < 3; i++) {
            _p[i] = 2.0 * M_PI * n[i] / L[i];
            _x0[i] = x0[i];
        }
        if (halfLinkDir >= 0 && halfLinkDir < 3) {
            _shift = 0.5 * _p[halfLinkDir];
        }
    }

    __host__ __device__ COMPLEX(double) operator()(gSite site) {
        typedef GIndexer<All, HaloDepth> GInd;
        LatticeData lat = GInd::getLatData();
        const sitexyzt g = lat.globalPos(site.coord);
        const double phi = _p[0] * (static_cast<int>(g.x) - _x0[0]) + _p[1] * (static_cast<int>(g.y) - _x0[1])
                           + _p[2] * (static_cast<int>(g.z) - _x0[2]) + _shift;
        return _acc.getElement<COMPLEX(double)>(mdwfPionAccIndex<HaloDepth>(site, _ch))
               * COMPLEX(double)(cos(phi), sin(phi));
    }
};

struct MDWFPionRow {
    std::string direction;
    int n2, nvec;
    std::string channel;
    int coord;
    COMPLEX(double) value;
};

struct MDWFPionResult {
    std::vector<MDWFPionRow> rows;
    int iterations;
    double maxResidue;
    double seconds;
    double maxPcacDeviation;   // max_tau |dA0cP - 2(m PP + J5qP)| / (|dA0cP| + |2(m PP + J5qP)|), tau = 1..Nt-1
    double maxPcacDeviationA0; // max_tau |dA0cA0 - 2(m PA0 + J5qA0)| / (|dA0cA0| + |2(m PA0 + J5qA0)|), tau = 1..Nt-1
    double maxPcacDeviationZ;  // max_r |dAzcP - 2(m PP + J5qP)| / (|dAzcP| + |2(m PP + J5qP)|) along z, r = 1..Nz-1
    double maxVzConservation;  // max_r |VzcVz(r) - VzcVz(r-1)| / max_r |VzcVz|, r = 1..Nz-1
};

/*
 * The full measurement: 12 solves at x0, contractions, projections. `cols` are four 5D work fields, `acc` and `red`
 * reduction containers (resized here).
 */
template<size_t HaloDepth, size_t Ls, class Solver>
MDWFPionResult mdwfMeasurePionDynamics(MDWFResidualMassMeasurement<HaloDepth, Ls, Solver> &meas,
                                       Gaugefield<double, true, HaloDepth, R18> &fermionGauge,
                                       std::array<MDWFSpinor<double, true, All, HaloDepth, Ls> *, 4> cols,
                                       LatticeContainer<true, COMPLEX(double)> &acc,
                                       LatticeContainer<true, COMPLEX(double)> &red, const std::array<int, 4> &x0,
                                       double mass, double b5, int n2max, int maxIter, double precision) {
    typedef GIndexer<All, HaloDepth> GInd;
    const LatticeData lat = GInd::getLatData();
    const double c5 = b5 - 1.0;
    CalcGSite<All, HaloDepth> calcGSite;
    MDWFPionResult result;
    result.iterations = 0;
    result.maxResidue = 0.0;
    result.seconds = 0.0;
    result.maxPcacDeviation = 0.0;
    result.maxPcacDeviationA0 = 0.0;
    result.maxPcacDeviationZ = 0.0;
    result.maxVzConservation = 0.0;

    acc.adjustSize(PD_NACC * lat.vol4);
    red.adjustSize(lat.vol4);
    iterateFunctorNoReturn<true, BLOCKSIZE>(MDWFPionZero<HaloDepth>(acc.getAccessor()), calcGSite, lat.vol4);

    const MDWFSpinMatrix g5 = mdwfGammaMatrix("5");
    auto prod = [](const MDWFSpinMatrix &a, const MDWFSpinMatrix &b) { return mdwfSpinProduct(a, b); };
    auto dag = [](const MDWFSpinMatrix &a) { return mdwfSpinDagger(a); };

    // Local terms: L = g5 Gsnk, R = Gsrc g5.
    std::vector<MDWFPionLocalTerm> local;
    auto addLocal = [&](const MDWFSpinMatrix &snk, const MDWFSpinMatrix &src, double w, int ch) {
        local.push_back({mdwfMonomial(prod(g5, snk)), mdwfMonomial(prod(src, g5)), w, ch});
    };
    const MDWFSpinMatrix gT = mdwfGammaMatrix("T"), one = mdwfSpinIdentity();
    const MDWFSpinMatrix A0 = prod(gT, g5);
    addLocal(g5, g5, 1.0, PD_PP);
    addLocal(A0, dag(A0), 1.0, PD_A0A0);
    addLocal(A0, g5, 1.0, PD_A0P);
    addLocal(g5, dag(A0), 1.0, PD_PA0);
    addLocal(gT, dag(gT), 1.0, PD_V0V0);
    addLocal(one, one, 1.0, PD_SS);
    static const char *spatial[3] = {"X", "Y", "Z"};
    for (int i = 0; i < 3; i++) {
        const MDWFSpinMatrix gi = mdwfGammaMatrix(spatial[i]);
        const MDWFSpinMatrix ai = prod(gi, g5);
        addLocal(gi, dag(gi), 1.0 / 3.0, PD_ViVi);
        addLocal(ai, dag(ai), 1.0 / 3.0, PD_AiAi);
    }

    // Spatial tensor (sink i, source j^+), local-local: two passes of 9 terms (MDWF_PION_MAX_LOCAL).
    std::vector<MDWFPionLocalTerm> tensorV, tensorA;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            const MDWFSpinMatrix gi = mdwfGammaMatrix(spatial[i]), gj = mdwfGammaMatrix(spatial[j]);
            tensorV.push_back({mdwfMonomial(prod(g5, gi)), mdwfMonomial(prod(dag(gj), g5)), 1.0,
                               mdwfPionTensorIndex(0, i, j)});
            tensorA.push_back({mdwfMonomial(prod(g5, prod(gi, g5))), mdwfMonomial(prod(dag(prod(gj, g5)), g5)), 1.0,
                               mdwfPionTensorIndex(1, i, j)});
        }
    }
    // AzP (local Az sink, g5 source) rides on the V tensor pass (10 <= MDWF_PION_MAX_LOCAL terms).
    const MDWFSpinMatrix Az = prod(mdwfGammaMatrix("Z"), g5);
    tensorV.push_back({mdwfMonomial(prod(g5, Az)), mdwfMonomial(prod(g5, g5)), 1.0, PD_AzP});

    // Conserved terms per mu: R = X g5.
    std::array<std::vector<MDWFPionConservedTerm>, 4> conserved;
    conserved[3].push_back({mdwfMonomial(prod(g5, g5)), 1.0, PD_A0cP, 1});
    conserved[3].push_back({mdwfMonomial(prod(dag(A0), g5)), 1.0, PD_A0cA0, 1});
    conserved[3].push_back({mdwfMonomial(prod(dag(gT), g5)), 1.0, PD_V0cV0, 0});
    for (int i = 0; i < 3; i++) {
        const MDWFSpinMatrix gi = mdwfGammaMatrix(spatial[i]);
        const MDWFSpinMatrix ai = prod(gi, g5);
        conserved[i].push_back({mdwfMonomial(prod(dag(gi), g5)), 1.0 / 3.0, PD_VicVi, 0});
        conserved[i].push_back({mdwfMonomial(prod(dag(ai), g5)), 1.0 / 3.0, PD_AicAi, 1});
        for (int j = 0; j < 3; j++) {   // tensor: conserved sink mu = i, local source j^+
            const MDWFSpinMatrix gj = mdwfGammaMatrix(spatial[j]);
            conserved[i].push_back({mdwfMonomial(prod(dag(gj), g5)), 1.0, mdwfPionTensorIndex(2, i, j), 0});
            conserved[i].push_back({mdwfMonomial(prod(dag(prod(gj, g5)), g5)), 1.0, mdwfPionTensorIndex(3, i, j), 1});
        }
    }
    conserved[2].push_back({mdwfMonomial(prod(g5, g5)), 1.0, PD_AzcP, 1});   // AzcP: conserved axial z, g5 source

    for (int colour = 0; colour < 3; colour++) {
        for (int spin = 0; spin < 4; spin++) {
            const MDWFResidualMassComponent c = meas.measureComponent(3 * spin + colour, x0[0], x0[1], x0[2], x0[3],
                                                                      maxIter, precision);
            result.iterations += c.iterations;
            result.maxResidue = std::max(result.maxResidue, c.residue);
            result.seconds += c.seconds;
            *cols[spin] = meas.solution5D();
            mdwfUpdateHalo(*cols[spin]);
        }
        const MDWFPionColumns<HaloDepth, Ls> q(*cols[0], *cols[1], *cols[2], *cols[3]);
        iterateFunctorNoReturn<true, BLOCKSIZE>(MDWFPionLocalKernel<HaloDepth, Ls>(q, acc.getAccessor(), local,
                                                                                    mdwfMonomial(prod(dag(A0), g5))),
                                                calcGSite, lat.vol4);
        for (const auto *tl : {&tensorV, &tensorA}) {
            iterateFunctorNoReturn<true, BLOCKSIZE>(
                MDWFPionLocalKernel<HaloDepth, Ls>(q, acc.getAccessor(), *tl, mdwfMonomial(prod(dag(A0), g5)), false),
                calcGSite, lat.vol4);
        }
        for (int mu = 0; mu < 4; mu++) {
            iterateFunctorNoReturn<true, BLOCKSIZE>(
                MDWFPionConservedKernel<HaloDepth, Ls>(q, fermionGauge, acc.getAccessor(), mu, colour, x0, mass, b5,
                                                       c5, conserved[mu]),
                calcGSite, lat.vol4);
        }
    }

    // Momentum orbits |n|^2 <= n2max, as GPT (itertools.product order does not matter for the orbit average).
    std::map<int, std::vector<std::array<int, 3>>> orbits;
    const int nmax = static_cast<int>(std::floor(std::sqrt(static_cast<double>(n2max))));
    for (int nx = -nmax; nx <= nmax; nx++) {
        for (int ny = -nmax; ny <= nmax; ny++) {
            for (int nz = -nmax; nz <= nmax; nz++) {
                const int n2 = nx * nx + ny * ny + nz * nz;
                if (n2 <= n2max) {
                    orbits[n2].push_back({nx, ny, nz});
                }
            }
        }
    }
    const int Nt = static_cast<int>(lat.globLT), Nz = static_cast<int>(lat.globLZ);
    static const int temporal[14] = {PD_PP,   PD_A0A0, PD_A0P,  PD_PA0, PD_V0V0,  PD_J5qP, PD_A0cP,
                                     PD_A0cA0, PD_V0cV0, PD_ViVi, PD_AiAi, PD_VicVi, PD_AicAi, PD_J5qA0};
    std::vector<COMPLEX(double)> slices;
    std::map<int, std::vector<COMPLEX(double)>> zeroMomentum;
    for (const auto &orbit : orbits) {
        for (const int ch : temporal) {
            std::vector<COMPLEX(double)> sum(Nt, COMPLEX(double)(0.0, 0.0));
            for (const auto &n : orbit.second) {
                red.template iterateFunctor<BLOCKSIZE>(MDWFPionProjection<HaloDepth>(acc.getAccessor(), ch, n, x0),
                                                       calcGSite, MDWFWriteAtSlices<HaloDepth>(3), lat.vol4);
                red.reduceStackedLocal(slices, Nt, lat.vol4 / Nt);
                for (int t = 0; t < Nt; t++) {
                    sum[t] += slices[t];
                }
            }
            const double inv = 1.0 / static_cast<double>(orbit.second.size());
            for (int tau = 0; tau < Nt; tau++) {
                result.rows.push_back({"temporal_t", orbit.first, static_cast<int>(orbit.second.size()),
                                       mdwfPionChannelName(ch), tau, inv * sum[(x0[3] + tau) % Nt]});
            }
            if (orbit.first == 0) {
                zeroMomentum[ch] = sum;
            }
        }
    }
    // Transverse / longitudinal split of the spatial tensors (see header).
    static const char *tlName[4][2] = {{"ViVi_T", "ViVi_L"}, {"AiAi_T", "AiAi_L"}, {"VicVi_T", "VicVi_L"},
                                       {"AicAi_T", "AicAi_L"}};
    for (const auto &orbit : orbits) {
        for (int type = 0; type < 4; type++) {
            std::vector<COMPLEX(double)> sT(Nt, COMPLEX(double)(0.0, 0.0)), sL(Nt, COMPLEX(double)(0.0, 0.0));
            for (const auto &n : orbit.second) {
                std::vector<COMPLEX(double)> G[3][3];
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        red.template iterateFunctor<BLOCKSIZE>(
                            MDWFPionProjection<HaloDepth>(acc.getAccessor(), mdwfPionTensorIndex(type, i, j), n, x0,
                                                          type >= 2 ? i : -1),
                            calcGSite, MDWFWriteAtSlices<HaloDepth>(3), lat.vol4);
                        red.reduceStackedLocal(G[i][j], Nt, lat.vol4 / Nt);
                    }
                }
                const double nn = std::sqrt(static_cast<double>(orbit.first));
                for (int t = 0; t < Nt; t++) {
                    const COMPLEX(double) tr = G[0][0][t] + G[1][1][t] + G[2][2][t];
                    if (orbit.first == 0) {
                        sT[t] += tr / 3.0;
                        sL[t] += tr / 3.0;
                        continue;
                    }
                    COMPLEX(double) gl(0.0, 0.0);
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {
                            gl += (n[i] / nn) * (n[j] / nn) * G[i][j][t];
                        }
                    }
                    sL[t] += gl;
                    sT[t] += 0.5 * (tr - gl);
                }
            }
            const double inv = 1.0 / static_cast<double>(orbit.second.size());
            for (int k = 0; k < 2; k++) {
                const std::vector<COMPLEX(double)> &s = k ? sL : sT;
                for (int tau = 0; tau < Nt; tau++) {
                    result.rows.push_back({"temporal_t", orbit.first, static_cast<int>(orbit.second.size()),
                                           tlName[type][k], tau, inv * s[(x0[3] + tau) % Nt]});
                }
            }
        }
    }

    // Screening direction z at p = 0 (see header): plain channels, then the tensor diagonals.
    std::vector<std::pair<int, std::string>> zch;
    for (const int ch : {PD_PP, PD_SS, PD_A0A0, PD_J5qP, PD_AzP, PD_AzcP, PD_V0V0, PD_V0cV0, PD_A0cA0}) {
        zch.push_back({ch, mdwfPionChannelName(ch)});
    }
    static const char *tdiag[4] = {"V%sV%s", "A%sA%s", "V%scV%s", "A%scA%s"};
    static const char *xyz[3] = {"x", "y", "z"};
    for (int type = 0; type < 4; type++) {
        for (int i = 0; i < 3; i++) {
            char nm[16];
            snprintf(nm, sizeof(nm), tdiag[type], xyz[i], xyz[i]);
            zch.push_back({mdwfPionTensorIndex(type, i, i), nm});
        }
    }
    std::map<std::string, std::vector<COMPLEX(double)>> zcorr;
    for (const auto &c : zch) {
        red.template iterateFunctor<BLOCKSIZE>(MDWFPionProjection<HaloDepth>(acc.getAccessor(), c.first, {0, 0, 0}, x0),
                                               calcGSite, MDWFWriteAtSlices<HaloDepth>(2), lat.vol4);
        red.reduceStackedLocal(slices, Nz, lat.vol4 / Nz);
        std::vector<COMPLEX(double)> v(Nz);
        for (int r = 0; r < Nz; r++) {
            v[r] = slices[(x0[2] + r) % Nz];
            result.rows.push_back({"spatial_z", 0, 1, c.second, r, v[r]});
        }
        zcorr[c.second] = v;
    }
    // Spatial PCAC along z: AzcP(r) - AzcP(r-1) = 2 [m PP(r) + J5qP(r)], r != 0.
    for (int r = 1; r < Nz; r++) {
        const double lhs = real(zcorr["AzcP"][r]) - real(zcorr["AzcP"][r - 1]);
        const double rhs = 2.0 * (mass * real(zcorr["PP"][r]) + real(zcorr["J5qP"][r]));
        const double rel = std::abs(lhs - rhs) / std::max(std::abs(lhs) + std::abs(rhs), 1e-300);
        rootLogger.info("MDWF pion dynamics: z-PCAC r = ", r, ": dAzcP = ", lhs, ", 2(m PP + J5qP) = ", rhs,
                        ", rel = ", rel);
        result.maxPcacDeviationZ = std::max(result.maxPcacDeviationZ, rel);
    }
    // Vector current conservation along z: VzcVz(r) independent of r != 0.
    double vzScale = 1e-300;
    for (int r = 0; r < Nz; r++) {
        vzScale = std::max(vzScale, std::abs(real(zcorr["VzcVz"][r])));
    }
    for (int r = 1; r < Nz; r++) {
        const double d = std::abs(real(zcorr["VzcVz"][r]) - real(zcorr["VzcVz"][r - 1])) / vzScale;
        result.maxVzConservation = std::max(result.maxVzConservation, d);
    }
    rootLogger.info("MDWF pion dynamics: VzcVz(r) = ", real(zcorr["VzcVz"][1]), " ... ", real(zcorr["VzcVz"][Nz / 2]),
                    ", max |dVzcVz| / max |VzcVz| = ", result.maxVzConservation);

    // PCAC: A0cP(tau) - A0cP(tau-1) = 2 [m PP(tau) + J5qP(tau)], tau != 0 (real parts, as GPT's diagnostic).
    auto at = [&](int ch, int tau) { return real(zeroMomentum[ch][(x0[3] + tau) % Nt]); };
    for (int tau = 1; tau < Nt; tau++) {
        const double lhs = at(PD_A0cP, tau) - at(PD_A0cP, tau - 1);
        const double rhs = 2.0 * (mass * at(PD_PP, tau) + at(PD_J5qP, tau));
        const double rel = std::abs(lhs - rhs) / std::max(std::abs(lhs) + std::abs(rhs), 1e-300);
        rootLogger.info("MDWF pion dynamics: PCAC tau = ", tau, ": dA0cP = ", lhs, ", 2(m PP + J5qP) = ", rhs,
                        ", rel = ", rel);
        result.maxPcacDeviation = std::max(result.maxPcacDeviation, rel);
    }
    // Axial-charge WI: A0cA0(tau) - A0cA0(tau-1) = 2 [m PA0(tau) + J5qA0(tau)], tau != 0.
    for (int tau = 1; tau < Nt; tau++) {
        const double lhs = at(PD_A0cA0, tau) - at(PD_A0cA0, tau - 1);
        const double rhs = 2.0 * (mass * at(PD_PA0, tau) + at(PD_J5qA0, tau));
        const double rel = std::abs(lhs - rhs) / std::max(std::abs(lhs) + std::abs(rhs), 1e-300);
        rootLogger.info("MDWF pion dynamics: axial-charge WI tau = ", tau, ": dA0cA0 = ", lhs,
                        ", 2(m PA0 + J5qA0) = ", rhs, ", rel = ", rel);
        result.maxPcacDeviationA0 = std::max(result.maxPcacDeviationA0, rel);
    }
    return result;
}
