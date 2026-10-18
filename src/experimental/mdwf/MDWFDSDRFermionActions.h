/*
 * DSDR (dislocation-suppressing determinant ratio) factor for the MDWF HMC (MDWF_DETERMINANT_FACTORIZATION.md
 * section 6), in the convention of the GPT production driver (dsdr_hmc_redesigned_flexible_spec.py: twisted-mass
 * regulator ratios g.qcd.fermion.wilson_twisted_mass(mass = -M5, mu) with
 * g.qcd.pseudofermion.action.two_flavor_ratio_evenodd_schur).
 *
 * Operator. The 4D twisted-mass Wilson operator with the domain-wall kernel's negative mass,
 *
 *   M(mu) = D_W(-M5) + i mu gamma5,   D_W(-M5) = (4 - M5) - (1/2) sum_nu [(1 - gamma_nu) U psi(x+nu)
 *                                                                   + (1 + gamma_nu) U^+ psi(x-nu)],
 *
 * plain Wilson (no clover term), with the fermion boundary conditions of the gauge field it is given (the HMC driver
 * passes the antiperiodic copy for antiperiodic_t). This is Grid's WilsonTMFermion (Mooee = (4 + mass) + i mu gamma5)
 * with mass = -M5. The gamma matrices are SIMULATeQCD's chiral basis (fullSpinor.h), gamma5 = diag(1, 1, -1, -1).
 * Since D_W is gamma5-hermitian, M(mu)^+ M(mu) = D_W^+ D_W + mu^2 and M(mu) M(mu)^+ = D_W D_W^+ + mu^2, so the DSDR
 * weight
 *
 *   W = det(D_W^+ D_W + eps_f^2) / det(D_W^+ D_W + eps_b^2)   (eps_f < eps_b; RBC/UKQCD: eps_f = 0.02, eps_b = 0.5)
 *
 * is a two-flavour ratio det(M_f M_f^+) / det(M_b M_b^+) of twisted-mass operators. It suppresses configurations
 * with near-zero modes of D_W(-M5) (dislocations, topology change) and leaves the continuum limit unchanged.
 *
 * Even/odd. Hop connects even and odd sites only and the site-diagonal part d = (4 - M5) + i mu gamma5 is gauge
 * independent, so det M = det(d_oo) det Mhat with Mhat = d - H_eo d^-1 H_oe on the even sites, and det d_oo is a
 * constant (it cancels in Delta H; it is different for the two twists, so actions of different twists are not
 * comparable in absolute value, only their changes). d is diagonal in chirality: d = (4 - M5 + i mu) on spins 0, 1
 * and (4 - M5 - i mu) on spins 2, 3, and H_eo^+ = gamma5 H_oe gamma5.
 *
 * Ratio factor (GPT's two_flavor_ratio_evenodd_schur with M1 = M(mu_f), M2 = M(mu_b)):
 *
 *   S = phi^+ Mhat_b^+ (Mhat_f Mhat_f^+)^-1 Mhat_b phi,
 *   heatbath  phi = Mhat_b^+ (Mhat_b Mhat_b^+)^-1 Mhat_f eta = Mhat_b^-1 Mhat_f eta  (so S = eta^+ eta),
 *   force     chi = (Mhat_f Mhat_f^+)^-1 Mhat_b phi,
 *             dS = 2 Re[chi^+ dMhat_b phi] - 2 Re[chi^+ dMhat_f (Mhat_f^+ chi)].
 *
 * As in MDWFEvenOddFermionActions.h, Re[u^+ dMhat v] = Re[U^+ dD V] with the full-lattice vectors V = (v, -d^-1 H_oe v),
 * U = (u, -d^-+ H_eo^+ u) and dD the variation of the Wilson hopping term, which is what the all-link force storage
 * computes for one stack (Ls = 1, c_sw = 0; MDWFDeviceForceStorage.h). So the storage terms are (right V(Mhat_f^+ chi),
 * left U(chi), numerator +1) and (right V(phi), left U(chi), numerator -1).
 *
 * Ladder. As GPT's tm_ratios = "0.02:0.07,0.07:0.18,0.18:0.5", MDWFHmcParameters::dsdr_twists = eps_0 < eps_1 < ...
 * < eps_k gives the k factors det(M(eps_i) M(eps_i)^+) / det(M(eps_(i+1)) M(eps_(i+1))^+), whose product telescopes to
 * W(eps_0, eps_k); each factor is a separate force term (lightest first) for the multi-level integrator.
 *
 * Single rank, double precision, unpreconditioned CG on Mhat Mhat^+ from a zero start (4D: cheap next to the 5D
 * MDWF solves).
 */

#pragma once

#include "MDWFSpinor.h"

#include "MDWFEvenOddFermionActions.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// out = up P_+ in + lo P_- in (chiral basis: gamma5 = +1 on spins 0, 1 and -1 on spins 2, 3).
template<class floatT, Layout LatLayout, size_t HaloDepthSpin>
struct MDWFChiralScale {
    SpinorColorAcc<floatT> _in;
    COMPLEX(floatT) _up;
    COMPLEX(floatT) _lo;

    MDWFChiralScale(const Spinorfield<floatT, true, LatLayout, HaloDepthSpin, 12, 1> &in, COMPLEX(floatT) up,
                    COMPLEX(floatT) lo)
        : _in(in.getAccessor()), _up(up), _lo(lo) {}

    __device__ __host__ Vect12<floatT> operator()(gSiteStack site) {
        const ColorVect<floatT> v = _in.getColorVect(site);
        ColorVect<floatT> r;
        r[0] = _up * v[0];
        r[1] = _up * v[1];
        r[2] = _lo * v[2];
        r[3] = _lo * v[3];
        return convertColorVectToVect12(r);
    }
};

// Unit vector e_component at one lattice site (local coordinates; single rank), zero elsewhere.
template<class floatT, Layout LatLayout, size_t HaloDepthSpin>
struct MDWFPointSourceField {
    sitexyzt _x;
    int _component;

    MDWFPointSourceField(sitexyzt x, int component) : _x(x), _component(component) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);
        const sitexyzt c = site.coord;
        if (c.x == _x.x && c.y == _x.y && c.z == _x.z && c.t == _x.t) {
            out.data[_component] = COMPLEX(floatT)(1.0, 0.0);
        }
        return out;
    }
};

/*
 * Even/odd blocks of M(mu) = D_W(-M5) + i mu gamma5 (4D, one stack). Inputs need no halo (the hopping blocks copy
 * their input into a work field and refresh its halo). The full operator applyFull (All layout) is for tests and
 * for the GPT cross-check (unpreconditioned traces).
 */
template<size_t HaloDepth>
class MDWFTwistedWilsonEvenOdd {
public:
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using SpinorAll = MDWFSpinor<double, true, All, HaloDepth, 1>;
    using SpinorE = MDWFSpinor<double, true, Even, HaloDepth, 1>;
    using SpinorO = MDWFSpinor<double, true, Odd, HaloDepth, 1>;

private:
    Gauge &_gauge;
    double _M5;
    double _mass;   // 4 - M5, the site-diagonal part of D_W(-M5)
    double _mu;
    SpinorE _e1, _e2, _e3;
    SpinorO _o1, _o2, _o3;
    SpinorAll _a1, _a2;

    COMPLEX(double) diagUp(bool dagger, bool inverse) const {
        COMPLEX(double) c(_mass, dagger ? -_mu : _mu);
        return inverse ? COMPLEX(double)(1.0, 0.0) / c : c;
    }

    COMPLEX(double) diagLo(bool dagger, bool inverse) const {
        COMPLEX(double) c(_mass, dagger ? _mu : -_mu);
        return inverse ? COMPLEX(double)(1.0, 0.0) / c : c;
    }

    template<Layout L>
    void diag(MDWFSpinor<double, true, L, HaloDepth, 1> &out, const MDWFSpinor<double, true, L, HaloDepth, 1> &in,
              bool dagger, bool inverse) {
        out.template iterateOverBulk<BLOCKSIZE>(
            MDWFChiralScale<double, L, HaloDepth>(in, diagUp(dagger, inverse), diagLo(dagger, inverse)));
    }

    // out = H_{LOut,LIn} in (the Wilson hopping block, -1/2 included); tIn is a work field of parity LIn.
    template<Layout LOut, Layout LIn>
    void hop(MDWFSpinor<double, true, LOut, HaloDepth, 1> &out, const MDWFSpinor<double, true, LIn, HaloDepth, 1> &in,
             MDWFSpinor<double, true, LIn, HaloDepth, 1> &tIn) {
        tIn = in;
        mdwfUpdateHalo(tIn);
        out.template iterateOverBulk<BLOCKSIZE>(
            DiracWilsonEvenOdd2<double, LOut, LIn, HaloDepth, HaloDepth, 1, false>(_gauge, tIn, 0.0, 0.0));
    }

    // out = (H_{LIn,LOut})^+ in = gamma5 H_{LOut,LIn} gamma5 in; tIn / tOut work fields of parity LIn / LOut.
    template<Layout LOut, Layout LIn>
    void hopDagger(MDWFSpinor<double, true, LOut, HaloDepth, 1> &out,
                   const MDWFSpinor<double, true, LIn, HaloDepth, 1> &in, MDWFSpinor<double, true, LIn, HaloDepth, 1> &tIn,
                   MDWFSpinor<double, true, LOut, HaloDepth, 1> &tOut) {
        applyMDWFGamma5<double, true, LIn, HaloDepth, 1>(tIn, in, true);
        tOut.template iterateOverBulk<BLOCKSIZE>(
            DiracWilsonEvenOdd2<double, LOut, LIn, HaloDepth, HaloDepth, 1, false>(_gauge, tIn, 0.0, 0.0));
        applyMDWFGamma5<double, true, LOut, HaloDepth, 1>(out, tOut, false);
    }

public:
    MDWFTwistedWilsonEvenOdd(Gauge &gauge, double M5, double mu, const std::string &name)
        : _gauge(gauge),
          _M5(M5),
          _mass(4.0 - M5),
          _mu(mu),
          _e1(gauge.getComm(), name + "_twe1"),
          _e2(gauge.getComm(), name + "_twe2"),
          _e3(gauge.getComm(), name + "_twe3"),
          _o1(gauge.getComm(), name + "_two1"),
          _o2(gauge.getComm(), name + "_two2"),
          _o3(gauge.getComm(), name + "_two3"),
          _a1(gauge.getComm(), name + "_twa1"),
          _a2(gauge.getComm(), name + "_twa2") {}

    double M5() const { return _M5; }
    double mu() const { return _mu; }

    // Site-diagonal blocks d = (4 - M5) + i mu gamma5, their adjoints and inverses.
    void Mee(SpinorE &out, const SpinorE &in, bool dagger = false) { diag<Even>(out, in, dagger, false); }
    void Moo(SpinorO &out, const SpinorO &in, bool dagger = false) { diag<Odd>(out, in, dagger, false); }
    void MeeInv(SpinorE &out, const SpinorE &in, bool dagger = false) { diag<Even>(out, in, dagger, true); }
    void MooInv(SpinorO &out, const SpinorO &in, bool dagger = false) { diag<Odd>(out, in, dagger, true); }

    // Hopping blocks M_eo = H_eo (odd -> even), M_oe = H_oe (even -> odd), and their adjoints.
    void Meo(SpinorE &out, const SpinorO &in) { hop<Even, Odd>(out, in, _o1); }
    void Moe(SpinorO &out, const SpinorE &in) { hop<Odd, Even>(out, in, _e1); }
    void MeoDagger(SpinorO &out, const SpinorE &in) { hopDagger<Odd, Even>(out, in, _e1, _o1); }
    void MoeDagger(SpinorE &out, const SpinorO &in) { hopDagger<Even, Odd>(out, in, _o1, _e1); }

    // Mhat = M_ee - M_eo M_oo^-1 M_oe, or Mhat^+ = M_ee^+ - M_oe^+ M_oo^-+ M_eo^+.
    void schur(SpinorE &out, const SpinorE &in, bool dagger = false) {
        if (!dagger) {
            Moe(_o2, in);
            MooInv(_o3, _o2);
            Meo(_e3, _o3);
            Mee(out, in);
        } else {
            MeoDagger(_o2, in);
            MooInv(_o3, _o2, true);
            MoeDagger(_e3, _o3);
            Mee(out, in, true);
        }
        out.template axpyThisB<64>(-1.0, _e3);
    }

    // Unpreconditioned M (or M^+) on the whole lattice.
    void applyFull(SpinorAll &out, const SpinorAll &in, bool dagger = false) {
        if (!dagger) {
            _a1 = in;
            mdwfUpdateHalo(_a1);
            out.template iterateOverBulk<BLOCKSIZE>(
                DiracWilsonEvenOdd2<double, All, All, HaloDepth, HaloDepth, 1, false>(_gauge, _a1, 0.0, 0.0));
        } else {
            applyMDWFGamma5<double, true, All, HaloDepth, 1>(_a1, in, true);
            _a2.template iterateOverBulk<BLOCKSIZE>(
                DiracWilsonEvenOdd2<double, All, All, HaloDepth, HaloDepth, 1, false>(_gauge, _a1, 0.0, 0.0));
            applyMDWFGamma5<double, true, All, HaloDepth, 1>(out, _a2, false);
        }
        diag<All>(_a2, in, dagger, false);
        out += _a2;
    }

    static void split(SpinorE &even, SpinorO &odd, const SpinorAll &all) {
        even.template iterateOverBulk<BLOCKSIZE>(MDWFEvenOddExtract<double, Even, HaloDepth, 1>(all));
        odd.template iterateOverBulk<BLOCKSIZE>(MDWFEvenOddExtract<double, Odd, HaloDepth, 1>(all));
    }

    static void merge(SpinorAll &all, const SpinorE &even, const SpinorO &odd) {
        all.template iterateOverBulk<BLOCKSIZE>(MDWFEvenOddMerge<double, HaloDepth, 1>(even, odd));
        mdwfUpdateHalo(all);
    }
};

// Mhat Mhat^+ on the even sites (GPT's MMdag_evenodd), in the shape MDWFCoupledSolverAdapter / MDWFCoupledCG expect.
template<size_t HaloDepth>
class MDWFTwistedSchurNormalOperator {
public:
    using EvenOdd = MDWFTwistedWilsonEvenOdd<HaloDepth>;
    using Spinor = typename EvenOdd::SpinorE;

private:
    EvenOdd &_eo;
    Spinor _tmp;

public:
    MDWFTwistedSchurNormalOperator(EvenOdd &eo, CommunicationBase &comm, const std::string &name)
        : _eo(eo), _tmp(comm, name + "_tmp") {}

    void apply(Spinor &out, const Spinor &in, bool update = false) {
        _eo.schur(_tmp, in, true);
        _eo.schur(out, _tmp, false);
        if (update) {
            mdwfUpdateHalo(out);
        }
    }
};

// M^+ M on the whole lattice (unpreconditioned; tests and the GPT cross-check).
template<size_t HaloDepth>
class MDWFTwistedFullNormalOperator {
public:
    using EvenOdd = MDWFTwistedWilsonEvenOdd<HaloDepth>;
    using Spinor = typename EvenOdd::SpinorAll;

private:
    EvenOdd &_eo;
    Spinor _tmp;

public:
    MDWFTwistedFullNormalOperator(EvenOdd &eo, CommunicationBase &comm, const std::string &name)
        : _eo(eo), _tmp(comm, name + "_tmp") {}

    void apply(Spinor &out, const Spinor &in, bool update = false) {
        _eo.applyFull(_tmp, in, false);
        _eo.applyFull(out, _tmp, true);
        if (update) {
            mdwfUpdateHalo(out);
        }
    }
};

/*
 * Force terms -2 n Re[u^+ dMhat v] as full-lattice one-stack storage terms: right V = (v, -d^-1 H_oe v), left
 * U = (u, -d^-+ H_eo^+ u) (see the header comment). Vectors are formed when chi(term)/eta(term) is read; the
 * buffers remember their term.
 */
template<size_t HaloDepth>
class MDWFTwistedForceTerms {
public:
    using EvenOdd = MDWFTwistedWilsonEvenOdd<HaloDepth>;
    using SpinorE = typename EvenOdd::SpinorE;
    using SpinorO = typename EvenOdd::SpinorO;
    using SpinorAll = typename EvenOdd::SpinorAll;

    struct Buffers {
        SpinorAll chi, eta;
        SpinorO odd1, odd2;

        Buffers(CommunicationBase &comm, const std::string &name)
            : chi(comm, name + "_tchi"), eta(comm, name + "_teta"), odd1(comm, name + "_todd1"),
              odd2(comm, name + "_todd2") {}
    };

private:
    struct Term {
        const SpinorE *v;
        const SpinorE *u;
        EvenOdd *eo;
    };
    static constexpr size_t none = std::numeric_limits<size_t>::max();

    std::vector<Term> _terms;
    Buffers *_buf;
    mutable size_t _chiTerm;
    mutable size_t _etaTerm;

public:
    explicit MDWFTwistedForceTerms(Buffers &buffers) : _buf(&buffers), _chiTerm(none), _etaTerm(none) {}

    void add(const SpinorE &v, const SpinorE &u, EvenOdd &eo) {
        _terms.push_back({&v, &u, &eo});
    }

    size_t size() const {
        return _terms.size();
    }

    // V = (v, -M_oo^-1 M_oe v).
    const SpinorAll &chi(size_t term) const {
        if (_chiTerm != term) {
            const Term &t = _terms.at(term);
            t.eo->Moe(_buf->odd1, *t.v);
            t.eo->MooInv(_buf->odd2, _buf->odd1);
            _buf->odd2 *= COMPLEX(double)(-1.0, 0.0);
            EvenOdd::merge(_buf->chi, *t.v, _buf->odd2);
            _chiTerm = term;
        }
        return _buf->chi;
    }

    // U = (u, -M_oo^-+ M_eo^+ u).
    const SpinorAll &eta(size_t term) const {
        if (_etaTerm != term) {
            const Term &t = _terms.at(term);
            t.eo->MeoDagger(_buf->odd1, *t.u);
            t.eo->MooInv(_buf->odd2, _buf->odd1, true);
            _buf->odd2 *= COMPLEX(double)(-1.0, 0.0);
            EvenOdd::merge(_buf->eta, *t.u, _buf->odd2);
            _etaTerm = term;
        }
        return _buf->eta;
    }
};

/*
 * One DSDR factor det(Mhat_f Mhat_f^+) / det(Mhat_b Mhat_b^+), Mhat_x the even-site Schur complement of
 * M(mu_x) = D_W(-M5) + i mu_x gamma5 (header comment). mu_f = mu_b is allowed (the identity, for tests).
 */
template<size_t HaloDepth>
class MDWFDSDRRatioFermionAction {
public:
    using EvenOdd = MDWFTwistedWilsonEvenOdd<HaloDepth>;
    using NormalOp = MDWFTwistedSchurNormalOperator<HaloDepth>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, 1, NormalOp>;
    using Spinor = typename EvenOdd::SpinorE;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using CG = MDWFCoupledCG<double, Adapter>;
    using Terms = MDWFTwistedForceTerms<HaloDepth>;

private:
    CommunicationBase &_commBase;
    MDWFHmcParameters _param;
    double _muF;
    double _muB;
    std::string _prefix;
    EvenOdd _eoF;
    EvenOdd _eoB;
    NormalOp _normalF;
    NormalOp _normalB;
    Adapter _adapterF;
    Adapter _adapterB;
    Spinor _phi, _eta, _tmp, _y, _psi, _chi, _mchi;
    typename Terms::Buffers _buffers;
    double _noiseNorm2;
    double _lastAction;
    int _lastIterations;

    static double validatedM5(const MDWFHmcParameters &param, double muF, double muB) {
        const double M5 = param.dsdr_M5 > 0.0 ? param.dsdr_M5 : param.M5;
        if (!(muF > 0.0) || !(muB >= muF) || !(M5 > 0.0)) {
            throw std::runtime_error(stdLogger.fatal("MDWF DSDR factor requires 0 < mu_f <= mu_b and M5 > 0, got mu_f = ",
                                                     muF, ", mu_b = ", muB, ", M5 = ", M5));
        }
        return M5;
    }

    void solve(Adapter &adapter, Spinor &out, const Spinor &in, const char *what) {
        CG cg;
        const MDWFCoupledCGResult<double> result = cg.invert(adapter, out, in, _param.max_iter, _param.precision, true);
        _lastIterations = std::max(_lastIterations, result.iterations);
        if (!result.converged) {
            throw std::runtime_error(stdLogger.fatal("MDWF DSDR factor (", _muF, " / ", _muB, ") ", what,
                                                     " solve did not converge: iterations = ", result.iterations));
        }
    }

public:
    MDWFDSDRRatioFermionAction(CommunicationBase &commBase, Gauge &gauge, const MDWFHmcParameters &param, double muF,
                               double muB, const std::string &name = "MDWF_hmc_dsdr")
        : _commBase(commBase),
          _param(param),
          _muF(muF),
          _muB(muB),
          _prefix(mdwfHmcInstancePrefix(name)),
          _eoF(gauge, validatedM5(param, muF, muB), muF, _prefix + "_eof"),
          _eoB(gauge, validatedM5(param, muF, muB), muB, _prefix + "_eob"),
          _normalF(_eoF, commBase, _prefix + "_nrmf"),
          _normalB(_eoB, commBase, _prefix + "_nrmb"),
          _adapterF(_normalF),
          _adapterB(_normalB),
          _phi(commBase, _prefix + "_phi"),
          _eta(commBase, _prefix + "_eta"),
          _tmp(commBase, _prefix + "_tmp"),
          _y(commBase, _prefix + "_y"),
          _psi(commBase, _prefix + "_psi"),
          _chi(commBase, _prefix + "_chi"),
          _mchi(commBase, _prefix + "_mchi"),
          _buffers(commBase, _prefix + "_fb"),
          _noiseNorm2(0.0),
          _lastAction(0.0),
          _lastIterations(0) {}

    // phi = Mhat_b^+ (Mhat_b Mhat_b^+)^-1 Mhat_f eta; then S = eta^+ eta.
    void heatbath(uint4 *randState) {
        _eta.gauss(randState);
        heatbathFromNoise();
    }

    // The same with eta already in noise() (regression test T7 fills it from hash fields).
    void heatbathFromNoise() {
        _lastIterations = 0;
        mdwfUpdateHalo(_eta);
        _noiseNorm2 = _adapterF.norm2(_eta);
        _eoF.schur(_tmp, _eta, false);
        solve(_adapterB, _y, _tmp, "heatbath");
        _eoB.schur(_phi, _y, true);
    }

    double noiseNorm2() const {
        return _noiseNorm2;
    }

    // S = psi^+ (Mhat_f Mhat_f^+)^-1 psi, psi = Mhat_b phi.
    double action() {
        _lastIterations = 0;
        _eoB.schur(_psi, _phi, false);
        solve(_adapterF, _chi, _psi, "action");
        _lastAction = real<double>(_adapterF.dotProduct5D(_psi, _chi));
        return _lastAction;
    }

    double lastPseudofermionAction() const {
        return _lastAction;
    }

    double lastDetAction() const {
        return 0.0;
    }

    // dS = 2 Re[chi^+ dMhat_b phi] - 2 Re[chi^+ dMhat_f (Mhat_f^+ chi)], chi = (Mhat_f Mhat_f^+)^-1 Mhat_b phi.
    void force(HostGauge &ipdotHost, const HostGauge &gaugeHost) {
        _lastIterations = 0;
        _eoB.schur(_psi, _phi, false);
        solve(_adapterF, _chi, _psi, "force");
        _eoF.schur(_mchi, _chi, true);

        Terms terms(_buffers);
        terms.add(_mchi, _chi, _eoF);
        terms.add(_phi, _chi, _eoB);
        const MDWFRationalCoefficients<double> coefficients{0.0, {1.0, -1.0}, {0.0, 0.0}};
        mdwfHmcStoreFermionForce<HaloDepth, 1>(ipdotHost, gaugeHost, terms, coefficients, 0.0, _commBase,
                                               _prefix + "_fstore");
    }

    Spinor &phi() {
        return _phi;
    }

    Spinor &noise() {
        return _eta;
    }

    double phiNorm2() {
        return _adapterF.norm2(_phi);
    }

    int lastIterations() const {
        return _lastIterations;
    }

    double muNumerator() const {
        return _muF;
    }

    double muDenominator() const {
        return _muB;
    }

    EvenOdd &numeratorOperator() {
        return _eoF;
    }

    EvenOdd &denominatorOperator() {
        return _eoB;
    }
};

/*
 * DSDR ladder: factors (eps_i, eps_(i+1)) of MDWFHmcParameters::dsdr_twists = eps_0 < ... < eps_k (k >= 1), product
 * det(D_W^+ D_W + eps_0^2) / det(D_W^+ D_W + eps_k^2). Same interface as the Hasenbusch ladder; one force term per
 * factor, lightest first.
 */
template<size_t HaloDepth>
class MDWFDSDRFermionAction {
public:
    using Factor = MDWFDSDRRatioFermionAction<HaloDepth>;
    using Spinor = typename Factor::Spinor;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;

private:
    std::string _prefix;
    std::vector<double> _twists;
    std::vector<std::unique_ptr<Factor>> _factors;
    HostGauge _ipdotPart;

    static std::vector<double> ladder(const MDWFHmcParameters &param) {
        const std::vector<double> &t = param.dsdr_twists;
        if (t.size() < 2) {
            throw std::runtime_error(stdLogger.fatal("MDWF DSDR needs at least two twists (eps_f eps_b), got ", t.size()));
        }
        if (!(t.front() > 0.0)) {
            throw std::runtime_error(stdLogger.fatal("MDWF DSDR twists must be positive, got ", t.front()));
        }
        for (size_t i = 0; i + 1 < t.size(); i++) {
            if (!(t[i + 1] > t[i])) {
                throw std::runtime_error(stdLogger.fatal("MDWF DSDR twists must increase strictly; twist ", i + 1, " = ",
                                                         t[i + 1], " after ", t[i]));
            }
        }
        return t;
    }

public:
    MDWFDSDRFermionAction(CommunicationBase &commBase, Gauge &gauge, const MDWFHmcParameters &param,
                          const std::string &name = "MDWF_hmc_dsdrlad")
        : _prefix(mdwfHmcInstancePrefix(name)),
          _twists(ladder(param)),
          _ipdotPart(commBase, _prefix + "_ipdot_part") {
        for (size_t i = 0; i + 1 < _twists.size(); i++) {
            _factors.push_back(std::make_unique<Factor>(commBase, gauge, param, _twists[i], _twists[i + 1],
                                                        _prefix + "_f" + std::to_string(i)));
        }
    }

    size_t factorCount() const {
        return _factors.size();
    }

    Factor &factor(size_t i) {
        return *_factors.at(i);
    }

    const std::vector<double> &twists() const {
        return _twists;
    }

    void heatbath(uint4 *randState) {
        for (auto &f : _factors) {
            f->heatbath(randState);
        }
    }

    double noiseNorm2() const {
        double sum = 0.0;
        for (const auto &f : _factors) {
            sum += f->noiseNorm2();
        }
        return sum;
    }

    double action() {
        double sum = 0.0;
        for (auto &f : _factors) {
            sum += f->action();
        }
        return sum;
    }

    double lastPseudofermionAction() const {
        double sum = 0.0;
        for (const auto &f : _factors) {
            sum += f->lastPseudofermionAction();
        }
        return sum;
    }

    double lastDetAction() const {
        return 0.0;
    }

    void force(HostGauge &ipdotHost, const HostGauge &gaugeHost) {
        _factors.front()->force(ipdotHost, gaugeHost);
        for (size_t i = 1; i < _factors.size(); i++) {
            _factors[i]->force(_ipdotPart, gaugeHost);
            mdwfHmcAddHostForce<HaloDepth>(ipdotHost, _ipdotPart);
        }
    }

    void forceFactor(size_t i, HostGauge &ipdotHost, const HostGauge &gaugeHost) {
        _factors.at(i)->force(ipdotHost, gaugeHost);
    }

    size_t forceTermCount() const {
        return _factors.size();
    }

    void forceTerm(size_t i, HostGauge &ipdotHost, const HostGauge &gaugeHost) {
        forceFactor(i, ipdotHost, gaugeHost);
    }

    Spinor &phi() {
        return _factors.front()->phi();
    }

    int lastIterations() const {
        int m = 0;
        for (const auto &f : _factors) {
            m = std::max(m, f->lastIterations());
        }
        return m;
    }

    std::string describe() const {
        std::ostringstream s;
        for (size_t i = 0; i < _twists.size(); i++) {
            s << (i ? " < " : "") << _twists[i];
        }
        return s.str();
    }
};

// 2+1 (even/odd, light Hasenbusch ladder, strange RHMC) times the DSDR ladder. Force terms: Hasenbusch factors
// (lightest first), strange RHMC, DSDR factors (lightest first).
template<size_t HaloDepth, size_t Ls>
using MDWFEvenOddHasenbuschDSDRTwoPlusOneFermionAction =
    MDWFSumFermionAction<HaloDepth, MDWFEvenOddHasenbuschTwoPlusOneFermionAction<HaloDepth, Ls>,
                         MDWFDSDRFermionAction<HaloDepth>>;

template<size_t HaloDepth, size_t Ls>
using MDWFEvenOddHasenbuschDSDRTwoPlusOneHmc =
    MDWFHmcDriver<HaloDepth, Ls, MDWFEvenOddHasenbuschDSDRTwoPlusOneFermionAction<HaloDepth, Ls>>;

// The DSDR ladder alone (pure gauge times DSDR), for tests.
template<size_t HaloDepth, size_t Ls>
using MDWFDSDRHmc = MDWFHmcDriver<HaloDepth, Ls, MDWFDSDRFermionAction<HaloDepth>>;
