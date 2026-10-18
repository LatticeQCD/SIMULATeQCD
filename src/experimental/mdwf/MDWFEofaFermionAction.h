/*
 * Exact one-flavour algorithm (EOFA; Chen and Izubuchi, arXiv:1403.1683) for the MDWF strange determinant ratio
 * det(M(m1)) / det(M(m2)) (m1 = ms, m2 = pv_mass), in the form of GPT's
 * g.qcd.pseudofermion.action.exact_one_flavor_ratio (lib/gpt/qcd/pseudofermion/action/exact_one_flavor.py) with
 * Grid's CayleyFermion5D conventions, so that the two codes can be compared term by term.
 *
 * Operators. M(m_+, m_-) is the Mobius operator with boundary mass m_+ for P_+ (wraps at s = 0) and m_- for P_-
 * (wraps at s = Ls - 1), Grid's mass_plus / mass_minus (MDWFMobiusOperatorParameters, setMasses). 4D <-> 5D maps:
 *   I_u(phi)  = (P_+ phi at s = 0, P_- phi at s = Ls - 1, else 0)       ImportUnphysicalFermion
 *   D_-       = 1 - c5 D_W on every slice                                  Dminus
 *   I         = D_- I_u                                                    ImportPhysicalFermionSource
 *   E(psi)    = P_- psi(0) + P_+ psi(Ls - 1)                               ExportPhysicalFermionSolution
 *   R         : psi(s) -> psi(Ls - 1 - s)
 *   Prop(m_+, m_-) = E M(m_+, m_-)^-1 I                                   (the 4D propagator)
 *
 * Action, with a 4D pseudofermion phi and P12 = Prop(m1, m2), P11 = Prop(m1, m1):
 *   K = 1 + (m2 - m1) P_+ g5 P12 P_+ + (m1 - m2) P_- g5 P11 P_-,   S = phi^+ K phi      (2 solves).
 *
 * Heatbath: phi = r(K) eta, r ~ x^(-1/2) on [eofa.lambda_low, eofa.lambda_high] (AlgRemez, or GPT's Zolotarev with
 *   eofa.zolotarev_order > 0; r = c0 + sum a_i/(x + s_i)),
 * with each pole applied through the shifted-mass identity (Woodbury; GPT inv_sqrt_matrix), g_i = 1 / (1 + s_i):
 *   (K + s_i)^-1 v = g_i [ v - g_i (m2 - m1) P_+ g5 Prop(m1 + g_i (m2 - m1), m2) P_+ v
 *                            + g_i (m2 - m1) P_- g5 Prop(m1, m1 + g_i (m2 - m1)) P_- v ]   (2 solves per pole).
 *
 * Force (2 adjoint solves, as GPT): with the term T_s = c_s phi^+ P_s g5 E M_s^-1 I P_s phi, (s, M_s, c_s) =
 * (+, M(m1, m2), m2 - m1) and (-, M(m1, m1), m1 - m2),
 *   y = M_s^-+ R g5 I_u P_s phi   (= M_s^-+ E^+ g5 P_s phi),   x = g5 R D_-^+ y   (= M_s^-1 I P_s phi),
 *   dT_s = -c_s Re[y^+ dM_s x] + c_s Re[y^+ dD_- I_u P_s phi],
 * dM = dD_W Din(m_+, m_-), dD_- = -c5 dD_W: both are storage terms -2 n Re[left^+ dD_W right] (mdwfHmcStoreFermionForce)
 * with (left, right, n) = (y, Din x, c_s / 2) and (y, I_u P_s phi, c_s c5 / 2).
 *
 * c_sw = 0 only (the operators with m_+ != m_- use the dense M_oo^-1 kernel). Solves: the even/odd full-operator
 * solvers (MDWFMobiusEvenOddSolver, or MDWFMobiusEvenOddMixedSolver with mixed_precision), one operator re-pointed
 * at each mass pair by setMasses.
 */

#pragma once

#include "MDWFSpinor.h"

#include "MDWFEvenOddFermionActions.h"
#include "MDWFWilsonSlice.h"

#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// GPT's zolotarev_inverse_square_root (lib/gpt/algorithms/rational/zolotarev.py, rational_function.py), transcribed
// line by line so that both codes use the same poles: x^(-1/2) on [lambda_low, lambda_high] as
// constant + sum residues_i / (x + poles_i), max_relative_error = GPT's delta. GPT passes k (not k^2) as the
// parameter m of ellipj; kept as is.
inline void mdwfZolotarevEllipj(double u, double m, double &sn, double &cn, double &dn, double &K) {
    const double EPS = 1e-16;
    double a[9] = {0.0}, c[9] = {0.0};
    a[0] = 1.0;
    double b = std::sqrt(1.0 - m);
    c[0] = std::sqrt(m);
    double twon = 1.0;
    int i = 0;
    while (std::abs(c[i] / a[i]) > EPS) {
        if (i > 7) {
            break;
        }
        const double ai = a[i];
        i++;
        c[i] = (ai - b) / 2.0;
        const double t = std::sqrt(ai * b);
        a[i] = (ai + b) / 2.0;
        b = t;
        twon *= 2.0;
    }
    double phi = twon * a[i] * u;
    K = M_PI / (2.0 * a[i]);
    while (i > 0) {
        const double t = c[i] * std::sin(phi) / a[i];
        phi = (std::asin(t) + phi) / 2.0;
        i--;
    }
    sn = std::sin(phi);
    cn = std::cos(phi);
    dn = std::sqrt(1.0 - m * sn * sn);
}

inline MDWFRemezApproximation mdwfZolotarevInverseSqrt(double lambda_low, double lambda_high, int n) {
    if (n <= 0 || !(lambda_low > 0.0) || !(lambda_high > lambda_low)) {
        throw std::runtime_error(stdLogger.fatal("MDWF Zolotarev approximation needs order > 0 and 0 < low < high"));
    }
    // zolotarev_approx_inverse_square_root(n, eps), eps = (ra / rb)^2 with ra^2 = lambda_low, rb^2 = lambda_high
    const double eps = lambda_low / lambda_high;
    const double k = std::sqrt(1.0 - eps);
    double sn, cn, dn, Kk;
    mdwfZolotarevEllipj(0.0, k, sn, cn, dn, Kk);
    const double v = Kk / (2 * n + 1);
    std::vector<double> a(2 * n), c(2 * n);
    for (int i = 0; i < 2 * n; i++) {
        double K;
        mdwfZolotarevEllipj((i + 1) * v, k, sn, cn, dn, K);
        a[i] = (cn / sn) * (cn / sn);
        c[i] = sn * sn;
    }
    double cOdd = 1.0, cEven = 1.0;
    for (int i = 0; i < 2 * n; i += 2) {
        cOdd *= c[i];
        cEven *= c[i + 1];
    }
    const double d = std::pow(k, 2 * n + 1) * cOdd * cOdd;
    const double den = 1.0 + std::sqrt(1.0 - d * d);
    const double A = 2.0 / den * cOdd / cEven;
    // zeros u_i = -a[2i] rb^2, poles v_i = -a[2i+1] rb^2 in x; norm = A / rb; partial_fractions(u, v)
    std::vector<double> zeros(n), poles(n);
    for (int i = 0; i < n; i++) {
        zeros[i] = -a[2 * i] * lambda_high;
        poles[i] = -a[2 * i + 1] * lambda_high;
    }
    const double norm = A / std::sqrt(lambda_high);
    MDWFRemezApproximation r;
    r.pnum = 1;
    r.pden = 2;
    r.lambda_low = lambda_low;
    r.lambda_high = lambda_high;
    r.order = n;
    r.digits = 0;
    r.max_relative_error = d * d / (den * den);
    r.inverse.constant = norm;
    for (int i = 0; i < n; i++) {
        const double z = poles[i];
        double num = 1.0, dnm = 1.0;
        for (int j = 0; j < n; j++) {
            num *= z - zeros[j];
            if (j != i) {
                dnm *= z - poles[j];
            }
        }
        r.inverse.residues.push_back(norm * num / dnm);
        r.inverse.poles.push_back(-z);
    }
    return r;
}

// 5D <- 4D: P_+ in at s = 0, P_- in at s = Ls - 1, zero elsewhere (Grid ImportUnphysicalFermion).
template<class floatT, size_t HaloDepth, size_t Ls>
struct MDWFEofaImportUnphysical {
    Vect12ArrayAcc<floatT> _in;

    explicit MDWFEofaImportUnphysical(const MDWFSpinor<floatT, true, All, HaloDepth, 1> &in) : _in(in.getAccessor()) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        if (site.stack == 0) {
            return mdwfProjectPlus(_in.getElement(static_cast<const gSite &>(site)));
        }
        if (site.stack == Ls - 1) {
            return mdwfProjectMinus(_in.getElement(static_cast<const gSite &>(site)));
        }
        return Vect12<floatT>(0.0);
    }
};

// 4D <- 5D: P_- in(s = 0) + P_+ in(s = Ls - 1) (Grid ExportPhysicalFermionSolution).
template<class floatT, size_t HaloDepth, size_t Ls>
struct MDWFEofaExport {
    Vect12ArrayAcc<floatT> _in;
    typedef GIndexer<All, HaloDepth> GInd;

    explicit MDWFEofaExport(const MDWFSpinor<floatT, true, All, HaloDepth, Ls> &in) : _in(in.getAccessor()) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        const gSite &s4 = static_cast<const gSite &>(site);
        Vect12<floatT> out = mdwfProjectMinus(_in.getElement(GInd::getSiteStack(s4, 0)));
        out += mdwfProjectPlus(_in.getElement(GInd::getSiteStack(s4, Ls - 1)));
        return out;
    }
};

// psi(s) -> psi(Ls - 1 - s) (Grid's R in GPT's mobius operator).
template<class floatT, size_t HaloDepth, size_t Ls>
struct MDWFEofaReflect {
    Vect12ArrayAcc<floatT> _in;
    typedef GIndexer<All, HaloDepth> GInd;

    explicit MDWFEofaReflect(const MDWFSpinor<floatT, true, All, HaloDepth, Ls> &in) : _in(in.getAccessor()) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        return _in.getElement(GInd::getSiteStack(static_cast<const gSite &>(site), Ls - 1 - site.stack));
    }
};

// P_+ in (plus) or P_- in on a field of any Ls.
template<class floatT, size_t HaloDepth, size_t Ls>
struct MDWFEofaChiralProject {
    Vect12ArrayAcc<floatT> _in;
    bool _plus;

    MDWFEofaChiralProject(const MDWFSpinor<floatT, true, All, HaloDepth, Ls> &in, bool plus)
        : _in(in.getAccessor()), _plus(plus) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        const Vect12<floatT> v = _in.getElement(site);
        return _plus ? mdwfProjectPlus(v) : mdwfProjectMinus(v);
    }
};

template<size_t HaloDepth, size_t Ls>
class MDWFEofaRatioFermionAction {
public:
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, 1>;    // the 4D pseudofermion
    using Spinor5 = MDWFSpinor<double, true, All, HaloDepth, Ls>;
    using Solver = MDWFMobiusEvenOddSolver<double, HaloDepth, HaloDepth, Ls>;
    using MixedSolver = MDWFMobiusEvenOddMixedSolver<HaloDepth, HaloDepth, Ls>;
    using EvenOdd = typename Solver::EvenOdd;
    using Types = MDWFEvenOddTypes<HaloDepth, Ls>;
    using SchurMixed = MDWFEvenOddMixedSchurSolver<HaloDepth, Ls>;
    using History = typename SchurMixed::History;

    // Storage terms: (left, right) pairs of 5D fields owned by the action.
    class Terms {
        std::vector<const Spinor5 *> _left, _right;

    public:
        void add(const Spinor5 &left, const Spinor5 &right) {
            _left.push_back(&left);
            _right.push_back(&right);
        }
        size_t size() const { return _left.size(); }
        const Spinor5 &chi(size_t i) const { return *_right.at(i); }
        const Spinor5 &eta(size_t i) const { return *_left.at(i); }
    };

private:
    CommunicationBase &_commBase;
    MDWFHmcParameters _param;
    std::string _prefix;
    Gauge &_gauge;
    double _m1, _m2;
    MDWFRemezApproximation _invSqrt;
    MDWFRationalCoefficients<double> _heatbath;   // x^(-1/2) on the EOFA interval
    std::unique_ptr<Solver> _solver;
    std::unique_ptr<MixedSolver> _mixed;
    // Chronological force solves (mixed_precision and chrono_size > 0, as GPT's EOFA solver with solution history):
    // own two-mass operator, the Schur-normal mixed solver, one history per force operator (M(m1, m2), M(m1, m1)).
    std::unique_ptr<EvenOdd> _eoC;
    std::unique_ptr<typename Types::NormalOp> _normalC;
    std::unique_ptr<typename Types::Adapter> _adapterC;
    std::unique_ptr<SchurMixed> _schurC;
    std::unique_ptr<History> _history[2];
    std::unique_ptr<typename Types::SpinorE> _ce1, _ce2, _ce3;
    std::unique_ptr<typename Types::SpinorO> _co1, _co2;
    Spinor _phi, _eta, _fa, _fb, _fc;
    Spinor5 _va, _vb, _vc, _vd, _ve;
    HostGauge _ipdotPart;
    double _noiseNorm2;
    double _lastHeatbathError;
    int _lastIterations;
    int _solves;

    static MDWFHmcParameters validated(const MDWFHmcParameters &param) {
        if (param.csw != 0.0) {
            throw std::runtime_error(stdLogger.fatal("MDWF EOFA action: c_sw = 0 only"));
        }
        if (param.eofa_schur != 0 && param.eofa_schur != 2) {
            throw std::runtime_error(stdLogger.fatal("MDWF EOFA action: eofa_schur must be 0 or 2, got ", param.eofa_schur));
        }
        if (param.eofa_schur == 2 && param.mixed_precision) {
            throw std::runtime_error(stdLogger.fatal("MDWF EOFA action: eofa_schur = 2 is implemented for double-precision "
                                                     "solves only (mixed_precision = 0)"));
        }
        if (!(param.rhmc.ms > 0.0) || !(param.pv_mass > param.rhmc.ms)) {
            throw std::runtime_error(stdLogger.fatal("MDWF EOFA action needs 0 < ms < pv_mass, got ", param.rhmc.ms,
                                                     ", ", param.pv_mass));
        }
        if (!(param.eofa.lambda_low > 0.0) || !(param.eofa.lambda_high > param.eofa.lambda_low)
            || !(param.eofa.error > 0.0)) {
            throw std::runtime_error(stdLogger.fatal("MDWF EOFA action: invalid interval or error"));
        }
        return param;
    }

    static double dot4(Spinor &a, const Spinor &b) {
        const std::vector<COMPLEX(double)> d = a.dotProductStacked(b);
        return real<double>(d[0]);
    }

    EvenOdd &evenOdd() {
        return _mixed ? _mixed->evenOdd() : _solver->evenOdd();
    }

    // out = M(mPlus, mMinus)^-1 in (dagger: M^-+ in).
    void solve(Spinor5 &out, const Spinor5 &in, double mPlus, double mMinus, bool dagger, double precision) {
        evenOdd().setMasses(mPlus, mMinus);
        int iterations = 0;
        if (_mixed) {
            if (dagger) {
                _mixed->solveDagger(out, in, _param.max_iter, precision);
            } else {
                _mixed->solve(out, in, _param.max_iter, precision);
            }
            iterations = _mixed->lastIterations();
        } else {
            if (dagger) {
                _solver->solveDagger(out, in, _param.max_iter, precision);
            } else {
                _solver->solve(out, in, _param.max_iter, precision);
            }
            iterations = _solver->lastIterations();
        }
        _lastIterations = std::max(_lastIterations, iterations);
        _solves++;
    }

    /*
     * out = M(mPlus, mMinus)^-+ in with the chronological guess of history slot (as MDWFMobiusEvenOddSolver::
     * solveDagger): bhat_e = b_e - M_oe^+ M_oo^-+ b_o, Mhat^+ Mhat w = bhat_e (mixed CG from the Galerkin guess of
     * the previous w of this operator), y_e = Mhat w, y_o = M_oo^-+ (b_o - M_eo^+ y_e).
     */
    void solveDaggerChrono(Spinor5 &out, const Spinor5 &in, double mPlus, double mMinus, int slot, double precision) {
        EvenOdd &eo = *_eoC;
        eo.setMasses(mPlus, mMinus);
        eo.refresh();
        EvenOdd::split(*_ce1, *_co1, in);
        eo.MooInv(*_co2, *_co1, true);
        eo.MoeDagger(*_ce2, *_co2);
        *_ce1 -= *_ce2;   // bhat_e
        const MDWFCoupledMixedCGResult r = _schurC->solve(eo, *_adapterC, *_ce3, *_ce1, _param.max_iter, precision,
                                                          _history[slot].get());
        if (!r.converged) {
            throw std::runtime_error(stdLogger.fatal("MDWF EOFA chronological force solve did not converge: ",
                                                     r.iterations, " iterations, residue ", r.residue));
        }
        rootLogger.info("MDWF EOFA force solve (M(", mPlus, ", ", mMinus, ")^+, chrono): ", r.iterations,
                        " single-precision iterations, ", r.reliableUpdates, " reliable updates");
        _lastIterations = std::max(_lastIterations, r.iterations);
        _solves++;
        eo.schur(*_ce2, *_ce3, false);   // y_e
        eo.MeoDagger(*_co2, *_ce2);
        *_co1 -= *_co2;
        eo.MooInv(*_co2, *_co1, true);   // y_o
        EvenOdd::merge(out, *_ce2, *_co2);
    }

    // out = D_- in = in - c5 D_W in (dagger: in - c5 g5 D_W g5 in); uses _vd, _ve.
    void dminus(Spinor5 &out, const Spinor5 &in, bool dagger) {
        const double c5 = _param.b5 - 1.0;
        const double mass = mdwfShamirKernelMass(_param.M5);
        if (!dagger) {
            _vd = in;
            mdwfUpdateHalo(_vd);
            applyMDWFWilsonSlice<double, HaloDepth, HaloDepth, Ls>(_ve, _gauge, out, _vd, mass, 0.0);
        } else {
            applyMDWFGamma5<double, true, All, HaloDepth, Ls>(_vd, in, true);
            applyMDWFWilsonSlice<double, HaloDepth, HaloDepth, Ls>(out, _gauge, _ve, _vd, mass, 0.0);
            applyMDWFGamma5<double, true, All, HaloDepth, Ls>(_ve, out, false);
        }
        out = in;
        out.template axpyThisB<64>(-c5, _ve);
    }

    // out = Prop(mPlus, mMinus) in = E M^-1 D_- I_u in (4D); uses _va, _vb, _vc (+ dminus scratch).
    void propagator(Spinor &out, const Spinor &in, double mPlus, double mMinus, double precision) {
        _va.template iterateOverBulk<BLOCKSIZE>(MDWFEofaImportUnphysical<double, HaloDepth, Ls>(in));
        dminus(_vb, _va, false);
        solve(_vc, _vb, mPlus, mMinus, false, precision);
        out.template iterateOverBulk<BLOCKSIZE>(MDWFEofaExport<double, HaloDepth, Ls>(_vc));
    }

    // out += coef P_s g5 Prop(mPlus, mMinus) P_s in; uses _fb, _fc.
    void addChiralTerm(Spinor &out, const Spinor &in, bool plus, double coef, double mPlus, double mMinus,
                       double precision) {
        _fb.template iterateOverBulk<BLOCKSIZE>(MDWFEofaChiralProject<double, HaloDepth, 1>(in, plus));
        propagator(_fc, _fb, mPlus, mMinus, precision);
        applyMDWFGamma5<double, true, All, HaloDepth, 1>(_fb, _fc, false);
        _fc.template iterateOverBulk<BLOCKSIZE>(MDWFEofaChiralProject<double, HaloDepth, 1>(_fb, plus));
        out.template axpyThisB<64>(coef, _fc);
    }

    // out = K in (out != in).
    void applyK(Spinor &out, const Spinor &in) {
        out = in;
        addChiralTerm(out, in, true, _m2 - _m1, _m1, _m2, _param.precision);
        addChiralTerm(out, in, false, _m1 - _m2, _m1, _m1, _param.precision);
    }

public:
    MDWFEofaRatioFermionAction(CommunicationBase &commBase, Gauge &gauge, const MDWFHmcParameters &param,
                               const std::string &name = "MDWF_hmc_eofa")
        : _commBase(commBase),
          _param(validated(param)),
          _prefix(mdwfHmcInstancePrefix(name)),
          _gauge(gauge),
          _m1(param.rhmc.ms),
          _m2(param.pv_mass),
          _invSqrt(param.eofa.zolotarev_order > 0
                       ? mdwfZolotarevInverseSqrt(param.eofa.lambda_low, param.eofa.lambda_high,
                                                  param.eofa.zolotarev_order)
                       : mdwfRemezPowerForError(1, 2, param.eofa.lambda_low, param.eofa.lambda_high, param.eofa.error,
                                                param.eofa.max_order, param.eofa.digits)),
          _heatbath(mdwfRhmcCoefficients(_invSqrt.inverse)),
          _phi(commBase, _prefix + "_phi"),
          _eta(commBase, _prefix + "_noise"),
          _fa(commBase, _prefix + "_fa"),
          _fb(commBase, _prefix + "_fb"),
          _fc(commBase, _prefix + "_fc"),
          _va(commBase, _prefix + "_va"),
          _vb(commBase, _prefix + "_vb"),
          _vc(commBase, _prefix + "_vc"),
          _vd(commBase, _prefix + "_vd"),
          _ve(commBase, _prefix + "_ve"),
          _ipdotPart(commBase, _prefix + "_ipart"),
          _noiseNorm2(0.0),
          _lastHeatbathError(0.0),
          _lastIterations(0),
          _solves(0) {
        if (param.mixed_precision && param.chrono_size > 0) {
            _eoC = std::make_unique<EvenOdd>(gauge, param.M5, param.rhmc.ms, param.b5, 0.0, _prefix + "_ceo");
            _normalC = std::make_unique<typename Types::NormalOp>(*_eoC, commBase, _prefix + "_cnrm");
            _adapterC = std::make_unique<typename Types::Adapter>(*_normalC);
            _schurC = std::make_unique<SchurMixed>(gauge, param.M5, param.b5, 0.0, param.mixed_delta, _prefix + "_cms");
            for (int i = 0; i < 2; i++) {
                _history[i] = std::make_unique<History>(commBase, static_cast<size_t>(param.chrono_size),
                                                        _prefix + "_hist" + std::to_string(i));
            }
            _ce1 = std::make_unique<typename Types::SpinorE>(commBase, _prefix + "_cea");
            _ce2 = std::make_unique<typename Types::SpinorE>(commBase, _prefix + "_ceb");
            _ce3 = std::make_unique<typename Types::SpinorE>(commBase, _prefix + "_cec");
            _co1 = std::make_unique<typename Types::SpinorO>(commBase, _prefix + "_coa");
            _co2 = std::make_unique<typename Types::SpinorO>(commBase, _prefix + "_cob");
        }
        if (param.mixed_precision) {
            _mixed = std::make_unique<MixedSolver>(gauge, param.M5, param.rhmc.ms, param.b5, 0.0, _prefix + "_msol",
                                                   param.mixed_delta);
        } else {
            _solver = std::make_unique<Solver>(gauge, param.M5, param.rhmc.ms, param.b5, 0.0, _prefix + "_dsol");
            _solver->setSchurTwo(param.eofa_schur == 2);
        }
    }

    // phi = r(K) eta, r ~ K^(-1/2); S(phi) = |eta|^2 up to the approximation error.
    void heatbath(uint4 *randState) {
        _eta.gauss(randState);
        heatbathFromNoise();
    }

    // The same with eta already in noise() (regression test T7 fills it from hash fields).
    void heatbathFromNoise() {
        _lastIterations = 0;
        _solves = 0;
        mdwfUpdateHalo(_eta);
        _noiseNorm2 = dot4(_eta, _eta);
        const MDWFRationalCoefficients<double> &r = _heatbath;
        _phi = r.constant * _eta;
        for (size_t i = 0; i < r.shift.size(); i++) {
            const double g = 1.0 / (1.0 + r.shift[i]);
            const double mi = _m1 + g * (_m2 - _m1);
            // _fa = (K + s_i)^-1 eta / g
            _fa = _eta;
            addChiralTerm(_fa, _eta, true, -g * (_m2 - _m1), mi, _m2, _param.precision);
            addChiralTerm(_fa, _eta, false, g * (_m2 - _m1), _m1, mi, _param.precision);
            _phi.template axpyThisB<64>(r.numerator[i] * g, _fa);
        }
        mdwfUpdateHalo(_phi);
        if (_param.eofa.check_heatbath) {
            const double s = action();
            _lastHeatbathError = std::abs(s / _noiseNorm2 - 1.0);
            rootLogger.info("MDWF EOFA heatbath check: S(phi) = ", s, ", |eta|^2 = ", _noiseNorm2,
                            ", relative error ", _lastHeatbathError, " (approximation order ", r.shift.size(), ")");
        }
    }

    double noiseNorm2() const {
        return _noiseNorm2;
    }

    double action() {
        _lastIterations = 0;
        applyK(_fa, _phi);
        return dot4(_phi, _fa);
    }

    void force(HostGauge &ipdotHost, const HostGauge &gaugeHost) {
        _lastIterations = 0;
        const double precision = _param.force_precision > 0.0 ? _param.force_precision : _param.precision;
        const double c5 = _param.b5 - 1.0;
        for (int chirality = 0; chirality < 2; chirality++) {
            const bool plus = chirality == 0;
            const double mPlus = _m1;
            const double mMinus = plus ? _m2 : _m1;
            const double c = plus ? (_m2 - _m1) : (_m1 - _m2);
            // _va = u = I_u P_s phi; _vc = R g5 u; _vb = y = M^-+ R g5 u; _vc = x = g5 R D_-^+ y, then Din x.
            _fa.template iterateOverBulk<BLOCKSIZE>(MDWFEofaChiralProject<double, HaloDepth, 1>(_phi, plus));
            _va.template iterateOverBulk<BLOCKSIZE>(MDWFEofaImportUnphysical<double, HaloDepth, Ls>(_fa));
            applyMDWFGamma5<double, true, All, HaloDepth, Ls>(_vb, _va, false);
            _vc.template iterateOverBulk<BLOCKSIZE>(MDWFEofaReflect<double, HaloDepth, Ls>(_vb));
            if (_schurC) {
                solveDaggerChrono(_vb, _vc, mPlus, mMinus, chirality, precision);
            } else {
                solve(_vb, _vc, mPlus, mMinus, true, precision);
            }
            dminus(_vc, _vb, true);
            _vd.template iterateOverBulk<BLOCKSIZE>(MDWFEofaReflect<double, HaloDepth, Ls>(_vc));
            applyMDWFGamma5<double, true, All, HaloDepth, Ls>(_vc, _vd, false);
            evenOdd().setMasses(mPlus, mMinus);   // Din of this operator
            applyMDWFFifthDimCoupling<double, true, All, HaloDepth, Ls>(_vd, _vc, evenOdd().params().dinCoeff, true);
            mdwfUpdateHalo(_vb);
            mdwfUpdateHalo(_va);
            Terms terms;
            terms.add(_vb, _vd);   // -c Re[y^+ dD_W Din x]: n = c / 2
            terms.add(_vb, _va);   // -c c5 Re[y^+ dD_W u]: n = c c5 / 2
            const MDWFRationalCoefficients<double> coefficients{0.0, {c / 2.0, c * c5 / 2.0}, {0.0, 0.0}};
            mdwfHmcStoreFermionForce<HaloDepth, Ls>(plus ? ipdotHost : _ipdotPart, gaugeHost, terms, coefficients,
                                                    0.0, _commBase, _prefix + "_fstore");
        }
        mdwfHmcAddHostForce<HaloDepth>(ipdotHost, _ipdotPart);
    }

    Spinor &phi() {
        return _phi;
    }

    Spinor &noise() {
        return _eta;
    }

    double phiNorm2() {
        return dot4(_phi, _phi);
    }

    int lastIterations() const {
        return _lastIterations;
    }

    double lastHeatbathError() const {
        return _lastHeatbathError;
    }

    const MDWFRemezApproximation &inverseSqrt() const {
        return _invSqrt;
    }

    // Test access: out = K in, out = Prop(mPlus, mMinus) in, and the two force vectors of a chirality.
    void applyKForTest(Spinor &out, const Spinor &in) {
        applyK(out, in);
    }

    void propagatorForTest(Spinor &out, const Spinor &in, double mPlus, double mMinus) {
        propagator(out, in, mPlus, mMinus, _param.precision);
    }

    // x via the identity x = g5 R D_-^+ M^-+ R g5 I_u P_s phi, and directly as M^-1 D_- I_u P_s phi; returns
    // |x_identity - x_direct| / |x_direct|.
    double forceIdentityCheck(bool plus) {
        const double mMinus = plus ? _m2 : _m1;
        _fa.template iterateOverBulk<BLOCKSIZE>(MDWFEofaChiralProject<double, HaloDepth, 1>(_phi, plus));
        _va.template iterateOverBulk<BLOCKSIZE>(MDWFEofaImportUnphysical<double, HaloDepth, Ls>(_fa));
        applyMDWFGamma5<double, true, All, HaloDepth, Ls>(_vb, _va, false);
        _vc.template iterateOverBulk<BLOCKSIZE>(MDWFEofaReflect<double, HaloDepth, Ls>(_vb));
        solve(_vb, _vc, _m1, mMinus, true, _param.precision);
        dminus(_vc, _vb, true);
        _vd.template iterateOverBulk<BLOCKSIZE>(MDWFEofaReflect<double, HaloDepth, Ls>(_vc));
        applyMDWFGamma5<double, true, All, HaloDepth, Ls>(_vb, _vd, false);   // _vb = x (identity)
        dminus(_vc, _va, false);
        solve(_vd, _vc, _m1, mMinus, false, _param.precision);               // _vd = x (direct)
        _vb.template axpyThisB<64>(-1.0, _vd);
        const std::vector<COMPLEX(double)> dd = _vb.dotProductStacked(_vb);
        const std::vector<COMPLEX(double)> nn = _vd.dotProductStacked(_vd);
        double d = 0.0, n = 0.0;
        for (size_t s = 0; s < Ls; s++) {
            d += real<double>(dd[s]);
            n += real<double>(nn[s]);
        }
        return std::sqrt(d / n);
    }
};
