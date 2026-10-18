/*
 * Even-site (even/odd preconditioned) MDWF pseudofermion actions, stages E2a
 * (c_sw = 0) and E2b (clover) of EVEN_ODD_DESIGN.md.
 *
 * With Mhat = M_ee - M_eo M_oo^-1 M_oe (MDWFMobiusEvenOdd.h), det M = det M_oo
 * det Mhat. For c_sw = 0, M_oo = A Din + Shift with A = mass is gauge
 * independent, so det M_oo is a constant and the actions below sample the same
 * determinant ratios as MDWFPauliVillarsTwoFlavorFermionAction and
 * MDWFOneFlavorRhmcFermionAction with M^+M replaced by Mhat^+ Mhat on the even
 * sites (half-size pseudofermions, and the Schur-complement CG needs about
 * half the iterations of the unpreconditioned one, mdwfMobiusEvenOddTest):
 *
 *   two-flavour:  S = phi^+ Mhat_1 (Mhat_f^+ Mhat_f)^-1 Mhat_1^+ phi,
 *   one-flavour:  S = phi^+ r_B^(1/4) r_A^(-1/2) r_B^(1/4) phi,
 *                 A = Mhat_s^+ Mhat_s,  B = Mhat_1^+ Mhat_1,
 *
 * with heatbaths and force expressions identical to the unpreconditioned ones
 * (MDWFHmcFermionActions.h, MDWFRhmcFermionActions.h).
 *
 * Force. For c_sw = 0 only the hopping blocks depend on the gauge field
 * (dM_ee = dM_oo = 0), so for even-site vectors u, v
 *
 *   Re[u^+ dMhat v] = -Re[u^+ dM_eo w] - Re[u~^+ dM_oe v] = Re[U^+ dM V],
 *   w = M_oo^-1 M_oe v,  u~ = M_oo^-+ M_eo^+ u,
 *   U = (u, -u~),  V = (v, -w)   (full-lattice vectors, even and odd parts),
 *
 * with dM = dD_W Din the full-operator variation. Every even-site force term
 * -2 n Re[u^+ dMhat v] is therefore the existing all-link storage term with
 * right = Din V and left = U (MDWFEvenOddForceTerms builds them one term at a
 * time as the storage reads them).
 *
 * The one-flavour approximation intervals (MDWFRhmcParameters lambda_*) are
 * those of Mhat^+ Mhat, not of M^+ M.
 *
 * Clover (stage E2b). For c_sw != 0, dM_ee = dA_e Din and dM_oo = dA_o Din
 * no longer vanish, and
 *
 *   Re[u^+ dMhat v] = Re[u^+ dM_ee v] - Re[u^+ dM_eo w] - Re[u~^+ dM_oe v] + Re[u~^+ dM_oo w]
 *                   = Re[U^+ dM V]
 *
 * with the same U, V: the all-link storage of full-lattice vectors already
 * contains the site-diagonal clover variation, so the pseudofermion force
 * terms are unchanged. det M_oo now depends on the gauge field, so the actions
 * carry S_oo = -n_f log det M_oo(m) + n_f log det M_oo(pv_mass), n_f = 2 for the
 * two-flavour and 1 for the one-flavour action (det M = det M_oo det Mhat).
 * Since log det M_oo = sum_{x odd, chi} [Ls log det P - log det W] and P = b5 A + 1
 * does not depend on the boundary mass, the P terms cancel exactly in the
 * ratio and S_oo = n_f sum [log det W(m) - log det W(pv)], which is O(m R^Ls)
 * per site (|R| ~ 0.15, so ~1e-7 at Ls = 8). Its force is sum_x Tr[H dA] with
 * the closed-form 6x6 H = G_W(m) - G_W(pv), G_W(m) = (-1)^Ls Ls R^(Ls-1) P^-1
 * (c5 - b5 R) m W(m) (MDWFMobiusCloverEvenOdd::logDetWDifferenceColumns), fed to
 * the storage as sum_b e_b^+ dA h_b with left e_b and right h_b = H e_b on the
 * odd sites at s = 0: the hopping part of the storage then vanishes (it only
 * connects even and odd sites), leaving exactly the clover part. (The general
 * single-mass form, G = sum_s [M_oo^-1 Din]_ss from logDetMooColumns, is kept
 * as a cross-check: mdwfEvenOddStoreLogDetForce.) For c_sw = 0, S_oo is a
 * constant and is omitted.
 */

#pragma once

#include "MDWFSpinor.h"

#include "MDWFMixedPrecisionSolver.h"
#include "MDWFMobiusEvenOdd.h"
#include "MDWFRhmcFermionActions.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

template<size_t HaloDepth, size_t Ls>
struct MDWFEvenOddTypes {
    using EvenOdd = MDWFMobiusCloverEvenOdd<double, HaloDepth, HaloDepth, Ls>;
    using NormalOp = MDWFMobiusSchurNormalOperator<EvenOdd>;
    using Adapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, NormalOp>;
    using SpinorE = typename EvenOdd::SpinorE;
    using SpinorO = typename EvenOdd::SpinorO;
    using SpinorAll = typename EvenOdd::SpinorAll;
};

/*
 * Force terms -2 n Re[u^+ dMhat v] as full-lattice storage terms (right = Din V,
 * left = U, see the header comment). With schurLeft the left even vector is
 * Mhat u instead of u. Vectors are formed when chi(term)/eta(term) is read; the
 * buffers remember their term. All EvenOdd objects must be refresh()ed.
 */
template<size_t HaloDepth, size_t Ls>
class MDWFEvenOddForceTerms {
public:
    using Types = MDWFEvenOddTypes<HaloDepth, Ls>;
    using EvenOdd = typename Types::EvenOdd;
    using SpinorE = typename Types::SpinorE;
    using SpinorO = typename Types::SpinorO;
    using SpinorAll = typename Types::SpinorAll;

    struct Buffers {
        SpinorAll chi, eta, merged;
        SpinorE even;
        SpinorO odd1, odd2;

        // Shared by all force-term users of this type (filled per term, nothing kept between force calls).
        Buffers(CommunicationBase &comm, const std::string &name)
            : chi(comm, EvenOdd::sharedAllName("fterm_chi")), eta(comm, EvenOdd::sharedAllName("fterm_eta")),
              merged(comm, EvenOdd::sharedAllName("fterm_merged")), even(comm, EvenOdd::sharedEvenName("fterm_even")),
              odd1(comm, EvenOdd::sharedOddName("fterm_odd1")), odd2(comm, EvenOdd::sharedOddName("fterm_odd2")) {
            (void)name;
        }
    };

private:
    struct Term {
        const SpinorE *v;
        const SpinorE *u;
        EvenOdd *eo;
        bool schurLeft;
    };
    static constexpr size_t none = std::numeric_limits<size_t>::max();

    std::vector<Term> _terms;
    Buffers *_buf;
    mutable size_t _chiTerm;
    mutable size_t _etaTerm;

public:
    explicit MDWFEvenOddForceTerms(Buffers &buffers) : _buf(&buffers), _chiTerm(none), _etaTerm(none) {}

    void add(const SpinorE &v, const SpinorE &u, EvenOdd &eo, bool schurLeft) {
        _terms.push_back({&v, &u, &eo, schurLeft});
    }

    size_t size() const {
        return _terms.size();
    }

    // Din V, V = (v, -M_oo^-1 M_oe v).
    const SpinorAll &chi(size_t term) const {
        if (_chiTerm != term) {
            const Term &t = _terms.at(term);
            t.eo->Moe(_buf->odd1, *t.v);
            t.eo->MooInv(_buf->odd2, _buf->odd1);
            _buf->odd2 *= COMPLEX(double)(-1.0, 0.0);
            EvenOdd::merge(_buf->merged, *t.v, _buf->odd2);
            applyMDWFFifthDimCoupling<double, true, All, HaloDepth, Ls>(_buf->chi, _buf->merged,
                                                                       t.eo->params().dinCoeff, true);
            _chiTerm = term;
        }
        return _buf->chi;
    }

    // U = (u', -M_oo^-+ M_eo^+ u'), u' = u or Mhat u.
    const SpinorAll &eta(size_t term) const {
        if (_etaTerm != term) {
            const Term &t = _terms.at(term);
            if (t.schurLeft) {
                t.eo->schur(_buf->even, *t.u, false);
            } else {
                _buf->even = *t.u;
            }
            t.eo->MeoDagger(_buf->odd1, _buf->even);
            t.eo->MooInv(_buf->odd2, _buf->odd1, true);
            _buf->odd2 *= COMPLEX(double)(-1.0, 0.0);
            EvenOdd::merge(_buf->eta, _buf->even, _buf->odd2);
            _etaTerm = term;
        }
        return _buf->eta;
    }
};

// Force terms sum_b -2 n Re[e_b^+ dD_W g_b] with left e_b (unit, odd sites, s = 0) and right g_b = G e_b.
template<size_t HaloDepth, size_t Ls>
class MDWFEvenOddLogDetForceTerms {
public:
    using Types = MDWFEvenOddTypes<HaloDepth, Ls>;
    using EvenOdd = typename Types::EvenOdd;
    using SpinorE = typename Types::SpinorE;
    using SpinorO = typename Types::SpinorO;
    using SpinorAll = typename Types::SpinorAll;
    using Columns = std::vector<std::unique_ptr<SpinorO>>;

private:
    const Columns *_g;
    SpinorAll *_chi;
    SpinorAll *_eta;
    SpinorE *_zero;
    SpinorO *_unit;

public:
    MDWFEvenOddLogDetForceTerms(const Columns &g, SpinorAll &chi, SpinorAll &eta, SpinorE &zero, SpinorO &unit)
        : _g(&g), _chi(&chi), _eta(&eta), _zero(&zero), _unit(&unit) {
        _zero->template iterateOverBulk<BLOCKSIZE>(MDWFUnitSliceField<double, Even, HaloDepth, Ls>(-1, 0));
    }

    size_t size() const {
        return 12;
    }

    const SpinorAll &chi(size_t b) const {
        EvenOdd::merge(*_chi, *_zero, *(*_g).at(b));
        return *_chi;
    }

    const SpinorAll &eta(size_t b) const {
        _unit->template iterateOverBulk<BLOCKSIZE>(MDWFUnitSliceField<double, Odd, HaloDepth, Ls>(static_cast<int>(b), 0));
        EvenOdd::merge(*_eta, *_zero, *_unit);
        return *_eta;
    }
};

/*
 * Overwrites ipdotHost with the force of S = -nf log det M_oo(eo), i.e. storage numerators nf / 2
 * for the 12 terms e_b^+ dA g_b (dS = -nf sum Tr[G dA] = sum_b -2 (nf/2) Re[e_b^+ dA g_b]).
 * eo must be refresh()ed for the current gauge field; c_sw must be nonzero.
 */
template<size_t HaloDepth, size_t Ls>
void mdwfEvenOddStoreLogDetForce(Gaugefield<double, false, HaloDepth, R18> &ipdotHost,
                                 const Gaugefield<double, false, HaloDepth, R18> &gaugeHost,
                                 typename MDWFEvenOddTypes<HaloDepth, Ls>::EvenOdd &eo, double nf, double csw,
                                 CommunicationBase &commBase, const std::string &name) {
    using Types = MDWFEvenOddTypes<HaloDepth, Ls>;
    using SpinorE = typename Types::SpinorE;
    using SpinorO = typename Types::SpinorO;
    using SpinorAll = typename Types::SpinorAll;

    std::vector<std::unique_ptr<SpinorO>> g;
    for (int b = 0; b < 12; b++) {
        g.emplace_back(new SpinorO(commBase, name + "_g" + std::to_string(b) + "c"));
    }
    SpinorO unit(commBase, name + "_unit");
    SpinorO work1(commBase, name + "_ow1");
    SpinorO work2(commBase, name + "_ow2");
    eo.logDetMooColumns(g, unit, work1, work2);

    SpinorAll chi(commBase, name + "_dchi");
    SpinorAll eta(commBase, name + "_deta");
    SpinorE zero(commBase, name + "_dzero");
    MDWFEvenOddLogDetForceTerms<HaloDepth, Ls> terms(g, chi, eta, zero, unit);
    const MDWFRationalCoefficients<double> coefficients{0.0, std::vector<double>(12, 0.5 * nf),
                                                         std::vector<double>(12, 0.0)};
    mdwfHmcStoreFermionForce<HaloDepth, Ls>(ipdotHost, gaugeHost, terms, coefficients, csw, commBase,
                                            name + "_store");
}

/*
 * Overwrites ipdotHost with the force of S = nf [log det W(m) - log det W(m')] = -nf [log det M_oo(m)
 * - log det M_oo(m')] for two operators differing only in the boundary mass: dS = -nf sum Tr[H dA],
 * storage numerators nf / 2 for the 12 terms e_b^+ dA h_b. Both must be refresh()ed; c_sw nonzero.
 */
template<size_t HaloDepth, size_t Ls>
void mdwfEvenOddStoreOddDetRatioForce(Gaugefield<double, false, HaloDepth, R18> &ipdotHost,
                                      const Gaugefield<double, false, HaloDepth, R18> &gaugeHost,
                                      typename MDWFEvenOddTypes<HaloDepth, Ls>::EvenOdd &eoM,
                                      typename MDWFEvenOddTypes<HaloDepth, Ls>::EvenOdd &eoPrime, double nf,
                                      double csw, CommunicationBase &commBase, const std::string &name) {
    using Types = MDWFEvenOddTypes<HaloDepth, Ls>;
    using SpinorE = typename Types::SpinorE;
    using SpinorO = typename Types::SpinorO;
    using SpinorAll = typename Types::SpinorAll;

    std::vector<std::unique_ptr<SpinorO>> h;
    for (int b = 0; b < 12; b++) {
        h.emplace_back(new SpinorO(commBase, name + "_h" + std::to_string(b) + "c"));
    }
    Types::EvenOdd::logDetWDifferenceColumns(eoM, eoPrime, h);

    SpinorO unit(commBase, name + "_unit");
    SpinorAll chi(commBase, name + "_dchi");
    SpinorAll eta(commBase, name + "_deta");
    SpinorE zero(commBase, name + "_dzero");
    MDWFEvenOddLogDetForceTerms<HaloDepth, Ls> terms(h, chi, eta, zero, unit);
    const MDWFRationalCoefficients<double> coefficients{0.0, std::vector<double>(12, 0.5 * nf),
                                                         std::vector<double>(12, 0.0)};
    mdwfHmcStoreFermionForce<HaloDepth, Ls>(ipdotHost, gaugeHost, terms, coefficients, csw, commBase,
                                            name + "_store");
}

/*
 * Two-flavour even-site ratio det(Mhat_a^+ Mhat_a) / det(Mhat_b^+ Mhat_b) of two boundary masses a (numerator, the
 * lighter one in a Hasenbusch ladder) and b (denominator), all other operator parameters equal, including the clover
 * term S_oo = 2 [log det W(a) - log det W(b)] for c_sw != 0. With a = mf and b = pv_mass it is the Pauli-Villars light
 * pair (MDWFEvenOddPauliVillarsTwoFlavorFermionAction); a chain of these is the Hasenbusch ladder.
 */
template<size_t HaloDepth, size_t Ls>
class MDWFEvenOddTwoFlavorRatioFermionAction {
public:
    using Types = MDWFEvenOddTypes<HaloDepth, Ls>;
    using EvenOdd = typename Types::EvenOdd;
    using NormalOp = typename Types::NormalOp;
    using Adapter = typename Types::Adapter;
    using Spinor = typename Types::SpinorE;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using CG = MDWFCoupledCG<double, Adapter>;
    using Terms = MDWFEvenOddForceTerms<HaloDepth, Ls>;
    using Mixed = MDWFEvenOddMixedSchurSolver<HaloDepth, Ls>;

private:
    CommunicationBase &_commBase;
    MDWFHmcParameters _param;
    double _massNum;
    double _massDen;
    std::string _prefix;
    std::unique_ptr<EvenOdd> _ownF;  // empty when the operators are shared (Hasenbusch ladder)
    std::unique_ptr<EvenOdd> _own1;
    EvenOdd &_eoF;
    EvenOdd &_eo1;
    NormalOp _normalF;
    NormalOp _normal1;
    Adapter _adapterF;
    Adapter _adapter1;
    Spinor _phi, _eta, _tmp, _y, _psi, _chi, _mchi;
    typename Terms::Buffers _buffers;
    HostGauge _ipdotDet;
    Mixed *_mixed;                                     // shared mixed-precision solver, or nullptr (double CG)
    std::unique_ptr<typename Mixed::History> _forceHistory;
    double _noiseNorm2;
    double _lastPseudofermionAction;
    double _lastDetAction;
    int _lastIterations;

    static MDWFHmcParameters validated(const MDWFHmcParameters &param, double massNum, double massDen) {
        if (!(massDen > 0.0) || !(massDen > massNum)) {
            throw std::runtime_error(stdLogger.fatal("MDWF even/odd two-flavour ratio requires a denominator mass > 0 "
                                                     "and above the numerator mass, got ", massNum, " / ", massDen));
        }
        if (param.light_form != 0 && param.light_form != 1) {
            throw std::runtime_error(stdLogger.fatal("MDWF light_form must be 0 (SIMULATeQCD) or 1 (GPT), got ",
                                                     param.light_form));
        }
        if (param.light_form == 1 && param.csw != 0.0) {
            throw std::runtime_error(stdLogger.fatal("MDWF light_form = 1 (GPT form) is implemented for c_sw = 0 only"));
        }
        return param;
    }

    void refresh() {
        _eoF.refresh();
        _eo1.refresh();
    }

    // (eo^+ eo) out = in; MD force solves use force_precision and the chronological guess.
    void solve(EvenOdd &eo, Adapter &adapter, Spinor &out, Spinor &in, const char *what, bool forceSolve = false) {
        const double precision = (forceSolve && _param.force_precision > 0.0) ? _param.force_precision
                                                                             : _param.precision;
        if (_mixed != nullptr) {
            const auto start = std::chrono::steady_clock::now();
            const MDWFCoupledMixedCGResult result = _mixed->solve(eo, adapter, out, in, _param.max_iter, precision,
                                                                  forceSolve ? _forceHistory.get() : nullptr,
                                                                  gptForm());
            _lastIterations = std::max(_lastIterations, result.iterations);
            rootLogger.info("MDWF even/odd ratio solve (", _prefix, " ", what, "): ", result.iterations,
                            " single-precision iterations, ", result.reliableUpdates, " reliable updates, ",
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), " s");
            if (!result.converged) {
                throw std::runtime_error(stdLogger.fatal("MDWF even/odd ratio ", what, " mixed-precision solve did not "
                                                         "converge: iterations = ", result.iterations, ", residue = ",
                                                         result.residue));
            }
            return;
        }
        CG cg;
        const MDWFCoupledCGResult<double> result = cg.invert(adapter, out, in, _param.max_iter, precision, true);
        _lastIterations = std::max(_lastIterations, result.iterations);
        if (!result.converged) {
            throw std::runtime_error(stdLogger.fatal("MDWF even/odd Pauli-Villars ", what,
                                                     " solve did not converge: iterations = ", result.iterations));
        }
    }

public:
    MDWFEvenOddTwoFlavorRatioFermionAction(CommunicationBase &commBase, Gauge &gauge, const MDWFHmcParameters &param,
                                           double massNum, double massDen,
                                           const std::string &name = "MDWF_hmc_eoratio")
        : MDWFEvenOddTwoFlavorRatioFermionAction(commBase, param, massNum, massDen, mdwfHmcInstancePrefix(name), &gauge,
                                                 nullptr, nullptr) {}

    // Operators owned elsewhere (one per mass, shared by adjacent Hasenbusch factors; refreshed by every user).
    // mixed: shared mixed-precision solver (nullptr: double CG); with chrono_size > 0 the factor keeps its own
    // history of force solutions.
    MDWFEvenOddTwoFlavorRatioFermionAction(CommunicationBase &commBase, const MDWFHmcParameters &param, EvenOdd &eoNum,
                                           EvenOdd &eoDen, double massNum, double massDen, const std::string &name,
                                           Mixed *mixed = nullptr)
        : MDWFEvenOddTwoFlavorRatioFermionAction(commBase, param, massNum, massDen, mdwfHmcInstancePrefix(name), nullptr,
                                                 &eoNum, &eoDen) {
        _mixed = mixed;
        if (_mixed != nullptr && param.chrono_size > 0) {
            _forceHistory = std::make_unique<typename Mixed::History>(commBase, static_cast<size_t>(param.chrono_size),
                                                                      _prefix + "_force");
        }
    }

private:
    // gauge != nullptr: own both operators; otherwise use eoNum / eoDen.
    MDWFEvenOddTwoFlavorRatioFermionAction(CommunicationBase &commBase, const MDWFHmcParameters &param, double massNum,
                                           double massDen, const std::string &prefix, Gauge *gauge, EvenOdd *eoNum,
                                           EvenOdd *eoDen)
        : _commBase(commBase),
          _param(validated(param, massNum, massDen)),
          _massNum(massNum),
          _massDen(massDen),
          _prefix(prefix),
          _ownF(gauge ? std::make_unique<EvenOdd>(*gauge, param.M5, massNum, param.b5, param.csw, prefix + "_eof")
                      : nullptr),
          _own1(gauge ? std::make_unique<EvenOdd>(*gauge, param.M5, massDen, param.b5, param.csw, prefix + "_eo1")
                      : nullptr),
          _eoF(gauge ? *_ownF : *eoNum),
          _eo1(gauge ? *_own1 : *eoDen),
          _normalF(_eoF, commBase, _prefix + "_nrmf"),
          _normal1(_eo1, commBase, _prefix + "_nrm1"),
          _adapterF(_normalF),
          _adapter1(_normal1),
          _phi(commBase, _prefix + "_phi"),
          // Scratch (written before it is read in every call): the shared pool, also used by the strange RHMC.
          _eta(commBase, EvenOdd::sharedEvenName("pool0")),
          _tmp(commBase, EvenOdd::sharedEvenName("pool1")),
          _y(commBase, EvenOdd::sharedEvenName("pool2")),
          _psi(commBase, EvenOdd::sharedEvenName("pool3")),
          _chi(commBase, EvenOdd::sharedEvenName("pool4")),
          _mchi(commBase, EvenOdd::sharedEvenName("pool5")),
          _buffers(commBase, _prefix + "_fb"),
          _ipdotDet(commBase, _prefix + "_ipdet"),
          _mixed(nullptr),
          _noiseNorm2(0.0),
          _lastPseudofermionAction(0.0),
          _lastDetAction(0.0),
          _lastIterations(0) {
        // GPT form: every solve of this factor is with Mhat Mhat^+ (heatbath with B, action and force with A).
        _normalF.setMMdag(gptForm());
        _normal1.setMMdag(gptForm());
    }

    bool gptForm() const {
        return _param.light_form == 1;
    }

public:
    // phi = Mhat_1 (Mhat_1^+ Mhat_1)^-1 Mhat_f^+ eta; then the pseudofermion part of S is eta^+ eta.
    void heatbath(uint4 *randState) {
        _eta.gauss(randState);
        heatbathFromNoise();
    }

    // The same with eta already in noise() (exact-comparison suite tests/exact8 fills it from hash fields). noise() is
    // the shared scratch pool0: fill it right before this call.
    void heatbathFromNoise() {
        refresh();
        _lastIterations = 0;
        mdwfUpdateHalo(_eta);
        _noiseNorm2 = _adapterF.norm2(_eta);
        if (gptForm()) {
            // phi = B^+ (B B^+)^-1 A eta (GPT two_flavor_ratio_base.draw)
            _eoF.schur(_tmp, _eta, false);
            solve(_eo1, _adapter1, _y, _tmp, "heatbath");
            _eo1.schur(_phi, _y, true);
        } else {
            _eoF.schur(_tmp, _eta, true);
            solve(_eo1, _adapter1, _y, _tmp, "heatbath");
            _eo1.schur(_phi, _y, false);
        }
    }

    Spinor &noise() {
        return _eta;
    }

    double noiseNorm2() const {
        return _noiseNorm2;
    }

    double phiNorm2() {
        return _adapterF.norm2(_phi);
    }

    // S = psi^+ (Mhat_f^+ Mhat_f)^-1 psi + S_oo, S_oo = 2 [log det W(mf) - log det W(pv)] (c_sw != 0).
    double action() {
        refresh();
        _lastIterations = 0;
        _eo1.schur(_psi, _phi, !gptForm());   // psi = B^+ phi, or B phi (GPT form)
        solve(_eoF, _adapterF, _chi, _psi, "action");
        _lastPseudofermionAction = real<double>(_adapterF.dotProduct5D(_psi, _chi));
        _lastDetAction = (_param.csw != 0.0) ? 2.0 * (_eoF.logDetW() - _eo1.logDetW()) : 0.0;
        return _lastPseudofermionAction + _lastDetAction;
    }

    double lastPseudofermionAction() const {
        return _lastPseudofermionAction;
    }

    double lastDetAction() const {
        return _lastDetAction;
    }

    // dS = -2 Re[(Mhat_f chi)^+ dMhat_f chi] + 2 Re[phi^+ dMhat_1 chi], chi = (Mhat_f^+ Mhat_f)^-1 Mhat_1^+ phi.
    void force(HostGauge &ipdotHost, const HostGauge &gaugeHost) {
        refresh();
        _lastIterations = 0;
        _eo1.schur(_psi, _phi, !gptForm());
        solve(_eoF, _adapterF, _chi, _psi, "force", true);
        _eoF.schur(_mchi, _chi, gptForm());

        Terms terms(_buffers);
        if (gptForm()) {
            // S = phi^+ B^+ (A A^+)^-1 B phi, chi = (A A^+)^-1 B phi, w = A^+ chi (in _mchi):
            // dS = -2 Re[chi^+ dA w] + 2 Re[chi^+ dB phi] (GPT two_flavor_ratio_base.gradient)
            terms.add(_mchi, _chi, _eoF, false);
            terms.add(_phi, _chi, _eo1, false);
        } else {
            terms.add(_chi, _mchi, _eoF, false);
            terms.add(_chi, _phi, _eo1, false);
        }
        const MDWFRationalCoefficients<double> coefficients{0.0, {1.0, -1.0}, {0.0, 0.0}};
        mdwfHmcStoreFermionForce<HaloDepth, Ls>(ipdotHost, gaugeHost, terms, coefficients, _param.csw, _commBase,
                                                _prefix + "_fstore");
        if (_param.csw != 0.0) {
            mdwfEvenOddStoreOddDetRatioForce<HaloDepth, Ls>(_ipdotDet, gaugeHost, _eoF, _eo1, 2.0, _param.csw,
                                                            _commBase, _prefix + "_det");
            mdwfHmcAddHostForce<HaloDepth>(ipdotHost, _ipdotDet);
        }
    }

    Spinor &phi() {
        return _phi;
    }

    int lastIterations() const {
        return _lastIterations;
    }

    double massNumerator() const {
        return _massNum;
    }

    double massDenominator() const {
        return _massDen;
    }
};

// The light pair det(Mhat_f^+ Mhat_f) / det(Mhat_1^+ Mhat_1), masses mf and pv_mass.
template<size_t HaloDepth, size_t Ls>
class MDWFEvenOddPauliVillarsTwoFlavorFermionAction : public MDWFEvenOddTwoFlavorRatioFermionAction<HaloDepth, Ls> {
public:
    using Base = MDWFEvenOddTwoFlavorRatioFermionAction<HaloDepth, Ls>;
    using Gauge = typename Base::Gauge;

    MDWFEvenOddPauliVillarsTwoFlavorFermionAction(CommunicationBase &commBase, Gauge &gauge,
                                                  const MDWFHmcParameters &param,
                                                  const std::string &name = "MDWF_hmc_eopv")
        : Base(commBase, gauge, param, param.mf, param.pv_mass, name) {}
};

/*
 * Hasenbusch ladder of the two-flavour light determinant ratio (MDWF_DETERMINANT_FACTORIZATION.md section 3):
 *
 *   det(Mhat_f^+ Mhat_f) / det(Mhat_1^+ Mhat_1) = prod_{i=0}^{k} det(Mhat(m_i)^+ Mhat(m_i)) / det(Mhat(m_{i+1})^+ Mhat(m_{i+1})),
 *   m_0 = mf < m_1 < ... < m_k < m_{k+1} = pv_mass,   m_1 ... m_k = MDWFHmcParameters::hasenbusch_masses,
 *
 * an identity of determinants (adjacent endpoints cancel; the clover log det W terms telescope as well), with one
 * pseudofermion per factor. Each factor's force is smaller and smoother than the single ratio's, so the factors can
 * be integrated on different time scales (factor 0, the lightest, carries the most fluctuating, smallest force).
 * With no intermediate masses it is the single ratio, identical to MDWFEvenOddPauliVillarsTwoFlavorFermionAction.
 */
template<size_t HaloDepth, size_t Ls>
class MDWFEvenOddHasenbuschTwoFlavorFermionAction {
public:
    using Factor = MDWFEvenOddTwoFlavorRatioFermionAction<HaloDepth, Ls>;
    using EvenOdd = typename Factor::EvenOdd;
    using Spinor = typename Factor::Spinor;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;

private:
    std::string _prefix;
    std::vector<double> _masses;
    std::vector<std::unique_ptr<EvenOdd>> _ops;  // one per ladder mass; factor i uses _ops[i] / _ops[i + 1]
    std::unique_ptr<typename Factor::Mixed> _mixed;  // one single-precision operator for all masses (mixed_precision)
    std::vector<std::unique_ptr<Factor>> _factors;
    HostGauge _ipdotPart;

    static std::vector<double> ladder(const MDWFHmcParameters &param) {
        std::vector<double> m{param.mf};
        m.insert(m.end(), param.hasenbusch_masses.begin(), param.hasenbusch_masses.end());
        m.push_back(param.pv_mass);
        for (size_t i = 0; i + 1 < m.size(); i++) {
            if (!(m[i + 1] > m[i])) {
                throw std::runtime_error(stdLogger.fatal("MDWF Hasenbusch ladder must increase strictly from mf to "
                                                         "pv_mass; mass ", i + 1, " = ", m[i + 1], " after ", m[i]));
            }
        }
        return m;
    }

public:
    MDWFEvenOddHasenbuschTwoFlavorFermionAction(CommunicationBase &commBase, Gauge &gauge,
                                                const MDWFHmcParameters &param,
                                                const std::string &name = "MDWF_hmc_eohb")
        : _prefix(mdwfHmcInstancePrefix(name)),
          _masses(ladder(param)),
          _ipdotPart(commBase, _prefix + "_ipdot_part") {
        for (size_t i = 0; i < _masses.size(); i++) {
            _ops.push_back(std::make_unique<EvenOdd>(gauge, param.M5, _masses[i], param.b5, param.csw,
                                                     _prefix + "_m" + std::to_string(i)));
        }
        if (param.mixed_precision) {
            _mixed = std::make_unique<typename Factor::Mixed>(gauge, param.M5, param.b5, param.csw, param.mixed_delta,
                                                              _prefix + "_mixed");
        }
        for (size_t i = 0; i + 1 < _masses.size(); i++) {
            _factors.push_back(std::make_unique<Factor>(commBase, param, *_ops[i], *_ops[i + 1], _masses[i],
                                                        _masses[i + 1], _prefix + "_f" + std::to_string(i),
                                                        _mixed.get()));
        }
    }

    size_t factorCount() const {
        return _factors.size();
    }

    Factor &factor(size_t i) {
        return *_factors.at(i);
    }

    const std::vector<double> &masses() const {
        return _masses;
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
        double sum = 0.0;
        for (const auto &f : _factors) {
            sum += f->lastDetAction();
        }
        return sum;
    }

    // Force of all factors (a single-time-scale integrator).
    void force(HostGauge &ipdotHost, const HostGauge &gaugeHost) {
        _factors.front()->force(ipdotHost, gaugeHost);
        for (size_t i = 1; i < _factors.size(); i++) {
            _factors[i]->force(_ipdotPart, gaugeHost);
            mdwfHmcAddHostForce<HaloDepth>(ipdotHost, _ipdotPart);
        }
    }

    // Force of factor i alone (a multi-level integrator puts the factors on separate time scales).
    void forceFactor(size_t i, HostGauge &ipdotHost, const HostGauge &gaugeHost) {
        _factors.at(i)->force(ipdotHost, gaugeHost);
    }

    // Force terms for the multi-level integrator (MDWFHmc.h): one per factor, lightest first.
    size_t forceTermCount() const {
        return _factors.size();
    }

    void forceTerm(size_t i, HostGauge &ipdotHost, const HostGauge &gaugeHost) {
        forceFactor(i, ipdotHost, gaugeHost);
    }

    // The lightest factor's pseudofermion (the driver's phi() accessor).
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
};

template<size_t HaloDepth, size_t Ls>
class MDWFEvenOddOneFlavorRhmcFermionAction {
public:
    using Types = MDWFEvenOddTypes<HaloDepth, Ls>;
    using EvenOdd = typename Types::EvenOdd;
    using NormalOp = typename Types::NormalOp;
    using Adapter = typename Types::Adapter;
    using Spinor = typename Types::SpinorE;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using MultiShiftCG = MDWFCoupledMultiShiftCG<double, Adapter>;
    using Terms = MDWFEvenOddForceTerms<HaloDepth, Ls>;
    using Solutions = std::vector<std::unique_ptr<Spinor>>;

private:
    CommunicationBase &_commBase;
    MDWFHmcParameters _param;
    std::string _prefix;
    MDWFRemezApproximation _quarterS, _halfS, _halfSForce, _quarterPv, _quarterPvForce;
    MDWFRationalCoefficients<double> _heatbathS, _heatbathPv, _actionS, _actionPv, _forceS, _forcePv;
    EvenOdd _eoS;
    EvenOdd _eo1;
    NormalOp _normalS;
    NormalOp _normal1;
    Adapter _adapterS;
    Adapter _adapter1;
    Spinor _phi, _eta, _hbtmp, _chi, _psi;
    Solutions _solS, _solPv;  // force: z_j (B shifts on psi) reuse _solS once the x_i terms are stored
    typename Terms::Buffers _buffers;
    HostGauge _ipdotDet;
    HostGauge _ipdotPart;
    double _noiseNorm2;
    double _lastPseudofermionAction;
    double _lastDetAction;
    int _lastIterations;

    static MDWFHmcParameters validated(const MDWFHmcParameters &param) {
        const MDWFRhmcParameters &r = param.rhmc;
        if (!(param.pv_mass > 0.0) || !(r.ms > 0.0) || !(r.lambda_low_s > 0.0) || !(r.lambda_high_s > r.lambda_low_s)
            || !(r.lambda_low_pv > 0.0) || !(r.lambda_high_pv > r.lambda_low_pv) || !(r.action_error > 0.0)) {
            throw std::runtime_error(stdLogger.fatal("MDWF even/odd one-flavour RHMC action: invalid masses, "
                                                     "intervals, or action_error"));
        }
        return param;
    }

    static MDWFRemezApproximation remez(int pnum, int pden, double low, double high, double error,
                                        const MDWFHmcParameters &param) {
        const int maxOrder = param.rhmc.max_order > 0 ? param.rhmc.max_order : 30;
        const int digits = param.rhmc.digits > 0 ? param.rhmc.digits : 50;
        return mdwfRemezPowerForError(pnum, pden, low, high, error, maxOrder, digits);
    }

    static double forceError(const MDWFHmcParameters &param) {
        return param.rhmc.force_error > 0.0 ? param.rhmc.force_error : param.rhmc.action_error;
    }

    // Solutions take consecutive pool slots after the four scratch fields (pool0 ... pool3).
    void allocate(Solutions &solutions, size_t count, size_t &slot) {
        for (size_t i = 0; i < count; i++) {
            solutions.emplace_back(new Spinor(_commBase, EvenOdd::sharedEvenName("pool" + std::to_string(slot++))));
        }
    }

    void refresh() {
        _eoS.refresh();
        _eo1.refresh();
    }

    // MD force solves use force_precision (<= 0: precision), heatbath and action solves precision.
    double precisionFor(bool forceSolve) const {
        return (forceSolve && _param.force_precision > 0.0) ? _param.force_precision : _param.precision;
    }

    void multishift(Adapter &adapter, Solutions &solutions, const Spinor &in,
                    const MDWFRationalCoefficients<double> &c, const char *what, bool forceSolve = false) {
        if (c.shift.empty()) {
            return;
        }
        std::vector<Spinor *> out;
        for (size_t i = 0; i < c.shift.size(); i++) {
            out.push_back(solutions.at(i).get());
        }
        MultiShiftCG cg(MDWFMultiShiftStrategy::Simultaneous);
        const auto start = std::chrono::steady_clock::now();
        const MDWFCoupledMultiShiftCGResults<double> results
            = cg.invert(adapter, out, in, c.shift, _param.max_iter, precisionFor(forceSolve), true);
        int iterations = 0;
        for (const auto &result : results.shifts) {
            iterations = std::max(iterations, result.iterations);
        }
        _lastIterations = std::max(_lastIterations, iterations);
        rootLogger.info("MDWF even/odd RHMC multishift (", what, "): ", c.shift.size(), " shifts, ", iterations,
                        " iterations, ", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
                        " s");
        if (!results.converged()) {
            throw std::runtime_error(stdLogger.fatal("MDWF even/odd one-flavour RHMC ", what,
                                                     " multishift solve did not converge: iterations = ", iterations));
        }
    }

    void combine(Spinor &out, const Spinor &in, const Solutions &solutions, const MDWFRationalCoefficients<double> &c) {
        out = c.constant * in;
        for (size_t i = 0; i < c.numerator.size(); i++) {
            out.template axpyThisB<64>(c.numerator[i], *solutions[i]);
        }
        mdwfUpdateHalo(out);
    }

    void applyRational(Adapter &adapter, Solutions &solutions, Spinor &out, const Spinor &in,
                       const MDWFRationalCoefficients<double> &c, const char *what, bool forceSolve = false) {
        multishift(adapter, solutions, in, c, what, forceSolve);
        combine(out, in, solutions, c);
    }

public:
    MDWFEvenOddOneFlavorRhmcFermionAction(CommunicationBase &commBase, Gauge &gauge, const MDWFHmcParameters &param,
                                          const std::string &name = "MDWF_hmc_eorhmc1")
        : _commBase(commBase),
          _param(validated(param)),
          _prefix(mdwfHmcInstancePrefix(name)),
          _quarterS(remez(1, 4, param.rhmc.lambda_low_s, param.rhmc.lambda_high_s, param.rhmc.action_error, param)),
          _halfS(remez(1, 2, param.rhmc.lambda_low_s, param.rhmc.lambda_high_s, param.rhmc.action_error, param)),
          _halfSForce(remez(1, 2, param.rhmc.lambda_low_s, param.rhmc.lambda_high_s, forceError(param), param)),
          _quarterPv(remez(1, 4, param.rhmc.lambda_low_pv, param.rhmc.lambda_high_pv, param.rhmc.action_error, param)),
          _quarterPvForce(remez(1, 4, param.rhmc.lambda_low_pv, param.rhmc.lambda_high_pv, forceError(param), param)),
          _heatbathS(mdwfRhmcCoefficients(_quarterS.power)),
          _heatbathPv(mdwfRhmcCoefficients(_quarterPv.inverse)),
          _actionS(mdwfRhmcCoefficients(_halfS.inverse)),
          _actionPv(mdwfRhmcCoefficients(_quarterPv.power)),
          _forceS(mdwfRhmcCoefficients(_halfSForce.inverse)),
          _forcePv(mdwfRhmcCoefficients(_quarterPvForce.power)),
          _eoS(gauge, param.M5, param.rhmc.ms, param.b5, param.csw, _prefix + "_eos"),
          _eo1(gauge, param.M5, param.pv_mass, param.b5, param.csw, _prefix + "_eo1"),
          _normalS(_eoS, commBase, _prefix + "_nrms"),
          _normal1(_eo1, commBase, _prefix + "_nrm1"),
          _adapterS(_normalS),
          _adapter1(_normal1),
          _phi(commBase, _prefix + "_phi"),
          // Scratch and the per-pole solutions below: the shared pool (pool0 ...), also used by the light factors.
          _eta(commBase, EvenOdd::sharedEvenName("pool0")),
          _hbtmp(commBase, EvenOdd::sharedEvenName("pool1")),
          _chi(commBase, EvenOdd::sharedEvenName("pool2")),
          _psi(commBase, EvenOdd::sharedEvenName("pool3")),
          _buffers(commBase, _prefix + "_fb"),
          _ipdotDet(commBase, _prefix + "_ipdet"),
          _ipdotPart(commBase, _prefix + "_ipdot_part"),
          _noiseNorm2(0.0),
          _lastPseudofermionAction(0.0),
          _lastDetAction(0.0),
          _lastIterations(0) {
        size_t slot = 4;
        allocate(_solS, std::max({_heatbathS.shift.size(), _actionS.shift.size(), _forceS.shift.size(),
                                  _forcePv.shift.size()}), slot);
        allocate(_solPv, std::max({_heatbathPv.shift.size(), _actionPv.shift.size(), _forcePv.shift.size()}), slot);
    }

    void heatbath(uint4 *randState) {
        refresh();
        _lastIterations = 0;
        _eta.gauss(randState);
        mdwfUpdateHalo(_eta);
        _noiseNorm2 = _adapterS.norm2(_eta);
        applyRational(_adapterS, _solS, _hbtmp, _eta, _heatbathS, "heatbath (A^(1/4))");
        applyRational(_adapter1, _solPv, _phi, _hbtmp, _heatbathPv, "heatbath (B^(-1/4))");
    }

    double noiseNorm2() const {
        return _noiseNorm2;
    }

    double action() {
        refresh();
        _lastIterations = 0;
        applyRational(_adapter1, _solPv, _chi, _phi, _actionPv, "action (B^(1/4))");
        applyRational(_adapterS, _solS, _psi, _chi, _actionS, "action (A^(-1/2))");
        _lastPseudofermionAction = real<double>(_adapterS.dotProduct5D(_chi, _psi));
        _lastDetAction = (_param.csw != 0.0) ? _eoS.logDetW() - _eo1.logDetW() : 0.0;
        return _lastPseudofermionAction + _lastDetAction;
    }

    double lastPseudofermionAction() const {
        return _lastPseudofermionAction;
    }

    double lastDetAction() const {
        return _lastDetAction;
    }

    void force(HostGauge &ipdotHost, const HostGauge &gaugeHost) {
        refresh();
        _lastIterations = 0;
        applyRational(_adapter1, _solPv, _chi, _phi, _forcePv, "force (B^(1/4) phi)", true);   // y_j = _solPv[j]
        applyRational(_adapterS, _solS, _psi, _chi, _forceS, "force (A^(-1/2) chi)", true);    // x_i = _solS[i]

        // Strange-mass terms first; then z_j overwrite the x_i in _solS (one set of pole vectors less in memory).
        {
            Terms terms(_buffers);
            std::vector<double> numerators;
            for (size_t i = 0; i < _forceS.shift.size(); i++) {
                terms.add(*_solS[i], *_solS[i], _eoS, true);    // -2 a_i Re[(Mhat_s x_i)^+ dMhat_s x_i]
                numerators.push_back(_forceS.numerator[i]);
            }
            const MDWFRationalCoefficients<double> coefficients{0.0, numerators,
                                                                std::vector<double>(numerators.size(), 0.0)};
            mdwfHmcStoreFermionForce<HaloDepth, Ls>(ipdotHost, gaugeHost, terms, coefficients, _param.csw, _commBase,
                                                    _prefix + "_fstore");
        }
        multishift(_adapter1, _solS, _psi, _forcePv, "force (B shifts on psi)", true);       // z_j = _solS[j]
        {
            Terms terms(_buffers);
            std::vector<double> numerators;
            for (size_t j = 0; j < _forcePv.shift.size(); j++) {
                terms.add(*_solPv[j], *_solS[j], _eo1, true);   // -2 b_j Re[(Mhat_1 z_j)^+ dMhat_1 y_j]
                numerators.push_back(_forcePv.numerator[j]);
                terms.add(*_solS[j], *_solPv[j], _eo1, true);   // -2 b_j Re[(Mhat_1 y_j)^+ dMhat_1 z_j]
                numerators.push_back(_forcePv.numerator[j]);
            }
            const MDWFRationalCoefficients<double> coefficients{0.0, numerators,
                                                                std::vector<double>(numerators.size(), 0.0)};
            mdwfHmcStoreFermionForce<HaloDepth, Ls>(_ipdotPart, gaugeHost, terms, coefficients, _param.csw, _commBase,
                                                    _prefix + "_fstore");
            mdwfHmcAddHostForce<HaloDepth>(ipdotHost, _ipdotPart);
        }
        if (_param.csw != 0.0) {
            mdwfEvenOddStoreOddDetRatioForce<HaloDepth, Ls>(_ipdotDet, gaugeHost, _eoS, _eo1, 1.0, _param.csw,
                                                            _commBase, _prefix + "_det");
            mdwfHmcAddHostForce<HaloDepth>(ipdotHost, _ipdotDet);
        }
    }

    Spinor &phi() {
        return _phi;
    }

    int lastIterations() const {
        return _lastIterations;
    }

    const MDWFRemezApproximation &quarterS() const { return _quarterS; }
    const MDWFRemezApproximation &halfS() const { return _halfS; }
    const MDWFRemezApproximation &halfSForce() const { return _halfSForce; }
    const MDWFRemezApproximation &quarterPv() const { return _quarterPv; }
    const MDWFRemezApproximation &quarterPvForce() const { return _quarterPvForce; }
};

template<size_t HaloDepth, size_t Ls>
using MDWFEvenOddTwoPlusOneFermionAction =
    MDWFSumFermionAction<HaloDepth, MDWFEvenOddPauliVillarsTwoFlavorFermionAction<HaloDepth, Ls>,
                         MDWFEvenOddOneFlavorRhmcFermionAction<HaloDepth, Ls>>;

template<size_t HaloDepth, size_t Ls>
using MDWFEvenOddTwoPlusOneHmc = MDWFHmcDriver<HaloDepth, Ls, MDWFEvenOddTwoPlusOneFermionAction<HaloDepth, Ls>>;

// 2+1 with the light pair as a Hasenbusch ladder (MDWFHmcParameters::hasenbusch_masses).
template<size_t HaloDepth, size_t Ls>
using MDWFEvenOddHasenbuschTwoPlusOneFermionAction =
    MDWFSumFermionAction<HaloDepth, MDWFEvenOddHasenbuschTwoFlavorFermionAction<HaloDepth, Ls>,
                         MDWFEvenOddOneFlavorRhmcFermionAction<HaloDepth, Ls>>;

template<size_t HaloDepth, size_t Ls>
using MDWFEvenOddHasenbuschTwoPlusOneHmc =
    MDWFHmcDriver<HaloDepth, Ls, MDWFEvenOddHasenbuschTwoPlusOneFermionAction<HaloDepth, Ls>>;

// The light Hasenbusch ladder alone (two flavours), for tests.
template<size_t HaloDepth, size_t Ls>
using MDWFEvenOddHasenbuschTwoFlavorHmc =
    MDWFHmcDriver<HaloDepth, Ls, MDWFEvenOddHasenbuschTwoFlavorFermionAction<HaloDepth, Ls>>;

template<size_t HaloDepth, size_t Ls>
using MDWFEvenOddPauliVillarsTwoFlavorHmc =
    MDWFHmcDriver<HaloDepth, Ls, MDWFEvenOddPauliVillarsTwoFlavorFermionAction<HaloDepth, Ls>>;
