/*
 * Test-only MDWF one-flavour Pauli-Villars RHMC action and a sum action for
 * 2+1 flavours, for the single-rank HMC driver (MDWFHmc.h). Same interface and
 * force convention as MDWFHmcFermionActions.h: force() overwrites the host
 * destination with ipdot_l = K_l = TA(B_l), dS(H) = Re tr(H B_l) for
 * U -> exp(eps H) U.
 *
 * MDWFOneFlavorRhmcFermionAction, with M_s = M(rhmc.ms), M_1 = M(pv_mass),
 * A = M_s^\dagger M_s, B = M_1^\dagger M_1, samples det(A)^(1/2) / det(B)^(1/2):
 *
 *   S = phi^\dagger r_B^(1/4) r_A^(-1/2) r_B^(1/4) phi,
 *   heatbath phi = r_B^(-1/4) r_A^(1/4) eta   (S = eta^\dagger eta up to the approximation error),
 *
 * with r_X^(p) the AlgRemez approximation of X^p on the interval given in
 * MDWFRhmcParameters, of the lowest order meeting the target error:
 *   A^(1/4)        power   of x^(1/4) on the M_s interval (heatbath),
 *   A^(-1/2)       inverse of x^(1/2) on the M_s interval (action; force with force_error),
 *   B^(+-1/4)      power / inverse of x^(1/4) on the Pauli-Villars interval
 *                  (+1/4: action, force with force_error; -1/4: heatbath).
 * The B^(-1/4) heatbath approximation is the exact reciprocal of the B^(1/4)
 * action approximation (both from one AlgRemez run).
 *
 * Force, with r_B^(1/4) = c_B + sum_j b_j (B + tau_j)^(-1), r_A^(-1/2) = c_A + sum_i a_i (A + sigma_i)^(-1):
 *   y_j = (B + tau_j)^(-1) phi,   chi = r_B^(1/4) phi,
 *   x_i = (A + sigma_i)^(-1) chi, psi = r_A^(-1/2) chi,
 *   z_j = (B + tau_j)^(-1) psi,
 *   dS = sum_i -2 a_i Re[(M_s x_i)^\dagger dM_s x_i]
 *      + sum_j -2 b_j (Re[(M_1 z_j)^\dagger dM_1 y_j] + Re[(M_1 y_j)^\dagger dM_1 z_j]),
 * three simultaneous multishift solves. With dM = dD_W Din(m), each term is a
 * storage term (right = Din v, left = M u, numerator n), since the storage
 * weight is -2 * numerator (as for the Pauli-Villars two-flavour action).
 * The Din v and M u vectors are formed one term at a time as the storage reads
 * them (MDWFLazyForceTerms), so only the solutions x_i, y_j, z_j are held.
 *
 * MDWFSumFermionAction<First, Second> adds two actions (independent
 * pseudofermions, summed actions and forces); MDWFTwoPlusOneFermionAction is
 * the Pauli-Villars two-flavour action at mf plus the one-flavour action at
 * rhmc.ms, i.e. det(M_l^\dagger M_l / M_1^\dagger M_1) det(M_s^\dagger M_s / M_1^\dagger M_1)^(1/2).
 *
 * The AlgRemez code is host-only: targets using this header must link the
 * mdwfRemez library (GMP/MPFR), see CMakeLists.txt.
 */

#pragma once

#include "MDWFHmc.h"
#include "MDWFRemez.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

inline MDWFRationalCoefficients<double> mdwfRhmcCoefficients(const MDWFRemezPartialFractions &pf) {
    MDWFRationalCoefficients<double> coefficients{pf.constant, pf.residues, pf.poles};
    coefficients.validate();
    return coefficients;
}

/*
 * Force terms (right = Din(m_r) r, left = M(m_l) l, or left = l when no left
 * operator is given) in the shape the all-link storage reads. The two vectors
 * of a term are computed into caller-owned buffers when chi(term)/eta(term) is
 * requested; the buffers remember the term they hold, so any access order is
 * correct and the storage's sequential access costs one Din and one M per term.
 */
template<size_t HaloDepth, size_t Ls, class Forward>
class MDWFLazyForceTerms {
public:
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;

private:
    struct Term {
        const Spinor *right;
        Forward *rightOperator;
        const Spinor *left;
        Forward *leftOperator;
    };
    static constexpr size_t none = std::numeric_limits<size_t>::max();

    std::vector<Term> _terms;
    Spinor *_chiBuffer;
    Spinor *_etaBuffer;
    mutable size_t _chiTerm;
    mutable size_t _etaTerm;

public:
    MDWFLazyForceTerms(Spinor &chiBuffer, Spinor &etaBuffer)
        : _chiBuffer(&chiBuffer), _etaBuffer(&etaBuffer), _chiTerm(none), _etaTerm(none) {}

    void add(const Spinor &right, Forward &rightOperator, const Spinor &left, Forward *leftOperator) {
        _terms.push_back({&right, &rightOperator, &left, leftOperator});
    }

    size_t size() const {
        return _terms.size();
    }

    const Spinor &chi(size_t term) const {
        if (_chiTerm != term) {
            const Term &t = _terms.at(term);
            applyMDWFFifthDimCoupling<double, true, All, HaloDepth, Ls>(*_chiBuffer, *t.right,
                                                                       t.rightOperator->params().dinCoeff, true);
            _chiTerm = term;
        }
        return *_chiBuffer;
    }

    const Spinor &eta(size_t term) const {
        const Term &t = _terms.at(term);
        if (t.leftOperator == nullptr) {
            return *t.left;
        }
        if (_etaTerm != term) {
            t.leftOperator->apply(*_etaBuffer, *t.left, true);
            _etaTerm = term;
        }
        return *_etaBuffer;
    }
};

template<size_t HaloDepth, size_t Ls>
class MDWFOneFlavorRhmcFermionAction {
public:
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;
    using Forward = MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Adjoint = MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Normal = MDWFNormalOperator<Forward, Adjoint>;
    using NormalAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, Normal>;
    using MultiShiftCG = MDWFCoupledMultiShiftCG<double, NormalAdapter>;
    using Terms = MDWFLazyForceTerms<HaloDepth, Ls, Forward>;
    using Solutions = std::vector<std::unique_ptr<Spinor>>;

private:
    CommunicationBase &_commBase;
    MDWFHmcParameters _param;
    std::string _prefix;
    MDWFRemezApproximation _quarterS;
    MDWFRemezApproximation _halfS;
    MDWFRemezApproximation _halfSForce;
    MDWFRemezApproximation _quarterPv;
    MDWFRemezApproximation _quarterPvForce;
    MDWFRationalCoefficients<double> _heatbathS;   // A^(1/4)
    MDWFRationalCoefficients<double> _heatbathPv;  // B^(-1/4)
    MDWFRationalCoefficients<double> _actionS;     // A^(-1/2)
    MDWFRationalCoefficients<double> _actionPv;    // B^(1/4)
    MDWFRationalCoefficients<double> _forceS;      // A^(-1/2)
    MDWFRationalCoefficients<double> _forcePv;     // B^(1/4)
    Spinor _phi;
    Spinor _eta;
    Spinor _hbtmp;
    Spinor _chi;
    Spinor _psi;
    Spinor _termChi;
    Spinor _termEta;
    Solutions _solS;
    Solutions _solPv;
    Solutions _solPvZ;
    Forward _forwardS;
    Adjoint _adjointS;
    Normal _normalS;
    NormalAdapter _adapterS;
    Forward _forward1;
    Adjoint _adjoint1;
    Normal _normal1;
    NormalAdapter _adapter1;
    double _noiseNorm2;
    int _lastIterations;

    static MDWFHmcParameters validated(const MDWFHmcParameters &param) {
        const MDWFRhmcParameters &r = param.rhmc;
        if (!(param.pv_mass > 0.0) || !(r.ms > 0.0)) {
            throw std::runtime_error(stdLogger.fatal("MDWF one-flavour RHMC action requires pv_mass > 0 and ms > 0, got ",
                                                     param.pv_mass, ", ", r.ms));
        }
        if (!(r.lambda_low_s > 0.0) || !(r.lambda_high_s > r.lambda_low_s)
            || !(r.lambda_low_pv > 0.0) || !(r.lambda_high_pv > r.lambda_low_pv)) {
            throw std::runtime_error(stdLogger.fatal("MDWF one-flavour RHMC action requires 0 < lambda_low < lambda_high "
                                                     "for both approximation intervals"));
        }
        if (!(r.action_error > 0.0)) {
            throw std::runtime_error(stdLogger.fatal("MDWF one-flavour RHMC action requires action_error > 0"));
        }
        return param;
    }

    static int maxOrder(const MDWFHmcParameters &param) {
        return param.rhmc.max_order > 0 ? param.rhmc.max_order : 30;
    }

    static int digits(const MDWFHmcParameters &param) {
        return param.rhmc.digits > 0 ? param.rhmc.digits : 50;
    }

    static double forceError(const MDWFHmcParameters &param) {
        return param.rhmc.force_error > 0.0 ? param.rhmc.force_error : param.rhmc.action_error;
    }

    static MDWFRemezApproximation remez(int pnum, int pden, double low, double high, double error,
                                        const MDWFHmcParameters &param) {
        return mdwfRemezPowerForError(pnum, pden, low, high, error, maxOrder(param), digits(param));
    }

    static size_t maxPoles(const MDWFRationalCoefficients<double> &a, const MDWFRationalCoefficients<double> &b,
                           const MDWFRationalCoefficients<double> &c) {
        return std::max(a.shift.size(), std::max(b.shift.size(), c.shift.size()));
    }

    void allocate(Solutions &solutions, size_t count, const std::string &stem) {
        for (size_t i = 0; i < count; i++) {
            solutions.emplace_back(new Spinor(_commBase, _prefix + stem + std::to_string(i) + "v"));
        }
    }

    // Solutions (X + shift_i)^(-1) in for the shifts of c, simultaneous multishift CG.
    void multishift(NormalAdapter &adapter, Solutions &solutions, const Spinor &in,
                    const MDWFRationalCoefficients<double> &c, const char *what) {
        if (c.shift.empty()) {
            return;
        }
        std::vector<Spinor *> out;
        for (size_t i = 0; i < c.shift.size(); i++) {
            out.push_back(solutions.at(i).get());
        }
        MultiShiftCG cg(MDWFMultiShiftStrategy::Simultaneous);
        const MDWFCoupledMultiShiftCGResults<double> results
            = cg.invert(adapter, out, in, c.shift, _param.max_iter, _param.precision, true);
        int iterations = 0;
        for (const auto &result : results.shifts) {
            iterations = std::max(iterations, result.iterations);
        }
        _lastIterations = std::max(_lastIterations, iterations);
        if (!results.converged()) {
            throw std::runtime_error(stdLogger.fatal("MDWF one-flavour RHMC ", what,
                                                     " multishift solve did not converge: iterations = ", iterations));
        }
    }

    // out = c.constant in + sum_i c.numerator_i solutions_i, with the solutions of multishift(..., in, c, ...).
    void combine(Spinor &out, const Spinor &in, const Solutions &solutions, const MDWFRationalCoefficients<double> &c) {
        out = c.constant * in;
        for (size_t i = 0; i < c.numerator.size(); i++) {
            out.template axpyThisB<64>(c.numerator[i], *solutions[i]);
        }
        out.updateAll();
    }

    void applyRational(NormalAdapter &adapter, Solutions &solutions, Spinor &out, const Spinor &in,
                       const MDWFRationalCoefficients<double> &c, const char *what) {
        multishift(adapter, solutions, in, c, what);
        combine(out, in, solutions, c);
    }

public:
    MDWFOneFlavorRhmcFermionAction(CommunicationBase &commBase, Gauge &gauge, const MDWFHmcParameters &param,
                                   const std::string &name = "MDWF_hmc_rhmc1")
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
          _phi(commBase, _prefix + "_phi"),
          _eta(commBase, _prefix + "_eta"),
          _hbtmp(commBase, _prefix + "_hbtmp"),
          _chi(commBase, _prefix + "_chi"),
          _psi(commBase, _prefix + "_psi"),
          _termChi(commBase, _prefix + "_term_chi"),
          _termEta(commBase, _prefix + "_term_eta"),
          _forwardS(gauge, param.M5, param.rhmc.ms, param.b5, param.csw, _prefix + "_forward_s"),
          _adjointS(gauge, param.M5, param.rhmc.ms, param.b5, param.csw, _prefix + "_adjoint_s"),
          _normalS(commBase, _forwardS, _adjointS, _prefix + "_normal_s"),
          _adapterS(_normalS),
          _forward1(gauge, param.M5, param.pv_mass, param.b5, param.csw, _prefix + "_forward_1"),
          _adjoint1(gauge, param.M5, param.pv_mass, param.b5, param.csw, _prefix + "_adjoint_1"),
          _normal1(commBase, _forward1, _adjoint1, _prefix + "_normal_1"),
          _adapter1(_normal1),
          _noiseNorm2(0.0),
          _lastIterations(0) {
        allocate(_solS, maxPoles(_heatbathS, _actionS, _forceS), "_xs");
        allocate(_solPv, maxPoles(_heatbathPv, _actionPv, _forcePv), "_ypv");
        allocate(_solPvZ, _forcePv.shift.size(), "_zpv");
    }

    // phi = r_B^(-1/4) r_A^(1/4) eta with eta distributed as exp(-eta^\dagger eta); then S = eta^\dagger eta.
    void heatbath(uint4 *randState) {
        _lastIterations = 0;
        _eta.gauss(randState);
        _eta.updateAll();
        _noiseNorm2 = _adapterS.norm2(_eta);
        applyRational(_adapterS, _solS, _hbtmp, _eta, _heatbathS, "heatbath (A^(1/4))");
        applyRational(_adapter1, _solPv, _phi, _hbtmp, _heatbathPv, "heatbath (B^(-1/4))");
    }

    double noiseNorm2() const {
        return _noiseNorm2;
    }

    // S = chi^\dagger r_A^(-1/2) chi with chi = r_B^(1/4) phi.
    double action() {
        _lastIterations = 0;
        applyRational(_adapter1, _solPv, _chi, _phi, _actionPv, "action (B^(1/4))");
        applyRational(_adapterS, _solS, _psi, _chi, _actionS, "action (A^(-1/2))");
        return real<double>(_adapterS.dotProduct5D(_chi, _psi));
    }

    void force(HostGauge &ipdotHost, const HostGauge &gaugeHost) {
        _lastIterations = 0;
        applyRational(_adapter1, _solPv, _chi, _phi, _forcePv, "force (B^(1/4) phi)");     // y_j = _solPv[j]
        applyRational(_adapterS, _solS, _psi, _chi, _forceS, "force (A^(-1/2) chi)");      // x_i = _solS[i]
        multishift(_adapter1, _solPvZ, _psi, _forcePv, "force (B shifts on psi)");          // z_j = _solPvZ[j]

        Terms terms(_termChi, _termEta);
        std::vector<double> numerators;
        for (size_t i = 0; i < _forceS.shift.size(); i++) {
            terms.add(*_solS[i], _forwardS, *_solS[i], &_forwardS);
            numerators.push_back(_forceS.numerator[i]);
        }
        for (size_t j = 0; j < _forcePv.shift.size(); j++) {
            terms.add(*_solPv[j], _forward1, *_solPvZ[j], &_forward1);   // -2 b_j Re[(M_1 z_j)^\dagger dM_1 y_j]
            numerators.push_back(_forcePv.numerator[j]);
            terms.add(*_solPvZ[j], _forward1, *_solPv[j], &_forward1);   // -2 b_j Re[(M_1 y_j)^\dagger dM_1 z_j]
            numerators.push_back(_forcePv.numerator[j]);
        }
        const MDWFRationalCoefficients<double> coefficients{0.0, numerators, std::vector<double>(numerators.size(), 0.0)};
        mdwfHmcStoreFermionForce<HaloDepth, Ls>(ipdotHost, gaugeHost, terms, coefficients, _param.csw, _commBase,
                                                _prefix + "_fstore");
    }

    Spinor &phi() {
        return _phi;
    }

    // Largest multishift iteration count of the last heatbath(), action(), or force().
    int lastIterations() const {
        return _lastIterations;
    }

    // Human-readable description of the five AlgRemez approximations in use.
    std::string describeApproximations() const {
        return mdwfRemezDescribe(_quarterS) + mdwfRemezDescribe(_halfS) + mdwfRemezDescribe(_halfSForce)
               + mdwfRemezDescribe(_quarterPv) + mdwfRemezDescribe(_quarterPvForce);
    }

    const MDWFRationalCoefficients<double> &heatbathS() const { return _heatbathS; }
    const MDWFRationalCoefficients<double> &heatbathPv() const { return _heatbathPv; }
    const MDWFRationalCoefficients<double> &actionS() const { return _actionS; }
    const MDWFRationalCoefficients<double> &actionPv() const { return _actionPv; }
    const MDWFRationalCoefficients<double> &forceS() const { return _forceS; }
    const MDWFRationalCoefficients<double> &forcePv() const { return _forcePv; }
    const MDWFRemezApproximation &quarterS() const { return _quarterS; }
    const MDWFRemezApproximation &halfS() const { return _halfS; }
    const MDWFRemezApproximation &halfSForce() const { return _halfSForce; }
    const MDWFRemezApproximation &quarterPv() const { return _quarterPv; }
    const MDWFRemezApproximation &quarterPvForce() const { return _quarterPvForce; }

    NormalAdapter &adapterS() { return _adapterS; }
    NormalAdapter &adapterPv() { return _adapter1; }

    // out = r(X) in for X = A (useS) or B, through the same simultaneous multishift path as the action.
    void applyApproximation(bool useS, const MDWFRationalCoefficients<double> &c, Spinor &out, const Spinor &in) {
        applyRational(useS ? _adapterS : _adapter1, useS ? _solS : _solPv, out, in, c, "approximation check");
    }
};

template<size_t HaloDepth>
void mdwfHmcAddHostForce(Gaugefield<double, false, HaloDepth, R18> &destination,
                         const Gaugefield<double, false, HaloDepth, R18> &addend) {
    typedef GIndexer<All, HaloDepth> GInd;
    SU3Accessor<double, R18> dAcc = destination.getAccessor();
    const SU3Accessor<double, R18> aAcc = addend.getAccessor();
    for (size_t siteIndex = 0; siteIndex < GInd::getLatData().vol4; siteIndex++) {
        const gSite site = GInd::getSite(siteIndex);
        for (uint8_t mu = 0; mu < 4; mu++) {
            const gSiteMu siteMu = GInd::getSiteMu(site, mu);
            dAcc.setLink(siteMu, dAcc.getLink(siteMu) + aAcc.getLink(siteMu));
        }
    }
}

template<size_t HaloDepth, class First, class Second>
class MDWFSumFermionAction {
public:
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;

private:
    std::string _prefix;
    First _first;
    Second _second;
    HostGauge _ipdotPart;

public:
    MDWFSumFermionAction(CommunicationBase &commBase, Gauge &gauge, const MDWFHmcParameters &param,
                         const std::string &name = "MDWF_hmc_sum")
        : _prefix(mdwfHmcInstancePrefix(name)),
          _first(commBase, gauge, param),
          _second(commBase, gauge, param),
          _ipdotPart(commBase, _prefix + "_ipdot_part") {}

    void heatbath(uint4 *randState) {
        _first.heatbath(randState);
        _second.heatbath(randState);
    }

    double noiseNorm2() const {
        return _first.noiseNorm2() + _second.noiseNorm2();
    }

    double action() {
        return _first.action() + _second.action();
    }

    void force(HostGauge &ipdotHost, const HostGauge &gaugeHost) {
        _first.force(ipdotHost, gaugeHost);
        _second.force(_ipdotPart, gaugeHost);
        mdwfHmcAddHostForce<HaloDepth>(ipdotHost, _ipdotPart);
    }

    // Force terms for the multi-level integrator: the first action's terms, then the second's.
    size_t forceTermCount() {
        return mdwfForceTermCount(_first) + mdwfForceTermCount(_second);
    }

    void forceTerm(size_t i, HostGauge &ipdotHost, const HostGauge &gaugeHost) {
        const size_t n1 = mdwfForceTermCount(_first);
        if (i < n1) {
            mdwfForceTerm(_first, i, ipdotHost, gaugeHost);
        } else {
            mdwfForceTerm(_second, i - n1, ipdotHost, gaugeHost);
        }
    }

    // The first action's pseudofermion (the driver's phi() accessor).
    typename First::Spinor &phi() {
        return _first.phi();
    }

    First &first() {
        return _first;
    }

    Second &second() {
        return _second;
    }
};

template<size_t HaloDepth, size_t Ls>
using MDWFTwoPlusOneFermionAction = MDWFSumFermionAction<HaloDepth, MDWFPauliVillarsTwoFlavorFermionAction<HaloDepth, Ls>,
                                                         MDWFOneFlavorRhmcFermionAction<HaloDepth, Ls>>;

template<size_t HaloDepth, size_t Ls>
using MDWFOneFlavorRhmcHmc = MDWFHmcDriver<HaloDepth, Ls, MDWFOneFlavorRhmcFermionAction<HaloDepth, Ls>>;

template<size_t HaloDepth, size_t Ls>
using MDWFTwoPlusOneHmc = MDWFHmcDriver<HaloDepth, Ls, MDWFTwoPlusOneFermionAction<HaloDepth, Ls>>;
