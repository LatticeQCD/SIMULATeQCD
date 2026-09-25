/*
 * Test-only MDWF pseudofermion actions for the single-rank HMC driver
 * (MDWFHmc.h). Each action provides heatbath(rand), action(), and
 * force(ipdotHost, gaugeHost); force() overwrites the host destination with
 * ipdot_l = K_l = TA(B_l), dS(H) = Re tr(H B_l) for U -> exp(eps H) U
 * (MDWF_HMC_CONVENTIONS.md).
 *
 * All operators are the Mobius clover operator M(m) = D_W Din(m) + Shift(m);
 * only Din and Shift depend on the boundary mass m, so dM(m) = dD_W Din(m).
 * Forces are evaluated through the existing single-rank all-link storage.
 *
 * MDWFBareTwoFlavorFermionAction:
 *   S = phi^\dagger (M^\dagger M)^{-1} phi,  phi = M^\dagger eta.  Samples det(M^\dagger M)
 *   of the bare 5D operator (no Pauli-Villars factor).
 *
 * MDWFPauliVillarsTwoFlavorFermionAction, with M_f = M(mf), M_1 = M(pv_mass):
 *   S = phi^\dagger M_1 (M_f^\dagger M_f)^{-1} M_1^\dagger phi,
 *   heatbath phi = M_1 (M_1^\dagger M_1)^{-1} M_f^\dagger eta   (so S = eta^\dagger eta),
 *   which samples det(M_f^\dagger M_f) / det(M_1^\dagger M_1).
 *   With psi = M_1^\dagger phi and chi = (M_f^\dagger M_f)^{-1} psi,
 *   dS = 2 Re[phi^\dagger dD_W (Din_1 chi)] - 2 Re[(M_f chi)^\dagger dD_W (Din_f chi)],
 *   i.e. two storage terms (right = Din_f chi, left = M_f chi, numerator +1) and
 *   (right = Din_1 chi, left = phi, numerator -1), since the storage weight is
 *   -2 * numerator. At mf = pv_mass the two terms cancel and the force vanishes.
 *
 * Neither class defines a production pseudofermion or determinant convention
 * beyond what is written here; no Hasenbusch splitting, no RHMC, single rank.
 */

#pragma once

#include "MDWFAllLinkDirectionIndependentStorage.h"
#include "MDWFCoupledCG.h"
#include "MDWFFermionForceWorkspace.h"
#include "MDWFMobiusForceWorkspace.h"
#include "MDWFMobiusMapping.h"
#include "MDWFNormalOperator.h"
#include "MDWFPseudofermionAction.h"
#include "MDWFRationalCoefficientAdapter.h"

#include <stdexcept>
#include <string>
#include <vector>

struct MDWFHmcParameters {
    double beta;
    double M5;
    double mf;
    double b5;
    double csw;
    double tau;
    int steps;
    int max_iter;
    double precision;
    double pv_mass;   // used only by the Pauli-Villars action
};

/*
 * Unique per-instance field-name prefix.
 *
 * SIMULATeQCD's MemoryManagement::getSmartName (src/base/memoryManagement.h)
 * does not give coexisting fields with the same name separate memory: when a
 * requested name such as "X_phi" already exists as "X_phi_0", it strips the
 * last "_"-segment of the untagged request and returns "X_1", so every field
 * of a second object with names "X_phi", "X_eta", ... aliases one buffer.
 * Every object that owns fields therefore uses a unique prefix, and no field
 * name may contain another field name of the same object as a substring.
 */
inline std::string mdwfHmcInstancePrefix(const std::string &base) {
    static unsigned long counter = 0;
    return base + "_i" + std::to_string(counter++);
}

inline MDWFRationalCoefficients<double> mdwfHmcInverseCoefficients(MDWFRationalCoefficientRole role,
                                                                   const std::string &name) {
    MDWFExplicitRationalInput<double> input{name, role, 0.0, {1.0}, {0.0}};
    return makeMDWFRationalCoefficients(input);
}

// Force terms (right vector chi(i), left vector eta(i)) in the shape the all-link storage reads.
template<class Spinor>
class MDWFExplicitForceTerms {
    std::vector<const Spinor *> _chi;
    std::vector<const Spinor *> _eta;

public:
    void add(const Spinor &chi, const Spinor &eta) {
        _chi.push_back(&chi);
        _eta.push_back(&eta);
    }

    size_t size() const {
        return _chi.size();
    }

    const Spinor &chi(size_t term) const {
        return *_chi.at(term);
    }

    const Spinor &eta(size_t term) const {
        return *_eta.at(term);
    }
};

template<size_t HaloDepth, size_t Ls, class Terms>
void mdwfHmcStoreFermionForce(Gaugefield<double, false, HaloDepth, R18> &ipdotHost,
                              const Gaugefield<double, false, HaloDepth, R18> &gaugeHost,
                              const Terms &terms,
                              const MDWFRationalCoefficients<double> &coefficients,
                              double csw,
                              CommunicationBase &commBase,
                              const std::string &name) {
    if (csw != 0.0) {
        overwriteMDWFCloverAllLinkDirectionIndependentStorageNonzero<HaloDepth, Ls>(
            ipdotHost, gaugeHost, terms, coefficients, csw, commBase, name);
    } else {
        overwriteMDWFWilsonAllLinkDirectionIndependentStorageCsw0<HaloDepth, Ls>(
            ipdotHost, gaugeHost, terms, coefficients, commBase, name);
    }
}

template<size_t HaloDepth, size_t Ls>
class MDWFBareTwoFlavorFermionAction {
public:
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;
    using Forward = MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Adjoint = MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Normal = MDWFNormalOperator<Forward, Adjoint>;
    using NormalAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, Normal>;
    using Workspace = MDWFFermionForceWorkspace<double, HaloDepth, HaloDepth, Ls, Normal, Forward>;
    using View = MDWFMobiusForceWorkspaceView<double, HaloDepth, Ls, Workspace>;

private:
    CommunicationBase &_commBase;
    MDWFHmcParameters _param;
    std::string _prefix;
    Spinor _phi;
    Spinor _eta;
    Spinor _actionWorkspace;
    // The operators hold a reference to the gauge field and recompute the clover term on
    // every application, so they stay valid while the gauge field evolves in place.
    Forward _forward;
    Adjoint _adjoint;
    Normal _normal;
    NormalAdapter _normalAdapter;
    MDWFRationalCoefficients<double> _actionCoefficients;
    MDWFRationalCoefficients<double> _forceCoefficients;
    double _noiseNorm2;

public:
    MDWFBareTwoFlavorFermionAction(CommunicationBase &commBase, Gauge &gauge, const MDWFHmcParameters &param,
                                   const std::string &name = "MDWF_hmc_bare")
        : _commBase(commBase),
          _param(param),
          _prefix(mdwfHmcInstancePrefix(name)),
          _phi(commBase, _prefix + "_phi"),
          _eta(commBase, _prefix + "_eta"),
          _actionWorkspace(commBase, _prefix + "_action_workspace"),
          _forward(gauge, param.M5, param.mf, param.b5, param.csw, _prefix + "_forward"),
          _adjoint(gauge, param.M5, param.mf, param.b5, param.csw, _prefix + "_adjoint"),
          _normal(commBase, _forward, _adjoint, _prefix + "_normal"),
          _normalAdapter(_normal),
          _actionCoefficients(mdwfHmcInverseCoefficients(MDWFRationalCoefficientRole::Action, _prefix + "_s")),
          _forceCoefficients(mdwfHmcInverseCoefficients(MDWFRationalCoefficientRole::Force, _prefix + "_f")),
          _noiseNorm2(0.0) {}

    // phi = M^\dagger eta with eta distributed as exp(-eta^\dagger eta); then S = eta^\dagger eta.
    void heatbath(uint4 *randState) {
        _eta.gauss(randState);
        _eta.updateAll();
        _noiseNorm2 = _normalAdapter.norm2(_eta);
        _adjoint.apply(_phi, _eta, true);
    }

    double noiseNorm2() const {
        return _noiseNorm2;
    }

    double action() {
        const MDWFRationalActionResult<double> result = computeMDWFRationalAction<double, NormalAdapter>(
            _normalAdapter, _actionWorkspace, _phi, _actionCoefficients, _param.max_iter, _param.precision,
            _prefix + "_act");
        if (!result.rational_result.converged()) {
            throw std::runtime_error(stdLogger.fatal("MDWF bare two-flavour action solve did not converge"));
        }
        return result.action_real;
    }

    void force(HostGauge &ipdotHost, const HostGauge &gaugeHost) {
        Workspace workspace;
        workspace.prepare(_normal, _forward, _phi, _forceCoefficients, _param.max_iter, _param.precision,
                          _prefix + "_fws");
        if (!workspace.converged()) {
            throw std::runtime_error(stdLogger.fatal("MDWF bare two-flavour force solve did not converge"));
        }
        View view(workspace, _forward.params().dinCoeff, _prefix + "_fview");
        mdwfHmcStoreFermionForce<HaloDepth, Ls>(ipdotHost, gaugeHost, view, _forceCoefficients, _param.csw,
                                                _commBase, _prefix + "_fstore");
    }

    Spinor &phi() {
        return _phi;
    }
};

template<size_t HaloDepth, size_t Ls>
class MDWFPauliVillarsTwoFlavorFermionAction {
public:
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;
    using Forward = MDWFMobiusCloverLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Adjoint = MDWFMobiusCloverAdjointLinearOperator<double, HaloDepth, HaloDepth, Ls>;
    using Normal = MDWFNormalOperator<Forward, Adjoint>;
    using NormalAdapter = MDWFCoupledSolverAdapter<double, HaloDepth, HaloDepth, Ls, Normal>;
    using CG = MDWFCoupledCG<double, NormalAdapter>;

private:
    CommunicationBase &_commBase;
    MDWFHmcParameters _param;
    std::string _prefix;
    Spinor _phi;
    Spinor _eta;
    Spinor _tmp;
    Spinor _y;
    Spinor _psi;
    Spinor _chi;
    Spinor _etaF;
    Spinor _dinChiF;
    Spinor _dinChi1;
    Spinor _actionWorkspace;
    Forward _forwardF;
    Adjoint _adjointF;
    Normal _normalF;
    NormalAdapter _adapterF;
    Forward _forward1;
    Adjoint _adjoint1;
    Normal _normal1;
    NormalAdapter _adapter1;
    MDWFRationalCoefficients<double> _actionCoefficients;
    MDWFRationalCoefficients<double> _forceCoefficients;
    double _noiseNorm2;

    static MDWFRationalCoefficients<double> makeForceCoefficients(const std::string &name) {
        MDWFExplicitRationalInput<double> input{
            name, MDWFRationalCoefficientRole::Force, 0.0, {1.0, -1.0}, {0.0, 0.0}};
        return makeMDWFRationalCoefficients(input);
    }

    void solve(NormalAdapter &adapter, Spinor &out, const Spinor &in, const char *what) {
        CG cg;
        const MDWFCoupledCGResult<double> result = cg.invert(adapter, out, in, _param.max_iter, _param.precision, true);
        if (!result.converged) {
            throw std::runtime_error(stdLogger.fatal("MDWF Pauli-Villars ", what, " solve did not converge: iterations = ",
                                                     result.iterations, ", residue = ", result.residue));
        }
    }

public:
    MDWFPauliVillarsTwoFlavorFermionAction(CommunicationBase &commBase, Gauge &gauge, const MDWFHmcParameters &param,
                                           const std::string &name = "MDWF_hmc_pv")
        : _commBase(commBase),
          _param(param),
          _prefix(mdwfHmcInstancePrefix(name)),
          _phi(commBase, _prefix + "_phi"),
          _eta(commBase, _prefix + "_eta"),
          _tmp(commBase, _prefix + "_tmp"),
          _y(commBase, _prefix + "_y"),
          _psi(commBase, _prefix + "_psi"),
          _chi(commBase, _prefix + "_chi"),
          _etaF(commBase, _prefix + "_mf_chi"),
          _dinChiF(commBase, _prefix + "_din_chi_f"),
          _dinChi1(commBase, _prefix + "_din_chi_1"),
          _actionWorkspace(commBase, _prefix + "_action_workspace"),
          _forwardF(gauge, param.M5, param.mf, param.b5, param.csw, _prefix + "_forward_f"),
          _adjointF(gauge, param.M5, param.mf, param.b5, param.csw, _prefix + "_adjoint_f"),
          _normalF(commBase, _forwardF, _adjointF, _prefix + "_normal_f"),
          _adapterF(_normalF),
          _forward1(gauge, param.M5, param.pv_mass, param.b5, param.csw, _prefix + "_forward_1"),
          _adjoint1(gauge, param.M5, param.pv_mass, param.b5, param.csw, _prefix + "_adjoint_1"),
          _normal1(commBase, _forward1, _adjoint1, _prefix + "_normal_1"),
          _adapter1(_normal1),
          _actionCoefficients(mdwfHmcInverseCoefficients(MDWFRationalCoefficientRole::Action, _prefix + "_s")),
          _forceCoefficients(makeForceCoefficients(_prefix + "_f")),
          _noiseNorm2(0.0) {
        if (!(param.pv_mass > 0.0)) {
            throw std::runtime_error(stdLogger.fatal("MDWF Pauli-Villars action requires pv_mass > 0, got ",
                                                     param.pv_mass));
        }
    }

    // phi = M_1 (M_1^\dagger M_1)^{-1} M_f^\dagger eta = (M_1^\dagger)^{-1} M_f^\dagger eta; then S = eta^\dagger eta.
    void heatbath(uint4 *randState) {
        _eta.gauss(randState);
        _eta.updateAll();
        _noiseNorm2 = _adapterF.norm2(_eta);
        _adjointF.apply(_tmp, _eta, true);
        solve(_adapter1, _y, _tmp, "heatbath");
        _forward1.apply(_phi, _y, true);
    }

    double noiseNorm2() const {
        return _noiseNorm2;
    }

    double action() {
        _adjoint1.apply(_psi, _phi, true);
        const MDWFRationalActionResult<double> result = computeMDWFRationalAction<double, NormalAdapter>(
            _adapterF, _actionWorkspace, _psi, _actionCoefficients, _param.max_iter, _param.precision,
            _prefix + "_act");
        if (!result.rational_result.converged()) {
            throw std::runtime_error(stdLogger.fatal("MDWF Pauli-Villars action solve did not converge"));
        }
        return result.action_real;
    }

    void force(HostGauge &ipdotHost, const HostGauge &gaugeHost) {
        _adjoint1.apply(_psi, _phi, true);
        solve(_adapterF, _chi, _psi, "force");
        _forwardF.apply(_etaF, _chi, true);
        applyMDWFFifthDimCoupling<double, true, All, HaloDepth, Ls>(_dinChiF, _chi, _forwardF.params().dinCoeff, true);
        applyMDWFFifthDimCoupling<double, true, All, HaloDepth, Ls>(_dinChi1, _chi, _forward1.params().dinCoeff, true);

        MDWFExplicitForceTerms<Spinor> terms;
        terms.add(_dinChiF, _etaF);
        terms.add(_dinChi1, _phi);
        mdwfHmcStoreFermionForce<HaloDepth, Ls>(ipdotHost, gaugeHost, terms, _forceCoefficients, _param.csw,
                                                _commBase, _prefix + "_fstore");
    }

    Spinor &phi() {
        return _phi;
    }
};
