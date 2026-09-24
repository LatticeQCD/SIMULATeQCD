/*
 * Test-only single-rank MDWF two-flavour HMC driver (step 2 of the MDWF RHMC
 * plan in TODO.md), following MDWF_HMC_CONVENTIONS.md:
 *
 *   H = (1/2) sum_links tr(P P) + S_g + S_f,
 *   U -> exp(i eps P) U,   P -> P - i eps ipdot,   ipdot_l = K_l = TA(B_l),
 *
 * with
 *
 *   S_g = -(beta/3) sum_plaquettes Re tr U_p          (Wilson gauge action),
 *   ipdot_g = -(beta/3) gaugeActionDerivPlaq,          (identity confirmed by
 *                                                       mdwfMobiusHmcConventionTest)
 *   S_f = phi^\dagger (M^\dagger M)^{-1} phi,          heatbath phi = M^\dagger eta,
 *   ipdot_f = stored all-link MDWF matrices K_l        (Mobius clover operator,
 *            from MDWFMobiusForceWorkspaceView)         rational c0 = 0, {1}, {0}).
 *
 * The Symanzik gauge action is deliberately not offered: its rectangle force
 * does not yet match its action (TODO.md). The two MD update formulas
 * reproduce do_evolve_Q / do_evolve_P from src/modules/rhmc/integrator.cpp,
 * which are file-local there; that module is not modified.
 *
 * This is a correctness scaffold: single rank, no Pauli-Villars factor (so
 * it samples det(M^\dagger M) of the bare 5D operator, not physical 2-flavour
 * MDWF), plain leapfrog, the force stored on the host each step, and
 * unpreconditioned solves from a zero initial guess. It is intended for
 * reversibility, Delta H scaling, and <exp(-Delta H)> checks on small
 * lattices, not for production.
 */

#pragma once

#include "MDWFAllLinkDirectionIndependentStorage.h"
#include "MDWFFermionForceWorkspace.h"
#include "MDWFMobiusForceWorkspace.h"
#include "MDWFMobiusMapping.h"
#include "MDWFNormalOperator.h"
#include "MDWFPseudofermionAction.h"
#include "MDWFRationalCoefficientAdapter.h"
#include "../../gauge/gaugeAction.h"
#include "../../gauge/gaugeActionDeriv.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

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
};

struct MDWFHmcEnergy {
    double kinetic;
    double gauge;
    double fermion;

    double total() const {
        return kinetic + gauge + fermion;
    }
};

struct MDWFHmcTrajectoryResult {
    MDWFHmcEnergy before;
    MDWFHmcEnergy after;
    double delta_h;
    bool accepted;
    int force_evaluations;
};

// Same formula as do_evolve_Q in src/modules/rhmc/integrator.cpp.
template<size_t HaloDepth>
struct MDWFHmcEvolveQ {
    SU3Accessor<double, R18> _gAcc;
    SU3Accessor<double> _pAcc;
    double _stepsize;

    MDWFHmcEvolveQ(SU3Accessor<double, R18> gAcc, SU3Accessor<double> pAcc, double stepsize)
        : _gAcc(gAcc), _pAcc(pAcc), _stepsize(stepsize) {}

    __host__ __device__ SU3<double> operator()(gSiteMu site) {
        SU3<double> temp = su3_exp<double>(COMPLEX(double)(0.0, 1.0) * _stepsize * _pAcc.getLink(site))
                           * _gAcc.getLink(site);
        temp.su3unitarize();
        return temp;
    }
};

// Same formula as do_evolve_P in src/modules/rhmc/integrator.cpp.
template<size_t HaloDepth>
struct MDWFHmcEvolveP {
    SU3Accessor<double> _pAcc;
    SU3Accessor<double> _ipdotAcc;
    double _stepsize;

    MDWFHmcEvolveP(SU3Accessor<double> pAcc, SU3Accessor<double> ipdotAcc, double stepsize)
        : _pAcc(pAcc), _ipdotAcc(ipdotAcc), _stepsize(stepsize) {}

    __host__ __device__ SU3<double> operator()(gSiteMu site) {
        SU3<double> temp = _pAcc.getLink(site);
        temp -= COMPLEX(double)(0.0, 1.0) * _stepsize * _ipdotAcc.getLink(site);
        return temp;
    }
};

template<size_t HaloDepth>
struct MDWFHmcWilsonGaugeForce {
    SU3Accessor<double, R18> _gAcc;
    double _beta;

    MDWFHmcWilsonGaugeForce(SU3Accessor<double, R18> gAcc, double beta) : _gAcc(gAcc), _beta(beta) {}

    __host__ __device__ SU3<double> operator()(gSiteMu siteMu) {
        typedef GIndexer<All, HaloDepth> GInd;
        const gSite site = GInd::getSite(siteMu.isite);
        return (-_beta / 3.0) * gaugeActionDerivPlaq<double, HaloDepth>(_gAcc, site, siteMu.mu);
    }
};

template<size_t HaloDepth, size_t Ls>
class MDWFTwoFlavorHmc {
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
    typedef GIndexer<All, HaloDepth> GInd;

    CommunicationBase &_commBase;
    Gauge &_gauge;
    MDWFHmcParameters _param;
    uint4 *_randState;

    Gauge _momenta;
    Gauge _savedGauge;
    Gauge _ipdot;
    HostGauge _gaugeHost;
    HostGauge _ipdotHost;
    Spinor _phi;
    Spinor _eta;
    Spinor _actionWorkspace;

    // The operators hold a reference to _gauge and recompute the clover term on every
    // application, so they stay valid while the gauge field evolves in place.
    Forward _forward;
    Adjoint _adjoint;
    Normal _normal;
    NormalAdapter _normalAdapter;

    MDWFRationalCoefficients<double> _actionCoefficients;
    MDWFRationalCoefficients<double> _forceCoefficients;
    int _forceEvaluations;
    double _noiseNorm2;
    double _maxGaugeForceRms;
    double _maxFermionForceRms;

    // sqrt of the mean over bulk links of -tr(K K) for anti-Hermitian K (cf. forceinfo in integrator.cpp).
    static double forceRms(const HostGauge &ipdot) {
        const SU3Accessor<double, R18> acc = ipdot.getAccessor();
        double sum = 0.0;
        for (size_t siteIndex = 0; siteIndex < GInd::getLatData().vol4; siteIndex++) {
            const gSite site = GInd::getSite(siteIndex);
            for (uint8_t mu = 0; mu < 4; mu++) {
                const SU3<double> k = acc.getLink(GInd::getSiteMu(site, mu));
                sum += -tr_d(k, k);
            }
        }
        return std::sqrt(sum / (4.0 * static_cast<double>(GInd::getLatData().vol4)));
    }

    static MDWFRationalCoefficients<double> makeInverse(MDWFRationalCoefficientRole role, const std::string &name) {
        MDWFExplicitRationalInput<double> input{name, role, 0.0, {1.0}, {0.0}};
        return makeMDWFRationalCoefficients(input);
    }

    void updateFermionForce() {
        Workspace workspace;
        workspace.prepare(_normal, _forward, _phi, _forceCoefficients, _param.max_iter, _param.precision,
                          "MDWF_hmc_force_workspace");
        if (!workspace.converged()) {
            throw std::runtime_error(stdLogger.fatal("MDWF HMC fermion force solve did not converge"));
        }
        View view(workspace, _forward.params().dinCoeff, "MDWF_hmc_force_view");

        _gaugeHost = _gauge;
        if (_param.csw != 0.0) {
            overwriteMDWFCloverAllLinkDirectionIndependentStorageNonzero<HaloDepth, Ls>(
                _ipdotHost, _gaugeHost, view, _forceCoefficients, _param.csw, _commBase, "MDWF_hmc_force_storage");
        } else {
            overwriteMDWFWilsonAllLinkDirectionIndependentStorageCsw0<HaloDepth, Ls>(
                _ipdotHost, _gaugeHost, view, _forceCoefficients, _commBase, "MDWF_hmc_force_storage");
        }
        _maxFermionForceRms = std::max(_maxFermionForceRms, forceRms(_ipdotHost));
        _ipdot = _ipdotHost;
    }

public:
    MDWFTwoFlavorHmc(CommunicationBase &commBase, Gauge &gauge, const MDWFHmcParameters &param, uint4 *randState)
        : _commBase(commBase),
          _gauge(gauge),
          _param(param),
          _randState(randState),
          _momenta(commBase, "MDWF_hmc_momenta"),
          _savedGauge(commBase, "MDWF_hmc_saved_gauge"),
          _ipdot(commBase, "MDWF_hmc_ipdot"),
          _gaugeHost(commBase, "MDWF_hmc_gauge_host"),
          _ipdotHost(commBase, "MDWF_hmc_ipdot_host"),
          _phi(commBase, "MDWF_hmc_phi"),
          _eta(commBase, "MDWF_hmc_eta"),
          _actionWorkspace(commBase, "MDWF_hmc_action_workspace"),
          _forward(gauge, param.M5, param.mf, param.b5, param.csw, "MDWF_hmc_forward"),
          _adjoint(gauge, param.M5, param.mf, param.b5, param.csw, "MDWF_hmc_adjoint"),
          _normal(commBase, _forward, _adjoint, "MDWF_hmc_normal"),
          _normalAdapter(_normal),
          _actionCoefficients(makeInverse(MDWFRationalCoefficientRole::Action, "MDWF_hmc_action")),
          _forceCoefficients(makeInverse(MDWFRationalCoefficientRole::Force, "MDWF_hmc_force")),
          _forceEvaluations(0),
          _noiseNorm2(0.0),
          _maxGaugeForceRms(0.0),
          _maxFermionForceRms(0.0) {
        const LatticeData lat = GInd::getLatData();
        if (lat.vol4 != lat.globvol4) {
            throw std::runtime_error(stdLogger.fatal("MDWF two-flavour HMC scaffold is single-rank only"));
        }
        if (param.steps <= 0 || !(param.tau > 0.0) || param.max_iter <= 0 || !(param.precision > 0.0)) {
            throw std::runtime_error(stdLogger.fatal("MDWF HMC requires positive tau, steps, max_iter, precision"));
        }
    }

    void refreshMomenta() {
        _momenta.gauss(_randState);
        _momenta.updateAll();
    }

    // phi = M^\dagger eta with eta distributed as exp(-eta^\dagger eta); then S_f = eta^\dagger eta.
    void heatbath() {
        _eta.gauss(_randState);
        _eta.updateAll();
        _noiseNorm2 = _normalAdapter.norm2(_eta);
        _adjoint.apply(_phi, _eta, true);
    }

    void flipMomenta() {
        _momenta = -1.0 * _momenta;
        _momenta.updateAll();
    }

    double noiseNorm2() const {
        return _noiseNorm2;
    }

    double kineticEnergy() {
        HostGauge momentaHost(_commBase, "MDWF_hmc_momenta_host");
        momentaHost = _momenta;
        const SU3Accessor<double> pAcc = momentaHost.getAccessor();
        double sum = 0.0;
        for (size_t siteIndex = 0; siteIndex < GInd::getLatData().vol4; siteIndex++) {
            const gSite site = GInd::getSite(siteIndex);
            for (uint8_t mu = 0; mu < 4; mu++) {
                const SU3<double> p = pAcc.getLink(GInd::getSiteMu(site, mu));
                sum += tr_d(p, p);
            }
        }
        return 0.5 * sum;
    }

    double gaugeAction() {
        GaugeAction<double, true, HaloDepth, R18> action(_gauge);
        return -(_param.beta / 3.0) * 18.0 * static_cast<double>(GInd::getLatData().globvol4)
               * static_cast<double>(action.plaquette());
    }

    double fermionAction() {
        const MDWFRationalActionResult<double> result = computeMDWFRationalAction<double, NormalAdapter>(
            _normalAdapter, _actionWorkspace, _phi, _actionCoefficients, _param.max_iter, _param.precision,
            "MDWF_hmc_action");
        if (!result.rational_result.converged()) {
            throw std::runtime_error(stdLogger.fatal("MDWF HMC fermion action solve did not converge"));
        }
        return result.action_real;
    }

    MDWFHmcEnergy energy() {
        return {kineticEnergy(), gaugeAction(), fermionAction()};
    }

    void evolveQ(double stepsize) {
        _gauge.iterateOverBulkAllMu(MDWFHmcEvolveQ<HaloDepth>(_gauge.getAccessor(), _momenta.getAccessor(), stepsize));
        _gauge.updateAll();
    }

    // Applies the gauge and fermion forces as two separate momentum updates, as integrator.cpp does.
    void evolveP(double stepsize) {
        _ipdot.iterateOverBulkAllMu(MDWFHmcWilsonGaugeForce<HaloDepth>(_gauge.getAccessor(), _param.beta));
        _ipdotHost = _ipdot;
        _maxGaugeForceRms = std::max(_maxGaugeForceRms, forceRms(_ipdotHost));
        _momenta.iterateOverBulkAllMu(MDWFHmcEvolveP<HaloDepth>(_momenta.getAccessor(), _ipdot.getAccessor(), stepsize));

        updateFermionForce();
        _momenta.iterateOverBulkAllMu(MDWFHmcEvolveP<HaloDepth>(_momenta.getAccessor(), _ipdot.getAccessor(), stepsize));
        _momenta.updateAll();
        _forceEvaluations++;
    }

    // Leapfrog P(eps/2) [Q(eps) P(eps)]^(steps-1) Q(eps) P(eps/2), eps = tau / steps.
    void integrate(int steps) {
        const double eps = _param.tau / static_cast<double>(steps);
        evolveP(0.5 * eps);
        for (int step = 0; step < steps - 1; step++) {
            evolveQ(eps);
            evolveP(eps);
        }
        evolveQ(eps);
        evolveP(0.5 * eps);
    }

    int forceEvaluations() const {
        return _forceEvaluations;
    }

    void resetForceStatistics() {
        _maxGaugeForceRms = 0.0;
        _maxFermionForceRms = 0.0;
    }

    double maxGaugeForceRms() const {
        return _maxGaugeForceRms;
    }

    double maxFermionForceRms() const {
        return _maxFermionForceRms;
    }

    Gauge &momenta() {
        return _momenta;
    }

    Spinor &phi() {
        return _phi;
    }

    MDWFHmcTrajectoryResult trajectory(bool metropolis, std::mt19937_64 &acceptRng) {
        _savedGauge = _gauge;
        refreshMomenta();
        heatbath();
        const int forceEvaluationsBefore = _forceEvaluations;

        const MDWFHmcEnergy before = energy();
        integrate(_param.steps);
        const MDWFHmcEnergy after = energy();
        const double deltaH = after.total() - before.total();

        bool accepted = true;
        if (metropolis) {
            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            accepted = deltaH <= 0.0 || uniform(acceptRng) < std::exp(-deltaH);
            if (!accepted) {
                _gauge = _savedGauge;
                _gauge.updateAll();
            }
        }
        return {before, after, deltaH, accepted, _forceEvaluations - forceEvaluationsBefore};
    }
};
