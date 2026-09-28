/*
 * Test-only single-rank MDWF HMC driver (steps 2 and 4 of the MDWF RHMC plan
 * in TODO.md), following MDWF_HMC_CONVENTIONS.md:
 *
 *   H = (1/2) sum_links tr(P P) + S_g + S_f,
 *   U -> exp(i eps P) U,   P -> P - i eps ipdot,   ipdot_l = K_l = TA(B_l),
 *
 * with the Wilson gauge action S_g = -(beta/3) sum_plaquettes Re tr U_p and
 * ipdot_g = -(beta/3) gaugeActionDerivPlaq (identity confirmed by
 * mdwfMobiusHmcConventionTest), or, with MDWFHmcParameters::symanzik_gauge,
 * SIMULATeQCD's tree-level Symanzik action S_g = -(3 beta/5) symanzik() =
 * -(beta/3) sum Re tr U_p + (beta/60) sum Re tr U_rect with ipdot_g =
 * gauge_force (gaugeActionDeriv.h), as in SIMULATeQCD's own HMC. Both gauge
 * forces are evaluated on the device. gauge_force must not be evaluated on the
 * host: there GIndexer::site_up_2dn(s, mu, nu) falls back to
 * site_up_dn_dn(s, mu, mu, nu) = s - nu instead of s + mu - 2 nu, which
 * corrupts one of the six rectangle staples (the GPU path site_move<1, -2> is
 * correct; see TODO.md and mdwfSymanzikGaugeForceTest). The fermion action is
 * a template parameter from MDWFHmcFermionActions.h:
 *
 *   MDWFTwoFlavorHmc              bare det(M^\dagger M)          (validated by mdwfHmcTrajectoryTest)
 *   MDWFPauliVillarsTwoFlavorHmc  det(M_f^\dagger M_f) / det(M_1^\dagger M_1)
 *
 * The two MD update formulas
 * reproduce do_evolve_Q / do_evolve_P from src/modules/rhmc/integrator.cpp,
 * which are file-local there; that module is not modified.
 *
 * This is a correctness scaffold: single rank, plain leapfrog with the gauge
 * and fermion forces applied as separate momentum updates, the fermion force
 * stored on the host each step, and unpreconditioned solves from a zero
 * initial guess. It is intended for reversibility, Delta H scaling, and
 * <exp(-Delta H)> checks on small lattices, not for production.
 */

#pragma once

#include "MDWFHmcFermionActions.h"
#include "../../gauge/gaugeAction.h"
#include "../../gauge/gaugeActionDeriv.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

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

// Symanzik gauge force; device only (see the header comment on site_up_2dn).
template<size_t HaloDepth>
struct MDWFHmcSymanzikGaugeForce {
    SU3Accessor<double, R18> _gAcc;
    double _beta;

    MDWFHmcSymanzikGaugeForce(SU3Accessor<double, R18> gAcc, double beta) : _gAcc(gAcc), _beta(beta) {}

    __host__ __device__ SU3<double> operator()(gSiteMu siteMu) {
        return gauge_force<double, HaloDepth, R18>(_gAcc, siteMu, _beta);
    }
};

template<size_t HaloDepth, size_t Ls, class FermionAction>
class MDWFHmcDriver {
public:
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;
    using HostGauge = Gaugefield<double, false, HaloDepth, R18>;
    using Spinor = MDWFSpinor<double, true, All, HaloDepth, Ls>;

private:
    typedef GIndexer<All, HaloDepth> GInd;

    CommunicationBase &_commBase;
    Gauge &_gauge;
    MDWFHmcParameters _param;
    uint4 *_randState;
    std::string _prefix;

    Gauge _momenta;
    Gauge _savedGauge;
    Gauge _ipdot;
    HostGauge _gaugeHost;
    HostGauge _ipdotHost;
    FermionAction _fermion;

    int _forceEvaluations;
    int _gaugeUpdates;
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

    void updateFermionForce() {
        _gaugeHost = _gauge;
        _fermion.force(_ipdotHost, _gaugeHost);
        _maxFermionForceRms = std::max(_maxFermionForceRms, forceRms(_ipdotHost));
        _ipdot = _ipdotHost;
    }

public:
    MDWFHmcDriver(CommunicationBase &commBase, Gauge &gauge, const MDWFHmcParameters &param, uint4 *randState)
        : _commBase(commBase),
          _gauge(gauge),
          _param(param),
          _randState(randState),
          _prefix(mdwfHmcInstancePrefix("MDWF_hmc_drv")),
          _momenta(commBase, _prefix + "_momenta"),
          _savedGauge(commBase, _prefix + "_saved_gauge"),
          _ipdot(commBase, _prefix + "_ipdot"),
          _gaugeHost(commBase, _prefix + "_gauge_host"),
          _ipdotHost(commBase, _prefix + "_ipdot_host"),
          _fermion(commBase, gauge, param),
          _forceEvaluations(0),
          _gaugeUpdates(0),
          _maxGaugeForceRms(0.0),
          _maxFermionForceRms(0.0) {
        const LatticeData lat = GInd::getLatData();
        if (lat.vol4 != lat.globvol4) {
            throw std::runtime_error(stdLogger.fatal("MDWF HMC scaffold is single-rank only"));
        }
        if (param.steps <= 0 || !(param.tau > 0.0) || param.max_iter <= 0 || !(param.precision > 0.0)) {
            throw std::runtime_error(stdLogger.fatal("MDWF HMC requires positive tau, steps, max_iter, precision"));
        }
    }

    void refreshMomenta() {
        _momenta.gauss(_randState);
        _momenta.updateAll();
    }

    void heatbath() {
        _fermion.heatbath(_randState);
    }

    void flipMomenta() {
        _momenta = -1.0 * _momenta;
        _momenta.updateAll();
    }

    double noiseNorm2() const {
        return _fermion.noiseNorm2();
    }

    double kineticEnergy() {
        HostGauge momentaHost(_commBase, _prefix + "_kinetic_host");
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
        if (_param.symanzik_gauge) {
            return -(3.0 * _param.beta / 5.0) * static_cast<double>(action.symanzik());
        }
        return -(_param.beta / 3.0) * 18.0 * static_cast<double>(GInd::getLatData().globvol4)
               * static_cast<double>(action.plaquette());
    }

    double fermionAction() {
        return _fermion.action();
    }

    MDWFHmcEnergy energy() {
        return {kineticEnergy(), gaugeAction(), fermionAction()};
    }

    void evolveQ(double stepsize) {
        _gauge.iterateOverBulkAllMu(MDWFHmcEvolveQ<HaloDepth>(_gauge.getAccessor(), _momenta.getAccessor(), stepsize));
        _gauge.updateAll();
    }

    void updatePGauge(double stepsize) {
        if (_param.symanzik_gauge) {
            _ipdot.iterateOverBulkAllMu(MDWFHmcSymanzikGaugeForce<HaloDepth>(_gauge.getAccessor(), _param.beta));
        } else {
            _ipdot.iterateOverBulkAllMu(MDWFHmcWilsonGaugeForce<HaloDepth>(_gauge.getAccessor(), _param.beta));
        }
        _ipdotHost = _ipdot;
        _maxGaugeForceRms = std::max(_maxGaugeForceRms, forceRms(_ipdotHost));
        _momenta.iterateOverBulkAllMu(MDWFHmcEvolveP<HaloDepth>(_momenta.getAccessor(), _ipdot.getAccessor(), stepsize));
        _momenta.updateAll();
        _gaugeUpdates++;
    }

    void updatePFermion(double stepsize) {
        updateFermionForce();
        _momenta.iterateOverBulkAllMu(MDWFHmcEvolveP<HaloDepth>(_momenta.getAccessor(), _ipdot.getAccessor(), stepsize));
        _momenta.updateAll();
        _forceEvaluations++;
    }

    // Both forces with the same step size, gauge first (a plain leapfrog momentum update).
    void evolveP(double stepsize) {
        updatePGauge(stepsize);
        updatePFermion(stepsize);
    }

    /*
     * Sexton-Weingarten two-scale leapfrog, as SWleapfrog in integrator.cpp with one fermion
     * scale: fermion step eps = tau / steps, gauge step delta = eps / gaugeSubsteps. Each fermion
     * step is P_f(eps/2) [inner leapfrog of Q and P_g over time eps] P_f(eps/2), with adjacent
     * half steps merged. At gaugeSubsteps = 1 this performs exactly the plain leapfrog
     * P(eps/2) [Q(eps) P(eps)]^(steps-1) Q(eps) P(eps/2), operation by operation.
     */
    void integrate(int steps, int gaugeSubsteps) {
        if (steps <= 0) {
            throw std::runtime_error(stdLogger.fatal("MDWF HMC integrate requires steps > 0"));
        }
        const int substeps = std::max(1, gaugeSubsteps);
        const double eps = _param.tau / static_cast<double>(steps);
        const double delta = eps / static_cast<double>(substeps);

        updatePGauge(0.5 * delta);
        updatePFermion(0.5 * eps);
        for (int step = 0; step < steps - 1; step++) {
            for (int sub = 0; sub < substeps; sub++) {
                evolveQ(delta);
                updatePGauge(delta);
            }
            updatePFermion(eps);
        }
        for (int sub = 0; sub < substeps - 1; sub++) {
            evolveQ(delta);
            updatePGauge(delta);
        }
        evolveQ(delta);
        updatePGauge(0.5 * delta);
        updatePFermion(0.5 * eps);
    }

    void integrate(int steps) {
        integrate(steps, _param.gauge_substeps);
    }

    int forceEvaluations() const {
        return _forceEvaluations;
    }

    int gaugeUpdates() const {
        return _gaugeUpdates;
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

    // Fermion force of the current state and phi, without changing the momenta.
    double currentFermionForceRms() {
        _gaugeHost = _gauge;
        _fermion.force(_ipdotHost, _gaugeHost);
        return forceRms(_ipdotHost);
    }

    Gauge &momenta() {
        return _momenta;
    }

    Spinor &phi() {
        return _fermion.phi();
    }

    FermionAction &fermion() {
        return _fermion;
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

template<size_t HaloDepth, size_t Ls>
using MDWFTwoFlavorHmc = MDWFHmcDriver<HaloDepth, Ls, MDWFBareTwoFlavorFermionAction<HaloDepth, Ls>>;

template<size_t HaloDepth, size_t Ls>
using MDWFPauliVillarsTwoFlavorHmc = MDWFHmcDriver<HaloDepth, Ls, MDWFPauliVillarsTwoFlavorFermionAction<HaloDepth, Ls>>;
