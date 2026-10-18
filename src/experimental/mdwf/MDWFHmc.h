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
 * Fermion boundary conditions in time: periodic by default; with
 * MDWFHmcParameters::antiperiodic_t the fermion action, its heatbath and its
 * force all see MDWFFermionGauge (MDWFFermionBoundary.h), a copy of the gauge
 * field with -U_t on the last time slice, re-derived from the thin field
 * before every heatbath, action and force evaluation (so external changes to
 * the gauge field, such as a restore or a configuration read, are picked up).
 * The force is then the force with respect to the thin links (see
 * MDWFFermionBoundary.h); the gauge action and force use the thin field.
 *
 * This is a correctness scaffold: single rank, plain leapfrog with the gauge
 * and fermion forces applied as separate momentum updates, the fermion force
 * stored on the host each step, and unpreconditioned solves from a zero
 * initial guess. It is intended for reversibility, Delta H scaling, and
 * <exp(-Delta H)> checks on small lattices, not for production.
 */

#pragma once

#include "MDWFSpinor.h"

#include "MDWFFermionBoundary.h"
#include "MDWFHmcFermionActions.h"
#include "../../gauge/gaugeAction.h"
#include "../../gauge/gaugeActionDeriv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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

// Forward trajectory, momentum flip, trajectory back, flip (MDWFHmcDriver::reversibilityCheck).
struct MDWFHmcReversibilityResult {
    MDWFHmcEnergy start;
    MDWFHmcEnergy forward;
    MDWFHmcEnergy back;
    double max_link_diff;
    double max_momentum_diff;
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

// Force-gradient displacement U -> exp(i c Pf) U = exp(c ipdot) U, Pf = -i ipdot the unit kick from zero momentum.
template<size_t HaloDepth>
struct MDWFHmcDisplaceFG {
    SU3Accessor<double, R18> _gAcc;
    SU3Accessor<double> _fAcc;
    double _c;

    MDWFHmcDisplaceFG(SU3Accessor<double, R18> gAcc, SU3Accessor<double> fAcc, double c)
        : _gAcc(gAcc), _fAcc(fAcc), _c(c) {}

    __host__ __device__ SU3<double> operator()(gSiteMu site) {
        SU3<double> temp = su3_exp<double>(COMPLEX(double)(_c, 0.0) * _fAcc.getLink(site)) * _gAcc.getLink(site);
        temp.su3unitarize();
        return temp;
    }
};

template<size_t HaloDepth>
struct MDWFHmcAddFields {
    SU3Accessor<double> _aAcc;
    SU3Accessor<double> _bAcc;

    MDWFHmcAddFields(SU3Accessor<double> aAcc, SU3Accessor<double> bAcc) : _aAcc(aAcc), _bAcc(bAcc) {}

    __host__ __device__ SU3<double> operator()(gSiteMu site) {
        return _aAcc.getLink(site) + _bAcc.getLink(site);
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

/*
 * Plaquette + rectangle gauge force for S_g = -(beta/3) [c0 sum Re tr U_p + c1 sum Re tr U_rect], c0 = 1 - 8 c1
 * (MDWFHmcParameters::rectangle_gauge): SIMULATeQCD's gauge_force (gaugeActionDeriv.h) with its fixed Symanzik
 * coefficients replaced, g_c1 = -c0, g_c2 = -c1, r_1 = beta/3. gauge_force itself is S = -(3 beta/5) symanzik() =
 * -(beta/3) [sum Re tr U_p - (1/20) sum Re tr U_rect], i.e. this functor at (beta, c0 = 1, c1 = -1/20), or at
 * (3 beta/5, c1 = -1/12) in the c0 = 1 - 8 c1 normalization (mdwfDSDRTest checks the equality). Device only, as
 * gauge_force (site_up_2dn, see the header comment).
 */
template<size_t HaloDepth>
struct MDWFHmcRectangleGaugeForce {
    SU3Accessor<double, R18> _gAcc;
    double _beta;
    double _c0;
    double _c1;

    MDWFHmcRectangleGaugeForce(SU3Accessor<double, R18> gAcc, double beta, double c0, double c1)
        : _gAcc(gAcc), _beta(beta), _c0(c0), _c1(c1) {}

    __host__ __device__ SU3<double> operator()(gSiteMu site) {
        typedef GIndexer<All, HaloDepth> GInd;
        SU3Accessor<double, R18> &latacc = _gAcc;

        const gSite s(GInd::getSite(site.isite));
        const int mu = site.mu;

        const double g_c1 = -_c0;
        const double g_c2 = -_c1;
        const COMPLEX(double) r_1 = COMPLEX(double)(_beta / 3.0, 0.0);

        SU3<double> m_0, m_res1, staple;
        staple = su3_zero<double>();

        for (int nu_aux = 1; nu_aux < 4; nu_aux++) {
            const int nu = (mu + nu_aux) % 4;
            m_0 = latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(s, mu, nu), nu));
            m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up_2dn(s, mu, nu), nu));
            m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_2dn(s, nu), mu));
            m_0 *= latacc.getLink(GInd::getSiteMu(GInd::site_2dn(s, nu), nu));
            m_0 *= latacc.getLink(GInd::getSiteMu(GInd::site_dn(s, nu), nu));
            staple += g_c2 * m_0;

            m_0 = latacc.getLink(GInd::getSiteMu(GInd::site_up(s, mu), nu));
            m_0 *= latacc.getLink(GInd::getSiteMu(GInd::site_up_up(s, nu, mu), nu));
            m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_2up(s, nu), mu));
            m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up(s, nu), nu));
            m_0 *= latacc.getLinkDagger(GInd::getSiteMu(s, nu));
            staple += g_c2 * m_0;

            m_res1 = latacc.getLink(GInd::getSiteMu(GInd::site_2up(s, mu), nu));
            m_res1 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up_up(s, nu, mu), mu));
            m_res1 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up(s, nu), mu));
            m_res1 *= latacc.getLinkDagger(GInd::getSiteMu(s, nu));

            m_0 = latacc.getLinkDagger(GInd::getSiteMu(GInd::site_2up_dn(s, mu, nu), nu));
            m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(s, mu, nu), mu));
            m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(s, nu), mu));
            m_0 *= latacc.getLink(GInd::getSiteMu(GInd::site_dn(s, nu), nu));

            m_0 += m_res1;

            m_res1 = g_c2 * latacc.getLink(GInd::getSiteMu(GInd::site_up(s, mu), mu)) * m_0;
            staple += m_res1;

            m_0 = latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(s, nu, mu), mu));
            m_0 *= latacc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(s, mu), nu));
            m_0 *= latacc.getLink(GInd::getSiteMu(GInd::site_dn(s, mu), mu));

            m_res1 = latacc.getLink(GInd::getSiteMu(GInd::site_up(s, mu), nu));
            m_res1 = m_res1 * latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up(s, nu), mu));
            m_res1 *= (g_c1 * latacc.getLinkDagger(GInd::getSiteMu(s, nu)) + g_c2 * m_0);
            staple += m_res1;

            m_0 = latacc.getLinkDagger(GInd::getSiteMu(GInd::site_dn_dn(s, mu, nu), mu));
            m_0 = m_0 * latacc.getLink(GInd::getSiteMu(GInd::site_dn_dn(s, mu, nu), nu));
            m_0 = m_0 * latacc.getLink(GInd::getSiteMu(GInd::site_dn(s, mu), mu));

            m_res1 = latacc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(s, mu, nu), nu));
            m_res1 = m_res1 * latacc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(s, nu), mu));
            m_res1 *= (g_c1 * latacc.getLink(GInd::getSiteMu(GInd::site_dn(s, nu), nu)) + g_c2 * m_0);
            staple += m_res1;
        }

        m_res1 = r_1 * latacc.getLink(GInd::getSiteMu(s, mu)) * staple;
        m_res1.TA();
        return m_res1;
    }
};

// S_g of MDWFHmcRectangleGaugeForce (no additive constant): -(beta/3) [(1 - 8 c1) sum Re tr U_p + c1 sum Re tr U_rect].
template<size_t HaloDepth>
double mdwfRectangleGaugeAction(Gaugefield<double, true, HaloDepth, R18> &gauge, double beta, double c1) {
    typedef GIndexer<All, HaloDepth> GInd;
    GaugeAction<double, true, HaloDepth, R18> action(gauge);
    const double volume = static_cast<double>(GInd::getLatData().globvol4);
    // plaquette() = sum Re tr U_p / (18 V) over the 6 V plaquettes, rectangle() = sum Re tr U_rect / (36 V) over 12 V.
    const double sumPlaq = 18.0 * volume * static_cast<double>(action.plaquette());
    const double sumRect = 36.0 * volume * static_cast<double>(action.rectangle());
    return -(beta / 3.0) * ((1.0 - 8.0 * c1) * sumPlaq + c1 * sumRect);
}

/*
 * Force terms of a fermion action, for the multi-level integrator. An action with forceTermCount() and
 * forceTerm(i, ipdotHost, gaugeHost) splits its force into separately integrable terms (Hasenbusch factors, the strange
 * RHMC, ...); any other action is a single term (its force()).
 */
template<class Action, class = void>
struct MDWFHasForceTerms : std::false_type {};

template<class Action>
struct MDWFHasForceTerms<Action, std::void_t<decltype(std::declval<Action &>().forceTermCount())>> : std::true_type {};

template<class Action>
size_t mdwfForceTermCount(Action &action) {
    if constexpr (MDWFHasForceTerms<Action>::value) {
        return action.forceTermCount();
    } else {
        return 1;
    }
}

template<class Action, class HostGauge>
void mdwfForceTerm(Action &action, size_t term, HostGauge &ipdotHost, const HostGauge &gaugeHost) {
    if constexpr (MDWFHasForceTerms<Action>::value) {
        action.forceTerm(term, ipdotHost, gaugeHost);
    } else {
        action.force(ipdotHost, gaugeHost);
    }
}

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
    MDWFFermionGauge<HaloDepth> _fermionGauge;   // before _fermion: the action binds to _fermionGauge.get()
    FermionAction _fermion;

    int _forceEvaluations;
    int _gaugeUpdates;
    double _fermionForceSeconds;
    double _maxGaugeForceRms;
    double _maxFermionForceRms;

    // Multi-level integrator: force caches, valid while the gauge field is unchanged (_gaugeVersion counts evolveQ).
    long _gaugeVersion;
    std::vector<std::unique_ptr<Gauge>> _termForce;
    std::vector<long> _termVersion;
    std::vector<int> _termEvaluations;
    std::vector<double> _termMaxForceRms;
    std::vector<double> _termSeconds;
    std::vector<long long> _termCgIterations;
    std::vector<double> _termNorm2Sum;     // sum and sum of squares of rms^2 (= GPT's force norm2/sites) per term
    std::vector<double> _termNorm2SumSq;
    // Wall-clock of the heatbath and of the two action evaluations per trajectory (accumulated).
    double _heatbathSeconds = 0.0;
    double _actionSeconds = 0.0;
    std::unique_ptr<Gauge> _gaugeForce;
    long _gaugeForceVersion;
    // Force-gradient step: summed force of the level and the gauge field before the displacement.
    std::unique_ptr<Gauge> _fgForce;
    std::unique_ptr<Gauge> _fgSavedGauge;

    // sqrt of the mean over bulk links of -tr(K K) for anti-Hermitian K (cf. forceinfo in integrator.cpp), all ranks.
    double forceRms(const HostGauge &ipdot) {
        const SU3Accessor<double, R18> acc = ipdot.getAccessor();
        double sum = 0.0;
        for (size_t siteIndex = 0; siteIndex < GInd::getLatData().vol4; siteIndex++) {
            const gSite site = GInd::getSite(siteIndex);
            for (uint8_t mu = 0; mu < 4; mu++) {
                const SU3<double> k = acc.getLink(GInd::getSiteMu(site, mu));
                sum += -tr_d(k, k);
            }
        }
        return std::sqrt(_commBase.reduce(sum) / (4.0 * static_cast<double>(GInd::getLatData().globvol4)));
    }

    void updateFermionForce() {
        const auto start = std::chrono::steady_clock::now();
        _fermionGauge.sync();
        _gaugeHost = _fermionGauge.get();
        _fermion.force(_ipdotHost, _gaugeHost);
        _maxFermionForceRms = std::max(_maxFermionForceRms, forceRms(_ipdotHost));
        _ipdot = _ipdotHost;
        _fermionForceSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
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
          _fermionGauge(commBase, gauge, param.antiperiodic_t, _prefix + "_fermion_bc_links"),
          _fermion(commBase, _fermionGauge.get(), param),
          _forceEvaluations(0),
          _gaugeUpdates(0),
          _fermionForceSeconds(0.0),
          _maxGaugeForceRms(0.0),
          _maxFermionForceRms(0.0),
          _gaugeVersion(0),
          _gaugeForceVersion(-1) {
        if (param.steps <= 0 || !(param.tau > 0.0) || param.max_iter <= 0 || !(param.precision > 0.0)) {
            throw std::runtime_error(stdLogger.fatal("MDWF HMC requires positive tau, steps, max_iter, precision"));
        }
    }

    void refreshMomenta() {
        _momenta.gauss(_randState);
        mdwfUpdateHalo(_momenta);
    }

    void heatbath() {
        _fermionGauge.sync();
        _fermion.heatbath(_randState);
    }

    void flipMomenta() {
        _momenta = -1.0 * _momenta;
        mdwfUpdateHalo(_momenta);
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
        return 0.5 * _commBase.reduce(sum);
    }

    double gaugeAction() {
        if (_param.rectangle_gauge) {
            return mdwfRectangleGaugeAction<HaloDepth>(_gauge, _param.beta, _param.gauge_c1);
        }
        GaugeAction<double, true, HaloDepth, R18> action(_gauge);
        if (_param.symanzik_gauge) {
            return -(3.0 * _param.beta / 5.0) * static_cast<double>(action.symanzik());
        }
        return -(_param.beta / 3.0) * 18.0 * static_cast<double>(GInd::getLatData().globvol4)
               * static_cast<double>(action.plaquette());
    }

    double fermionAction() {
        _fermionGauge.sync();
        return _fermion.action();
    }

    MDWFHmcEnergy energy() {
        return {kineticEnergy(), gaugeAction(), fermionAction()};
    }

    void evolveQ(double stepsize) {
        _gaugeVersion++;
        _gauge.iterateOverBulkAllMu(MDWFHmcEvolveQ<HaloDepth>(_gauge.getAccessor(), _momenta.getAccessor(), stepsize));
        mdwfUpdateHalo(_gauge);
    }

    // ipdot of the gauge action (Wilson, Symanzik or plaquette + rectangle), on the device.
    void gaugeForce(Gauge &destination) {
        if (_param.rectangle_gauge) {
            destination.iterateOverBulkAllMu(MDWFHmcRectangleGaugeForce<HaloDepth>(
                _gauge.getAccessor(), _param.beta, 1.0 - 8.0 * _param.gauge_c1, _param.gauge_c1));
        } else if (_param.symanzik_gauge) {
            destination.iterateOverBulkAllMu(MDWFHmcSymanzikGaugeForce<HaloDepth>(_gauge.getAccessor(), _param.beta));
        } else {
            destination.iterateOverBulkAllMu(MDWFHmcWilsonGaugeForce<HaloDepth>(_gauge.getAccessor(), _param.beta));
        }
    }

    void updatePGauge(double stepsize) {
        gaugeForce(_ipdot);
        _ipdotHost = _ipdot;
        _maxGaugeForceRms = std::max(_maxGaugeForceRms, forceRms(_ipdotHost));
        _momenta.iterateOverBulkAllMu(MDWFHmcEvolveP<HaloDepth>(_momenta.getAccessor(), _ipdot.getAccessor(), stepsize));
        mdwfUpdateHalo(_momenta);
        _gaugeUpdates++;
    }

    void updatePFermion(double stepsize) {
        updateFermionForce();
        _momenta.iterateOverBulkAllMu(MDWFHmcEvolveP<HaloDepth>(_momenta.getAccessor(), _ipdot.getAccessor(), stepsize));
        mdwfUpdateHalo(_momenta);
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

    int gaugeScheme() const {
        return _param.gauge_scheme < 0 ? _param.md_scheme : _param.gauge_scheme;
    }

    void integrate(int steps) {
        if (!_param.term_level.empty() || _param.md_scheme != 0 || gaugeScheme() != 0) {
            integrateMultiLevel(steps);
        } else {
            integrate(steps, _param.gauge_substeps);
        }
    }

    /*
     * Nested (Sexton-Weingarten) leapfrog over the fermion force terms grouped by MDWFHmcParameters::term_level, with
     * the gauge force on the finest level: level l evolves time dt in n_l steps h = dt / n_l, each
     * P_l(h/2) [level l + 1 over h] P_l(h/2), P_l = all terms of level l, n_0 = steps, n_l = level_substeps[l - 1],
     * the gauge level gauge_substeps steps of P_g(d/2) Q(d) P_g(d/2). A term's force is computed once per gauge
     * configuration (cache by _gaugeVersion) and reused for the adjacent half kicks, so a level makes n + 1 force
     * evaluations per evolution and level l in total n_0 n_1 ... n_l + 1 per trajectory. With one level, term_level
     * all 0, this is the two-scale integrator above with the merged kicks split in two halves.
     */
    void integrateMultiLevel(int steps) {
        integrateMultiLevel(steps, _param.tau);
    }

    // The multi-level integrator over MD time dt instead of tau (n steps of dt / n). integrateTime(tau / n, 1) n times
    // is the same trajectory as integrateMultiLevel(n) (same arithmetic; forces at the step boundaries are recomputed
    // instead of taken from the cache). Used by the exact-comparison suite tests/exact8 to stop after every step.
    void integrateTime(double dt, int steps) {
        integrateMultiLevel(steps, dt);
    }

    void integrateMultiLevel(int steps, double dt) {
        const size_t terms = mdwfForceTermCount(_fermion);
        const size_t levels = 1 + _param.level_substeps.size();
        if (_param.term_level.empty()) {
            _param.term_level.assign(terms, 0);
        }
        for (const int s : {_param.md_scheme, gaugeScheme()}) {
            if (s < 0 || s > 2) {
                throw std::runtime_error(stdLogger.fatal("MDWF multi-level integrator: scheme ", s,
                                                         " (0 leapfrog, 1 OMF2, 2 OMF2 force-gradient)"));
            }
        }
        if (_param.term_level.size() != terms) {
            throw std::runtime_error(stdLogger.fatal("MDWF multi-level integrator: term_level has ", _param.term_level.size(),
                                                     " entries, the fermion action has ", terms, " force terms"));
        }
        for (const int l : _param.term_level) {
            if (l < 0 || static_cast<size_t>(l) >= levels) {
                throw std::runtime_error(stdLogger.fatal("MDWF multi-level integrator: term level ", l, " outside 0 ... ",
                                                         levels - 1, " (level_substeps has ", levels - 1, " entries)"));
            }
        }
        for (const int n : _param.level_substeps) {
            if (n <= 0) {
                throw std::runtime_error(stdLogger.fatal("MDWF multi-level integrator: level_substeps must be positive"));
            }
        }
        if (steps <= 0) {
            throw std::runtime_error(stdLogger.fatal("MDWF HMC integrate requires steps > 0"));
        }
        if (_termForce.size() != terms) {
            _termForce.clear();
            for (size_t t = 0; t < terms; t++) {
                _termForce.push_back(std::make_unique<Gauge>(_commBase, _prefix + "_tforce" + std::to_string(t) + "x"));
            }
            _termVersion.assign(terms, -1);
            _termEvaluations.assign(terms, 0);
            _termMaxForceRms.assign(terms, 0.0);
            _termSeconds.assign(terms, 0.0);
            _termCgIterations.assign(terms, 0);
            _termNorm2Sum.assign(terms, 0.0);
            _termNorm2SumSq.assign(terms, 0.0);
        }
        if (!_gaugeForce) {
            _gaugeForce = std::make_unique<Gauge>(_commBase, _prefix + "_gforcecache");
        }
        if ((_param.md_scheme == 2 || gaugeScheme() == 2) && !_fgForce) {
            _fgForce = std::make_unique<Gauge>(_commBase, _prefix + "_fgforce");
            _fgSavedGauge = std::make_unique<Gauge>(_commBase, _prefix + "_fgsavedgauge");
        }
        // The gauge field may have been set from outside since the last call.
        std::fill(_termVersion.begin(), _termVersion.end(), -1);
        _gaugeForceVersion = -1;
        evolveLevel(0, dt, steps);
    }

    // Number of separately integrable fermion force terms, and the force of term t at the current gauge field and
    // pseudofermions (diagnostics: no statistics, no effect on the integrator caches).
    size_t forceTermCount() {
        return mdwfForceTermCount(_fermion);
    }

    void termForce(size_t t, HostGauge &ipdotHost) {
        _fermionGauge.sync();
        _gaugeHost = _fermionGauge.get();
        mdwfForceTerm(_fermion, t, ipdotHost, _gaugeHost);
    }

    // Force evaluations per fermion term and largest rms force per term (multi-level integrator).
    const std::vector<int> &termForceEvaluations() const {
        return _termEvaluations;
    }

    const std::vector<double> &termMaxForceRms() const {
        return _termMaxForceRms;
    }

    // Accumulated rms^2 and (rms^2)^2 per fermion force term over all evaluations; rms^2 = mean over links of
    // -tr(K K) equals GPT's "force norm2/sites" (sum_mu norm2(F_mu) / V, K = -2i F).
    const std::vector<double> &termForceNorm2Sum() const {
        return _termNorm2Sum;
    }

    const std::vector<double> &termForceNorm2SumSq() const {
        return _termNorm2SumSq;
    }

    // Accumulated wall-clock seconds per fermion force term (multi-level integrator).
    const std::vector<double> &termForceSeconds() const {
        return _termSeconds;
    }

    // Accumulated MDWFCoupledCG iterations of the force evaluations per fermion term (double-precision solves only;
    // mixed-precision solves are counted by their own solver).
    const std::vector<long long> &termForceCgIterations() const {
        return _termCgIterations;
    }

    double heatbathSeconds() const {
        return _heatbathSeconds;
    }

    double actionSeconds() const {
        return _actionSeconds;
    }

    void setIntegratorLevels(const std::vector<int> &termLevel, const std::vector<int> &levelSubsteps) {
        _param.term_level = termLevel;
        _param.level_substeps = levelSubsteps;
    }

private:
    void computeTermForce(size_t t) {
        if (_termVersion[t] != _gaugeVersion) {
            const auto start = std::chrono::steady_clock::now();
            const long long cgStart = mdwfCoupledCGIterationCount();
            _fermionGauge.sync();
            _gaugeHost = _fermionGauge.get();
            mdwfForceTerm(_fermion, t, _ipdotHost, _gaugeHost);
            _termCgIterations[t] += mdwfCoupledCGIterationCount() - cgStart;
            const double rms = forceRms(_ipdotHost);
            _termMaxForceRms[t] = std::max(_termMaxForceRms[t], rms);
            _termNorm2Sum[t] += rms * rms;
            _termNorm2SumSq[t] += rms * rms * rms * rms;
            _maxFermionForceRms = std::max(_maxFermionForceRms, rms);
            *_termForce[t] = _ipdotHost;
            _termVersion[t] = _gaugeVersion;
            _termEvaluations[t]++;
            _forceEvaluations++;
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            _termSeconds[t] += seconds;
            _fermionForceSeconds += seconds;
        }
    }

    void kickTerm(size_t t, double stepsize) {
        computeTermForce(t);
        _momenta.iterateOverBulkAllMu(
            MDWFHmcEvolveP<HaloDepth>(_momenta.getAccessor(), _termForce[t]->getAccessor(), stepsize));
        mdwfUpdateHalo(_momenta);
    }

    void kickLevel(size_t level, double stepsize) {
        for (size_t t = 0; t < _param.term_level.size(); t++) {
            if (static_cast<size_t>(_param.term_level[t]) == level) {
                kickTerm(t, stepsize);
            }
        }
    }

    void computeGaugeForce() {
        if (_gaugeForceVersion != _gaugeVersion) {
            gaugeForce(*_gaugeForce);
            _ipdotHost = *_gaugeForce;
            _maxGaugeForceRms = std::max(_maxGaugeForceRms, forceRms(_ipdotHost));
            _gaugeForceVersion = _gaugeVersion;
            _gaugeUpdates++;
        }
    }

    void kickGauge(double stepsize) {
        computeGaugeForce();
        _momenta.iterateOverBulkAllMu(
            MDWFHmcEvolveP<HaloDepth>(_momenta.getAccessor(), _gaugeForce->getAccessor(), stepsize));
        mdwfUpdateHalo(_momenta);
    }

    /*
     * Force-gradient kick (GPT update_p_force_gradient): with F the force of the level at U (already in _fgForce),
     * move U to exp(c F) U (exp(i c Pf) U, Pf = -i F the unit kick from zero), kick the momenta by stepsize with the
     * level's force there, and restore U. Version numbers are never reused, so no cache survives at the wrong U.
     */
    template<class Kick>
    void forceGradientKick(double stepsize, double c, Kick kick) {
        *_fgSavedGauge = _gauge;
        _gaugeVersion++;
        _gauge.iterateOverBulkAllMu(MDWFHmcDisplaceFG<HaloDepth>(_gauge.getAccessor(), _fgForce->getAccessor(), c));
        mdwfUpdateHalo(_gauge);
        kick(stepsize);
        _gauge = *_fgSavedGauge;
        mdwfUpdateHalo(_gauge);
        _gaugeVersion++;
    }

    void forceGradientKickLevel(size_t level, double stepsize, double c) {
        bool first = true;
        for (size_t t = 0; t < _param.term_level.size(); t++) {
            if (static_cast<size_t>(_param.term_level[t]) == level) {
                computeTermForce(t);
                if (first) {
                    *_fgForce = *_termForce[t];
                    first = false;
                } else {
                    _fgForce->iterateOverBulkAllMu(
                        MDWFHmcAddFields<HaloDepth>(_fgForce->getAccessor(), _termForce[t]->getAccessor()));
                }
            }
        }
        if (first) {
            return;   // empty level
        }
        forceGradientKick(stepsize, c, [&](double s) { kickLevel(level, s); });
    }

    void forceGradientKickGauge(double stepsize, double c) {
        computeGaugeForce();
        *_fgForce = *_gaugeForce;
        forceGradientKick(stepsize, c, [&](double s) { kickGauge(s); });
    }

    // n steps of size h of a scheme (MDWFHmcParameters::md_scheme) with momentum kick, inner evolution and
    // force-gradient kick; consecutive kicks at the same U reuse the cached forces.
    template<class Kick, class Inner, class FgKick>
    void schemeSteps(int scheme, int n, double h, Kick kick, Inner inner, FgKick fgKick) {
        const double l = _param.omf2_lambda;
        for (int k = 0; k < n; k++) {
            if (scheme == 0) {
                kick(0.5 * h);
                inner(h);
                kick(0.5 * h);
            } else if (scheme == 1) {
                kick(l * h);
                inner(0.5 * h);
                kick((1.0 - 2.0 * l) * h);
                inner(0.5 * h);
                kick(l * h);
            } else {
                kick(h / 6.0);
                inner(0.5 * h);
                fgKick(2.0 * h / 3.0, h * h / 24.0);
                inner(0.5 * h);
                kick(h / 6.0);
            }
        }
    }

    void evolveGaugeLevel(double dt) {
        const int m = std::max(1, _param.gauge_substeps);
        schemeSteps(gaugeScheme(), m, dt / static_cast<double>(m), [&](double s) { kickGauge(s); },
                    [&](double s) { evolveQ(s); }, [&](double s, double c) { forceGradientKickGauge(s, c); });
    }

    void evolveLevel(size_t level, double dt, int n) {
        const size_t levels = 1 + _param.level_substeps.size();
        schemeSteps(
            _param.md_scheme, n, dt / static_cast<double>(n), [&](double s) { kickLevel(level, s); },
            [&](double s) {
                if (level + 1 < levels) {
                    evolveLevel(level + 1, s, _param.level_substeps[level]);
                } else {
                    evolveGaugeLevel(s);
                }
            },
            [&](double s, double c) { forceGradientKickLevel(level, s, c); });
    }

public:

    int forceEvaluations() const {
        return _forceEvaluations;
    }

    int gaugeUpdates() const {
        return _gaugeUpdates;
    }

    // Accumulated wall-clock time of the fermion force evaluations (solves, host storage, transfers).
    double fermionForceSeconds() const {
        return _fermionForceSeconds;
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
        _fermionGauge.sync();
        _gaugeHost = _fermionGauge.get();
        _fermion.force(_ipdotHost, _gaugeHost);
        return forceRms(_ipdotHost);
    }

    /*
     * i sum_links tr(P_l ipdot_l) of the fermion force at the current gauge field and pseudofermions, with the
     * current momenta P: the derivative d S_f / dt along U(t) = exp(i t P) U predicted by the force (MD equations
     * U' = i P U, P' = -i ipdot conserve H only if this matches), for finite-difference checks.
     */
    double fermionForceAlongMomenta() {
        _fermionGauge.sync();
        _gaugeHost = _fermionGauge.get();
        _fermion.force(_ipdotHost, _gaugeHost);
        HostGauge momentaHost(_commBase, _prefix + "_fdmom_host");
        momentaHost = _momenta;
        const SU3Accessor<double> pAcc = momentaHost.getAccessor();
        const SU3Accessor<double, R18> fAcc = _ipdotHost.getAccessor();
        COMPLEX(double) sum(0.0, 0.0);
        for (size_t siteIndex = 0; siteIndex < GInd::getLatData().vol4; siteIndex++) {
            const gSite site = GInd::getSite(siteIndex);
            for (uint8_t mu = 0; mu < 4; mu++) {
                const gSiteMu siteMu = GInd::getSiteMu(site, mu);
                sum += tr_c(pAcc.getLink(siteMu), fAcc.getLink(siteMu));
            }
        }
        return real(COMPLEX(double)(0.0, 1.0) * _commBase.reduce(sum));
    }

    // The gauge field the fermion action sees (the thin field, or its antiperiodic copy).
    Gauge &fermionGauge() {
        _fermionGauge.sync();
        return _fermionGauge.get();
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
        auto start = std::chrono::steady_clock::now();
        heatbath();
        _heatbathSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const int forceEvaluationsBefore = _forceEvaluations;

        start = std::chrono::steady_clock::now();
        const MDWFHmcEnergy before = energy();
        _actionSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        integrate(_param.steps);
        start = std::chrono::steady_clock::now();
        const MDWFHmcEnergy after = energy();
        _actionSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const double deltaH = after.total() - before.total();

        bool accepted = true;
        if (metropolis) {
            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            accepted = deltaH <= 0.0 || uniform(acceptRng) < std::exp(-deltaH);
            _commBase.root2all(accepted);   // one decision even if Delta H differs in the last bits between ranks
            if (!accepted) {
                _gauge = _savedGauge;
                mdwfUpdateHalo(_gauge);
            }
        }
        return {before, after, deltaH, accepted, _forceEvaluations - forceEvaluationsBefore};
    }

    // Reversibility of the production MD (all terms, solver settings, chronological guesses): fresh momenta and
    // pseudofermions, integrate tau, flip P, integrate tau, flip P; compare links and momenta with the start (max
    // infnorm per link). Afterwards the gauge field is the start configuration again (exactly).
    MDWFHmcReversibilityResult reversibilityCheck() {
        _savedGauge = _gauge;
        refreshMomenta();
        heatbath();
        Gauge startMomenta(_commBase, _prefix + "_rev_start_momenta");
        startMomenta = _momenta;
        const int forceEvaluationsBefore = _forceEvaluations;
        MDWFHmcReversibilityResult result;
        result.start = energy();
        integrate(_param.steps);
        result.forward = energy();
        flipMomenta();
        integrate(_param.steps);
        flipMomenta();
        result.back = energy();
        result.max_link_diff = maxLinkDiff(_gauge, _savedGauge);
        result.max_momentum_diff = maxLinkDiff(_momenta, startMomenta);
        result.force_evaluations = _forceEvaluations - forceEvaluationsBefore;
        _gauge = _savedGauge;
        mdwfUpdateHalo(_gauge);
        return result;
    }

private:
    double maxLinkDiff(Gauge &a, Gauge &b) {
        HostGauge aHost(_commBase, _prefix + "_rev_diff_a");
        HostGauge bHost(_commBase, _prefix + "_rev_diff_b");
        aHost = a;
        bHost = b;
        const SU3Accessor<double, R18> aAcc = aHost.getAccessor();
        const SU3Accessor<double, R18> bAcc = bHost.getAccessor();
        double maxDiff = 0.0;
        for (size_t siteIndex = 0; siteIndex < GInd::getLatData().vol4; siteIndex++) {
            const gSite site = GInd::getSite(siteIndex);
            for (uint8_t mu = 0; mu < 4; mu++) {
                const gSiteMu siteMu = GInd::getSiteMu(site, mu);
                maxDiff = std::max(maxDiff, static_cast<double>(infnorm(aAcc.getLink(siteMu) - bAcc.getLink(siteMu))));
            }
        }
        return _commBase.reduceMax(maxDiff);
    }
};

template<size_t HaloDepth, size_t Ls>
using MDWFTwoFlavorHmc = MDWFHmcDriver<HaloDepth, Ls, MDWFBareTwoFlavorFermionAction<HaloDepth, Ls>>;

template<size_t HaloDepth, size_t Ls>
using MDWFPauliVillarsTwoFlavorHmc = MDWFHmcDriver<HaloDepth, Ls, MDWFPauliVillarsTwoFlavorFermionAction<HaloDepth, Ls>>;
