/*
 * hisqForce.cpp
 *
 * D. Bollweg
 *
 */

#include "hisqForce.h"
#include "staggeredPhasesKernel.h"

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, bool lvl1>
contribution_3link<floatT, onDevice, HaloDepth, comp, lvl1>::contribution_3link(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn,
                                                                                Gaugefield<floatT, onDevice, HaloDepth> &ForceIn)
    : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, bool lvl1>
__host__ __device__ SU3<floatT> contribution_3link<floatT, onDevice, HaloDepth, comp, lvl1>::operator()(gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);
    return threeLinkContribution<floatT, HaloDepth, comp>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _smParams);
}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
contribution_lepagelink<floatT, onDevice, HaloDepth, comp>::contribution_lepagelink(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn,
                                                                                    Gaugefield<floatT, onDevice, HaloDepth> &ForceIn)
    : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
__host__ __device__ SU3<floatT> contribution_lepagelink<floatT, onDevice, HaloDepth, comp>::operator()(gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);
    return lepagelinkContribution<floatT, HaloDepth, comp>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c_lp);
}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int Part, size_t term>
contribution_7link<floatT, onDevice, HaloDepth, comp, Part, term>::contribution_7link(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn,
                                                                                      Gaugefield<floatT, onDevice, HaloDepth> &ForceIn)
    : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int Part, size_t term>
__host__ __device__ SU3<floatT> contribution_7link<floatT, onDevice, HaloDepth, comp, Part, term>::operator()(gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);
    switch (Part) {
    case 1:
        return sevenLinkContribution_1_alt<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    case 2:
        return sevenLinkContribution_2_alt<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    case 3:
        return sevenLinkContribution_3_alt<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    case 4:
        return sevenLinkContribution_4_alt<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    case 5:
        return sevenLinkContribution_5_alt<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    case 6:
        return sevenLinkContribution_6_alt<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    case 7:
        return sevenLinkContribution_7_alt<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    default:
        return su3_zero<floatT>();
    }
}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part>
contribution_5link<floatT, onDevice, HaloDepth, comp, part>::contribution_5link(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn,
                                                                                Gaugefield<floatT, onDevice, HaloDepth> &ForceIn)
    : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part>
__host__ __device__ SU3<floatT> contribution_5link<floatT, onDevice, HaloDepth, comp, part>::operator()(gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);
    switch (part) {
    case 1:
        return fiveLinkContribution_11<floatT, HaloDepth, comp>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
    case 2:
        return fiveLinkContribution_12<floatT, HaloDepth, comp>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
    case 3:
        return fiveLinkContribution_13<floatT, HaloDepth, comp>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
    case 4:
        return fiveLinkContribution_14<floatT, HaloDepth, comp>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
    default:
        return su3_zero<floatT>();
    }
}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part, size_t term>
contribution_5link_large<floatT, onDevice, HaloDepth, comp, part, term>::contribution_5link_large(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn,
                                                                                                  Gaugefield<floatT, onDevice, HaloDepth> &ForceIn)
    : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part, size_t term>
__host__ __device__ SU3<floatT> contribution_5link_large<floatT, onDevice, HaloDepth, comp, part, term>::operator()(gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);
    switch (part) {
    case 5:
        return fiveLinkContribution_20<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
    case 6:
        return fiveLinkContribution_30<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
    default:
        return su3_zero<floatT>();
    }
}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
constructU3ProjForce<floatT, onDevice, HaloDepth, comp>::constructU3ProjForce(Gaugefield<floatT, onDevice, HaloDepth> &GaugeIn,
                                                                              Gaugefield<floatT, onDevice, HaloDepth, comp> &ForceIn)
    : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
__host__ __device__ SU3<floatT> constructU3ProjForce<floatT, onDevice, HaloDepth, comp>::operator()(gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);
    return derivativeProjectU3<floatT, HaloDepth>(_SU3Accessor, _forceAccessor, site, siteMu.mu);
}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
finalizeForce<floatT, onDevice, HaloDepth, comp>::finalizeForce(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn,
                                                                Gaugefield<floatT, onDevice, HaloDepth, comp> &ForceIn)
    : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
__host__ __device__ SU3<floatT> finalizeForce<floatT, onDevice, HaloDepth, comp>::operator()(gSiteMu siteMu) {
    SU3<floatT> tmp = floatT(2.0) * _SU3Accessor.getLink(siteMu) * _forceAccessor.getLink(siteMu);
    tmp.TA();

    return tmp;
}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
constructNaikDerivForce<floatT, onDevice, HaloDepth, comp>::constructNaikDerivForce(Gaugefield<floatT, onDevice, HaloDepth> &GaugeIn,
                                                                                    Gaugefield<floatT, onDevice, HaloDepth, comp> &ForceIn)
    : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
__host__ __device__ SU3<floatT> constructNaikDerivForce<floatT, onDevice, HaloDepth, comp>::operator()(gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);
    return naikLinkDerivative<floatT, HaloDepth>(_SU3Accessor, _forceAccessor, site, siteMu.mu);
}

template <class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks> struct multiplySimpleArraySpinor {

    SimpleArray<floatT, NStacks> _Arr;
    Vect3arrayAcc<floatT> _Spinor;
    multiplySimpleArraySpinor(SimpleArray<floatT, NStacks> &Arr, Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> &Spinor)
        : _Arr(Arr), _Spinor(Spinor.getAccessor()) {}

    __host__ __device__ void initialize(__attribute__((unused)) gSite &site) {}

    __host__ __device__ Vect3<floatT> operator()(gSite &site, size_t stack) {

        gSiteStack siteStack = GIndexer<LatticeLayout, HaloDepthSpin>::getSiteStack(site, stack);
        Vect3<floatT> tmp;
        tmp = _Spinor.getElement(siteStack);
        tmp = tmp * _Arr[stack];
        return tmp;
    }
};

template <class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, int steps, bool runTesting, const int rdeg>
tensor_product<floatT, onDevice, HaloDepth, HaloDepthSpin, steps, runTesting, rdeg>::tensor_product(Gaugefield<floatT, onDevice, HaloDepth> &gaugeIn,
                                                                                                    Vect3arrayAcc<floatT> x, Vect3arrayAcc<floatT> y,
                                                                                                    SimpleArray<floatT, rdeg> rat_num)
    : _x(x), _y(y), gAccessor(gaugeIn.getAccessor()), _rat_num(rat_num) {}

template <class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, int steps, bool runTesting, const int rdeg>
__host__ __device__ SU3<floatT> tensor_product<floatT, onDevice, HaloDepth, HaloDepthSpin, steps, runTesting, rdeg>::operator()(gSiteMu site) {
    typedef GIndexer<Even, HaloDepthSpin> GInd_even;
    typedef GIndexer<Odd, HaloDepthSpin> GInd_odd;

    floatT actConstant;
    if (steps == 1) {
        actConstant = 1. / 2.; // c1000
    } else if (steps == 3) {
        actConstant = -1. / 48.; // c3000
    }

    SU3<floatT> tmp = su3_zero<floatT>();
    for (int i = 0; i < rdeg; i++) {
        sitexyzt here = site.coord;

        bool oddness = (isOdd(here.x) ^ isOdd(here.y)) ^ (isOdd(here.z) ^ isOdd(here.t));

        if (!oddness) {
            gSiteStack even_site = GInd_even::getSiteStack(here.x, here.y, here.z, here.t, i);
            gSiteStack odd_site = GInd_even::template site_move<steps>(even_site, site.mu);

            tmp += _rat_num[i] * tensor_prod(_y.getElement(odd_site), conj(_x.getElement(even_site)));
        } else {
            gSiteStack odd_site = GInd_odd::getSiteStack(here.x, here.y, here.z, here.t, i);
            gSiteStack even_site = GInd_odd::template site_move<steps>(odd_site, site.mu);

            tmp -= _rat_num[i] * tensor_prod(_x.getElement(even_site), conj(_y.getElement(odd_site)));
        }
    }

    return actConstant * tmp;
}

template <class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, CompressionType comp, bool runTesting, const int rdeg>
HisqForce<floatT, onDevice, HaloDepth, HaloDepthSpin, comp, runTesting, rdeg>::HisqForce(
    Gaugefield<floatT, onDevice, HaloDepth, R18> &GaugeBase, Gaugefield<floatT, onDevice, HaloDepth, comp> &Force, AdvancedMultiShiftCG<floatT, rdeg> &cg,
    HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, 1> &dslash, HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, rdeg> &dslash_multi,
    RhmcParameters &rhmc_param, RationalCoeff &rat, HisqSmearing<floatT, onDevice, HaloDepth, R18, R18, R18, U3R14> &smearing)
    : _GaugeU3P(GaugeBase.getComm(), "SHARED_GAUGELVL2"), _GaugeLvl1(GaugeBase.getComm(), "SHARED_GAUGENAIK"), _TmpForce(GaugeBase.getComm()),
      _GaugeBase(GaugeBase), _Dummy(GaugeBase.getComm(), "SHARED_DUMMY"),

      _ForceNu(GaugeBase.getComm(), "HisqForceRecursiveScratch"),

      _spinor_x(GaugeBase.getComm()), _spinor_y(GaugeBase.getComm(), "SHARED_tmp"), _createF2(_GaugeLvl1, _TmpForce), _finalizeF3(_GaugeU3P, _TmpForce),
      _createNaikF1(_GaugeU3P, _TmpForce), F1_create_3Link(_GaugeU3P, Force), F1_lepagelink(_GaugeU3P, Force),
      _smearing(smearing), _cg(cg), _dslash(dslash), _dslash_multi(dslash_multi), _rhmc_param(rhmc_param), _rat(rat) {}

template <class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, CompressionType comp, bool runTesting, const int rdeg>
void HisqForce<floatT, onDevice, HaloDepth, HaloDepthSpin, comp, runTesting, rdeg>::make_f0(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &SpinorIn,
                                                                                            Gaugefield<floatT, onDevice, HaloDepth, comp> &Force,
                                                                                            Gaugefield<floatT, onDevice, HaloDepth, comp> &NaikForce,
                                                                                            bool isLight) {

    Force.iterateWithConst(su3_zero<floatT>());
    NaikForce.iterateWithConst(su3_zero<floatT>());
    SimpleArray<floatT, rdeg> shifts;
    SimpleArray<floatT, rdeg> rat_num;

    if (isLight) {
        shifts[0] = _rat.r_bar_lf_den[0] + _rhmc_param.m_ud() * _rhmc_param.m_ud();
        rat_num[0] = _rat.r_bar_lf_num[0];
        for (int i = 1; i < rdeg; i++) {
            shifts[i] = _rat.r_bar_lf_den[i] - _rat.r_bar_lf_den[0];
            rat_num[i] = _rat.r_bar_lf_num[i];
        }
    } else {
        shifts[0] = _rat.r_bar_sf_den[0] + _rhmc_param.m_s() * _rhmc_param.m_s();
        rat_num[0] = _rat.r_bar_sf_num[0];
        for (int i = 1; i < rdeg; i++) {
            shifts[i] = _rat.r_bar_sf_den[i] - _rat.r_bar_sf_den[0];
            rat_num[i] = _rat.r_bar_sf_num[i];
        }
    }

    _cg.invert(_dslash, _spinor_x, SpinorIn, shifts, _rhmc_param.cgMax(), _rhmc_param.residue_force());

    _dslash_multi.Dslash(_spinor_y, _spinor_x, true);

    Force.iterateOverBulkAllMu(
        tensor_product<floatT, onDevice, HaloDepth, HaloDepthSpin, 1, runTesting>(Force, _spinor_x.getAccessor(), _spinor_y.getAccessor(), rat_num));

    Force.updateAll();
    _TmpForce.iterateOverBulkAllMu(
        tensor_product<floatT, onDevice, HaloDepth, HaloDepthSpin, 3, runTesting>(NaikForce, _spinor_x.getAccessor(), _spinor_y.getAccessor(), rat_num));

    _TmpForce.updateAll();
    return;
}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h>
rho_middle_force<floatT, onDevice, HaloDepth, comp, nu_h, rho_h>::rho_middle_force(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn,
                                                                                   Gaugefield<floatT, onDevice, HaloDepth> &ForceNu)
    : _SU3Accessor(GaugeIn.getAccessor()), _forceNuAccessor(ForceNu.getAccessor()) {}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h>
__host__ __device__ SU3<floatT> rho_middle_force<floatT, onDevice, HaloDepth, comp, nu_h, rho_h>::operator()(gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;

    gSite site = GInd::getSite(siteMu.isite);
    int mu = siteMu.mu;

    int nu = (mu + nu_h) % 4;

    int rho = (((mu + nu) % 2) * ((40 * (mu + nu) - 6 * mu * nu - 18 * (mu * mu + nu * nu) + 2 * (mu * mu * mu + nu * nu * nu)) / 12 + rho_h) +
               ((mu + nu + 1) % 2) * (mu + 1 + 2 * rho_h)) %
              4;

    return rhoMiddleGather<floatT, HaloDepth, comp, R18>(_SU3Accessor, _forceNuAccessor, site, mu, rho);
}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h>
__host__ __device__ SU3<floatT> sigma_dressed_primal<floatT, onDevice, HaloDepth, comp, nu_h, rho_h>::operator()(gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;

    gSite site = GInd::getSite(siteMu.isite);

    int mu = siteMu.mu;

    int nu = (mu + nu_h) % 4;

    int rho = (((mu + nu) % 2) * ((40 * (mu + nu) - 6 * mu * nu - 18 * (mu * mu + nu * nu) + 2 * (mu * mu * mu + nu * nu * nu)) / 12 + rho_h) +
               ((mu + nu + 1) % 2) * (mu + 1 + 2 * rho_h)) %
              4;

    int sigma = 6 - mu - nu - rho;

    gSite site_p_sigma = GInd::site_up(site, sigma);

    gSite site_p_mu = GInd::site_up(site, mu);

    gSite site_m_sigma = GInd::site_dn(site, sigma);

    gSite site_m_sigma_p_mu = GInd::site_up(site_m_sigma, mu);

    SU3<floatT> pos = _SU3Accessor.getLink(GInd::getSiteMu(site, sigma)) * _SU3Accessor.getLink(GInd::getSiteMu(site_p_sigma, mu)) *
                      _SU3Accessor.getLinkDagger(GInd::getSiteMu(site_p_mu, sigma));

    SU3<floatT> neg = _SU3Accessor.getLinkDagger(GInd::getSiteMu(site_m_sigma, sigma)) * _SU3Accessor.getLink(GInd::getSiteMu(site_m_sigma, mu)) *
                      _SU3Accessor.getLink(GInd::getSiteMu(site_m_sigma_p_mu, sigma));

    return pos + neg;
}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h>
__host__ __device__ SU3<floatT> rho_side_sigma_dressed_force<floatT, onDevice, HaloDepth, comp, nu_h, rho_h>::operator()(gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;

    gSite site = GInd::getSite(siteMu.isite);

    // We are producing force on the rho side link.
    int targetRho = siteMu.mu;

    int centerMu;
    int outerNu;

    bool found = invertRhoDirection<nu_h, rho_h>(targetRho, centerMu, outerNu);

    if (!found)
        return su3_zero<floatT>();

    gSite upCenter = GInd::site_up(site, centerMu);

    gSite right = GInd::site_up(site, targetRho);

    gSite downCenter = GInd::site_dn(site, centerMu);

    gSite rightDn = GInd::site_up_dn(site, targetRho, centerMu);

    SU3<floatT> temp = _forceNuAccessor.getLinkDagger(GInd::getSiteMu(right, centerMu)) * _SU3Accessor.getLinkDagger(GInd::getSiteMu(upCenter, targetRho)) *
                       _sigmaAccessor.getLinkDagger(GInd::getSiteMu(site, centerMu));

    temp += _sigmaAccessor.getLink(GInd::getSiteMu(right, centerMu)) * _SU3Accessor.getLinkDagger(GInd::getSiteMu(upCenter, targetRho)) *
            _forceNuAccessor.getLink(GInd::getSiteMu(site, centerMu));

    temp += _forceNuAccessor.getLink(GInd::getSiteMu(rightDn, centerMu)) * _SU3Accessor.getLinkDagger(GInd::getSiteMu(downCenter, targetRho)) *
            _sigmaAccessor.getLink(GInd::getSiteMu(downCenter, centerMu));

    temp += _sigmaAccessor.getLinkDagger(GInd::getSiteMu(rightDn, centerMu)) * _SU3Accessor.getLinkDagger(GInd::getSiteMu(downCenter, targetRho)) *
            _forceNuAccessor.getLinkDagger(GInd::getSiteMu(downCenter, centerMu));

    return temp;
}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h>
rho_sigma_dressed_primal<floatT, onDevice, HaloDepth, comp, nu_h, rho_h>::rho_sigma_dressed_primal(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn,
                                                                                                   Gaugefield<floatT, onDevice, HaloDepth> &SigmaPrimal)
    : _SU3Accessor(GaugeIn.getAccessor()), _sigmaAccessor(SigmaPrimal.getAccessor()) {}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h>
__host__ __device__ SU3<floatT> rho_sigma_dressed_primal<floatT, onDevice, HaloDepth, comp, nu_h, rho_h>::operator()(gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;

    gSite site = GInd::getSite(siteMu.isite);

    int mu = siteMu.mu;

    int nu = (mu + nu_h) % 4;

    int rho = (((mu + nu) % 2) * ((40 * (mu + nu) - 6 * mu * nu - 18 * (mu * mu + nu * nu) + 2 * (mu * mu * mu + nu * nu * nu)) / 12 + rho_h) +
               ((mu + nu + 1) % 2) * (mu + 1 + 2 * rho_h)) %
              4;

    // ------------------------------------------------------------
    // Positive rho dressing:
    //
    // U_rho(x)
    // X_sigma_mu(x+rho)
    // U_rho^\dagger(x+mu)
    // ------------------------------------------------------------

    gSite upRho = GInd::site_up(site, rho);

    gSite upMu = GInd::site_up(site, mu);

    SU3<floatT> positive = _SU3Accessor.getLink(GInd::getSiteMu(site, rho))

                           * _sigmaAccessor.getLink(GInd::getSiteMu(upRho, mu))

                           * _SU3Accessor.getLinkDagger(GInd::getSiteMu(upMu, rho));

    // ------------------------------------------------------------
    // Negative rho dressing:
    //
    // U_rho^\dagger(x-rho)
    // X_sigma_mu(x-rho)
    // U_rho(x-rho+mu)
    // ------------------------------------------------------------

    gSite dnRho = GInd::site_dn(site, rho);

    gSite dnRhoUpMu = GInd::site_up(dnRho, mu);

    SU3<floatT> negative = _SU3Accessor.getLinkDagger(GInd::getSiteMu(dnRho, rho))

                           * _sigmaAccessor.getLink(GInd::getSiteMu(dnRho, mu))

                           * _SU3Accessor.getLink(GInd::getSiteMu(dnRhoUpMu, rho));

    return positive + negative;
}

template <class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, CompressionType comp, bool runTesting, const int rdeg>
void HisqForce<floatT, onDevice, HaloDepth, HaloDepthSpin, comp, runTesting, rdeg>::constructF1(
    Gaugefield<floatT, onDevice, HaloDepth, comp> &Force) {

    const SmearingParameters<floatT> smParams = getLevel2Params<floatT>();

    // D3 starts in the shared dummy. Stream each middle branch into that
    // accumulator, so three simultaneously live ForceNu fields are unnecessary.
    _Dummy.template iterateOverBulkAllMu<64>(
        recursive_three_link_base_force<floatT, onDevice, HaloDepth, R18>(
            _GaugeU3P, Force));

    static_for<1, 4>::apply([&](auto nu_h) {
        _Dummy.template iterateOverBulkAllMu<64>(
            make_accumulate_scaled_force(
                _Dummy,
                outer_nu_middle_force<floatT, onDevice, HaloDepth, R18, nu_h>(
                    _GaugeU3P, Force),
                smParams._c_3));
    });

    // Preserve the Naik derivative before _TmpForce stops being the pre-F1
    // Naik source. Then make _TmpForce the physical F1 accumulator.
    _ForceNu.template iterateOverBulkAllMu<64>(_createNaikF1);
    _TmpForce = _Dummy + _ForceNu;

    // Fuse the D5 and signed-D7 reverse paths. Within each (nu_h, rho_h)
    // branch, both derivatives use the same
    //
    //   F_N = outer_nu_middle_force(F)
    //   F_R = rho_middle_force(F_N).
    //
    // _ForceNu first holds F_N. _Dummy first holds F_R, then X_sigma.
    // Only after all users of F_N are finished is _ForceNu overwritten by
    // the combined outer-nu primal field.
    static_for<1, 4>::apply([&](auto nu_h) {
        static_for<0, 2>::apply([&](auto rho_h) {
            _ForceNu.template iterateOverBulkAllMu<64>(
                outer_nu_middle_force<floatT, onDevice, HaloDepth, R18, nu_h>(
                    _GaugeU3P, Force));
            _ForceNu.updateAll();

            _Dummy.template iterateOverBulkAllMu<64>(
                rho_middle_force<floatT, onDevice, HaloDepth, R18, nu_h, rho_h>(
                    _GaugeU3P, _ForceNu));
            _Dummy.updateAll();

            // D5 middle contribution. The same F_R below drives both
            // unchanged D7 sigma contributions.
            _TmpForce.template iterateOverBulkAllMu<64>(
                make_accumulate_scaled_force(
                    _TmpForce,
                    _Dummy.getAccessor(),
                    smParams._c_5));

            // Unchanged D7 sigma-middle and sigma-side contributions.
            _TmpForce.template iterateOverBulkAllMu<64>(
                make_accumulate_scaled_force(
                    _TmpForce,
                    sigma_middle_force<floatT, onDevice, HaloDepth, R18, nu_h, rho_h>(
                        _GaugeU3P, _Dummy),
                    smParams._c_7));

            _TmpForce.template iterateOverBulkAllMu<64>(
                make_accumulate_scaled_force(
                    _TmpForce,
                    sigma_side_force<floatT, onDevice, HaloDepth, R18, nu_h, rho_h>(
                        _GaugeU3P, _Dummy),
                    smParams._c_7));

            _Dummy.template iterateOverBulkAllMu<64>(
                sigma_dressed_primal<floatT, onDevice, HaloDepth, R18, nu_h, rho_h>(
                    _GaugeU3P));
            _Dummy.updateAll();

            // Fuse the rho-side kernels:
            //
            //   c5 * rho_side[U, F_N]
            //     + c7 * rho_side[X_sigma, F_N].
            _TmpForce.template iterateOverBulkAllMu<64>(
                make_accumulate_scaled_force(
                    _TmpForce,
                    combined_rho_side_force<floatT, onDevice, HaloDepth, R18, nu_h, rho_h>(
                        _GaugeU3P, _Dummy, _ForceNu,
                        smParams._c_5, smParams._c_7),
                    static_cast<floatT>(1)));

            // By linearity of D_rho, build one outer-nu primal field:
            //
            //   D_rho[c5 * U - c7 * X_sigma]
            //     = c5 * D_rho[U] - c7 * D_rho[X_sigma].
            //
            // The minus preserves the signed-D7 positions 1+7 convention.
            _ForceNu.template iterateOverBulkAllMu<64>(
                combined_rho_dressed_primal<floatT, onDevice, HaloDepth, R18, nu_h, rho_h>(
                    _GaugeU3P, _Dummy,
                    smParams._c_5, smParams._c_7));
            _ForceNu.updateAll();

            // One outer-nu-side gather now supplies the D5 and signed-D7
            // positions 1+7 contributions together.
            _TmpForce.template iterateOverBulkAllMu<64>(
                make_accumulate_scaled_force(
                    _TmpForce,
                    outer_nu_side_dressed_force<floatT, onDevice, HaloDepth, R18, nu_h, rho_h>(
                        _GaugeU3P, _ForceNu, Force),
                    static_cast<floatT>(1)));
        });
    });

    // Unchanged production Lepage contribution.
    _Dummy.iterateOverBulkAllMu(F1_lepagelink);
    _TmpForce = _TmpForce + _Dummy;
    _TmpForce.updateAll();
}

template <class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, CompressionType comp, bool runTesting, const int rdeg>
void HisqForce<floatT, onDevice, HaloDepth, HaloDepthSpin, comp, runTesting, rdeg>::constructF3Recursive(
    Gaugefield<floatT, onDevice, HaloDepth, comp> &Force,
    Gaugefield<floatT, onDevice, HaloDepth, comp> &ForceOut) {

    const SmearingParameters<floatT> smParams = getLevel1Params<floatT>();

    _Dummy.template iterateOverBulkAllMu<64>(
        recursive_three_link_base_force<floatT, onDevice, HaloDepth, R18, true>(
            _GaugeU3P, Force));

    static_for<1, 4>::apply([&](auto nu_h) {
        _Dummy.template iterateOverBulkAllMu<64>(
            make_accumulate_scaled_force(
                _Dummy,
                outer_nu_middle_force<floatT, onDevice, HaloDepth, R18, nu_h>(
                    _GaugeU3P, Force),
                smParams._c_3));
    });

    ForceOut = _Dummy;

    static_for<1, 4>::apply([&](auto nu_h) {
        _ForceNu.template iterateOverBulkAllMu<64>(
            outer_nu_middle_force<floatT, onDevice, HaloDepth, R18, nu_h>(
                _GaugeU3P, Force));
        _ForceNu.updateAll();

        static_for<0, 2>::apply([&](auto rho_h) {
            ForceOut.template iterateOverBulkAllMu<64>(
                make_accumulate_scaled_force(
                    ForceOut,
                    rho_middle_force<floatT, onDevice, HaloDepth, R18, nu_h, rho_h>(
                        _GaugeU3P, _ForceNu),
                    smParams._c_5));

            ForceOut.template iterateOverBulkAllMu<64>(
                make_accumulate_scaled_force(
                    ForceOut,
                    rho_side_force<floatT, onDevice, HaloDepth, R18, nu_h, rho_h>(
                        _GaugeU3P, _ForceNu),
                    smParams._c_5));

            _Dummy.template iterateOverBulkAllMu<64>(
                rho_dressed_primal<floatT, onDevice, HaloDepth, R18, nu_h, rho_h>(
                    _GaugeU3P));
            _Dummy.updateAll();

            ForceOut.template iterateOverBulkAllMu<64>(
                make_accumulate_scaled_force(
                    ForceOut,
                    outer_nu_side_dressed_force<floatT, onDevice, HaloDepth, R18, nu_h, rho_h>(
                        _GaugeU3P, _Dummy, Force),
                    smParams._c_5));
        });
    });

    static_for<1, 4>::apply([&](auto nu_h) {
        static_for<0, 2>::apply([&](auto rho_h) {
            _ForceNu.template iterateOverBulkAllMu<64>(
                outer_nu_middle_force<floatT, onDevice, HaloDepth, R18, nu_h>(
                    _GaugeU3P, Force));
            _ForceNu.updateAll();

            _Dummy.template iterateOverBulkAllMu<64>(
                rho_middle_force<floatT, onDevice, HaloDepth, R18, nu_h, rho_h>(
                    _GaugeU3P, _ForceNu));
            _Dummy.updateAll();

            ForceOut.template iterateOverBulkAllMu<64>(
                make_accumulate_scaled_force(
                    ForceOut,
                    sigma_middle_force<floatT, onDevice, HaloDepth, R18, nu_h, rho_h>(
                        _GaugeU3P, _Dummy),
                    smParams._c_7));

            ForceOut.template iterateOverBulkAllMu<64>(
                make_accumulate_scaled_force(
                    ForceOut,
                    sigma_side_force<floatT, onDevice, HaloDepth, R18, nu_h, rho_h>(
                        _GaugeU3P, _Dummy),
                    smParams._c_7));

            _Dummy.template iterateOverBulkAllMu<64>(
                sigma_dressed_primal<floatT, onDevice, HaloDepth, R18, nu_h, rho_h>(
                    _GaugeU3P));
            _Dummy.updateAll();

            ForceOut.template iterateOverBulkAllMu<64>(
                make_accumulate_scaled_force(
                    ForceOut,
                    rho_side_sigma_dressed_force<floatT, onDevice, HaloDepth, R18, nu_h, rho_h>(
                        _GaugeU3P, _Dummy, _ForceNu),
                    smParams._c_7));

            _ForceNu.template iterateOverBulkAllMu<64>(
                rho_sigma_dressed_primal<floatT, onDevice, HaloDepth, R18, nu_h, rho_h>(
                    _GaugeU3P, _Dummy));
            _ForceNu.updateAll();

            ForceOut.template iterateOverBulkAllMu<64>(
                make_accumulate_scaled_force(
                    ForceOut,
                    outer_nu_side_dressed_force<floatT, onDevice, HaloDepth, R18, nu_h, rho_h>(
                        _GaugeU3P, _ForceNu, Force),
                    -smParams._c_7));
        });
    });
}

template <class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, CompressionType comp, bool runTesting, const int rdeg>
void HisqForce<floatT, onDevice, HaloDepth, HaloDepthSpin, comp, runTesting, rdeg>::TestForce(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &SpinorIn,
                                                                                              Gaugefield<floatT, onDevice, HaloDepth, comp> &Force,
                                                                                              grnd_state<true> &d_rand) {

    if (!runTesting) {
        rootLogger.error("Calling member function TestForce should only be used when the template parameter runTesting is set to true!");
    }

    Force.iterateWithConst(su3_zero<floatT>());
    _TmpForce.iterateWithConst(su3_zero<floatT>());

    Spinorfield<floatT, onDevice, Even, HaloDepthSpin> temp(SpinorIn.getComm());
    for (int i = 0; i < rdeg; i++) {
        temp.gauss(d_rand.state);
        _spinor_x.copyFromStackToStack(temp, i, 0);
    }

    SimpleArray<floatT, rdeg> rat_num;

    for (int i = 0; i < rdeg; i++) {
        rat_num[i] = _rat.r_inv_lf_num[i];
    }

    _dslash_multi.Dslash(_spinor_y, _spinor_x);
    Force.iterateOverBulkAllMu(
        tensor_product<floatT, onDevice, HaloDepth, HaloDepthSpin, 1, runTesting>(Force, _spinor_x.getAccessor(), _spinor_y.getAccessor(), rat_num));
    _TmpForce.iterateOverBulkAllMu(
        tensor_product<floatT, onDevice, HaloDepth, HaloDepthSpin, 3, runTesting>(_TmpForce, _spinor_x.getAccessor(), _spinor_y.getAccessor(), rat_num));

    Force.updateAll();
    _TmpForce.updateAll();

    _smearing.template SmearLvl1<R18>(_GaugeLvl1);
    _smearing.template ProjectU3<R18, R18>(_GaugeLvl1, _GaugeU3P);

    staggeredPhaseKernel<floatT, onDevice, HaloDepth, R18> multPhase(_GaugeU3P, _rhmc_param.mu_f());
    _GaugeU3P.iterateOverBulkAllMu(multPhase);
    _GaugeU3P.updateAll();

    constructF1(Force);

    // ============================================================
    // F1 -> F2 : derivative of U(3) projection
    // ============================================================

    staggeredPhaseKernel<floatT, onDevice, HaloDepth, R18> multPhaselv1(_GaugeLvl1, _rhmc_param.mu_f());

    _GaugeLvl1.iterateOverBulkAllMu(multPhaselv1);

    Force.iterateOverBulkAllMu(_createF2);
    Force.updateAll();

    // ============================================================
    // Prepare thin links for F3
    // ============================================================

    staggeredPhaseKernel<floatT, onDevice, HaloDepth, R18> multPhaseB(_GaugeBase, _rhmc_param.mu_f());

    _GaugeU3P.iterateOverBulkAllMu(multPhaseB);

    constructF3Recursive(Force, _TmpForce);

    // ============================================================
    // Final thin-link force
    // ============================================================

    Force.iterateOverBulkAllMu(_finalizeF3);

    return;
}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h>
__host__ __device__ SU3<floatT> sigma_middle_force<floatT, onDevice, HaloDepth, comp, nu_h, rho_h>::operator()(gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;

    gSite site = GInd::getSite(siteMu.isite);

    int mu = siteMu.mu;

    int nu = (mu + nu_h) % 4;

    int rho = (((mu + nu) % 2) * ((40 * (mu + nu) - 6 * mu * nu - 18 * (mu * mu + nu * nu) + 2 * (mu * mu * mu + nu * nu * nu)) / 12 + rho_h) +
               ((mu + nu + 1) % 2) * (mu + 1 + 2 * rho_h)) %
              4;

    int sigma = 6 - mu - nu - rho;

    return rhoMiddleGather<floatT, HaloDepth, comp, R18>(_SU3Accessor, _forceRAccessor, site, mu, sigma);
}

template <class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, CompressionType comp, bool runTesting, const int rdeg>
void HisqForce<floatT, onDevice, HaloDepth, HaloDepthSpin, comp, runTesting, rdeg>::updateForce(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &SpinorIn,
                                                                                                Gaugefield<floatT, onDevice, HaloDepth, comp> &Force,
                                                                                                bool isLight) {

    make_f0(SpinorIn, Force, _TmpForce, isLight);

    // Level 1 smearing.
    _smearing.template SmearLvl1<R18>(_GaugeLvl1);

    // U3 projection.
    _smearing.template ProjectU3<R18, R18>(_GaugeLvl1, _GaugeU3P);

    staggeredPhaseKernel<floatT, onDevice, HaloDepth, R18> multPhase(_GaugeU3P, _rhmc_param.mu_f());
    _GaugeU3P.iterateOverBulkAllMu(multPhase);
    _GaugeU3P.updateAll();

    constructF1(Force);

    staggeredPhaseKernel<floatT, onDevice, HaloDepth, R18> multPhaselv1(_GaugeLvl1, _rhmc_param.mu_f());
    _GaugeLvl1.iterateOverBulkAllMu(multPhaselv1);

    Force.iterateOverBulkAllMu(_createF2);
    Force.updateAll();

    staggeredPhaseKernel<floatT, onDevice, HaloDepth, R18> multPhaseB(_GaugeBase, _rhmc_param.mu_f());
    _GaugeU3P.iterateOverBulkAllMu(multPhaseB); // reuse U3P Field here

    _GaugeU3P.updateAll();

    constructF3Recursive(Force, _TmpForce);

    Force.iterateOverBulkAllMu(_finalizeF3);

    return;
}

#define HFORCE_INIT(floatT, HALO, HALOSPIN)                                                                                                                    \
    template class HisqForce<floatT, true, HALO, HALOSPIN, R18, false>;                                                                                        \
    template class HisqForce<floatT, true, HALO, HALOSPIN, R18, true>;
INIT_PHHS(HFORCE_INIT)
