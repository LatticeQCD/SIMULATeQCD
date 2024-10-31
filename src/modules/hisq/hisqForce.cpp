/*
 * hisqForce.cu
 *
 * D. Bollweg
 *
 */

#include "hisqForce.h"
#include "staggeredPhasesKernel.h"

/// Method for printing results, only needed for testing.
template<class floatT, size_t HaloDepth, CompressionType comp>
void printResult(Gaugefield<floatT, true, HaloDepth, comp> &g_dev) {
    typedef GIndexer<All, HaloDepth> GInd;
    Gaugefield<floatT, false, HaloDepth, comp> g_host(g_dev.getComm());
    g_host = g_dev;
    gSite site = GInd::getSite(0,0,0,0);
    SU3<floatT> mat = g_host.getAccessor().getLink(GInd::getSiteMu(site,3));
    rootLogger.info(mat.getLink00() ,  mat.getLink01() ,  mat.getLink02());
    rootLogger.info(mat.getLink10() ,  mat.getLink11() ,  mat.getLink12());
    rootLogger.info(mat.getLink20() ,  mat.getLink21() ,  mat.getLink22());
    rootLogger.info(" ");
    site = GInd::getSite(0,0,0,1);
    mat = g_host.getAccessor().getLink(GInd::getSiteMu(site,3));
    rootLogger.info(mat.getLink00() ,  mat.getLink01() ,  mat.getLink02());
    rootLogger.info(mat.getLink10() ,  mat.getLink11() ,  mat.getLink12());
    rootLogger.info(mat.getLink20() ,  mat.getLink21() ,  mat.getLink22());
    rootLogger.info(" ");
    site = GInd::getSite(0,0,0,2);
    mat = g_host.getAccessor().getLink(GInd::getSiteMu(site,3));
    rootLogger.info(mat.getLink00() ,  mat.getLink01() ,  mat.getLink02());
    rootLogger.info(mat.getLink10() ,  mat.getLink11() ,  mat.getLink12());
    rootLogger.info(mat.getLink20() ,  mat.getLink21() ,  mat.getLink22());
    rootLogger.info(" ");
    site = GInd::getSite(0,0,0,3);
    mat = g_host.getAccessor().getLink(GInd::getSiteMu(site,3));
    rootLogger.info(mat.getLink00() ,  mat.getLink01() ,  mat.getLink02());
    rootLogger.info(mat.getLink10() ,  mat.getLink11() ,  mat.getLink12());
    rootLogger.info(mat.getLink20() ,  mat.getLink21() ,  mat.getLink22());
    rootLogger.info(" ");
    site = GInd::getSite(0,0,1,1);
    mat = g_host.getAccessor().getLink(GInd::getSiteMu(site,1));
    rootLogger.info(mat.getLink00() ,  mat.getLink01() ,  mat.getLink02());
    rootLogger.info(mat.getLink10() ,  mat.getLink11() ,  mat.getLink12());
    rootLogger.info(mat.getLink20() ,  mat.getLink21() ,  mat.getLink22());
    rootLogger.info(" ");
    return;
}


// 3-link derivative
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, bool lvl1>
contribution_3link<floatT, onDevice, HaloDepth, comp, lvl1>::contribution_3link(
    Gaugefield<floatT, onDevice,HaloDepth,comp> &GaugeIn, 
    Gaugefield<floatT, onDevice, HaloDepth> &ForceIn) : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, bool lvl1>
__host__ __device__ SU3<floatT> contribution_3link<floatT, onDevice, HaloDepth, comp, lvl1>::operator() (gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);
    return threeLinkContribution<floatT, HaloDepth, comp>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _smParams);
}


// LePage derivative
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
contribution_lepagelink<floatT, onDevice, HaloDepth, comp>::contribution_lepagelink(
    Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, 
    Gaugefield<floatT, onDevice, HaloDepth> &ForceIn) : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
__host__ __device__ SU3<floatT> contribution_lepagelink<floatT, onDevice, HaloDepth, comp>::operator() (gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);
    return lepagelinkContribution<floatT, HaloDepth, comp> (_SU3Accessor, _forceAccessor, site, siteMu.mu, _c_lp);
}


// 7-link derivative
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int Part, size_t term>
contribution_7link<floatT, onDevice, HaloDepth, comp, Part, term>::contribution_7link(
    Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, 
    Gaugefield<floatT, onDevice, HaloDepth> &ForceIn) : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int Part, size_t term>
__host__ __device__ SU3<floatT> contribution_7link<floatT, onDevice, HaloDepth, comp, Part, term>::operator() (gSiteMu siteMu) {
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


// 5-link derivative
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part>
contribution_5link<floatT, onDevice, HaloDepth, comp, part>::contribution_5link(
    Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, 
    Gaugefield<floatT, onDevice, HaloDepth> &ForceIn) : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part>
__host__ __device__ SU3<floatT> contribution_5link<floatT, onDevice, HaloDepth, comp, part>::operator() (gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);
    switch (part) {
    case 1: return fiveLinkContribution_11<floatT, HaloDepth, comp>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
    case 2: return fiveLinkContribution_12<floatT, HaloDepth, comp>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
    case 3: return fiveLinkContribution_13<floatT, HaloDepth, comp>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
    case 4: return fiveLinkContribution_14<floatT, HaloDepth, comp>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
    default: return su3_zero<floatT>();
    }
}


// 5-link large derivative
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part, size_t term>
contribution_5link_large<floatT, onDevice, HaloDepth, comp, part, term>::contribution_5link_large(
    Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, 
    Gaugefield<floatT, onDevice, HaloDepth> &ForceIn) : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part, size_t term>
__host__ __device__ SU3<floatT> contribution_5link_large<floatT, onDevice, HaloDepth, comp, part, term>::operator() (gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);
    switch (part) {
        case 5: return fiveLinkContribution_20<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
        case 6: return fiveLinkContribution_30<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c5);
    default: return su3_zero<floatT>();
    }
}


// derivative of U3 projection
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
constructU3ProjForce<floatT, onDevice, HaloDepth, comp>::constructU3ProjForce(
    Gaugefield<floatT, onDevice, HaloDepth> &GaugeIn, 
    Gaugefield<floatT, onDevice, HaloDepth, comp> &ForceIn) : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
__host__ __device__ SU3<floatT> constructU3ProjForce<floatT, onDevice, HaloDepth, comp>::operator()(gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);
    return derivativeProjectU3<floatT, HaloDepth>(_SU3Accessor, _forceAccessor, site, siteMu.mu);
}


// traceless, antihermitian part of 
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
finalizeForce<floatT, onDevice, HaloDepth, comp>::finalizeForce(
    Gaugefield<floatT, onDevice, HaloDepth,comp> &GaugeIn, 
    Gaugefield<floatT, onDevice, HaloDepth, comp> &ForceIn) : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
__host__ __device__ SU3<floatT> finalizeForce<floatT, onDevice, HaloDepth, comp>::operator()(gSiteMu siteMu) {
    SU3<floatT> tmp = floatT(2.0)*_SU3Accessor.getLink(siteMu)*_forceAccessor.getLink(siteMu);
    tmp.TA();
    return tmp;
}


// Naik derivative
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
constructNaikDerivForce<floatT, onDevice, HaloDepth, comp>::constructNaikDerivForce(
    Gaugefield<floatT, onDevice, HaloDepth> &GaugeIn, 
    Gaugefield<floatT, onDevice, HaloDepth, comp> &ForceIn) : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
__host__ __device__ SU3<floatT> constructNaikDerivForce<floatT, onDevice, HaloDepth, comp>::operator()(gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);
    return naikLinkDerivative<floatT, HaloDepth>(_SU3Accessor,_forceAccessor,site, siteMu.mu);
}


template<class floatT,bool onDevice, Layout LatticeLayout, size_t HaloDepthSpin, size_t NStacks>
struct multiplySimpleArraySpinor {

    SimpleArray<floatT, NStacks> _Arr;
    Vect3arrayAcc<floatT> _Spinor;
    multiplySimpleArraySpinor(SimpleArray<floatT, NStacks>& Arr, Spinorfield<floatT,onDevice,LatticeLayout,HaloDepthSpin,NStacks>& Spinor) : _Arr(Arr), _Spinor(Spinor.getAccessor()) {}

    __host__ __device__ void initialize(__attribute__((unused)) gSite& site) {
    }

    __host__ __device__ Vect3<floatT> operator()(gSite& site, size_t stack) {

        gSiteStack siteStack = GIndexer<LatticeLayout,HaloDepthSpin>::getSiteStack(site,stack);
        Vect3<floatT> tmp;
        tmp = _Spinor.getElement(siteStack);
        tmp = tmp * _Arr[stack];
        return tmp;
    }
};


// ~ alpha_l |X><Y| (order depends on site parity) 
template<class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, int steps, bool runTesting, const int rdeg>
tensor_product<floatT, onDevice, HaloDepth, HaloDepthSpin, steps, runTesting,rdeg>::tensor_product(
    Gaugefield<floatT, onDevice,  HaloDepth> &gaugeIn,
    Vect3arrayAcc<floatT> x, 
    Vect3arrayAcc<floatT> y,
    SimpleArray<floatT, rdeg> rat_num)
    : _x(x), _y(y), gAccessor(gaugeIn.getAccessor()), _rat_num(rat_num) {}

template<class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, int steps, bool runTesting, const int rdeg>
__host__ __device__ SU3<floatT> tensor_product<floatT, onDevice, HaloDepth, HaloDepthSpin, steps, runTesting,rdeg>::operator()(gSiteMu site) {
    typedef GIndexer<Even,HaloDepthSpin> GInd_even;
    typedef GIndexer<Odd, HaloDepthSpin> GInd_odd;

    floatT actConstant;
    if (steps == 1) {
        actConstant = 1./2.; //c1000
    } else if (steps == 3) {
        actConstant = -1./48.; //c3000
    }

    SU3<floatT> tmp = su3_zero<floatT>();
    for (int i = 0; i < rdeg; i++) {
        sitexyzt here =  site.coord;

        // C++ XOR operator is ^. The boolean variable `oddness` will be `true` if the total number of 
        // `true` values among `isOdd(here.x)`, `isOdd(here.y)`, `isOdd(here.z)`, and `isOdd(here.t)` 
        // is odd, and `false` otherwise.
        bool oddness = (isOdd(here.x) ^ isOdd(here.y)) ^ (isOdd(here.z) ^ isOdd(here.t));

        if (!oddness) {
            gSiteStack even_site = GInd_even::getSiteStack(here.x, here.y, here.z, here.t,i);
            gSiteStack odd_site  = GInd_even::template site_move<steps>(even_site,site.mu);

            tmp += _rat_num[i]*tensor_prod(_y.getElement(odd_site),conj(_x.getElement(even_site)));

        } else {
            gSiteStack odd_site  = GInd_odd::getSiteStack(here.x, here.y, here.z, here.t,i);
            gSiteStack even_site = GInd_odd::template site_move<steps>(odd_site,site.mu);

            tmp -= _rat_num[i]*tensor_prod(_x.getElement(even_site), conj(_y.getElement(odd_site)));
        }
    }
    return actConstant*tmp;
}


template<class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin,  CompressionType comp, bool runTesting, const int rdeg>
HisqForce<floatT, onDevice, HaloDepth, HaloDepthSpin, comp, runTesting, rdeg>::HisqForce(
    Gaugefield<floatT, onDevice,  HaloDepth,R18> &GaugeBase,
    Gaugefield<floatT, onDevice,  HaloDepth, comp> &Force,
    AdvancedMultiShiftCG<floatT, rdeg> &cg,
    HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, 1> &dslash,
    HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, rdeg> &dslash_multi,
    RhmcParameters &rhmc_param,
    RationalCoeff &rat,
    HisqSmearing<floatT, onDevice, HaloDepth, R18, R18, R18, U3R14> &smearing)
    : _GaugeU3P(GaugeBase.getComm(),"SHARED_GAUGELVL2"),
      _GaugeLvl1(GaugeBase.getComm(),"SHARED_GAUGENAIK"),
      _TmpForce(GaugeBase.getComm()),
      _GaugeBase(GaugeBase),
      _Dummy(GaugeBase.getComm(), "SHARED_DUMMY"),
      _spinor_x(GaugeBase.getComm()),
      _spinor_y(GaugeBase.getComm(), "SHARED_tmp"),
      _createF2(_GaugeLvl1,_TmpForce),
      _finalizeF3(_GaugeU3P,_TmpForce),
      _createNaikF1(_GaugeU3P,_TmpForce),
      F1_create_3Link(_GaugeU3P,Force),
      F1_lepagelink(_GaugeU3P, Force),
      F3_create_3Link(_GaugeU3P,Force),
      _smearing(smearing),
      _cg(cg),
      _dslash(dslash),
      _dslash_multi(dslash_multi),
      _rhmc_param(rhmc_param),
      _rat(rat)  {}


template<class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, CompressionType comp, bool runTesting, const int rdeg>
void HisqForce<floatT,onDevice, HaloDepth, HaloDepthSpin, comp, runTesting,rdeg>::make_f0(
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &SpinorIn, // |PHI>
    Gaugefield<floatT, onDevice, HaloDepth, comp> &Force,
    Gaugefield<floatT, onDevice, HaloDepth, comp> &NaikForce,
    bool isLight)
{
    Force.iterateWithConst(su3_zero<floatT>());
    NaikForce.iterateWithConst(su3_zero<floatT>());
    SimpleArray<floatT, rdeg> shifts;
    SimpleArray<floatT, rdeg> rat_num;

    if (isLight) {
        shifts[0]  = _rat.r_bar_lf_den[0] + _rhmc_param.m_ud()*_rhmc_param.m_ud();
        rat_num[0] = _rat.r_bar_lf_num[0];
        for (int i = 1; i < rdeg; i++) {
            shifts[i]  = _rat.r_bar_lf_den[i] -_rat.r_bar_lf_den[0];
            rat_num[i] = _rat.r_bar_lf_num[i];
        }
    } else {
        shifts[0]  = _rat.r_bar_sf_den[0] + _rhmc_param.m_s()*_rhmc_param.m_s();
        rat_num[0] = _rat.r_bar_sf_num[0];
        for (int i = 1; i < rdeg; i++) {
            shifts[i]  = _rat.r_bar_sf_den[i] -_rat.r_bar_sf_den[0];
            rat_num[i] = _rat.r_bar_sf_num[i];
        }
    }

    // |X> ~ (D^dagger D)^-1 |PHI>. Here D is the massless Dirac operator.
    _cg.invert(_dslash,_spinor_x,SpinorIn,shifts,_rhmc_param.cgMax(),_rhmc_param.residue_force());

    // |Y> = D |X>. Again, massless DSlash.
    _dslash_multi.Dslash(_spinor_y,_spinor_x,true);

    // Force ~ alpha_l |X><Y| (depends on site parity, separated by one site)
    Force.iterateOverBulkAllMu(tensor_product<floatT, onDevice, HaloDepth, HaloDepthSpin, 1, runTesting>(Force, _spinor_x.getAccessor(), _spinor_y.getAccessor(),rat_num));
    Force.updateAll();

    // _TmpForce ~ alpha_l |X><Y| (depends on site parity, separated by three sites)
    _TmpForce.iterateOverBulkAllMu(tensor_product<floatT, onDevice, HaloDepth, HaloDepthSpin, 3, runTesting>(NaikForce, _spinor_x.getAccessor(), _spinor_y.getAccessor(),rat_num));
    _TmpForce.updateAll();
    return;
}


template<class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, CompressionType comp, bool runTesting, const int rdeg>
void HisqForce<floatT,onDevice, HaloDepth,HaloDepthSpin,comp,runTesting,rdeg>::TestForce(
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &SpinorIn, 
    Gaugefield<floatT,onDevice,HaloDepth,comp> &Force, 
    grnd_state<true> &d_rand) 
{
    if (!runTesting) {
        rootLogger.error("Calling member function TestForce should only be used when the template parameter runTesting is set to true!");
    }

    Force.iterateWithConst(su3_zero<floatT>());
    _TmpForce.iterateWithConst(su3_zero<floatT>());

    Spinorfield<floatT, onDevice, Even, HaloDepthSpin> temp(SpinorIn.getComm());
    Spinorfield<floatT, false, Even, HaloDepthSpin> SpinorOutput(SpinorIn.getComm());

    Vect3<floatT> SpinorTestOutput;
    for (int i = 0; i < rdeg; i++) {
        temp.gauss(d_rand.state);
        _spinor_x.copyFromStackToStack(temp,i,0);
        SpinorOutput = temp;
        SpinorTestOutput = SpinorOutput.getAccessor().getElement(GIndexer<Even,HaloDepthSpin>::getSite(0,0,0,0));
        rootLogger.info("Xi_" ,  i ,  " " ,  SpinorTestOutput);
    }

    SimpleArray<floatT, rdeg> rat_num;
    for (int i = 0; i < rdeg; i++) {
        rat_num[i] = _rat.r_inv_lf_num[i];
    }

    _dslash_multi.Dslash(_spinor_y,_spinor_x);
    Force.iterateOverBulkAllMu(tensor_product<floatT, onDevice, HaloDepth, HaloDepthSpin, 1, runTesting>(Force, _spinor_x.getAccessor(), _spinor_y.getAccessor(),rat_num));
    _TmpForce.iterateOverBulkAllMu(tensor_product<floatT, onDevice, HaloDepth, HaloDepthSpin, 3, runTesting>(_TmpForce, _spinor_x.getAccessor(), _spinor_y.getAccessor(),rat_num));

    Force.updateAll();
    _TmpForce.updateAll();

    rootLogger.info("f0 intermediate result");
    printResult<floatT,HaloDepth,comp>(_TmpForce);
    _smearing.template SmearLvl1<R18>(_GaugeLvl1);
    _smearing.template ProjectU3<R18,R18>(_GaugeLvl1,_GaugeU3P);

    printResult<floatT,HaloDepth,R18>(_GaugeU3P);

    staggeredPhaseKernel<floatT, onDevice, HaloDepth,R18> multPhase(_GaugeU3P,_rhmc_param.mu_f());
    _GaugeU3P.iterateOverBulkAllMu(multPhase);

    _Dummy.template iterateOverBulkAllMu<64>(_createNaikF1);
    _TmpForce = _Dummy;
    _Dummy.template iterateOverBulkAllMu<64>(F1_create_3Link);
    _TmpForce = _TmpForce + _Dummy;

    static_for<1,5>::apply([&] (auto part) {
            _Dummy.template iterateOverBulkAllMu<64>(contribution_5link<floatT, onDevice, HaloDepth, R18, part>(_GaugeU3P,Force));
            _TmpForce = _TmpForce + _Dummy;
        });

    static_for<5,7>::apply([&] (auto part) {
        _Dummy.template iterateOverBulkAllMu<64>(contribution_5link_large<floatT, onDevice, HaloDepth, R18, part, 0>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
        
        _Dummy.template iterateOverBulkAllMu<64>(contribution_5link_large<floatT, onDevice, HaloDepth, R18, part, 1>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_5link_large<floatT, onDevice, HaloDepth, R18, part, 2>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_5link_large<floatT, onDevice, HaloDepth, R18, part, 3>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
    });

    _Dummy.iterateOverBulkAllMu(F1_lepagelink);
    _TmpForce = _TmpForce + _Dummy;

    static_for<1,8>::apply([&] (auto part) {
        _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 0>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

         _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 1>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

         _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 2>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
        
         _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 3>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
        
         _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 4>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
        
         _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 5>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
        
         _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 6>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
        
         _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 7>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
    });

    rootLogger.info("f1 intermediate result");
    printResult<floatT,HaloDepth,comp>(_TmpForce);
    _TmpForce.updateAll();

    staggeredPhaseKernel<floatT,onDevice, HaloDepth,R18> multPhaselv1(_GaugeLvl1,_rhmc_param.mu_f());
    _GaugeLvl1.iterateOverBulkAllMu(multPhaselv1);

    Force.iterateOverBulkAllMu(_createF2);
    Force.updateAll();

    rootLogger.info("f2 intermediate result");
    printResult<floatT,HaloDepth,comp>(Force);
    staggeredPhaseKernel<floatT,onDevice, HaloDepth,R18> multPhaseB(_GaugeBase,_rhmc_param.mu_f());
    _GaugeU3P.iterateOverBulkAllMu(multPhaseB); //reuse U3P Field here

    _TmpForce.template iterateOverBulkAllMu<64>(F3_create_3Link);
    
    static_for<1,5>::apply([&] (auto part) {
        _Dummy.template iterateOverBulkAllMu<64>(contribution_5link<floatT, onDevice, HaloDepth, R18, part>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
    });

    static_for<5,7>::apply([&] (auto part) {
        _Dummy.template iterateOverBulkAllMu<64>(contribution_5link_large<floatT, onDevice, HaloDepth, R18, part, 0>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
        
        _Dummy.template iterateOverBulkAllMu<64>(contribution_5link_large<floatT, onDevice, HaloDepth, R18, part, 1>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_5link_large<floatT, onDevice, HaloDepth, R18, part, 2>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_5link_large<floatT, onDevice, HaloDepth, R18, part, 3>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
    });

    static_for<1,8>::apply([&] (auto part) {
        _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 0>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

         _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 1>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

         _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 2>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
        
         _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 3>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
        
         _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 4>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
        
         _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 5>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
        
         _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 6>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
        
         _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 7>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
    });

    rootLogger.info("f3 intermediate result");
    printResult<floatT,HaloDepth,comp>(_TmpForce);

    Force.iterateOverBulkAllMu(_finalizeF3);

    rootLogger.info("f3 final result");
    printResult<floatT,HaloDepth,comp>(Force);

    return;
}


template<class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, CompressionType comp, bool runTesting, const int rdeg>
void HisqForce<floatT,onDevice, HaloDepth, HaloDepthSpin, comp, runTesting, rdeg>::updateForce(
    Spinorfield<floatT,onDevice,Even,HaloDepthSpin> &SpinorIn,
    Gaugefield<floatT,onDevice,HaloDepth,comp> &Force, bool isLight) 
{
    // Populate Force and _TmpForce ~ |X><Y| for 1-step and Naik, respectively. We start the _TmpForce with the
    // Naik separation because _TmpForce will accumulate all the force contributions and we're starting with
    // the Naik derivative.
    make_f0(SpinorIn,Force,_TmpForce,isLight);

    // Level 1 smearing, _GaugeLvl1 = V = SmearLvl1(U)
    _smearing.template SmearLvl1<R18>(_GaugeLvl1);

    // U3 projection, _GaugeU3P = W = U3projection(V)
    _smearing.template ProjectU3<R18,R18>(_GaugeLvl1,_GaugeU3P);

    // Multiply staggered phase into W (also where pure imaginary mu is applied) 
    staggeredPhaseKernel<floatT, onDevice, HaloDepth,R18> multPhase(_GaugeU3P,_rhmc_param.mu_f());
    _GaugeU3P.iterateOverBulkAllMu(multPhase);
    _GaugeU3P.updateAll();

    // --- Now start adding together the contributions to the force. Notes: 
    //     * The coefficients c3, c5, and c7 are the same on both levels of smearing. The coefficient c1 is different
    //       on each level. The initial contribution_3link object is thus instantiated with lvl1=False, which means it
    //       is using the lvl2 parameter for c1. Presumably this is why there is a separate F1_create_3Link functor
    //       while the other contributions are called explicitly.
    //     * dA/dB for matrices A,B is generally a rank-4 tensor. The contributions are written so that the
    //       contraction with |X><Y| is built in, which yields a rank-2 tensor again. So when I write derivatives
    //       like dX/dW from now on, it is understood to be contracted with |X><Y|
    //     * To figure out what the arguments of functors like F1_create_3Link are, you have to follow backwards where 
    //       they were instantiated or initialized.

    _Dummy.iterateOverBulkAllMu(_createNaikF1); // _TmpForce = derivativeNaik(W, |X><Y| naik)
    _TmpForce = _Dummy;
    _Dummy.template iterateOverBulkAllMu<64>(F1_create_3Link); // _TmpForce += d_3link(W, |X><Y|) using level 2 parameters ~ dX/dW
    _TmpForce = _TmpForce + _Dummy;
    
    static_for<1,5>::apply([&] (auto part) { // _TmpForce += d_5link(W, |X><Y|) ~ dX/dW
        _Dummy.template iterateOverBulkAllMu<64>(contribution_5link<floatT, onDevice, HaloDepth, R18, part>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
    });

    static_for<5,7>::apply([&] (auto part) { // _TmpForce += d_5link_large(W, |X><Y|) ~ dX/dW
        // static_for<0,4>::apply([&](auto term) { //nvidia compiler cannot handle nested static_for...
        _Dummy.template iterateOverBulkAllMu<64>(contribution_5link_large<floatT, onDevice, HaloDepth, R18, part, 0>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_5link_large<floatT, onDevice, HaloDepth, R18, part, 1>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_5link_large<floatT, onDevice, HaloDepth, R18, part, 2>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_5link_large<floatT, onDevice, HaloDepth, R18, part, 3>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
        // });
    });

    _Dummy.iterateOverBulkAllMu(F1_lepagelink); // _TmpForce += d_lepage(W, |X><Y|) ~ dX/dW
    _TmpForce = _TmpForce + _Dummy;

    static_for<1,8>::apply([&] (auto part) { // _TmpForce += d_7link(W, |X><Y|) ~ dX/dW
        _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 0>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 1>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 2>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 3>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 4>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 5>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 6>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 7>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
    });


    // Multiply staggered phases into V.
    staggeredPhaseKernel<floatT,onDevice, HaloDepth,R18> multPhaselv1(_GaugeLvl1, _rhmc_param.mu_f());
    _GaugeLvl1.iterateOverBulkAllMu(multPhaselv1);

    // Force = derivativeU3proj(W,_TmpForce) = dW/dV * dX/dW
    Force.iterateOverBulkAllMu(_createF2);
    Force.updateAll();

    // Multiply staggered phases into U. 
    staggeredPhaseKernel<floatT,onDevice, HaloDepth,R18> multPhaseB(_GaugeBase,_rhmc_param.mu_f());
    _GaugeU3P.iterateOverBulkAllMu(multPhaseB); // _GaugeU3P = U 
    _GaugeU3P.updateAll();


    // Overwrite _TmpForce = d_3link(U, dW/dV * dX/dW) using level 1 parameters = dV/dU * dW/dV * dX/dW. In this
    // way it is still keeping track of all the contributions to the force up to now.
    _TmpForce.template iterateOverBulkAllMu<64>(F3_create_3Link);
    
    static_for<1,5>::apply([&] (auto part) { // _TmpForce += d_5link(W, dW/dV * dX/dW) ~ dV/dU * dW/dV * dX/dW
        _Dummy.template iterateOverBulkAllMu<64>(contribution_5link<floatT, onDevice, HaloDepth, R18, part>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
    });

    static_for<5,7>::apply([&] (auto part) { // _TmpForce += d_5link(W, dW/dV * dX/dW) ~ dV/dU * dW/dV * dX/dW
        _Dummy.template iterateOverBulkAllMu<64>(contribution_5link_large<floatT, onDevice, HaloDepth, R18, part, 0>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_5link_large<floatT, onDevice, HaloDepth, R18, part, 1>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_5link_large<floatT, onDevice, HaloDepth, R18, part, 2>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_5link_large<floatT, onDevice, HaloDepth, R18, part, 3>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
    });

    static_for<1,8>::apply([&] (auto part) { // _TmpForce += d_7link(W, dW/dV * dX/dW) ~ dV/dU * dW/dV * dX/dW
        _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 0>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 1>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 2>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 3>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 4>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 5>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 6>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;

        _Dummy.template iterateOverBulkAllMu<64>(contribution_7link<floatT, onDevice, HaloDepth, R18, part, 7>(_GaugeU3P,Force));
        _TmpForce = _TmpForce + _Dummy;
    });

    // Force = 2 TA( U_mu(x) _TmpForce_mu(x) )
    Force.iterateOverBulkAllMu(_finalizeF3);

    return;
}

#define HFORCE_INIT(floatT, HALO, HALOSPIN)			       \
  template class HisqForce<floatT, true, HALO, HALOSPIN, R18, false>; \
  template class HisqForce<floatT, true, HALO, HALOSPIN, R18, true>;
INIT_PHHS(HFORCE_INIT)

