/*
 * hisqForce.h
 *
 * D. Bollweg
 *
 * The methods related to the fictitious HISQ force, which drives the RHMC trajectory.
 *
 */

#include "../../gauge/constructs/hisqForceConstructs.h"
#include "../../gauge/constructs/derivativeProjectU3.h"
#include "../../gauge/constructs/naikDerivativeConstructs.h"
#include "hisqSmearing.h"
#include "../inverter/inverter.h"
#include "../dslash/dslash.h"
#include "../rhmc/rhmcParameters.h"

template <bool RunTesting> // template selector to change between rat approx used in RHMC and rat approx used in Testing
class RatDegreeSelector;

template <> class RatDegreeSelector<true> {
  public:
    static const int RatDegree = 14;
};
template <> class RatDegreeSelector<false> {
  public:
    static const int RatDegree = 12;
};

template <class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, int steps, bool runTesting = false,
          const int rdeg = RatDegreeSelector<runTesting>::RatDegree>
struct tensor_product {
    Vect3arrayAcc<floatT> _x;
    Vect3arrayAcc<floatT> _y;
    SU3Accessor<floatT> gAccessor;
    SimpleArray<floatT, rdeg> _rat_num;
    tensor_product(Gaugefield<floatT, onDevice, HaloDepth> &gaugeIn, Vect3arrayAcc<floatT> x, Vect3arrayAcc<floatT> y, SimpleArray<floatT, rdeg> rat_num);
    __device__ __host__ SU3<floatT> operator()(gSiteMu site);
};

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, bool lvl1> class contribution_3link {
  private:
    SU3Accessor<floatT, comp> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;
    SmearingParameters<floatT> _smParams = (lvl1 ? getLevel1Params<floatT>() : getLevel2Params<floatT>());

  public:
    contribution_3link(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceIn);
    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

// ============================================================
// Reverse recursive Fat7: outer-nu middle gather
// ============================================================
template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h> class outer_nu_middle_force {
  private:
    SU3Accessor<floatT, comp> _gAcc;
    SU3Accessor<floatT> _finAcc;

  public:
    outer_nu_middle_force(Gaugefield<floatT, onDevice, HaloDepth, comp> &Gauge, Gaugefield<floatT, onDevice, HaloDepth> &ForceIn)
        : _gAcc(Gauge.getAccessor()), _finAcc(ForceIn.getAccessor()) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu) {
        using GInd = GIndexer<All, HaloDepth>;

        gSite site = GInd::getSite(siteMu.isite);

        int mu = siteMu.mu;
        int nu = (mu + nu_h) % 4;

        return outerNuMiddleGather<floatT, HaloDepth, comp, R18>(_gAcc, _finAcc, site, mu, nu);
    }
};
template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h> class rho_middle_force {
  private:
    SU3Accessor<floatT, comp> _SU3Accessor;
    SU3Accessor<floatT> _forceNuAccessor;

  public:
    rho_middle_force(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceNu);

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

// D3 without its middle-link branches. The three middle branches are streamed
// into the same accumulator with accumulate_scaled_force below.
template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, bool lvl1 = false>
class recursive_three_link_base_force {
  private:
    SU3Accessor<floatT, comp> _gAcc;
    SU3Accessor<floatT> _finAcc;
    SmearingParameters<floatT> _smParams
        = lvl1 ? getLevel1Params<floatT>() : getLevel2Params<floatT>();

  public:
    recursive_three_link_base_force(Gaugefield<floatT, onDevice, HaloDepth, comp> &Gauge,
                                    Gaugefield<floatT, onDevice, HaloDepth> &ForceIn)
        : _gAcc(Gauge.getAccessor()), _finAcc(ForceIn.getAccessor()) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu) {
        using GInd = GIndexer<All, HaloDepth>;

        gSite site = GInd::getSite(siteMu.isite);
        int mu = siteMu.mu;
        SU3<floatT> side = su3_zero<floatT>();

        for (int nu_h = 1; nu_h < 4; ++nu_h) {
            int nu = (mu + nu_h) % 4;
            side += outerNuSideGather<floatT, HaloDepth, comp, R18>(_gAcc, _finAcc, site, mu, nu);
        }

        return _smParams._c_1 * _finAcc.getLink(siteMu) + _smParams._c_3 * side;
    }
};

// In-place accumulation is safe because each thread reads and writes only its
// own accumulator link. The contribution may read other, separately stored
// fields, but never a neighboring accumulator link.
template <class floatT, bool onDevice, size_t HaloDepth, CompressionType accumulatorComp, class Contribution>
class accumulate_scaled_force {
  private:
    SU3Accessor<floatT, accumulatorComp> _accumulatorAcc;
    Contribution _contribution;
    floatT _coefficient;

  public:
    accumulate_scaled_force(Gaugefield<floatT, onDevice, HaloDepth, accumulatorComp> &Accumulator,
                            Contribution contribution, floatT coefficient)
        : _accumulatorAcc(Accumulator.getAccessor()), _contribution(contribution), _coefficient(coefficient) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu) {
        return _accumulatorAcc.getLink(siteMu) + _coefficient * _contribution(siteMu);
    }
};

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType accumulatorComp, class Contribution>
inline
accumulate_scaled_force<floatT, onDevice, HaloDepth, accumulatorComp, Contribution>
make_accumulate_scaled_force(Gaugefield<floatT, onDevice, HaloDepth, accumulatorComp> &Accumulator,
                             Contribution contribution, floatT coefficient) {
    return accumulate_scaled_force<floatT, onDevice, HaloDepth, accumulatorComp, Contribution>(
        Accumulator, contribution, coefficient);
}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h> class rho_side_force {
  private:
    SU3Accessor<floatT, comp> _SU3Accessor;
    SU3Accessor<floatT> _forceNuAccessor;

  public:
    rho_side_force(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceNu);

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h>
rho_side_force<floatT, onDevice, HaloDepth, comp, nu_h, rho_h>::rho_side_force(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn,
                                                                               Gaugefield<floatT, onDevice, HaloDepth> &ForceNu)
    : _SU3Accessor(GaugeIn.getAccessor()), _forceNuAccessor(ForceNu.getAccessor()) {}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h>
__host__ __device__ SU3<floatT> rho_side_force<floatT, onDevice, HaloDepth, comp, nu_h, rho_h>::operator()(gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;

    gSite site = GInd::getSite(siteMu.isite);

    // The output direction is now the inner rho direction.
    int targetRho = siteMu.mu;

    int centerMu;
    int outerNu;

    bool found = invertRhoDirection<nu_h, rho_h>(targetRho, centerMu, outerNu);

    if (!found)
        return su3_zero<floatT>();

    /*
     * The side derivative of D_rho[U_mu] has exactly the
     * same structure as the already validated side derivative
     * of D_nu[U_mu].
     *
     * Here:
     *   output direction  = targetRho
     *   central direction = centerMu
     *   upstream force    = F_N
     */
    return outerNuSideGather<floatT, HaloDepth, comp, R18>(_SU3Accessor, _forceNuAccessor, site, targetRho, centerMu);
}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h> class rho_dressed_primal {
  private:
    SU3Accessor<floatT, comp> _SU3Accessor;

  public:
    rho_dressed_primal(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn);

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h>
rho_dressed_primal<floatT, onDevice, HaloDepth, comp, nu_h, rho_h>::rho_dressed_primal(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn)
    : _SU3Accessor(GaugeIn.getAccessor()) {}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h>
__host__ __device__ SU3<floatT> rho_dressed_primal<floatT, onDevice, HaloDepth, comp, nu_h, rho_h>::operator()(gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;

    gSite site = GInd::getSite(siteMu.isite);

    // This field is indexed by the original central direction mu.
    int mu = siteMu.mu;

    int nu = (mu + nu_h) % 4;

    int rho = (((mu + nu) % 2) * ((40 * (mu + nu) - 6 * mu * nu - 18 * (mu * mu + nu * nu) + 2 * (mu * mu * mu + nu * nu * nu)) / 12 + rho_h) +
               ((mu + nu + 1) % 2) * (mu + 1 + 2 * rho_h)) %
              4;

    return rhoDressPrimalGather<floatT, HaloDepth, comp>(_SU3Accessor, site, mu, rho);
}

template <class floatT, size_t HaloDepth, CompressionType compGauge = R18, CompressionType compDressed = R18, CompressionType compForce = R18>
__host__ __device__ SU3<floatT> outerNuSideDressedGather(SU3Accessor<floatT, compGauge> gAcc, SU3Accessor<floatT, compDressed> xAcc,
                                                         SU3Accessor<floatT, compForce> fAcc, gSite site, int sideMu, int middleNu) {
    typedef GIndexer<All, HaloDepth> GInd;

    gSite origin = site;
    gSite up = GInd::site_up(site, middleNu);
    gSite right = GInd::site_up(site, sideMu);
    gSite down = GInd::site_dn(site, middleNu);
    gSite rightDn = GInd::site_dn(right, middleNu);

    /*
     * These are exactly the four SIDE terms of the
     * 3-link reverse derivative, except that the
     * middle primal U_nu has been replaced by X_nu.
     *
     * IMPORTANT:
     * there is NO overall minus here.
     */

    SU3<floatT> temp = fAcc.getLinkDagger(GInd::getSiteMu(right, middleNu)) * gAcc.getLinkDagger(GInd::getSiteMu(up, sideMu)) *
                       xAcc.getLinkDagger(GInd::getSiteMu(origin, middleNu));

    temp += xAcc.getLink(GInd::getSiteMu(right, middleNu)) * gAcc.getLinkDagger(GInd::getSiteMu(up, sideMu)) * fAcc.getLink(GInd::getSiteMu(origin, middleNu));

    temp +=
        fAcc.getLink(GInd::getSiteMu(rightDn, middleNu)) * gAcc.getLinkDagger(GInd::getSiteMu(down, sideMu)) * xAcc.getLink(GInd::getSiteMu(down, middleNu));

    temp += xAcc.getLinkDagger(GInd::getSiteMu(rightDn, middleNu)) * gAcc.getLinkDagger(GInd::getSiteMu(down, sideMu)) *
            fAcc.getLinkDagger(GInd::getSiteMu(down, middleNu));

    return temp;
}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h> class outer_nu_side_dressed_force {
  private:
    SU3Accessor<floatT, comp> _SU3Accessor;
    SU3Accessor<floatT> _dressedAccessor;
    SU3Accessor<floatT> _forceAccessor;

  public:
    outer_nu_side_dressed_force(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &Dressed,
                                Gaugefield<floatT, onDevice, HaloDepth> &ForceIn);

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h>
outer_nu_side_dressed_force<floatT, onDevice, HaloDepth, comp, nu_h, rho_h>::outer_nu_side_dressed_force(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn,
                                                                                                         Gaugefield<floatT, onDevice, HaloDepth> &Dressed,
                                                                                                         Gaugefield<floatT, onDevice, HaloDepth> &ForceIn)
    : _SU3Accessor(GaugeIn.getAccessor()), _dressedAccessor(Dressed.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h>
__host__ __device__ SU3<floatT> outer_nu_side_dressed_force<floatT, onDevice, HaloDepth, comp, nu_h, rho_h>::operator()(gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;

    gSite site = GInd::getSite(siteMu.isite);

    /*
     * Output link is the OUTER side direction nu.
     */
    int targetNu = siteMu.mu;

    /*
     * Forward mapping:
     *
     * targetNu = (centerMu + nu_h) % 4
     *
     * so invert it.
     */
    int centerMu = (targetNu + 4 - nu_h) % 4;

    return outerNuSideDressedGather<floatT, HaloDepth, comp, R18, R18>(_SU3Accessor, _dressedAccessor, _forceAccessor, site, targetNu, centerMu);
}

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h> class sigma_middle_force {
  private:
    SU3Accessor<floatT, comp> _SU3Accessor;
    SU3Accessor<floatT> _forceRAccessor;

  public:
    sigma_middle_force(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceR)
        : _SU3Accessor(GaugeIn.getAccessor()), _forceRAccessor(ForceR.getAccessor()) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h> class sigma_side_force {
  private:
    SU3Accessor<floatT, comp> _SU3Accessor;
    SU3Accessor<floatT> _forceRAccessor;

  public:
    sigma_side_force(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceR)
        : _SU3Accessor(GaugeIn.getAccessor()), _forceRAccessor(ForceR.getAccessor()) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu) {
        typedef GIndexer<All, HaloDepth> GInd;

        gSite site = GInd::getSite(siteMu.isite);

        int targetSigma = siteMu.mu;
        int centerMu;

        invertSigmaDirection<nu_h, rho_h>(targetSigma, centerMu);

        return outerNuSideGather<floatT, HaloDepth, comp, R18>(_SU3Accessor, _forceRAccessor, site, targetSigma, centerMu);
    }
};

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h> class sigma_dressed_primal {
  private:
    SU3Accessor<floatT, comp> _SU3Accessor;

  public:
    sigma_dressed_primal(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn) : _SU3Accessor(GaugeIn.getAccessor()) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h> class rho_side_sigma_dressed_force {
  private:
    SU3Accessor<floatT, comp> _SU3Accessor;
    SU3Accessor<floatT> _sigmaAccessor;
    SU3Accessor<floatT> _forceNuAccessor;

  public:
    rho_side_sigma_dressed_force(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &SigmaPrimal,
                                 Gaugefield<floatT, onDevice, HaloDepth> &ForceNu)
        : _SU3Accessor(GaugeIn.getAccessor()), _sigmaAccessor(SigmaPrimal.getAccessor()), _forceNuAccessor(ForceNu.getAccessor()) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h> class rho_sigma_dressed_primal {
  private:
    SU3Accessor<floatT, comp> _SU3Accessor;
    SU3Accessor<floatT> _sigmaAccessor;

  public:
    rho_sigma_dressed_primal(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &SigmaPrimal);

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

// The D5 and D7 rho-side derivatives have the same upstream force F_N.
// Fuse their accumulation using linearity:
//
//   c5 * rho_side_force[U, F_N]
//     + c7 * rho_side_sigma_dressed_force[X_sigma, F_N].
template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h>
class combined_rho_side_force {
  private:
    rho_side_force<floatT, onDevice, HaloDepth, comp, nu_h, rho_h> _d5RhoSide;
    rho_side_sigma_dressed_force<floatT, onDevice, HaloDepth, comp, nu_h, rho_h> _d7RhoSide;
    floatT _c5;
    floatT _c7;

  public:
    combined_rho_side_force(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn,
                            Gaugefield<floatT, onDevice, HaloDepth> &SigmaPrimal,
                            Gaugefield<floatT, onDevice, HaloDepth> &ForceNu,
                            floatT c5, floatT c7)
        : _d5RhoSide(GaugeIn, ForceNu),
          _d7RhoSide(GaugeIn, SigmaPrimal, ForceNu),
          _c5(c5), _c7(c7) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu) {
        return _c5 * _d5RhoSide(siteMu) + _c7 * _d7RhoSide(siteMu);
    }
};

// Linearity of the rho dressing combines the two outer-nu primal fields:
//
//   c5 * D_rho[U] - c7 * D_rho[X_sigma]
//     = D_rho[c5 * U - c7 * X_sigma].
//
// The minus is the already validated signed-D7 convention for positions 1+7.
template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int nu_h, int rho_h>
class combined_rho_dressed_primal {
  private:
    rho_dressed_primal<floatT, onDevice, HaloDepth, comp, nu_h, rho_h> _d5Primal;
    rho_sigma_dressed_primal<floatT, onDevice, HaloDepth, comp, nu_h, rho_h> _d7Primal;
    floatT _c5;
    floatT _c7;

  public:
    combined_rho_dressed_primal(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn,
                                Gaugefield<floatT, onDevice, HaloDepth> &SigmaPrimal,
                                floatT c5, floatT c7)
        : _d5Primal(GaugeIn), _d7Primal(GaugeIn, SigmaPrimal),
          _c5(c5), _c7(c7) {}

    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu) {
        return _c5 * _d5Primal(siteMu) - _c7 * _d7Primal(siteMu);
    }
};

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp> class contribution_lepagelink {
  private:
    SU3Accessor<floatT, comp> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;
    floatT _c_lp = -1 / 8.0;

  public:
    contribution_lepagelink(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceIn);
    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part> class contribution_5link {
  private:
    SU3Accessor<floatT, comp> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;
    floatT _c5 = 1 / 8. / 8.;

  public:
    contribution_5link(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceIn);
    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part, size_t term> class contribution_5link_large {
  private:
    SU3Accessor<floatT, comp> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;
    floatT _c5 = 1 / 8. / 8.;

  public:
    contribution_5link_large(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceIn);
    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part, size_t term> class contribution_7link {
  private:
    SU3Accessor<floatT, comp> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;
    floatT _c7 = 1 / 48. / 8.;

  public:
    contribution_7link(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceInm);
    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp> class constructU3ProjForce {
  private:
    SU3Accessor<floatT> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;

  public:
    constructU3ProjForce(Gaugefield<floatT, onDevice, HaloDepth> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth, comp> &ForceIn);
    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp> class finalizeForce {
  private:
    SU3Accessor<floatT, comp> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;

  public:
    finalizeForce(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth, comp> &ForceIn);
    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp> class constructNaikDerivForce {
  private:
    SU3Accessor<floatT> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;

  public:
    constructNaikDerivForce(Gaugefield<floatT, onDevice, HaloDepth> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth, comp> &ForceIn);
    __host__ __device__ SU3<floatT> operator()(gSiteMu siteMu);
};

template <class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, CompressionType comp = R18, bool runTesting = false,
          const int rdeg = RatDegreeSelector<runTesting>::RatDegree>
class HisqForce {
  private:
    Gaugefield<floatT, onDevice, HaloDepth> _GaugeU3P;
    Gaugefield<floatT, onDevice, HaloDepth> _GaugeLvl1;
    Gaugefield<floatT, onDevice, HaloDepth, comp> _TmpForce; // One of its uses is for NaikForce
    Gaugefield<floatT, onDevice, HaloDepth, R18> &_GaugeBase;
    Gaugefield<floatT, onDevice, HaloDepth, R18> _Dummy;

    Gaugefield<floatT, onDevice, HaloDepth, R18> _ForceNu;

    Spinorfield<floatT, onDevice, Even, HaloDepthSpin, rdeg> _spinor_x;
    Spinorfield<floatT, onDevice, Odd, HaloDepthSpin, rdeg> _spinor_y;

    constructU3ProjForce<floatT, onDevice, HaloDepth, comp> _createF2;

    finalizeForce<floatT, onDevice, HaloDepth, comp> _finalizeF3;

    constructNaikDerivForce<floatT, onDevice, HaloDepth, comp> _createNaikF1;

    // F1 part
    contribution_3link<floatT, onDevice, HaloDepth, comp, false> F1_create_3Link;

    // contribution_7link<floatT, onDevice, HaloDepth, comp, 1> F1_7link_part_1;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 2> F1_7link_part_2;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 3> F1_7link_part_3;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 4> F1_7link_part_4;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 5> F1_7link_part_5;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 6> F1_7link_part_6;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 7> F1_7link_part_7;

    // contribution_5link<floatT, onDevice, HaloDepth, comp, 11> F1_5link_part11;
    // contribution_5link<floatT, onDevice, HaloDepth, comp, 12> F1_5link_part12;
    // contribution_5link<floatT, onDevice, HaloDepth, comp, 13> F1_5link_part13;
    // contribution_5link<floatT, onDevice, HaloDepth, comp, 14> F1_5link_part14;
    // contribution_5link<floatT, onDevice, HaloDepth, comp, 20> F1_5link_part20;
    // contribution_5link<floatT, onDevice, HaloDepth, comp, 30> F1_5link_part30;

    contribution_lepagelink<floatT, onDevice, HaloDepth, comp> F1_lepagelink;

    // contribution_7link<floatT, onDevice, HaloDepth, comp, 1> F3_7link_part_1;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 2> F3_7link_part_2;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 3> F3_7link_part_3;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 4> F3_7link_part_4;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 5> F3_7link_part_5;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 6> F3_7link_part_6;
    // contribution_7link<floatT, onDevice, HaloDepth, comp, 7> F3_7link_part_7;

    // contribution_5link<floatT, onDevice, HaloDepth, comp, 11> F3_5link_part11;
    // contribution_5link<floatT, onDevice, HaloDepth, comp, 12> F3_5link_part12;
    // contribution_5link<floatT, onDevice, HaloDepth, comp, 13> F3_5link_part13;
    // contribution_5link<floatT, onDevice, HaloDepth, comp, 14> F3_5link_part14;
    // contribution_5link<floatT, onDevice, HaloDepth, comp, 20> F3_5link_part20;
    // contribution_5link<floatT, onDevice, HaloDepth, comp, 30> F3_5link_part30;

    HisqSmearing<floatT, onDevice, HaloDepth, R18, R18, R18, U3R14> &_smearing;
    AdvancedMultiShiftCG<floatT, rdeg> &_cg;
    HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, 1> &_dslash;
    HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, rdeg> &_dslash_multi;
    RhmcParameters _rhmc_param;
    RationalCoeff _rat;

    void constructF1(Gaugefield<floatT, onDevice, HaloDepth, comp> &Force);
    void constructF3Recursive(
        Gaugefield<floatT, onDevice, HaloDepth, comp> &Force,
        Gaugefield<floatT, onDevice, HaloDepth, comp> &ForceOut);

  public:
    // Initializer list is in cpp file.
    HisqForce(Gaugefield<floatT, onDevice, HaloDepth, R18> &GaugeBase, Gaugefield<floatT, onDevice, HaloDepth, comp> &Force,
              AdvancedMultiShiftCG<floatT, rdeg> &cg, HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, 1> &dslash,
              HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, rdeg> &dslash_multi, RhmcParameters &rhmc_param, RationalCoeff &rat,
              HisqSmearing<floatT, onDevice, HaloDepth, R18, R18, R18, U3R14> &smearing);

    void make_f0(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &SpinorIn, Gaugefield<floatT, onDevice, HaloDepth, comp> &Force,
                 Gaugefield<floatT, onDevice, HaloDepth, comp> &NaikForce, bool isLight);

    void updateForce(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &SpinorIn, Gaugefield<floatT, onDevice, HaloDepth, comp> &Force, bool isLight);

    void TestForce(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &SpinorIn, Gaugefield<floatT, onDevice, HaloDepth, comp> &Force,
                   grnd_state<true> &d_rand);

};
