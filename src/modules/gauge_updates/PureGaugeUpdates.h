/*
 * PureGaugeUpdates.h
 *
 * v1.0: D. Clarke, 1 Feb 2019
 *
 * Some methods to update pure gauge systems.
 *
 */

#pragma once

#include "../../gauge/gaugefield.h"
#include "../../base/math/gcomplex.h"
#include "../../define.h"
#include "../../base/math/gsu2.h"
#include "../../base/math/grnd.h"


template<class floatT, bool onDevice, size_t HaloDepth>
class GaugeUpdate {
protected:
    Gaugefield<floatT, onDevice, HaloDepth> &_gauge;

private:
    typedef GIndexer<All, HaloDepth> GInd;
    const size_t elems = GInd::getLatData().sizeh;

public:
    explicit GaugeUpdate(Gaugefield<floatT, onDevice, HaloDepth> &gaugefield) : _gauge(gaugefield) {}

    void updateOR();                                              /// OR update of entire lattice
    void updateHB(uint4 *state, floatT beta, bool ltest=false);   /// HB update of entire lattice
    void set_gauge_to_reference();
};

/// Even/odd read index
template<Layout LatLayout, size_t HaloDepth>
struct ReadIndexEvenOdd {
    inline __host__ __device__ gSite operator()(const dim3 &blockDim, const uint3 &blockIdx, const uint3 &threadIdx) {
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        typedef GIndexer<LatLayout, HaloDepth> GInd;
        gSite site = GInd::getSite(i);
        return site;
    }
};

/// Get staple attached to U_mu(site).
template<class floatT, Layout LatLayout, size_t HaloDepth>
__device__ __host__ inline GSU3<floatT> SU3Staple(gaugeAccessor<floatT> gaugeAccessor,
                                                  const gSite &site, const uint8_t &mu) {
    typedef GIndexer<LatLayout, HaloDepth> GInd;
    GSU3<floatT> Ustaple, U1, U2, U3;
    Ustaple = gsu3_zero<floatT>();
    for (uint8_t nu = 0; nu < 4; nu++) {
        if (nu != mu) {
            gSite x_p_mu = GInd::site_up(site, mu);
            gSite x_p_nu = GInd::site_up(site, nu);
            /// Right part of the staple.
            U1 = gaugeAccessor.getLink(GInd::getSiteMu(x_p_mu, nu));
            U2 = gaugeAccessor.getLinkDagger(GInd::getSiteMu(x_p_nu, mu));
            U3 = gaugeAccessor.getLinkDagger(GInd::getSiteMu(site, nu));
            Ustaple += U1 * U2 * U3;
            gSite x_m_nu = GInd::site_dn(site, nu);
            gSite x_p_mu_m_nu = GInd::site_dn(x_p_mu, nu);
            /// Left part of the staple.
            U1 = gaugeAccessor.getLinkDagger(GInd::getSiteMu(x_p_mu_m_nu, nu));
            U2 = gaugeAccessor.getLinkDagger(GInd::getSiteMu(x_m_nu, mu));
            U3 = gaugeAccessor.getLink(GInd::getSiteMu(x_m_nu, nu));
            Ustaple += U1 * U2 * U3;
        }
    }
    return Ustaple;
}

/// SU3 over-relaxation update.
template<class floatT>
__device__ __host__ inline GSU3<floatT> OR_GPUSU3(const GSU3<floatT> &U, const GSU3<floatT> &Ustaple) {
    GSU3<floatT> w;
    GSU2<floatT> a, z;
    floatT znorm;
    w = U;

    /// First subgroup
    z = sub12(w, Ustaple);      /// Extract staple in first subgroup in link*staple
    a = (z * z);                /// Calculate OR link and project on to SU(2)
    a = a.dagger();
    znorm = z.norm2();
    a = a / znorm;
    w = sub12(a, w);            /// Multiply into SU(3)

    /// Repeat for second subgroup
    z = sub13(w, Ustaple);
    a = (z * z);
    a = a.dagger();
    znorm = z.norm2();
    a = a / znorm;
    w = sub13(a, w);

    /// And third subgroup
    z = sub23(w, Ustaple);
    a = (z * z);
    a = a.dagger();
    znorm = z.norm2();
    a = a / znorm;
    w = sub23(a, w);

    return (w);
}

/// Heatbath update for SU2 matrices. If the ltest flag is true, RNG is turned off. This is to check whether the update
/// works independently of the RNG.
template<typename floatT>
__device__ __host__ inline GSU2<floatT> HB_GPUSU2(const GSU2<floatT> &U, const floatT inv_4beta, uint4 *stateElem,
                                                  bool ltest)
{
    GSU2<floatT> z, g;
    floatT znorm, alpha;
    floatT x0, x1, x2, x3, x4;
    floatT hb_1, hb_2;

    znorm = 1.0 / sqrt(norm2(U)); /// According to Lüscher: =1/a
    z = znorm * U;
    alpha = inv_4beta * znorm;    /// for SU2 this would be 1/(2*a*beta). With SU3 subgroup update this is 3/(4*a*beta)

    hb_1 = get_rand_excl0<floatT>(stateElem);
    if(ltest) hb_1 = 0.215;
    x1 = log(hb_1);

    hb_1 = get_rand_excl0<floatT>(stateElem);
    if(ltest) hb_1 = 0.839;
    x2 = log(hb_1);

    hb_1 = get_rand_excl0<floatT>(stateElem);
    if(ltest) hb_1 = 0.390;
    x3 = cos(2.0 * M_PI * hb_1);

    x0 = (x1 + x2 * x3 * x3) * alpha;        /// lambda^2 Gattringer Lang eq. (4.45)

    hb_1 = get_rand_excl0<floatT>(stateElem);
    if(ltest) hb_1 = 0.004;
    x4 = hb_1 * hb_1 - x0;                   /// eq. (4.46)

    hb_1 = get_rand_excl0<floatT>(stateElem);
    hb_2 = get_rand_excl0<floatT>(stateElem);
    if(ltest) hb_1 = 0.294;
    if(ltest) hb_2 = 0.082;

    if (x4 < 1.0) {
        x0 = 1.0 + 2.0 * x0;
        x4 = sqrt(1.0 - x0 * x0);            /// |x|
        x1 = x4 * (2.0 * hb_1 - 1.0);
        x4 = sqrt((x4 - x1) * (x4 + x1));
        x2 = x4 * cos(2.0 * M_PI * hb_2);
        x3 = x4 * sin(2.0 * M_PI * hb_2);
        g = GSU2<floatT>(GCOMPLEX(floatT)(x0, x1), GCOMPLEX(floatT)(x2, x3));
        g = g * dagger(z);
    } else {
        g = GSU2<floatT>(1.0, 0.0);
    }

    return (g);
}

/// Heatbath update of SU2 subgroups embedded in SU3.
template<typename floatT>
__device__ __host__ inline GSU3<float> HB_GPUSU3(const GSU3<floatT> &U, GSU3<floatT> &Ustaple,
                                                 uint4 *state, const floatT beta, bool ltest) {
    GSU2<floatT> z;
    GSU3<floatT> x;
    floatT inv_75beta = 0.75 / beta;

    x = U;
    z = sub12(x, Ustaple);                              /// According to Gattringer: Calc relevant part of W eq. (4.48)
    z = HB_GPUSU2<floatT>(z, inv_75beta, state, ltest); /// distribution according to eq. (4.41)
    x = sub12(z, x);                                    /// Mult new SU2 into U (4.49)

    z = sub13(x, Ustaple);
    z = HB_GPUSU2<floatT>(z, inv_75beta, state, ltest);
    x = sub13(z, x);

    z = sub23(x, Ustaple);
    z = HB_GPUSU2<floatT>(z, inv_75beta, state, ltest);
    x = sub23(z, x);

    return x;
}

