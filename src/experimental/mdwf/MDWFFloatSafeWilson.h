/*
 * Precision-generic copies of the Wilson pieces the even/odd MDWF operator
 * uses (MDWFMobiusEvenOdd.h), for the single-precision operator of the
 * mixed-precision solver (MDWFMixedPrecisionSolver.h).
 *
 * The shared originals (src/experimental/DWilson.h: DiracWilsonEvenOdd2,
 * gamma5; src/experimental/fullSpinor.h: Gamma*MultVec) multiply by double
 * literals such as (-1.0) * ColorVect<floatT>, which only compile for
 * floatT = double, and they are also used by the Wilson meson applications, so
 * they are not changed (AGENTS.md: prefer MDWF files over shared code). The
 * copies below are the same arithmetic with the literals cast to floatT and the
 * same (chiral) gamma basis. MDWFMobiusCloverEvenOdd keeps the originals for
 * double, so the validated double path is unchanged, and uses these for float.
 */

#pragma once

#include "../DWilson.h"
#include "../fullSpinor.h"

template<class floatT>
__host__ __device__ inline ColorVect<floatT> mdwfGammaXMult(const ColorVect<floatT> &v) {
    const COMPLEX(floatT) i(0.0, 1.0);
    ColorVect<floatT> r;
    r[0] = i * v[3];
    r[1] = i * v[2];
    r[2] = i * (static_cast<floatT>(-1.0) * v[1]);
    r[3] = i * (static_cast<floatT>(-1.0) * v[0]);
    return r;
}

template<class floatT>
__host__ __device__ inline ColorVect<floatT> mdwfGammaYMult(const ColorVect<floatT> &v) {
    ColorVect<floatT> r;
    r[0] = static_cast<floatT>(-1.0) * v[3];
    r[1] = v[2];
    r[2] = v[1];
    r[3] = static_cast<floatT>(-1.0) * v[0];
    return r;
}

template<class floatT>
__host__ __device__ inline ColorVect<floatT> mdwfGammaZMult(const ColorVect<floatT> &v) {
    ColorVect<floatT> r;
    r[0] = COMPLEX(floatT)(0.0, 1.0) * v[2];
    r[1] = COMPLEX(floatT)(0.0, -1.0) * v[3];
    r[2] = COMPLEX(floatT)(0.0, -1.0) * v[0];
    r[3] = COMPLEX(floatT)(0.0, 1.0) * v[1];
    return r;
}

// Chiral representation, as the active GammaTMultVec in fullSpinor.h.
template<class floatT>
__host__ __device__ inline ColorVect<floatT> mdwfGammaTMult(const ColorVect<floatT> &v) {
    ColorVect<floatT> r;
    r[0] = v[2];
    r[1] = v[3];
    r[2] = v[0];
    r[3] = v[1];
    return r;
}

template<class floatT>
__host__ __device__ inline ColorVect<floatT> mdwfGamma5Mult(const ColorVect<floatT> &v) {
    ColorVect<floatT> r;
    r[0] = v[0];
    r[1] = v[1];
    r[2] = static_cast<floatT>(-1.0) * v[2];
    r[3] = static_cast<floatT>(-1.0) * v[3];
    return r;
}

// DiracWilsonEvenOdd2<..., g5 = false>: (1/2) sum_mu [ -(1 - gamma_mu) U psi(x+mu) - (1 + gamma_mu) U^+ psi(x-mu) ].
template<class floatT, Layout LatLayoutLHS, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin,
         size_t NStacks>
struct MDWFWilsonHopEvenOdd {
    SU3Accessor<floatT> _SU3Accessor;
    SpinorColorAcc<floatT> _SpinorColorAccessor;

    typedef GIndexer<LatLayoutLHS, HaloDepthSpin> GInd;

    MDWFWilsonHopEvenOdd(Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
                         const Spinorfield<floatT, true, LatLayoutRHS, HaloDepthSpin, 12, NStacks> &spinorIn)
        : _SU3Accessor(gauge.getAccessor()), _SpinorColorAccessor(spinorIn.getAccessor()) {}

    __device__ __host__ Vect12<floatT> operator()(gSiteStack site) {
        ColorVect<floatT> outSC;
        outSC = static_cast<floatT>(0.0) * outSC;
        ColorVect<floatT> temp, temp2;

        temp = _SU3Accessor.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, 0)))
               * _SpinorColorAccessor.getColorVect(GInd::site_up(site, 0));
        temp2 = _SU3Accessor.getLinkDagger(
                    GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, 0), 0)))
                * _SpinorColorAccessor.getColorVect(GInd::site_dn(site, 0));
        outSC = outSC - temp - temp2 + mdwfGammaXMult(temp - temp2);

        temp = _SU3Accessor.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, 1)))
               * _SpinorColorAccessor.getColorVect(GInd::site_up(site, 1));
        temp2 = _SU3Accessor.getLinkDagger(
                    GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, 1), 1)))
                * _SpinorColorAccessor.getColorVect(GInd::site_dn(site, 1));
        outSC = outSC - temp - temp2 + mdwfGammaYMult(temp - temp2);

        temp = _SU3Accessor.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, 2)))
               * _SpinorColorAccessor.getColorVect(GInd::site_up(site, 2));
        temp2 = _SU3Accessor.getLinkDagger(
                    GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, 2), 2)))
                * _SpinorColorAccessor.getColorVect(GInd::site_dn(site, 2));
        outSC = outSC - temp - temp2 + mdwfGammaZMult(temp - temp2);

        temp = _SU3Accessor.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, 3)))
               * _SpinorColorAccessor.getColorVect(GInd::site_up(site, 3));
        temp2 = _SU3Accessor.getLinkDagger(
                    GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, 3), 3)))
                * _SpinorColorAccessor.getColorVect(GInd::site_dn(site, 3));
        outSC = outSC - temp - temp2 + mdwfGammaTMult(temp - temp2);

        outSC = static_cast<floatT>(0.5) * outSC;
        return convertColorVectToVect12(outSC);
    }
};

template<class floatT, Layout LatLayout, size_t HaloDepthSpin, size_t NStacks>
struct MDWFGamma5Functor {
    SpinorColorAcc<floatT> _SpinorColorAccessor;

    explicit MDWFGamma5Functor(const Spinorfield<floatT, true, LatLayout, HaloDepthSpin, 12, NStacks> &spinorIn)
        : _SpinorColorAccessor(spinorIn.getAccessor()) {}

    __device__ __host__ Vect12<floatT> operator()(gSiteStack site) {
        ColorVect<floatT> out = mdwfGamma5Mult(_SpinorColorAccessor.getColorVect(site));
        return convertColorVectToVect12(out);   // takes a non-const reference
    }
};
