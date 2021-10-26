//
// Created by Lukas Mazur on 19.11.18.
//

#ifndef GRADIENTFLOW_H
#define GRADIENTFLOW_H

#if ADAPTIVE_STEPSIZE
#include "su3rungeKutta3AdStepSize.h"
#include "su3rungeKutta3AdStepSizeAllGPU.h"
#endif //ADAPTIVE_STEPSIZE

#include "su3rungeKutta3.h"
#include "../../gauge/gaugeActionDeriv.h"

enum RungeKuttaMethod { fixed_stepsize, adaptive_stepsize, adaptive_stepsize_allgpu };
enum Force { wilson, zeuthen };
///convert string to enum for switch()
 std::map<std::string, Force> Force_map = {
        {std::string("wilson"), wilson},
        {std::string("zeuthen"), zeuthen}
};
std::map<std::string, RungeKuttaMethod > RK_map = {
        {std::string("fixed_stepsize"), fixed_stepsize},
        {std::string("adaptive_stepsize"), adaptive_stepsize},
        {std::string("adaptive_stepsize_allgpu"), adaptive_stepsize_allgpu}
};

/*Z_i( GSU3<floatT> &, GSU3<floatT> &,  GSU3<floatT> &, floatT )
inhomogenity Kernel
The derivative of the Wilson gauge action can be reduced to

	d/dt S_W(V_mu(x,t)) = 1/6 tr(omega - omega^+) * I - 0.5 * (omega - omega^+)

Where I denotes the SU(3) unitary element and omega is defined as

	omega = V_mu(x,t) * W^+(x,mu)

W^+(x,rho) denotes the staple matrix. Then a runge kutta step is given by

	Z(V_mu(x,t)) = d/dt S_W(V_mu(x,t)) * V_mu(x,t)
*/

#if WILSON_FLOW
template<class floatT, const size_t HaloDepth, bool onDevice>
struct Z_i_wilson {

    gaugeAccessor<floatT> gaugeAccessor;
    floatT _stepSize;

    Z_i_wilson(Gaugefield<floatT, onDevice, HaloDepth> &gauge, floatT stepSize) :
            gaugeAccessor(gauge.getAccessor()),
            _stepSize(stepSize) {}

    typedef GIndexer<All, HaloDepth> GInd;
    __device__ __host__ inline GSU3<floatT> operator()(gSiteMu siteMu) {
        return _stepSize * floatT(-1.0) *
               gaugeActionDerivPlaq<floatT, HaloDepth>(gaugeAccessor, siteMu, siteMu.mu); // for Wilson Flow
    }
};


template<class floatT, const size_t HaloDepth, RungeKuttaMethod _RK_method>
class wilsonFlow {
};

#if FIXED_STEPSIZE
template<class floatT, const size_t HaloDepth>
class wilsonFlow<floatT, HaloDepth, fixed_stepsize>
        : public su3rungeKutta3<floatT, HaloDepth, Z_i_wilson<floatT, HaloDepth, true> > {

    using Z_i = Z_i_wilson<floatT, HaloDepth, true>;
public:
    wilsonFlow(Gaugefield<floatT, true, HaloDepth> &inGaugeA, floatT stepSize, floatT start, floatT stop,
                   std::vector<floatT> necessaryFlowTime, floatT dummy_accuracy = 0)
            : su3rungeKutta3<floatT, HaloDepth, Z_i>(inGaugeA, stepSize, start, stop, necessaryFlowTime) {}

};
#endif

#if ADAPTIVE_STEPSIZE
template<class floatT, const size_t HaloDepth>
class wilsonFlow<floatT, HaloDepth, adaptive_stepsize>
        : public su3rungeKutta3AdStepSize<floatT, HaloDepth, Z_i_wilson<floatT, HaloDepth, true> > {

    using Z_i = Z_i_wilson<floatT, HaloDepth, true>;
public:
    wilsonFlow(Gaugefield<floatT, true, HaloDepth> &inGaugeA,
                floatT stepSize, floatT start, floatT stop,
                             std::vector<floatT> necessaryFlowTime, floatT accuracy)
            : su3rungeKutta3AdStepSize<floatT, HaloDepth, Z_i>(inGaugeA, stepSize, start, stop,
                             necessaryFlowTime, accuracy) {}

};


template<class floatT, const size_t HaloDepth>
class wilsonFlow<floatT, HaloDepth, adaptive_stepsize_allgpu>
        : public su3rungeKutta3AdStepSizeAllGPU<floatT, HaloDepth, Z_i_wilson<floatT, HaloDepth, true> > {

    using Z_i = Z_i_wilson<floatT, HaloDepth, true>;
public:
    wilsonFlow(Gaugefield<floatT, true, HaloDepth> &inGaugeA,
               floatT stepSize, floatT start, floatT stop,
               std::vector<floatT> necessaryFlowTime, floatT accuracy)
            : su3rungeKutta3AdStepSizeAllGPU<floatT, HaloDepth, Z_i>(inGaugeA, stepSize, start, stop,
                                                               necessaryFlowTime, accuracy) {}

};
#endif
#endif

#if ZEUTHEN_FLOW
template<class floatT, const size_t HaloDepth, bool onDevice>
struct Z_i_zeuthen {

    gaugeAccessor<floatT> gaugeAccessor;
    floatT _stepSize;

    typedef GIndexer<All, HaloDepth> GInd;

    Z_i_zeuthen(Gaugefield<floatT, onDevice, HaloDepth> &gauge, floatT stepSize) :
            gaugeAccessor(gauge.getAccessor()),
            _stepSize(stepSize) {}

    __device__ __host__ inline GSU3<floatT> operator()(gSiteMu siteMu) {

        gSiteMu siteDn = GInd::getSiteMu(GInd::site_dn(siteMu, siteMu.mu), siteMu.mu);
        gSiteMu siteUp = GInd::getSiteMu(GInd::site_up(siteMu, siteMu.mu), siteMu.mu);

        GSU3<floatT> symForce = symanzikGaugeActionDeriv<floatT,HaloDepth>(gaugeAccessor,siteMu,siteMu.mu);
        GSU3<floatT> symForcePlusMu = symanzikGaugeActionDeriv<floatT,HaloDepth>(gaugeAccessor,siteUp,siteMu.mu);
        GSU3<floatT> symForceMinusMu = symanzikGaugeActionDeriv<floatT,HaloDepth>(gaugeAccessor,siteDn,siteMu.mu);
        GSU3<floatT> Link = gaugeAccessor.getLink(siteMu);
        GSU3<floatT> LinkD = gaugeAccessor.getLinkDagger(siteMu);

        GSU3<floatT> LinkMu = gaugeAccessor.getLink(siteDn);
        GSU3<floatT> LinkDMu = gaugeAccessor.getLinkDagger(siteDn);

        return floatT(-1.0) * _stepSize *(
            floatT(5./6.) * symForce + floatT(1./12.) * Link * symForcePlusMu * LinkD + floatT(1./12.) * LinkDMu * symForceMinusMu * LinkMu);
    }
};


template<class floatT, const size_t HaloDepth, RungeKuttaMethod _RK_method>
class zeuthenFlow {
};

#if FIXED_STEPSIZE
template<class floatT, const size_t HaloDepth>
class zeuthenFlow<floatT, HaloDepth, fixed_stepsize>
        : public su3rungeKutta3<floatT, HaloDepth, Z_i_zeuthen<floatT, HaloDepth, true> > {

    using Z_i = Z_i_zeuthen<floatT, HaloDepth, true>;
public:
    zeuthenFlow(Gaugefield<floatT, true, HaloDepth> &inGaugeA, floatT stepSize, floatT start, floatT stop,
                   std::vector<floatT> necessaryFlowTime, floatT dummy_accuracy = 0)
            : su3rungeKutta3<floatT, HaloDepth, Z_i>(inGaugeA, stepSize, start, stop, necessaryFlowTime) {}

};
#endif

#if ADAPTIVE_STEPSIZE
template<class floatT, const size_t HaloDepth>
class zeuthenFlow<floatT, HaloDepth, adaptive_stepsize>
        : public su3rungeKutta3AdStepSize<floatT, HaloDepth, Z_i_zeuthen<floatT, HaloDepth, true> > {

    using Z_i = Z_i_zeuthen<floatT, HaloDepth, true>;
public:
    zeuthenFlow(Gaugefield<floatT, true, HaloDepth> &inGaugeA,
                 floatT stepSize, floatT start, floatT stop,
                             std::vector<floatT> necessaryFlowTime, floatT accuracy)
            : su3rungeKutta3AdStepSize<floatT, HaloDepth, Z_i>(inGaugeA, stepSize, start, stop,
                             necessaryFlowTime, accuracy) {}

};

template<class floatT, const size_t HaloDepth>
class zeuthenFlow<floatT, HaloDepth, adaptive_stepsize_allgpu>
        : public su3rungeKutta3AdStepSizeAllGPU<floatT, HaloDepth, Z_i_zeuthen<floatT, HaloDepth, true> > {

    using Z_i = Z_i_zeuthen<floatT, HaloDepth, true>;
public:
    zeuthenFlow(Gaugefield<floatT, true, HaloDepth> &inGaugeA,
                floatT stepSize, floatT start, floatT stop,
                std::vector<floatT> necessaryFlowTime, floatT accuracy)
            : su3rungeKutta3AdStepSizeAllGPU<floatT, HaloDepth, Z_i>(inGaugeA, stepSize, start, stop,
                                                               necessaryFlowTime, accuracy) {}

};
#endif
#endif

#endif //GRADIENTFLOW_H
