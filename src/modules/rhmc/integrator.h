/*
 * integrator.h
 *
 * P. Scior
 *
 */

#ifndef INTEGRATOR
#define INTEGRATOR

#include "../../gauge/gaugefield.h"
#include "./rhmcParameters.h"
#include "../../base/math/su3.h"
#include "../../gauge/gaugeActionDeriv.h"
#include "../../base/latticeContainer.h"
#include <iostream>
#include "../../spinor/spinorfield.h"
#include "../dslash/dslash.h"
#include "../../base/gutils.h"
#include "../hisq/hisqForce.h"
#include "spinorfield_container.h"


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t HaloDepthSpin>
class integrator {
public:
    integrator(RhmcParameters rhmc_param, Gaugefield<floatT, onDevice, HaloDepth, R18> &gaugeField,
               Gaugefield<floatT, onDevice, HaloDepth> &p,
               Gaugefield<floatT, onDevice, HaloDepth, U3R14> &X,
               Gaugefield<floatT, onDevice, HaloDepth> &W,
               HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, 1> &dslash,
               RationalCoeff rat, HisqSmearing<floatT, onDevice, HaloDepth, R18, R18, R18, U3R14> &smearing)
        : _gaugeField(gaugeField),
          _p(p),
          _X(X),
          _W(W),
          _rhmc_param(rhmc_param),
          gAcc(gaugeField.getAccessor()),
          pAccessor(p.getAccessor()),
          _dslash(dslash),
          ipdot(gaugeField.getComm()),         // "force" gaugeField object, p-dot
          ipdotAccessor(ipdot.getAccessor()),
          _rat(rat),
          _smearing(smearing),
          _dslashM(_W, _X, 0.0), // ip_dot_f2_hisq is the hisq force object, used to calculate stuff
          ip_dot_f2_hisq(_gaugeField, ipdot, cgM, _dslash, _dslashM, _rhmc_param, _rat, _smearing) {};

    ~integrator() {};

    void integrate(Spinorfield_container<floatT, onDevice, Even, HaloDepthSpin> &_phi_lf_container, Spinorfield_container<floatT, onDevice, Even, HaloDepthSpin> &_phi_sf_container);

private:
    // methods to evolve P and Q
    void updateP_gaugeforce(floatT stepsize);

    void updateP_fermforce(floatT stepsize, Spinorfield_container<floatT, onDevice, Even, HaloDepthSpin> &phi, bool light);

    void evolveQ(floatT stepsize);

    void evolveP(floatT stepsize);

    void check_traceless();

    void forceinfo();
    floatT forceinfo2();

    //this is only a placeholder function for testing, real implementation is in HISQ force! REMOVE for production!
    void make_f0(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &phi, std::vector <floatT> rat_coeff);

    // The different integration schemes
    void SWleapfrog(Spinorfield_container<floatT, onDevice, Even, HaloDepthSpin> &_phi_lf_container, Spinorfield_container<floatT, onDevice, Even, HaloDepthSpin> &_phi_sf_container);
    void PQPQP2MN(Spinorfield_container<floatT, onDevice, Even, HaloDepthSpin> &_phi_lf_container, Spinorfield_container<floatT, onDevice, Even, HaloDepthSpin> &_phi_sf_container);

    void PureGaugeleapfrog();

    Gaugefield<floatT, onDevice, HaloDepth, R18> &_gaugeField;
    Gaugefield<floatT, onDevice, HaloDepth, U3R14> &_X;
    Gaugefield<floatT, onDevice, HaloDepth> &_W;
    Gaugefield<floatT, onDevice, HaloDepth> &_p;
    HisqSmearing<floatT, onDevice, HaloDepth, R18, R18, R18, U3R14> &_smearing;

    RhmcParameters _rhmc_param;
    RationalCoeff _rat;
    const int _no_pf = _rhmc_param.no_pf();

    HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, 1> &_dslash;
    HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, 12> _dslashM;
    AdvancedMultiShiftCG<floatT, 12> cgM;

    Gaugefield<floatT, onDevice, HaloDepth> ipdot;

    HisqForce<floatT, onDevice, HaloDepth, 4> ip_dot_f2_hisq;

    SU3Accessor<floatT, R18> gAcc;
    SU3Accessor<floatT> pAccessor;
    SU3Accessor<floatT> ipdotAccessor;
};




//Pure gauge Leapfrogger
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
class pure_gauge_integrator {

public:
    pure_gauge_integrator(RhmcParameters rhmc_param, Gaugefield<floatT, onDevice, HaloDepth, comp> &gaugeField,
                          Gaugefield<floatT, onDevice, HaloDepth> &p)
            : _gaugeField(gaugeField), _p(p), _rhmc_param(rhmc_param), gAcc(gaugeField.getAccessor()),
              pAccessor(p.getAccessor()), ipdot(gaugeField.getComm()),
              ipdotAccessor(ipdot.getAccessor()) {};

    ~pure_gauge_integrator() {};

    void integrate();

private:
    // methods to evolve P and Q
    void updateP_gaugeforce(floatT stepsize);

    void evolveQ(floatT stepsize);

    void evolveP(floatT stepsize);

    void check_traceless();

    void PureGaugeleapfrog();

    Gaugefield<floatT, onDevice, HaloDepth, comp> &_gaugeField;
    Gaugefield<floatT, onDevice, HaloDepth> &_p;

    RhmcParameters _rhmc_param;
    const int _no_pf = _rhmc_param.no_pf();

    Gaugefield<floatT, onDevice, HaloDepth> ipdot;

    SU3Accessor<floatT, comp> gAcc;
    SU3Accessor<floatT> pAccessor;
    SU3Accessor<floatT> ipdotAccessor;

};

#endif //INTEGRATOR
