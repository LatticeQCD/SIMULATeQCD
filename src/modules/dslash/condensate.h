#include "../../define.h"
#include "../../gauge/gaugefield.h"
#include "../rhmc/rhmcParameters.h"
#include "../../spinor/spinorfield.h"
#include "dslash.h"
#include "../inverter/inverter.h"
#include "../HISQ/hisqSmearing.h"

template<class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, size_t NStacks>
SimpleArray<double,NStacks> measure_condensate(CommunicationBase &commBase, RhmcParameters param, bool light, 
    Gaugefield<floatT, onDevice, HaloDepth, R18> &gauge, grnd_state<onDevice> &d_rand) {

    typedef GIndexer<All,HaloDepth> GInd;
    
    floatT mass;

    if (light) {
        mass = param.m_ud();
    } else {
        mass = param.m_s();
    }

    Gaugefield<floatT, onDevice, HaloDepth, U3R14> smeared_X(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, R18> smeared_W(commBase, "SHARED_GAUGELVL2");
    
    HisqSmearing<floatT, onDevice, HaloDepth, R18, R18, R18, U3R14> smearing(gauge, smeared_W, smeared_X);
    smearing.SmearAll(param.mu_f());
    
    ConjugateGradient<floatT, NStacks> cg;
    HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, NStacks> dslash_e(smeared_W, smeared_X, 0.0);
    HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, NStacks> dslash_e_inv(smeared_W, smeared_X, mass);
    HisqDSlash<floatT, onDevice, Odd, HaloDepth, HaloDepthSpin, NStacks> dslash_o(smeared_W, smeared_X, 0.0);

    Spinorfield<floatT, onDevice, Even, HaloDepthSpin, NStacks> eta_e(commBase);
    Spinorfield<floatT, onDevice, Odd, HaloDepthSpin, NStacks> eta_o(commBase);
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin, NStacks> x_e(commBase);
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin, NStacks> w_e(commBase);  
    Spinorfield<floatT, onDevice, Odd, HaloDepthSpin, NStacks> w_o(commBase);

    eta_o.gauss(d_rand.state);
    eta_e.gauss(d_rand.state);

    SimpleArray<double, NStacks> dot_e(0.0);
    SimpleArray<double, NStacks> dot_o(0.0);

    dslash_o.Dslash(x_e, eta_o);
    x_e = eta_e * mass - x_e;

    cg.invert_new(dslash_e_inv, w_e, x_e, param.cgMax_meas(), param.residue_meas());

    dslash_e.Dslash(w_o, w_e);
    w_o = floatT(1./mass)*(eta_o - w_o);

    dot_e = eta_e.realdotProductStacked(w_e);
    dot_o = eta_o.realdotProductStacked(w_o);

    return (dot_o + dot_e)/double(GInd::getLatData().globvol4);
};

