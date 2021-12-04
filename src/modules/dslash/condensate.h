#include "../../define.h"
#include "../../gauge/gaugefield.h"
#include "../rhmc/rhmcParameters.h"
#include "../../spinor/spinorfield.h"
#include "dslash.h"
#include "../inverter/inverter.h"
#include "../HISQ/hisqSmearing.h"

template<class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin, size_t NStacks>
void measure_condensate(CommunicationBase &commBase, RhmcParameters param, bool light, 
    Gaugefield<floatT, onDevice, HaloDepth, R18> &gauge, grnd_state<onDevice> &d_rand)
{
    typedef GIndexer<All,HaloDepth> GInd;

    // Gaugefield<floatT, onDevice, HaloDepth, U3R14> smeared_X(commBase, "SHARED_GAUGENAIK");
    // Gaugefield<floatT, onDevice, HaloDepth, R18> smeared_W(commBase, "SHARED_GAUGELVL2");

    // HisqSmearing<floatT, onDevice, HaloDepth, R18, R18, R18, U3R14> smearing(gauge, smeared_W, smeared_X);
    // smearing.SmearAll();
    
    floatT mass;

    if (light)
        mass = param.m_ud();
    else
        mass = param.m_s();


    // ConjugateGradient<floatT, NStacks> cg;
    // HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, NStacks> dslash_e(smeared_W, smeared_X, 0.0);
    // HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, NStacks> dslash_e_inv(smeared_W, smeared_X, mass);
    // HisqDSlash<floatT, onDevice, Odd, HaloDepth, HaloDepthSpin, NStacks> dslash_o(smeared_W, smeared_X, 0.0);

    // Spinorfield<floatT, onDevice, Even, HaloDepthSpin, NStacks> eta_e(commBase);
    // Spinorfield<floatT, onDevice, Odd, HaloDepthSpin, NStacks> eta_o(commBase);
    // Spinorfield<floatT, onDevice, Even, HaloDepthSpin, NStacks> x_e(commBase);
    // Spinorfield<floatT, onDevice, Even, HaloDepthSpin, NStacks> w_e(commBase);  
    // Spinorfield<floatT, onDevice, Odd, HaloDepthSpin, NStacks> w_o(commBase);

    // eta_e.gauss(d_rand.state);
    // eta_o.gauss(d_rand.state);


    // dslash_o.Dslash(x_e, eta_o);
    // x_e = eta_e * mass - x_e;

    // cg.invert(dslash_e_inv, w_e, x_e, param.cgMax(), param.residue());

    // dslash_e.Dslash(w_o, w_e);
    // w_o = static_cast<floatT>((1./mass))*(eta_o - w_o);

    // SimpleArray<floatT, NStacks> dot_e(0);
    // SimpleArray<floatT, NStacks> dot_o(0);

    // dot_e = eta_e.realdotProductStacked(w_e);
    // dot_o = eta_o.realdotProductStacked(w_o);

    // floatT condensate;

    // for (size_t i = 0; i < NStacks; ++i)
    // {

    //     condensate = dot_o[i]+dot_e[i];
    //     condensate/=GInd::getLatData().globvol4;

    //     if(light)
    //         rootLogger.info("chi_ud(" ,  i , ")= " ,  condensate);
    //     else
    //         rootLogger.info("chi_s(" ,  i , ")= " ,  condensate);
    // }


    Gaugefield<floatT, onDevice, HaloDepth, U3R14> smeared_X(commBase, "SHARED_GAUGENAIK");
    Gaugefield<floatT, onDevice, HaloDepth, R18> smeared_W(commBase, "SHARED_GAUGELVL2");
    /*
    //InitializeLinksHISQ<floatT,onDevice,HaloDepth> smearedlinks(gauge, smeared_X, smeared_W);
    //smearedlinks.SmearedLinks();
    getLvl2Smearing<floatT, onDevice, HaloDepth> smearlvl2(gauge, smeared_W);
    getNaikSmearing<floatT, onDevice, HaloDepth> smearnaik(gauge, smeared_X);
    smearlvl2.performSmearing();
    smearnaik.performSmearing();
    */
    HisqSmearing<floatT, onDevice, HaloDepth, R18, R18, R18, U3R14> smearing(gauge, smeared_W, smeared_X);
    smearing.SmearAll();
    
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

    // rootLogger.info("Starting inversion");

    // dot_e = x_e.dotProductStacked(x_e);

    // for (int i = 0; i < NStacks; ++i)
    // {
    //     rootLogger.info("x_e[" ,  i ,  "]^2 = " ,  dot_e[i]);
    // }

    cg.invert_new(dslash_e_inv, w_e, x_e, param.cgMax_meas(), param.residue_meas());

    // dot_o = w_e.dotProductStacked(w_e);

    // for (int i = 0; i < NStacks; ++i)
    // {
    //     rootLogger.info("w_e[" ,  i ,  "]^2 = " ,  dot_o[i]);
    // }

    dslash_e.Dslash(w_o, w_e);
    w_o = floatT(1./mass)*(eta_o - w_o);



    dot_e = eta_e.realdotProductStacked(w_e);
    dot_o = eta_o.realdotProductStacked(w_o);


    // double condensate=0.;

    for (size_t i = 0; i < NStacks; ++i)
    {
         if(light)
            rootLogger.info("CHI_UD = " ,   (dot_o[i]+dot_e[i])/floatT(GInd::getLatData().globvol4));
        else
            rootLogger.info("CHI_S = " ,   (dot_o[i]+dot_e[i])/floatT(GInd::getLatData().globvol4));
    }


};

