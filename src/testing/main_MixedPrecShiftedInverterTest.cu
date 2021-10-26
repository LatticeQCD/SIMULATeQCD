/* 
 * main_MixedPrecShiftedInverterTest.cu                                                               
 * 
 * Dennis Bollweg
 * 
 */

#include "../SIMULATeQCD.h"
#include "../modules/dslash/dslash.h"
#include "../modules/rhmc/rhmcParameters.h"
#include "../modules/HISQ/hisqSmearing.h"

//Run function. Please start reading here.
template<class floatT, Layout LatLayout, Layout LatLayoutRHS, size_t NStacks, size_t Multistack, bool onDevice>
void run_func(CommunicationBase &commBase, RhmcParameters &param, RationalCoeff &rat)
{
    bool success = true;

    //Initialization as usual
    
    const int HaloDepth = 2;
    const int HaloDepthSpin = 4;
    typedef GIndexer<LatLayoutRHS, HaloDepth> GInd;

    //Our gaugefield
    Gaugefield<floatT, onDevice, HaloDepth, R14> gauge(commBase);

    Gaugefield<floatT, onDevice, HaloDepth, R18> gauge_smeared(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, U3R14> gauge_Naik(commBase);
    Gaugefield<__half, onDevice, HaloDepth, R18> gauge_smeared_half(commBase);
    Gaugefield<__half, onDevice, HaloDepth, U3R14> gauge_naik_half(commBase);
    // Gaugefield<floatT, onDevice, HaloDepth, R18> gauge_smeared_stored(commBase);
    // Gaugefield<floatT, onDevice, HaloDepth, U3R14> gauge_Naik_stored(commBase);

    HisqSmearing<floatT, onDevice, HaloDepth> smearing(gauge, gauge_smeared, gauge_Naik);
    
    // rootLogger.info() << "Read configuration";
    // gauge.readconf_nersc("/work/conf/stli/mso320/l808f21b6390m000216875m0694s_01/l808f21b6390m000216875m0694s_01.2130");
    grnd_state<onDevice> d_rand;
    initialize_rng(13333+2130, d_rand);

    gauge.random(d_rand.state);

    // gauge.updateAll();
    smearing.SmearAll();
    
    gauge_smeared_half.template convert_precision<floatT>(gauge_smeared);
    gauge_naik_half.template convert_precision<floatT>(gauge_Naik);
    
    rootLogger.info() << "Initialize random state";

    GaugeAction<floatT, onDevice, HaloDepth, R18> gaugeaction(gauge_smeared);

    rootLogger.info() << "Whats the avg. plaquette of lvl2 smeared field? " << gaugeaction.plaquette();

    rootLogger.info() << "Initialize spinors";

    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, 1> eta(commBase);
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, 1> eta2(commBase);
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, 1> eta3(commBase);

    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> spinorIn(commBase);
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> spinorOut(commBase);
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> spinorOut2(commBase);
    Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks> spinorOut3(commBase);

    Spinorfield<floatT, false, LatLayoutRHS, HaloDepthSpin, NStacks> spinorHost(commBase);

    rootLogger.info() << "Randomize spinors";
    spinorIn.gauss(d_rand.state);

    // spinorIn.copyFromStackToStack(eta, 0, 0);
    // spinorIn.copyFromStackToStack(eta2, 1, 0);
    // spinorIn.copyFromStackToStack(eta3, 2, 0);

    SimpleArray<GCOMPLEX(double), NStacks> dot1(0.0);

    rootLogger.info() << "Simplest possible test!";

    dot1 = spinorIn.dotProductStacked(spinorIn);

    rootLogger.info() << "||spinorIn 0 ||^2 = " << dot1[0];

    spinorOut2.gauss(d_rand.state);

    rootLogger.info() << "Initialize DSlash";
    HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, NStacks> dslash_(gauge_smeared, gauge_Naik, 0.0);
    HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, NStacks> dslash(gauge_smeared, gauge_Naik, param.m_ud());
    HisqDSlash<__half, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, NStacks> dslash_half_(gauge_smeared_half, gauge_naik_half, 0.0);
    HisqDSlash<__half, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, NStacks> dslash_half(gauge_smeared_half, gauge_naik_half, param.m_ud());
    rootLogger.info() << "Very simple test";

    dslash_.Dslash(spinorOut3, spinorIn, true);
    
    dot1 = spinorOut3.dotProductStacked(spinorOut3);

    dslash.applyMdaggM(spinorOut, spinorIn);
    dot1 = spinorOut.dotProductStacked(spinorOut);

    rootLogger.info() << "|| D^+D eta ||^2 = " << dot1[0];     

    // spinorIn = spinorOut2 * floatT(param.m_ud()) - spinorOut3;

    dot1 = spinorIn.dotProductStacked(spinorIn);

    for (int i = 0; i < NStacks; ++i)
    {
        rootLogger.info() << " || SpinorIn ||^2 /vol4 [" << i << "] = " << dot1[i]/floatT(GInd::getLatData().globvol4);
    }

    ConjugateGradient<floatT, NStacks> cg;
    // AdvancedMultiShiftCG<floatT, NStacks> cgM;
    rootLogger.info() << "Run inversion";
    cg.invert_mrel(dslash, dslash_half, spinorOut, spinorIn, param.cgMax(), param.residue());

    SimpleArray<floatT, NStacks> mass(param.m_ud()*param.m_ud());

    // cgM.invert(dslash, spinorOut, spinorIn, mass, param.cgMax(), param.residue());

    dslash.applyMdaggM(spinorOut2, spinorOut);

    // spinorOut2 = spinorOut *floatT(param.m_ud()*param.m_ud()) - spinorOut2;

    spinorOut = spinorOut2 - spinorIn;

    // SimpleArray<GCOMPLEX(floatT), NStacks> dot1(0);
    SimpleArray<GCOMPLEX(double), NStacks> dot2(0.0);

    dot1 = spinorOut.dotProductStacked(spinorOut);
    dot2 = spinorIn.dotProductStacked(spinorIn);

    SimpleArray<double, NStacks> err_arr(0.0);

    err_arr = real<double>(dot1)/real<double>(dot2);

    for (size_t i = 0; i < NStacks; ++i)
    {
        rootLogger.info() << err_arr[i];
    }

    rootLogger.info() << "relative error of (D^+D) * (D^+D)^-1 * phi - phi : " << max(err_arr) ;

    if (!(max(err_arr) < 1e-5))
        success = success && false;


    rootLogger.info() << "Testing multishift inverter:";
    gSite origin;
    HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, 1> dslashMulti_ud(gauge_smeared, gauge_Naik, param.m_ud());
    HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, 1> dslashMulti_s(gauge_smeared, gauge_Naik, param.m_s());
    HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, 1> dslashMulti(gauge_smeared, gauge_Naik, 0.0);
    HisqDSlash<__half, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, 1> dslashMulti_ud_half(gauge_smeared_half, gauge_naik_half, param.m_ud());
    HisqDSlash<__half, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, 1> dslashMulti_s_half(gauge_smeared_half, gauge_naik_half, param.m_s());
    HisqDSlash<__half, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, 1> dslashMulti_half(gauge_smeared_half, gauge_naik_half, 0.0);
    // stdStagDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, 1> dslashMulti(gauge, 0.0);
    // stdStagDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, 1> dslashMulti_ud(gauge, param.m_ud());
    // stdStagDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, 1> dslashMulti_s(gauge, param.m_s());

    SimpleArray<floatT, Multistack> shifts(0.0);

    for (size_t i = 1; i <rat.r_inv_2f_den.get().size(); ++i)
    {
        shifts[i] = rat.r_inv_2f_den[i]-rat.r_inv_2f_den[0];
    }
    shifts[0] = rat.r_inv_2f_den[0] +param.m_ud()*param.m_ud();


    for (size_t i = 0; i < rat.r_inv_2f_den.get().size(); ++i)
    {   
        rootLogger.info() << "shifts: " << shifts[i];
    }


    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, 1> spinorInMulti(commBase);
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, 1> spinortmp(commBase);
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, 1> phi(commBase);
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, 1> spinorOutSingle(commBase);
    // spinorInMulti.one();
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, Multistack> spinorOutMulti(commBase);


    AdvancedMultiShiftCG<floatT, Multistack> cgM;
    // MultiShiftCG<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, Multistack> cgM;


    rootLogger.info() << "The xi test as in ye olde code:";
    spinorInMulti.one();

    spinortmp.copyFromStackToStack(spinorOutMulti, 0, 0);
    spinorHost = spinortmp;
    rootLogger.info() << "0th component of xi[" << 0 << "] =" << spinorHost.getAccessor().getElement(origin); 

    cgM.invert_mixed(dslashMulti, dslashMulti_half,spinorOutMulti, spinorInMulti, shifts, param.cgMax(), param.residue_force());


    for (size_t i = 0; i < rat.r_bar_1f_den.get().size(); ++i)
    {
        spinortmp.copyFromStackToStack(spinorOutMulti, 0, i);
        spinorHost = spinortmp;
        rootLogger.info() << "0th component of xi[" << i << "] =" << spinorHost.getAccessor().getElement(origin);    
    }


    //////////////////////////////////
    // r_2f test
    //////////////////////////////////

    rootLogger.info() << "Starting r_2f test";

    spinorInMulti.gauss(d_rand.state);

    // make phi:

    for (size_t i = 1; i <rat.r_inv_2f_den.get().size(); ++i)
    {
        shifts[i] = rat.r_inv_2f_den[i]-rat.r_inv_2f_den[0];
    }
    shifts[0] = rat.r_inv_2f_den[0] +param.m_ud()*param.m_ud();

    cgM.invert_mixed(dslashMulti, dslashMulti_half, spinorOutMulti, spinorInMulti, shifts, param.cgMax(), param.residue());

    phi = floatT(rat.r_inv_2f_const()) * spinorInMulti;

    for (size_t i = 0; i < rat.r_inv_2f_den.get().size(); ++i)
    {
        spinortmp.copyFromStackToStack(spinorOutMulti, 0, i);
        phi = phi + floatT(rat.r_inv_2f_num[i])*spinortmp;
    }

    

    

    ////////////
    // make r_2f^4

    spinorInMulti = phi;

    for (size_t i = 1; i <rat.r_2f_den.get().size(); ++i)
    {
        shifts[i] = rat.r_2f_den[i]-rat.r_2f_den[0];
    }
    shifts[0] = rat.r_2f_den[0] +param.m_ud()*param.m_ud();

    for (int j = 0; j < 4; ++j)
    {

        cgM.invert_mixed(dslashMulti, dslashMulti_half, spinorOutMulti, spinorInMulti, shifts, param.cgMax(), param.residue());

        spinorOutSingle = floatT(rat.r_2f_const()) * spinorInMulti;

        for (size_t i = 0; i < rat.r_2f_den.get().size(); ++i)
        {
            spinortmp.copyFromStackToStack(spinorOutMulti, 0, i);
            spinorOutSingle = spinorOutSingle + floatT(rat.r_2f_num[i])*spinortmp;
            // spinorOutSingle.axpyThisB(floatT(rat.r_2f_num[i]), spinortmp);
        } 
        spinorOutSingle.updateAll();

        spinorInMulti = spinorOutSingle;

    }

    spinorHost = spinorOutSingle;

    

    // rootLogger.info() << "0th component of r_2f^4 * phi =" << spinorHost.getAccessor().getElement(origin);


    ///////////
    // make (Ds^+ Ds)^-1
    cg.invert_mrel(dslashMulti_s, dslashMulti_s_half,spinorOutSingle, spinorInMulti, param.cgMax(), param.residue());
    ///////////
    // make (Dl^+ Dl)
    dslashMulti_ud.applyMdaggM(spinortmp, spinorOutSingle);

    spinortmp = spinortmp - phi;

    double diff_normsq = real<double>(spinortmp.dotProduct(spinortmp));
    double phi_sq = real<double>(phi.dotProduct(phi));

    rootLogger.info() << "relative error of (Dl^+Dl) * (Ds^+ Ds)^-1 * r_2f^4 * phi - phi = " << abs(diff_normsq/phi_sq);
    if (!(abs(diff_normsq/phi_sq) < 1e-5))
        success = success && false;


    //////////////////////////////////
    ////r_bar_2f_test
    //////////////////////////////////
    rootLogger.info() << "Starting r_bar_2f test";

    SimpleArray<floatT, 12> shifts_bar(0.0);

    for (size_t i = 1; i <rat.r_bar_2f_den.get().size(); ++i)
    {
        shifts_bar[i] = rat.r_bar_2f_den[i]-rat.r_bar_2f_den[0];
    }
    shifts_bar[0] = rat.r_bar_2f_den[0] +param.m_ud()*param.m_ud();


    spinorInMulti.gauss(d_rand.state);
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, 12> spinorOutMulti_bar(commBase);


    AdvancedMultiShiftCG<floatT, 12> cgM_bar;


    ////////////
    // make r_bar_2f^2

    spinorInMulti = phi;

    for (int j = 0; j < 2; ++j)
    {

        cgM_bar.invert_mixed(dslashMulti, dslashMulti_half, spinorOutMulti_bar, spinorInMulti, shifts_bar, param.cgMax(), param.residue_force());

        spinorOutSingle = floatT(rat.r_bar_2f_const()) * spinorInMulti;

        for (size_t i = 0; i < rat.r_bar_2f_den.get().size(); ++i)
        {
            spinortmp.copyFromStackToStack(spinorOutMulti_bar, 0, i);
            spinorOutSingle = spinorOutSingle + floatT(rat.r_bar_2f_num[i])*spinortmp;
            // spinorOutSingle.axpyThisB(floatT(rat.r_2f_num[i]), spinortmp);
        } 
        spinorOutSingle.updateAll();

        spinorInMulti = spinorOutSingle;

    }


    ///////////
    // make (Ds^+ Ds)^-1
    cg.invert_mrel(dslashMulti_s, dslashMulti_s_half, spinorOutSingle, spinorInMulti, param.cgMax(), param.residue());
    ///////////
    // make (Dl^+ Dl)
    dslashMulti_ud.applyMdaggM(spinortmp, spinorOutSingle);

    spinortmp = spinortmp - phi;

    diff_normsq = real<double>(spinortmp.dotProduct(spinortmp));
    phi_sq = real<double>(phi.dotProduct(phi));

    rootLogger.info() << "relative error of (Dl^+Dl) * (Ds^+ Ds)^-1 * r_bar_2f^4 * phi - phi = " << abs(diff_normsq/phi_sq);
    if (!(abs(diff_normsq/phi_sq) < 1e-5))
        success = success && false;





    // r_inv_2f test

    rootLogger.info() << "Starting r_inv_2f test";

    spinorInMulti = phi;

    for (size_t i = 1; i <rat.r_2f_den.get().size(); ++i)
    {
        shifts[i] = rat.r_inv_2f_den[i]- rat.r_inv_2f_den[0];
    }
    shifts[0] = rat.r_inv_2f_den[0] +param.m_ud()*param.m_ud();

    for (int j = 0; j < 4; ++j)
    {
        cgM.invert_mixed(dslashMulti, dslashMulti_half, spinorOutMulti, spinorInMulti, shifts, param.cgMax(), param.residue());

        spinorOutSingle = floatT(rat.r_inv_2f_const()) * spinorInMulti;

        for (size_t i = 0; i < rat.r_inv_2f_den.get().size(); ++i)
        {
            spinortmp.copyFromStackToStack(spinorOutMulti, 0, i);
            spinorOutSingle = spinorOutSingle + floatT(rat.r_inv_2f_num[i])*spinortmp;
        } 

        spinorInMulti = spinorOutSingle;

    }
    ///////////
    // make (Dl^+ Dl)

    dslashMulti_ud.applyMdaggM(spinorInMulti, phi);

    ///////////
    // make (Ds^+ Ds)^-1
    cg.invert_mrel(dslashMulti_s,dslashMulti_s_half, spinortmp, spinorInMulti, param.cgMax(), param.residue());

    spinortmp = spinortmp - spinorOutSingle;

    diff_normsq = real<double>(spinortmp.dotProduct(spinortmp));
    phi_sq = real<double>(spinorOutSingle.dotProduct(spinorOutSingle));

    rootLogger.info() << "relative error of r_inv_2f^4 * phi - (Dl^+Dl) * (Ds^+ Ds)^-1 * phi = " << abs(diff_normsq/phi_sq);
    if (!(abs(diff_normsq/phi_sq) < 1e-5))
        success = success && false;
    
    //////////////////////////////////
    // r_1f test
    //////////////////////////////////

    spinorInMulti.gauss(d_rand.state);

    for (size_t i = 1; i <rat.r_inv_1f_den.get().size(); ++i)
    {
        shifts[i] = rat.r_inv_1f_den[i]- rat.r_inv_1f_den[0];
    }
    shifts[0] = rat.r_inv_1f_den[0] + param.m_s() * param.m_s();

    // make phi:
    cgM.invert_mixed(dslashMulti, dslashMulti_half, spinorOutMulti, spinorInMulti, shifts, param.cgMax(), param.residue());

    phi = floatT(rat.r_1f_const()) * spinorInMulti;

    for (size_t i = 0; i < rat.r_1f_den.get().size(); ++i)
    {
        spinortmp.copyFromStackToStack(spinorOutMulti, 0, i);
        phi = phi + floatT(rat.r_inv_1f_num[i])*spinortmp;
    }

    // make r_1f^8

    spinorInMulti = phi;

    for (size_t i = 1; i <rat.r_1f_den.get().size(); ++i)
    {
        shifts[i] = rat.r_1f_den[i]- rat.r_1f_den[0];
    }
    shifts[0] = rat.r_1f_den[0] + param.m_s() * param.m_s();

    for (int j = 0; j < 8; ++j)
    {
        cgM.invert_mixed(dslashMulti, dslashMulti_half, spinorOutMulti, spinorInMulti, shifts, param.cgMax(), param.residue());

        spinorOutSingle = floatT(rat.r_1f_const()) * spinorInMulti;

        for (size_t i = 0; i < rat.r_1f_den.get().size(); ++i)
        {
            spinortmp.copyFromStackToStack(spinorOutMulti, 0, i);
            spinorOutSingle = spinorOutSingle + floatT(rat.r_1f_num[i])*spinortmp;
        } 
        spinorInMulti = spinorOutSingle;
    }

    // make (Ds^+Ds)^3
    for (int j = 0; j < 3; ++j)
    {
        dslashMulti_s.applyMdaggM(spinorOutSingle, spinorInMulti, true);

        spinorInMulti = spinorOutSingle;
    }

    spinorOutSingle = spinorOutSingle - phi;

    diff_normsq = real<double>(spinorOutSingle.dotProduct(spinorOutSingle));
    phi_sq = real<double>(phi.dotProduct(phi));

    rootLogger.info() << "relative error of (Ds^+*Ds)^3 * r_1f^8 * phi - phi = " << abs(diff_normsq/phi_sq);
    if (!(abs(diff_normsq/phi_sq) < 1e-5))
        success = success && false;


    //////////////////////////////////
    // r_bar_1f test
    //////////////////////////////////

    // make r_bar_1f^4


    spinorInMulti = phi;

    for (size_t i = 1; i <rat.r_bar_1f_den.get().size(); ++i)
    {
        shifts_bar[i] = rat.r_bar_1f_den[i]- rat.r_bar_1f_den[0];
    }
    shifts_bar[0] = rat.r_bar_1f_den[0] + param.m_s() * param.m_s();

    for (int j = 0; j < 4; ++j)
    {
        cgM_bar.invert_mixed(dslashMulti, dslashMulti_half,spinorOutMulti_bar, spinorInMulti, shifts_bar, param.cgMax(), param.residue_force());

        spinorOutSingle = floatT(rat.r_bar_1f_const()) * spinorInMulti;

        for (size_t i = 0; i < rat.r_bar_1f_den.get().size(); ++i)
        {
            spinortmp.copyFromStackToStack(spinorOutMulti_bar, 0, i);
            spinorOutSingle = spinorOutSingle + floatT(rat.r_bar_1f_num[i])*spinortmp;
        } 
        spinorInMulti = spinorOutSingle;
    }

    // make (Ds^+Ds)^3
    for (int j = 0; j < 3; ++j)
    {
        dslashMulti_s.applyMdaggM(spinorOutSingle, spinorInMulti, true);

        spinorInMulti = spinorOutSingle;
    }

    spinorOutSingle = spinorOutSingle - phi;

    diff_normsq = real<double>(spinorOutSingle.dotProduct(spinorOutSingle));
    phi_sq = real<double>(phi.dotProduct(phi));

    rootLogger.info() << "relative error of (Ds^+*Ds)^3 * r_bar_1f^4 * phi - phi = " << abs(diff_normsq/phi_sq);
    if (!(abs(diff_normsq/phi_sq) < 1e-5))
        success = success && false;
    //////////////////////////////////
    // r_inv_1f test
    //////////////////////////////////

    // make r_inv_1f^8

    spinorInMulti = phi;

    for (size_t i = 1; i <rat.r_inv_1f_den.get().size(); ++i)
    {
        shifts[i] = rat.r_inv_1f_den[i]- rat.r_inv_1f_den[0];
    }
    shifts[0] = rat.r_inv_1f_den[0] + param.m_s() * param.m_s();

    for (int j = 0; j < 8; ++j)
    {
        cgM.invert_mixed(dslashMulti, dslashMulti_half,spinorOutMulti, spinorInMulti, shifts, param.cgMax(), param.residue());

        spinorOutSingle = floatT(rat.r_inv_1f_const()) * spinorInMulti;

        for (size_t i = 0; i < rat.r_inv_1f_den.get().size(); ++i)
        {
            spinortmp.copyFromStackToStack(spinorOutMulti, 0, i);
            spinorOutSingle = spinorOutSingle + floatT(rat.r_inv_1f_num[i])*spinortmp;
        } 
        spinorInMulti = spinorOutSingle;
    }

    // make (Ds^+Ds)^3
    for (int j = 0; j < 3; ++j)
    {
        dslashMulti_s.applyMdaggM(spinorOutSingle, phi, true);

        phi = spinorOutSingle;
    }

    spinorOutSingle = spinorInMulti - phi;

    diff_normsq = real<double>(spinorOutSingle.dotProduct(spinorOutSingle));
    phi_sq = real<double>(phi.dotProduct(phi));

    rootLogger.info() << "relative error of  r_inv_1f^8 * phi - (Ds^+*Ds)^3 *phi = " << abs(diff_normsq/phi_sq);
    if (!(abs(diff_normsq/phi_sq) < 1e-5))
        success = success && false;

    if (!(abs(diff_normsq/phi_sq) < 1e-5))
        success = success && false;

    if (success)
        rootLogger.info() << "Inverter tests: " << CoutColors::green << "passed" << CoutColors::reset;
    else
        rootLogger.info() << "Inverter tests: " << CoutColors::red << "failed" << CoutColors::reset;
}



int main(int argc, char **argv) {

    stdLogger.setVerbosity(DEBUG);

    CommunicationBase commBase(&argc, &argv);
    RhmcParameters param;
    param.readfile(commBase, "../parameter/tests/MixedPrecInverterTest.param", argc, argv);

    RationalCoeff rat;
    rat.readfile(commBase, param.rat_file());

    commBase.init(param.nodeDim(), param.gpuTopo());

    // const int HaloDepth = 2;
    // initIndexer(HaloDepth,param, commBase);
    const int HaloDepthSpin = 4;
    initIndexer(HaloDepthSpin,param, commBase);

    rootLogger.info() << "-------------------------------------";
    rootLogger.info() << "Running on Device";
    rootLogger.info() << "-------------------------------------";

    rootLogger.info() << "Testing Even - Odd";
    rootLogger.info() << "------------------";
    // run_func<double, Even, Odd, 10, 14, true>(commBase, param, rat);
    run_func<float, Even, Odd, 1, 14, true>(commBase, param, rat);

    // rootLogger.info() << "------------------";
    // rootLogger.info() << "Testing Even - Odd";
    // rootLogger.info() << "------------------";
    // run_func<float, Even, Odd, 1, 14, true>(commBase, param, rat);
    // rootLogger.info() << "------------------";
    // rootLogger.info() << "Testing Odd - Even";
    // rootLogger.info() << "------------------";
    // run_func<double, Odd, Even, 1, 14, true>(commBase, param, rat);

    /// Apparently the host has trouble to store a configuration.
    //    rootLogger.info() << "-------------------------------------";
    //    rootLogger.info() << "Running on Host";
    //    rootLogger.info() << "-------------------------------------";
    //    rootLogger.info() << "Testing All - All";
    //    rootLogger.info() << "------------------";
    //    run_func<double, All, All, 1, 14, false>(commBase, param, rat);
    //    rootLogger.info() << "------------------";
    //    rootLogger.info() << "Testing Even - Odd";
    //    rootLogger.info() << "------------------";
    //    run_func<double, Even, Odd, 1, 14, false>(commBase, param, rat);
    //    rootLogger.info() << "------------------";
    //    rootLogger.info() << "Testing Odd - Even";
    //    rootLogger.info() << "------------------";
    //    run_func<double, Odd, Even, 1, 14,false>(commBase, param, rat);
}

template<Layout LatLayout, size_t HaloDepth>
size_t getGlobalIndex(LatticeDimensions coord) {
    typedef GIndexer<LatLayout, HaloDepth> GInd;

    LatticeData lat = GInd::getLatData();
    LatticeDimensions globCoord = lat.globalPos(coord);

    return globCoord[0] + globCoord[1] * lat.globLX + globCoord[2] * lat.globLX * lat.globLY +
        globCoord[3] * lat.globLX * lat.globLY * lat.globLZ;
}
