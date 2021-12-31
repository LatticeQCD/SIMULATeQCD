//
// Created by Jishnu on 07/04/20.
//

#include "../SIMULATeQCD.h"
#include "../modules/dslash/dslash.h"
#include "../modules/rhmc/rhmcParameters.h"
#include "../modules/HISQ/hisqSmearing.h"

template <class floatT>
__host__ bool operator==(const gVect3<floatT> &lhs, const gVect3<floatT> &rhs){

    bool ret = true;

    floatT val_lhs, val_rhs;

    val_lhs = lhs.getElement0().cREAL;
    val_rhs = rhs.getElement0().cREAL;

    ret = ret && cmp_rel(val_lhs, val_rhs, 1.e-4, 1e-4);

    val_lhs = lhs.getElement0().cIMAG;
    val_rhs = rhs.getElement0().cIMAG;

    ret = ret && cmp_rel(val_lhs, val_rhs, 1.e-4, 1e-4);

    val_lhs = lhs.getElement1().cREAL;
    val_rhs = rhs.getElement1().cREAL;

    ret = ret && cmp_rel(val_lhs, val_rhs, 1.e-4, 1e-4);

    val_lhs = lhs.getElement1().cIMAG;
    val_rhs = rhs.getElement1().cIMAG;

    ret = ret && cmp_rel(val_lhs, val_rhs, 1.e-4, 1e-4);

    val_lhs = lhs.getElement2().cREAL;
    val_rhs = rhs.getElement2().cREAL;

    ret = ret && cmp_rel(val_lhs, val_rhs, 1.e-4, 1e-4);

    val_lhs = lhs.getElement2().cIMAG;
    val_rhs = rhs.getElement2().cIMAG;

    ret = ret && cmp_rel(val_lhs, val_rhs, 1.e-4, 1e-4);

    return ret;
}

//the Dslash test function. Please start reading here.
template<class floatT, Layout LatLayout, Layout LatLayoutRHS, size_t NStacks, bool onDevice>
void test_dslash(CommunicationBase &commBase){

    //Initialization as usual
    const int HaloDepth = 0;
    const int HaloDepthSpin = 0;
    typedef GIndexer<LatLayout, HaloDepth> GInd;

    //Our gaugefield
    Gaugefield<floatT, onDevice, HaloDepth, R14> gauge(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, R18> gauge_smeared(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, U3R14> gauge_Naik(commBase);
    double mu=0.4;

    HisqSmearing<floatT, onDevice, HaloDepth> smearing(gauge, gauge_smeared, gauge_Naik);

    rootLogger.info("Read conf");

    gauge.readconf_nersc("../test_conf/gauge12750");

    gauge.updateAll();

    smearing.SmearAll(mu);

    rootLogger.info( "Initialize random state");
    grnd_state<false> h_rand;
    grnd_state<onDevice> d_rand;

    h_rand.make_rng_state(1337);
    d_rand = h_rand;

    rootLogger.info("Initialize spinors");
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> spinorIn(commBase);
    Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks> spinorOut(commBase);
    Spinorfield<floatT, false, LatLayout, HaloDepthSpin, NStacks> spinorOut2(commBase);

    rootLogger.info( "Randomize spinors");
    spinorIn.gauss(d_rand.state);

    rootLogger.info( "Initialize DSlash");
    HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, NStacks> dslash(gauge_smeared, gauge_Naik, 0.0);

    dslash.Dslash(spinorOut, spinorIn);

    spinorOut2 = spinorOut;

    gVect3<floatT> OE_vec_0 = gVect3<floatT>(GCOMPLEX(floatT)(2.28094,0.578288),GCOMPLEX(floatT)(-1.57489,-3.40721),
                                             GCOMPLEX(floatT)(-0.0483184,-1.14394));
    gVect3<floatT> OE_vec_512 = gVect3<floatT>(GCOMPLEX(floatT)(-1.29579,0.339849),GCOMPLEX(floatT)(-1.46313,1.60306),
                                               GCOMPLEX(floatT)(-1.52663,-2.28068));
    gVect3<floatT> OE_vec_697 = gVect3<floatT>(GCOMPLEX(floatT)(0.993064,-0.721346),GCOMPLEX(floatT)(-2.17415,-0.643141),
                                               GCOMPLEX(floatT)(-0.931775,-0.588071));
    gVect3<floatT> OE_vec_1023 = gVect3<floatT>(GCOMPLEX(floatT)(-0.683339,-0.542339),GCOMPLEX(floatT)(-1.36262,0.125968),
                                                GCOMPLEX(floatT)(0.255348,1.68404));

    gVect3<floatT> EO_vec_0 = gVect3<floatT>(GCOMPLEX(floatT)(0.483484,-0.159519),GCOMPLEX(floatT)(-2.79428,-1.11986),
                                             GCOMPLEX(floatT)(0.440891,-1.96765));
    gVect3<floatT> EO_vec_319 = gVect3<floatT>(GCOMPLEX(floatT)(1.26807,0.474252),GCOMPLEX(floatT)(-0.327903,-1.13687),
                                               GCOMPLEX(floatT)(-1.17472,0.362931));
    gVect3<floatT> EO_vec_550 = gVect3<floatT>(GCOMPLEX(floatT)(-0.230879,-1.36073),GCOMPLEX(floatT)(1.12482,-2.19585),
                                               GCOMPLEX(floatT)(1.15326,0.459365));
    gVect3<floatT> EO_vec_1023 = gVect3<floatT>(GCOMPLEX(floatT)(1.23632,-1.13617),GCOMPLEX(floatT)(-0.0638833,-0.507776),
                                                GCOMPLEX(floatT)(0.0369125,-0.770414));

    bool success[4]={false};

    if (LatLayoutRHS==Even)
    {
        gSite site = GInd::getSite(0);
        gVect3<floatT> vec = spinorOut2.getAccessor().getElement(site);
        if (vec==EO_vec_0)
        {
            success[0]=true;
        }
        site = GInd::getSite(319);
        vec = spinorOut2.getAccessor().getElement(site);
        if (vec==EO_vec_319)
        {
            success[1]=true;
        }
        site = GInd::getSite(550);
        vec = spinorOut2.getAccessor().getElement(site);
        if (vec==EO_vec_550)
        {
            success[2]=true;
        }
        site = GInd::getSite(1023);
        vec = spinorOut2.getAccessor().getElement(site);
        if (vec==EO_vec_1023)
        {
            success[3]=true;
        }
    }
    if (LatLayoutRHS==Odd)
    {
        gSite site = GInd::getSite(0);
        gVect3<floatT> vec = spinorOut2.getAccessor().getElement(site);
        if (vec==OE_vec_0)
        {
            success[0]=true;
        }
        site = GInd::getSite(512);
        vec = spinorOut2.getAccessor().getElement(site);
        if (vec==OE_vec_512)
        {
            success[1]=true;
        }
        site = GInd::getSite(697);
        vec = spinorOut2.getAccessor().getElement(site);
        if (vec==OE_vec_697)
        {
            success[2]=true;
        }
        site = GInd::getSite(1023);
        vec = spinorOut2.getAccessor().getElement(site);
        if (vec==OE_vec_1023)
        {
            success[3]=true;
        }
    }

    if (success[0] && success[1] && success[2] && success[3])
        rootLogger.info("Test of Dslash against old values: ",CoutColors::green, "passed" ,CoutColors::reset);
    else
        rootLogger.info("Test of Dslash against old values: " , CoutColors::red , "failed" , CoutColors::reset);

}

int main(int argc, char **argv) {

    stdLogger.setVerbosity(INFO);

    LatticeParameters param;
    const int LatDim[] = {8, 8, 8, 4};
    const int NodeDim[] = {1, 1, 1, 1};

    CommunicationBase commBase(&argc, &argv);

    param.latDim.set(LatDim);
    param.nodeDim.set(NodeDim);

    commBase.init(param.nodeDim());

    const int HaloDepth = 0;
    initIndexer(HaloDepth,param, commBase);
    const int HaloDepthSpin = 0;
    initIndexer(HaloDepthSpin,param, commBase);
    stdLogger.setVerbosity(INFO);

    rootLogger.info("Test with old values done for chemical potential::0.4");
    rootLogger.info("-------------------------------------");
    rootLogger.info( "Running on Device");
    rootLogger.info("-------------------------------------");
    rootLogger.info("Testing Even - Odd");
    rootLogger.info("------------------");
    test_dslash<float, Even, Odd, 1, true>(commBase);
    rootLogger.info( "------------------");
    rootLogger.info( "Testing Odd - Even");
    rootLogger.info("------------------");
    test_dslash<float, Odd, Even, 1, true>(commBase);
}


//template<Layout LatLayout, size_t HaloDepth>
//size_t getGlobalIndex(LatticeDimensions coord) {
//    typedef GIndexer<LatLayout, HaloDepth> GInd;
//
//    LatticeData lat = GInd::getLatData();
//    LatticeDimensions globCoord = lat.globalPos(coord);
//
//    return globCoord[0] + globCoord[1] * lat.globLX + globCoord[2] * lat.globLX * lat.globLY +
//           globCoord[3] * lat.globLX * lat.globLY * lat.globLZ;
//}
