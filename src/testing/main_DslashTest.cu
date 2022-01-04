/* 
 * main_DslashTest.cu                                                               
 * 
 */

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
bool test_dslash(CommunicationBase &commBase){

    //Initialization as usual
    const int HaloDepth = 0;
    const int HaloDepthSpin = 0;
    typedef GIndexer<LatLayout, HaloDepth> GInd;

    //Our gaugefield
    Gaugefield<floatT, onDevice, HaloDepth, R14> gauge(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, R18> gauge_smeared(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, U3R14> gauge_Naik(commBase);

    HisqSmearing<floatT, onDevice, HaloDepth> smearing(gauge, gauge_smeared, gauge_Naik);
    
    rootLogger.info("Read conf");

    gauge.readconf_nersc("../test_conf/gauge12750");

    gauge.updateAll();

    smearing.SmearAll();

    rootLogger.info("Initialize random state");
    grnd_state<false> h_rand;
    grnd_state<onDevice> d_rand;

    h_rand.make_rng_state(1337);
    d_rand = h_rand;

    rootLogger.info("Initialize spinors");
    Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> spinorIn(commBase);
    Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks> spinorOut(commBase);
    Spinorfield<floatT, false, LatLayout, HaloDepthSpin, NStacks> spinorOut2(commBase);

    rootLogger.info("Randomize spinors");
    spinorIn.gauss(d_rand.state);

    rootLogger.info("Initialize DSlash");
    HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, NStacks> dslash(gauge_smeared, gauge_Naik, 0.0);
    
    dslash.Dslash(spinorOut, spinorIn);

    spinorOut2 = spinorOut;

    gVect3<floatT> OE_vec_0 = gVect3<floatT>(GCOMPLEX(floatT)(2.3391,0.133572),GCOMPLEX(floatT)(-1.2575,-3.71507),
        GCOMPLEX(floatT)(0.0448819,-1.03748));
    gVect3<floatT> OE_vec_512 = gVect3<floatT>(GCOMPLEX(floatT)(-1.41934,0.264733),GCOMPLEX(floatT)(-1.63475,1.55185),
        GCOMPLEX(floatT)(-1.71302,-2.05072));
    gVect3<floatT> OE_vec_697 = gVect3<floatT>(GCOMPLEX(floatT)(1.10125,-0.909113),GCOMPLEX(floatT)(-1.92929,-0.893694),
        GCOMPLEX(floatT)(-1.21505,-0.491814));
    gVect3<floatT> OE_vec_1023 = gVect3<floatT>(GCOMPLEX(floatT)(-1.01115,-0.838747),GCOMPLEX(floatT)(-1.45297,-0.0797096),
        GCOMPLEX(floatT)(-0.0330309,1.81019));

    gVect3<floatT> EO_vec_0 = gVect3<floatT>(GCOMPLEX(floatT)(0.479281,-0.210938),GCOMPLEX(floatT)(-2.60014,-1.10176),
        GCOMPLEX(floatT)(0.417858,-1.59543));
    gVect3<floatT> EO_vec_319 = gVect3<floatT>(GCOMPLEX(floatT)(0.877146,0.391793),GCOMPLEX(floatT)(-0.177829,-0.995388),
        GCOMPLEX(floatT)(-1.37693,0.670404));
    gVect3<floatT> EO_vec_550 = gVect3<floatT>(GCOMPLEX(floatT)(-0.102483,-1.74452),GCOMPLEX(floatT)(1.02125,-2.43817),
        GCOMPLEX(floatT)(1.1127,0.309778));
    gVect3<floatT> EO_vec_1023 = gVect3<floatT>(GCOMPLEX(floatT)(1.25156,-1.10202),GCOMPLEX(floatT)(0.190079,-0.295352),
        GCOMPLEX(floatT)(0.273009,-0.809811));

    bool success[4]={false};

    if (LatLayoutRHS==Even) {

        gSite site = GInd::getSite(0);
        gVect3<floatT> vec = spinorOut2.getAccessor().getElement(site);
        if (vec==EO_vec_0) {
            success[0]=true;
        }
        site = GInd::getSite(319);
        vec = spinorOut2.getAccessor().getElement(site);
        if (vec==EO_vec_319) {
            success[1]=true;
        }
        site = GInd::getSite(550);
        vec = spinorOut2.getAccessor().getElement(site);
        if (vec==EO_vec_550) {
            success[2]=true;
        }
        site = GInd::getSite(1023);
        vec = spinorOut2.getAccessor().getElement(site);
        if (vec==EO_vec_1023) {
            success[3]=true;
        }
    }

    if (LatLayoutRHS==Odd) {
        gSite site = GInd::getSite(0);
        gVect3<floatT> vec = spinorOut2.getAccessor().getElement(site);
        if (vec==OE_vec_0) {
            success[0]=true;
        }
        site = GInd::getSite(512);
        vec = spinorOut2.getAccessor().getElement(site);
        if (vec==OE_vec_512) {
            success[1]=true;
        }
        site = GInd::getSite(697);
        vec = spinorOut2.getAccessor().getElement(site);
        if (vec==OE_vec_697) {
            success[2]=true;
        }
        site = GInd::getSite(1023);
        vec = spinorOut2.getAccessor().getElement(site);
        if (vec==OE_vec_1023) {
            success[3]=true;
        }
    }

    if (success[0] && success[1] && success[2] && success[3]) {
        rootLogger.info("Test of Dslash against old values: " ,  CoutColors::green ,  "passed" ,  CoutColors::reset);
        return false;
    } else {
        rootLogger.error("Test of Dslash against old values failed!");
        return true;
    }
}

int main(int argc, char **argv) {

    stdLogger.setVerbosity(INFO);

    LatticeParameters param;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/tests/DslashTest.param", argc, argv);
    bool lerror=false;

    commBase.init(param.nodeDim());

    const int HaloDepth = 0;
    initIndexer(HaloDepth,param, commBase);
    const int HaloDepthSpin = 0;
    initIndexer(HaloDepthSpin,param, commBase);
    stdLogger.setVerbosity(INFO);

    rootLogger.info("-------------------------------------");
    rootLogger.info("Running on Device");
    rootLogger.info("-------------------------------------");
    rootLogger.info("Testing Even - Odd");
    rootLogger.info("------------------");
    lerror = (lerror || test_dslash<float, Even, Odd, 1, true>(commBase));
    rootLogger.info("------------------");
    rootLogger.info("Testing Odd - Even");
    rootLogger.info("------------------");
    lerror = (lerror || test_dslash<float, Odd, Even, 1, true>(commBase));
}


template<Layout LatLayout, size_t HaloDepth>
size_t getGlobalIndex(LatticeDimensions coord) {
    typedef GIndexer<LatLayout, HaloDepth> GInd;

    LatticeData lat = GInd::getLatData();
    LatticeDimensions globCoord = lat.globalPos(coord);

    return globCoord[0] + globCoord[1] * lat.globLX + globCoord[2] * lat.globLX * lat.globLY +
           globCoord[3] * lat.globLX * lat.globLY * lat.globLZ;
}

