/*
 * main_DslashMultiTest.cpp
 *
 */

#include "../simulateqcd.h"
#include "../modules/dslash/dslash.h"
#include "../modules/rhmc/rhmcParameters.h"
#include "../modules/hisq/hisqSmearing.h"

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
struct compare_smearing {
    SU3Accessor<floatT,comp> gL;
    SU3Accessor<floatT,comp> gR;
    compare_smearing(Gaugefield<floatT,onDevice,HaloDepth, comp> &GaugeL, Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeR) : gL(GaugeL.getAccessor()), gR(GaugeR.getAccessor()) {}

    __host__ __device__ int operator() (gSite site) {
        floatT sum = 0.0;
        for (int mu = 0; mu < 4; mu++) {

            gSiteMu siteMu=GIndexer<All,HaloDepth>::getSiteMu(site,mu);
            SU3<floatT> diff = gL.getLink(siteMu) - gR.getLink(siteMu);
            floatT norm = 0.0;

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    norm += abs2(diff(i,j));
                }
            }
            sum += sqrt(norm);
        }
    return (sum < 1e-5 ? 0 : 1);
    }
};

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
bool checkfields(Gaugefield<floatT,onDevice,HaloDepth, comp> &GaugeL, Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeR) {
    LatticeContainer<onDevice,int> redBase(GaugeL.getComm());
    const size_t elems = GIndexer<All,HaloDepth>::getLatData().vol4;

    redBase.adjustSize(elems);

    redBase.template iterateOverBulk<All,HaloDepth>(compare_smearing<floatT, onDevice, HaloDepth, comp>(GaugeL,GaugeR));

    int faults = 0;
    redBase.reduce(faults,elems);

    rootLogger.info(faults ,  " faults detected");

    if (faults > 0) {
      return false;
    } else {
      return true;
    }
}


template <class floatT>
__host__ bool operator==(const Vect3<floatT> &lhs, const Vect3<floatT> &rhs){

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

template <Layout LatLayoutRHS, size_t HaloDepth>
bool local_site(sitexyzt globalPos, gSite * site, CommunicationBase &commBase){

    typedef GIndexer<LatLayoutRHS, HaloDepth> GInd;

    LatticeDimensions offset = commBase.mycoords()*GInd::getLatData().localLattice();

    bool local[4] = {false};

    int x = globalPos.x;
    int y = globalPos.y;
    int z = globalPos.z;
    int t = globalPos.z;

    if (int(offset[0]+GInd::getLatData().lx) > x && x >= offset[0]) {
        x-=offset[0];
        local[0] = true;
    }
    if (int(offset[1]+GInd::getLatData().ly) > y && y >= offset[1]) {
        y-=offset[1];
        local[1] = true;
    }
    if (int(offset[2]+GInd::getLatData().lz) > z && z >= offset[2]) {
        z-=offset[2];
        local[2] = true;}
    if (int(offset[3]+GInd::getLatData().lt) > t && t >= offset[3]) {
        t-=offset[3];
        local[3] = true;
    }

    *site = GInd::getSite(x,y,z,t);

    return (local[0] && local[1] && local[2] && local[3]);
}


//the Dslash test function. Please start reading here.
template<class floatT, Layout LatLayout, Layout LatLayoutRHS, size_t NStacks, bool onDevice>
bool test_dslash2(CommunicationBase &commBase){

    //Initialization as usual
    const int HaloDepth = 2;
    const int HaloDepthSpin = 4;
    typedef GIndexer<LatLayoutRHS, HaloDepth> GInd;

    //Our gaugefield
    Gaugefield<floatT, onDevice, HaloDepth, R14> gauge(commBase);

    Gaugefield<floatT, onDevice, HaloDepth, R18> gauge_smeared(commBase);
    Gaugefield<floatT, onDevice, HaloDepth, U3R14> gauge_Naik(commBase);

    HisqSmearing<floatT, onDevice, HaloDepth> smearing(gauge, gauge_smeared, gauge_Naik);

    rootLogger.info("Read configuration");
    gauge.readconf_nersc("../test_conf/l20t20b06498a_nersc.302500");

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
    Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks> spinor(commBase);

    rootLogger.info("Randomize spinors");
    spinorIn.gauss(d_rand.state);
    spinor.gauss(d_rand.state);

    rootLogger.info("Initialize DSlash");
    HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepth, HaloDepthSpin, NStacks> dslash(gauge_smeared, gauge_Naik, 0.0);

    dslash.Dslash(spinorOut, spinorIn);

    spinorOut2 = spinorOut;

    Vect3<floatT> OE_vec_0 = Vect3<floatT>(COMPLEX(floatT)(1.2236,-0.475367),COMPLEX(floatT)(0.450346,0.710892),
        COMPLEX(floatT)(-0.444914,0.586732));
    Vect3<floatT> OE_vec_4315 = Vect3<floatT>(COMPLEX(floatT)(-1.02087,-0.99984),COMPLEX(floatT)(0.65547,-0.0330726),
        COMPLEX(floatT)(-1.98085,-0.0921614));
    Vect3<floatT> OE_vec_25242 = Vect3<floatT>(COMPLEX(floatT)(-0.376823,-1.6069),COMPLEX(floatT)(0.143999,0.235529),
        COMPLEX(floatT)(-0.223067,0.499678));
    Vect3<floatT> OE_vec_79999 = Vect3<floatT>(COMPLEX(floatT)(-1.16874,-0.447845),COMPLEX(floatT)(2.64052,0.548778),
        COMPLEX(floatT)(-0.4966,-3.00779));

    Vect3<floatT> EO_vec_0 = Vect3<floatT>(COMPLEX(floatT)(-1.00401,1.90105),COMPLEX(floatT)(0.629232,-0.4621),
        COMPLEX(floatT)(0.244322,0.437392));
    Vect3<floatT> EO_vec_54729 = Vect3<floatT>(COMPLEX(floatT)(1.42788,-1.46884),COMPLEX(floatT)(0.196411,-0.676345),
        COMPLEX(floatT)(-1.54544,0.981528));
    Vect3<floatT> EO_vec_29512 = Vect3<floatT>(COMPLEX(floatT)(0.495192,0.172826),COMPLEX(floatT)(1.73708,0.363861),
        COMPLEX(floatT)(1.82399,-0.531962));
    Vect3<floatT> EO_vec_79999 = Vect3<floatT>(COMPLEX(floatT)(-1.7404,-0.955973),COMPLEX(floatT)(2.45754,-0.650599),
        COMPLEX(floatT)(2.03533,0.636551));

    Vect3<floatT> AA_vec_0 = Vect3<floatT>(COMPLEX(floatT)(-1.60503,-1.01415),COMPLEX(floatT)(-1.20935,0.210231),
        COMPLEX(floatT)(-0.0256269,-0.636353));
    Vect3<floatT> AA_vec_79912 = Vect3<floatT>(COMPLEX(floatT)(1.94435,0.211277),COMPLEX(floatT)(0.540431,0.0692634),
        COMPLEX(floatT)(0.0145495,-0.483509));
    Vect3<floatT> AA_vec_155500 = Vect3<floatT>(COMPLEX(floatT)(-0.762942,-2.22293),COMPLEX(floatT)(-0.678135,-2.19784),
        COMPLEX(floatT)(-0.96457,-1.29161));
    Vect3<floatT> AA_vec_159999 = Vect3<floatT>(COMPLEX(floatT)(-1.16874,-0.447845),COMPLEX(floatT)(2.64052,0.548778),
        COMPLEX(floatT)(-0.4966,-3.00779));

    sitexyzt OE_0(0,0,0,0);
    sitexyzt OE_4315(11,11,1,1);
    sitexyzt OE_25242(4,4,6,6);
    sitexyzt OE_79999(19,19,19,19);

    sitexyzt EO_0(0,0,0,0);
    sitexyzt EO_54729(18,12,13,13);
    sitexyzt EO_29512(4,11,7,7);
    sitexyzt EO_79999(18,19,19,19);

    sitexyzt AA_0(0,0,0,0);
    sitexyzt AA_79912(5,11,19,19);
    sitexyzt AA_155500(1,10,17,17);
    sitexyzt AA_159999(18,19,19,19);

    gSite site;

    bool success[4]={false};

    Vect3<floatT> vec;

    if (LatLayoutRHS==Even) {

        if(local_site<Even, HaloDepthSpin>(OE_0, &site, commBase)) {
            vec = spinorOut2.getAccessor().getElement(site);
            if (vec==OE_vec_0)
                success[0]=true;
        } else {
            success[0]=true;
        }

        if(local_site<Even, HaloDepthSpin>(OE_4315, &site, commBase)) {
            vec = spinorOut2.getAccessor().getElement(site);
            if (vec==OE_vec_4315)
                success[1]=true;
        } else {
            success[1]=true;
        }

        if(local_site<Even, HaloDepthSpin>(OE_25242, &site, commBase)) {
            vec = spinorOut2.getAccessor().getElement(site);
            if (vec==OE_vec_25242)
                success[2]=true;
        } else {
            success[2]=true;
        }

        if(local_site<Even, HaloDepthSpin>(OE_79999, &site, commBase)) {
            vec = spinorOut2.getAccessor().getElement(site);
            if (vec==OE_vec_79999)
                success[3]=true;
        } else {
            success[3]=true;
        }
    }

    if (LatLayoutRHS==Odd) {

        if(local_site<Odd, HaloDepthSpin>(EO_0, &site, commBase)) {
            vec = spinorOut2.getAccessor().getElement(site);
            if (vec==EO_vec_0)
                success[0]=true;
        } else {
            success[0]=true;
        }

        if(local_site<Odd, HaloDepthSpin>(EO_54729, &site, commBase)) {
            vec = spinorOut2.getAccessor().getElement(site);
            if (vec==EO_vec_54729)
                success[1]=true;
        } else {
            success[1]=true;
        }

        if(local_site<Odd, HaloDepthSpin>(EO_29512 , &site, commBase)) {
            vec = spinorOut2.getAccessor().getElement(site);
            if (vec==EO_vec_29512)
                success[2]=true;
        } else {
            success[2]=true;
        }

        if(local_site<Odd, HaloDepthSpin>(EO_79999, &site, commBase)) {
            vec = spinorOut2.getAccessor().getElement(site);
            if (vec==EO_vec_79999)
                success[3]=true;
        } else {
            success[3]=true;
        }
    }

    if (LatLayoutRHS==All) {

        if(local_site<All, HaloDepthSpin>(AA_0, &site, commBase)) {
            vec = spinorOut2.getAccessor().getElement(site);
            if (vec==AA_vec_0)
                success[0]=true;
        } else {
            success[0]=true;
        }

        if(local_site<All, HaloDepthSpin>(AA_79912 , &site, commBase)) {
            vec = spinorOut2.getAccessor().getElement(site);
            if (vec==AA_vec_79912)
                success[1]=true;
        } else {
            success[1]=true;
        }

        if(local_site<All, HaloDepthSpin>(AA_155500 , &site, commBase)) {
            vec = spinorOut2.getAccessor().getElement(site);
            if (vec==AA_vec_155500)
                success[2]=true;
        } else {
            success[2]=true;
        }

        if(local_site<All, HaloDepthSpin>(AA_159999, &site, commBase)) {
            vec = spinorOut2.getAccessor().getElement(site);
            if (vec==AA_vec_159999)
                success[3]=true;
        } else {
            success[3]=true;
        }
    }

    if (success[0] && success[1] && success[2] && success[3]) {
        stdLogger.info("Test of Dslash: " ,  CoutColors::green ,  "passed" ,  CoutColors::reset);
    } else {
        stdLogger.error("Test of Dslash: failed");
        stdLogger.info("mycoords = " ,  commBase.mycoords()*GInd::getLatData().localLattice(),
                       ": " ,   success[0] ,  " " ,  success[1] ,  " " ,  success[2] ,  " " ,  success[3]);
        return true;
    }

    bool simple = false;

    COMPLEX(floatT) scalarProduct, reference;

    spinorOut.updateAll();

    if (LatLayoutRHS==All)
        reference = COMPLEX(floatT)(-20.616344023,-374.68626006);

    else if(LatLayoutRHS==Odd)
        reference = COMPLEX(floatT)(-245.143001,983.5859418);

    else
        reference = COMPLEX(floatT)(-34.25902057,-404.02228596);

    scalarProduct = spinor.dotProduct(spinorOut);
    if (scalarProduct == reference)
        simple = true;

    if (simple) {
        stdLogger.info("Simple test using scalar product: " ,  CoutColors::green ,  "passed" ,  CoutColors::reset);
        return false;
    } else {
        stdLogger.error("Simple test using scalar product failed!");
        return true;
    }
}



int main(int argc, char **argv) {

    stdLogger.setVerbosity(INFO);

    LatticeParameters param;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/tests/dslashMultiTest.param", argc, argv);
    bool lerror=false;

    commBase.init(param.nodeDim());
    initIndexer(2,param, commBase);
    initIndexer(4,param, commBase);
    stdLogger.setVerbosity(INFO);

    rootLogger.info("-------------------------------------");
    rootLogger.info("Running on Device");
    rootLogger.info("-------------------------------------");
    rootLogger.info("Testing All - All");
    rootLogger.info("------------------");
    lerror = (lerror || test_dslash2<double, All, All, 1, true>(commBase));
    rootLogger.info("------------------");
    rootLogger.info("Testing Even - Odd");
    rootLogger.info("------------------");
    lerror = (lerror || test_dslash2<double, Even, Odd, 1, true>(commBase));
    rootLogger.info("------------------");
    rootLogger.info("Testing Odd - Even");
    rootLogger.info("------------------");
    lerror = (lerror || test_dslash2<double, Odd, Even, 1, true>(commBase));
    rootLogger.info("------------------");

    if(lerror) {
        rootLogger.error("At least one test failed!");
        return -1;
    } else {
        rootLogger.info("All tests " ,  CoutColors::green ,  "passed!" ,  CoutColors::reset);
    }
}
