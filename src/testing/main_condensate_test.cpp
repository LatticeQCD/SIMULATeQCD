#include "../simulateqcd.h"
#include "../modules/rhmc/rhmcParameters.h"
#include "../modules/dslash/condensate.h"
#include "../modules/hisq/hisqSmearing.h"

#define PREC double

int main(int argc, char *argv[])
{
    stdLogger.setVerbosity(INFO);
    CommunicationBase commBase(&argc, &argv);
    RhmcParameters param;
    param.readfile(commBase, "../parameter/tests/condensate.param", argc, argv);
    commBase.init(param.nodeDim());
    initIndexer(2,param,commBase);
    Gaugefield<PREC,true,2> gauge(commBase);

    gauge.readconf_nersc(param.GaugefileName());
    gauge.updateAll();

    grnd_state<true> d_rand;
    initialize_rng(param.seed(), d_rand);

    const size_t numVec=10; // Do not change this.
    SimpleArray<PREC,numVec> chi_l = measure_condensate<PREC, true, 2, 4, numVec>(commBase, param, param.m_ud(), gauge, d_rand);

    SimpleArray<PREC,numVec> chi_lCONTROL(0.0);
    chi_lCONTROL[0]=0.0671406;
    chi_lCONTROL[1]=0.0785372;
    chi_lCONTROL[2]=0.0473219;
    chi_lCONTROL[3]=0.0387776;
    chi_lCONTROL[4]=0.0611802;
    chi_lCONTROL[5]=0.0582478;
    chi_lCONTROL[6]=0.0650505;
    chi_lCONTROL[7]=0.072139;
    chi_lCONTROL[8]=0.0637511;
    chi_lCONTROL[9]=0.052524;

    bool lerror=false;
    for (int i = 0; i < numVec; ++i) {
        if(!cmp_rel(chi_l[i],chi_lCONTROL[i],1e-5,1e-5)) {
            rootLogger.error("chi_l, chi_lCONTROL = ",chi_l[i],"  ",chi_lCONTROL[i]);
            lerror=true;
        };
    }

    if(lerror) {
        rootLogger.error("At least one test failed!");
        return -1;
    } else {
        rootLogger.info("All tests " ,  CoutColors::green ,  "passed!" ,  CoutColors::reset);
    }

    return 0;
}
