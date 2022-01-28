#include "../SIMULATeQCD.h"
#include "../modules/rhmc/rhmcParameters.h"
#include "../modules/dslash/condensate.h"
#include "../modules/HISQ/hisqSmearing.h"

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
    SimpleArray<PREC,numVec> chi_l = measure_condensate<PREC, true, 2, 4, numVec>(commBase, param, true, gauge, d_rand);

    SimpleArray<PREC,numVec> chi_lCONTROL(0.0);
    chi_lCONTROL[0]=0.0632172;
    chi_lCONTROL[1]=0.0880454;
    chi_lCONTROL[2]=0.0421913;
    chi_lCONTROL[3]=0.0458335;
    chi_lCONTROL[4]=0.0723446;
    chi_lCONTROL[5]=0.059815;
    chi_lCONTROL[6]=0.0734244;
    chi_lCONTROL[7]=0.0751981;
    chi_lCONTROL[8]=0.0620134;
    chi_lCONTROL[9]=0.0537205;

    bool lerror=false;
    for (int i = 0; i < numVec; ++i) {
        if(!cmp_rel(chi_l[i],chi_lCONTROL[i],1e-6,1e-6)) {
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
