#include "../SIMULATeQCD.h"
#include "../modules/rhmc/rhmcParameters.h"
#include "../modules/dslash/condensate.h"
#include "../modules/HISQ/hisqSmearing.h"

int main(int argc, char *argv[])
{
    stdLogger.setVerbosity(INFO);
    CommunicationBase commBase(&argc, &argv);
    RhmcParameters param;
    param.readfile(commBase, "condensate.param");
    commBase.init(param.nodeDim());
    initIndexer(2,param,commBase);
    Gaugefield<double,true,2> gauge(commBase);

    gauge.readconf_nersc(param.GaugefileName());
    gauge.updateAll();

    grnd_state<true> d_rand;
    initialize_rng(param.seed(), d_rand);

    const size_t numVec=10;
    
    measure_condensate<double, true, 2, 4, numVec>(commBase, param, true, gauge, d_rand);

    return 0;
}
