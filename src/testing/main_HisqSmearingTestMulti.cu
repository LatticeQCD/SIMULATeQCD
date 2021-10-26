/* 
 * main_HisqSmearingTestMulti.cu                                                               
 * 
 */

#include "../SIMULATeQCD.h"

#define PREC double
#define MY_BLOCKSIZE 256
#define USE_GPU true

template <class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
struct compare_smearing {
    gaugeAccessor<floatT,comp> gL;
    gaugeAccessor<floatT,comp> gR;
    compare_smearing(Gaugefield<floatT,onDevice,HaloDepth, comp> &GaugeL, Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeR) : gL(GaugeL.getAccessor()), gR(GaugeR.getAccessor()) {}

    __host__ __device__ int operator() (gSite site) {
        floatT sum = 0.0;
        for (int mu = 0; mu < 4; mu++) {

            gSiteMu siteMu=GIndexer<All,HaloDepth>::getSiteMu(site,mu);
            GSU3<floatT> diff = gL.getLink(siteMu) - gR.getLink(siteMu);
            floatT norm = 0.0;

            for (int i = 0; i < 3; i++)
	        for (int j = 0; j < 3; j++) {
	            norm += abs2(diff(i,j));
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

    rootLogger.info() << faults << " faults detected";

    if (faults > 0) {
        return false;
    }
    else {
        return true;
    }
}
						    
int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);
    LatticeParameters param;
    const int LatDim[] = {20, 20, 20, 20};
    const int NodeDim[] = {1, 1, 1, 1};

    param.latDim.set(LatDim);
    param.nodeDim.set(NodeDim);

    CommunicationBase commBase(&argc, &argv);
    commBase.init(param.nodeDim());

    const size_t HaloDepth = 0;

    rootLogger.info() << "Initialize Lattice";
    typedef GIndexer<All,HaloDepth> GInd;
    initIndexer(HaloDepth,param,commBase);

    Gaugefield<PREC,true, HaloDepth> gaugeSingleGPU(commBase);
    Gaugefield<PREC,true, HaloDepth> gaugeMultiGPU(commBase);
    Gaugefield<PREC,true, HaloDepth> gaugeMultiXGPU(commBase);
    gaugeSingleGPU.readconf_nersc("../test_conf/pgpu_naik_smearing_single.nersc");
    gaugeMultiGPU.readconf_nersc("../test_conf/pgpu_naik_smearing_multi.nersc");
    gaugeMultiXGPU.readconf_nersc("../test_conf/pgpu_naik_smearing_multi_x.nersc");

    bool pass = checkfields<PREC,true,HaloDepth,R18>(gaugeSingleGPU,gaugeMultiGPU);
    bool pass2 = checkfields<PREC,true,HaloDepth,R18>(gaugeSingleGPU,gaugeMultiXGPU);

    if (pass) {
      rootLogger.info() << "Fields Single and Multi are identical";
    }
    else {
      rootLogger.info() << "Fields Single and Multi are not identical";
    }
    if (pass2) {
      rootLogger.info() << "Fields Single and MultiX are identical";
    }
    else {
      rootLogger.info() << "Fields Single and MultiX are not identical";
    }
    if (pass && pass2) {
      rootLogger.info() << CoutColors::green << "Test passed!";
    }
    else {
      rootLogger.info() << CoutColors::red << "Test failed!";
    }
    
    return 0;
}
