/* 
 * main_SaveTest.cu                                                               
 * 
 * Marcel Rodekamp, 19 Jul 2018
 * 
 */

#include "../SIMULATeQCD.h"

#define PREC double
#define USE_GPU true
#define MY_BLOCKSIZE 256
#define error 0.00001

__host__ __device__ PREC dabs(PREC z){
	if(z < 0){return -z;}
	else{return z;}
}

template <class floatT, size_t HaloDepth, bool onDevice, CompressionType comp>
struct compare_links {
    gaugeAccessor<floatT,comp> gL;
    gaugeAccessor<floatT,comp> gR;
    LatticeParameters param;

    compare_links(Gaugefield<floatT, onDevice, HaloDepth, comp> &gaugeL, Gaugefield<floatT, onDevice, HaloDepth, comp> &gaugeR) : gL(gaugeL.getAccessor()), gR(gaugeR.getAccessor()) {}

    __host__ __device__ int operator() (gSite site) {

        floatT sum = 0.0;
        for (int mu = 0; mu < 4; mu++) {

            gSiteMu siteMu=GIndexer<All,HaloDepth>::getSiteMu(site,mu);
            GSU3<floatT> diff = gL.getLink(siteMu) - gR.getLink(siteMu);
            floatT norm = 0.0;

            for (int i = 0; i < 3; i++) {
            	for (int j = 0; j < 3; j++) {
    	            norm += abs2(diff(i,j));
    	        }
            }
            sum += norm;
        }
        sum /= 4.0;
        /*if (param.prec_out() == 1 || (param.prec_out() == 0 && sizeof(floatT) == sizeof(float)))
            return (sum < 1e-7 ? 0 : 1);
        else if (param.prec_out() == 2 || (param.prec_out() == 0 && sizeof(floatT) == sizeof(double)))*/
            return (sum < 1e-15 ? 0 : 1);
    }
};

template <class floatT, size_t HaloDepth, bool onDevice, CompressionType comp>
bool compare_fields(Gaugefield<floatT, onDevice, HaloDepth, comp> &gaugeL, Gaugefield<floatT, onDevice, HaloDepth, comp> &gaugeR) {
    LatticeContainer<onDevice, int> redBase(gaugeL.getComm());
    const size_t elems = GIndexer<All,HaloDepth>::getLatData().vol4;

    redBase.adjustSize(elems);

    redBase.template iterateOverBulk<All,HaloDepth>(compare_links<floatT,HaloDepth,onDevice,comp>(gaugeL, gaugeR));

    int faults = 0;
    redBase.reduce(faults, elems);

    rootLogger.info(faults ,  " faults detected!");

    if (faults > 0) {
        return false;
    } else {
        return true;
    }
}

int main(int argc, char *argv[]){
	stdLogger.setVerbosity(DEBUG);

	LatticeParameters param;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "/home/sali/measurements/measurePlaquette/readWriteConf/SaveTest.param", argc, argv);
	commBase.init(param.nodeDim());

	rootLogger.info("Initialize Lattice");

	const size_t HaloDepth = 1;
	typedef GIndexer<All,HaloDepth> GInd;
    initIndexer(HaloDepth,param,commBase);
    rootLogger.info("output precision = ", param.prec_out());

	Gaugefield<PREC, USE_GPU,HaloDepth> gauge( commBase);
	Gaugefield<PREC, USE_GPU,HaloDepth> gauge_test( commBase);
	Gaugefield<PREC, USE_GPU,HaloDepth> gauge_test1(commBase);

    rootLogger.info("Read ildg sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss");
	gauge.readconf_nersc("/home/sali/measurements/measurePlaquette/readWriteConf/nersc.l8t4b3360_bieHB");
	//gauge.readconf_ildg("/home/sali/measurements/measurePlaquette/readWriteConf/ildg.l8t4b3360_bieHB");
	gauge.updateAll();

    //rootLogger.info("Write ildg sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss");
	//gauge.writeconf_nersc("/home/sali/measurements/measurePlaquette/readWriteConf/nersc.l8t4b3360_bieHB_test");
    //gauge.writeconf_ildg("/home/sali/measurements/measurePlaquette/readWriteConf/ildg.l8t4b3360_bieHB_test",3,param.prec_out());

    //rootLogger.info("Read ildg sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss");
	//gauge.readconf_nersc("/home/sali/measurements/measurePlaquette/readWriteConf/nersc.l8t4b3360_bieHB_test");
	//gauge.readconf_ildg("/home/sali/measurements/measurePlaquette/readWriteConf/nersc_ildg.l8t4b3360_bieHB_test");
    //gauge.updateAll();

    rootLogger.info("Write ildg sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss");
	gauge.writeconf_ildg("/home/sali/measurements/measurePlaquette/readWriteConf/nersc_ildg.l8t4b3360_bieHB_test",2,2);
	/*gauge.writeconf_nersc("/home/sali/measurements/measurePlaquette/readWriteConf/nersc.l8t4b3360_bieHB_test",3,param.prec_out());
    gauge.readconf_nersc("/home/sali/measurements/measurePlaquette/readWriteConf/nersc.l8t4b3360_bieHB_test");
    gauge.updateAll();
    gauge.writeconf_nersc("/home/sali/measurements/measurePlaquette/readWriteConf/nersc.l8t4b3360_bieHB_test_",3,param.prec_out());*/
    rootLogger.info("Read ildg sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss");
    gauge_test.readconf_ildg("/home/sali/measurements/measurePlaquette/readWriteConf/nersc_ildg.l8t4b3360_bieHB_test");
    gauge_test.updateAll();
    rootLogger.info("Write ildg sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss");
	gauge_test.writeconf_ildg("/home/sali/measurements/measurePlaquette/readWriteConf/ildg_nersc_ildg.l8t4b3360_bieHB_test",3,2);
	//gauge_test.updateAll();



    rootLogger.info("Testing the configuration");

    GaugeAction<PREC, USE_GPU,HaloDepth,R18> gAction(gauge);

	GaugeAction<PREC, USE_GPU,HaloDepth,R18> gAction_test(gauge_test);

    PREC plaq = gAction.plaquette();

    PREC plaq_test = gAction_test.plaquette();

	if(abs(plaq - plaq_test) > error){
		rootLogger.info(CoutColors::red, "writeconf failed: Computed plaquettes are not equal ", plaq ," != ", plaq_test ,  CoutColors::reset);
	} else {
	    rootLogger.info(CoutColors::green, "writeconf succeeded: Computed plaquettes are equal", CoutColors::reset);
	}

    PREC clov = gAction.clover();

    PREC clov_test = gAction_test.clover();

	if(abs(clov - clov_test) > error){
		rootLogger.info(CoutColors::red,  "writeconf failed: Computed clovers are not equal ", clov, " != ", clov_test ,  CoutColors::reset);
	} else {
	    rootLogger.info(CoutColors::green , "writeconf succeeded: Computed clovers are equal ", CoutColors::reset);
	}

    bool pass = compare_fields<PREC,HaloDepth,USE_GPU,R18>(gauge,gauge_test);

    if(!pass){
		rootLogger.info(CoutColors::red ,  "writeconf failed: Binaries are not equal.", CoutColors::reset);
		remove("../test_conf/test_l20t20b06498a_nersc.302500");
	} else {
		rootLogger.info(CoutColors::green , "Congratulations, writeconf worked!", CoutColors::reset);
		remove("../test_conf/test_l20t20b06498a_nersc.302500");
    }

    /*pass = compare_fields<PREC,HaloDepth,USE_GPU,R18>(gauge_test,gauge_test1);

    if(!pass){
		rootLogger.info(CoutColors::red ,  "writeconf failed: Binaries are not equal.", CoutColors::reset);
		remove("../test_conf/test_l20t20b06498a_nersc.302500");
	} else {
		rootLogger.info(CoutColors::green , "Congratulations, writeconf worked!", CoutColors::reset);
		remove("../test_conf/test_l20t20b06498a_nersc.302500");
    }*/
   
    return 0;
}

