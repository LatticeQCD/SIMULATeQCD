/* 
 * main_confReadWriteTest.cu                                                               
 * 
 * Test to check whether that the NERSC and ILDG read/write as well as the MILC read work correctly.
 * 
 * M. Rodekamp, S. Ali, D. Clarke 
 * 
 */

#include "../SIMULATeQCD.h"
#include "testing.h"

#define PREC double
#define MY_BLOCKSIZE 256
#define epsilon 0.00001


int main(int argc, char *argv[]) {

	stdLogger.setVerbosity(INFO);

	LatticeParameters param;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/tests/confReadWriteTest.param", argc, argv);
	commBase.init(param.nodeDim());

	rootLogger.info("Initialize Lattice");

	const size_t HaloDepth = 0;
	typedef GIndexer<All,HaloDepth> GInd;
    initIndexer(HaloDepth,param,commBase);

	Gaugefield<PREC,true,HaloDepth> gauge(commBase);
	Gaugefield<PREC,true,HaloDepth> gauge_test(commBase);

    rootLogger.info("Try NERSC read...");
	gauge.readconf_nersc("../test_conf/nersc.l8t4b3360_bieHB");

    rootLogger.info("Try NERSC write...");
	gauge.writeconf_nersc("nersc.l8t4b3360_bieHB_test");

    rootLogger.info("Try ILDG write...");
    gauge.writeconf_ildg("nersc_ildg.l8t4b3360_bieHB_test",3,param.prec_out());

    rootLogger.info("Try ILDG read...");
	gauge_test.readconf_ildg("nersc_ildg.l8t4b3360_bieHB_test");

    rootLogger.info("One last ILDG write to verify the checksum worked...");
	gauge_test.writeconf_ildg("ildg_ildg.l8t4b3360_bieHB_test",3,param.prec_out());

    rootLogger.info("Link-by-link comparison of NERSC config with written ILDG config...");
    bool pass = compare_fields<PREC,HaloDepth,true,R18>(gauge,gauge_test,1e-15);
    if(!pass) {
		rootLogger.error("Binaries are not equal.");
        return -1;
	} else {
		rootLogger.info(CoutColors::green , "All tests passed!", CoutColors::reset);
    }
   
    return 0;
}
