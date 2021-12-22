//
// Created by sajid Ali on 12/21/21.
//
// converts NERSC file format to ILDG and vice versa.

#define PREC double
#define USE_GPU false

#include "../SIMULATeQCD.h"
using namespace std;

int main(int argc, char *argv[]) {

    const size_t HaloDepth = 2;
    LatticeParameters param;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "/home/sali/builddirsimulateQCD/parameter/gradientFlow8c4.param", argc, argv);
    commBase.init(param.nodeDim());

	typedef GIndexer<All,HaloDepth> GInd;
    initIndexer(HaloDepth,param,commBase);

    Gaugefield<PREC, USE_GPU, HaloDepth> gauge_field_in(commBase);
    gauge_field_in.readconf_nersc("/home/sali/measurements/measurePlaquette/readWriteConf/nersc.l8t4b3360_bieHB");
    //gauge_field_in.one();
    gauge_field_in.updateAll();

    std::string file_name_out = "l8t4b3360_bieHB_ildg_prec_"+std::to_string(2);
    gauge_field_in.writeconf_ildg(  file_name_out,3,2);

    return 0;
    }