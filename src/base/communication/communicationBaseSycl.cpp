#include "../../define.h"
#include <mpi.h>
#include <vector>
#include <cstring>

#include "communicationBaseSycl.h"


Logger rootLogger(OFF);
Logger stdLogger(ALL);

CommunicationBaseSycl::CommunicationBaseSycl(int *argc, char ***argv, bool forceHalos) : _forceHalos(forceHalos) {
    int ret;

    ret = MPI_Init(argc, argv);
    _MPI_fail(ret, "MPI_Init");

    ret = MPI_Comm_rank(MPI_COMM_WORLD, &myInfo.world_rank);
    _MPI_fail(ret, "MPI_Comm_rank");

    ret = MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    _MPI_fail(ret, "MPI_Comm_size");

    stdLogger.set_additional_prefix(sjoin("[Rank ", myInfo.world_rank, "] "));

    if (IamRoot()) {
        rootLogger.setVerbosity(stdLogger.getVerbosity());
    }


    rootLogger.info("Running SIMULATeQCD");
    rootLogger.info("Git commit version: ", GIT_HASH);
    rootLogger.info("Initializing MPI with (", world_size, " proc)");


    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);


    std::string nodeName = processor_name;

    myInfo.nodeName = nodeName;
}

void CommunicationBaseSycl::initNodeComm() {
    int ret; 
    ret = MPI_Info_create(&mpi_info);
    _MPI_fail(ret, "MPI_Info_create");

    ret = MPI_Comm_split_type(getCart_comm(), MPI_COMM_TYPE_SHARED, MyRank(), mpi_info, &node_comm);
    _MPI_fail(ret, "MPI_Comm_split_type");

    ret = MPI_Comm_rank(node_comm, &myInfo.node_rank);
    _MPI_fail(ret, "MPI_Cart_Comm_rank");

}

void CommunicationBaseSycl::init(const LatticeDimensions &Dim, __attribute__((unused)) const LatticeDimensions &Topo) {
    if (Dim.mult() != getNumberProcesses()) {
        throw std::runtime_error(stdLogger.fatal("Number of processes does not match Node dimensions: ", Dim[0], " * ", Dim[1], " * ", Dim[2], " * ", Dim[3],
                    " != ", getNumberProcesses()));
    }

    int ret; 

    dims = Dim;
    int periods[] = {true, true, true, true};
    int reorder = true;

    ret = MPI_Cart_create(MPI_COMM_WORLD, 4, dims, periods, reorder, &cart_comm);
    _MPI_fail(ret, "MPI_Cart_create");

    ret = MPI_Comm_rank(cart_comm, &myInfo.world_rank);
    _MPI_fail(ret, "MPI_Comm_rank (cart)");

    initNodeComm();

    ret = MPI_Cart_get(cart_comm, 4, dims, periods, myInfo.coord);
    _MPI_fail(ret, "MPI_Cart_get");

    _initialized = true;

    int num_devices = 0;
    //get device count
    std::string sycl_target = SYCL_TARGET;
    
    
    bool gpu_selection = (sycl_target.find("amd") != std::string::npos || sycl_target.find("nvidia") != std::string::npos || sycl_target.find("intel_gpu") != std::string::npos);
    if (gpu_selection) {
        auto sycl_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        num_devices = sycl_devices.size();
    }
    else {
        rootLogger.warn("SYCL_TARGET does not contain substrings \"amd\", \"nvidia\" or \"intel_gpu\". Using CPUs as sycl devices. Specify sycl target architecture directly if this is not intended.");
        auto sycl_devices = sycl::device::get_devices(sycl::info::device_type::cpu);
        num_devices = sycl_devices.size();
    }
    //What device should we chose if only -fsycl but not -fsycl-targets is used?

    if (Topo.summed() == 0) {
        myInfo.deviceRank = myInfo.node_rank % num_devices;
    }
    else {
        myInfo.deviceRank = myInfo.coord[0] * Topo[0] + myInfo.coord[1] * Topo[1] + myInfo.coord[2] * Topo[2]
            + myInfo.coord[3] * Topo[3];

        for (int i = 0; i < 4; ++i) {
            if (((Dim[i] == 1) && (Topo[i] != 0)) || ((Dim[i] != 1) && (Topo[i] == 0))) {
                rootLogger.warn( "GPU topology and chosen lattice splitting seem incompatible!");
            }
        }
    }


    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);





    
}