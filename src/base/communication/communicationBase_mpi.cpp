//
// Created by Lukas Mazur on 11.10.17.
//

#include "../../define.h"

#ifdef COMPILE_WITH_MPI

#include <sched.h>
#include <mpi.h>
#include <vector>
#include <cstring>
#include "communicationBase.h"


Logger rootLogger(OFF);
Logger stdLogger(ALL);

CommunicationBase::CommunicationBase(int *argc, char ***argv) {

    int ret;
    ret = MPI_Init(argc, argv);
    _MPI_fail(ret, "MPI_Init");

    ret = MPI_Comm_rank(MPI_COMM_WORLD, &myInfo.world_rank);
    _MPI_fail(ret, "MPI_Comm_rank");

    ret = MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    _MPI_fail(ret, "MPI_Comm_size");

    stdLogger.set_additional_prefix(sjoin("[Rank ", myInfo.world_rank, "] "));

    if (IamRoot()){
        rootLogger.setVerbosity(stdLogger.getVerbosity());
    }

    rootLogger.info("Initializing MPI with (", world_size, " proc)");

    /// Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);


    //! clean up nodeName (remove ".clusternet") for Bielefeld GPU Cluster:
    std::string nodeName = processor_name;
    std::string pattern = ".clusternet";
    std::string::size_type i = nodeName.find(pattern);
    while (i != std::string::npos) {
        nodeName.erase(i, pattern.length());
        i = nodeName.find(pattern, i);
    }
    //! end clean up

    myInfo.nodeName = nodeName;

}

void CommunicationBase::initNodeComm() {
    int ret;
    ret = MPI_Info_create(&mpi_info);
    _MPI_fail(ret, "MPI_Info_create");
    ret = MPI_Comm_split_type(getCart_comm(), MPI_COMM_TYPE_SHARED, MyRank(), mpi_info, &node_comm);
    _MPI_fail(ret, "MPI_Comm_split_type");
    ret = MPI_Comm_rank(node_comm, &myInfo.node_rank);
    _MPI_fail(ret, "MPI_Cart_Comm_rank (Node)");
}

inline std::string CommunicationBase::gpuAwareMPICheck() {

    std::stringstream check_result;

    check_result << "@compilation ";

    //! We can only check this reliably for openmpi. TODO: find out how to do it for other mpi libraries
#if OPEN_MPI \
    //! Check if the program was compiled with a CUDA-aware MPI library:
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    check_result << "YES, ";
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
    check_result << "NO, ";
#else
    check_result << "UNKNOWN, ";
#endif

    //! Check if the program is running using a CUDA-aware MPI library:
    check_result << "@runtime ";
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    if (1 == MPIX_Query_cuda_support()) {
        check_result << "YES";
    } else {
        check_result << "NO";
    }
#else /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
    check_result << "UNKNOWN";
#endif
#endif
    return check_result.str();

}

/// Initialize MPI
void CommunicationBase::init(const LatticeDimensions &Dim, const LatticeDimensions &Topo) {

    if (Dim.mult() != getNumberProcesses()) {
        throw std::runtime_error(stdLogger.fatal(
                    "Number of processes does not match with the Node dimensions: ", Dim[0],
                    " * ", Dim[1], " * ", Dim[2], " * ", Dim[3],
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
    _MPI_fail(ret, "MPI_Cart_get");  //now we have process coordinates


    int num_devices = 0;
    gpuGetDeviceCount(&num_devices);
    if (num_devices == 0) {
        throw std::runtime_error(stdLogger.fatal(myInfo.nodeName, " CPU_", sched_getcpu(),
                    " MPI world_rank=", myInfo.world_rank,
                    ": You didn't give me any GPU to work with! >:("));
    }

    if (Topo.summed() == 0)
        myInfo.deviceRank = myInfo.node_rank % num_devices; // choose GPU

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

    gpuSetDevice(myInfo.deviceRank);

    gpuDeviceProp tmpProp;
    gpuGetDeviceProperties(&tmpProp, myInfo.deviceRank);

    globalBarrier();
    rootLogger.info("> Running on:");
    globalBarrier();
    stdLogger.info("> ", myInfo.nodeName, " CPU_", sched_getcpu(), " GPU_",
            std::uppercase, std::hex, tmpProp.pciBusID,
            "; MPI: world_rank=", myInfo.world_rank,
            ", node_rank=", myInfo.node_rank, ", coord: ", myInfo.coord);
    globalBarrier();

    rootLogger.info("> Is MPI CUDA-aware? ", gpuAwareMPICheck());

    const int gpu_arch = tmpProp.major * static_cast<int>(10) + tmpProp.minor;
    rootLogger.info("> GPU compute capability: ", gpu_arch);

#ifdef ARCHITECTURE
    if (static_cast<int>(ARCHITECTURE) != gpu_arch) {
        throw std::runtime_error(stdLogger.fatal("You compiled for ARCHITECTURE=", ARCHITECTURE,
                    " but the GPUs here are ", gpu_arch));
    }
#else
    rootLogger.warn("Cannot determine for which compute capability the code was compiled!");
#endif
    globalBarrier();
    neighbor_info = NeighborInfo(cart_comm, myInfo);

}


CommunicationBase::~CommunicationBase() {
    rootLogger.info("Finalize CommunicationBase");
    int ret;
    ret = MPI_Comm_free(&cart_comm);
    _MPI_fail(ret, "MPI_Comm_free");

    MPI_Comm_free(&node_comm);
    MPI_Info_free(&mpi_info);

    /// Finalize MPI
    ret = MPI_Finalize();
    _MPI_fail(ret, "MPI_Finalize");
}

void CommunicationBase::_MPI_fail(int ret, const std::string &func) {
    if (ret != MPI_SUCCESS) {
        throw std::runtime_error(stdLogger.fatal(func, " failed!"));
    }
}

bool CommunicationBase::IamRoot() const {
    return (myInfo.world_rank == 0);
}

const LatticeDimensions &CommunicationBase::nodes() { return dims; }            /// Number of nodes in Cartesian grid
int CommunicationBase::getRank(LatticeDimensions c) const {
    int result;
    MPI_Cart_rank(cart_comm, c, &result);
    return result;
};

void CommunicationBase::root2all(int &value) const {
    MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void CommunicationBase::root2all(int64_t &value) const {
    MPI_Bcast(&value, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
}

void CommunicationBase::root2all(bool &value) const {
    MPI_Bcast(&value, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
}

void CommunicationBase::root2all(float &value) const {
    MPI_Bcast(&value, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void CommunicationBase::root2all(double &value) const {
    MPI_Bcast(&value, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void CommunicationBase::root2all(GCOMPLEX(float) &value) const {
    std::complex<float> v(value.cREAL, value.cIMAG);
    MPI_Bcast(&v, 1, MPI_COMPLEX, 0, MPI_COMM_WORLD);
    value.cREAL = v.real();
    value.cIMAG = v.imag();
}

void CommunicationBase::root2all(GCOMPLEX(double) &value) const {
    std::complex<double> v(value.cREAL, value.cIMAG);
    MPI_Bcast(&v, 1, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    value.cREAL = v.real();
    value.cIMAG = v.imag();
}

void CommunicationBase::root2all(Matrix4x4Sym<float> &value) const {
    MPI_Bcast(&value, 10, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void CommunicationBase::root2all(Matrix4x4Sym<double> &value) const {
    MPI_Bcast(&value, 10, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void CommunicationBase::root2all(GSU3<float> &value) const {
    MPI_Bcast(&value, 9, MPI_COMPLEX, 0, MPI_COMM_WORLD);
}

void CommunicationBase::root2all(GSU3<double> &value) const {
    MPI_Bcast(&value, 9, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
}

void CommunicationBase::root2all(std::vector<float> &v) const {
    MPI_Bcast(&v[0], v.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void CommunicationBase::root2all(std::vector<double> &v) const {
    MPI_Bcast(&v[0], v.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void CommunicationBase::root2all(std::vector<int> &v) const {
    MPI_Bcast(&v[0], v.size(), MPI_INT, 0, MPI_COMM_WORLD);
}

void CommunicationBase::root2all(std::vector<std::complex<float>> &v) const {
    MPI_Bcast(&v[0], v.size(), MPI_COMPLEX, 0, MPI_COMM_WORLD);
}

void CommunicationBase::root2all(std::vector<std::complex<double>> &v) const {
    MPI_Bcast(&v[0], v.size(), MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
}

void CommunicationBase::root2all(std::string &s) const {
    unsigned int size = s.size() + 1;
    MPI_Bcast(&size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    char *buf = new char[size];
    if (IamRoot())
        memcpy(buf, s.c_str(), size);
    MPI_Bcast(buf, size, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (!IamRoot())
        s = buf;
    delete[] buf;
}

float CommunicationBase::reduceMax(float in) const {
    float recv;
    MPI_Allreduce(&in, &recv, 1, MPI_FLOAT, MPI_MAX, cart_comm);
    return recv;
}


double CommunicationBase::reduceMax(double in) const {
    double recv;
    MPI_Allreduce(&in, &recv, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
    return recv;
}

int CommunicationBase::reduce(int in) const {
    int recv;
    MPI_Allreduce(&in, &recv, 1, MPI_INT, MPI_SUM, cart_comm);
    return recv;
}

uint32_t CommunicationBase::reduce(uint32_t in) const {
    uint32_t recv;
    MPI_Allreduce(&in, &recv, 1, MPI_UINT32_T, MPI_SUM, cart_comm);
    return recv;
}

float CommunicationBase::reduce(float in) const {
    float recv;
    MPI_Allreduce(&in, &recv, 1, MPI_FLOAT, MPI_SUM, cart_comm);
    return recv;
}

double CommunicationBase::reduce(double in) const {
    double recv;
    MPI_Allreduce(&in, &recv, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
    return recv;
}

std::complex<float> CommunicationBase::reduce(std::complex<float> in) const {
    std::complex<float> recv;
    MPI_Allreduce(&in, &recv, 1, MPI_COMPLEX, MPI_SUM, cart_comm);
    return recv;
}

std::complex<double> CommunicationBase::reduce(std::complex<double> in) const {
    std::complex<double> recv;
    MPI_Allreduce(&in, &recv, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, cart_comm);
    return recv;
}

GCOMPLEX(float) CommunicationBase::reduce(GCOMPLEX(float) in) const {
    std::complex<float> recv(in.cREAL, in.cIMAG);
    MPI_Allreduce(&in, &recv, 1, MPI_COMPLEX, MPI_SUM, cart_comm);
    return recv;
}

GCOMPLEX(double) CommunicationBase::reduce(GCOMPLEX(double) in) const {
    std::complex<double> recv(in.cREAL, in.cIMAG);
    MPI_Allreduce(&in, &recv, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, cart_comm);
    return recv;
}


Matrix4x4Sym<float> CommunicationBase::reduce(Matrix4x4Sym<float> in) const {
    Matrix4x4Sym<float> recv;
    MPI_Allreduce(&in, &recv, 10, MPI_FLOAT, MPI_SUM, cart_comm);
    return recv;
}

Matrix4x4Sym<double> CommunicationBase::reduce(Matrix4x4Sym<double> in) const {
    Matrix4x4Sym<double> recv;
    MPI_Allreduce(&in, &recv, 10, MPI_DOUBLE, MPI_SUM, cart_comm);
    return recv;
}

GSU3<float> CommunicationBase::reduce(GSU3<float> in) const {
    GSU3<float> recv;
    MPI_Allreduce(&in, &recv, 9, MPI_COMPLEX, MPI_SUM, cart_comm);
    return recv;
}

GSU3<double> CommunicationBase::reduce(GSU3<double> in) const {
    GSU3<double> recv;
    MPI_Allreduce(&in, &recv, 9, MPI_DOUBLE_COMPLEX, MPI_SUM, cart_comm);
    return recv;
}

void CommunicationBase::reduce(float *in, int nr) const {
    float *buf = new float[nr];
    MPI_Allreduce(in, buf, nr, MPI_FLOAT, MPI_SUM, cart_comm);
    for (int i = 0; i < nr; i++) in[i] = buf[i];
    delete[] buf;
}

void CommunicationBase::reduce(double *in, int nr) const {
    double *buf = new double[nr];
    MPI_Allreduce(in, buf, nr, MPI_DOUBLE, MPI_SUM, cart_comm);
    for (int i = 0; i < nr; i++) in[i] = buf[i];
    delete[] buf;
}

void CommunicationBase::reduce(GCOMPLEX(float) *in, int nr) const {
    GCOMPLEX(float) *buf = new GCOMPLEX(float)[nr];
    MPI_Allreduce(in, buf, nr, MPI_COMPLEX, MPI_SUM, cart_comm);
    for (int i = 0; i < nr; i++) in[i] = buf[i];
    delete[] buf;
}


void CommunicationBase::reduce(GCOMPLEX(double) *in, int nr) const {
    GCOMPLEX(double) *buf = new GCOMPLEX(double)[nr];
    MPI_Allreduce(in, buf, nr, MPI_DOUBLE_COMPLEX, MPI_SUM, cart_comm);
    for (int i = 0; i < nr; i++) in[i] = buf[i];
    delete[] buf;
}


void CommunicationBase::reduce(Matrix4x4Sym<float> *in, int nr) const {
    Matrix4x4Sym<float> *buf = new Matrix4x4Sym<float>[nr];
    MPI_Allreduce(reinterpret_cast<float *>(in), reinterpret_cast<float *>(buf), nr * 10, MPI_FLOAT, MPI_SUM,
            cart_comm);
    for (int i = 0; i < nr; i++) in[i] = buf[i];
    delete[] buf;
}

void CommunicationBase::reduce(Matrix4x4Sym<double> *in, int nr) const {
    Matrix4x4Sym<double> *buf = new Matrix4x4Sym<double>[nr];
    MPI_Allreduce(reinterpret_cast<double *>(in), reinterpret_cast<double *>(buf), nr * 10, MPI_DOUBLE, MPI_SUM,
            cart_comm);
    for (int i = 0; i < nr; i++) in[i] = buf[i];
    delete[] buf;
}

void CommunicationBase::reduce(GSU3<float> *in, int nr) const {
    GSU3<float> *buf = new GSU3<float>[nr];
    MPI_Allreduce(reinterpret_cast<float *>(in), reinterpret_cast<float *>(buf), nr * 9, MPI_COMPLEX, MPI_SUM,
            cart_comm);
    for (int i = 0; i < nr; i++) in[i] = buf[i];
    delete[] buf;
}


void CommunicationBase::reduce(GSU3<double> *in, int nr) const {
    GSU3<double> *buf = new GSU3<double>[nr];
    MPI_Allreduce(reinterpret_cast<double *>(in), reinterpret_cast<double *>(buf), nr * 9, MPI_DOUBLE_COMPLEX, MPI_SUM,
            cart_comm);
    for (int i = 0; i < nr; i++) in[i] = buf[i];
    delete[] buf;
}

void CommunicationBase::reduce(std::complex<float> *in, int nr) const {
    std::complex<float> *buf = new std::complex<float>[nr];
    MPI_Allreduce(in, buf, nr, MPI_COMPLEX, MPI_SUM, cart_comm);
    for (int i = 0; i < nr; i++) in[i] = buf[i];
    delete[] buf;
}

void CommunicationBase::reduce(std::complex<double> *in, int nr) const {
    std::complex<double> *buf = new std::complex<double>[nr];
    MPI_Allreduce(in, buf, nr, MPI_DOUBLE_COMPLEX, MPI_SUM, cart_comm);
    for (int i = 0; i < nr; i++) in[i] = buf[i];
    delete[] buf;
}

double CommunicationBase::globalAverage(double in) const {
    double recv;
    MPI_Allreduce(&in, &recv, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
    recv /= (double) world_size;
    return recv;
}

std::complex<double> CommunicationBase::globalAverage(std::complex<double> in) const {
    std::complex<double> recv;
    MPI_Allreduce(&in, &recv, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, cart_comm);
    recv /= world_size;
    return recv;
}

double CommunicationBase::globalMinimum(double in) const {
    double recv;
    MPI_Allreduce(&in, &recv, 1, MPI_DOUBLE, MPI_MIN, cart_comm);
    return recv;
}

double CommunicationBase::globalMaximum(double in) const {
    double recv;
    MPI_Allreduce(&in, &recv, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
    return recv;
}

float CommunicationBase::globalAverage(float in) const {
    float recv;
    MPI_Allreduce(&in, &recv, 1, MPI_FLOAT, MPI_SUM, cart_comm);
    recv /= (float) world_size;
    return recv;
}

std::complex<float> CommunicationBase::globalAverage(std::complex<float> in) const {
    std::complex<float> recv;
    MPI_Allreduce(&in, &recv, 1, MPI_COMPLEX, MPI_SUM, cart_comm);
    recv /= (float) world_size;
    return recv;
}

float CommunicationBase::globalMinimum(float in) const {
    float recv;
    MPI_Allreduce(&in, &recv, 1, MPI_FLOAT, MPI_MIN, cart_comm);
    return recv;
}

float CommunicationBase::globalMaximum(float in) const {
    float recv;
    MPI_Allreduce(&in, &recv, 1, MPI_FLOAT, MPI_MAX, cart_comm);
    return recv;
}

template <bool onDevice>
int CommunicationBase::updateSegment(HaloSegment hseg, size_t direction,
        int leftRight, HaloOffsetInfo<onDevice> &HalInfo) {
    HaloSegmentInfo &seg = HalInfo.get(hseg, direction, leftRight);

    NeighborInfo &NInfo = HalInfo.getNeighborInfo();

    ProcessInfo &info = NInfo.getNeighborInfo(hseg, direction, leftRight);

    int rank = info.world_rank;


    gpuError_t gpuErr;
    if (seg.getLength() != 0) {
        if (onDevice && info.p2p && useGpuP2P()) {
            // communicate via GPUDirect
            uint8_t *sendBase = seg.getMyDeviceSourcePtr();
            uint8_t *recvBase = seg.getDeviceDestinationPtrP2P();

            IF(COMBASE_DEBUG)(stdLogger.debug("gpuMemcpyAsync: Copy ", seg.getLength(),
                                 " bytes from rank ", MyRank(), " to rank ", info.world_rank);)

                gpuErr = gpuMemcpyAsync(recvBase, sendBase, seg.getLength(), gpuMemcpyDeviceToDevice,
                        seg.getDeviceStream());
            if (gpuErr != gpuSuccess)
                GpuError("communicationBase_mpi.cpp: Failed to copy data (DeviceToDevice) (1a)", gpuErr);

        } else if (onDevice && info.sameRank && (useGpuP2P() || gpuAwareMPIAvail())) {
            uint8_t *sendBase = seg.getMyDeviceSourcePtr();
            uint8_t *recvBase = seg.getMyDeviceDestinationPtr();

            IF(COMBASE_DEBUG) (stdLogger.debug("gpuMemcpyAsync (same rank): Copy ", seg.getLength(),
                                 " bytes on rank ", MyRank());)

                gpuErr = gpuMemcpyAsync(recvBase, sendBase, seg.getLength(), gpuMemcpyDeviceToDevice,
                        seg.getDeviceStream(0));
            if (gpuErr != gpuSuccess)
                GpuError("communicationBase_mpi.cpp: Failed to copy data (DeviceToDevice) (1b)", gpuErr);

        } else if (onDevice && gpuAwareMPIAvail()) {
            uint8_t *sendBase = seg.getMyDeviceSourcePtr();
            uint8_t *recvBase = seg.getDeviceDestinationPtrGPUAwareMPI();

            int index = haloSegmentCoordToIndex(hseg, direction, leftRight);
            int indexDest = haloSegmentCoordToIndex(hseg, direction, !leftRight);

            IF(COMBASE_DEBUG) (stdLogger.debug("MPI_Isend (gpu): Send ", seg.getLength(),
                                 " bytes from rank ", MyRank(), " to rank ", info.world_rank);)

                MPI_Isend(sendBase, 1, seg.getMpiType(), rank, indexDest, cart_comm, &seg.getRequestSend());
            MPI_Irecv(recvBase, 1, seg.getMpiType(), rank, index, cart_comm, &seg.getRequestRecv());
        } else {

            uint8_t *sendBase = seg.getHostSendPtr();
            uint8_t *recvBase = seg.getHostRecvPtr();

            int index = haloSegmentCoordToIndex(hseg, direction, leftRight);
            int indexDest = haloSegmentCoordToIndex(hseg, direction, !leftRight);

            IF(COMBASE_DEBUG) (stdLogger.debug("MPI_Isend (cpu): Send ", seg.getLength(),
                                 " bytes from rank ", MyRank(), " to rank ", info.world_rank);)
                MPI_Isend(sendBase, 1, seg.getMpiType(), rank, indexDest, cart_comm, &seg.getRequestSend());
            MPI_Irecv(recvBase, 1, seg.getMpiType(), rank, index, cart_comm, &seg.getRequestRecv());
        }
    }
    return seg.getLength();
}


template<bool onDevice>
void CommunicationBase::updateAll(HaloOffsetInfo<onDevice> &HalInfo, unsigned int param) {

    if (param & COMM_START) {

        if (param & Hyperplane) {
            for (const HaloSegment &hseg : HaloHypPlanes) {
                for (int dir = 0; dir < HaloSegmentDirections(hseg); dir++) {
                    updateSegment<onDevice>(hseg, dir, 0, HalInfo);
                    updateSegment<onDevice>(hseg, dir, 1, HalInfo);
                }
            }
        }
        if (param & Plane) {
            for (const HaloSegment &hseg : HaloPlanes) {
                for (int dir = 0; dir < HaloSegmentDirections(hseg); dir++) {
                    updateSegment<onDevice>(hseg, dir, 0, HalInfo);
                    updateSegment<onDevice>(hseg, dir, 1, HalInfo);
                }
            }
        }
        if (param & Stripe) {
            for (const HaloSegment &hseg : HaloStripes) {
                for (int dir = 0; dir < HaloSegmentDirections(hseg); dir++) {
                    updateSegment<onDevice>(hseg, dir, 0, HalInfo);
                    updateSegment<onDevice>(hseg, dir, 1, HalInfo);
                }
            }
        }
        if (param & Corner) {
            for (const HaloSegment &hseg : HaloCorners) {
                for (int dir = 0; dir < HaloSegmentDirections(hseg); dir++) {
                    updateSegment<onDevice>(hseg, dir, 0, HalInfo);
                    updateSegment<onDevice>(hseg, dir, 1, HalInfo);
                }
            }
        }


    }
    if (param & COMM_FINISH) {
        HalInfo.syncAllStreamRequests();
    }

}


// set MPI Barrier
void CommunicationBase::globalBarrier() const {
    MPI_Barrier(cart_comm);
}

void CommunicationBase::nodeBarrier() const {
    MPI_Barrier(node_comm);
}

void CommunicationBase::initIOBinary(std::string fileName, size_t filesize, size_t bytesPerSite, size_t displacement,
        LatticeDimensions globalLattice, LatticeDimensions localLattice, IO_Mode mode) {

    LatticeDimensions offset = mycoords() * localLattice;

    MPI_Type_contiguous(bytesPerSite, MPI_BYTE, &basetype);
    MPI_Type_commit(&basetype);

    // create and set a view
    MPI_Type_create_subarray(4, globalLattice, localLattice, offset, MPI_ORDER_FORTRAN, basetype, &fvtype);
    MPI_Type_commit(&fvtype);

    stdLogger.debug("MPI File [", Pad0(2, MyRank()), "] ", globalLattice, " ",
            localLattice, " ", offset);

    int mpi_mode = 0;
    if (mode == READ) mpi_mode = MPI_MODE_RDONLY;
    if (mode == WRITE) mpi_mode = MPI_MODE_WRONLY;
    if (mode == READWRITE) mpi_mode = MPI_MODE_RDWR;

    if (MPI_File_open(MPI_COMM_WORLD, const_cast<char *>(fileName.c_str()),
                mpi_mode, MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
        throw std::runtime_error(stdLogger.fatal("Unable to read/write binary file: ", fileName));
    }

    if (mode != READ) MPI_File_set_size(fh, filesize); //truncate if file exists and is too large

    char fnative[10] = "native";
    MPI_File_set_view(fh, displacement, basetype, fvtype, fnative, MPI_INFO_NULL);
}

void CommunicationBase::writeBinary(void *buffer, size_t elemCount) {

    MPI_File_write_all(fh, buffer, elemCount, basetype, MPI_STATUS_IGNORE);
}

void CommunicationBase::readBinary(void *buffer, size_t elemCount) {
    MPI_File_read_all(fh, buffer, elemCount, basetype, MPI_STATUS_IGNORE);
}

void CommunicationBase::closeIOBinary() {

    MPI_File_close(&fh);
    MPI_Type_free(&fvtype);
    MPI_Type_free(&basetype);
}

template int CommunicationBase::updateSegment<true>(HaloSegment hseg, size_t direction, int leftRight,
        HaloOffsetInfo<true> &HalInfo);

template int CommunicationBase::updateSegment<false>(HaloSegment hseg, size_t direction, int leftRight,
        HaloOffsetInfo<false> &HalInfo);

template void CommunicationBase::updateAll<true>(HaloOffsetInfo<true> &HalInfo, unsigned int haltype);

template void CommunicationBase::updateAll<false>(HaloOffsetInfo<false> &HalInfo, unsigned int haltype);



#endif
