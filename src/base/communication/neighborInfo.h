/* 
 * neighborInfo.h                                                               
 * 
 * L. Mazur 
 *
 * In order to communicate, each sublattice needs to know something about his neighbors,
 * for example their rank or whether they are on the same node. (See ProcessInfo struct.)
 *
 */

#ifndef NEIGHBORINFO_H
#define NEIGHBORINFO_H

#include "../../define.h"
#include <mpi.h>

#include <utility>
#include "../LatticeDimension.h"
#include "../wrapper/gpu_wrapper.h"

//#define PEERACCESSINFO

struct ProcessInfo {
    int world_rank;
    int node_rank;
    LatticeDimensions coord;
    int deviceRank;
    int nodeNameSize;
    std::string nodeName;
    bool onNode;
    bool sameRank = false;
    bool p2p = false;

    ProcessInfo() : world_rank(0), coord(LatticeDimensions(0, 0, 0, 0)), deviceRank(0),
                    onNode(true) {}


    void checkLocation(const int myRank, const std::string &myNodeName) {
        onNode = (nodeName == myNodeName);
        sameRank = (world_rank == myRank);
    }
};

class NeighborInfo {
private:

    MPI_Comm cart_comm;
    ProcessInfo myInfo;

    ProcessInfo _X[2];
    ProcessInfo _Y[2];
    ProcessInfo _Z[2];
    ProcessInfo _T[2];
    ProcessInfo _XY[2][2];
    ProcessInfo _XZ[2][2];
    ProcessInfo _XT[2][2];
    ProcessInfo _YZ[2][2];
    ProcessInfo _YT[2][2];
    ProcessInfo _ZT[2][2];
    ProcessInfo _XYZ[4][2];
    ProcessInfo _XYT[4][2];
    ProcessInfo _XZT[4][2];
    ProcessInfo _YZT[4][2];
    ProcessInfo _XYZT[8][2];

    ProcessInfo fail;

    gpuDeviceProp myProp;

    inline void _fill2DNeighbors(ProcessInfo array[][2], int mu, int nu);

    inline void _fill3DNeighbors(ProcessInfo array[][2], int mu, int nu, int rho);

    inline void _fill4DNeighbors(ProcessInfo array[][2]);


    inline short _opposite(short number, short block);

    inline LatticeDimensions _shortToDim(short number);

    inline void sendRecvCoord(ProcessInfo &info, MPI_Request *sendReq, MPI_Request *recvReq, int index, int indexDest);

    inline void
    sendRecvDeviceRank(ProcessInfo &info, MPI_Request *sendReq, MPI_Request *recvReq, int index, int indexDest);

    inline void
    sendRecvNodeRank(ProcessInfo &info, MPI_Request *sendReq, MPI_Request *recvReq, int index, int indexDest);

    inline void
    sendRecvNodeNameSize(ProcessInfo &info, MPI_Request *sendReq, MPI_Request *recvReq, int index, int indexDest);

    inline void
    sendRecvNodeName(ProcessInfo &info, MPI_Request *sendReq, MPI_Request *recvReq, int index, int indexDest);


    inline bool IsGPUCapableP2P() const;

    inline void checkP2P();


public:
    NeighborInfo(const NeighborInfo &info) = delete;

    NeighborInfo(MPI_Comm CartComm, ProcessInfo info) : cart_comm(CartComm), myInfo(info) {

        rootLogger.debug("> Collecting neighbor information...");
        MPI_Cart_shift(cart_comm, 0, 1, &_X[0].world_rank, &_X[1].world_rank);
        MPI_Cart_shift(cart_comm, 1, 1, &_Y[0].world_rank, &_Y[1].world_rank);
        MPI_Cart_shift(cart_comm, 2, 1, &_Z[0].world_rank, &_Z[1].world_rank);
        MPI_Cart_shift(cart_comm, 3, 1, &_T[0].world_rank, &_T[1].world_rank);

        _fill2DNeighbors(_XY, 0, 1);
        _fill2DNeighbors(_XZ, 0, 2);
        _fill2DNeighbors(_XT, 0, 3);
        _fill2DNeighbors(_YZ, 1, 2);
        _fill2DNeighbors(_YT, 1, 3);
        _fill2DNeighbors(_ZT, 2, 3);

        _fill3DNeighbors(_XYZ, 0, 1, 2);
        _fill3DNeighbors(_XYT, 0, 1, 3);
        _fill3DNeighbors(_XZT, 0, 2, 3);
        _fill3DNeighbors(_YZT, 1, 2, 3);

        _fill4DNeighbors(_XYZT);

        myInfo.nodeNameSize = myInfo.nodeName.length();
        exchangeProcessInfo();
        /// I don't know of a better solution yet...
        fail.world_rank = -99;
        fail.deviceRank = -99;
        fail.coord = LatticeDimensions(-99, -99, -99, -99);
        fail.onNode = false;

#ifndef CPUONLY
        gpuError_t gpuErr = gpuGetDeviceProperties(&myProp, myInfo.deviceRank);
        if (gpuErr != gpuSuccess) {
            GpuError("neighborInfo.h: gpuGetDeviceProperties failed:", gpuErr);
        }

        rootLogger.info("> Checking support for P2P (Peer-to-Peer) and UVA (Unified Virtual Addressing):");
        MPI_Barrier(cart_comm);
        checkP2P();
#endif
        MPI_Barrier(cart_comm);
    }

    NeighborInfo() = default;

    ~NeighborInfo() = default;

    inline void exchangeProcessInfo();


    ProcessInfo &getMyInfo() { return myInfo; }

    ProcessInfo &getNeighborInfo(HaloSegment hseg, size_t direction, int leftRight) {
        if (hseg == X) { return _X[leftRight]; }
        else if (hseg == Y) return _Y[leftRight];
        else if (hseg == Z) return _Z[leftRight];
        else if (hseg == T) return _T[leftRight];

        else if (hseg == XY) {
            if (direction < 2 && leftRight < 2) return _XY[direction][leftRight];
            else {
                rootLogger.error("neighborinfo.h: getNeighborInfo(...): dir or lr is wrong!");
                return fail;
            }
        } else if (hseg == XZ) {
            if (direction < 2 && leftRight < 2) return _XZ[direction][leftRight];
            else {
                rootLogger.error("neighborinfo.h: getNeighborInfo(...): dir or lr is wrong!");
                return fail;
            }
        } else if (hseg == XT) {
            if (direction < 2 && leftRight < 2) return _XT[direction][leftRight];
            else {
                rootLogger.error("neighborinfo.h: getNeighborInfo(...): dir or lr is wrong!");
                return fail;
            }
        } else if (hseg == YZ) {
            if (direction < 2 && leftRight < 2) return _YZ[direction][leftRight];
            else {
                rootLogger.error("neighborinfo.h: getNeighborInfo(...): dir or lr is wrong!");
                return fail;
            }
        } else if (hseg == YT) {
            if (direction < 2 && leftRight < 2) return _YT[direction][leftRight];
            else {
                rootLogger.error("neighborinfo.h: getNeighborInfo(...): dir or lr is wrong!");
                return fail;
            }
        } else if (hseg == ZT) {
            if (direction < 2 && leftRight < 2) return _ZT[direction][leftRight];
            else {
                rootLogger.error("neighborinfo.h: getNeighborInfo(...): dir or lr is wrong!");
                return fail;
            }
        } else if (hseg == XYZ) return _XYZ[direction][leftRight];
        else if (hseg == XYT) return _XYT[direction][leftRight];
        else if (hseg == XZT) return _XZT[direction][leftRight];
        else if (hseg == YZT) return _YZT[direction][leftRight];

        else if (hseg == XYZT) return _XYZT[direction][leftRight];
        else {
            rootLogger.error("neighborinfo.h: getNeighborInfo(...): hseg is wrong!");
            return fail;
        }
    }

};


inline short NeighborInfo::_opposite(short number, short block) {
    if (block == 1)
        return (~number) & (short) 1;
    if (block == 2)
        return (~number) & (short) 3;
    if (block == 3)
        return (~number) & (short) 7;
    if (block == 4)
        return (~number) & (short) 15;
    return 999; // One should think of a better solution ...
}


inline LatticeDimensions NeighborInfo::_shortToDim(short number) {
    LatticeDimensions ret;
    ret[0] = (number & (short) 1) ? 1 : -1;
    ret[1] = (number & (short) 2) ? 1 : -1;
    ret[2] = (number & (short) 4) ? 1 : -1;
    ret[3] = (number & (short) 8) ? 1 : -1;


    return ret;
}

inline void NeighborInfo::_fill2DNeighbors(ProcessInfo array[][2], int mu, int nu) {
    for (short i = 0; i < 2; i++) {
        LatticeDimensions c = myInfo.coord;
        LatticeDimensions left = _shortToDim(i);
        c[mu] += left[0];
        c[nu] += left[1];
        MPI_Cart_rank(cart_comm, c, &array[i][0].world_rank);

        c = myInfo.coord;
        LatticeDimensions right = _shortToDim(_opposite(i, 2));
        c[mu] += right[0];
        c[nu] += right[1];
        MPI_Cart_rank(cart_comm, c, &array[i][1].world_rank);
        // rootLogger.info("2D: (" , left[0] ,  " " ,  left[1] ,  ") (" ,  right[0] ,  " " ,  right[1]  ,  ") i=" , i , " i_opp=" ,  _opposite(i, 2), std::endl);
    }
}


inline void NeighborInfo::_fill3DNeighbors(ProcessInfo array[][2], int mu, int nu, int rho) {
    for (short i = 0; i < 4; i++) {
        LatticeDimensions c = myInfo.coord;
        LatticeDimensions left = _shortToDim(i);
        c[mu] += left[0];
        c[nu] += left[1];
        c[rho] += left[2];
        MPI_Cart_rank(cart_comm, c, &array[i][0].world_rank);

        c = myInfo.coord;
        LatticeDimensions right = _shortToDim(_opposite(i, 3));
        c[mu] += right[0];
        c[nu] += right[1];
        c[rho] += right[2];
        MPI_Cart_rank(cart_comm, c, &array[i][1].world_rank);
        //rootLogger.info("3D: (" , left[0] ,  " " ,  left[1] ,  " " ,  left[2] ,  ") (" ,  right[0] ,  " " ,  right[1] ,  " " ,  right[2] ,  ") i=" , i , " i_opp=" ,  _opposite(i, 3), std::endl);
    }
}


inline void NeighborInfo::_fill4DNeighbors(ProcessInfo array[][2]) {
    for (short i = 0; i < 8; i++) {
        LatticeDimensions c = myInfo.coord;
        LatticeDimensions left = _shortToDim(i);
        c = c + left;
        //c[0] += left[0];
        //c[1] += left[1];
        //c[2] += left[2];
        //c[3] += left[3];
        MPI_Cart_rank(cart_comm, c, &array[i][0].world_rank);

        c = myInfo.coord;
        LatticeDimensions right = _shortToDim(_opposite(i, 4));
        //if(IamRoot())std::cout <<"4D:" <<left << " " << right << " i=" <<i <<" i_opp=" << _opposite(i, 4)<<std::endl;
        //  rootLogger.info("4D: (" , left[0] ,  " " ,  left[1] ,  " " ,  left[2] ,  " " , left[3] ,  ") (" ,  right[0] ,  " " ,  right[1] ,  " " ,  right[2], " ",  right[3] ,  ") i=" , i , " i_opp=" ,  _opposite(i, 4), std::endl);
        c = c + right;
        //c[0] += right[0];
        //c[1] += right[1];
        //c[2] += right[2];
        //c[3] += right[3];
        MPI_Cart_rank(cart_comm, c, &array[i][1].world_rank);
    }
}


inline void
NeighborInfo::sendRecvCoord(ProcessInfo &info, MPI_Request *sendReq, MPI_Request *recvReq, int index,
                            int indexDest) {
    /// send coord
    MPI_Isend(&myInfo.coord, 4, MPI_INT, info.world_rank, indexDest, cart_comm, sendReq);
    MPI_Irecv(&info.coord, 4, MPI_INT, info.world_rank, index, cart_comm, recvReq);

}

inline void
NeighborInfo::sendRecvDeviceRank(ProcessInfo &info, MPI_Request *sendReq, MPI_Request *recvReq, int index,
                                 int indexDest) {
    /// send deviceRank
    MPI_Isend(&myInfo.deviceRank, 1, MPI_INT, info.world_rank, indexDest, cart_comm, sendReq);
    MPI_Irecv(&info.deviceRank, 1, MPI_INT, info.world_rank, index, cart_comm, recvReq);
}

inline void NeighborInfo::sendRecvNodeRank(ProcessInfo &info, MPI_Request *sendReq, MPI_Request *recvReq, int index,
                                           int indexDest) {
    /// send nodeRank
    MPI_Isend(&myInfo.node_rank, 1, MPI_INT, info.world_rank, indexDest, cart_comm, sendReq);
    MPI_Irecv(&info.node_rank, 1, MPI_INT, info.world_rank, index, cart_comm, recvReq);
}


inline void
NeighborInfo::sendRecvNodeNameSize(ProcessInfo &info, MPI_Request *sendReq, MPI_Request *recvReq, int index,
                                   int indexDest) {

    /// send nodeName
    MPI_Isend(&myInfo.nodeNameSize, 1, MPI_INT, info.world_rank, indexDest, cart_comm, sendReq);
    MPI_Irecv(&info.nodeNameSize, 1, MPI_INT, info.world_rank, index, cart_comm, recvReq);

}

/// info.nodeName should be resized properly beforehand!
inline void NeighborInfo::sendRecvNodeName(ProcessInfo &info, MPI_Request *sendReq, MPI_Request *recvReq, int index,
                                           int indexDest) {
    int sendSize = myInfo.nodeName.length();
    int recvSize = info.nodeNameSize;

    MPI_Isend(myInfo.nodeName.c_str(), sendSize + 1, MPI_CHAR, info.world_rank, indexDest, cart_comm, sendReq);
    MPI_Irecv(&info.nodeName[0], recvSize + 1, MPI_CHAR, info.world_rank, index, cart_comm, recvReq);

}

inline void NeighborInfo::exchangeProcessInfo() {

    MPI_Request sendReq_coord[80];
    MPI_Request recvReq_coord[80];
    for (const HaloSegment &hseg : AllHaloSegments) {
        for (int dir = 0; dir < HaloSegmentDirections(hseg); dir++) {
            for (int leftRight = 0; leftRight <= 1; leftRight++) {
                int index = haloSegmentCoordToIndex(hseg, dir, leftRight);
                int indexDest = haloSegmentCoordToIndex(hseg, dir, !leftRight);
                sendRecvCoord(getNeighborInfo(hseg, dir, leftRight), &sendReq_coord[index], &recvReq_coord[index],
                              index, indexDest);
            }
        }
    }

    MPI_Waitall(80, sendReq_coord, MPI_STATUSES_IGNORE);
    MPI_Waitall(80, recvReq_coord, MPI_STATUSES_IGNORE);

    MPI_Request sendReq_devRank[80];
    MPI_Request recvReq_devRank[80];
    for (const HaloSegment &hseg : AllHaloSegments) {
        for (int dir = 0; dir < HaloSegmentDirections(hseg); dir++) {
            for (int leftRight = 0; leftRight <= 1; leftRight++) {
                int index = haloSegmentCoordToIndex(hseg, dir, leftRight);
                int indexDest = haloSegmentCoordToIndex(hseg, dir, !leftRight);
                sendRecvDeviceRank(getNeighborInfo(hseg, dir, leftRight), &sendReq_devRank[index],
                                   &recvReq_devRank[index], index, indexDest);
            }
        }
    }
    MPI_Waitall(80, sendReq_devRank, MPI_STATUSES_IGNORE);
    MPI_Waitall(80, recvReq_devRank, MPI_STATUSES_IGNORE);

    MPI_Request sendReq_nodRank[80];
    MPI_Request recvReq_nodRank[80];
    for (const HaloSegment &hseg : AllHaloSegments) {
        for (int dir = 0; dir < HaloSegmentDirections(hseg); dir++) {
            for (int leftRight = 0; leftRight <= 1; leftRight++) {
                int index = haloSegmentCoordToIndex(hseg, dir, leftRight);
                int indexDest = haloSegmentCoordToIndex(hseg, dir, !leftRight);
                sendRecvNodeRank(getNeighborInfo(hseg, dir, leftRight), &sendReq_nodRank[index],
                                 &recvReq_nodRank[index], index, indexDest);
            }
        }
    }
    MPI_Waitall(80, sendReq_nodRank, MPI_STATUSES_IGNORE);
    MPI_Waitall(80, recvReq_nodRank, MPI_STATUSES_IGNORE);

    MPI_Request sendReq_nameSize[80];
    MPI_Request recvReq_nameSize[80];
    for (const HaloSegment &hseg : AllHaloSegments) {
        for (int dir = 0; dir < HaloSegmentDirections(hseg); dir++) {
            for (int leftRight = 0; leftRight <= 1; leftRight++) {
                int index = haloSegmentCoordToIndex(hseg, dir, leftRight);
                int indexDest = haloSegmentCoordToIndex(hseg, dir, !leftRight);
                sendRecvNodeNameSize(getNeighborInfo(hseg, dir, leftRight), &sendReq_nameSize[index],
                                     &recvReq_nameSize[index], index, indexDest);
            }
        }
    }
    MPI_Waitall(80, sendReq_nameSize, MPI_STATUSES_IGNORE);
    MPI_Waitall(80, recvReq_nameSize, MPI_STATUSES_IGNORE);

    for (const HaloSegment &hseg : AllHaloSegments) {
        for (int dir = 0; dir < HaloSegmentDirections(hseg); dir++) {
            for (int leftRight = 0; leftRight <= 1; leftRight++) {
                ProcessInfo &info = getNeighborInfo(hseg, dir, leftRight);
                info.nodeName.resize(info.nodeNameSize);
            }
        }
    }
    MPI_Request sendReq_name[80];
    MPI_Request recvReq_name[80];
    for (const HaloSegment &hseg : AllHaloSegments) {
        for (int dir = 0; dir < HaloSegmentDirections(hseg); dir++) {
            for (int leftRight = 0; leftRight <= 1; leftRight++) {
                int index = haloSegmentCoordToIndex(hseg, dir, leftRight);
                int indexDest = haloSegmentCoordToIndex(hseg, dir, !leftRight);
                sendRecvNodeName(getNeighborInfo(hseg, dir, leftRight), &sendReq_name[index], &recvReq_name[index],
                                 index, indexDest);
            }
        }
    }
    MPI_Waitall(80, sendReq_name, MPI_STATUSES_IGNORE);
    MPI_Waitall(80, recvReq_name, MPI_STATUSES_IGNORE);

    for (const HaloSegment &hseg : AllHaloSegments) {
        for (int dir = 0; dir < HaloSegmentDirections(hseg); dir++) {
            for (int leftRight = 0; leftRight <= 1; leftRight++) {
                ProcessInfo &info = getNeighborInfo(hseg, dir, leftRight);
                info.checkLocation(myInfo.world_rank, myInfo.nodeName);
            }
        }
    }
    MPI_Barrier(cart_comm);
    rootLogger.debug("> Neighbor information collected!");
}

inline bool NeighborInfo::IsGPUCapableP2P() const {
    // This requires two processes accessing each device, so we need
    // to ensure exclusive or prohibited mode is not set
    if (myProp.computeMode != gpuComputeModeDefault) {
        throw std::runtime_error(stdLogger.fatal("Device ", myProp.name, " is in an unsupported compute mode (exclusive or prohibited mode is NOT allowed)"));
    }
    return (bool) (myProp.major >= 2);

}


inline void NeighborInfo::checkP2P() {

//! This checks
    for (const HaloSegment &hseg : AllHaloSegments) {
        for (int dir = 0; dir < HaloSegmentDirections(hseg); dir++) {
            for (int leftRight = 0; leftRight <= 1; leftRight++) {
                ProcessInfo &nInfo = getNeighborInfo(hseg, dir, leftRight);
                if (nInfo.onNode) {
                    int can_access_peer;
                    gpuError_t gpuErr = gpuDeviceCanAccessPeer(&can_access_peer, myInfo.deviceRank, nInfo.deviceRank);
                    if (gpuErr != gpuSuccess) {
                        GpuError("neighborInfo.h: gpuDeviceCanAccessPeer failed:", gpuErr);
                    }
                    nInfo.p2p = (bool) can_access_peer;
#ifdef PEERACCESSINFO
                    if (!nInfo.sameRank)
                        stdLogger.debug("> Peer access on node " ,  myInfo.nodeName ,  " from GPU "
                                          ,  myInfo.deviceRank
                                          ,  " -> GPU " ,  nInfo.deviceRank ,  (can_access_peer ? " Yes" : " No"));
#endif

                }
            }
        }
    }

    stdLogger.info("> " ,  myInfo.nodeName ,  " GPU_" ,  std::uppercase ,  std::hex ,  myProp.pciBusID ,  "(" ,  myProp.name ,  "): "
#if USE_GPU_P2P
     ,  "P2P " ,  (IsGPUCapableP2P() ? "YES" : "NO") ,  "; "
#else
     , "P2P NO ; "
#endif
#ifdef USE_CUDA
      , "UVA ", (myProp.unifiedAddressing ? "YES" : "NO"));
#elif defined USE_HIP
      , "UVA ", "Unknown (HIP does not support this!)");
#endif
}

#endif //NEIGHBORINFO_H
