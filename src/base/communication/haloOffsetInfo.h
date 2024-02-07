/*
 * haloOffsetInfo.h
 *
 * L. Mazur
 *
 * The sites on a sublattice's halo have some cartesian index. When communicating, information
 * from these sites are saved into a buffer. The cardinality of this communication buffer is
 * smaller than the sublattice because it only contains halo information; hence it has its own
 * indexing scheme. Loosely speaking, the difference between these indexing schemes is the offset.
 * This header contains methods to calculate that.
 *
 */

#pragma once
#include <mpi.h>
#include "../../define.h"
#include <map>
#include <utility>
#include "neighborInfo.h"
#include "../memoryManagement.h"
#include <array>

const size_t MAX_NUM_HALOSEGMENTS = 80;

#ifdef COMMS_STREAMS
const size_t MAX_NUM_STREAMS = 160;
#else
const size_t MAX_NUM_STREAMS = 2;
#endif

typedef std::array<gpuStream_t, MAX_NUM_STREAMS> commStreams_t;

struct HaloSegmentInfo {
private:
    gMemoryPtr<false> sendBase = MemoryManagement::getMemAt<false>("SHARED_NULL");
    gMemoryPtr<false> recvBase = MemoryManagement::getMemAt<false>("SHARED_NULL");

    gMemoryPtr<true> destinationBase = MemoryManagement::getMemAt<true>("SHARED_NULL");
    gMemoryPtr<true> sourceBase = MemoryManagement::getMemAt<true>("SHARED_NULL");
    size_t offset = 0;
    size_t reverseOffset = 0;
    bool p2p = false;
    int oppositeP2PRank = 0;
    size_t length = 0;
    MPI_Datatype MpiType;
    bool sendReqUsed = false;
    bool recvReqUsed = false;
    MPI_Request hostRequestSend = 0;
    MPI_Request hostRequestRecv = 0;
    gpuStream_t& sendStream;
    gpuStream_t& receiveStream;
    bool sendStreamUsed = false;
    bool receiveStreamUsed = false;

public:

    HaloSegmentInfo(gpuStream_t &sendStream, gpuStream_t &recvStream) : sendStream(sendStream), receiveStream(recvStream) {
        MpiType = MPI_DATATYPE_NULL;
        hostRequestSend = MPI_REQUEST_NULL;
        hostRequestRecv = MPI_REQUEST_NULL;
    }

    //! copy constructor
    HaloSegmentInfo(HaloSegmentInfo&) = delete;

    //! copy assignment
    HaloSegmentInfo& operator=(HaloSegmentInfo&) = delete;

    //! move assignment
    HaloSegmentInfo& operator=(HaloSegmentInfo&&) = delete;

    //! move constructor
    HaloSegmentInfo(HaloSegmentInfo&& source) noexcept :
    //! move the gMemoryPtr's
    sendBase(std::move(source.sendBase)),
    recvBase(std::move(source.recvBase)),
    destinationBase(std::move(source.destinationBase)),
    sourceBase(std::move(source.sourceBase)),

    //! these can be copied as they are just basic types
    offset(source.offset),
    reverseOffset(source.reverseOffset),
    p2p(source.p2p),
    oppositeP2PRank(source.oppositeP2PRank),
    length(source.length),

    //! This is just a pointer
    MpiType(source.MpiType),

    //! again basic types
    sendReqUsed(source.sendReqUsed),
    recvReqUsed(source.recvReqUsed),

    //! MPI_Request. These are just pointers
    hostRequestSend(source.hostRequestSend),
    hostRequestRecv(source.hostRequestRecv),

    //! these are vectors of basic types
    // deviceStream(std::move(source.deviceStream)),
    sendStream(source.sendStream),
    receiveStream(source.receiveStream),
    sendStreamUsed(std::move(source.sendStreamUsed)),
    receiveStreamUsed(std::move(source.receiveStreamUsed))
    {
        //! make source hollow
        source.offset = 0;
        source.reverseOffset = 0;
        source.p2p = false;
        source.oppositeP2PRank = 0;
        source.length = 0;
        source.MpiType = MPI_DATATYPE_NULL;
        source.hostRequestSend = MPI_REQUEST_NULL;
        source.hostRequestRecv = MPI_REQUEST_NULL;
    }

    void setSendBase(const gMemoryPtr<false> &_sendBase) { HaloSegmentInfo::sendBase = _sendBase; }

    void setRecvBase(const gMemoryPtr<false> &_recvBase) { HaloSegmentInfo::recvBase = _recvBase; }

    void setDestinationBase(
            const gMemoryPtr<true> &_destinationBase) { HaloSegmentInfo::destinationBase = _destinationBase; }

    void setSourceBase(const gMemoryPtr<true> &_sourceBase) { HaloSegmentInfo::sourceBase = _sourceBase; }

    const uint8_t *getSendBase() const { return sendBase->template getPointer<uint8_t>(); }

    const uint8_t *getRecvBase() const { return recvBase->template getPointer<uint8_t>(); }

    const uint8_t *getDestinationBase() const { return destinationBase->template getPointer<uint8_t>(); }

    const uint8_t *getSourceBase() const { return sourceBase->template getPointer<uint8_t>(); }

    void setOffset(size_t _offset) { HaloSegmentInfo::offset = _offset; }

    void setReverseOffset(size_t _reverseOffset) { HaloSegmentInfo::reverseOffset = _reverseOffset; }

    void setP2P(bool _p2p) { p2p = _p2p; }

    void setOppositeP2PRank(int _oppositeP2PRank) { HaloSegmentInfo::oppositeP2PRank = _oppositeP2PRank; }

    void setLength(size_t _length) { HaloSegmentInfo::length = _length; }

    size_t getOffset() const { return offset; }

    size_t getReverseOffset() const { return reverseOffset; }

    bool isP2P() const { return p2p; }

    int getOppositeP2PRank() const { return oppositeP2PRank; }

    size_t getLength() const { return length; }

    MPI_Datatype &getMpiType() { return MpiType; }


    MPI_Request &getRequestSend() {
        sendReqUsed = true;
        return hostRequestSend;
    }

    MPI_Request &getRequestRecv() {
        recvReqUsed = true;
        return hostRequestRecv;
    }

    // gpuStream_t &getDeviceStream(int ind = 0) {
    //     streamUsed[ind] = true;

    //     return deviceStream[ind];
    // }

    gpuStream_t &getSendStream() {
        sendStreamUsed = true;
        return sendStream;
    }

    gpuStream_t &getReceiveStream() {
        receiveStreamUsed = true;
        return receiveStream;
    }

  
    // int addDeviceStream() {
    //     deviceStream.emplace_back();

    //     int lastIndex = deviceStream.size() - 1;
    //     gpuError_t gpuErr = gpuStreamCreate(&deviceStream[lastIndex]);
    //     if (gpuErr != gpuSuccess) GpuError("haloOffsetInfo.h: gpuStreamCreate", gpuErr);

    //     streamUsed.emplace_back();
    //     streamUsed[lastIndex] = false;
    //     return lastIndex;
    // }

    void synchronizeAll() {
        synchronizeSendStream();
        synchronizeReceiveStream();
        synchronizeRequest();
    }

    void synchronizeSendStream() {
        if (sendStreamUsed) {
                gpuError_t gpuErr = gpuStreamSynchronize(sendStream);
                if (gpuErr != gpuSuccess) GpuError("haloOffsetInfo.h: gpuStreamSynchronize sendStream", gpuErr);
                sendStreamUsed = false;
        }
    }
    
     void synchronizeReceiveStream() {
        if (receiveStreamUsed) {
                gpuError_t gpuErr = gpuStreamSynchronize(receiveStream);
                if (gpuErr != gpuSuccess) GpuError("haloOffsetInfo.h: gpuStreamSynchronize receiveStream", gpuErr);
                receiveStreamUsed = false;
        }
    }
    
    

    void synchronizeRequest(int sendRecv = -1) {

        if(sendRecv == 0 || sendRecv == -1) {
            if (sendReqUsed) {
                MPI_Wait(&hostRequestSend, MPI_STATUS_IGNORE);
                sendReqUsed = false;
            }
        }

        if(sendRecv == 1 || sendRecv == -1) {
            if (recvReqUsed) {
                MPI_Wait(&hostRequestRecv, MPI_STATUS_IGNORE);
                recvReqUsed = false;
            }
        }
    }

    uint8_t *getHostSendPtr() {
        return &sendBase->template getPointer<uint8_t>()[offset];
    }

    uint8_t *getHostRecvPtr() {
        return &recvBase->template getPointer<uint8_t>()[offset];
    }

    uint8_t *getMyDeviceSourcePtr() {
        return sourceBase->template getPointer<uint8_t>() + offset;
    }

    uint8_t *getMyDeviceDestinationPtr() {
        return &destinationBase->template getPointer<uint8_t>()[reverseOffset];
    }

    uint8_t *getDeviceDestinationPtrP2P() {
        return destinationBase->getOppositeP2PPointer(oppositeP2PRank) + reverseOffset;
    }

    uint8_t *getDeviceDestinationPtrGPUAwareMPI() {
        return &destinationBase->template getPointer<uint8_t>()[offset];
    }

    ~HaloSegmentInfo() {

        // gpuError_t gpuErr = gpuStreamDestroy(sendStream);
        //     if (gpuErr != gpuSuccess) GpuError("haloOffsetInfo.h: gpuStreamDestroy sendStream", gpuErr);
        
        // gpuError_t gpuErr = gpuStreamDestroy(receiveStream);
        //     if (gpuErr != gpuSuccess) GpuError("haloOffsetInfo.h: gpuStreamDestroy recieveStream", gpuErr);

        if ((hostRequestSend != MPI_REQUEST_NULL) && (hostRequestSend != 0)) {
            MPI_Request_free(&hostRequestSend);
        }
        if ((hostRequestRecv != MPI_REQUEST_NULL) && (hostRequestRecv != 0)) {
            MPI_Request_free(&hostRequestRecv);
        }

        if (MpiType != MPI_DATATYPE_NULL) {
            MPI_Type_free(&MpiType);
        }
    }
};

template<bool onDevice>
class HaloOffsetInfo {
private:
    NeighborInfo &neighbor_info;
    MPI_Comm cart_comm;
    int _myRank;
    gMemoryPtr<false> sendBase;
    gMemoryPtr<false> recvBase;
    gMemoryPtr<true> sendBaseP2P;
    gMemoryPtr<true> recvBaseP2P;

    bool _gpuAwareMPI;
    bool _gpuP2P;

    std::vector<HaloSegmentInfo> _hypPlaneInfo;
    std::vector<HaloSegmentInfo> _planeInfo;
    std::vector<HaloSegmentInfo> _stripeInfo;
    std::vector<HaloSegmentInfo> _cornerInfo;
    
    commStreams_t& commStreams;
    
    gpuIPCEvent _cIPCEvent;


public:

    std::map<HaloSegment, HaloSegmentInfo *> _HalSegMapLeft;
    std::map<HaloSegment, HaloSegmentInfo *> _HalSegMapRight;

    

    HaloOffsetInfo() = delete;

    //! constructor
    HaloOffsetInfo(commStreams_t &cstreams, NeighborInfo &NInfo, MPI_Comm comm, int myRank, bool gpuAwareMPI = false, bool gpuP2P = false) :
            neighbor_info(NInfo),
            cart_comm(comm), _myRank(myRank),
            sendBase(MemoryManagement::getMemAt<false>("sendBase")),
            recvBase(MemoryManagement::getMemAt<false>("recvBase")),
            sendBaseP2P(MemoryManagement::getMemAt<true>("sendBaseP2P")),
            recvBaseP2P(MemoryManagement::getMemAt<true>("recvBaseP2P")),
            _gpuAwareMPI(gpuAwareMPI),
            _gpuP2P(gpuP2P),
            commStreams(cstreams),
            _cIPCEvent(cart_comm, _myRank) {

#ifdef COMMS_STREAMS       
        for (size_t i = 0; i < 2*MAX_NUM_HALOSEGMENTS; i+=2) {
            if (i < 16) {
                _hypPlaneInfo.push_back(HaloSegmentInfo(commStreams[i],commStreams[i+1]));
            }
            else if (i < 64) {
                _planeInfo.push_back(HaloSegmentInfo(commStreams[i],commStreams[i+1]));
            }
            else if (i < 128) {
                _stripeInfo.push_back(HaloSegmentInfo(commStreams[i],commStreams[i+1]));
            }
            else {
                _cornerInfo.push_back(HaloSegmentInfo(commStreams[i],commStreams[i+1]));
            }

        }
#else
        for (size_t i = 0; i < 2*MAX_NUM_HALOSEGMENTS; i+=2) {
            if (i < 16) {
                _hypPlaneInfo.push_back(HaloSegmentInfo(commStreams[0],commStreams[1]));
            }
            else if (i < 64) {
                _planeInfo.push_back(HaloSegmentInfo(commStreams[0],commStreams[1]));
            }
            else if (i < 128) {
                _stripeInfo.push_back(HaloSegmentInfo(commStreams[0],commStreams[1]));
            }
            else {
                _cornerInfo.push_back(HaloSegmentInfo(commStreams[0],commStreams[1]));
            }

        }
#endif
        for (auto &HypPlane : HaloHypPlanes) {
            _HalSegMapLeft[HypPlane] = _hypPlaneInfo.data();
            _HalSegMapRight[HypPlane] = _hypPlaneInfo.data();
        }

        for (auto &plane : HaloPlanes) {
            _HalSegMapLeft[plane] = _planeInfo.data();
            _HalSegMapRight[plane] = _planeInfo.data();
        }

        for (auto &stripe : HaloStripes) {
            _HalSegMapLeft[stripe] = _stripeInfo.data();
            _HalSegMapRight[stripe] = _stripeInfo.data();
        }

        for (auto &corner : HaloCorners) {
            _HalSegMapLeft[corner] = _cornerInfo.data();
            _HalSegMapRight[corner] = _cornerInfo.data();
        }
    }

    //! move constructor
    HaloOffsetInfo(HaloOffsetInfo<onDevice>&& source)  noexcept :
        neighbor_info(source.neighbor_info), //! this is a reference and doesn't need to be moved
        cart_comm(source.cart_comm),
        _myRank(source._myRank),
        //! move gMemoryPtr's:
        sendBase(std::move(source.sendBase)),
        recvBase(std::move(source.recvBase)),
        sendBaseP2P(std::move(source.sendBaseP2P)),
        recvBaseP2P(std::move(source.recvBaseP2P)),

        //! these are just bools
        _gpuAwareMPI(source._gpuAwareMPI),
        _gpuP2P(source._gpuP2P),

        //! move HaloSegmentInfo arrays
        _hypPlaneInfo(std::move(source._hypPlaneInfo)),
        _planeInfo(std::move(source._planeInfo)),
        _stripeInfo(std::move(source._stripeInfo)),
        _cornerInfo(std::move(source._cornerInfo)),
        commStreams(source.commStreams),
        _cIPCEvent(std::move(source._cIPCEvent))
    {
        //! reset other object somewhat
        source._myRank = -1;
        source.cart_comm = MPI_COMM_NULL;

        //! this is basically the normal constructor
        for (auto &HypPlane : HaloHypPlanes) {
            _HalSegMapLeft[HypPlane] = _hypPlaneInfo.data();
            _HalSegMapRight[HypPlane] = _hypPlaneInfo.data();
        }

        for (auto &plane : HaloPlanes) {
            _HalSegMapLeft[plane] = _planeInfo.data();
            _HalSegMapRight[plane] = _planeInfo.data();
        }

        for (auto &stripe : HaloStripes) {
            _HalSegMapLeft[stripe] = _stripeInfo.data();
            _HalSegMapRight[stripe] = _stripeInfo.data();
        }

        for (auto &corner : HaloCorners) {
            _HalSegMapLeft[corner] = _cornerInfo.data();
            _HalSegMapRight[corner] = _cornerInfo.data();
        }
    }


    ~HaloOffsetInfo() = default;

    void syncAllStreamRequests() {
        IF(COMBASE_DEBUG) (stdLogger.debug("Synchronize all");)
        for (const HaloSegment &hseg : AllHaloSegments) {
            for (int dir = 0; dir < HaloSegmentDirections(hseg); dir++) {
                for (int leftRight = 0; leftRight <= 1; leftRight++) {
                    size_t pos_l = getSegTypeOffset(hseg, dir);
                    size_t pos_r = getSegTypeOffset(hseg, dir) + 1;
                    _HalSegMapLeft[hseg][pos_l].synchronizeAll();
                    _HalSegMapRight[hseg][pos_r].synchronizeAll();
                }
            }
        }
    }

    int MyRank() { return _myRank; }

    void setSendBase(gMemoryPtr<false> ptr) {
        sendBase = ptr;
    }

    void setRecvBase(gMemoryPtr<false> ptr) {
        recvBase = ptr;

    }

    gMemoryPtr<false> getSendBase() {
        return sendBase;
    }

    gMemoryPtr<false> getRecvBase() {
        return recvBase;
    }

    void setSendBaseP2P(gMemoryPtr<true> ptr) {
        sendBaseP2P = ptr;
    }

    gMemoryPtr<true> getSendBaseP2P() {
        return sendBaseP2P;
    }

    void setRecvBaseP2P(gMemoryPtr<true> ptr) {
        recvBaseP2P = ptr;
    }

    void initP2P() {
        if (_gpuP2P && onDevice) {
            sendBaseP2P->initP2P(cart_comm, _myRank);
            recvBaseP2P->initP2P(cart_comm, _myRank);
            //      _cIPCEvent = gpuIPCEvent(cart_comm,_myRank);
        }
    }

    gMemoryPtr<true> getRecvBaseP2P() {
        return recvBaseP2P;
    }

    void setMemoryPointer(HaloSegmentInfo &segInfo, ProcessInfo &neighborInfo, int index, int oppositeIndex) {
        segInfo.setRecvBase(recvBase);
        segInfo.setSendBase(sendBase);

        if (onDevice) {
            segInfo.setSourceBase(sendBaseP2P);
            if (_gpuP2P || _gpuAwareMPI)segInfo.setDestinationBase(recvBaseP2P);
        }

        segInfo.setP2P(false);

        if (neighborInfo.p2p && (segInfo.getLength() != 0) && onDevice && _gpuP2P) {
            segInfo.setP2P(neighborInfo.p2p && _gpuP2P);

            segInfo.setOppositeP2PRank(neighborInfo.world_rank);
            if (!neighborInfo.sameRank) {
                sendBaseP2P->addP2PRank(neighborInfo.world_rank);
                recvBaseP2P->addP2PRank(neighborInfo.world_rank);
                _cIPCEvent.addP2PRank(index, oppositeIndex, neighborInfo.world_rank);
            }
        }
    }

    void syncAndInitP2PRanks() {
        if (onDevice && _gpuP2P) {
            sendBaseP2P->syncAndInitP2PRanks();
            recvBaseP2P->syncAndInitP2PRanks();
            _cIPCEvent.syncAndInitAllP2PRanks();
        }
    }

    void exchangeHandles() {
        for (const HaloSegment &hseg : AllHaloSegments) {
            for (int dir = 0; dir < HaloSegmentDirections(hseg); dir++) {
                for (int leftRight = 0; leftRight <= 1; leftRight++) {
                    ProcessInfo &nInfo = neighbor_info.getNeighborInfo(hseg, dir, leftRight);
                    HaloSegmentInfo &segInfo = get(hseg, dir, leftRight);
                    setMemoryPointer(segInfo, nInfo);
                }
            }
        }
    }

    NeighborInfo &getNeighborInfo() { return neighbor_info; }


    size_t getSegTypeOffset(HaloSegment halseg, size_t direction) {
        if (halseg < 4) return halseg * 2;
        if (halseg < 10) return ((size_t) halseg - 4) * 4 + direction * 2;
        if (halseg < 14) return ((size_t) halseg - 10) * 8 + direction * 2;
        return ((size_t) halseg - 14) * 16 + direction * 2;
    }

    void create(HaloSegment hseg, size_t direction, size_t off_l, size_t length_l, size_t off_r, size_t length_r) {
        size_t pos_l = getSegTypeOffset(hseg, direction);
        size_t pos_r = getSegTypeOffset(hseg, direction) + 1;

        _HalSegMapLeft[hseg][pos_l].setOffset(off_l);
        _HalSegMapRight[hseg][pos_r].setOffset(off_r);

        _HalSegMapLeft[hseg][pos_l].setReverseOffset(off_r);
        _HalSegMapRight[hseg][pos_r].setReverseOffset(off_l);

        _HalSegMapLeft[hseg][pos_l].setLength(length_l);
        _HalSegMapRight[hseg][pos_r].setLength(length_r);

        ProcessInfo &leftInfo = neighbor_info.getNeighborInfo(hseg, direction, 0);
        ProcessInfo &rightInfo = neighbor_info.getNeighborInfo(hseg, direction, 1);


        int indexLeft = haloSegmentCoordToIndex(hseg, direction, 0);
        int indexRight = haloSegmentCoordToIndex(hseg, direction, 1);
        setMemoryPointer(_HalSegMapLeft[hseg][pos_l], leftInfo, indexLeft, indexRight);
        setMemoryPointer(_HalSegMapRight[hseg][pos_r], rightInfo, indexRight, indexLeft);

        if (_HalSegMapLeft[hseg][pos_l].getLength() != 0) {
            MPI_Type_contiguous(length_l, MPI_BYTE, &_HalSegMapLeft[hseg][pos_l].getMpiType());
            MPI_Type_commit(&_HalSegMapLeft[hseg][pos_l].getMpiType());
        }
        if (_HalSegMapRight[hseg][pos_r].getLength() != 0) {
            MPI_Type_contiguous(length_r, MPI_BYTE, &_HalSegMapRight[hseg][pos_r].getMpiType());
            MPI_Type_commit(&_HalSegMapRight[hseg][pos_r].getMpiType());
        }
    }

    deviceEventPair &getGpuEventPair(HaloSegment hseg, size_t direction, bool leftRight) {
        int oppositeIndex = haloSegmentCoordToIndex(hseg, direction, !leftRight);
        int rank = neighbor_info.getNeighborInfo(hseg, direction, leftRight).world_rank;
        return _cIPCEvent.getOppositeEventPair(oppositeIndex, rank);
    }

    deviceEventPair &getMyGpuEventPair(HaloSegment hseg, size_t direction, bool leftRight) {
        int index = haloSegmentCoordToIndex(hseg, direction, leftRight);
        int rank = neighbor_info.getNeighborInfo(hseg, direction, leftRight).world_rank;
        return _cIPCEvent.getMyEventPair(index, rank);
    }

    HaloSegmentInfo &get(HaloSegment hseg, size_t direction, bool leftRight) {
        size_t pos_l = getSegTypeOffset(hseg, direction);
        size_t pos_r = getSegTypeOffset(hseg, direction) + 1;

        if (leftRight) {
            return _HalSegMapRight[hseg][pos_r];
        } else return _HalSegMapLeft[hseg][pos_l];
    }

};

