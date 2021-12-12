//
// Created by Lukas Mazur on 28.04.19.
//

#ifndef GPUIPC_H
#define GPUIPC_H

#include <mpi.h>
#include "../../define.h"
#include <map>
#include <vector>
#include "deviceEvent.h"
#include <list>


struct gpuIpcInfo {
    gpuIpcMemHandle_t handle;
    gpuIpcEventHandle_t eventHandle;
};

struct p2pNeighbor {
    int oppositeRank;
    bool RanksExchanged;
};

class gpuIPC {

private:
    MPI_Comm _cart_comm;
    int _myRank;

    std::vector<p2pNeighbor> _oppositeInfo;
    std::vector<uint8_t *> _oppositeMemoryPtr;
    std::vector<gpuEvent_t> _oppositeEvent;
    uint8_t *_myMemory;
    gpuIpcInfo _myHandle;

    gpuEvent_t _myEvent;


    void initMyHandle() {
        if (!initialized) {
            gpuError_t gpuErr;
            gpuErr = gpuIpcGetMemHandle((gpuIpcMemHandle_t *) &_myHandle.handle, _myMemory);
            if (gpuErr != gpuSuccess) GpuError("gpuIPC.h: exchangeHandle: gpuIpcGetMemHandle", gpuErr);

            gpuErr = gpuEventCreate(&_myEvent, gpuEventDisableTiming | gpuEventInterprocess);
            if (gpuErr != gpuSuccess) GpuError("gpuIPC.h: exchangeHandle: gpuEventCreate", gpuErr);

            gpuErr = gpuIpcGetEventHandle((gpuIpcEventHandle_t *) &_myHandle.eventHandle, _myEvent);
            if (gpuErr != gpuSuccess) GpuError("gpuIPC.h: exchangeHandle: gpuIpcGetEventHandle", gpuErr);

            gpuErr = gpuEventSynchronize(_myEvent);
            if (gpuErr != gpuSuccess) GpuError("gpuIPC.h: exchangeHandle: gpuEventSynchronize(_myEvent)", gpuErr);

            initialized = true;
        }
    }


    bool initialized;

public:
    gpuIPC() : initialized(false) {}


    gpuIPC(MPI_Comm cart_comm, int myRank, uint8_t *myMemory) : _cart_comm(cart_comm), _myRank(myRank),
                                                                 _myMemory(myMemory), initialized(false) {
        initMyHandle();

    }

    ~gpuIPC() {
        destroy();
    }

    void destroy() {
        if (initialized) {

            gpuError_t gpuErr;

            for (unsigned int i = 0; i < _oppositeInfo.size(); i++) {
                gpuErr = gpuIpcCloseMemHandle(_oppositeMemoryPtr[i]);
                if (gpuErr != gpuSuccess) {
                    GpuError("gpuIPC.h: destroy: gpuIpcCloseMemHandle", gpuErr);
                    stdLogger.error("Rank = " ,  i);
                }
                gpuErr = gpuEventDestroy(_oppositeEvent[i]);
                if (gpuErr != gpuSuccess) {
                    GpuError("gpuIPC.h: destroy: gpuEventDestroy", gpuErr);
                    stdLogger.error("Rank = " ,  i);
                }
            }
            initialized = false;
        }
    }

    void addP2PRank(int oppositeRank) {
        if (!initialized) initMyHandle();

        for (unsigned int i = 0; i < _oppositeInfo.size(); i++) {
            if (_oppositeInfo[i].oppositeRank == oppositeRank) return;
        }
        _oppositeInfo.emplace_back();
        int lastIndex = _oppositeInfo.size() - 1;
        _oppositeInfo[lastIndex].oppositeRank = oppositeRank;
        _oppositeInfo[lastIndex].RanksExchanged = false;
    }

    void syncAndInitAllP2PRanks() {
        if (!initialized) initMyHandle();

        std::vector<gpuIpcInfo> oppositeHandles;
        std::vector<MPI_Request> sendReq;
        std::vector<MPI_Request> recvReq;
        for (unsigned int i = 0; i < _oppositeInfo.size(); i++) {
            if (!_oppositeInfo[i].RanksExchanged) {
                sendReq.emplace_back();
                recvReq.emplace_back();
                oppositeHandles.emplace_back();
            }
        }
        for (unsigned int i = 0; i < _oppositeInfo.size(); i++) {
            if (!_oppositeInfo[i].RanksExchanged) {
                MPI_Isend((uint8_t *) &_myHandle, sizeof(gpuIpcInfo), MPI_UINT8_T, _oppositeInfo[i].oppositeRank,
                          _myRank, _cart_comm, &sendReq[i]);
                MPI_Irecv((uint8_t *) &oppositeHandles[i], sizeof(gpuIpcInfo), MPI_UINT8_T,
                          _oppositeInfo[i].oppositeRank, _oppositeInfo[i].oppositeRank, _cart_comm, &recvReq[i]);
            }
        }

        for (unsigned int i = 0; i < _oppositeInfo.size(); i++) {
            if (!_oppositeInfo[i].RanksExchanged) {


                int ret;
                ret = MPI_Waitall(_oppositeInfo.size(), &recvReq[0], MPI_STATUSES_IGNORE);
                if (ret != MPI_SUCCESS) {
                    stdLogger.error("gpuIPC.h: MPI_Wait(recv) failed");
                }
                ret = MPI_Waitall(_oppositeInfo.size(), &sendReq[0], MPI_STATUSES_IGNORE);
                if (ret != MPI_SUCCESS) {
                    stdLogger.error("gpuIPC.h: MPI_Wait(send) failed");
                }


                gpuError_t gpuErr;
                gpuEvent_t oppositeEvent;

                gpuErr = gpuEventSynchronize(_myEvent);
                if (gpuErr != gpuSuccess)
                    GpuError("gpuIPC.h: exchangeHandle: gpuEventSynchronize(_myEvent)", gpuErr);


                gpuErr = gpuIpcOpenEventHandle(&oppositeEvent,
                                                 *(gpuIpcEventHandle_t *) &oppositeHandles[i].eventHandle);
                if (gpuErr != gpuSuccess) GpuError("gpuIPC.h: exchangeHandle: gpuIpcOpenEventHandle", gpuErr);

                _oppositeEvent.emplace_back();
                _oppositeEvent[i] = oppositeEvent;

                gpuErr = gpuEventSynchronize(_oppositeEvent[i]);
                if (gpuErr != gpuSuccess)
                    GpuError("gpuIPC.h: exchangeHandle: gpuEventSynchronize(_oppositeEvent)", gpuErr);

                _oppositeMemoryPtr.emplace_back();
                gpuErr = gpuIpcOpenMemHandle((void **) &_oppositeMemoryPtr[i],
                                               *(gpuIpcMemHandle_t *) &oppositeHandles[i].handle,
                                               gpuIpcMemLazyEnablePeerAccess);
                if (gpuErr != gpuSuccess) GpuError("gpuIPC.h: exchangeHandle: gpuIpcOpenMemHandle", gpuErr);


                _oppositeInfo[i].RanksExchanged = true;
            }
        }

    }

    uint8_t *getPointer(int oppositeRank) {
        for (unsigned int i = 0; i < _oppositeInfo.size(); i++) {
            if (_oppositeInfo[i].oppositeRank == oppositeRank) {

                /* gpuError_t gpuErr;
                 gpuErr = gpuEventSynchronize(_oppositeEvent[i]);
                 if (gpuErr != gpuSuccess)
                     GpuError("gpuIPC.h: getPointer: gpuEventSynchronize(_oppositeEvent)", gpuErr);*/
                return _oppositeMemoryPtr[i];
            }
        }
        rootLogger.error("gpuIPC.h: getPointer: Rank is not listed!");
        return nullptr;
    }

    void updateAllHandles(uint8_t *myMemory) {
        destroy();
        _myMemory = myMemory;
        initMyHandle();
        std::vector<int> ranks;
        for (unsigned int i = 0; i < _oppositeInfo.size(); i++) {
            ranks.push_back(_oppositeInfo[i].oppositeRank);
        }
        _oppositeInfo.clear();
        _oppositeMemoryPtr.clear();
        _oppositeEvent.clear();


        for (unsigned int i = 0; i < ranks.size(); i++) {
            addP2PRank(ranks[i]);
        }
        syncAndInitAllP2PRanks();
    }


};

struct p2pNeighborEvent : public p2pNeighbor {
    int index;
    int oppositeIndex;
};

struct deviceEventHandlePair {

    gpuIpcEventHandle_t start;
    gpuIpcEventHandle_t stop;

};

struct deviceEventPair {
    deviceEventPair(deviceEvent &&start, deviceEvent &&stop) : start(std::move(start)), stop(std::move(stop)) { }
    deviceEventPair() : start(3), stop(3) { }

    //! move constructor
    deviceEventPair(deviceEventPair &&obj) noexcept :
      start(std::move(obj.start)),
      stop(std::move(obj.stop)){}

   deviceEventPair& operator=(deviceEventPair&& obj) {
        start = std::move(obj.start);
        stop = std::move(obj.stop);
        return *this;
    }

    deviceEvent start;
    deviceEvent stop;
};

class gpuIPCEvent {

private:
    MPI_Comm _cart_comm;
    bool initialized;
    int _myRank;

    std::vector<p2pNeighborEvent> _oppositeInfo;
    std::vector<deviceEventPair> _oppositeEvents;
    std::vector<deviceEventPair> _myEvents;
    //deviceEventPair _oppositeEvent[80];

    std::vector<deviceEventHandlePair> _myHandles;
   // deviceEventPair _myEvent;


    void addMyEvent() {

        if (initialized) {
            _myEvents.emplace_back(deviceEvent(2), deviceEvent(2));
            deviceEventPair &_myEvent = _myEvents[_myEvents.size()-1];
            _myHandles.emplace_back();
            deviceEventHandlePair &_myHandle = _myHandles[_myEvents.size()-1];

            _myEvent.start.synchronize();
            gpuError_t gpuErr = gpuIpcGetEventHandle((gpuIpcEventHandle_t *) &_myHandle.start, _myEvent.start._event);
            if (gpuErr != gpuSuccess) GpuError("gpuIPC.h: gpuIPCEvent(..): gpuIpcGetEventHandle(start)", gpuErr);

            _myEvent.start.synchronize();
            _myEvent.stop.synchronize();
            gpuErr = gpuIpcGetEventHandle((gpuIpcEventHandle_t *) &_myHandle.stop, _myEvent.stop._event);
            if (gpuErr != gpuSuccess) GpuError("gpuIPC.h: gpuIPCEvent(..): gpuIpcGetEventHandle(stop)", gpuErr);

            _myEvent.stop.synchronize();
        }
    }

public:
    //! constructor
    gpuIPCEvent() : initialized(false) {}
    //! construcotr
    gpuIPCEvent(const gpuIPCEvent &obj) : _cart_comm(obj._cart_comm), initialized(obj.initialized),
                                            _myRank(obj._myRank) {}
    //! constructor
    gpuIPCEvent(MPI_Comm cart_comm, int myRank) : _cart_comm(cart_comm), initialized(true), _myRank(myRank){}
    //! copy assignment
    gpuIPCEvent& operator=(const gpuIPCEvent &obj) {
        _cart_comm = obj._cart_comm;
        initialized = obj.initialized;
        _myRank = obj._myRank;
        return *this;
    }
    //! copy constructor
    gpuIPCEvent(gpuIPCEvent&) = delete;
    //! move assignment
    gpuIPCEvent& operator=(gpuIPCEvent&&) = delete;
    //! move constructor
    gpuIPCEvent(gpuIPCEvent&& source) noexcept :
    //! simple types
    _cart_comm(source._cart_comm),
    initialized(source.initialized),
    _myRank(source._myRank),
    //! these are all vectors
    _oppositeInfo(std::move(source._oppositeInfo)), //! vector of simple structs
    _oppositeEvents(std::move(source._oppositeEvents)),
    _myEvents(std::move(source._myEvents)),
    _myHandles(std::move(source._myHandles))
    {
        source._cart_comm = MPI_COMM_NULL;
        source.initialized = false;
        source._myRank = -1;
    }

    //! destructor
    ~gpuIPCEvent() {
        initialized = false;
    }


    void addP2PRank(int index, int oppositeIndex, int oppositeRank) {
        if (!initialized) rootLogger.error("gpuIPC.h: gpuIPCEvent.addP2PRank: Event not initialized");

    //    rootLogger.info("Set index: " ,  index ,  " with rank: " ,  oppositeRank);
        _oppositeInfo.emplace_back();
        int lastIndex = _oppositeInfo.size() - 1;
        _oppositeInfo[lastIndex].oppositeRank = oppositeRank;
        _oppositeInfo[lastIndex].RanksExchanged = false;
        _oppositeInfo[lastIndex].index = index;
        _oppositeInfo[lastIndex].oppositeIndex = oppositeIndex;
        addMyEvent();
    }

    void syncAndInitAllP2PRanks() {
        if (!initialized) rootLogger.error("gpuIPC.h: gpuIPCEvent.syncAndInitAllP2PRanks: Event not initialized");

        std::vector<deviceEventHandlePair> oppositeHandles;
        std::vector<MPI_Request> sendReq;
        std::vector<MPI_Request> recvReq;
        for (unsigned int i = 0; i < _oppositeInfo.size(); i++) {
            if (!_oppositeInfo[i].RanksExchanged) {
                sendReq.emplace_back();
                recvReq.emplace_back();
                oppositeHandles.emplace_back();
            }
        }
        for (unsigned int i = 0; i < _oppositeInfo.size(); i++) {
            if (!_oppositeInfo[i].RanksExchanged) {
             //   int index = _oppositeInfo[i].index;
            //    int oppositeIndex = _oppositeInfo[i].oppositeIndex;

                MPI_Isend((uint8_t *) &_myHandles[i], sizeof(deviceEventHandlePair), MPI_UINT8_T,
                          _oppositeInfo[i].oppositeRank, _oppositeInfo[i].oppositeIndex, _cart_comm, &sendReq[i]);
                MPI_Irecv((uint8_t *) &oppositeHandles[i], sizeof(deviceEventHandlePair), MPI_UINT8_T,
                          _oppositeInfo[i].oppositeRank, _oppositeInfo[i].index, _cart_comm, &recvReq[i]);
            }
        }

        int ret;
        ret = MPI_Waitall(_oppositeInfo.size(), &recvReq[0], MPI_STATUSES_IGNORE);
        if (ret != MPI_SUCCESS) {
            stdLogger.error("gpuIPC.h: MPI_Wait(recv) failed");
        }
        ret = MPI_Waitall(_oppositeInfo.size(), &sendReq[0], MPI_STATUSES_IGNORE);
        if (ret != MPI_SUCCESS) {
            stdLogger.error("gpuIPC.h: MPI_Wait(send) failed");
        }

        for (unsigned int i = 0; i < _oppositeInfo.size(); i++) {
            if (!_oppositeInfo[i].RanksExchanged) {


                gpuError_t gpuErr;

                _myEvents[i].start.synchronize();
                _myEvents[i].stop.synchronize();

                _oppositeEvents.emplace_back(deviceEvent(3), deviceEvent(3));

                //deviceEvent start(3);
                //deviceEvent stop(3);
                gpuErr = gpuIpcOpenEventHandle(&(_oppositeEvents[i].start._event),
                                                 *(gpuIpcEventHandle_t *) &oppositeHandles[i].start);
                if (gpuErr != gpuSuccess)
                    GpuError("gpuIPC.h: gpuIPCEvent.exchangeHandle: gpuIpcOpenEventHandle 1", gpuErr);



                gpuErr = gpuIpcOpenEventHandle(&(_oppositeEvents[i].stop._event),
                                                 *(gpuIpcEventHandle_t *) &oppositeHandles[i].stop);
                if (gpuErr != gpuSuccess)
                    GpuError("gpuIPC.h: gpuIPCEvent.exchangeHandle: gpuIpcOpenEventHandle 2", gpuErr);

               // _oppositeEvent[i] = deviceEventPair(std::move(start),std::move(stop));

                _oppositeEvents[i].start._mode =2;
                _oppositeEvents[i].start.synchronize();

                _oppositeEvents[i].stop._mode =2 ;
                _oppositeEvents[i].stop.synchronize();
                _oppositeInfo[i].RanksExchanged = true;
            }
        }

    }

    deviceEventPair &getOppositeEventPair(int oppositeIndex, int rank) {
        if (!initialized) rootLogger.error("gpuIPC.h: gpuIPCEvent.getOppositeEventPair: Event not initialized");
        for (unsigned int i = 0; i < _oppositeInfo.size(); i++) {
            if (_oppositeInfo[i].oppositeIndex == oppositeIndex) {

                if(_oppositeInfo[i].oppositeRank != rank) rootLogger.error("gpuIPCEvent: getOppositeEventPair: Rank is not the same!!");
                return _oppositeEvents[i];
            }
        }
        rootLogger.error("gpuIPC.h: getPointer: Rank is not listed!");
        return _myEvents[0];
    }

    deviceEventPair &getMyEventPair(int index, int rank) {
        if (!initialized) rootLogger.error("gpuIPC.h: gpuIPCEvent.getMyEventPair: Event not initialized");
        for (unsigned int i = 0; i < _oppositeInfo.size(); i++) {
            if (_oppositeInfo[i].index == index) {

                if(_oppositeInfo[i].oppositeRank != rank) rootLogger.error("gpuIPCEvent: getMyEventPair: Rank is not the same!!");
                return _myEvents[i];
            }
        }
        rootLogger.error("gpuIPC.h: getPointer: Rank is not listed!");
        return _myEvents[0];
    }


};

#endif //GPUIPC_H
