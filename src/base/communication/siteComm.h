/*
 * siteComm.h
 *
 * L. Mazur
 *
 * This class glues together all other classes needed to exchange halos. Again loosely speaking,
 * to exchange some information for my stencil calculation between a sublattice and his neighbor,
 * I need to
 *
 *     1. allocate some dynamic memory for the buffer (MemoryManagement);
 *     2. translate the local halo coordinates to buffer coordinates to be able to copy
 *        sites into the buffer (haloOffsetInfo); and
 *     3. communicate these buffers to my neighbors (I know who they are because of neighborInfo)
 *        using the wrappers in the CommunicationBase.
 *
 */

#pragma once

#include "../../define.h"
#include <iostream>
#include "communicationBase.h"
#include <memory>
#include "../stopWatch.h"
#include <unordered_set>
#include "../runFunctors.h"
#ifndef USE_HIP_AMD
    #include "nvToolsExt.h"
#endif
#include "deviceEvent.h"
#include "deviceStream.h"


#include "../indexer/HaloIndexer.h"

#include "calcGSiteHalo_dynamic.h"


template<bool onDevice, class floatT, class Accessor, size_t ElemCount, size_t EntryCount, Layout LatLayout, size_t HaloDepth>
class HaloSegmentConfig {
    int N;

    typedef HaloIndexer<LatLayout, HaloDepth> HInd;
    HaloType _haloType;
    HaloSegment _haloSeg;
    int _subIndex;
    int _dir;
    int _leftRight;

    HaloSegmentInfo& _segmentInfo;
    NeighborInfo& _NInfo;
    ProcessInfo& _PInfo;
    int _index;
    int _size;
    int _length;
    Accessor _hal_acc;

    public:
    HaloSegmentConfig(int N, HaloOffsetInfo<onDevice> &HalInfo, GCOMPLEX(floatT) *haloBuffer) : 
        N(N),
        _segmentInfo(HalInfo.get(HSegSelector(N).haloSeg(), HSegSelector(N).dir(), HSegSelector(N).leftRight())),
        _NInfo(HalInfo.getNeighborInfo()),
        _PInfo(HalInfo.getNeighborInfo().getNeighborInfo(HSegSelector(N).haloSeg(), HSegSelector(N).dir(), HSegSelector(N).leftRight())),
        _index(haloSegmentCoordToIndex(HSegSelector(N).haloSeg(), HSegSelector(N).dir(), HSegSelector(N).leftRight())),
        _size(HInd::get_SubHaloSize(haloSegmentCoordToIndex(HSegSelector(N).haloSeg(), HSegSelector(N).dir(), HSegSelector(N).leftRight())) * ElemCount),
        _length(HInd::get_SubHaloSize(haloSegmentCoordToIndex(HSegSelector(N).haloSeg(), HSegSelector(N).dir(), HSegSelector(N).leftRight()))),
        _hal_acc(Accessor(haloBuffer + HInd::get_SubHaloOffset(N) * EntryCount * ElemCount,  HInd::get_SubHaloSize(haloSegmentCoordToIndex(HSegSelector(N).haloSeg(), HSegSelector(N).dir(), HSegSelector(N).leftRight())) * ElemCount)) {
        }

    HaloType haloType() { return _haloType; }
    HaloSegment haloSeg() { return _haloSeg; }
    int subIndex() { return _subIndex; }
    int dir() { return _dir; }
    int leftRight() { return _leftRight; } 

    HaloSegmentInfo& segmentInfo() { return _segmentInfo; }
    NeighborInfo& NInfo() {return _NInfo; }
    ProcessInfo& PInfo() {return _PInfo; }
    int index() {return _index; }
    int length() {return _length; }
    int size() { return _size; }

    Accessor hal_acc() {return _hal_acc; }
};




template<class floatT, bool onDevice, class Accessor, class AccType, size_t EntryCount, size_t ElemCount, Layout LatLayout, size_t HaloDepth>
class siteComm : public RunFunctors<onDevice, Accessor> {
private:
    CommunicationBase &_commBase;

    typedef GIndexer<LatLayout, HaloDepth> GInd;
    typedef HaloIndexer<LatLayout, HaloDepth> HInd;

    size_t _elems;
    int _halElementSize;
    size_t _bufferLength;
    size_t _bufferSize;

    gMemoryPtr<false> _haloBuffer_Host = MemoryManagement::getMemAt<false>("SHARED_HaloAndReductionA");
    gMemoryPtr<false> _haloBuffer_Host_recv = MemoryManagement::getMemAt<false>("SHARED_HaloAndReductionD");
    gMemoryPtr<true> _haloBuffer_Device = MemoryManagement::getMemAt<true>("SHARED_HaloAndReductionA");
    gMemoryPtr<true> _haloBuffer_Device_recv = MemoryManagement::getMemAt<true>("SHARED_HaloAndReductionB");

    HaloOffsetInfo<onDevice> HaloInfo;

    void _injectHalos(Accessor lattice, GCOMPLEX(floatT) *HaloBuffer);

    void _extractHalos(Accessor lattice, GCOMPLEX(floatT) *HaloBuffer);

    void _extractHalosSeg(Accessor acc, GCOMPLEX(floatT) *HaloBuffer, unsigned int param);

    void _injectHalosSeg(Accessor acc, GCOMPLEX(floatT) *HaloBuffer, unsigned int param);
    

    std::vector<HaloSegmentConfig<onDevice, floatT, Accessor, ElemCount, EntryCount, LatLayout, HaloDepth>> HSegConfig_send_vec;
    std::vector<HaloSegmentConfig<onDevice, floatT, Accessor, ElemCount, EntryCount, LatLayout, HaloDepth>> HSegConfig_recv_vec;

public:
    //! constructor
    explicit siteComm(CommunicationBase &commB) :
            _commBase(commB),
            HaloInfo(HaloOffsetInfo<onDevice>(commB.getNeighborInfo(), _commBase.getCart_comm(), _commBase.MyRank(),
                                              _commBase.gpuAwareMPIAvail(), _commBase.useGpuP2P())) {

        _elems = HInd::getHalData().getBufferSize(LatLayout);

        _halElementSize = (int) ElemCount * sizeof(GCOMPLEX(floatT)) * EntryCount;
        _bufferLength = ElemCount * _elems;
        _bufferSize = sizeof(AccType) * _bufferLength;


        _haloBuffer_Host->adjustSize(_bufferSize);
        _haloBuffer_Host_recv->adjustSize(_bufferSize);

        if (onDevice) {
            _haloBuffer_Device->adjustSize(_bufferSize);

            if (commB.gpuAwareMPIAvail() || commB.useGpuP2P()) {
                _haloBuffer_Device_recv->adjustSize(_bufferSize);
            }
        }

        /// Set the base pointer of our buffer.
        HaloInfo.setSendBase(_haloBuffer_Host);
        HaloInfo.setRecvBase(_haloBuffer_Host_recv);

        if (onDevice) {
            HaloInfo.setSendBaseP2P(_haloBuffer_Device);
            if (commB.gpuAwareMPIAvail() || commB.useGpuP2P()) {
                HaloInfo.setRecvBaseP2P(_haloBuffer_Device_recv);
            }
        }

        HaloData haldat = HInd::getHalData();

        if (commB.getNumberProcesses() > 1 && onDevice) HaloInfo.initP2P();

        for (const auto &hypPlane : HaloHypPlanes) {
            size_t pos_l = 0, pos_r = 0;
            size_t off_l = 0, off_r = 0;
            HInd::getHypPlanePos(hypPlane, pos_l, pos_r);
            for (size_t k = 0; k < pos_l; ++k)off_l += haldat.get_SubHaloSize(k, LatLayout) * _halElementSize;
            for (size_t k = 0; k < pos_r; ++k)off_r += haldat.get_SubHaloSize(k, LatLayout) * _halElementSize;

            HaloInfo.create((HaloSegment) hypPlane, 0,
                            off_l, haldat.get_SubHaloSize(pos_l, LatLayout) * _halElementSize,
                            off_r, haldat.get_SubHaloSize(pos_r, LatLayout) * _halElementSize);
        }

        for (const auto &plane : HaloPlanes) {
            for (size_t dir = 0; dir < 2; dir++) {

                size_t pos_l = 0, pos_r = 0;
                size_t off_l = 0, off_r = 0;
                HInd::getPlanePos(plane, dir, pos_l, pos_r);
                for (size_t k = 0; k < pos_l; ++k)off_l += haldat.get_SubHaloSize(k, LatLayout) * _halElementSize;
                for (size_t k = 0; k < pos_r; ++k)off_r += haldat.get_SubHaloSize(k, LatLayout) * _halElementSize;


                HaloInfo.create((HaloSegment) (plane), dir,
                                off_l, haldat.get_SubHaloSize(pos_l, LatLayout) * _halElementSize,
                                off_r, haldat.get_SubHaloSize(pos_r, LatLayout) * _halElementSize);
            }
        }

        for (const auto &stripe : HaloStripes) {
            for (size_t dir = 0; dir < 4; dir++) {

                size_t pos_l = 0, pos_r = 0;
                size_t off_l = 0, off_r = 0;
                HInd::getStripePos(stripe, dir, pos_l, pos_r);
                for (size_t k = 0; k < pos_l; ++k)off_l += haldat.get_SubHaloSize(k, LatLayout) * _halElementSize;
                for (size_t k = 0; k < pos_r; ++k)off_r += haldat.get_SubHaloSize(k, LatLayout) * _halElementSize;

                HaloInfo.create((HaloSegment) (stripe), dir,
                                off_l, haldat.get_SubHaloSize(pos_l, LatLayout) * _halElementSize,
                                off_r, haldat.get_SubHaloSize(pos_r, LatLayout) * _halElementSize);
            }
        }

        for (const auto &corner : HaloCorners) {
            for (size_t dir = 0; dir < 8; dir++) {

                size_t pos_l = 0, pos_r = 0;
                size_t off_l = 0, off_r = 0;
                HInd::getCornerPos(corner, dir, pos_l, pos_r);
                for (size_t k = 0; k < pos_l; ++k)off_l += haldat.get_SubHaloSize(k, LatLayout) * _halElementSize;
                for (size_t k = 0; k < pos_r; ++k)off_r += haldat.get_SubHaloSize(k, LatLayout) * _halElementSize;

                HaloInfo.create((HaloSegment) (corner), dir,
                                off_l, haldat.get_SubHaloSize(pos_l, LatLayout) * _halElementSize,
                                off_r, haldat.get_SubHaloSize(pos_r, LatLayout) * _halElementSize);
            }
        }
        if (commB.getNumberProcesses() > 1 && onDevice) HaloInfo.syncAndInitP2PRanks();

#ifndef CPUONLY
        gpuError_t gpuErr = gpuDeviceSynchronize();
        if (gpuErr != gpuSuccess) {
            GpuError("siteComm.h: siteComm constructor, gpuDeviceSynchronize failed:", gpuErr);
        }
#endif
        
        if (onDevice) {
            if (commB.gpuAwareMPIAvail() || commB.useGpuP2P()) {
                for(int N = 0; N < 80; N++){
                    using HaloSegmentConfig_def = HaloSegmentConfig<onDevice, floatT, Accessor, ElemCount, EntryCount, LatLayout, HaloDepth>;
                    
                    HaloSegmentConfig_def HSegConfig_send = HaloSegmentConfig_def(N,HaloInfo,_haloBuffer_Device->template getPointer<GCOMPLEX(floatT) >());
                    HaloSegmentConfig_def HSegConfig_recv = HaloSegmentConfig_def(N,HaloInfo,_haloBuffer_Device_recv->template getPointer<GCOMPLEX(floatT) >());

                    if(HSegConfig_send.size() != 0){
                        HSegConfig_send_vec.push_back(HSegConfig_send);
                            
                    }
                    if(HSegConfig_recv.size() != 0){
                        HSegConfig_recv_vec.push_back(HSegConfig_recv);
                    }
                }
            }
        }
        commB.globalBarrier();
    }

    //! copy constructor
    siteComm(siteComm<floatT,onDevice,Accessor,AccType,EntryCount,ElemCount,LatLayout,HaloDepth>&) = delete;

    //! copy assignment
    siteComm<floatT,onDevice,Accessor,AccType,EntryCount,ElemCount,LatLayout,HaloDepth>&
            operator=(siteComm<floatT,onDevice,Accessor,AccType,EntryCount,ElemCount,LatLayout,HaloDepth>&) = delete;

    //! move assignment
    siteComm<floatT,onDevice,Accessor,AccType,EntryCount,ElemCount,LatLayout,HaloDepth>&
            operator=(siteComm<floatT,onDevice,Accessor,AccType,EntryCount,ElemCount,LatLayout,HaloDepth>&&) = delete;

    //! move constructor
    siteComm(siteComm<floatT,onDevice,Accessor,AccType,EntryCount,ElemCount,LatLayout,HaloDepth>&& source) noexcept :
    _commBase(source._commBase), //! this is a reference and shouldn't be moved
    _elems(source._elems),
    _halElementSize(source._halElementSize),
    _bufferLength(source._bufferLength),
    _bufferSize(source._bufferSize),

    //! move the gMemoryPtr's
    _haloBuffer_Host(std::move(source._haloBuffer_Host)),
    _haloBuffer_Host_recv(std::move(source._haloBuffer_Host_recv)),
    _haloBuffer_Device(std::move(source._haloBuffer_Device)),
    _haloBuffer_Device_recv(std::move(source._haloBuffer_Device_recv)),

    //! move the HaloInfo
    HaloInfo(std::move(source.HaloInfo))
    {
        source._elems = 0;
        source._halElementSize = 0;
        source._bufferLength = 0;
        source._bufferSize = 0;
    }

    virtual ~siteComm() {
    }

    virtual Accessor getAccessor() const = 0;

    CommunicationBase &getComm() const { return _commBase; }


    void updateAll(unsigned int param = AllTypes | COMM_BOTH) {
        gpuError_t gpuErr;

        /// A check that we don't have multiGPU and halosize=0:
        if (_commBase.getNumberProcesses() != 1 && HaloDepth == 0) {
            throw std::runtime_error(stdLogger.fatal("Useless call of CommunicationBase.updateAll() with multiGPU and HaloDepth=0!"));
        }
        if (HaloDepth == 0)
        {
            return;
        }

        unsigned int haltype = param & 15;
        unsigned int commtype = param & COMM_BOTH;
        if(haltype == 0) {
            haltype = AllTypes;
            param = param | AllTypes;
        }
        if(commtype == 0) {
            commtype = COMM_BOTH;
            param = param | COMM_BOTH;
        }

        if (commtype & COMM_START) {

            if (onDevice) {
                if (!(_commBase.gpuAwareMPIAvail() || _commBase.useGpuP2P())) {
                    _extractHalos(getAccessor(), _haloBuffer_Device->template getPointer<GCOMPLEX(floatT) >());
                    gpuErr = gpuMemcpy(_haloBuffer_Host->template getPointer<GCOMPLEX(floatT) >(),
                                         _haloBuffer_Device->template getPointer<GCOMPLEX(floatT) >(), _bufferSize,
                                         gpuMemcpyDeviceToHost);
                    if (gpuErr) {
                        GpuError("_haloBuffer_Device: Failed to copy to host", gpuErr);
                    }
                    _commBase.updateAll<onDevice>(HaloInfo, COMM_START | haltype);
                } else {
                    _extractHalosSeg(getAccessor(), _haloBuffer_Device->template getPointer<GCOMPLEX(floatT) >(),
                                     param);
                }
            } else {
                _extractHalos(getAccessor(), _haloBuffer_Host->template getPointer<GCOMPLEX(floatT) >());
                _commBase.updateAll<onDevice>(HaloInfo, COMM_START | haltype);
            }
        }

        if (commtype & COMM_FINISH) {


            if (onDevice) {
                if (_commBase.gpuAwareMPIAvail() || _commBase.useGpuP2P()) {
                    _injectHalosSeg(getAccessor(), _haloBuffer_Device_recv->template getPointer<GCOMPLEX(floatT) >(),
                                    param);
                } else {
		            HaloInfo.syncAllStreamRequests();
                    gpuErr = gpuMemcpy(_haloBuffer_Device->template getPointer<GCOMPLEX(floatT) >(),
                                         _haloBuffer_Host_recv->template getPointer<GCOMPLEX(floatT) >(), _bufferSize,
                                         gpuMemcpyHostToDevice);
                    if (gpuErr)
                        GpuError("_haloBuffer_Device: Failed to copy to device", gpuErr);
                    _injectHalos(getAccessor(), _haloBuffer_Device->template getPointer<GCOMPLEX(floatT) >());
                }
            } else {
                _commBase.updateAll<onDevice>(HaloInfo, COMM_FINISH | haltype);
                _injectHalos(getAccessor(), _haloBuffer_Host_recv->template getPointer<GCOMPLEX(floatT) >());
            }
        }
    }
};


template<class floatT, class Accessor, class AccType, size_t EntryCount, size_t ElemCount, Layout LatLayout, size_t HaloDepth>
struct ExtractInnerHalo {

    Accessor _acc;
    GCOMPLEX(floatT) *pointer[80];
    size_t size[80];
    typedef GIndexer<LatLayout, HaloDepth> GInd;
    typedef HaloIndexer<LatLayout, HaloDepth> HInd;

    ExtractInnerHalo(Accessor acc, GCOMPLEX(floatT) *HaloBuffer) :
            _acc(acc) {

        for (int i = 0; i < 80; ++i) {
            pointer[i] = HaloBuffer + HInd::get_SubHaloOffset(i) * EntryCount * ElemCount;
            size[i] = HInd::get_SubHaloSize(i) * ElemCount;

        }
    }

    inline __host__ __device__ void operator()(HaloSite site) {

        Accessor _hal_acc(pointer[site.HalNumber], size[site.HalNumber]);

        for (size_t mu = 0; mu < ElemCount; mu++) {
            size_t index = _acc.template getIndexComm<LatLayout, HaloDepth>(site.LatticeIndex, mu);
            _hal_acc.setEntriesComm(_acc, site.LocHalIndex * ElemCount + mu, index);
        }
    }
};

template<class floatT, bool onDevice, class Accessor, class AccType, size_t EntryCount, size_t ElemCount, Layout LatLayout, size_t HaloDepth>
void siteComm<floatT, onDevice, Accessor, AccType, EntryCount, ElemCount, LatLayout, HaloDepth>::_extractHalos(
        Accessor acc,
        GCOMPLEX(floatT) *HaloBuffer) {

    size_t size = HaloIndexer<LatLayout, HaloDepth>::getBufferSize();
    if (size == 0) return;

    CalcInnerHaloIndexComm<floatT, LatLayout, HaloDepth> calcIndex;

    ExtractInnerHalo<floatT, Accessor, AccType, EntryCount, ElemCount, LatLayout, HaloDepth> extract(acc, HaloBuffer);

    iterateFunctorNoReturn<onDevice>(extract, calcIndex, size);
}


template<class floatT, class Accessor, class AccType, size_t EntryCount, size_t ElemCount, Layout LatLayout, size_t HaloDepth>
struct InjectOuterHalo {

    Accessor _acc;
    GCOMPLEX(floatT) *pointer[80];
    size_t size[80];
    typedef GIndexer<LatLayout, HaloDepth> GInd;
    typedef HaloIndexer<LatLayout, HaloDepth> HInd;

    InjectOuterHalo(Accessor acc, GCOMPLEX(floatT) *HaloBuffer) :
            _acc(acc) {
        for (int i = 0; i < 80; ++i) {
            pointer[i] = HaloBuffer + HInd::get_SubHaloOffset(i) * EntryCount * ElemCount;
            size[i] = HInd::get_SubHaloSize(i) * ElemCount;
        }
    }

    inline __host__ __device__ void operator()(HaloSite site) {

        Accessor _hal_acc(pointer[site.HalNumber], size[site.HalNumber]);
        for (size_t mu = 0; mu < ElemCount; mu++) {
            size_t index = _acc.template getIndexComm<LatLayout, HaloDepth>(site.LatticeIndex, mu);
            _acc.setEntriesComm(_hal_acc, index, site.LocHalIndex * ElemCount + mu);

        }
    }
};


template<class floatT, bool onDevice, class Accessor, class AccType, size_t EntryCount, size_t ElemCount, Layout LatLayout, size_t HaloDepth>
void
siteComm<floatT, onDevice, Accessor, AccType, EntryCount, ElemCount, LatLayout, HaloDepth>::_injectHalos(
        Accessor acc,
        GCOMPLEX(floatT) *HaloBuffer) {

    size_t size = HaloIndexer<LatLayout, HaloDepth>::getBufferSize();
    if (size == 0) return;

    CalcOuterHaloIndexComm<floatT, LatLayout, HaloDepth> calcIndex;

    InjectOuterHalo<floatT, Accessor, AccType, EntryCount, ElemCount, LatLayout, HaloDepth> inject(acc, HaloBuffer);

    iterateFunctorNoReturn<onDevice>(inject, calcIndex, size);
}




template<class floatT, class Accessor, size_t ElemCount, Layout LatLayout, size_t HaloDepth>
struct ExtractInnerHaloSeg {

    Accessor _acc;
    Accessor _hal_acc;
    typedef GIndexer<LatLayout, HaloDepth> GInd;
    typedef HaloIndexer<LatLayout, HaloDepth> HInd;

    ExtractInnerHaloSeg(Accessor acc, Accessor hal_acc) :
        _acc(acc), _hal_acc(hal_acc) {}

    inline __host__ __device__ void operator()(HaloSite site) {

        for (size_t mu = 0; mu < ElemCount; mu++) {
            size_t index = _acc.template getIndexComm<LatLayout, HaloDepth>(site.LatticeIndex, mu);
            _hal_acc.setEntriesComm(_acc, site.LocHalIndex * ElemCount + mu, index);
        }
    }
};


template<class floatT, bool onDevice, class Accessor, class AccType, size_t EntryCount, size_t ElemCount, Layout LatLayout, size_t HaloDepth>
void siteComm<floatT, onDevice, Accessor, AccType, EntryCount, ElemCount, LatLayout, HaloDepth>::_extractHalosSeg(
        Accessor acc,
        GCOMPLEX(floatT) *HaloBuffer,
        unsigned int param) {

    gpuError_t gpuErr = gpuDeviceSynchronize();
    if (gpuErr != gpuSuccess) {
        GpuError("siteComm.h: _extractHalosSeg gpuDeviceSynchronize failed:", gpuErr);
    }

    _commBase.globalBarrier();

    for (auto &HSegConfig : HSegConfig_send_vec){

        typedef HaloIndexer<LatLayout, HaloDepth> HInd;


        HaloSegment hseg = HSegConfig.haloSeg();
        int dir = HSegConfig.dir();
        int leftRight = HSegConfig.leftRight();
        int subIndex = HSegConfig.subIndex();
        HaloType currentHaltype = HSegConfig.haloType();

        HaloSegmentInfo& segmentInfo = HSegConfig.segmentInfo();

        NeighborInfo& NInfo = HSegConfig.NInfo();
        ProcessInfo& PInfo = HSegConfig.PInfo();

        int size = HSegConfig.size();
        int length = HSegConfig.length();

        if (param & currentHaltype) {
                Accessor hal_acc = HSegConfig.hal_acc();

                int streamNo = 0;
                ExtractInnerHaloSeg<floatT, Accessor, ElemCount, LatLayout, HaloDepth> extractLeft(acc, hal_acc);
                iterateFunctorNoReturn<onDevice>(extractLeft, CalcInnerHaloSegIndexComm<floatT, LatLayout, HaloDepth>(hseg, subIndex),
                        length, 1, 1, segmentInfo.getDeviceStream(streamNo));

                if (PInfo.p2p && onDevice && _commBase.useGpuP2P()) {
                    deviceEventPair &p2pCopyEvent = HaloInfo.getMyGpuEventPair(hseg, dir, leftRight);
                    p2pCopyEvent.start.record(segmentInfo.getDeviceStream());
                }

                if ( (onDevice && _commBase.useGpuP2P() && PInfo.sameRank) ||
                        (onDevice && _commBase.gpuAwareMPIAvail()) )
                {
                    segmentInfo.synchronizeStream(streamNo);
                }

                _commBase.updateSegment(hseg, dir, leftRight, HaloInfo);

                if (PInfo.p2p && onDevice && _commBase.useGpuP2P()) {
                    deviceEventPair &p2pCopyEvent = HaloInfo.getMyGpuEventPair(hseg, dir, leftRight);
                    p2pCopyEvent.stop.record(segmentInfo.getDeviceStream());
                }
        }
    }

    HaloInfo.syncAllStreamRequests();
    _commBase.globalBarrier();
}



template<class floatT, class Accessor, size_t ElemCount, Layout LatLayout, size_t HaloDepth>
struct InjectOuterHaloSeg {

    Accessor _acc;
    Accessor _hal_acc;
    typedef GIndexer<LatLayout, HaloDepth> GInd;

    InjectOuterHaloSeg(Accessor acc, Accessor hal_acc) :
        _acc(acc), _hal_acc(hal_acc) {
        }

    inline __host__ __device__ void operator()(HaloSite site) {

        for (size_t mu = 0; mu < ElemCount; mu++) {
            size_t index = _acc.template getIndexComm<LatLayout, HaloDepth>(site.LatticeIndex, mu);
            _acc.setEntriesComm(_hal_acc, index, site.LocHalIndex * ElemCount + mu);
        }
    }
};

template<class floatT, bool onDevice, class Accessor, class AccType, size_t EntryCount, size_t ElemCount, Layout LatLayout, size_t HaloDepth>
void siteComm<floatT, onDevice, Accessor, AccType, EntryCount, ElemCount, LatLayout, HaloDepth>::_injectHalosSeg(
        Accessor acc,
        GCOMPLEX(floatT) *HaloBuffer, unsigned int param) {


    for (auto &HSegConfig : HSegConfig_recv_vec){
        
        typedef HaloIndexer<LatLayout, HaloDepth> HInd;



        HaloSegment hseg = HSegConfig.haloSeg();
        int dir = HSegConfig.dir();
        int leftRight = HSegConfig.leftRight();
        int subIndex = HSegConfig.subIndex();
        HaloType currentHaltype = HSegConfig.haloType();

        HaloSegmentInfo& segmentInfo = HSegConfig.segmentInfo();

        NeighborInfo& NInfo = HSegConfig.NInfo();
        ProcessInfo& PInfo = HSegConfig.PInfo();

        int size = HSegConfig.size();
        int length = HSegConfig.length();



        if (param & currentHaltype) {


                Accessor hal_acc = HSegConfig.hal_acc();
                int streamNo = 1;

                if (PInfo.p2p && onDevice && _commBase.useGpuP2P()) {
                    deviceEvent &p2pCopyEvent = HaloInfo.getGpuEventPair(hseg, dir, leftRight).stop;
                    p2pCopyEvent.streamWaitForMe(segmentInfo.getDeviceStream(streamNo));
                }

                if (onDevice && _commBase.useGpuP2P() && PInfo.sameRank) {
                    segmentInfo.synchronizeStream(0);
                }
                if (!onDevice || (onDevice && !_commBase.useGpuP2P())) {
                    segmentInfo.synchronizeRequest();
                }

                InjectOuterHaloSeg<floatT, Accessor, ElemCount, LatLayout, HaloDepth> injectLeft(acc, hal_acc);

                iterateFunctorNoReturn<onDevice>(injectLeft, CalcOuterHaloSegIndexComm<floatT, LatLayout, HaloDepth>(hseg, subIndex),
                        length, 1, 1, segmentInfo.getDeviceStream(streamNo));
        }
    }

    gpuError_t gpuErr = gpuDeviceSynchronize();
    if (gpuErr != gpuSuccess) {
        GpuError("siteComm.h: _injectHalosSe, gpuDeviceSynchronize failed:", gpuErr);
    }
    _commBase.globalBarrier();

}

