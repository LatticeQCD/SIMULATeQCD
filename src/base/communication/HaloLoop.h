/*
 * HaloLoop.h
 *
 * L. Mazur
 *
 */

#ifndef HALOLOOP_H
#define HALOLOOP_H

#define DYNAMIC_HALO_LOOP

#include "communicationBase.h"
#include "../runFunctors.h"

#ifdef DYNAMIC_HALO_LOOP

#include "calcGSiteHalo_dynamic.h"

#else
#include "calcGSiteHalo.h"
#endif

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

template<bool onDevice, class floatT, class Accessor, size_t ElemCount, size_t EntryCount, Layout LatLayout, size_t HaloDepth, int N = 0>
class extractLoop {
    typedef HaloIndexer<LatLayout, HaloDepth> HInd;
public:
    extractLoop(
            Accessor acc,
            CommunicationBase &commBase,
            HaloOffsetInfo<onDevice> &HalInfo,
            GCOMPLEX(floatT) *haloBuffer, unsigned int param) {


        const HaloSegment hseg = HSegSelector<N>().haloSeg();
        const int dir = HSegSelector<N>().dir();
        const int leftRight = HSegSelector<N>().leftRight();
        const int subIndex = HSegSelector<N>().subIndex();
        const HaloType currentHaltype = HSegSelector<N>().haloType();

        HaloSegmentInfo &segmentInfo = HalInfo.get(hseg, dir, leftRight);

        NeighborInfo &NInfo = HalInfo.getNeighborInfo();
        ProcessInfo &info = NInfo.getNeighborInfo(hseg, dir, leftRight);

        int index = haloSegmentCoordToIndex(hseg, dir, leftRight);
        int size = HInd::get_SubHaloSize(index) * ElemCount;

        if (param & currentHaltype) {
            if (size != 0) {
                const int index = N;
                int length = HInd::get_SubHaloSize(index);
                int size = HInd::get_SubHaloSize(index) * ElemCount;
                GCOMPLEX(floatT) *pointer = haloBuffer + HInd::get_SubHaloOffset(index) * EntryCount * ElemCount;
                if (size != 0) {
                    Accessor hal_acc(pointer, size);

                    int streamNo = 0;
                    ExtractInnerHaloSeg<floatT, Accessor, ElemCount, LatLayout, HaloDepth> extractLeft(acc,
                                                                                                       hal_acc);
#ifdef DYNAMIC_HALO_LOOP
                    iterateFunctorNoReturn<onDevice>(extractLeft,
                                                            CalcInnerHaloSegIndexComm<floatT, LatLayout, HaloDepth>(
                                                                    hseg, subIndex),
                                                            length, 1, 1, segmentInfo.getDeviceStream(streamNo));
#else
                    iterateFunctorNoReturn<onDevice>(extractLeft,
                                                    CalcInnerHaloSegIndexComm<floatT, LatLayout, HaloDepth, hseg, subIndex>(),
                                                    length, 1, 1, segmentInfo.getDeviceStream(streamNo));
#endif

                    if (info.p2p && onDevice && commBase.useGpuP2P()) {
                        deviceEventPair &p2pCopyEvent = HalInfo.getMyGpuEventPair(hseg, dir, leftRight);
                        p2pCopyEvent.start.record(segmentInfo.getDeviceStream());
                    }

                    if ( (onDevice && commBase.useGpuP2P() && info.sameRank) ||
                            (onDevice && commBase.gpuAwareMPIAvail()) )
                    {
                        segmentInfo.synchronizeStream(streamNo);
                    }

                    commBase.updateSegment(hseg, dir, leftRight, HalInfo);

                    if (info.p2p && onDevice && commBase.useGpuP2P()) {
                        deviceEventPair &p2pCopyEvent = HalInfo.getMyGpuEventPair(hseg, dir, leftRight);
                        p2pCopyEvent.stop.record(segmentInfo.getDeviceStream());
                    }
                }
            }
        }
        extractLoop<onDevice, floatT, Accessor, ElemCount, EntryCount, LatLayout, HaloDepth,
                N + 1> tmp(acc, commBase, HalInfo, haloBuffer, param);

    }
};

//Specialized template
template<bool onDevice, class floatT, class Accessor, size_t ElemCount, size_t EntryCount, Layout LatLayout, size_t HaloDepth>
class extractLoop<onDevice, floatT, Accessor, ElemCount, EntryCount, LatLayout, HaloDepth, 80> {
    //Do Nothing, our loop has finished
public:

    extractLoop(
            __attribute__((unused)) Accessor acc,
            __attribute__((unused)) CommunicationBase &commBase,
            __attribute__((unused)) HaloOffsetInfo<onDevice> &HalInfo,
            __attribute__((unused)) GCOMPLEX(floatT) *haloBuffer,
            __attribute__((unused)) unsigned int param
    ) {
    }
};

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

template<bool onDevice, class floatT, class Accessor, size_t ElemCount, size_t EntryCount, Layout LatLayout, size_t HaloDepth, int N = 0>
class injectLoop {
    //Execute some code
    typedef HaloIndexer<LatLayout, HaloDepth> HInd;
public:
    injectLoop(CommunicationBase &commBase, HaloOffsetInfo<onDevice> &HalInfo, Accessor acc,
               GCOMPLEX(floatT) *haloBuffer, unsigned int param) {

        const HaloSegment hseg = HSegSelector<N>().haloSeg();
        const int dir = HSegSelector<N>().dir();
        const int leftRight = HSegSelector<N>().leftRight();
        const int subIndex = HSegSelector<N>().subIndex();

        HaloSegmentInfo &segmentInfo = HalInfo.get(hseg, dir, leftRight);

        int index = haloSegmentCoordToIndex(hseg, dir, leftRight);
        GCOMPLEX(floatT) *pointer = haloBuffer + HInd::get_SubHaloOffset(index) * EntryCount * ElemCount;
        int length = HInd::get_SubHaloSize(index);
        int size = HInd::get_SubHaloSize(index) * ElemCount;
        const HaloType currentHaltype = HSegSelector<N>().haloType();

        NeighborInfo &NInfo = HalInfo.getNeighborInfo();
        ProcessInfo &info = NInfo.getNeighborInfo(hseg, dir, leftRight);

        if (param & currentHaltype) {

            if (size != 0) {

                int streamNo = 1;

                if (info.p2p && onDevice && commBase.useGpuP2P()) {
                    deviceEvent &p2pCopyEvent = HalInfo.getGpuEventPair(hseg, dir, leftRight).stop;
                    p2pCopyEvent.streamWaitForMe(segmentInfo.getDeviceStream(streamNo));
                }

                if (onDevice && commBase.useGpuP2P() && info.sameRank) {
                    segmentInfo.synchronizeStream(0);
                }
                if (!onDevice || (onDevice && !commBase.useGpuP2P())) {
                    segmentInfo.synchronizeRequest();
                }

                InjectOuterHaloSeg<floatT, Accessor, ElemCount, LatLayout, HaloDepth> injectLeft(acc,
                                                                                                 Accessor(pointer,
                                                                                                          size));
#ifdef DYNAMIC_HALO_LOOP
                iterateFunctorNoReturn<onDevice>(injectLeft,
                                                        CalcOuterHaloSegIndexComm<floatT, LatLayout, HaloDepth>(
                                                                hseg,
                                                                subIndex),
                                                        length, 1, 1, segmentInfo.getDeviceStream(streamNo));
#else
                iterateFunctorNoReturn<onDevice>(injectLeft,
                                                        CalcOuterHaloSegIndexComm<floatT, LatLayout, HaloDepth, hseg, subIndex>(),
                                                        length, 1, 1, segmentInfo.getDeviceStream(streamNo));
#endif
            }
        }
        //Recurse
        injectLoop<onDevice, floatT, Accessor, ElemCount, EntryCount, LatLayout, HaloDepth, N + 1> tmp(commBase,
                                                                                                       HalInfo, acc,
                                                                                                       haloBuffer,
                                                                                                       param);
    }
};

//Specialized template
template<bool onDevice, class floatT, class Accessor, size_t ElemCount, size_t EntryCount, Layout LatLayout, size_t HaloDepth>
class injectLoop<onDevice, floatT, Accessor, ElemCount, EntryCount, LatLayout, HaloDepth, 80> {
    //Do Nothing, our loop has finished
public:
    injectLoop(__attribute__((unused)) CommunicationBase &commBase,
               __attribute__((unused)) HaloOffsetInfo<onDevice> &HalInfo,
               __attribute__((unused)) Accessor acc,
               __attribute__((unused)) GCOMPLEX(floatT) *haloBuffer,
               __attribute__((unused)) unsigned int param) {
    }
};

#endif //HALOLOOP_H
