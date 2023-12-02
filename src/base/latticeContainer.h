/*
 * latticeContainer.h
 *
 * L. Mazur
 *
 * This class oversees LatticeContainer objects, which are essentially intermediate containers
 * used to store intermediate results that will be reduced later. For instance if one calculates
 * the action, one first finds each local contribution, sums these contributions over a sublattice,
 * then sums this result over all sublattices. This whole process is carried out with reduce call.
 *
 * The LatticeContainer can hold elements of arbitrary type, and it is spread over the processes
 * in a similar way as the Gaugefield or Spinorfield. The memory of the LatticeContainer is by
 * default shared with the memory of the halo buffer, because in general the intermediate results
 * have to be re-calculated after a halo update.
 *
 */

#pragma once
#include "communication/communicationBase.h"
#include "indexer/bulkIndexer.h"
#include "memoryManagement.h"
#include "../base/runFunctors.h"
#include "math/operators.h"
#include "wrapper/marker.h"

template<class floatT>
gpuError_t CubReduce(void *helpArr, size_t *temp_storage_bytes, floatT *Arr, floatT *out, size_t size, gpuStream_t stream = NULL);

template<class floatT>
gpuError_t CubReduceMax(void *helpArr, size_t *temp_storage_bytes, void *Arr, floatT *out, size_t size);

template<class floatT>
gpuError_t CubReduceStacked(void *helpArr, size_t *temp_storage_bytes,
        void *Arr, void *out, int Nt, void *TimeSliceOffsets);


class LatticeContainerAccessor : public MemoryAccessor {
public:

    template<bool onDevice>
    explicit LatticeContainerAccessor(gMemoryPtr<onDevice> reductionMemory) : MemoryAccessor(reductionMemory) {}

    explicit LatticeContainerAccessor(void *reductionArray) : MemoryAccessor(reductionArray) {}

    LatticeContainerAccessor() : MemoryAccessor() {}

    /// Set values.
    template<class floatT>
    __device__  __host__ inline void setElement(const size_t isite, const floatT value) {
        auto *arr = reinterpret_cast<floatT *>(Array);
        arr[isite] = value;
    }
    template<class floatT>
    __device__  __host__ inline void setElement(const gSite& site, const floatT value) {
        setValue(site.isite, value);
    }
    template<class floatT>
    __device__  __host__ inline void setElement(const gSiteStack& site, const floatT value) {
        setValue(site.isiteStack, value);
    }

    /// Get values.
    template<class floatT>
    __device__  __host__ floatT getElement(const gSite& site) {
        return getElement<floatT>(site.isite);
    }
    template<class floatT>
    __device__  __host__ floatT getElement(const gSiteStack& site) {
        return getElement<floatT>(site.isiteStack);
    }
    template<class floatT>
    __device__  __host__ inline floatT getElement(const size_t isite) {
        auto *arr = reinterpret_cast<floatT *>(Array);
        return arr[isite];
    }
};



template<bool onDevice, typename elemType>
class LatticeContainer : public RunFunctors<onDevice, LatticeContainerAccessor>  {
private:
    CommunicationBase    &comm;
    gMemoryPtr<onDevice> ContainerArray; /// Points to the array holding your data.
    gMemoryPtr<onDevice> HelperArray;
    gMemoryPtr<onDevice> ReductionResult;
    gMemoryPtr<false>    ReductionResultHost;
    gMemoryPtr<onDevice> d_out;
    gMemoryPtr<false>    StackOffsetsHostTemp;
    gMemoryPtr<onDevice> StackOffsetsTemp;
    gpuStream_t reductionStream;

public:

    //! constructor
    explicit LatticeContainer(CommunicationBase &commBase,
                     std::string nameContainerArr="SHARED_HaloAndReductionA",
                     std::string nameHelpArr     ="SHARED_HaloAndReductionB",
                     std::string nameResult      ="SHARED_HaloAndReductionC",
                     std::string nameResultHost  ="SHARED_HaloAndReductionD" ) :
            comm(commBase),
            ContainerArray(MemoryManagement::getMemAt<onDevice>(nameContainerArr)),
            HelperArray(MemoryManagement::getMemAt<onDevice>(nameHelpArr)),
            ReductionResult(MemoryManagement::getMemAt<onDevice>(nameResult)),
            ReductionResultHost(MemoryManagement::getMemAt<false>(nameResultHost)),
            d_out(MemoryManagement::getMemAt<onDevice>("red_base:d_out")),
            StackOffsetsHostTemp(MemoryManagement::getMemAt<false>("StackOffsetsHostTemp")),
            StackOffsetsTemp(MemoryManagement::getMemAt<onDevice>("StackOffSetsTemp"))
            {
                d_out->template adjustSize<elemType>(1);
                gpuError_t gpuErr = gpuStreamCreate(&reductionStream);
                if (gpuErr != gpuSuccess) GpuError("latticeContainer.h: gpuStreamCreate", gpuErr);
            }

    //! copy constructor
    LatticeContainer(LatticeContainer<onDevice, elemType> &) = delete;

    //! copy assignment
    LatticeContainer<onDevice, elemType>& operator=(LatticeContainer<onDevice, elemType> &) = delete;

    //! move constructor
    //! MAKE SURE TO NOT USE THE DEFAULT SHARED MEMORY NAMES WHEN USING MULTIPLE INDEPENDENT INSTANCES OF THIS CLASS
    LatticeContainer(LatticeContainer<onDevice, elemType>&& source) noexcept :
    comm(source.comm), //! this is a reference, it shouldn't be moved
    //! move all of the gMemoryPtr's:
    ContainerArray(std::move(source.ContainerArray)),
    HelperArray(std::move(source.HelperArray)),
    ReductionResult(std::move(source.ReductionResult)),
    ReductionResultHost(std::move(source.ReductionResultHost)),
    d_out(std::move(source.d_out)),
    StackOffsetsHostTemp(std::move(source.StackOffsetsHostTemp)),
    StackOffsetsTemp(std::move(source.StackOffsetsTemp)),
    reductionStream(std::move(source.reductionStream)){}
    //! move assignment
    LatticeContainer<onDevice, elemType>& operator=(LatticeContainer<onDevice, elemType>&&) = delete;

    //! destructor
    ~LatticeContainer(){
        gpuError_t gpuErr =  gpuStreamDestroy(reductionStream);
        if (gpuErr != gpuSuccess) GpuError("latticeContainer.h: gpuStreamDestroy", gpuErr);
    }

    void adjustSize(size_t size) {

        ContainerArray->template adjustSize<elemType>(size);
    }

    void reduceTimeSlices(std::vector<elemType> &values) {

        int Nt = GIndexer<All, MAXUSEDHALO>::getLatData().lt;
        int vol3 = (int) GIndexer<All, MAXUSEDHALO>::getLatData().vol3;
        std::vector<elemType> values_per_rank;

        reduceStackedLocal(values_per_rank, Nt, vol3);

        size_t globNt = GIndexer<All, MAXUSEDHALO>::getLatData().globLT;
        int NtOffset = GIndexer<All, MAXUSEDHALO>::getLatData().globalPos(LatticeDimensions(0, 0, 0, 0))[3];
        if (values.size() < globNt) {
            values.resize(globNt);
        }
        std::fill(values.begin(), values.end(), 0.0);

        for (int i = 0; i < Nt; i++) {
            values[i + NtOffset] = values_per_rank[i];
        }
        comm.reduce(&values[0], values.size());
    }


    /// Reduce local per rank.
    void reduceStackedLocal(std::vector<elemType> &values, size_t NStacks, size_t stackSize, bool sequentialLoop = false){
        markerBegin("reduceStackedLocal", "Reduction");
        if (values.size() < NStacks) {
            values.resize(NStacks);
        }

        if (onDevice) {
            ReductionResult->template adjustSize<elemType>(NStacks);
            ReductionResultHost->template adjustSize<elemType>(NStacks);

            if (sequentialLoop) {

                for (size_t i = 0; i < NStacks; i++) {
                    /// Determine temporary device storage requirements
                    size_t temp_storage_bytes = 0;
                    gpuError_t gpuErr = CubReduce<elemType>(NULL, &temp_storage_bytes,
                                                            ContainerArray->template getPointer<elemType>(i*stackSize), ReductionResult->template getPointer<elemType>(i), stackSize);

                    if (gpuErr) GpuError("LatticeContainer::reduceStackedLocal: gpucub::DeviceReduce::Sum (1)", gpuErr);

                    HelperArray->template adjustSize<void *>(temp_storage_bytes);

                    gpuErr = CubReduce<elemType>(HelperArray->getPointer(), &temp_storage_bytes,
                                                 ContainerArray->template getPointer<elemType>(i*stackSize), ReductionResult->template getPointer<elemType>(i), stackSize);

                    if (gpuErr) GpuError("LatticeContainer::reduceStackedLocal: gpucub::DeviceReduce::Sum (2)", gpuErr);
                }

            }
            else {

                StackOffsetsHostTemp->template adjustSize<size_t>(NStacks+1);
                LatticeContainerAccessor acc(StackOffsetsHostTemp->getPointer());
                StackOffsetsTemp->template adjustSize<size_t>(NStacks+1);

                for (size_t i = 0; i < NStacks + 1; i++) {
                    acc.setValue(i, i * stackSize);
                }

                StackOffsetsTemp->copyFrom(StackOffsetsHostTemp, sizeof(size_t)*(NStacks+1));
                /// Determine temporary device storage requirements
                size_t temp_storage_bytes = 0;
                gpuError_t gpuErr = CubReduceStacked<elemType>(NULL, &temp_storage_bytes,
                                                               ContainerArray->getPointer(), ReductionResult->getPointer(), NStacks,
                                                               StackOffsetsTemp->getPointer());

                if (gpuErr) GpuError("LatticeContainer::reduceStackedLocal: gpucub::DeviceSegmentedReduce::Sum (1)", gpuErr);

                HelperArray->template adjustSize<void *>(temp_storage_bytes);

                gpuErr = CubReduceStacked<elemType>(HelperArray->getPointer(), &temp_storage_bytes,
                                                    ContainerArray->getPointer(), ReductionResult->getPointer(), NStacks,
                                                    StackOffsetsTemp->getPointer());

                if (gpuErr) GpuError("LatticeContainer::reduceStackedLocal: gpucub::DeviceSegmentedReduce::Sum (2)", gpuErr);
            }

            ReductionResultHost->copyFrom(ReductionResult, NStacks* sizeof(elemType));

            LatticeContainerAccessor accRes(ReductionResultHost->getPointer());
            elemType tmp;
            for (size_t i = 0; i < NStacks; i++) {
                accRes.getValue((size_t) i, tmp);
                values[i] = tmp;
            }

        } else {
            LatticeContainerAccessor acc = getAccessor();
            for (size_t stack = 0; stack < NStacks; stack++) {
                values[stack] = 0;
                for (size_t i = 0; i < stackSize; i++){
                    values[stack] += acc.getElement<elemType>(i + stack * stackSize);
                }
            }
        }
        markerEnd();
    }


    /// Reduce over all processes
    void reduceStacked(std::vector<elemType> &values, size_t NStacks, size_t stackSize, bool sequentialLoop) {
        reduceStackedLocal(values, NStacks, stackSize, sequentialLoop);
        comm.reduce(&values[0], values.size());
    }


    void reduce(elemType &value, size_t size, bool rootToAll = false) {
        markerBegin("reduce", "Reduction");
        elemType result = 0;
        

        if (onDevice) {
            ReductionResultHost->template adjustSize<elemType>(1);
            
            markerBegin("DeviceReduce::Sum (1)", "Reduction");
            // Determine temporary device storage requirements
            size_t temp_storage_bytes = 0;

            gpuError_t gpuErr = CubReduce(NULL, &temp_storage_bytes, ContainerArray->template getPointer<elemType>(),
                                    d_out->template getPointer<elemType>(), size);
            if (gpuErr) GpuError("LatticeContainer::reduce: gpucub::DeviceReduce::Sum (1)", gpuErr);
            markerEnd();

            HelperArray->template adjustSize<void *>(temp_storage_bytes);

            markerBegin("DeviceReduce::Sum (2)", "Reduction");
            gpuErr = CubReduce(HelperArray->getPointer(), &temp_storage_bytes, ContainerArray->template getPointer<elemType>(),
                            d_out->template getPointer<elemType>(), size);
            if (gpuErr) GpuError("LatticeContainer::reduce: gpucub::DeviceReduce::Sum (2)", gpuErr);
            markerEnd();

            markerBegin("DtH copy result", "Reduction");
            ReductionResultHost->copyFrom(d_out, sizeof(elemType));
    
            // LatticeContainerAccessor accRes(ReductionResultHost->getPointer());
            // accRes.getValue((size_t) 0, result);
            result = *(ReductionResultHost->template getPointer<elemType>());
            markerEnd();
        } else{
            LatticeContainerAccessor acc = getAccessor();
            for (size_t i = 0; i < size; i++){
                result += acc.getElement<elemType>(i);
            }
        }
        markerBegin("MPI reduce", "Reduction");
        value = comm.reduce(result);
        if (rootToAll) comm.root2all(result);
        markerEnd();
        markerEnd();
    }

#ifdef USE_NCCL
    void reduce_nccl(gMemoryPtr<onDevice> value, size_t size) {
        markerBegin("reduce", "Reduction");
        auto dcount = d_out->getSize()/sizeof(elemType);
        value->template adjustSize<elemType*>(dcount);

        markerBegin("DeviceReduce::Sum (1)", "Reduction");
        // Determine temporary device storage requirements
        size_t temp_storage_bytes = 0;

        gpuError_t gpuErr = CubReduce(NULL, &temp_storage_bytes, ContainerArray->template getPointer<elemType>(),
                                d_out->template getPointer<elemType>(), size);
        if (gpuErr) GpuError("LatticeContainer::reduce: gpucub::DeviceReduce::Sum (1)", gpuErr);
        markerEnd();

        HelperArray->template adjustSize<void *>(temp_storage_bytes);

        markerBegin("DeviceReduce::Sum (2)", "Reduction");
        gpuErr = CubReduce(HelperArray->getPointer(), &temp_storage_bytes, ContainerArray->template getPointer<elemType>(),
                        d_out->template getPointer<elemType>(), size);
        if (gpuErr) GpuError("LatticeContainer::reduce: gpucub::DeviceReduce::Sum (2)", gpuErr);
        markerEnd();

        // markerBegin("DtH copy result", "Reduction");
        // ReductionResultHost->copyFrom(d_out, sizeof(elemType));

        // // LatticeContainerAccessor accRes(ReductionResultHost->getPointer());
        // // accRes.getValue((size_t) 0, result);
        // result = *(ReductionResultHost->template getPointer<elemType>());
        // markerEnd();
        
        ncclResult_t r = ncclAllReduce(static_cast<const void*>(d_out->getPointer()), static_cast<void*>(value->getPointer()),
         comm.getNumberProcesses(), ncclDouble, ncclSum, comm.getNccl_communicator(), reductionStream);
        if (r != ncclSuccess) {
            NcclError("LatticeContainer::reduce_nccl: ncclAllReduce failed ", r);
        }

        gpuErr = gpuStreamSynchronize(reductionStream);
        if (gpuErr) GpuError("LatticeContainer::reduce: gpuStreamSynchronize", gpuErr);
        markerEnd();
        markerEnd();
    }
#endif

    void reduceMax(elemType &value, size_t size, bool rootToAll = false) {
        markerBegin("reduceMax", "Reduction");
        elemType result = 0;

        if (onDevice) {
            // elemType *d_out = NULL;
            // gpuMalloc((void **) &d_out, sizeof(elemType));
            // Determine temporary device storage requirements
            size_t temp_storage_bytes = 0;

            gpuError_t gpuErr = CubReduceMax(NULL, &temp_storage_bytes, ContainerArray->getPointer(),
                                    d_out->template getPointer<elemType>(), size);
            if (gpuErr) GpuError("LatticeContainer::reduceMax: gpucub::DeviceReduce::Max (1)", gpuErr);

            HelperArray->template adjustSize<void *>(temp_storage_bytes);

            gpuErr = CubReduceMax(HelperArray->getPointer(), &temp_storage_bytes, ContainerArray->getPointer(),
                             d_out->template getPointer<elemType>() , size);
            if (gpuErr) GpuError("LatticeContainer::reduceMax: gpucub::DeviceReduce::Max (2)", gpuErr);
            gpuErr = gpuMemcpy(&result, d_out->template getPointer<elemType>(), sizeof(result), gpuMemcpyDeviceToHost);
            if (gpuErr)
                GpuError("Reductionbase.h: Failed to copy data", gpuErr);
            // gpuFree(d_out);
        } else {
            rootLogger.warn("Max Host reduction has not been properly tested. Check the results and remove this warning");
            LatticeContainerAccessor acc = getAccessor();
            for (size_t i = 0; i < size; i++){
                result = std::max(result, acc.getElement<elemType>(i));
            }
        }
        value = comm.globalMaximum(result);
        if (rootToAll) comm.root2all(result);
        markerEnd();
    }

    LatticeContainerAccessor getAccessor() const { return LatticeContainerAccessor(ContainerArray->getPointer()); }

    gMemoryPtr<onDevice> getMemPointer() { return ContainerArray; }

    template<Layout LatticeLayout, size_t HaloDepth, unsigned BlockSize = DEFAULT_NBLOCKS, typename Functor>
    void iterateOverBulk(Functor op);

    template<Layout LatticeLayout, size_t HaloDepth, unsigned BlockSize = DEFAULT_NBLOCKS, typename Functor>
    void iterateOverCenter(Functor op);

    template<Layout LatticeLayout, size_t HaloDepth, unsigned BlockSize = DEFAULT_NBLOCKS, typename Functor>
    void iterateOverHalo(Functor op);
    
    template<Layout LatticeLayout, size_t HaloDepth, unsigned BlockSize = DEFAULT_NBLOCKS, typename Functor>
    void iterateOverHaloLookup(gSite* Halsites, Functor op);

    template<Layout LatticeLayout, size_t HaloDepth, size_t NStacks, unsigned BlockSize = DEFAULT_NBLOCKS, typename Functor>
    void iterateOverBulkStacked(Functor op);

    template<Layout LatticeLayout, size_t HaloDepth, unsigned BlockSize = DEFAULT_NBLOCKS, typename Functor>
    void iterateOverTimeslices(Functor op);

    template<Layout LatticeLayout, size_t HaloDepth, unsigned BlockSize = DEFAULT_NBLOCKS, typename Functor>

    void iterateOverSpatialBulk(Functor op);


    /// Access to private pointers and references.
    const gMemoryPtr<onDevice>& get_ContainerArrayPtr() const { return ContainerArray; }
    const gMemoryPtr<onDevice>& get_HelperArrayPtr() const { return HelperArray; }
    const gMemoryPtr<onDevice>& get_ReductionResultPtr() const { return ReductionResult; }
    const gMemoryPtr<onDevice>& get_d_outPtr() const { return d_out; }
    const gMemoryPtr<onDevice>& get_StackOffsetsTempPtr() const { return StackOffsetsTemp; }
    CommunicationBase&          get_CommBase() { return comm; }

    /// Equal device assignment operator.
    LatticeContainer<onDevice, elemType> &operator=(const LatticeContainer<onDevice, elemType> &containerRHS) {
        ContainerArray   = containerRHS.get_ContainerArrayPtr();
        HelperArray      = containerRHS.get_HelperArrayPtr();
        ReductionResult  = containerRHS.get_ReductionResultPtr();
        d_out            = containerRHS.get_d_outPtr();
        StackOffsetsTemp = containerRHS.get_StackOffsetsTempPtr();
        return *this;
    }
};


template<bool onDevice, typename elemType>
template<Layout LatticeLayout, size_t HaloDepth, unsigned BlockSize, typename Functor>
void LatticeContainer<onDevice, elemType>::iterateOverBulk(Functor op) {
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;
    CalcGSite<LatticeLayout, HaloDepth> calcGSite;
    WriteAtRead writeAtRead;
    size_t elems;
    if (LatticeLayout == All) {
        elems = GInd::getLatData().vol4;
    } else {
        elems = GInd::getLatData().vol4 / 2;
    }
    this->template iterateFunctor<BlockSize>(op, calcGSite, writeAtRead, elems);
}

template<bool onDevice, typename elemType>
template<Layout LatticeLayout, size_t HaloDepth, unsigned BlockSize, typename Functor>
void LatticeContainer<onDevice, elemType>::iterateOverCenter(Functor op) {
    typedef HaloIndexer<LatticeLayout, HaloDepth> HInd;
    CalcGSiteInnerBulk<LatticeLayout,HaloDepth> calcCenterSite;
    WriteAtRead writeAtRead;
    size_t elems;
    if (LatticeLayout == All) {
        elems = HInd::getCenterSize();
    } else {
        elems = HInd::getCenterSize()/2;
    }
    this->template iterateFunctor<BlockSize>(op, calcCenterSite, writeAtRead, elems);    
}


template<bool onDevice, typename elemType>
template<Layout LatticeLayout, size_t HaloDepth, unsigned BlockSize, typename Functor>
void LatticeContainer<onDevice, elemType>::iterateOverHalo(Functor op) {
    typedef HaloIndexer<LatticeLayout, HaloDepth> HInd;
    CalcGSiteHalo<LatticeLayout,HaloDepth> calcHaloSite;
    WriteAtRead writeAtRead;
    size_t elems;
    if (LatticeLayout == All) {
        elems = HInd::getInnerHaloSize();
    } else {
        elems = HInd::getInnerHaloSize()/2;
    }
    this->template iterateFunctor<BlockSize>(op, calcHaloSite, writeAtRead, elems);    
}

template<bool onDevice, typename elemType>
template<Layout LatticeLayout, size_t HaloDepth, unsigned BlockSize, typename Functor>
void LatticeContainer<onDevice, elemType>::iterateOverHaloLookup(gSite* Halsites, Functor op) {
    typedef HaloIndexer<LatticeLayout, HaloDepth> HInd;
    CalcGSiteHaloLookup calcHaloSite(Halsites);
    WriteAtRead writeAtRead;
    size_t elems;
    if (LatticeLayout == All) {
        elems = HInd::getInnerHaloSize();
    } else {
        elems = HInd::getInnerHaloSize()/2;
    }
    this->template iterateFunctor<BlockSize>(op, calcHaloSite, writeAtRead, elems);    
}


template<bool onDevice, typename elemType>
template<Layout LatticeLayout, size_t HaloDepth, size_t NStacks, unsigned BlockSize, typename Functor>
void LatticeContainer<onDevice, elemType>::iterateOverBulkStacked(Functor op) {
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;
    CalcGSiteStack<LatticeLayout, HaloDepth> calcGSiteStack;
    WriteAtReadStack writeAtReadStack;
    size_t elems;
    if (LatticeLayout == All) {
        elems = GInd::getLatData().vol4;
    } else {
        elems = GInd::getLatData().sizeh;
    }
    this->template iterateFunctor<BlockSize>(op, calcGSiteStack, writeAtReadStack, elems, NStacks);
    if(!onDevice){
        LatticeContainerAccessor acc = getAccessor();
    }
}


template<Layout LatticeLayout, size_t HaloDepth>
struct WriteAtTimeSlices {
    inline __host__ __device__ size_t operator()(const gSite &site) {
        return GIndexer<LatticeLayout, HaloDepth>::siteTimeOrdered(site);
    }
};

template<bool onDevice, typename elemType>
template<Layout LatticeLayout, size_t HaloDepth, unsigned BlockSize, typename Functor>
void LatticeContainer<onDevice, elemType>::iterateOverTimeslices(Functor op) {
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;
    CalcGSite<LatticeLayout, HaloDepth> calcGSite;
    WriteAtTimeSlices<LatticeLayout, HaloDepth> writeAtTimesl;
    size_t elems;
    if (LatticeLayout == All) {
        elems = GInd::getLatData().vol4;
    } else {
        elems = GInd::getLatData().sizeh;
    }
    this->template iterateFunctor<BlockSize>(op, calcGSite, writeAtTimesl, elems);
}

template<bool onDevice, typename elemType>
template<Layout LatticeLayout, size_t HaloDepth, unsigned BlockSize, typename Functor>
void LatticeContainer<onDevice, elemType>::iterateOverSpatialBulk(Functor op) {
    typedef GIndexer<LatticeLayout, HaloDepth> GInd;
    CalcGSiteSpatial<LatticeLayout, HaloDepth> calcGSiteSpatial;
    WriteAtRead writeAtRead;
    size_t elems;
    if (LatticeLayout == All) {
        elems = GInd::getLatData().vol3;
    } else {
        elems = GInd::getLatData().vol3 / 2;
    }
    this->template iterateFunctor<BlockSize>(op, calcGSiteSpatial, writeAtRead, elems);
}

