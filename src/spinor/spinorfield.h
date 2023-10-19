//
// Created by Lukas Mazur on 08.08.18.
//

#pragma once

#include "../base/math/operators.h"
#include "../define.h"
#include "../base/math/vect3array.h"
#include "../base/gutils.h"
#include "../base/latticeContainer.h"
#include "../base/IO/misc.h"
#include "../base/communication/siteComm.h"
#include "../base/communication/communicationBase.h"
#include "../base/math/simpleArray.h"
#include "../base/utilities/static_for_loop.h"
#include <memory>

template <Layout parity>
__host__ __device__ constexpr inline Layout LayoutSwitcher();

template <>
__host__ __device__ constexpr inline Layout LayoutSwitcher<All>() {
    return All;
}
template <>
__host__ __device__ constexpr inline Layout LayoutSwitcher<Odd>() {
    return Even;
}
template <>
__host__ __device__ constexpr inline Layout LayoutSwitcher<Even>() {
    return Odd;
}

template<class floatT_source, class floatT_target, bool onDevice, Layout LatLayout, size_t HaloDepthSpin, size_t NStacks>
    struct convert_spinor_precision;

template<typename floatT, bool onDevice, size_t HaloDepth, size_t NStacks>
    class SpinorfieldAll;

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks = 1>
class Spinorfield : public siteComm<floatT, onDevice, Vect3arrayAcc<floatT>, Vect3<floatT>, 3, NStacks, LatticeLayout, HaloDepth>
{
private:
    Vect3array<floatT, onDevice> _lattice;
    LatticeContainer<onDevice,COMPLEX(double)> _redBase;

    typedef GIndexer<LatticeLayout, HaloDepth> GInd;
    gpuStream_t runStream;
public:
typedef floatT floatT_inner;
    //! constructor
    explicit Spinorfield(CommunicationBase &comm, std::string spinorfieldName="Spinorfield") :
            siteComm<floatT, onDevice, Vect3arrayAcc<floatT>,
            Vect3<floatT>,3, NStacks, LatticeLayout, HaloDepth>(comm),
            _lattice( (int)(NStacks*( (LatticeLayout == All) ? GInd::getLatData().vol4Full : GInd::getLatData().sizehFull )), spinorfieldName ),
            _redBase(comm)
    {
        if (LatticeLayout == All){
            _redBase.adjustSize(GIndexer<LatticeLayout, HaloDepth>::getLatData().vol4 * NStacks);
        }else{
            _redBase.adjustSize(GIndexer<LatticeLayout, HaloDepth>::getLatData().vol4 * NStacks / 2);
        }
        gpuError_t gpuErr = gpuStreamCreate(&runStream);
        if (gpuErr)
            GpuError("Spinorfield constructor: Failed to create runStream", gpuErr);
    }

    //! copy constructor
    Spinorfield(Spinorfield&) = delete;

    //! copy assignment: host to device / device to host
    template<bool onDevice2>
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &
    operator=(const Spinorfield<floatT, onDevice2, LatticeLayout, HaloDepth, NStacks> &spinorRHS) {
        _lattice.copyFrom(spinorRHS.getArray());
        return *this;
    }

    //! copy assignment: host to host / device to device
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &
    operator=(const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &spinorRHS) {
        _lattice.copyFrom(spinorRHS.getArray());
        return *this;
    }

    //! move constructor
    Spinorfield(Spinorfield<floatT,onDevice,LatticeLayout,HaloDepth,NStacks>&& source) noexcept :
            siteComm<floatT, onDevice, Vect3arrayAcc<floatT>,
                    Vect3<floatT>,3, NStacks, LatticeLayout, HaloDepth>(std::move(source)),
            _lattice(std::move(source._lattice)),
            _redBase(std::move(source._redBase)){}

    //! move assignment
    Spinorfield<floatT,onDevice,LatticeLayout,HaloDepth,NStacks>&
            operator=(Spinorfield<floatT,onDevice,LatticeLayout,HaloDepth,NStacks>&&) = delete;

    //! destructor
    ~Spinorfield() {
        gpuError_t gpuErr = gpuStreamDestroy(runStream);
    };

    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &
            operator=(const SpinorfieldAll<floatT, onDevice, HaloDepth, NStacks> &spinorRHS);

    template<bool onDevice2, size_t NStacks2>
    void copyFromStackToStack(const Spinorfield<floatT, onDevice2, LatticeLayout, HaloDepth, NStacks2> &spinorRHS, size_t stackSelf, size_t stackSrc){
        if (stackSelf >= NStacks){
            throw std::runtime_error(stdLogger.fatal("stackSelf larger than NStacks"));
        }
        if (stackSrc >= NStacks2){
            throw std::runtime_error(stdLogger.fatal("stackSrc larger than NStacks"));
        }
        _lattice.copyFromPartial(spinorRHS.getArray(), getNumberLatticePointsFull(), getNumberLatticePointsFull() * stackSelf,
                getNumberLatticePointsFull() * stackSrc);
    }

    const Vect3array<floatT, onDevice>& getArray() const {
        return _lattice;
    }

    COMPLEX(double) dotProduct(const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &y);
    double realdotProduct(const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &y);

    std::vector<COMPLEX(double)> dotProductStacked(const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &y);
    std::vector<double> realdotProductStacked(const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &y);

    void operator*=(const COMPLEX(floatT) &y);
    void operator+=(const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> & S2);
    void operator-=(const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> & S2);

    void gauss(uint4* rand_state); // generate gaussian spinors.

    void one();

    // TODO MOVE THIS WHERE IT BELONGS
    //! this takes a gSite that involves halos!
    template<typename Object>
    void setOneSiteToConst(Object const_obj, const gSite site_full){
        ReadDummy<LatticeLayout, HaloDepth> readIndex; //! we don't actually use the read index here, so we use the dummy
        WriteAtFixedSite<LatticeLayout,HaloDepth> writeIndex(site_full);
        //! we're only changing one entry:
        this->template iterateWithConstObject<1>(const_obj, readIndex, writeIndex, 1, NStacks);
    }

    // TODO MOVE THIS WHERE IT BELONGS
    void setPointSource(const sitexyzt pointsource, const int i_color, const floatT mass){
        //! set whole spinor to zero
        iterateWithConst(vect3_zero<floatT>()); //TODO change this to just memset to 0 for possible performance gain?

        //! if we're on the correct GPU, set one entry to one
        if ( GInd::getLatData().isLocal(pointsource) ){
            sitexyzt pointsource_local(GInd::globalCoordToLocalCoord(pointsource));
            stdLogger.info("Pointsource at " ,  pointsource ,  ": " ,  vect3_unity<double>(i_color)*(double)mass);
            //! TODO add support for multiple RHS (stacks)
            sitexyzt pointsource_full = GIndexer<LatticeLayout,HaloDepth>::coordToFullCoord(pointsource_local);
            gSite tmp = GIndexer<LatticeLayout,HaloDepth>::getSiteFull(pointsource_full);
            setOneSiteToConst(mass*vect3_unity<floatT>(i_color), tmp);
        }
        this->updateAll();
    }

    template<typename const_T>
    void axpyThis(const const_T &x, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> & y);

    void axpyThis(const floatT &x, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> & y);

    template<size_t Blocksize = DEFAULT_NBLOCKS, typename const_T>
    void axpyThisB(const const_T &x, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> & y);

    template<size_t Blocksize = DEFAULT_NBLOCKS, typename const_T>
    void axpyThisLoop(const const_T &x, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &y, size_t stack_entry, gpuStream_t &stream);

    template<size_t Blocksize = DEFAULT_NBLOCKS, typename const_T>
    void axpyThisLoopd(const const_T &x, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &y, size_t stack_entry);

    template<size_t Blocksize = DEFAULT_NBLOCKS>
    void axpyThisB(const floatT &x, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> & y);

    template<typename const_T, size_t Blocksize = DEFAULT_NBLOCKS>
    void xpayThisB(const const_T &x, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> & y);

    template<typename const_T, size_t Blocksize = DEFAULT_NBLOCKS>
    void xpayThisBd(const const_T &x, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> & y);

    template<size_t Blocksize = DEFAULT_NBLOCKS, typename const_T>
    void axupbyThisB(const const_T &a, const const_T &b, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, 1> &y);

    template<size_t Blocksize = DEFAULT_NBLOCKS, typename const_T>
    void axupbyThisLoop(const const_T &a, const const_T &b, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, 1> &y, size_t stack_entry,gpuStream_t &stream);

    template<size_t Blocksize = DEFAULT_NBLOCKS, typename const_T>
    void axupbyThisLoopd(const const_T &a, const const_T &b, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, 1> &y, size_t stack_entry);

    virtual Vect3arrayAcc<floatT> getAccessor() const;

    template<unsigned Blocksize = DEFAULT_NBLOCKS, typename Functor>
    void iterateOverFull(Functor op, size_t Nmax = NStacks);

    template<unsigned Blocksize = DEFAULT_NBLOCKS, typename Functor>
    void iterateOverBulk(Functor op, size_t Nmax = NStacks);

    template<size_t stack, unsigned Blocksize = DEFAULT_NBLOCKS, typename Functor>
    void iterateOverFullAtStack(Functor op);

    template<size_t stack, unsigned Blocksize = DEFAULT_NBLOCKS, typename Functor>
    void iterateOverBulkAtStack(Functor op);

    template<size_t stack, unsigned Blocksize = DEFAULT_NBLOCKS, typename Functor>
    void iterateOverEvenBulkAtStack(Functor op);

    template<size_t stack, unsigned Blocksize = DEFAULT_NBLOCKS, typename Functor>
    void iterateOverOddBulkAtStack(Functor op);

    template<unsigned Blocksize = DEFAULT_NBLOCKS, typename Functor>
    void iterateOverFullLoopStack(Functor op);

    template<unsigned Blocksize = DEFAULT_NBLOCKS, typename Functor>
    void iterateOverBulkLoopStack(Functor op, size_t Nmax=NStacks);

    template<unsigned Blocksize = DEFAULT_NBLOCKS, typename Object>
    void iterateWithConst(Object ob);

    template<typename Functor>
    Spinorfield& operator=(Functor op);

//TODO: put that into the cpp file and fix explicit instantiation macros to reduce compile time
    template<class floatT_source>
    void convert_precision(Spinorfield<floatT_source, onDevice, LatticeLayout, HaloDepth, NStacks> & spinorIn) {
        iterateOverFull(convert_spinor_precision<floatT_source,floatT,onDevice,LatticeLayout, HaloDepth, NStacks>(spinorIn));
    }

    size_t getNumberLatticePoints(){
        size_t elems;
        if (LatticeLayout == All) {
            elems = GInd::getLatData().vol4;
        } else {
            elems = GInd::getLatData().sizeh;
        }
        return elems;
    }

    size_t getNumberLatticePointsFull(){
        size_t elems;
        if (LatticeLayout == All) {
            elems = GInd::getLatData().vol4Full;
        } else {
            elems = GInd::getLatData().sizehFull;
        }
        return elems;
    }

    size_t getNumberElements(){
        return getNumberLatticePoints() * NStacks;
    }

    size_t getNumberElementsFull(){
        return getNumberLatticePointsFull() * NStacks;
    }

};

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
inline Vect3arrayAcc<floatT> Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::getAccessor() const {
    return (_lattice.getAccessor());
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
template<unsigned BlockSize, typename Functor>
void Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::iterateOverFull(Functor op, size_t Nmax){
    CalcGSiteStackFull<LatticeLayout, HaloDepth> calcGSiteFull;
    WriteAtReadStack writeAtRead;
    size_t elems = getNumberLatticePointsFull();
    this->template iterateFunctor<BlockSize>(op, calcGSiteFull, writeAtRead, elems, Nmax, 1, runStream);
  gpuError_t   gpuErr = gpuStreamSynchronize(runStream);
    if (gpuErr)
            GpuError("Spinorfield: Failed to synchronize runStream", gpuErr);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
template<unsigned BlockSize, typename Functor>
void Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::iterateOverFullLoopStack(Functor op){
    CalcGSiteLoopStackFull<LatticeLayout, HaloDepth> calcGSiteFull;
    WriteAtLoopStack<LatticeLayout, HaloDepth> writeAtRead;
    size_t elems = getNumberLatticePointsFull();
    this->template iterateFunctorLoop<NStacks, BlockSize>(op, calcGSiteFull, writeAtRead, elems, 1, 1, runStream);
   gpuError_t  gpuErr = gpuStreamSynchronize(runStream);
    if (gpuErr)
            GpuError("Spinorfield: Failed to synchronize runStream", gpuErr);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
template<unsigned BlockSize, typename Object>
void Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::iterateWithConst(Object ob){
    CalcGSiteStackFull<LatticeLayout, HaloDepth> calcGSiteFull;
    WriteAtReadStack writeAtRead;
    size_t elems = getNumberLatticePointsFull();
    this->template iterateWithConstObject<BlockSize>(ob, calcGSiteFull, writeAtRead, elems, NStacks, 1, runStream);
    gpuError_t gpuErr = gpuStreamSynchronize(runStream);
    if (gpuErr)
            GpuError("Spinorfield: Failed to synchronize runStream", gpuErr);
}



template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
template<unsigned BlockSize, typename Functor>
void Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::iterateOverBulk(Functor op, size_t Nmax){
    CalcGSiteStack<LatticeLayout, HaloDepth> calcGSite;
    WriteAtReadStack writeAtRead;
    size_t elems = getNumberLatticePoints();

    this->template iterateFunctor<BlockSize>(op, calcGSite, writeAtRead, elems, Nmax, 1, runStream);
    gpuError_t gpuErr = gpuStreamSynchronize(runStream);
    if (gpuErr)
            GpuError("Spinorfield: Failed to synchronize runStream", gpuErr);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
template<size_t stack, unsigned BlockSize, typename Functor>
void Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::iterateOverBulkAtStack(Functor op){

    // static_assert(stack < NStacks, "Stack must be smaller than NStacks");

    CalcGSiteAtStack<stack, LatticeLayout, HaloDepth> calcGSite;
    WriteAtReadStack writeAtRead;
    size_t elems = getNumberLatticePoints();

    this->template iterateFunctor<BlockSize>(op, calcGSite, writeAtRead, elems, 1, 1, runStream);
    gpuError_t gpuErr = gpuStreamSynchronize(runStream);
    if (gpuErr)
            GpuError("Spinorfield: Failed to synchronize runStream", gpuErr);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
template<size_t stack, unsigned BlockSize, typename Functor>
void Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::iterateOverEvenBulkAtStack(Functor op){
    if(LatticeLayout == Odd){
        throw std::runtime_error(stdLogger.fatal("You're trying to iterate over the even part of an odd spinorfield!"));
    }
    CalcGSiteAtStack<stack, LatticeLayout, HaloDepth> calcGSite;
    WriteAtReadStack writeAtRead;
    size_t elems = GInd::getLatData().sizeh;
    this->template iterateFunctor<BlockSize>(op, calcGSite, writeAtRead, elems, 1, 1, runStream);
    gpuError_t gpuErr = gpuStreamSynchronize(runStream);
    if (gpuErr)
            GpuError("Spinorfield: Failed to synchronize runStream", gpuErr);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
template<size_t stack, unsigned BlockSize, typename Functor>
void Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::iterateOverOddBulkAtStack(Functor op){
    if(LatticeLayout == Even){
        throw std::runtime_error(stdLogger.fatal("You're trying to iterate over the odd part of an even spinorfield!"));
    }

    //TODO write and read index may be wrong here! the read index we use here is actually the write index, and the true read index depends on whether source spinor latticelayout
    //! TODO i guess we can fix this in the functor that get the read gSite
    WriteAtReadStack writeAtRead;
    size_t elems = GInd::getLatData().sizeh;
    if(LatticeLayout == Odd){
        CalcGSiteAtStack<stack, LatticeLayout, HaloDepth> calcGSite;
        this->template iterateFunctor<BlockSize>(op, calcGSite, writeAtRead, elems, 1, 1, runStream);
   gpuError_t  gpuErr = gpuStreamSynchronize(runStream);
    if (gpuErr)
            GpuError("Spinorfield: Failed to synchronize runStream", gpuErr);
    }
    if(LatticeLayout == All){
        CalcOddGSiteAtStack<stack, LatticeLayout, HaloDepth> calcGSite;
        this->template iterateFunctor<BlockSize>(op, calcGSite, writeAtRead, elems, 1, 1, runStream);
    gpuError_t gpuErr = gpuStreamSynchronize(runStream);
    if (gpuErr)
            GpuError("Spinorfield: Failed to synchronize runStream", gpuErr);
    }

}



template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
template<size_t stack, unsigned BlockSize, typename Functor>
void Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::iterateOverFullAtStack(Functor op){

    static_assert(stack < NStacks, "Stack must be smaller than NStacks");

    CalcGSiteAtStackFull<stack, LatticeLayout, HaloDepth> calcGSiteFull;
    WriteAtReadStack writeAtRead;
    size_t elems = getNumberLatticePoints();

    this->template iterateFunctor<BlockSize>(op, calcGSiteFull, writeAtRead, elems, 1,1,runStream);
    gpuError_t gpuErr = gpuStreamSynchronize(runStream);
    if (gpuErr)
            GpuError("Spinorfield: Failed to synchronize runStream", gpuErr);
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
template<unsigned BlockSize, typename Functor>
void Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::iterateOverBulkLoopStack(Functor op, size_t Nmax){
    CalcGSiteLoopStack<LatticeLayout, HaloDepth> calcGSite;
    WriteAtLoopStack<LatticeLayout, HaloDepth> writeAtRead;
    size_t elems = getNumberLatticePoints();

    this->template iterateFunctorLoop<NStacks, BlockSize>(op, calcGSite, writeAtRead, elems, 1 ,1 , runStream, Nmax);
    gpuError_t gpuErr = gpuStreamSynchronize(runStream);
    if (gpuErr)
            GpuError("Spinorfield: Failed to synchronize runStream", gpuErr);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
template<typename Functor>
Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>&
Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::operator=(Functor op) {
    iterateOverFull(op);
    return *this;
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks, typename T>
auto operator*(const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>& lhs, T rhs)
{
    return general_mult(lhs, rhs);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks, typename T>
auto operator*(T lhs, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>& rhs)
{
    return general_mult(lhs, rhs);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
auto operator*(const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>& lhs,
        const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>& rhs)
{
    return general_mult(lhs, rhs);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks, typename T>
auto operator + (const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>& lhs, T rhs)
{
    return general_add(lhs, rhs);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks, typename T>
auto operator + (T lhs, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>& rhs)
{
    return general_add(lhs, rhs);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
auto operator + (const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>& lhs,
        const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>& rhs)
{
    return general_add(lhs, rhs);
}


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks, typename T>
auto operator - (const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>& lhs, T rhs)
{
    return general_subtract(lhs, rhs);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks, typename T>
auto operator - (T lhs, const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>& rhs)
{
    return general_subtract(lhs, rhs);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
auto operator - (const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>& lhs,
        const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>& rhs)
{
    return general_subtract(lhs, rhs);
}

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks, typename T>
auto operator / (const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>& lhs, T rhs)
{
    return general_divide(lhs, rhs);
}


template<class floatT_source, class floatT_target, bool onDevice, Layout LatLayout, size_t HaloDepthSpin, size_t NStacks>
struct convert_spinor_precision {

    __host__ __device__ void initialize(__attribute__((unused)) gSite& site){
        //We do not initialize anything
    }
    Vect3arrayAcc<floatT_source> spinor_source;

    convert_spinor_precision(Spinorfield<floatT_source, onDevice, LatLayout, HaloDepthSpin, NStacks> &spinorIn) : spinor_source(spinorIn.getAccessor()) {}

    __device__ __host__ auto operator()(gSiteStack site) {

        return spinor_source.template getElement<floatT_target>(site);
    }
};


template<typename floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t Nstacks>
struct returnSpinor {
    Vect3arrayAcc<floatT> _gAcc;

    explicit returnSpinor(const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, Nstacks> &spinorIn);
    __host__ __device__ Vect3<floatT> operator()(gSiteStack site);
};

template<typename floatT, bool onDevice, size_t HaloDepth, size_t NStacks = 1>
class SpinorfieldAll
{
public:
    Spinorfield<floatT, onDevice, Even, HaloDepth, NStacks> even;
    Spinorfield<floatT, onDevice, Odd, HaloDepth, NStacks> odd;

    //! constructor
    explicit SpinorfieldAll(CommunicationBase &comm, std::string spinorfieldName="Spinorfield") :
            even(comm, spinorfieldName),
            odd(comm, spinorfieldName)
    { }

    explicit SpinorfieldAll(Spinorfield<floatT, onDevice, Even, HaloDepth, NStacks> &even,
    Spinorfield<floatT, onDevice, Odd, HaloDepth, NStacks> &odd) :
            even(std::move(even)),
            odd(std::move(odd))
    { }

    //! copy constructor
    SpinorfieldAll(SpinorfieldAll&) = delete;

    //! copy assignment
    SpinorfieldAll<floatT, onDevice, HaloDepth, NStacks> &
            operator=(const SpinorfieldAll<floatT, onDevice, HaloDepth, NStacks> &spinorRHS) {
        even = spinorRHS.even;
        odd = spinorRHS.odd;
        return *this;
    }

    //! copy assignment from a Spinorfield<All> into this SpinorfieldAll
    SpinorfieldAll<floatT, onDevice, HaloDepth, NStacks> &
            operator=(const Spinorfield<floatT, onDevice, All, HaloDepth, NStacks> &spinorRHS) {
        static_for<0, NStacks>::apply([&](auto i) {
             even.template iterateOverEvenBulkAtStack<i>(
                returnSpinor<floatT, onDevice, All, HaloDepth, NStacks>(spinorRHS));
             odd.template iterateOverOddBulkAtStack<i>(
                returnSpinor<floatT, onDevice, All, HaloDepth, NStacks>(spinorRHS));
        });
        return *this;
    }

    //! move constructor
    SpinorfieldAll(SpinorfieldAll<floatT, onDevice, HaloDepth, NStacks>&& source) noexcept :
            even(std::move(source.even)),
            odd(std::move(source.odd))
    { }

    //! move assignment
    SpinorfieldAll<floatT, onDevice, HaloDepth, NStacks> &
    operator=(SpinorfieldAll<floatT, onDevice, HaloDepth, NStacks>&&) = delete;

    //! destructor
    ~SpinorfieldAll() = default;

    COMPLEX(double) dotProduct(const SpinorfieldAll<floatT, onDevice, HaloDepth, NStacks> &y) {
        return even.dotProduct(y.even) + odd.dotProduct(y.odd);
    }
    double realdotProduct(const SpinorfieldAll<floatT, onDevice, HaloDepth, NStacks> &y) {
        return even.realdotProduct(y.even) + odd.realdotProduct(y.odd);
    }

    std::vector<COMPLEX(double)> dotProductStacked(const SpinorfieldAll<floatT, onDevice, HaloDepth, NStacks> &y) {
        std::vector<COMPLEX(double)> even_res = even.dotProductStacked(y.even);
        std::vector<COMPLEX(double)> odd_res = odd.dotProductStacked(y.odd);
        for (std::size_t i = 0, e = even_res.size(); i != e; i++)
            even_res[i] += odd_res[i];
        return even_res;
    }
    std::vector<double> realdotProductStacked(const SpinorfieldAll<floatT, onDevice, HaloDepth, NStacks> &y) {
        std::vector<double> even_res = even.realdotProductStacked(y.even);
        std::vector<double> odd_res = odd.realdotProductStacked(y.odd);
        for (std::size_t i = 0, e = even_res.size(); i != e; i++)
            even_res[i] += odd_res[i];
        return even_res;
    }

    void operator*=(const COMPLEX(floatT) &y) {
        even *= y;
        odd *= y;
    }
    void operator+=(const SpinorfieldAll<floatT, onDevice, HaloDepth, NStacks> & S2) {
        even += S2.even;
        odd += S2.odd;
    }
    void operator-=(const SpinorfieldAll<floatT, onDevice, HaloDepth, NStacks> & S2) {
        even -= S2.even;
        odd -= S2.odd;
    }

    // generate gaussian spinors.
    void gauss(uint4* rand_state) {
        odd.gauss(rand_state);
        even.gauss(rand_state);
    }
    void one() {
        even.one();
        odd.one();
    }

    void updateAll() {
        even.updateAll();
        odd.updateAll();
    }
};

template<typename floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &
Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>::operator=(const SpinorfieldAll<floatT, onDevice, HaloDepth, NStacks> &spinorRHS) {
    if (LatticeLayout == All) {
        static_for<0, NStacks>::apply([&](auto i) {
            this->template iterateOverEvenBulkAtStack<i>(
                returnSpinor<floatT, onDevice, Even, HaloDepth, NStacks>(spinorRHS.even));
            this->template iterateOverOddBulkAtStack<i>(
                returnSpinor<floatT, onDevice, Odd, HaloDepth, NStacks>(spinorRHS.odd));
        });
    }
    else {
        throw std::runtime_error(stdLogger.fatal("SpinorfieldAll<> can only be copied into Spinorfield<All>"));
    }
    return *this;
}
