//
// Created by Lukas Mazur on 08.08.18.
//

#include "spinorfield.h"
#include "../base/utilities/static_for_loop.h"

template <class floatT,size_t elems>
struct fill_with_gauss_vec
{
    uint4 * state;

    explicit fill_with_gauss_vec(uint4 * rand_state) : state(rand_state) {}

    __host__ __device__ void initialize(__attribute__((unused)) gSite& site){}

    __device__ __host__ Vect<floatT,elems> operator()(gSite& site, __attribute__((unused)) size_t stack){

        Vect<floatT,elems> vec;
        vec.gauss(&state[site.isite]);
        return vec;
    }
};

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::gauss(uint4* rand_state)
{
    iterateOverBulkLoopStack(fill_with_gauss_vec<floatT,elems>(rand_state));
    this->updateAll();

}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::unit_basis(const int& i)
{
    iterateWithConst(unit_basis_vect<floatT,elems>(i));
    this->updateAll();
}


template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::one()
{
    iterateWithConst(vect_one<floatT,elems>());
    this->updateAll();
}


/// val = s_in * s_out
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
struct SpinorrealDotProduct{

    VectArrayAcc<floatT,elems> spinorAccOut;
    VectArrayAcc<floatT,elems> spinorAccIn;

    SpinorrealDotProduct(const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>& spinorIn,
            const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>& spinorOut)
        : spinorAccOut(spinorOut.getAccessor()), spinorAccIn(spinorIn.getAccessor()) {}

    __device__ __host__ double  operator()(gSiteStack& site){
        return re_dot_prod(spinorAccIn.template getElement<double>(site), spinorAccOut.template getElement<double>(site));
    }
};

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
struct SpinorReadFromStack{
    VectArrayAcc<floatT,elems> spinorAccIn;
    const size_t _readFromStack;
    SpinorReadFromStack(const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>& spinorIn, const size_t readFromStack) : spinorAccIn(spinorIn.getAccessor()), _readFromStack(readFromStack) {}

    __host__ __device__ Vect<floatT,elems> operator()(gSiteStack& site) {
        typedef GIndexer<LatLayout, HaloDepth> GInd;
        
        return spinorAccIn.getElement(GInd::getSiteStack(site,_readFromStack));
    }
};

/// val = S_in * S_out
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
double Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::realdotProduct(
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks> &y) {
    if (NStacks > 1){
        throw std::runtime_error(stdLogger.fatal("realDotProduct only possible for non stacked spinors"));
    }else{

        double result;

        size_t elems_ = getNumberElements();

        _redBase_real.adjustSize(elems_);

        _redBase_real.template iterateOverBulkStacked<LatLayout, HaloDepth, 1>(
                SpinorrealDotProduct<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>(*this, y));

        _redBase_real.reduce(result, elems_);
        return result;
    }
}


/// val = S_in * S_out
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
std::vector<double> Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::realdotProductStacked(
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks> &y)
{
    std::vector<COMPLEX(double)> result_complex;
    size_t elems_ = getNumberElements();

    _redBase.adjustSize(elems_);

    _redBase.template iterateOverBulkStacked<LatLayout, HaloDepth, elems, NStacks>(
            SpinorrealDotProduct<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>(*this, y));

    _redBase.reduceStacked(result_complex, NStacks, getNumberLatticePoints(), true);
    std::vector<double> result;
    result.resize(result_complex.size());
    for (size_t i = 0; i < result.size(); i++){
        result[i] = result_complex[i].cREAL;
    }
    return result;
}

/// val = s_in * s_out
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
struct SpinorDotProduct{

    VectArrayAcc<floatT,elems> spinorAccOut;
    VectArrayAcc<floatT,elems> spinorAccIn;

    SpinorDotProduct(const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>& spinorIn,
            const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>& spinorOut)
        : spinorAccOut(spinorOut.getAccessor()), spinorAccIn(spinorIn.getAccessor()) {}

    __device__ __host__ COMPLEX(double)  operator()(gSiteStack& site){

        COMPLEX(double) ret =  spinorAccIn.template getElement<double>(site) *  spinorAccOut.template getElement<double>(site);
        return ret;
    }
};

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
template<size_t NStacks2, size_t stackSelf, size_t stackSrc>
    void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::copyFromStackToStackDevice(const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks2> &spinorRHS){
        if (stackSelf >= NStacks){
            throw std::runtime_error(stdLogger.fatal("stackSelf larger than NStacks"));
        }
        if (stackSrc >= NStacks2){
            throw std::runtime_error(stdLogger.fatal("stackSrc larger than NStacks"));
        }
        this->template iterateOverBulkAtStack<stackSelf>(SpinorReadFromStack<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks2>(spinorRHS,stackSrc));
    }

/// val = S_in * S_out
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
COMPLEX(double) Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::dotProduct(
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks> &y)
{
    if (NStacks > 1){
        throw std::runtime_error(stdLogger.fatal("dotProduct only possible for non stacked spinors"));
    }else{

        size_t elems_ = getNumberElements();

        _redBase.adjustSize(elems_);

        _redBase.template iterateOverBulkStacked<LatLayout, HaloDepth, elems, NStacks>(
                SpinorDotProduct<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>(*this, y));

        COMPLEX(double) result = 0;
        _redBase.reduce(result, elems_);
        return result;
    }
}

/// val = S_in * S_out
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
std::vector<COMPLEX(double)> Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::dotProductStacked(
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks> &y)
{
    size_t elems_ = getNumberElements();

    _redBase.adjustSize(elems_);

    _redBase.template iterateOverBulkStacked<LatLayout, HaloDepth, elems, NStacks>(
            SpinorDotProduct<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>(*this, y));

    std::vector<COMPLEX(double)> result;

    _redBase.reduceStacked(result, NStacks, getNumberLatticePoints(), true);
    return result;


}

/// S_out *= val
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::operator*=(const COMPLEX(floatT) &y) {
    iterateOverFull(general_mult(*this, y));
}

template<class floatT, Layout LatLayout, size_t HaloDepth, typename const_T, size_t elems, size_t NStacks>
struct SpinorPlusConstTTimesSpinor{

    VectArrayAcc<floatT,elems> spinor1;
    VectArrayAcc<floatT,elems> spinor2;
    const_T val;
    SpinorPlusConstTTimesSpinor(VectArrayAcc<floatT,elems> spinor1,
                                  VectArrayAcc<floatT,elems> spinor2,
                                  const_T val) : spinor1(spinor1), spinor2(spinor2), val(val){}

    __device__ __host__ Vect<floatT,elems> operator()(gSiteStack& site){
        Vect<floatT,elems> Stmp;
        Stmp = spinor1.getElement(site);
        Stmp += val(site) * spinor2.getElement(site);

        return Stmp;
    }
};


// template<class floatT, Layout LatLayout, size_t HaloDepth, typename const_T, size_t elems, size_t NStacks>
// struct SpinorPlusConstTTimesSpinord{
// 
//     VectArrayAcc<floatT,elems> spinor1;
//     VectArrayAcc<floatT,elems> spinor2;
//     const_T val;
//     SpinorPlusConstTTimesSpinord(VectArrayAcc<floatT,elems> spinor1,
//                                  VectArrayAcc<floatT,elems> spinor2,
//                                   const_T val) : spinor1(spinor1), spinor2(spinor2), val(val){}
// 
//     __device__ __host__ Vect<floatT,elems> operator()(gSiteStack& site){
//         Vect<floatT,elems> Stmp;
//         Stmp = spinor1.template getElement<double>(site);
//         Stmp += val(site) * spinor2.template getElement<double>(site);
// 
//         return Stmp;
//     }
// };


        // pAp = pi0.realdotProduct(s); //Optimization: do dot product but dont copy result to host

        // B[0] = - norm_r2 / pAp; //use device-resident result to do this on the gpu

        // r.axpyThisB(B[0], s); //fuse with this kernel call

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
struct SpinorPlusFloatRatioTimesSpinor{

    VectArrayAcc<floatT,elems> spinor1;
    VectArrayAcc<floatT,elems> spinor2;
    MemoryAccessor val1_acc;
    MemoryAccessor val2_acc;

    SpinorPlusFloatRatioTimesSpinor(VectArrayAcc<floatT,elems> spinor1,
                                  VectArrayAcc<floatT,elems> spinor2,
                                  gMemoryPtr<onDevice> val1, gMemoryPtr<onDevice> val2) : spinor1(spinor1), spinor2(spinor2), 
                                  val1_acc(val1->template getPointer<double>()), val2_acc(val2->template getPointer<double>()){}

    __device__ __host__ Vect<floatT,elems> operator()(gSiteStack& site){
        double v1,v2,val;
        val1_acc.getScalar(v1);
        val2_acc.getScalar(v2); 
        val = -v1/v2;           
        Vect<floatT,elems> Stmp;
        Stmp = spinor1.template getElement<double>(site);
        Stmp += val * spinor2.template getElement<double>(site);

        return Stmp;
    }
};


template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::realDotProductNoCopy(const spin_t &y, __attribute__((unused)) gMemoryPtr<onDevice> pAp) {
    size_t elems_ = getNumberElements();
    _redBase_real.adjustSize(elems_);

    _redBase_real.template iterateOverBulkStacked<LatLayout, HaloDepth, elems, NStacks>(
        SpinorrealDotProduct<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>(*this, y));
#ifdef USE_NCCL
    _redBase_real.reduce_nccl(pAp, elems_);
#else
    rootLogger.fatal("Cannot use realDotProductNoCopy when compiling without NCCL/RCCL!");
#endif
}


template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::axpyThisPtr(const gMemoryPtr<onDevice> a,
        const gMemoryPtr<onDevice> b, const spin_t &y) {
        
          iterateOverBulk<DEFAULT_NBLOCKS>(SpinorPlusFloatRatioTimesSpinor<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>(
                  this->getAccessor(), y.getAccessor(), a,b ));
        
}


template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::fusedDotProdAndaxpy(spin_t &pi, const spin_t &s,
    gMemoryPtr<onDevice> pAp, gMemoryPtr<onDevice> norm_r2) {
    pi.realDotProductNoCopy(s,pAp);

    this->axpyThisPtr(norm_r2, pAp, s);
}

/* /// S_out *= val */
/* template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks> */
/* void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::operator*=(const COMPLEX(floatT) &y) { */
/*     iterateOverFull(general_mult(*this, y)); */
/* } */


template<class floatT, Layout LatLayout, size_t HaloDepth, typename const_T, size_t elems, size_t NStacks>
struct SpinorPlusConstTTimesSpinorNoReturn{

    VectArrayAcc<floatT, elems> spinor1;
    VectArrayAcc<floatT, elems> spinor2;
    const_T val;
    size_t MaxStack;
    SpinorPlusConstTTimesSpinorNoReturn(VectArrayAcc<floatT, elems> spinor1,
                                  VectArrayAcc<floatT, elems> spinor2,
                                  const_T val, size_t maxStack) : spinor1(spinor1), spinor2(spinor2), val(val), MaxStack(maxStack){}

    __device__ __host__ void operator()(gSiteStack& site) {
        typedef GIndexer<LatLayout, HaloDepth> GInd;

        static_for<0,NStacks>::apply([&] (auto stack) {
            if (stack < MaxStack) {
                Vect<floatT, elems> Stmp;
                gSiteStack sitestack = GInd::getSiteStack(site,stack);
                Stmp = spinor1.getElement(sitestack);
                Stmp += val(sitestack) * spinor2.getElement(sitestack);
                spinor1.setElement(sitestack,Stmp);
            }
        });
    }
};


template<class floatT, Layout LatLayout, size_t HaloDepth, typename const_T, size_t elems, size_t NStacks>
struct SpinorPlusConstTTimesSpinord{

    VectArrayAcc<floatT,elems> spinor1;
    VectArrayAcc<floatT,elems> spinor2;
    const_T val;
    SpinorPlusConstTTimesSpinord(VectArrayAcc<floatT,elems> spinor1,
                                 VectArrayAcc<floatT,elems> spinor2,
                                  const_T val) : spinor1(spinor1), spinor2(spinor2), val(val){}

    __device__ __host__ Vect<floatT,elems> operator()(gSiteStack& site){
        Vect<floatT,elems> Stmp;
        Stmp = spinor1.template getElement<double>(site);
        Stmp += val(site) * spinor2.template getElement<double>(site);

        return Stmp;
    }
};


template<class floatT, Layout LatLayout, size_t HaloDepth, typename const_T, size_t elems, size_t NStacks>
struct SpinorPlusConstTTimesSpinorLoop{

    VectArrayAcc<floatT,elems> spinor1;
    VectArrayAcc<floatT,elems> spinor2;
    const_T val;

    typedef GIndexer<LatLayout, HaloDepth> GInd;

    SpinorPlusConstTTimesSpinorLoop(VectArrayAcc<floatT,elems> spinor1,
                                    VectArrayAcc<floatT,elems> spinor2,
                                    const_T val) : spinor1(spinor1), spinor2(spinor2), val(val) {}

    __host__ __device__ void initialize(__attribute__((unused)) gSite& site){
    }

    __device__ __host__ Vect<floatT,elems> operator()(gSite& site, size_t stack){
        Vect<floatT,elems> Stmp;
        gSiteStack siteStack = GInd::getSiteStack(site, stack);

        Stmp = spinor1.getElement(siteStack);
        Stmp += val(siteStack) * spinor2.getElement(siteStack);

        return Stmp;
    }
};

template<class floatT, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
struct SpinorPlusFloatTimesSpinor{

    VectArrayAcc<floatT,elems> spinor1;
    VectArrayAcc<floatT,elems> spinor2;
    floatT val;
    SpinorPlusFloatTimesSpinor(VectArrayAcc<floatT,elems> spinor1,
                               VectArrayAcc<floatT,elems> spinor2,
                               floatT val) : spinor1(spinor1), spinor2(spinor2), val(val){}

    __device__ __host__ Vect<floatT,elems> operator()(gSiteStack& site){
        Vect<floatT,elems> Stmp;
        Stmp = spinor1.getElement(site);
        Stmp += val * spinor2.getElement(site);

        return Stmp;
    }
};

template<class floatT, Layout LatLayout, int HaloDepth, typename const_T, size_t elems, size_t NStacks>
struct StackTimesFloatPlusFloatTimesNoStack
{
    const VectArrayAcc<floatT,elems> spinorIn1;
    const VectArrayAcc<floatT,elems> spinorIn2;
    const_T _a;
    const_T _b;

    typedef GIndexer<LatLayout, HaloDepth> GInd;

    StackTimesFloatPlusFloatTimesNoStack(VectArrayAcc<floatT,elems> spinorIn1,
            SimpleArray<floatT, NStacks> a,
            VectArrayAcc<floatT, elems> spinorIn2,
            SimpleArray<floatT, NStacks> b) :
        spinorIn1(spinorIn1), spinorIn2(spinorIn2), _a(a), _b(b) {}


    __host__ __device__ Vect<floatT,elems> operator()(gSiteStack& siteStack){
        gSiteStack siteUnStack = GInd::getSiteStack(siteStack, 0);

        const Vect<floatT,elems> my_vec = spinorIn1.getElement(siteStack)*_a(siteStack) + spinorIn2.getElement(siteUnStack)*_b(siteStack);

        return my_vec;
    }
};

template<class floatT, Layout LatLayout, int HaloDepth, typename const_T, size_t elems, size_t NStacks>
struct StackTimesFloatPlusFloatTimesNoStackLoop
{
    VectArrayAcc<floatT,elems> spinorIn1;
    VectArrayAcc<floatT,elems> spinorIn2;
    const_T _a;
    const_T _b;

    typedef GIndexer<LatLayout, HaloDepth> GInd;

    StackTimesFloatPlusFloatTimesNoStackLoop(VectArrayAcc<floatT,elems> spinorIn1,
            SimpleArray<floatT, NStacks> a,
            VectArrayAcc<floatT,elems> spinorIn2,
            SimpleArray<floatT, NStacks> b) :
        spinorIn1(spinorIn1), spinorIn2(spinorIn2), _a(a), _b(b) {}

    __host__ __device__ void initialize(__attribute__((unused)) gSite& site){
    }


    __host__ __device__ Vect<floatT,elems> operator()(gSite& site, size_t stack){
        gSiteStack siteUnStack = GInd::getSiteStack(site, 0);
        gSiteStack siteStack = GInd::getSiteStack(site, stack);
        Vect<floatT,elems> my_vec;

        my_vec = spinorIn1.getElement(siteStack)*_a(siteStack) + spinorIn2.getElement(siteUnStack)*_b(siteStack);

        return my_vec;
    }
};

template<class floatT, Layout LatLayout, int HaloDepth, typename const_T, size_t elems, size_t NStacks>
struct StackTimesFloatPlusFloatTimesNoStackNoReturn
{
    VectArrayAcc<floatT,elems> spinorIn1;
    VectArrayAcc<floatT,elems> spinorIn2;
    const_T _a;
    const_T _b;
    size_t MaxStack;
    typedef GIndexer<LatLayout, HaloDepth> GInd;

    StackTimesFloatPlusFloatTimesNoStackNoReturn(VectArrayAcc<floatT,elems> spinorIn1,
            SimpleArray<floatT, NStacks> a,
            VectArrayAcc<floatT,elems> spinorIn2,
            SimpleArray<floatT, NStacks> b, size_t maxStack) :
        spinorIn1(spinorIn1), spinorIn2(spinorIn2), _a(a), _b(b), MaxStack(maxStack) {}

    __host__ __device__ void initialize(__attribute__((unused)) gSite& site){
    }


    __host__ __device__ void operator()(gSiteStack& site){
        static_for<0,NStacks>::apply([&] (auto stack){ 
            if (stack < MaxStack) {
                gSiteStack siteUnStack = GInd::getSiteStack(site, 0);
                gSiteStack siteStack = GInd::getSiteStack(site, stack);
                Vect<floatT,elems> my_vec;

                my_vec = spinorIn1.getElement(siteStack)*_a(siteStack) + spinorIn2.getElement(siteUnStack)*_b(siteStack);
                spinorIn1.setElement(siteStack,my_vec);
            }
        });
    }
};



template<class floatT, Layout LatLayout, int HaloDepth, typename const_T, size_t elems, size_t NStacks>
struct StackTimesFloatPlusFloatTimesNoStackLoop_d
{
    VectArrayAcc<floatT,elems> spinorIn1;
    VectArrayAcc<floatT,elems> spinorIn2;
    const_T _a;
    const_T _b;

    typedef GIndexer<LatLayout, HaloDepth> GInd;

    StackTimesFloatPlusFloatTimesNoStackLoop_d(VectArrayAcc<floatT,elems> spinorIn1,
            SimpleArray<double, NStacks> a,
            VectArrayAcc<floatT,elems> spinorIn2,
            SimpleArray<double, NStacks> b) :
        spinorIn1(spinorIn1), spinorIn2(spinorIn2), _a(a), _b(b) {}

    __host__ __device__ void initialize(__attribute__((unused)) gSite& site){
    }


    __host__ __device__ Vect<floatT,elems> operator()(gSite& site, size_t stack){
        gSiteStack siteUnStack = GInd::getSiteStack(site, 0);
        gSiteStack siteStack = GInd::getSiteStack(site, stack);
        Vect<double,elems> my_vec;

        my_vec = spinorIn1.template getElement<double>(siteStack)*_a(siteStack) + spinorIn2.template getElement<double>(siteUnStack)*_b(siteStack);

        return my_vec;
    }
};

/// S_this_in += S2
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::operator+=(
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks> &S2) {
    iterateOverFull(general_add(*this, S2));
}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::operator-=(
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks> &S2) {
    iterateOverFull(general_subtract(*this, S2));
}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
template<typename const_T>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::axpyThis(const const_T &x,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks> &y) {

    iterateOverFull(SpinorPlusConstTTimesSpinor<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
                this->getAccessor(), y.getAccessor(), x));
}


template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::axpyThis(const floatT &x,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks> &y) {

    iterateOverFull(SpinorPlusFloatTimesSpinor<floatT, LatLayout, HaloDepth, elems, NStacks>(
                this->getAccessor(), y.getAccessor(), x));
}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
template<size_t BlockSize, typename const_T>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::axpyThisB(const const_T &x,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks> &y) {

    iterateOverBulk<BlockSize>(SpinorPlusConstTTimesSpinor<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
                this->getAccessor(), y.getAccessor(), x));

}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
template<size_t BlockSize>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::axpyThisB(const floatT &x,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks> &y) {

    iterateOverBulk<BlockSize>(SpinorPlusFloatTimesSpinor<floatT, LatLayout, HaloDepth, elems, NStacks>(
                this->getAccessor(), y.getAccessor(), x));

}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
template<size_t BlockSize, typename const_T>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::axpyThisLoop(const const_T &x,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks> &y, size_t stack_entry) {

    // static_for<0, NStacks>::apply([&](auto i) {
    //      // change this to include a vector of bools, to see wether one needs to modify the vector, or if it has converged

    //         if(stack_entry >= i+1 ) {
    //             iterateOverBulkAtStack<i, BlockSize>(
    //                     SpinorPlusConstTTimesSpinor<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
    //                             this->getAccessor(), y.getAccessor(), x));
    //         }
    //         });
        CalcGSiteStackFull<LatLayout, HaloDepth> calcGSite;
        iterateFunctorNoReturn<onDevice, BlockSize>(SpinorPlusConstTTimesSpinorNoReturn<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
            this->getAccessor(), y.getAccessor(), x, stack_entry), calcGSite,this->getNumberLatticePointsFull(), 1);
}


template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
template<size_t BlockSize, typename const_T>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::axpyThisLoopd(const const_T &x,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks> &y, size_t stack_entry) {

    static_for<0, NStacks>::apply([&](auto i) {
         // change this to include a vector of bools, to see wether one needs to modify the vector, or if it has converged

            if(stack_entry >= i+1 ) {
                iterateOverBulkAtStack<i, BlockSize>(
                        SpinorPlusConstTTimesSpinord<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
                                this->getAccessor(), y.getAccessor(), x));
            }
            });
}


template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
template<size_t BlockSize, typename const_T>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::axupbyThisB(const const_T &a, const const_T &b,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, 1> &y) {

    iterateOverBulk<BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
                this->getAccessor(), b, y.getAccessor(), a));

}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
template<size_t BlockSize, typename const_T>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::axupbyThisLoopd(const const_T &a, const const_T &b,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, 1> &y, size_t stack_entry) {

    iterateOverBulkLoopStack<BlockSize>(StackTimesFloatPlusFloatTimesNoStackLoop_d<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
                this->getAccessor(), b, y.getAccessor(), a), stack_entry);


    //             this->getAccessor(), b, y.getAccessor(), a));

}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
template<size_t BlockSize, typename const_T>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::axupbyThisLoop(const const_T &a, const const_T &b,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, 1> &y, size_t stack_entry) {

    iterateOverBulkLoopStack<BlockSize>(StackTimesFloatPlusFloatTimesNoStackLoop<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
                this->getAccessor(), b, y.getAccessor(), a), stack_entry);


    // iterateOverBulkAtStack<0,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=2 )
    // iterateOverBulkAtStack<1,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=3 )
    // iterateOverBulkAtStack<2,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=4 )
    // iterateOverBulkAtStack<3,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=5 )
    // iterateOverBulkAtStack<4,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=6 )
    // iterateOverBulkAtStack<5,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=7 )
    // iterateOverBulkAtStack<6,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=8 )
    // iterateOverBulkAtStack<7,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=9 )
    // iterateOverBulkAtStack<8,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=10 )
    // iterateOverBulkAtStack<9,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=11 )
    // iterateOverBulkAtStack<10,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=12 )
    // iterateOverBulkAtStack<11,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >= 13)
    // iterateOverBulkAtStack<12,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry == 14)
    // iterateOverBulkAtStack<13,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));

}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
template<typename const_T, size_t BlockSize>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::xpayThisB(const const_T &x,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks> &y){

    iterateOverBulk<BlockSize>(SpinorPlusConstTTimesSpinor<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
                y.getAccessor(), this->getAccessor(), x));

}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t NStacks>
template<typename const_T, size_t BlockSize>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks>::xpayThisBd(const const_T &x,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, NStacks> &y){

    iterateOverBulk<BlockSize>(SpinorPlusConstTTimesSpinord<floatT, LatLayout, HaloDepth, const_T, elems, NStacks>(
                y.getAccessor(), this->getAccessor(), x));

}

/* template<typename floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t Nstacks> */
/* returnSpinor<floatT, onDevice, LatLayout, HaloDepth, elems, Nstacks>::returnSpinor(const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, elems, Nstacks> &spinorIn) : */
/*         _gAcc(spinorIn.getAccessor()) { */
/* } */

/* template<typename floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t elems, size_t Nstacks> */
/* __host__ __device__ Vect<floatT,elems> returnSpinor<floatT, onDevice, LatLayout, HaloDepth, elems, Nstacks>::operator()(gSiteStack site) { */
/*     //! Deduce gSiteStacked object for the source from the gSite object of the destination */
/*     gSite temp = GIndexer<LatLayout,HaloDepth>::getSite(site.coord); */
/*     return _gAcc.template getElement<floatT>(temp); */
/* } */

#define SPINOR_INIT_PLHSN(floatT,LO,HALOSPIN,STACKS)\
template class Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>;\
template void Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>::copyFromStackToStackDevice<1,0,0>(const Spinorfield<floatT,false,LO,HALOSPIN,3,1>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>::copyFromStackToStackDevice<2,0,0>(const Spinorfield<floatT,false,LO,HALOSPIN,3,2>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>::copyFromStackToStackDevice<3,0,0>(const Spinorfield<floatT,false,LO,HALOSPIN,3,3>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>::copyFromStackToStackDevice<4,0,0>(const Spinorfield<floatT,false,LO,HALOSPIN,3,4>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>::copyFromStackToStackDevice<8,0,0>(const Spinorfield<floatT,false,LO,HALOSPIN,3,8>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>::copyFromStackToStackDevice<10,0,0>(const Spinorfield<floatT,false,LO,HALOSPIN,3,10>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>::copyFromStackToStackDevice<12,0,0>(const Spinorfield<floatT,false,LO,HALOSPIN,3,12>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>::copyFromStackToStackDevice<14,0,0>(const Spinorfield<floatT,false,LO,HALOSPIN,3,14>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>::copyFromStackToStackDevice<32,0,0>(const Spinorfield<floatT,false,LO,HALOSPIN,3,32>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>::axpyThis(const COMPLEX(floatT)&, const Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>::axpyThisB(const COMPLEX(floatT)&, const Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>::axpyThisB<64>(const floatT&, const Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>::axpyThis(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>::axpyThisB(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>::axpyThisB<32>(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>::xpayThisB(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,3,STACKS>::axupbyThisB(const SimpleArray<floatT, STACKS>&, const SimpleArray<floatT, STACKS>&, const Spinorfield<floatT,false,LO,HALOSPIN,3,1>&);\
template void Spinorfield<floatT, false, LO, HALOSPIN,3, STACKS>::axupbyThisLoop<64>(const SimpleArray<floatT, STACKS> &a, const SimpleArray<floatT, STACKS> &b, const Spinorfield<floatT, false, LO, HALOSPIN,3, 1> &y, size_t stack_entry);\
template void Spinorfield<floatT, false, LO, HALOSPIN,3, STACKS>::axpyThisLoop<64>(const SimpleArray<floatT, STACKS> &x, const Spinorfield<floatT, false, LO, HALOSPIN,3, STACKS> &y, size_t stack_entry);\
template struct returnSpinor<floatT,false,LO,HALOSPIN,3 ,STACKS>;\


INIT_PLHSN(SPINOR_INIT_PLHSN)

#define SPINOR_INIT_PLHSN_HALF(floatT,LO,HALOSPIN,STACKS)\
template class Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>;\
template void Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>::copyFromStackToStackDevice<1,0,0>(const Spinorfield<floatT,true,LO,HALOSPIN,3,1>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>::copyFromStackToStackDevice<2,0,0>(const Spinorfield<floatT,true,LO,HALOSPIN,3,2>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>::copyFromStackToStackDevice<3,0,0>(const Spinorfield<floatT,true,LO,HALOSPIN,3,3>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>::copyFromStackToStackDevice<4,0,0>(const Spinorfield<floatT,true,LO,HALOSPIN,3,4>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>::copyFromStackToStackDevice<8,0,0>(const Spinorfield<floatT,true,LO,HALOSPIN,3,8>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>::copyFromStackToStackDevice<10,0,0>(const Spinorfield<floatT,true,LO,HALOSPIN,3,10>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>::copyFromStackToStackDevice<12,0,0>(const Spinorfield<floatT,true,LO,HALOSPIN,3,12>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>::copyFromStackToStackDevice<14,0,0>(const Spinorfield<floatT,true,LO,HALOSPIN,3,14>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>::copyFromStackToStackDevice<32,0,0>(const Spinorfield<floatT,true,LO,HALOSPIN,3,32>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>::axpyThis(const COMPLEX(floatT)&,const Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>::axpyThisB(const COMPLEX(floatT)&,const Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>::axpyThisB<64>(const floatT&,const Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>::axpyThis(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>::axpyThisB(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>::axpyThisB<32>(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>::xpayThisB(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,true,LO,HALOSPIN,3, STACKS>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>::axupbyThisB(const SimpleArray<floatT, STACKS>&, const SimpleArray<floatT, STACKS>&, const Spinorfield<floatT,true,LO,HALOSPIN,3,1>&);\
template void Spinorfield<floatT, true, LO, HALOSPIN,3, STACKS>::axupbyThisLoop<64>(const SimpleArray<floatT, STACKS> &a, const SimpleArray<floatT, STACKS> &b, const Spinorfield<floatT, true, LO, HALOSPIN,3, 1> &y, size_t stack_entry);\
template void Spinorfield<floatT, true, LO, HALOSPIN,3, STACKS>::axupbyThisLoopd<64>(const SimpleArray<double, STACKS> &a, const SimpleArray<double, STACKS> &b, const Spinorfield<floatT, true, LO, HALOSPIN,3, 1> &y, size_t stack_entry); \
template void Spinorfield<floatT, true, LO, HALOSPIN,3, STACKS>::axpyThisLoop<64>(const SimpleArray<floatT, STACKS> &x, const Spinorfield<floatT, true, LO, HALOSPIN,3, STACKS> &y, size_t stack_entry);\
template void Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>::xpayThisBd<SimpleArray<double,STACKS>,64>(const SimpleArray<double, STACKS>&,const Spinorfield<floatT,true,LO,HALOSPIN,3,STACKS>&); \
template void Spinorfield<floatT, true, LO, HALOSPIN,3, STACKS>::axpyThisLoopd<32>(const SimpleArray<double, STACKS> &x, const Spinorfield<floatT, true, LO, HALOSPIN,3, STACKS> &y, size_t stack_entry);\
template void Spinorfield<floatT, true, LO, HALOSPIN,3, STACKS>::axpyThisLoopd<64>(const SimpleArray<double, STACKS> &x, const Spinorfield<floatT, true, LO, HALOSPIN,3, STACKS> &y, size_t stack_entry);\
template struct returnSpinor<floatT,true,LO,HALOSPIN,3,STACKS>;\

INIT_PLHSN_HALF(SPINOR_INIT_PLHSN_HALF)
