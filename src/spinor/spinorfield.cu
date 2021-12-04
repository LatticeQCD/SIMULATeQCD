//
// Created by Lukas Mazur on 08.08.18.
//

#include "spinorfield.h"
#include "../base/static_for_loop.h"

template <class floatT>
struct fill_with_gauss_vec
{
    uint4 * state;

    explicit fill_with_gauss_vec(uint4 * rand_state) : state(rand_state) {}

    __host__ __device__ void initialize(__attribute__((unused)) gSite& site){}

    __device__ __host__ gVect3<floatT> operator()(gSite& site, __attribute__((unused)) size_t stack){

        gVect3<floatT> vec;
        vec.gauss(&state[site.isite]);
        return vec;
    }
};

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::gauss(uint4* rand_state)
{
    iterateOverBulkLoopStack(fill_with_gauss_vec<floatT>(rand_state));
    this->updateAll();

}


template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::one()
{
    iterateWithConst(gvect3_unity<floatT>(0));
    this->updateAll();
}


/// val = s_in * s_out
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
struct SpinorrealDotProduct{

    gVect3arrayAcc<floatT> spinorAccOut;
    gVect3arrayAcc<floatT> spinorAccIn;

    SpinorrealDotProduct(const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>& spinorIn,
            const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>& spinorOut)
        : spinorAccOut(spinorOut.getAccessor()), spinorAccIn(spinorIn.getAccessor()) {}

    __device__ __host__ GCOMPLEX(double)  operator()(gSiteStack& site){
        return re_dot_prod(spinorAccIn.template getElement<double>(site), spinorAccOut.template getElement<double>(site));
    }
};

/// val = S_in * S_out
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
double Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::realdotProduct(
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &y) {
    if (NStacks > 1){
        throw std::runtime_error(stdLogger.fatal("realDotProduct only possible for non stacked spinors");
    }else{

        GCOMPLEX(double) result = 0;

        size_t elems = getNumberElements();

        _redBase.adjustSize(elems);

        _redBase.template iterateOverBulkStacked<LatLayout, HaloDepth, 1>(
                SpinorrealDotProduct<floatT, onDevice, LatLayout, HaloDepth, NStacks>(*this, y));

        _redBase.reduce(result, elems);
        return result.cREAL;
    }
}


/// val = S_in * S_out
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
std::vector<double> Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::realdotProductStacked(
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &y)
{
    std::vector<GCOMPLEX(double)> result_complex;
    size_t elems = getNumberElements();

    _redBase.adjustSize(elems);

    _redBase.template iterateOverBulkStacked<LatLayout, HaloDepth, NStacks>(
            SpinorrealDotProduct<floatT, onDevice, LatLayout, HaloDepth, NStacks>(*this, y));

    _redBase.reduceStacked(result_complex, NStacks, getNumberLatticePoints(), true);
    std::vector<double> result;
    result.resize(result_complex.size());
    for (size_t i = 0; i < result.size(); i++){
        result[i] = result_complex[i].cREAL;
    }
    return result;
}

/// val = s_in * s_out
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
struct SpinorDotProduct{

    gVect3arrayAcc<floatT> spinorAccOut;
    gVect3arrayAcc<floatT> spinorAccIn;

    SpinorDotProduct(const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>& spinorIn,
            const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>& spinorOut)
        : spinorAccOut(spinorOut.getAccessor()), spinorAccIn(spinorIn.getAccessor()) {}

    __device__ __host__ GCOMPLEX(double)  operator()(gSiteStack& site){

        GCOMPLEX(double) ret =  spinorAccIn.template getElement<double>(site) *  spinorAccOut.template getElement<double>(site);
        return ret;
    }
};

/// val = S_in * S_out
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
GCOMPLEX(double) Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::dotProduct(
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &y)
{
    if (NStacks > 1){
        throw std::runtime_error(stdLogger.fatal("dotProduct only possible for non stacked spinors");
    }else{

        size_t elems = getNumberElements();

        _redBase.adjustSize(elems);

        _redBase.template iterateOverBulkStacked<LatLayout, HaloDepth, NStacks>(
                SpinorDotProduct<floatT, onDevice, LatLayout, HaloDepth, NStacks>(*this, y));

        GCOMPLEX(double) result = 0;
        _redBase.reduce(result, elems);
        return result;
    }
}

/// val = S_in * S_out
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
std::vector<GCOMPLEX(double)> Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::dotProductStacked(
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &y)
{
    size_t elems = getNumberElements();

    _redBase.adjustSize(elems);

    _redBase.template iterateOverBulkStacked<LatLayout, HaloDepth, NStacks>(
            SpinorDotProduct<floatT, onDevice, LatLayout, HaloDepth, NStacks>(*this, y));
    
    std::vector<GCOMPLEX(double)> result;

    _redBase.reduceStacked(result, NStacks, getNumberLatticePoints(),true);
    return result;

        
}

/// S_out *= val
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::operator*=(const GCOMPLEX(floatT) &y) {
    iterateOverFull(general_mult(*this, y));
}

template<class floatT, Layout LatLayout, size_t HaloDepth, typename const_T ,size_t NStacks>
struct SpinorPlusConstTTimesSpinor{

    gVect3arrayAcc<floatT> spinor1;
    gVect3arrayAcc<floatT> spinor2;
    const_T val;
    SpinorPlusConstTTimesSpinor(gVect3arrayAcc<floatT> spinor1,
                                  gVect3arrayAcc<floatT> spinor2,
                                  const_T val) : spinor1(spinor1), spinor2(spinor2), val(val){}

    __device__ __host__ gVect3<floatT> operator()(gSiteStack& site){
        gVect3<floatT> Stmp;
        Stmp = spinor1.getElement(site);
        Stmp += val(site) * spinor2.getElement(site);

        return Stmp;
    }
};

template<class floatT, Layout LatLayout, size_t HaloDepth, typename const_T ,size_t NStacks>
struct SpinorPlusConstTTimesSpinord{

    gVect3arrayAcc<floatT> spinor1;
    gVect3arrayAcc<floatT> spinor2;
    const_T val;
    SpinorPlusConstTTimesSpinord(gVect3arrayAcc<floatT> spinor1,
                                  gVect3arrayAcc<floatT> spinor2,
                                  const_T val) : spinor1(spinor1), spinor2(spinor2), val(val){}

    __device__ __host__ gVect3<floatT> operator()(gSiteStack& site){
        gVect3<floatT> Stmp;
        Stmp = spinor1.template getElement<double>(site);
        Stmp += val(site) * spinor2.template getElement<double>(site);

        return Stmp;
    }
};


template<class floatT, Layout LatLayout, size_t HaloDepth, typename const_T ,size_t NStacks>
struct SpinorPlusConstTTimesSpinorLoop{

    gVect3arrayAcc<floatT> spinor1;
    gVect3arrayAcc<floatT> spinor2;
    const_T val;

    typedef GIndexer<LatLayout, HaloDepth> GInd;

    SpinorPlusConstTTimesSpinorLoop(gVect3arrayAcc<floatT> spinor1,
                                  gVect3arrayAcc<floatT> spinor2,
                                  const_T val) : spinor1(spinor1), spinor2(spinor2), val(val) {}

    __host__ __device__ void initialize(__attribute__((unused)) gSite& site){
    }

    __device__ __host__ gVect3<floatT> operator()(gSite& site, size_t stack){
        gVect3<floatT> Stmp;
        gSiteStack siteStack = GInd::getSiteStack(site, stack);

        Stmp = spinor1.getElement(siteStack);
        Stmp += val(siteStack) * spinor2.getElement(siteStack);

        return Stmp;
    }
};

template<class floatT, Layout LatLayout, size_t HaloDepth ,size_t NStacks>
struct SpinorPlusFloatTimesSpinor{

    gVect3arrayAcc<floatT> spinor1;
    gVect3arrayAcc<floatT> spinor2;
    floatT val;
    SpinorPlusFloatTimesSpinor(gVect3arrayAcc<floatT> spinor1,
                                  gVect3arrayAcc<floatT> spinor2,
                                  floatT val) : spinor1(spinor1), spinor2(spinor2), val(val){}

    __device__ __host__ gVect3<floatT> operator()(gSiteStack& site){
        gVect3<floatT> Stmp;
        Stmp = spinor1.getElement(site);
        Stmp += val * spinor2.getElement(site);

        return Stmp;
    }
};

template<class floatT, Layout LatLayout, int HaloDepth, typename const_T, size_t NStacks>
struct StackTimesFloatPlusFloatTimesNoStack
{
    const gVect3arrayAcc<floatT> spinorIn1;
    const gVect3arrayAcc<floatT> spinorIn2;
    const_T _a;
    const_T _b;

    typedef GIndexer<LatLayout, HaloDepth> GInd;

    StackTimesFloatPlusFloatTimesNoStack(gVect3arrayAcc<floatT> spinorIn1,
            SimpleArray<floatT, NStacks> a,
            gVect3arrayAcc<floatT> spinorIn2,
            SimpleArray<floatT, NStacks> b) :
        spinorIn1(spinorIn1), spinorIn2(spinorIn2), _a(a), _b(b) {}


    __host__ __device__ gVect3<floatT> operator()(gSiteStack& siteStack){
        gSiteStack siteUnStack = GInd::getSiteStack(siteStack, 0);

        const gVect3<floatT> my_vec = spinorIn1.getElement(siteStack)*_a(siteStack) + spinorIn2.getElement(siteUnStack)*_b(siteStack);

        return my_vec;
    }
};

template<class floatT, Layout LatLayout, int HaloDepth, typename const_T, size_t NStacks>
struct StackTimesFloatPlusFloatTimesNoStackLoop
{
    gVect3arrayAcc<floatT> spinorIn1;
    gVect3arrayAcc<floatT> spinorIn2;
    const_T _a;
    const_T _b;

    typedef GIndexer<LatLayout, HaloDepth> GInd;

    StackTimesFloatPlusFloatTimesNoStackLoop(gVect3arrayAcc<floatT> spinorIn1,
            SimpleArray<floatT, NStacks> a,
            gVect3arrayAcc<floatT> spinorIn2,
            SimpleArray<floatT, NStacks> b) :
        spinorIn1(spinorIn1), spinorIn2(spinorIn2), _a(a), _b(b) {}

    __host__ __device__ void initialize(__attribute__((unused)) gSite& site){
    }


    __host__ __device__ gVect3<floatT> operator()(gSite& site, size_t stack){
        gSiteStack siteUnStack = GInd::getSiteStack(site, 0);
        gSiteStack siteStack = GInd::getSiteStack(site, stack);
        gVect3<floatT> my_vec;

        my_vec = spinorIn1.getElement(siteStack)*_a(siteStack) + spinorIn2.getElement(siteUnStack)*_b(siteStack);

        return my_vec;
    }
};


template<class floatT, Layout LatLayout, int HaloDepth, typename const_T, size_t NStacks>
struct StackTimesFloatPlusFloatTimesNoStackLoop_d
{
    gVect3arrayAcc<floatT> spinorIn1;
    gVect3arrayAcc<floatT> spinorIn2;
    const_T _a;
    const_T _b;

    typedef GIndexer<LatLayout, HaloDepth> GInd;

    StackTimesFloatPlusFloatTimesNoStackLoop_d(gVect3arrayAcc<floatT> spinorIn1,
            SimpleArray<double, NStacks> a,
            gVect3arrayAcc<floatT> spinorIn2,
            SimpleArray<double, NStacks> b) :
        spinorIn1(spinorIn1), spinorIn2(spinorIn2), _a(a), _b(b) {}

    __host__ __device__ void initialize(__attribute__((unused)) gSite& site){
    }


    __host__ __device__ gVect3<floatT> operator()(gSite& site, size_t stack){
        gSiteStack siteUnStack = GInd::getSiteStack(site, 0);
        gSiteStack siteStack = GInd::getSiteStack(site, stack);
        gVect3<double> my_vec;

        my_vec = spinorIn1.template getElement<double>(siteStack)*_a(siteStack) + spinorIn2.template getElement<double>(siteUnStack)*_b(siteStack);

        return my_vec;
    }
};

/// S_this_in += S2
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::operator+=(
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &S2) {
    iterateOverFull(general_add(*this, S2));
}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::operator-=(
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &S2) {
    iterateOverFull(general_subtract(*this, S2));
}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
template<typename const_T>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::axpyThis(const const_T &x,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &y) {

    iterateOverFull(SpinorPlusConstTTimesSpinor<floatT, LatLayout, HaloDepth, const_T, NStacks>(
                this->getAccessor(), y.getAccessor(), x));
}


template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::axpyThis(const floatT &x,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &y) {

    iterateOverFull(SpinorPlusFloatTimesSpinor<floatT, LatLayout, HaloDepth, NStacks>(
                this->getAccessor(), y.getAccessor(), x));
}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
template<size_t BlockSize, typename const_T>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::axpyThisB(const const_T &x,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &y) {

    iterateOverBulk<BlockSize>(SpinorPlusConstTTimesSpinor<floatT, LatLayout, HaloDepth, const_T, NStacks>(
                this->getAccessor(), y.getAccessor(), x));

}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
template<size_t BlockSize>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::axpyThisB(const floatT &x,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &y) {

    iterateOverBulk<BlockSize>(SpinorPlusFloatTimesSpinor<floatT, LatLayout, HaloDepth, NStacks>(
                this->getAccessor(), y.getAccessor(), x));

}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
template<size_t BlockSize, typename const_T>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::axpyThisLoop(const const_T &x,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &y, size_t stack_entry) {

    static_for<0, NStacks>::apply([&](auto i) {
         // change this to include a vector of bools, to see wether one needs to modify the vector, or if it has converged

            if(stack_entry >= i+1 ) {
                iterateOverBulkAtStack<i, BlockSize>(
                        SpinorPlusConstTTimesSpinor<floatT, LatLayout, HaloDepth, const_T, NStacks>(
                                this->getAccessor(), y.getAccessor(), x));
            }
            });
}


template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
template<size_t BlockSize, typename const_T>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::axpyThisLoopd(const const_T &x,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &y, size_t stack_entry) {

    static_for<0, NStacks>::apply([&](auto i) {
         // change this to include a vector of bools, to see wether one needs to modify the vector, or if it has converged

            if(stack_entry >= i+1 ) {
                iterateOverBulkAtStack<i, BlockSize>(
                        SpinorPlusConstTTimesSpinord<floatT, LatLayout, HaloDepth, const_T, NStacks>(
                                this->getAccessor(), y.getAccessor(), x));
            }
            });
}


template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
template<size_t BlockSize, typename const_T>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::axupbyThisB(const const_T &a, const const_T &b,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, 1> &y) {

    iterateOverBulk<BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, NStacks>(
                this->getAccessor(), b, y.getAccessor(), a));

}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
template<size_t BlockSize, typename const_T>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::axupbyThisLoopd(const const_T &a, const const_T &b,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, 1> &y, size_t stack_entry) {

    iterateOverBulkLoopStack<BlockSize>(StackTimesFloatPlusFloatTimesNoStackLoop_d<floatT, LatLayout, HaloDepth, const_T, NStacks>(
                this->getAccessor(), b, y.getAccessor(), a), stack_entry);


    //             this->getAccessor(), b, y.getAccessor(), a));

}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
template<size_t BlockSize, typename const_T>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::axupbyThisLoop(const const_T &a, const const_T &b,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, 1> &y, size_t stack_entry) {

    iterateOverBulkLoopStack<BlockSize>(StackTimesFloatPlusFloatTimesNoStackLoop<floatT, LatLayout, HaloDepth, const_T, NStacks>(
                this->getAccessor(), b, y.getAccessor(), a), stack_entry);


    // iterateOverBulkAtStack<0,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=2 )
    // iterateOverBulkAtStack<1,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=3 )
    // iterateOverBulkAtStack<2,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=4 )
    // iterateOverBulkAtStack<3,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=5 )
    // iterateOverBulkAtStack<4,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=6 )
    // iterateOverBulkAtStack<5,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=7 )
    // iterateOverBulkAtStack<6,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=8 )
    // iterateOverBulkAtStack<7,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=9 )
    // iterateOverBulkAtStack<8,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=10 )
    // iterateOverBulkAtStack<9,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=11 )
    // iterateOverBulkAtStack<10,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >=12 )
    // iterateOverBulkAtStack<11,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry >= 13)
    // iterateOverBulkAtStack<12,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));
    // if(stack_entry == 14)
    // iterateOverBulkAtStack<13,BlockSize>(StackTimesFloatPlusFloatTimesNoStack<floatT, LatLayout, HaloDepth, const_T, NStacks>(
    //             this->getAccessor(), b, y.getAccessor(), a));

}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
template<typename const_T, size_t BlockSize>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::xpayThisB(const const_T &x,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &y){

    iterateOverBulk<BlockSize>(SpinorPlusConstTTimesSpinor<floatT, LatLayout, HaloDepth, const_T, NStacks>(
                y.getAccessor(), this->getAccessor(), x));

}

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t NStacks>
template<typename const_T, size_t BlockSize>
void Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>::xpayThisBd(const const_T &x,
        const Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &y){

    iterateOverBulk<BlockSize>(SpinorPlusConstTTimesSpinord<floatT, LatLayout, HaloDepth, const_T, NStacks>(
                y.getAccessor(), this->getAccessor(), x));

}

#define SPINOR_INIT_PLHSN(floatT,LO,HALOSPIN,STACKS)\
template class Spinorfield<floatT,false,LO,HALOSPIN,STACKS>;\
template void Spinorfield<floatT,false,LO,HALOSPIN,STACKS>::axpyThis(const GCOMPLEX(floatT)&, const Spinorfield<floatT,false,LO,HALOSPIN,STACKS>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,STACKS>::axpyThisB(const GCOMPLEX(floatT)&, const Spinorfield<floatT,false,LO,HALOSPIN,STACKS>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,STACKS>::axpyThisB<64>(const floatT&, const Spinorfield<floatT,false,LO,HALOSPIN,STACKS>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,STACKS>::axpyThis(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,false,LO,HALOSPIN,STACKS>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,STACKS>::axpyThisB(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,false,LO,HALOSPIN,STACKS>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,STACKS>::axpyThisB<32>(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,false,LO,HALOSPIN,STACKS>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,STACKS>::axpyThisB<64>(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,false,LO,HALOSPIN,STACKS>&);\
 template void Spinorfield<floatT,false,LO,HALOSPIN,STACKS>::xpayThisB<SimpleArray<floatT,STACKS>,32>(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,false,LO,HALOSPIN,STACKS>&); \
template void Spinorfield<floatT,false,LO,HALOSPIN,STACKS>::xpayThisB(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,false,LO,HALOSPIN,STACKS>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,STACKS>::axupbyThisB(const SimpleArray<floatT, STACKS>&, const SimpleArray<floatT, STACKS>&, const Spinorfield<floatT,false,LO,HALOSPIN,1>&);\
template void Spinorfield<floatT,false,LO,HALOSPIN,STACKS>::axupbyThisB<32>(const SimpleArray<floatT, STACKS>&, const SimpleArray<floatT, STACKS>&, const Spinorfield<floatT,false,LO,HALOSPIN,1>&);\
template void Spinorfield<floatT, false, LO, HALOSPIN, STACKS>::axupbyThisLoop<32>(const SimpleArray<floatT, STACKS> &a, const SimpleArray<floatT, STACKS> &b, const Spinorfield<floatT, false, LO, HALOSPIN, 1> &y, size_t stack_entry);\
template void Spinorfield<floatT, false, LO, HALOSPIN, STACKS>::axpyThisLoop<32>(const SimpleArray<floatT, STACKS> &x, const Spinorfield<floatT, false, LO, HALOSPIN, STACKS> &y, size_t stack_entry);\


INIT_PLHSN(SPINOR_INIT_PLHSN)

#define SPINOR_INIT_PLHSN_HALF(floatT,LO,HALOSPIN,STACKS)\
template class Spinorfield<floatT,true,LO,HALOSPIN,STACKS>;\
template void Spinorfield<floatT,true,LO,HALOSPIN,STACKS>::axpyThis(const GCOMPLEX(floatT)&,const Spinorfield<floatT,true,LO,HALOSPIN,STACKS>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,STACKS>::axpyThisB(const GCOMPLEX(floatT)&,const Spinorfield<floatT,true,LO,HALOSPIN,STACKS>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,STACKS>::axpyThisB<64>(const floatT&,const Spinorfield<floatT,true,LO,HALOSPIN,STACKS>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,STACKS>::axpyThis(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,true,LO,HALOSPIN,STACKS>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,STACKS>::axpyThisB(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,true,LO,HALOSPIN,STACKS>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,STACKS>::axpyThisB<32>(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,true,LO,HALOSPIN,STACKS>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,STACKS>::axpyThisB<64>(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,true,LO,HALOSPIN,STACKS>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,STACKS>::xpayThisB<SimpleArray<floatT,STACKS>,32>(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,true,LO,HALOSPIN,STACKS>&); \
template void Spinorfield<floatT,true,LO,HALOSPIN,STACKS>::xpayThisB(const SimpleArray<floatT, STACKS>&,const Spinorfield<floatT,true,LO,HALOSPIN,STACKS>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,STACKS>::axupbyThisB(const SimpleArray<floatT, STACKS>&, const SimpleArray<floatT, STACKS>&, const Spinorfield<floatT,true,LO,HALOSPIN,1>&);\
template void Spinorfield<floatT,true,LO,HALOSPIN,STACKS>::axupbyThisB<32>(const SimpleArray<floatT, STACKS>&, const SimpleArray<floatT, STACKS>&, const Spinorfield<floatT,true,LO,HALOSPIN,1>&);\
template void Spinorfield<floatT, true, LO, HALOSPIN, STACKS>::axupbyThisLoop<32>(const SimpleArray<floatT, STACKS> &a, const SimpleArray<floatT, STACKS> &b, const Spinorfield<floatT, true, LO, HALOSPIN, 1> &y, size_t stack_entry);\
template void Spinorfield<floatT, true, LO, HALOSPIN, STACKS>::axupbyThisLoopd<32>(const SimpleArray<double, STACKS> &a, const SimpleArray<double, STACKS> &b, const Spinorfield<floatT, true, LO, HALOSPIN, 1> &y, size_t stack_entry); \
template void Spinorfield<floatT, true, LO, HALOSPIN, STACKS>::axpyThisLoop<32>(const SimpleArray<floatT, STACKS> &x, const Spinorfield<floatT, true, LO, HALOSPIN, STACKS> &y, size_t stack_entry);\
template void Spinorfield<floatT,true,LO,HALOSPIN,STACKS>::xpayThisBd<SimpleArray<double,STACKS>,32>(const SimpleArray<double, STACKS>&,const Spinorfield<floatT,true,LO,HALOSPIN,STACKS>&); \
template void Spinorfield<floatT, true, LO, HALOSPIN, STACKS>::axpyThisLoopd<32>(const SimpleArray<double, STACKS> &x, const Spinorfield<floatT, true, LO, HALOSPIN, STACKS> &y, size_t stack_entry);\

INIT_PLHSN_HALF(SPINOR_INIT_PLHSN_HALF)
