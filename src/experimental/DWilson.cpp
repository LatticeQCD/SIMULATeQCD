#include "DWilson.h"

#ifdef USE_HIP_AMD
#define BLOCKSIZE 64
#else
#define BLOCKSIZE 32
#endif


template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void DWilson<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::applyMdaggM(SpinorRHS_t& spinorOut, const SpinorRHS_t& spinorIn, bool update){

    //apply Gamma5 * the dirac wilson operator since CG requires hermitian operator
    //spinorOut.template iterateOverBulk<BLOCKSIZE>(gamma5DiracWilson<floatT,HaloDepthGauge,HaloDepthSpin,NStacks>(_gauge, spinorIn,_mass,_csw));
    if(NStacks > 100){
//        spinorOut.template iterateOverBulk<BLOCKSIZE>(gamma5DiracWilson<floatT,HaloDepthGauge,HaloDepthSpin,NStacks>(_gauge, spinorIn,_mass,_csw));
          typedef GIndexer<All, HaloDepthGauge> GInd;
          size_t _elems = GInd::getLatData().vol4;
          CalcGSite<All, HaloDepthSpin> calcGSite;
          iterateFunctorNoReturn<onDevice,BLOCKSIZE>(gamma5DiracWilsonStack<floatT,HaloDepthGauge,HaloDepthSpin,NStacks>(_gauge,spinorOut, spinorIn,_mass,_csw),calcGSite,_elems);

    }
    else{
        spinorOut.template iterateOverBulk<BLOCKSIZE>(gamma5DiracWilson<floatT,HaloDepthGauge,HaloDepthSpin,NStacks>(_gauge, spinorIn,_mass,_csw));
    }
//    _tmpSpin = spinorOut;
//    spinorOut.template iterateOverBulk<BLOCKSIZE>(gamma5DiracWilson<floatT,HaloDepthGauge,HaloDepthSpin,NStacks>(_gauge, _tmpSpin));
//    spinorOut.template iterateOverBulk<BLOCKSIZE>(ttest<floatT,HaloDepthGauge,HaloDepthSpin,NStacks>(_gauge, spinorIn));

        /*
        ReadIndexSpatial<HaloDepth> calcReadIndexSpatial;

        size_t _elems = GInd::getLatData().globvol4;
        CalcGSite<All, HaloDepthSpin> calcGSite;
        iterateFunctorNoReturn<onDevice>(Print<floatT,onDevice,HaloDepth>(_gauge, _wlineGPU), calcReadIndexSpatial, _elems);
        iterateFunctorNoReturn<onDevice>(Print<floatT,onDevice,HaloDepth>(_gauge, _wlineGPU), calcReadIndexSpatial, _elems);
        */

    if(update)
        spinorOut.updateAll();
}

/// val = S_in * S_in but only at spatial time t
template<class floatT, bool onDevice,size_t HaloDepthGauge, size_t HaloDepth, size_t NStacks>
COMPLEX(double) DWilsonInverse<floatT,onDevice,HaloDepthGauge,HaloDepth,NStacks>::Correlator(int t,
        const Spinorfield<floatT, onDevice, All, HaloDepth, 12, NStacks> & spinorIn) {
//    if (NStacks > 1){
//        throw std::runtime_error(stdLogger.fatal("Correlator currently only possible for non stacked spinors"));
//    }else{

        COMPLEX(double) result = 0;

        size_t elems_ = GInd::getLatData().vol3;

        _redBase.adjustSize(elems_);

        _redBase.template iterateOverSpatialBulk<All, HaloDepth>(
                Contract<floatT, HaloDepth,NStacks>(t, spinorIn));

        _redBase.reduce(result, elems_);
        return result;
//    }
}


template<class floatT, bool onDevice,size_t HaloDepthGauge, size_t HaloDepth, size_t NStacks>
void DWilsonInverse<floatT,onDevice,HaloDepthGauge,HaloDepth,NStacks>::gamma5MultVec(
                       Spinorfield<floatT, onDevice,All, HaloDepth, 12, NStacks> &spinorOut,
                       Spinorfield<floatT, onDevice,All, HaloDepth, 12, NStacks> &spinorIn ){

  spinorOut.template iterateOverBulk<BLOCKSIZE>(gamma5<floatT,All,HaloDepth,NStacks>(spinorIn));

}


/////////////////////////////

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void DWilsonEvenOdd<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::applyMdaggM(SpinorRHS_t& spinorOut, const SpinorRHS_t& spinorIn, bool update){

/*
    _tmpSpin.template iterateOverBulk<32>(DiracWilsonEvenOdd<floatT,Odd,Even,HaloDepthGauge,HaloDepthSpin,NStacks,false>(_gauge, spinorIn,_mass,_csw));
    _tmpSpin.updateAll();
    spinorOut.template iterateOverBulk<32>(DiracWilsonEvenOdd<floatT,Even,Odd,HaloDepthGauge,HaloDepthSpin,NStacks>(_gauge, _tmpSpin,_mass,_csw));

    _tmpSpinEven.template iterateOverBulk<32>(gamma5<floatT,Even,HaloDepthSpin,NStacks>(spinorIn));

    // remember to multiply by gamma5
    spinorOut = (_mass*_tmpSpinEven)-((1.0/_mass)*spinorOut);

*/

    _tmpSpin.template iterateOverBulk<32>(DiracWilsonEvenOdd<floatT,Odd,Even,HaloDepthGauge,HaloDepthSpin,NStacks,false>(_gauge, spinorIn,_mass,_csw));
    dslashDiagonalOdd( _tmpSpin, _tmpSpin,true);
    _tmpSpin.updateAll();
    spinorOut.template iterateOverBulk<32>(DiracWilsonEvenOdd<floatT,Even,Odd,HaloDepthGauge,HaloDepthSpin,NStacks>(_gauge, _tmpSpin,_mass,_csw));

    dslashDiagonalEven( _tmpSpinEven,spinorIn,false);
    _tmpSpinEven.template iterateOverBulk<32>(gamma5<floatT,Even,HaloDepthSpin,NStacks>(_tmpSpinEven));

    spinorOut = _tmpSpinEven-spinorOut;
    

    if(update)
        spinorOut.updateAll();
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void DWilsonEvenOdd<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::calcFmunu(){

    typedef GIndexer<All, HaloDepthGauge> GInd;
    size_t _elems = GInd::getLatData().vol4;
    CalcGSite<All, HaloDepthSpin> calcGSite;
    iterateFunctorNoReturn<onDevice>(preCalcFmunu<floatT,HaloDepthGauge>(_gauge,FmunuUpper,FmunuLower,FmunuInvUpper,FmunuInvLower, _mass,_csw), calcGSite, _elems);
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void DWilsonEvenOdd<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::dslashDiagonalOdd(Spinorfield<floatT, true, Odd, HaloDepthSpin, 12, NStacks> & spinorOut, 
                                                                                                         const Spinorfield<floatT, true, Odd, HaloDepthSpin, 12, NStacks> & spinorIn, bool inverse){

    if(inverse){
        spinorOut.template iterateOverBulk<32>(DiracWilsonEvenEven2<floatT,Odd,HaloDepthGauge,HaloDepthSpin,NStacks>(spinorIn,FmunuInvUpper,FmunuInvLower));
    }
    else{
        spinorOut.template iterateOverBulk<32>(DiracWilsonEvenEven2<floatT,Odd,HaloDepthGauge,HaloDepthSpin,NStacks>(spinorIn,FmunuUpper,FmunuLower));
    }
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void DWilsonEvenOdd<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::dslashDiagonalEven(Spinorfield<floatT, true, Even, HaloDepthSpin, 12, NStacks> & spinorOut,
                                                                                                          const Spinorfield<floatT, true, Even, HaloDepthSpin, 12, NStacks> & spinorIn, bool inverse){

    if(inverse){
        spinorOut.template iterateOverBulk<32>(DiracWilsonEvenEven2<floatT,Even,HaloDepthGauge,HaloDepthSpin,NStacks>(spinorIn,FmunuInvUpper,FmunuInvLower));
    }
    else{
        spinorOut.template iterateOverBulk<32>(DiracWilsonEvenEven2<floatT,Even,HaloDepthGauge,HaloDepthSpin,NStacks>(spinorIn,FmunuUpper,FmunuLower));
    }
}



template class DWilson<double,true,All,2,2,1>;
/*template class DWilson<double,true,All,2,2,2>;
template class DWilson<double,true,All,2,2,3>;
template class DWilson<double,true,All,2,2,4>;
template class DWilson<double,true,All,2,2,5>;
template class DWilson<double,true,All,2,2,6>;
*/
template class DWilson<double,true,All,2,2,12>;

template class DWilsonInverse<double,true,2,2,1>;
/*template class DWilsonInverse<double,true,2,2,2>;
template class DWilsonInverse<double,true,2,2,3>;
template class DWilsonInverse<double,true,2,2,4>;
template class DWilsonInverse<double,true,2,2,5>;
template class DWilsonInverse<double,true,2,2,6>;
*/
template class DWilsonInverse<double,true,2,2,12>;

template class DWilsonEvenOdd<double,true,Even,2,2,1>;
template class DWilsonInverseShurComplement<double,true,2,2,1>;

template class DWilsonEvenOdd<double,true,Even,2,2,12>;
template class DWilsonInverseShurComplement<double,true,2,2,12>;

template class DWilsonEvenOdd<double,true,Even,2,2,4>;
template class DWilsonInverseShurComplement<double,true,2,2,4>;



//template class DWilsonEvenOdd<double,true,Even,2,2,1>::dslashDiagonal<Odd>;
//template class DWilsonEvenOdd<double,true,Even,2,2,1>::dslashDiagonal<Even>;

/*
//! explicit template instantiations
#define DSLASH_INIT(floatT,LO, HaloDepth, HaloDepthSpin, NStacks) \
  template class DWilson<floatT,true,All,HaloDepth,HaloDepthSpin,NStacks>;\
  template class DWilsonEvenOdd<floatT,true,Even,HaloDepth,HaloDepthSpin,NStacks>;\

INIT_PHHSN(DSLASH_INIT)

#define DSLASHINV_INIT(floatT,LO, HaloDepth, HaloDepthSpin, NStacks) \
  template class DWilsonInverse<floatT,true,HaloDepth,HaloDepthSpin,NStacks>;\
  template class DWilsonInverseShurComplement<floatT,true,HaloDepth,HaloDepthSpin,NStacks>;\
INIT_PHHSN(DSLASHINV_INIT)
*/

