#include "source.h"

#ifdef USE_HIP_AMD
#define BLOCKSIZE 64
#else
#define BLOCKSIZE 32
#endif

template<typename floatT, size_t HaloDepth>
void Source::makePointSource(Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinorIn,
                      size_t posx, size_t posy,size_t posz,size_t post){

    typedef GIndexer<All, HaloDepth> GInd;
    size_t _elems = GInd::getLatData().vol4;
    ReadIndex<All,HaloDepth> index;

    iterateFunctorNoReturn<true,BLOCKSIZE>(MakePointSource12<floatT,HaloDepth>( spinorIn, posx,posy,posz,post),index,_elems);

}


template<typename floatT, size_t HaloDepth, size_t NStacks>
void Source::copyHalfFromAll(SpinorfieldAll<floatT, true,   HaloDepth, 12, NStacks> & spinorIn,
                             Spinorfield<floatT, true, All, HaloDepth, 12, 12     > & spinorInAll,
                             int offset){
    typedef GIndexer<All, HaloDepth> GInd;
    size_t _elems = GInd::getLatData().vol4/2;

    ReadIndex<Even,HaloDepth> indexEven;
    iterateFunctorNoReturn<true,BLOCKSIZE>(CopyHalfFromAll<floatT,Even,HaloDepth,NStacks>(spinorIn.even, spinorInAll,offset),indexEven,_elems);
 
    ReadIndex<Odd,HaloDepth> indexOdd;
    iterateFunctorNoReturn<true,BLOCKSIZE>(CopyHalfFromAll<floatT,Odd,HaloDepth,NStacks>(spinorIn.odd, spinorInAll,offset),indexOdd,_elems);

}

template<typename floatT, size_t HaloDepth, size_t NStacks>
void Source::copyAllFromHalf(Spinorfield<floatT, true, All, HaloDepth, 12, 12     > & spinorInAll,
                             SpinorfieldAll<floatT, true,   HaloDepth, 12, NStacks> & spinorIn,
                             int offset){
    typedef GIndexer<All, HaloDepth> GInd;
    size_t _elems = GInd::getLatData().vol4/2;

    ReadIndex<Even,HaloDepth> indexEven;
    iterateFunctorNoReturn<true,BLOCKSIZE>(CopyAllFromHalf<floatT,Even,HaloDepth,NStacks>(spinorInAll,spinorIn.even,offset),indexEven,_elems);

    ReadIndex<Odd,HaloDepth> indexOdd;
    iterateFunctorNoReturn<true,BLOCKSIZE>(CopyAllFromHalf<floatT,Odd,HaloDepth,NStacks>(spinorInAll,spinorIn.odd,offset),indexOdd,_elems);

}


template<class floatT,Layout LatLayout, size_t HaloDepthSpin,size_t NStacks,int gammamu>
void Source::gammaMu(Spinorfield<floatT, true,LatLayout , HaloDepthSpin, 12, NStacks> & spinorIn){

     spinorIn.template iterateOverBulk<BLOCKSIZE>(gamma_mu<floatT,LatLayout,HaloDepthSpin,NStacks,gammamu>(spinorIn));

};

template<class floatT,Layout LatLayout, size_t HaloDepthSpin,int gammamu>
void Source::gammaMuRight(Spinorfield<floatT, true,LatLayout , HaloDepthSpin, 12, 12> & spinorIn){

    typedef GIndexer<All, HaloDepthSpin> GInd;
    size_t _elems = GInd::getLatData().vol4;

    ReadIndex<All,HaloDepthSpin> index;
    iterateFunctorNoReturn<true,BLOCKSIZE>(gamma_mu_right<floatT,LatLayout,HaloDepthSpin,gammamu>(spinorIn),index,_elems);


};


template<class floatT,size_t HaloDepthGauge, size_t HaloDepthSpin>
void Source::smearSource(Gaugefield<floatT,true,HaloDepthGauge,R18> &gauge,
                         Spinorfield<floatT, true,All , HaloDepthSpin, 12, 12> & spinorOut,
                         Spinorfield<floatT, true,All , HaloDepthSpin, 12, 12> & spinorIn,
                         floatT lambda, int steps){
     for(int i =0;i < steps;i++){
         spinorOut.template iterateOverBulk<BLOCKSIZE>(SmearSource<floatT,HaloDepthGauge,HaloDepthSpin>(gauge,spinorIn,lambda*lambda/(4.0*steps)));
         spinorIn = spinorOut;
         spinorIn.updateAll();
     }
     spinorOut = spinorIn;
     spinorOut.updateAll();

};

template void Source::smearSource(Gaugefield<double,true,2,R18> &gauge,
                         Spinorfield<double, true,All ,2, 12, 12> & spinorOut,
                         Spinorfield<double, true,All ,2, 12, 12> & spinorIn,
                         double lambda, int steps);

template void Source::gammaMu<double,All,2, 12, 0>(Spinorfield<double,true,All,2, 12, 12> &spinorIn);
template void Source::gammaMu<double,All,2, 12, 1>(Spinorfield<double,true,All,2, 12, 12> &spinorIn);
template void Source::gammaMu<double,All,2, 12, 2>(Spinorfield<double,true,All,2, 12, 12> &spinorIn);
template void Source::gammaMu<double,All,2, 12, 3>(Spinorfield<double,true,All,2, 12, 12> &spinorIn);
template void Source::gammaMu<double,All,2, 12, 5>(Spinorfield<double,true,All,2, 12, 12> &spinorIn);



template void Source::gammaMuRight<double,All,2, 0>(Spinorfield<double,true,All,2, 12, 12> &spinorIn);
template void Source::gammaMuRight<double,All,2, 1>(Spinorfield<double,true,All,2, 12, 12> &spinorIn);
template void Source::gammaMuRight<double,All,2, 2>(Spinorfield<double,true,All,2, 12, 12> &spinorIn);
template void Source::gammaMuRight<double,All,2, 3>(Spinorfield<double,true,All,2, 12, 12> &spinorIn);
template void Source::gammaMuRight<double,All,2, 5>(Spinorfield<double,true,All,2, 12, 12> &spinorIn);


template void Source::makePointSource<double,2>(Spinorfield<double, true, All, 2, 12, 12> & spinorIn,
                      size_t posx, size_t posy,size_t posz,size_t post);


template void Source::copyHalfFromAll<double,2,12>(SpinorfieldAll<double, true, 2, 12, 12> &spinorIn,
                             Spinorfield<double, true, All, 2, 12, 12     > &spinorInAll,
                             int offset);
template void Source::copyAllFromHalf<double,2,12>(Spinorfield<double, true, All, 2, 12, 12     > &spinorInAll,
                             SpinorfieldAll<double, true, 2, 12, 12> &spinorIn,
                             int offset);



template void Source::copyHalfFromAll<double,2,1>(SpinorfieldAll<double, true, 2, 12, 1> &spinorIn,
                             Spinorfield<double, true, All, 2, 12, 12     > &spinorInAll,
                             int offset);
template void Source::copyAllFromHalf<double,2,1>(Spinorfield<double, true, All, 2, 12, 12     > &spinorInAll,
                             SpinorfieldAll<double, true, 2, 12, 1> &spinorIn,
                             int offset);


template void Source::copyHalfFromAll<double,2,4>(SpinorfieldAll<double, true, 2, 12, 4> &spinorIn,
                             Spinorfield<double, true, All, 2, 12, 12     > &spinorInAll,
                             int offset);
template void Source::copyAllFromHalf<double,2,4>(Spinorfield<double, true, All, 2, 12, 12     > &spinorInAll,
                             SpinorfieldAll<double, true, 2, 12, 4> &spinorIn,
                             int offset);


