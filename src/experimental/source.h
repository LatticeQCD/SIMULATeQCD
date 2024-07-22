#pragma once

#include "../simulateqcd.h"
#include "fullSpinor.h"

template<Layout LatLayout, size_t HaloDepth>
struct ReadIndex {
    inline __host__ __device__ gSite operator()(const dim3& blockDim, const uint3& blockIdx, const uint3& threadIdx) {
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        typedef GIndexer<LatLayout, HaloDepth> GInd;
        gSite site = GInd::getSite(i);
        return site;
    }
};

template<class floatT, size_t HaloDepth>
struct MakePointSource12{

    // accessor to access the spinor field
    Vect12ArrayAcc<floatT> _spinorIn;

    size_t _posx, _posy, _posz, _post;

    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    MakePointSource12(Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinorIn,
                      size_t posx, size_t posy,size_t posz,size_t post)
                : _spinorIn(spinorIn.getAccessor()),
                  _posx(posx), _posy(posy), _posz(posz), _post(post)
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ void operator()(gSite site) {

        for (size_t stack = 0; stack < 12; stack++) {
            Vect12<floatT> tmp(0.0);


            // get global coordinates and set the source to idendity
            sitexyzt coord = GIndexer<All, HaloDepth>::getLatData().globalPos(site.coord);
            if(coord[0] == _posx && coord[1] == _posy && coord[2] == _posz && coord[3] == _post ){
                tmp.data[stack] = 1.0;
            }
            const gSiteStack writeSite = GInd::getSiteStack(site,stack);
            _spinorIn.setElement(writeSite,tmp);

        }
    }
};



template<class floatT,Layout layoutRHS, size_t HaloDepth, size_t NStacks>
struct CopyAllFromHalf{

    //Gauge accessor to access the gauge field
    Vect12ArrayAcc<floatT> _SpinorAll;
    Vect12ArrayAcc<floatT> _Spinor;
    int _offset;

    typedef GIndexer<All, HaloDepth > GIndAll;
    typedef GIndexer<layoutRHS, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    CopyAllFromHalf(Spinorfield<floatT, true, All, HaloDepth, 12, 12> &spinorInAll, Spinorfield<floatT, true, layoutRHS, HaloDepth, 12, NStacks> &spinorIn, int offset)
                : _SpinorAll(spinorInAll.getAccessor()), _Spinor(spinorIn.getAccessor()), _offset(offset)
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ void operator()(gSite site) {
    
    for (size_t stack = 0; stack < NStacks; stack++) {
        // take from even or odd and put into all
        Vect12<floatT> tmp = _Spinor.getElement(GInd::getSiteStack(site,stack));
       _SpinorAll.setElement(GIndAll::getSiteStack(GInd::template convertSite<All, HaloDepth>(site),stack+_offset),tmp );
      }
    }
};

template<class floatT,Layout layoutRHS, size_t HaloDepth, size_t NStacks>
struct CopyHalfFromAll{

    //Gauge accessor to access the gauge field
    Vect12ArrayAcc<floatT> _SpinorAll;
    Vect12ArrayAcc<floatT> _Spinor;
    int _offset;

    typedef GIndexer<All, HaloDepth > GIndAll;
    typedef GIndexer<layoutRHS, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    CopyHalfFromAll(Spinorfield<floatT, true, layoutRHS, HaloDepth, 12, NStacks> &spinorIn,
                    Spinorfield<floatT, true, All, HaloDepth, 12, 12> &spinorInAll, int offset)
                : _SpinorAll(spinorInAll.getAccessor()), _Spinor(spinorIn.getAccessor()), _offset(offset)
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ void operator()(gSite site) {

    for (size_t stack = 0; stack < NStacks; stack++) {
        // take from all and put into even or odd
        Vect12<floatT> tmp = _SpinorAll.getElement(GIndAll::getSiteStack(GInd::template convertSite<All, HaloDepth>(site),stack+_offset));
       _Spinor.setElement(GInd::getSiteStack(site,stack),tmp );
      }
    }
};

class Source {
private:

public:
    Source()  {}

    template<typename floatT, size_t HaloDepth>
    void makePointSource(Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinorIn,
                      size_t posx, size_t posy,size_t posz,size_t post);

    template<typename floatT, size_t HaloDepth, size_t NStacks>
    void copyHalfFromAll(SpinorfieldAll<floatT, true,      HaloDepth, 12, NStacks> &spinorIn,
                         Spinorfield<floatT   , true, All, HaloDepth, 12, 12     > &spinorInAll,
                         int offset);

    template<typename floatT, size_t HaloDepth, size_t NStacks>
    void copyAllFromHalf(Spinorfield<floatT   , true, All      , HaloDepth, 12, 12     > &spinorInAll,
                         SpinorfieldAll<floatT, true,            HaloDepth, 12, NStacks> &spinorIn,
                         int offset);

};


