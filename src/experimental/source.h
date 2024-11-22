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


template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin>
struct SmearSource{

    // accessor to access the spinor field
    SpinorColorAcc<floatT> _spinorIn;
    SU3Accessor<floatT> _SU3Accessor;

    floatT _smear;

    typedef GIndexer<All, HaloDepthSpin > GInd;
    //Constructor to initialize all necessary members.
    SmearSource(Gaugefield<floatT,true,HaloDepthGauge,R18> &gauge,
                Spinorfield<floatT, true, All, HaloDepthSpin, 12, 12> & spinorIn,floatT smear)
                : _SU3Accessor(gauge.getAccessor()),
                  _spinorIn(spinorIn.getAccessor()),
                  _smear(smear)
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ Vect12<floatT> operator()(gSiteStack site) {

        ColorVect<floatT> tmp = (1.0-3*2.0*_smear)*_spinorIn.getColorVect(site);

        for(int dir=0; dir < 3; dir ++){
            tmp = tmp + _smear*_SU3Accessor.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site,dir)))*
                               _spinorIn.getColorVect(GInd::site_up(site, dir));
            tmp = tmp + _smear*_SU3Accessor.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, dir),dir)))*
                              _spinorIn.getColorVect(GInd::site_dn(site, dir));
        }

        return convertColorVectToVect12(tmp);
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

/*

template<class floatT,size_t HaloDepth>
__global__ void copySpinorToContainer(MemoryAccessor _redBase, Vect12ArrayAcc<floatT> _SpinorIn, const size_t size,int spincolor1, int spincolor2,int lx,int ly,int lz,const int xtopo,const int ytopo,const int ztopo) {

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size) {
        return;
    }

    typedef GIndexer<All,HaloDepth> GInd;

    int ix, iy, iz, it;
    it = 0;

    int  tmp;

    divmod(site, GInd::getLatData().vol2, iz, tmp);
    divmod(tmp,  GInd::getLatData().vol1, iy, ix);

    Vect12<floatT> tmp12 = _SpinorIn.getElement(GInd::getSiteStack(GInd::getSite((size_t)ix,(size_t)iy, (size_t)iz, (size_t)(it)) , spincolor2));

    ix += xtopo*GInd::getLatData().lx;
    iy += ytopo*GInd::getLatData().ly;
    iz += ztopo*GInd::getLatData().lz;

    _redBase.setValue<COMPLEX(floatT)>(ix+lx*(iy+ly*iz),tmp12.data[spincolor1]);
//    printf("%d %f \n" ,(int)(ix+GInd::getLatData().lx*(iy+GInd::getLatData().ly*iz)), tmp12.data[spincolor1].cREAL );
}



template<class floatT,size_t HaloDepth>
__global__ void copyContainerToSpinor(Vect12ArrayAcc<floatT> _SpinorOut,LatticeContainerAccessor _redBase, const size_t size,int spincolor1, int spincolor2,int lx,int ly,int lz,const int xtopo,const int ytopo,const int ztopo) {

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size) {
        return;
    }

    typedef GIndexer<All,HaloDepth> GInd;

    int ix, iy, iz, it;
    it = 0;

    int  tmp;

    divmod(site, GInd::getLatData().vol2, iz, tmp);
    divmod(tmp,  GInd::getLatData().vol1, iy, ix);

    Vect12<floatT> tmp12 = _SpinorIn.getElement(GInd::getSiteStack(GInd::getSite((size_t)ix,(size_t)iy, (size_t)iz, (size_t)(it)) , spincolor2));

    ix += xtopo*GInd::getLatData().lx;
    iy += ytopo*GInd::getLatData().ly;
    iz += ztopo*GInd::getLatData().lz;

    _redBase.setValue<COMPLEX(floatT)>(ix+lx*(iy+ly*iz),tmp12.data[spincolor1]);
//    printf("%d %f \n" ,(int)(ix+GInd::getLatData().lx*(iy+GInd::getLatData().ly*iz)), tmp12.data[spincolor1].cREAL );
}

*/

template<class floatT, size_t HaloDepth,size_t NStacks>
struct SumXYZ_TrM{
    using SpinorRHS_t = Spinorfield<floatT, true, All, HaloDepth, 12, NStacks>;


    SpinorColorAcc<floatT> _spinorIn;
    int _t;

    // adding spinor gives compile error
    typedef GIndexer<All, HaloDepth > GInd;
    SumXYZ_TrM(int t, const SpinorRHS_t &spinorIn)
          :  _t(t), _spinorIn(spinorIn.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ COMPLEX(double)  operator()(gSite site){

        sitexyzt coords=site.coord;
        gSite siteT = GInd::getSite(coords.x,coords.y, coords.z, _t);

        COMPLEX(double) temp(0.0,0.0);
        for (size_t stack = 0; stack < NStacks; stack++) {
            temp  = temp + _spinorIn.template getElement<double>(GInd::getSiteStack(siteT,stack)).data[stack];
        }

        return temp;
    }
};

template<class floatT, size_t HaloDepthGauge,size_t HaloDepthSpin,size_t NStacks>
struct ShiftSource{

     SU3Accessor<floatT> _SU3Accessor;
     SpinorColorAcc<floatT> _spinorIn;
 
     typedef GIndexer<All, HaloDepthSpin > GInd;
     ShiftSource( Gaugefield<floatT,true,HaloDepthGauge,R18> &gauge,
                 Spinorfield<floatT, true, All, HaloDepthSpin, 12, NStacks> &spinorIn):
                 _SU3Accessor(gauge.getAccessor()), _spinorIn(spinorIn.getAccessor())
     {}

      __device__ __host__ Vect12<floatT> operator()(gSiteStack site){
 
          //ColorVect<floatT> out = _SU3Accessor.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, 3),3)))*_spinorIn.getColorVect(GInd::site_dn(site, 3));
          ColorVect<floatT> out =_spinorIn.getColorVect(GInd::site_dn(site, 3));

          //ColorVect<floatT> tmp = GammaTMultVec(out);
          //out = (-0.5)*tmp+(-0.5)*out;     

          return convertColorVectToVect12(out);
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

    template<class floatT,Layout LatLayout, size_t HaloDepthSpin,size_t NStacks,int gammamu>
    void gammaMu(Spinorfield<floatT, true ,LatLayout, HaloDepthSpin, 12, NStacks> & spinorIn);

    template<class floatT,Layout LatLayout, size_t HaloDepthSpin, int gammamu>
    void gammaMuRight(Spinorfield<floatT, true ,LatLayout, HaloDepthSpin, 12, 12> & spinorIn);

    template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin>
    void smearSource(Gaugefield<floatT,true,HaloDepthGauge,R18> &gauge,
                     Spinorfield<floatT, true ,All, HaloDepthSpin, 12, 12> & spinorOut,
                     Spinorfield<floatT, true ,All, HaloDepthSpin, 12, 12> & spinorIn,
                     floatT lambda, int steps);

    template<class floatT, size_t HaloDepthGauge,size_t HaloDepthSpin,size_t NStacks>
    void shiftSource1t(Gaugefield<floatT,true,HaloDepthGauge,R18> &gauge,
                         Spinorfield<floatT, true ,All, HaloDepthSpin, 12, 12> & spinorOut,
                         Spinorfield<floatT, true ,All, HaloDepthSpin, 12, 12> & spinorIn);


    template<class floatT,Layout LatLayout, size_t HaloDepthSpin>
    void daggerSource(Spinorfield<floatT, true ,LatLayout, HaloDepthSpin, 12, 12> & spinorIn);

    template<class floatT,Layout LatLayout, size_t HaloDepthSpin>
    void conjugateSource(Spinorfield<floatT, true ,LatLayout, HaloDepthSpin, 12, 12> & spinorIn);

};


template<class floatT,Layout LatLayout, size_t HaloDepthSpin,size_t NStacks,int gammamu>
struct gamma_mu{

    //input
    SpinorColorAcc<floatT> _SpinorColorAccessor;

    typedef GIndexer<LatLayout, HaloDepthSpin > GInd;
    //Constructor to initialize all necessary members.
    gamma_mu(Spinorfield<floatT, true,LatLayout, HaloDepthSpin, 12, NStacks> &spinorIn)
                : _SpinorColorAccessor(spinorIn.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    //__device__ __host__ Vect12<floatT> operator()(gSite site) {
    __device__ __host__ Vect12<floatT> operator()(gSiteStack site) {

        ColorVect<floatT> outSC = _SpinorColorAccessor.getColorVect(site);
        if(gammamu == 0){
            outSC = GammaXMultVec(outSC);
        }
        if(gammamu == 1){
            outSC = GammaYMultVec(outSC);
        }
        if(gammamu == 2){
            outSC = GammaZMultVec(outSC);
        }
        if(gammamu == 3){
            outSC = GammaTMultVec(outSC);
        }
        if(gammamu == 5){
            outSC = Gamma5MultVec(outSC);
        }


        return convertColorVectToVect12(outSC);
    }
};

template<class floatT,Layout LatLayout, size_t HaloDepthSpin,int gammamu>
struct gamma_mu_right{

    //input
    Vect12ArrayAcc<floatT> _spinorIn;

    typedef GIndexer<LatLayout, HaloDepthSpin > GInd;
    //Constructor to initialize all necessary members.
    gamma_mu_right(Spinorfield<floatT, true,LatLayout, HaloDepthSpin, 12, 12> &spinorIn)
                : _spinorIn(spinorIn.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    //__device__ __host__ Vect12<floatT> operator()(gSite site) {
    __device__ __host__ void operator()(gSite site) {

        Vect12<floatT> tmp[12];
        Vect12<floatT> tmp2[12];
        for (size_t stack = 0; stack < 12; stack++) {
            tmp[stack] = _spinorIn.getElement(GInd::getSiteStack(site,stack));
        }

        for (size_t stack = 0; stack < 12; stack++) {
            for (size_t stack2 = 0; stack2 < 12; stack2++) {
                tmp2[stack2].data[stack] = conj(tmp[stack].data[stack2]);
            }
        }

        for (size_t stack = 0; stack < 12; stack++) {

            ColorVect<floatT> tmp3;
            if(gammamu == 0){
                tmp3 = GammaXMultVec(convertVect12ToColorVect(tmp2[stack]));
            }
            if(gammamu == 1){
                tmp3 = GammaYMultVec(convertVect12ToColorVect(tmp2[stack]));
            }
            if(gammamu == 2){
                tmp3 = GammaZMultVec(convertVect12ToColorVect(tmp2[stack]));
            }
            if(gammamu == 3){
                tmp3 = GammaTMultVec(convertVect12ToColorVect(tmp2[stack]));
            }
            if(gammamu == 5){
                tmp3 = Gamma5MultVec(convertVect12ToColorVect(tmp2[stack]));
            }
           
           tmp2[stack] = convertColorVectToVect12(tmp3);
        }

        for (size_t stack = 0; stack < 12; stack++) {
            for (size_t stack2 = 0; stack2 < 12; stack2++) {
                tmp[stack2].data[stack] = conj(tmp2[stack].data[stack2]);
            }
        }

        for (size_t stack = 0; stack < 12; stack++) {
            _spinorIn.setElement(GInd::getSiteStack(site,stack),tmp[stack]);
        }



    }
};

template<class floatT,Layout LatLayout, size_t HaloDepthSpin>
struct ConjugateSource{

    //input
    Vect12ArrayAcc<floatT> _spinorIn;

    typedef GIndexer<LatLayout, HaloDepthSpin > GInd;
    //Constructor to initialize all necessary members.
    ConjugateSource(Spinorfield<floatT, true,LatLayout, HaloDepthSpin, 12, 12> &spinorIn)
                : _spinorIn(spinorIn.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    //__device__ __host__ Vect12<floatT> operator()(gSite site) {
    __device__ __host__ void operator()(gSite site) {

        Vect12<floatT> tmp[12];
        Vect12<floatT> tmp2[12];
        for (size_t stack = 0; stack < 12; stack++) {
            tmp[stack] = conj(_spinorIn.getElement(GInd::getSiteStack(site,stack)));
        }

        for (size_t stack = 0; stack < 12; stack++) {
            _spinorIn.setElement(GInd::getSiteStack(site,stack),tmp[stack]);
        }



    }
};

template<class floatT,Layout LatLayout, size_t HaloDepthSpin>
struct DaggerSource{

    //input
    Vect12ArrayAcc<floatT> _spinorIn;

    typedef GIndexer<LatLayout, HaloDepthSpin > GInd;
    //Constructor to initialize all necessary members.
    DaggerSource(Spinorfield<floatT, true,LatLayout, HaloDepthSpin, 12, 12> &spinorIn)
                : _spinorIn(spinorIn.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    //__device__ __host__ Vect12<floatT> operator()(gSite site) {
    __device__ __host__ void operator()(gSite site) {

        Vect12<floatT> tmp[12];
        Vect12<floatT> tmp2[12];
        for (size_t stack = 0; stack < 12; stack++) {
            tmp[stack] = _spinorIn.getElement(GInd::getSiteStack(site,stack));
        }

        for (size_t stack = 0; stack < 12; stack++) {
            for (size_t stack2 = 0; stack2 < 12; stack2++) {
                tmp2[stack2].data[stack] = conj(tmp[stack].data[stack2]);
            }
        }

        for (size_t stack = 0; stack < 12; stack++) {
            _spinorIn.setElement(GInd::getSiteStack(site,stack),tmp2[stack]);
        }



    }
};

