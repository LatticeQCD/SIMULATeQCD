#pragma once

#include "../simulateqcd.h"
#include "fullSpinor.h"

#define LZ 64

// functions definitions

template<class floatT, size_t HaloDepth>
void fourier3D(Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinor_out,Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinor_in,LatticeContainer<true,COMPLEX(floatT)> & redBaseDevice,LatticeContainer<false,COMPLEX(floatT)> & redBaseHost,CommunicationBase & commBase);

template<typename floatT, bool onDevice, size_t HaloDepthSpin>
COMPLEX(floatT) sumXYZ_TrMdaggerM(int t,
        const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorInDagger,
        const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorIn,
        LatticeContainer<true,COMPLEX(floatT)> & _redBase);

template<typename floatT, bool onDevice,size_t HaloDepthSpin>
COMPLEX(floatT) sumXYZ_TrMdaggerMwave(int t,
        const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorInDagger,
        const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 12, 12> & spinorIn,
        const Spinorfield<floatT, onDevice, All, HaloDepthSpin, 3,1> & spinor_wave,
        LatticeContainer<true,COMPLEX(floatT)> & _redBase, int time, int col);

template<typename floatT, size_t HaloDepthSpin>
void loadWave(std::string fname, Spinorfield<floatT, true, All, HaloDepthSpin, 3,1> & spinor_device,
                                 Spinorfield<floatT, false, All, HaloDepthSpin, 3,1> & spinor_host,
                                 int time, int col,CommunicationBase & commBase);

template<typename floatT, size_t HaloDepth>
void makeWaveSource(Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinorIn, const Spinorfield<floatT, true, All, HaloDepth, 3,1> &spinor_wave,
                      size_t time, size_t col,size_t post);

template<typename floatT>
void gatherAllHost(std::complex<floatT> *in,CommunicationBase & commBase);

template<typename floatT,int direction>
void gatherHostXYZ(std::complex<floatT> *in,MPI_Comm & comm,int glx,int gly,int glz);

// gpu functions

template<class floatT,int direction>
__global__ void fourier(LatticeContainerAccessor _redBaseOut, LatticeContainerAccessor _redBaseIn, const size_t size,const size_t lx,const size_t ly,const size_t lz,const size_t lt,size_t lsIn) {

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size) {
        return;
    }

    int ix, iy, it;

    int  tmp;

    int ls=lsIn;
    int hf=lz/ls;

    divmod(site, lx*ly, it, tmp);
    divmod(tmp,  lx   , iy, ix );

   COMPLEX(floatT) v[LZ];
   COMPLEX(floatT) v0[LZ]; 

  //  COMPLEX(floatT) * v = new COMPLEX(floatT)[lz];
    
    if(direction == 0){
       for(int z =0; z < lz ; z++){
        v[z] = _redBaseOut.getElement<COMPLEX(floatT)>(z+lz*(ix+lx*(iy+ly*it)));
       }
    }
    if(direction == 1){
       for(int z =0; z < lz ; z++){
        v[z] = _redBaseOut.getElement<COMPLEX(floatT)>(ix+lx*(z+lz*(iy+ly*it)));
       }
    }
    if(direction == 2){
       for(int z =0; z < lz ; z++){
        v[z] = _redBaseOut.getElement<COMPLEX(floatT)>(ix+lx*(iy+ly*(z+lz*it)));
       }
    }
    // standard fourier transformation
    for(int z =0; z < lz ; z++){
       v0[z] = v[z];
    }    
    for(int i =0; i < hf ; i++){
       for(int k =0; k < ls ; k++){
          COMPLEX(floatT) sum = 0.0;
          for(int z =0; z < ls ; z++){
             sum = sum + v0[z*hf+i]*COMPLEX(floatT)(cos(2.0*k*z*M_PI/ls),sin(2.0*k*z*M_PI/ls));
          }
          v[i+k*hf] = sum;
       }
    }


    //fast part
    for(int j =0; j < (int)(log2(lz/lsIn)+0.1) ; j++){
       for(int z =0; z < lz ; z++){
          v0[z] = v[z];
       }
       ls=ls*2;
       hf=hf/2;
       for(int s =0; s < hf ; s++){
          for(int k =0; k < ls/2 ; k++){

             COMPLEX(floatT) phase = COMPLEX(floatT)(cos(2.0*k*M_PI/ls),sin(2.0*k*M_PI/ls));

             COMPLEX(floatT) even = v0[s + k*hf*2];
             COMPLEX(floatT) odd  = v0[s + k*hf*2 + hf];

             v[s + k*hf] = even + phase*odd;
             v[s + k*hf + hf*ls/2] = even - phase*odd;

          }
       }
    }

   for(int z =0; z < lz ; z++){
      v[z] = v[z]/sqrt(lz);
     // printf("%f %f \n", v[z].cREAL , v[z].cIMAG);
      if(direction == 0){
         _redBaseOut.setValue<COMPLEX(floatT)>(z+lz*(ix+lx*(iy+ly*it)),v[z]);
      }
      if(direction == 1){
         _redBaseOut.setValue<COMPLEX(floatT)>(ix+lx*(z+lz*(iy+ly*it)),v[z]);
      }
      if(direction == 2){
//         if(ix == 0 && iy == 0 && z==0)
//         printf("i%d %d %d %f %f \n",(int)ix, (int)iy, (int)z, v[z].cREAL , v[z].cIMAG);
         _redBaseOut.setValue<COMPLEX(floatT)>(ix+lx*(iy+ly*(z+lz*it)),v[z]);
      }
   }
    

//   delete [] v;
//   delete [] v0;
   
}



template<class floatT>
__global__ void setValues(MemoryAccessor _redBaseOut,const size_t size, const size_t lx,const size_t ly,const size_t lz){

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size) {
        return;
    }

    int ix, iy, iz;

    int  tmp;

    divmod(site, lx*ly, iz, tmp);
    divmod(tmp,  lx   , iy, ix );


    for(int z =0; z < lz ; z++){
      _redBaseOut.setValue<COMPLEX(floatT)>(ix+lx*(iy+ly*z),0.1*(z+1)+0.1*(iy)+0.1*(ix));
   }




}

template<class floatT>
__global__ void moveValues(LatticeContainerAccessor _redBaseOut, LatticeContainerAccessor _redBaseIn,const size_t size, const size_t lx,const size_t ly,const size_t lz, const size_t lx2, const size_t ly2, const size_t lz2,const size_t xtopo, const size_t ytopo, const size_t ztopo){

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size) {
        return;
    }

    int ix, iy, iz;

    int  tmp;

    divmod(site, lx2*ly2, iz, tmp);
    divmod(tmp,  lx2    , iy, ix );


      size_t pos , pos2;
      pos  = ix+lx2*(iy+ly2*iz);
      pos2 = (ix-lx*xtopo)+lx*((iy-ly*ytopo)+ly*(iz-lz*ztopo));

//      printf("%d %d %d %d %d %d %d \n", (int)site,(int)ix , (int)iy ,(int)iz, (int)(lz*(ztopo+1)), (int)(lz*ztopo), (int)(( ix >= lx*xtopo && ix < lx*(xtopo+1) && iy >= ly*ytopo && iy < ly*(ytopo+1) && iz >= lz*ztopo && iz < lz*(ztopo+1))));

      if( ix >= lx*xtopo && ix < lx*(xtopo+1) && iy >= ly*ytopo && iy < ly*(ytopo+1) && iz >= lz*ztopo && iz < lz*(ztopo+1)){
          printf("i  %d %d %d %d %d %f %f \n",(int)ix, (int)iy, (int)iz, (int)pos, (int)pos2,_redBaseIn.getElement<COMPLEX(floatT)>(pos2).cREAL , _redBaseIn.getElement<COMPLEX(floatT)>(pos2).cIMAG );
          _redBaseOut.setValue<COMPLEX(floatT)>(pos,_redBaseIn.getElement<COMPLEX(floatT)>(pos2));
      }
      else{
          printf("ii  %d %d %d %d \n",(int)ix, (int)iy, (int)iz, (int)pos);
          _redBaseOut.setValue<COMPLEX(floatT)>(pos,0.0);
      }

}


template<class floatT,size_t HaloDepth>
__global__ void copySpinorToContainer(MemoryAccessor _redBase, Vect12ArrayAcc<floatT> _SpinorIn, const size_t size, int spincolor1, int spincolor2,int lx,int ly,int lz,int lt,
                                      const int xtopo,const int ytopo,const int ztopo) {

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size) {
        return;
    }

    typedef GIndexer<All,HaloDepth> GInd;

    int ix, iy, iz, it;
//    it = tt;

    int  tmp;

    divmod(site, lx*ly*lz, it, tmp);
    divmod(tmp , lx*ly   , iz, tmp);
    divmod(tmp ,  lx     , iy, ix);

    if( ix >= xtopo*GInd::getLatData().lx && ix < (1+xtopo)*GInd::getLatData().lx &&
        iy >= ytopo*GInd::getLatData().ly && iy < (1+ytopo)*GInd::getLatData().ly &&
        iz >= ztopo*GInd::getLatData().lz && iz < (1+ztopo)*GInd::getLatData().lz  ){

    Vect12<floatT> tmp12 = _SpinorIn.getElement(GInd::getSiteStack(GInd::getSite(ix-xtopo*GInd::getLatData().lx,iy-ytopo*GInd::getLatData().ly, iz-ztopo*GInd::getLatData().lz, it) , spincolor2));
    _redBase.setValue<COMPLEX(floatT)>(ix+lx*(iy+ly*(iz+lz*it)),tmp12.data[spincolor1]);
    //printf("%d %f %d %d %d \n" ,(int)(site), tmp12.data[spincolor1].cREAL,xtopo,ytopo,ztopo );
    }
    else{
        _redBase.setValue<COMPLEX(floatT)>(ix+lx*(iy+ly*(iz+lz*it)),0.0);
    }


}

template<class floatT,size_t HaloDepth>
__global__ void copySpinorToContainerLocal(MemoryAccessor _redBase, Vect12ArrayAcc<floatT> _SpinorIn, const size_t size, int spincolor1, int spincolor2,int lx,int ly,int lz,int lt) {

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size) {
        return;
    }

    typedef GIndexer<All,HaloDepth> GInd;

    int ix, iy, iz, it;
//    it = tt;

    int  tmp;

    divmod(site, lx*ly*lz, it, tmp);
    divmod(tmp , lx*ly   , iz, tmp);
    divmod(tmp ,  lx     , iy, ix);

    Vect12<floatT> tmp12 = _SpinorIn.getElement(GInd::getSiteStack(GInd::getSite(ix,iy, iz, it) , spincolor2));
    _redBase.setValue<COMPLEX(floatT)>(ix+lx*(iy+ly*(iz+lz*it)),tmp12.data[spincolor1]);


}

template<class floatT,size_t HaloDepth>
__global__ void copyContainerToSpinor(Vect12ArrayAcc<floatT> _SpinorOut,LatticeContainerAccessor _redBase,
                                      const size_t size, int spincolor1, int spincolor2,int lx,int ly,int lz,int lt,
                                      const int xtopo,const int ytopo,const int ztopo) {

    size_t site = blockDim.x * blockIdx.x + threadIdx.x;
    if (site >= size) {
        return;
    }

    typedef GIndexer<All,HaloDepth> GInd;

    int ix, iy, iz, it;
//    it = tt;

    int  tmp;

    divmod(site, GInd::getLatData().vol3, it, tmp);
    divmod(tmp , GInd::getLatData().vol2, iz, tmp);
    divmod(tmp , GInd::getLatData().vol1, iy, ix);

    COMPLEX(floatT) val = _redBase.getElement<COMPLEX(floatT)>((ix+xtopo*GInd::getLatData().lx)+lx*((iy+ytopo*GInd::getLatData().ly)+ly*((iz+ztopo*GInd::getLatData().lz)+lz*it)));

    Vect12<floatT> tmp12 = _SpinorOut.getElement(GInd::getSiteStack(GInd::getSite((size_t)ix,(size_t)iy, (size_t)iz, (size_t)(it)) , spincolor2));

    tmp12.data[spincolor1] = val;
    _SpinorOut.setElement(GInd::getSiteStack(GInd::getSite((size_t)ix,(size_t)iy, (size_t)iz, (size_t)(it)) , spincolor2),tmp12);

}

template<class floatT, size_t HaloDepth,size_t NStacks>
struct SumXYZ_TrMdaggerM2{
    using SpinorRHS_t = Spinorfield<floatT, true, All, HaloDepth, 12, NStacks>;


    SpinorColorAcc<floatT> _spinorIn;
    SpinorColorAcc<floatT> _spinorInDagger;
    int _t;

    // adding spinor gives compile error
    typedef GIndexer<All, HaloDepth > GInd;
    SumXYZ_TrMdaggerM2(int t,const SpinorRHS_t &spinorInDagger, const SpinorRHS_t &spinorIn)
          :  _t(t), _spinorIn(spinorIn.getAccessor()), _spinorInDagger(spinorInDagger.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ COMPLEX(double)  operator()(gSite site){

        sitexyzt coords=site.coord;
        gSite siteT = GInd::getSite(coords.x,coords.y, coords.z, _t);

        COMPLEX(double) temp(0.0,0.0);
        for (size_t stack = 0; stack < NStacks; stack++) {
            temp  = temp + _spinorInDagger.template getElement<double>(GInd::getSiteStack(siteT,stack)) *
                                 _spinorIn.template getElement<double>(GInd::getSiteStack(siteT,stack));
        }

//        printf("tr %d %d %d %f %f \n", coords.x,coords.y,coords.z,temp.cREAL, temp.cIMAG);

        return temp;
    }
};

template<class floatT, size_t HaloDepth,size_t NStacks>
struct SumXYZ_TrMdaggerMwave{
    using SpinorRHS_t = Spinorfield<floatT, true, All, HaloDepth, 12, NStacks>;


    SpinorColorAcc<floatT> _spinorIn;
    SpinorColorAcc<floatT> _spinorInDagger;
    Vect3ArrayAcc<floatT> _spinor_wave;
    int _t, _time, _col;

    // adding spinor gives compile error
    typedef GIndexer<All, HaloDepth > GInd;
    SumXYZ_TrMdaggerMwave(int t,const SpinorRHS_t &spinorInDagger, const SpinorRHS_t &spinorIn, const Spinorfield<floatT, true, All, HaloDepth, 3,1> &spinor_wave, int time, int col)
          :  _t(t), _spinorIn(spinorIn.getAccessor()), _spinorInDagger(spinorInDagger.getAccessor()),
                   _spinor_wave(spinor_wave.getAccessor()), _time(time), _col(col)
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ COMPLEX(double)  operator()(gSite site){

        sitexyzt coords=site.coord;
        gSite siteT = GInd::getSite(coords.x,coords.y, coords.z, _t);

        COMPLEX(double) temp(0.0,0.0);
        for (size_t stack = 0; stack < NStacks; stack++) {
            temp  = temp + _spinorInDagger.template getElement<double>(GInd::getSiteStack(siteT,stack)) *
                                 _spinorIn.template getElement<double>(GInd::getSiteStack(siteT,stack));
        }
        temp = temp*(_spinor_wave.template getElement<double>(GInd::getSite(coords.x,coords.y, coords.z, _time))).data[_col];  
      //  temp = (_spinor_wave.template getElement<double>(GInd::getSite(coords.x,coords.y, coords.z, _time))).data[col];

//        printf("tr %d %d %d %f %f \n", coords.x,coords.y,coords.z,temp.cREAL, temp.cIMAG);

        return temp;
    }
};


template<class floatT, size_t HaloDepth>
struct MakeWaveSource12{

    // accessor to access the spinor field
    Vect12ArrayAcc<floatT> _spinorIn;
    Vect3ArrayAcc<floatT> _spinor_wave;

    size_t _time, _col, _post;

    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    MakeWaveSource12(Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinorIn, const Spinorfield<floatT, true, All, HaloDepth, 3,1> &spinor_wave
                      ,size_t time, size_t col, size_t post)
                : _spinorIn(spinorIn.getAccessor()), _spinor_wave(spinor_wave.getAccessor()),
                  _time(time), _col(col), _post(post)
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ void operator()(gSite site) {

        for (size_t stack = 0; stack < 12; stack++) {
            Vect12<floatT> tmp(0.0);

            sitexyzt coords=site.coord;
            gSite siteT = GInd::getSite(coords.x,coords.y, coords.z, _time);
            if(coords[3] == _post ){
                tmp.data[stack] = (_spinor_wave.template getElement<double>(siteT)).data[_col];
            }

            const gSiteStack writeSite = GInd::getSiteStack(site,stack);
            _spinorIn.setElement(writeSite,tmp);

        }
    }
};


