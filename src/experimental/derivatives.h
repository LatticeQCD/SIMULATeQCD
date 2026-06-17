#include "../simulateqcd.h"
#include "fullSpinor.h"
#include "matrix6x6Hermitian.h"
#include "source.h"

#ifdef USE_HIP_AMD
#define BLOCKSIZE 64
#else
#define BLOCKSIZE 32
#endif


template<typename floatT, bool onDevice, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void applyDmu(Gaugefield<floatT, onDevice, HaloDepthGauge, R18> & _gauge, Spinorfield<floatT, true, All, HaloDepthSpin, 12, NStacks>& spinorOut, Spinorfield<floatT, true, All, HaloDepthSpin, 12, NStacks>& spinorIn,COMPLEX(floatT) phase);	

template<typename floatT, bool onDevice, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks,int dir>
void applyDmu_Individual(Gaugefield<floatT, onDevice, HaloDepthGauge, R18> & _gauge, Spinorfield<floatT, true, All, HaloDepthSpin, 12, NStacks>& spinorOut, Spinorfield<floatT, true, All, HaloDepthSpin, 12, NStacks>& spinorIn,
		         COMPLEX(floatT) phase);

template<typename floatT, bool onDevice, size_t HaloDepthGauge>
COMPLEX(double) calc_emt_Fmunu(Gaugefield<floatT, onDevice, HaloDepthGauge, R18> & _gauge);

template<typename floatT, bool onDevice, size_t HaloDepthGauge,int dir>
COMPLEX(double) calc_emt_Fmunu_Individual(Gaugefield<floatT, onDevice, HaloDepthGauge, R18> & _gauge);


template<class floatT, size_t HaloDepth, size_t HaloDepthSpin,size_t NStacks>
struct Dmu{

    //input
    SU3Accessor<floatT> _SU3Accessor;
    SpinorColorAcc<floatT> _SpinorColorAccessor;
    COMPLEX(floatT) _phase;

    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    Dmu(Gaugefield<floatT,true,HaloDepth,R18> &gauge,const Spinorfield<floatT, true,All, HaloDepthSpin, NStacks, NStacks> &spinorIn, COMPLEX(floatT) phase)
                : _SU3Accessor(gauge.getAccessor()),
                  _SpinorColorAccessor(spinorIn.getAccessor()), _phase(phase)
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ Vect12<floatT> operator()(gSiteStack site) {
        SU3<floatT> link;

        ColorVect<floatT> outSC;
        ColorVect<floatT> temp;

        temp = _SU3Accessor.getLink(GInd::getSiteMu(site,0))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 0));
        outSC =         (1.0/3.0)*GammaXMultVec(temp);
        // x backwards direction 
        temp = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 0),0))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 0));
        outSC = outSC - (1.0/3.0)*GammaXMultVec(temp);

        // y forward direction 
        temp = _SU3Accessor.getLink(GInd::getSiteMu(site,1))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 1));
        outSC = outSC + (1.0/3.0)*GammaYMultVec(temp);
        // y backwards direction 
        temp = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 1),1))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 1));
        outSC = outSC - (1.0/3.0)*GammaYMultVec(temp);

        // z forward direction 
        temp = _SU3Accessor.getLink(GInd::getSiteMu(site,2))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 2));
        outSC = outSC + (1.0/3.0)*GammaZMultVec(temp);
        // z backwards direction 
        temp = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 2),2))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 2));
        outSC = outSC - (1.0/3.0)*GammaZMultVec(temp);

        // t forward direction 
        temp = _SU3Accessor.getLink(GInd::getSiteMu(site,3))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 3));
        outSC = outSC - (1.0/_phase)*GammaTMultVec(temp);
        // t backwards direction 
        temp = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 3),3))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 3));
        outSC = outSC + (1.0*_phase)*GammaTMultVec(temp);

        return convertColorVectToVect12(outSC);
    }
};

template<class floatT, size_t HaloDepth, size_t HaloDepthSpin,size_t NStacks,int dir>
struct Dmu_Individual{

    //input
    SU3Accessor<floatT> _SU3Accessor;
    SpinorColorAcc<floatT> _SpinorColorAccessor;
    COMPLEX(floatT) _phase;

    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    Dmu_Individual(Gaugefield<floatT,true,HaloDepth,R18> &gauge,const Spinorfield<floatT, true,All, HaloDepthSpin, NStacks, NStacks> &spinorIn, COMPLEX(floatT) phase)
                : _SU3Accessor(gauge.getAccessor()),
                  _SpinorColorAccessor(spinorIn.getAccessor()), _phase(phase)
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ Vect12<floatT> operator()(gSiteStack site) {
        SU3<floatT> link;

        ColorVect<floatT> outSC;
        ColorVect<floatT> temp;


        if(dir ==0){
           temp = _SU3Accessor.getLink(GInd::getSiteMu(site,0))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 0));
           outSC =         (1.0)*GammaXMultVec(temp);
           // x backwards direction 
           temp = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 0),0))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 0));
           outSC = outSC - (1.0)*GammaXMultVec(temp);
        }

        if(dir ==1){
	   // y forward direction 
           temp = _SU3Accessor.getLink(GInd::getSiteMu(site,1))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 1));
           outSC =         (1.0)*GammaYMultVec(temp);
           // y backwards direction 
           temp = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 1),1))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 1));
           outSC = outSC - (1.0)*GammaYMultVec(temp);
        }

	if(dir ==2){
           // z forward direction 
           temp = _SU3Accessor.getLink(GInd::getSiteMu(site,2))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 2));
           outSC =         (1.0)*GammaZMultVec(temp);
           // z backwards direction 
           temp = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 2),2))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 2));
           outSC = outSC - (1.0)*GammaZMultVec(temp);
        }

	if(dir ==3){
           // t forward direction 
           temp = _SU3Accessor.getLink(GInd::getSiteMu(site,3))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 3));
           outSC =         (1.0/_phase)*GammaTMultVec(temp);
           // t backwards direction 
           temp = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 3),3))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 3));
           outSC = outSC - (1.0*_phase)*GammaTMultVec(temp);
        }

        return convertColorVectToVect12(outSC);
    }
};


// calculate sigmunu fmunu that splits into 2 block matrices of size 6X6 and save to vector 18 complex
template<class floatT,size_t HaloDepthGauge>
struct emt_Fmunu{

    //input
    FieldStrengthTensor<floatT,HaloDepthGauge,true,R18> FT;


    typedef GIndexer<All, HaloDepthGauge > GInd;
    //Constructor to initialize all necessary members.
    emt_Fmunu(Gaugefield<floatT,true,HaloDepthGauge,R18> &gauge)
                : FT(gauge.getAccessor())
    { }

    __device__ __host__ COMPLEX(double) operator()(gSite site) {
        SU3<floatT> Fmunu ;
    	COMPLEX(double) tmp = 0.0;
	
	Fmunu = FT(site,0,1);
	tmp = tmp + tr_c(Fmunu*Fmunu);
        Fmunu = FT(site,0,2);
        tmp = tmp + tr_c(Fmunu*Fmunu);
	Fmunu = FT(site,1,2);
        tmp = tmp + tr_c(Fmunu*Fmunu);

	Fmunu = FT(site,3,0);
        tmp = tmp - tr_c(Fmunu*Fmunu);
        Fmunu = FT(site,3,1);
        tmp = tmp - tr_c(Fmunu*Fmunu);
        Fmunu = FT(site,3,2);
        tmp = tmp - tr_c(Fmunu*Fmunu);


        return tmp;
    }
};

// calculate sigmunu fmunu that splits into 2 block matrices of size 6X6 and save to vector 18 complex
template<class floatT,size_t HaloDepthGauge,int dir>
struct emt_Fmunu_Individual{

    //input
    FieldStrengthTensor<floatT,HaloDepthGauge,true,R18> FT;


    typedef GIndexer<All, HaloDepthGauge > GInd;
    //Constructor to initialize all necessary members.
    emt_Fmunu_Individual(Gaugefield<floatT,true,HaloDepthGauge,R18> &gauge)
                : FT(gauge.getAccessor())
    { }

    __device__ __host__ COMPLEX(double) operator()(gSite site) {
        SU3<floatT> Fmunu ;
        COMPLEX(double) tmp = 0.0;

	if(dir == 0){
           Fmunu = FT(site,0,1);
           tmp = tmp + tr_c(Fmunu*Fmunu);
           Fmunu = FT(site,0,2);
           tmp = tmp + tr_c(Fmunu*Fmunu);
           Fmunu = FT(site,1,2);
           tmp = tmp + tr_c(Fmunu*Fmunu);
        }

        if(dir == 1){
           Fmunu = FT(site,3,0);
           tmp = tmp + tr_c(Fmunu*Fmunu);
           Fmunu = FT(site,3,1);
           tmp = tmp + tr_c(Fmunu*Fmunu);
           Fmunu = FT(site,3,2);
           tmp = tmp + tr_c(Fmunu*Fmunu);
        }

        return tmp;
    }
};


      //set lin to -link if it reaches over the boundary
template<class floatT,size_t HaloDepthGauge>
struct setAnyPeriodicBoundary{

    //input
    SU3Accessor<floatT> _SU3Accessor;
    COMPLEX(floatT) _phase; 

    typedef GIndexer<All, HaloDepthGauge > GInd;
    setAnyPeriodicBoundary(Gaugefield<floatT,true,HaloDepthGauge,R18> &gauge, COMPLEX(floatT) phase): _SU3Accessor(gauge.getAccessor()) ,_phase(phase){}

    __device__ __host__ void operator()(gSite site) {

        SU3<floatT> tmp = (_phase)*_SU3Accessor.getLink(GInd::getSiteMu(site,3));

        size_t lt = GIndexer<All, HaloDepthGauge>::getLatData().globLT;

        sitexyzt coord = GIndexer<All, HaloDepthGauge>::getLatData().globalPos(site.coord);

        if(coord.t == (lt-1) ){
            _SU3Accessor.setLink(GInd::getSiteMu(site,3),tmp);
        }

    }

};

      template<typename floatT, size_t HaloDepthGauge>
      void anyperiodicBoundaries(Gaugefield<floatT,true,HaloDepthGauge,R18> &_gauge, COMPLEX(floatT) phase ){
          typedef GIndexer<All, HaloDepthGauge> GInd;
          size_t _elems = GInd::getLatData().vol4;
          CalcGSite<All, HaloDepthGauge> calcGSite;
          iterateFunctorNoReturn<true>(setAnyPeriodicBoundary<floatT,HaloDepthGauge>(_gauge,phase), calcGSite, _elems);
          _gauge.updateAll();
      }

template<class floatT,size_t HaloDepthGauge>
struct SetU4{

    //input
    SU3Accessor<floatT> _SU3Accessor;
    COMPLEX(floatT) _phase;

    typedef GIndexer<All, HaloDepthGauge > GInd;
    SetU4(Gaugefield<floatT,true,HaloDepthGauge,R18> &gauge, COMPLEX(floatT) phase): _SU3Accessor(gauge.getAccessor()) ,_phase(phase){}

    __device__ __host__ void operator()(gSite site) {

        SU3<floatT> tmp = (_phase)*_SU3Accessor.getLink(GInd::getSiteMu(site,3));

        _SU3Accessor.setLink(GInd::getSiteMu(site,3),tmp);

    }

};

      template<typename floatT, size_t HaloDepthGauge>
      void setU4(Gaugefield<floatT,true,HaloDepthGauge,R18> &_gauge, COMPLEX(floatT) phase ){
          typedef GIndexer<All, HaloDepthGauge> GInd;
          size_t _elems = GInd::getLatData().vol4;
          CalcGSite<All, HaloDepthGauge> calcGSite;
          iterateFunctorNoReturn<true>(SetU4<floatT,HaloDepthGauge>(_gauge,phase), calcGSite, _elems);
          _gauge.updateAll();
      }





