/*
 * Created by Rasmus Larsen on 17-02-2021
 *
 */

#pragma once

#include "../../define.h"
#include "../../gauge/gaugefield.h"
#include "../../base/stopWatch.h"
#include "../../base/latticeContainer.h"
#include "../../gauge/gaugeAction.h"
#include "../../spinor/spinorfield.h"
#include "../../base/math/simpleArray.h"

#include "../../modules/rhmc/rhmcParameters.h"
#include "../../modules/gaugeFixing/gfix.h"

#include "../../base/math/correlators.h"

#include <iostream>
using namespace std;

template<class floatT, size_t HaloDepth, int stacks>
class WilsonLineCorrelatorMultiGPU{

    public:

    //construct the class
        WilsonLineCorrelatorMultiGPU() {}

    // Function to compute the wilson line using the above struct CalcWilson.
    void gWilson(Gaugefield<floatT,true,HaloDepth> &gauge , size_t length);

    void gMoveOne( Gaugefield<floatT,true,HaloDepth> &gauge , int direction, int up);

    COMPLEX(floatT) gDotAlongXY( Gaugefield<floatT,true,HaloDepth> &gauge ,int shiftx, int shifty,  LatticeContainer<true,COMPLEX(floatT)> &redBase);

    COMPLEX(floatT) gDotAlongXYFull( Gaugefield<floatT,true,HaloDepth> &gauge ,int shiftx, int shifty,  LatticeContainer<true,COMPLEX(floatT)> &redBase);

    std::vector<floatT> gDotAlongXYStacked( Gaugefield<floatT,true,HaloDepth> &gauge ,int shiftx, int shifty,  LatticeContainer<true,floatT> &redBase);

    std::vector<floatT> gDotAlongXYStackedShared( Gaugefield<floatT,true,HaloDepth> &gauge , int shifty,  LatticeContainer<true,floatT> &redBase);
//<<<<<<< HEAD:src/modules/observables/WilsonLineCorrelatorMultiGPU.h


    std::vector<floatT> gWilsonLoop( Gaugefield<floatT,true,HaloDepth> &gauge , Gaugefield<floatT,true,HaloDepth> &gaugeX ,int wlt,  LatticeContainer<true,floatT> &redBase);

//=======
//>>>>>>> origin/main:src/modules/observables/wilsonLineCorrelatorMultiGPU.h
};

template<class floatT,size_t HaloDepth,Layout LatLayout,size_t direction,bool Up>
struct ShiftVectorOne{

    /// Gauge accessor to access the gauge field.
    SU3Accessor<floatT> _gaugeIn;

    /// Constructor to initialize all necessary members.
    ShiftVectorOne(Gaugefield<floatT,true,HaloDepth> &gaugeIn) :
            _gaugeIn(gaugeIn.getAccessor())
    {}

    __device__ __host__ auto operator()(gSiteMu siteMu) {

    typedef GIndexer<All,HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);


    SU3<floatT> Stmp;

    // takes vector from mu=1 direction from up or down 1 and saves and returns it
    // needed since A(x).A(x+r) product is done by making a copy of A and then move it around
    // and takes the dot product between original and moved vector
    if(Up == true){
        Stmp =  _gaugeIn.getLink(GInd::getSiteMu(GInd::site_up(site, direction),1));
    }else{
        Stmp =  _gaugeIn.getLink(GInd::getSiteMu(GInd::site_dn(site, direction),1));
    }

    return Stmp;

    }

    auto getAccessor() const
    {
        return *this;
    }
};



// copies from mu=direction and saves is back into the object that calls the function
template<class floatT,size_t HaloDepth,Layout LatLayout,size_t direction>
struct CopyFromMu{

    /// Gauge accessor to access the gauge field.
    SU3Accessor<floatT> _gaugeIn;

    /// Constructor to initialize all necessary members.
    CopyFromMu(Gaugefield<floatT,true,HaloDepth> &gaugeIn) :
            _gaugeIn(gaugeIn.getAccessor())
    {}

    __device__ __host__ auto operator()(gSite site) {

    typedef GIndexer<All,HaloDepth> GInd;

    return _gaugeIn.getLink(GInd::getSiteMu(site,direction));

    }

    auto getAccessor() const
    {
        return *this;
    }



};

// calculates A(x).B(x+r) where r can be any point in x and y direction
template<class floatT,size_t HaloDepth,Layout LatLayout>
struct DotAlongXYInterval{

    /// Gauge accessor to access the gauge field.
    SU3Accessor<floatT> _gaugeIn;
    int _shiftx;
    int _shifty;

    /// Constructor to initialize all necessary members.
    DotAlongXYInterval(Gaugefield<floatT,true,HaloDepth> &gaugeIn,int shiftx,int shifty) :
            _gaugeIn(gaugeIn.getAccessor()),_shiftx(shiftx),_shifty(shifty) {
    }

    /// This is the operator that is called inside the Kernel. We set the type to COMPLEX(floatT) because the
    /// Polyakov loop is complex valued.
    __device__ __host__ auto operator()(gSite site) {

        sitexyzt coords=site.coordFull;
        const int split = 1;
	if(((int)coords.x-split*((int)coords.x/split) ==0) && ((int)coords.y-split*((int)coords.y/split) ==0) && ((int)coords.z-split*((int)coords.z/split) ==0)){

        typedef GIndexer<All,HaloDepth> GInd;

        /// Get coordinates.
        int ix=(int)coords.x+_shiftx;
        int iy=(int)coords.y+_shifty;
        int it=(int)coords.t;

        if(ix >= (int)GInd::getLatData().lxFull){
            ix -=(int)GInd::getLatData().lxFull;
        }

        if(ix < 0){
            ix +=(int)GInd::getLatData().lxFull;
        }

        if(iy >= (int)GInd::getLatData().lyFull){
            iy -=(int)GInd::getLatData().lyFull;
        }

        if(iy < 0){
            iy +=(int)GInd::getLatData().lyFull;
        }

        COMPLEX(floatT) results(0.0,0.0);


        // loop over all t, was implemented like this, in case not all t should be used
        for(int tt = 0; tt < (int)GInd::getLatData().ltFull; tt += split){
                results = results +  tr_c(_gaugeIn.getLinkDagger(GInd::getSiteMu(GInd::getSiteFull(coords.x,coords.y, coords.z, (size_t)(it+tt)),0))
                                         *_gaugeIn.getLink(GInd::getSiteMu(GInd::getSiteFull((size_t)ix, (size_t)iy, coords.z, (size_t)(it+tt)),1)));
	}

	return results/3.0;


	}else{
		return COMPLEX(floatT) (0.0,0.0);
	}

    }
};

// calculates A(x).B(x+r) where r can be any point in x and y direction
template<class floatT,size_t HaloDepth,Layout LatLayout>
struct DotAlongXYIntervalFull{

    /// Gauge accessor to access the gauge field.
    SU3Accessor<floatT> _gaugeIn;
    int _shiftx;
    int _shifty;

    /// Constructor to initialize all necessary members.
    DotAlongXYIntervalFull(Gaugefield<floatT,true,HaloDepth> &gaugeIn,int shiftx,int shifty) :
            _gaugeIn(gaugeIn.getAccessor()),_shiftx(shiftx),_shifty(shifty) {
    }

    /// This is the operator that is called inside the Kernel. We set the type to COMPLEX(floatT) because the
    /// Polyakov loop is complex valued.
    __device__ __host__ auto operator()(gSite site) {

        sitexyzt coords=site.coordFull;

        typedef GIndexer<All,HaloDepth> GInd;

        /// Get coordinates.
        int ix=(int)coords.x+_shiftx;
        int iy=(int)coords.y+_shifty;

        if(ix >= (int)GInd::getLatData().lxFull){
            ix -=(int)GInd::getLatData().lxFull;
        }

        if(ix < 0){
            ix +=(int)GInd::getLatData().lxFull;
        }

        if(iy >= (int)GInd::getLatData().lyFull){
            iy -=(int)GInd::getLatData().lyFull;
        }

        if(iy < 0){
            iy +=(int)GInd::getLatData().lyFull;
        }

        COMPLEX(floatT) results(0.0,0.0);

        // loop over all t, was implemented like this, in case not all t should be used
                results = results +  tr_c(_gaugeIn.getLinkDagger(GInd::getSiteMu(GInd::getSiteFull(coords.x,coords.y, coords.z, coords.t),0))
                                         *_gaugeIn.getLink(GInd::getSiteMu(GInd::getSiteFull((size_t)ix, (size_t)iy, coords.z, coords.t),1)));

        return results/3.0;

    }
};


// calculates 1 wilson line of length length
// The wilson line is calculated from any spacetime point
template<class floatT,size_t HaloDepth>
struct CalcWilson{

    /// Gauge accessor to access the gauge field.
    SU3Accessor<floatT> SU3Accessor;
    size_t _length;

    /// Constructor to initialize all necessary members.
    CalcWilson(Gaugefield<floatT,true,HaloDepth> &gauge,size_t length) :
                                                                         SU3Accessor(gauge.getAccessor())
                                                                        ,_length(length)
    {
    }

    /// This is the operator that is called inside the Kernel. We set the type to COMPLEX(floatT) because the
    /// Polyakov loop is complex valued.
    __device__ __host__ auto operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;

        /// Define an SU(3) matrix and initialize result variable.
        SU3<floatT> temp;
        COMPLEX(floatT) result;

        /// Extension in timelike direction. In general unsigned declarations reduce compiler warnings.
        const size_t Ntau=GInd::getLatData().lt;

        /// Get coordinates.
        sitexyzt coords=site.coord;
        size_t ix=coords.x;
        size_t iy=coords.y;
        size_t iz=coords.z;
        size_t it=coords.t;

        /// Start off at this site, pointing in N_tau direction.
        temp=SU3Accessor.getLink(GInd::getSiteMu(GInd::getSite(ix, iy, iz, it), 3));

        /// Loop over N_tau direction.
        for (size_t itp = 1; itp < _length; itp++) {
            size_t itau=it+itp;
            if(itau >= Ntau){
                itau-=Ntau;
            }
            temp*=SU3Accessor.getLink(GInd::getSiteMu(GInd::getSite(ix, iy, iz, itau), 3));
        }

        return temp;
    }

    auto getAccessor() const
    {
        return *this;
    }

};

// calculates A(x).B(x+r) where r can be any point in x and y direction
template<class floatT,size_t HaloDepth,Layout LatLayout, int stacks>
struct DotAlongXYIntervalStacked{

    /// Gauge accessor to access the gauge field.
    MemoryAccessor _redBase;
    SU3Accessor<floatT> _gaugeIn;
    int _shiftx;
    int _shifty;

    /// Constructor to initialize all necessary members.
    DotAlongXYIntervalStacked(LatticeContainer<true,floatT> & redBase,Gaugefield<floatT,true,HaloDepth> &gaugeIn,int shiftx,int shifty) :
            _redBase(redBase.getAccessor()),_gaugeIn(gaugeIn.getAccessor()),_shiftx(shiftx),_shifty(shifty)
    {
    }

    /// This is the operator that is called inside the Kernel. We set the type to COMPLEX(floatT) because the
    /// Polyakov loop is complex valued.
    __device__ __host__ auto operator()(gSite site) {

        sitexyzt coords=site.coordFull;
        const int split = 1;
        if(((int)coords.x-split*((int)coords.x/split) ==0) && ((int)coords.y-split*((int)coords.y/split) ==0) && ((int)coords.z-split*((int)coords.z/split) ==0)){

        typedef GIndexer<All,HaloDepth> GInd;

        int ix[stacks];
        floatT results[stacks];

        /// Get coordinates.
        int iy=(int)coords.y+_shifty;
//        int iz=(int)coords.z;
        int it=(int)coords.t;


        for(int i =0; i < stacks ; i++){
            ix[i]=(int)coords.x+_shiftx+i;
            results[i] = 0.0;

        if(ix[i] >= (int)GInd::getLatData().lxFull){
            ix[i] -=(int)GInd::getLatData().lxFull;
        }

        if(ix[i] < 0){
            ix[i] +=(int)GInd::getLatData().lxFull;
        }
        }

        if(iy >= (int)GInd::getLatData().lyFull){
            iy -=(int)GInd::getLatData().lyFull;
        }

        if(iy < 0){
            iy +=(int)GInd::getLatData().lyFull;
        }




        // loop over all t, was implemented like this, in case not all t should be used
        for(int tt = 0; tt < (int)GInd::getLatData().ltFull; tt += split){
                SU3<floatT> su3Temp = _gaugeIn.getLinkDagger(GInd::getSiteMu(GInd::getSiteFull(coords.x,coords.y, coords.z, (size_t)(it+tt)),0));

                for(int i =0; i < stacks ; i++){
                results[i] = results[i] +  tr_c(su3Temp
                                         *_gaugeIn.getLink(GInd::getSiteMu(GInd::getSiteFull((size_t)ix[i], (size_t)iy, coords.z, (size_t)(it+tt)),1))).cREAL;
                }
        }

        for(int i =1; i < stacks ; i++){
            _redBase.setValue<floatT>(site.isite+i*GInd::getLatData().vol3, results[i]/3.0);
        }
        return results[0]/3.0;

        }
        else{
                return 0.0;
        }

    }
};

//<<<<<<< HEAD:src/modules/observables/WilsonLineCorrelatorMultiGPU.h
//////////wilson loop

template<class floatT,size_t HaloDepth,Layout LatLayout, int stacks>
struct WilsonLoop{

    /// Gauge accessor to access the gauge field.
    MemoryAccessor _redBase;
    SU3Accessor<floatT> _gaugeIn;
    SU3Accessor<floatT> _gaugeInX;
    int _wlt;

    /// Constructor to initialize all necessary members.
    WilsonLoop(LatticeContainer<true,floatT> & redBase,Gaugefield<floatT,true,HaloDepth> &gaugeIn,Gaugefield<floatT,true,HaloDepth> &gaugeInX, int wlt) :
            _redBase(redBase.getAccessor()),_gaugeIn(gaugeIn.getAccessor()),_gaugeInX(gaugeInX.getAccessor()),_wlt(wlt)
   {
    }

    __device__ __host__ auto operator()(gSite site) {

        sitexyzt coords=site.coordFull;
        const int split = 1;

        typedef GIndexer<All,HaloDepth> GInd;

        floatT results[stacks];

        for(int xx = 0; xx < (int)GInd::getLatData().lxFull; xx += 1){
             results[xx] = 0.0;
	}


        // loop over all t, was implemented like this, in case not all t should be used
        for(int tt = 0; tt < (int)GInd::getLatData().ltFull; tt += split){
              int tend = tt+_wlt;
	      if(tend >= GInd::getLatData().ltFull){
	           tend -= GInd::getLatData().ltFull;
	      }

///////////////x direction
	      SU3<floatT> su3Temp = _gaugeIn.getLink(GInd::getSiteMu(GInd::getSiteFull(coords.x,coords.y, coords.z, (size_t)(tt)),0));
	      SU3<floatT> su3TempXend   = su3_one<floatT>();
	      SU3<floatT> su3TempXstart = su3_one<floatT>();


              for(int xx = 1; xx < (int)GInd::getLatData().lxFull; xx += 1){
                   int xend = coords.x+xx;
                   if(xend >= GInd::getLatData().lxFull){
                       xend -= GInd::getLatData().lxFull;
                   }
                   int xm1 = xend-1;
                   if(xm1 < 0){
                       xm1 += GInd::getLatData().lxFull;
                   } 

                   su3TempXend   *= _gaugeInX.getLink(GInd::getSiteMu(GInd::getSiteFull(xm1,coords.y, coords.z, (size_t)(tend)),0));
		   su3TempXstart *= _gaugeInX.getLink(GInd::getSiteMu(GInd::getSiteFull(xm1,coords.y, coords.z, (size_t)(tt)),0));

                   
                   results[xx] = results[xx] +  tr_c(su3Temp*
				                   su3TempXend*
						   _gaugeIn.getLinkDagger(GInd::getSiteMu(GInd::getSiteFull(xend, coords.y, coords.z, (size_t)(tt)),0))*
						   dagger(su3TempXstart)).cREAL;
                   

	      }
	      
///////////////y direction
              su3TempXend   = su3_one<floatT>();
              su3TempXstart = su3_one<floatT>();


              for(int yy = 1; yy < (int)GInd::getLatData().lyFull; yy += 1){
                   int yend = coords.y+yy;
                   if(yend >= GInd::getLatData().lyFull){
                       yend -= GInd::getLatData().lyFull;
                   }
                   int ym1 = yend-1;
                   if(ym1 < 0){
                       ym1 += GInd::getLatData().lyFull;
                   }

                   su3TempXend   *= _gaugeInX.getLink(GInd::getSiteMu(GInd::getSiteFull(coords.x,ym1, coords.z, (size_t)(tend)),1));
                   su3TempXstart *= _gaugeInX.getLink(GInd::getSiteMu(GInd::getSiteFull(coords.x,ym1, coords.z, (size_t)(tt)),1));


                   results[yy] = results[yy] +  tr_c(su3Temp*
                                                   su3TempXend*
                                                   _gaugeIn.getLinkDagger(GInd::getSiteMu(GInd::getSiteFull(coords.x, yend, coords.z, (size_t)(tt)),0))*
                                                   dagger(su3TempXstart)).cREAL;


              }
	      
///////////////z direction
              su3TempXend   = su3_one<floatT>();
              su3TempXstart = su3_one<floatT>();


              for(int zz = 1; zz < (int)GInd::getLatData().lzFull; zz += 1){
                   int zend = coords.z+zz;
                   if(zend >= GInd::getLatData().lzFull){
                       zend -= GInd::getLatData().lzFull;
                   }
		   int zm1 = zend-1;
                   if(zm1 < 0){
                       zm1 += GInd::getLatData().lzFull;
                   }

                   su3TempXend   *= _gaugeInX.getLink(GInd::getSiteMu(GInd::getSiteFull(coords.x,coords.y, zm1, (size_t)(tend)),2));
                   su3TempXstart *= _gaugeInX.getLink(GInd::getSiteMu(GInd::getSiteFull(coords.x,coords.y, zm1, (size_t)(tt)),2));


                   results[zz] = results[zz] +  tr_c(su3Temp*
                                                   su3TempXend*
                                                   _gaugeIn.getLinkDagger(GInd::getSiteMu(GInd::getSiteFull( coords.x, coords.y, zend, (size_t)(tt)),0))*
                                                   dagger(su3TempXstart)).cREAL;


              }
	      
        }

        for(int i =1; i < stacks ; i++){
            _redBase.setValue<floatT>(site.isite+i*GInd::getLatData().vol3, results[i]/9.0);
        }
        return (floatT)GInd::getLatData().ltFull;


    }

};


//#endif

//=======
//>>>>>>> origin/main:src/modules/observables/wilsonLineCorrelatorMultiGPU.h
