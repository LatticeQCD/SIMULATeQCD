/*
 * Created by Rasmus Larsen on 17-02-2021
 *
 */

#pragma once

#include "../../define.h"
#include "../../gauge/gaugefield.h"
#include "../../base/stopWatch.h"
#include "../../base/LatticeContainer.h"
#include "../../gauge/GaugeAction.h"
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

    GCOMPLEX(floatT) gDotAlongXY( Gaugefield<floatT,true,HaloDepth> &gauge ,int shiftx, int shifty,  LatticeContainer<true,GCOMPLEX(floatT)> &redBase);

    GCOMPLEX(floatT) gDotAlongXYFull( Gaugefield<floatT,true,HaloDepth> &gauge ,int shiftx, int shifty,  LatticeContainer<true,GCOMPLEX(floatT)> &redBase);

    std::vector<floatT> gDotAlongXYStacked( Gaugefield<floatT,true,HaloDepth> &gauge ,int shiftx, int shifty,  LatticeContainer<true,floatT> &redBase);

    std::vector<floatT> gDotAlongXYStackedShared( Gaugefield<floatT,true,HaloDepth> &gauge , int shifty,  LatticeContainer<true,floatT> &redBase);
};

template<class floatT,size_t HaloDepth,Layout LatLayout,size_t direction,bool Up>
struct ShiftVectorOne{

    /// Gauge accessor to access the gauge field.
    gaugeAccessor<floatT> _gaugeIn;

    /// Constructor to initialize all necessary members.
    ShiftVectorOne(Gaugefield<floatT,true,HaloDepth> &gaugeIn) :
            _gaugeIn(gaugeIn.getAccessor())
    {}

    __device__ __host__ auto operator()(gSiteMu siteMu) {

    typedef GIndexer<All,HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);


    GSU3<floatT> Stmp;

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
    gaugeAccessor<floatT> _gaugeIn;

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
    gaugeAccessor<floatT> _gaugeIn;
    int _shiftx;
    int _shifty;

    /// Constructor to initialize all necessary members.
    DotAlongXYInterval(Gaugefield<floatT,true,HaloDepth> &gaugeIn,int shiftx,int shifty) :
            _gaugeIn(gaugeIn.getAccessor()),_shiftx(shiftx),_shifty(shifty) {
    }

    /// This is the operator that is called inside the Kernel. We set the type to GCOMPLEX(floatT) because the
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

        GCOMPLEX(floatT) results(0.0,0.0);


        // loop over all t, was implemented like this, in case not all t should be used
        for(int tt = 0; tt < (int)GInd::getLatData().ltFull; tt += split){
                results = results +  tr_c(_gaugeIn.getLinkDagger(GInd::getSiteMu(GInd::getSiteFull(coords.x,coords.y, coords.z, (size_t)(it+tt)),0))
                                         *_gaugeIn.getLink(GInd::getSiteMu(GInd::getSiteFull((size_t)ix, (size_t)iy, coords.z, (size_t)(it+tt)),1)));
	}

	return results/3.0;


	}else{
		return GCOMPLEX(floatT) (0.0,0.0);
	}

    }
};

// calculates A(x).B(x+r) where r can be any point in x and y direction
template<class floatT,size_t HaloDepth,Layout LatLayout>
struct DotAlongXYIntervalFull{

    /// Gauge accessor to access the gauge field.
    gaugeAccessor<floatT> _gaugeIn;
    int _shiftx;
    int _shifty;

    /// Constructor to initialize all necessary members.
    DotAlongXYIntervalFull(Gaugefield<floatT,true,HaloDepth> &gaugeIn,int shiftx,int shifty) :
            _gaugeIn(gaugeIn.getAccessor()),_shiftx(shiftx),_shifty(shifty) {
    }

    /// This is the operator that is called inside the Kernel. We set the type to GCOMPLEX(floatT) because the
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

        GCOMPLEX(floatT) results(0.0,0.0);

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
    gaugeAccessor<floatT> gaugeAccessor;
    size_t _length;

    /// Constructor to initialize all necessary members.
    CalcWilson(Gaugefield<floatT,true,HaloDepth> &gauge,size_t length) :
                                                                         gaugeAccessor(gauge.getAccessor())
                                                                        ,_length(length)
    {
    }

    /// This is the operator that is called inside the Kernel. We set the type to GCOMPLEX(floatT) because the
    /// Polyakov loop is complex valued.
    __device__ __host__ auto operator()(gSite site) {
        typedef GIndexer<All,HaloDepth> GInd;

        /// Define an SU(3) matrix and initialize result variable.
        GSU3<floatT> temp;
        GCOMPLEX(floatT) result;

        /// Extension in timelike direction. In general unsigned declarations reduce compiler warnings.
        const size_t Ntau=GInd::getLatData().lt;

        /// Get coordinates.
        sitexyzt coords=site.coord;
        size_t ix=coords.x;
        size_t iy=coords.y;
        size_t iz=coords.z;
        size_t it=coords.t;

        /// Start off at this site, pointing in N_tau direction.
        temp=gaugeAccessor.getLink(GInd::getSiteMu(GInd::getSite(ix, iy, iz, it), 3));

        /// Loop over N_tau direction.
        for (size_t itp = 1; itp < _length; itp++) {
            size_t itau=it+itp;
            if(itau >= Ntau){
                itau-=Ntau;
            }
            temp*=gaugeAccessor.getLink(GInd::getSiteMu(GInd::getSite(ix, iy, iz, itau), 3));
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
    gaugeAccessor<floatT> _gaugeIn;
    int _shiftx;
    int _shifty;

    /// Constructor to initialize all necessary members.
    DotAlongXYIntervalStacked(LatticeContainer<true,floatT> & redBase,Gaugefield<floatT,true,HaloDepth> &gaugeIn,int shiftx,int shifty) :
            _redBase(redBase.getAccessor()),_gaugeIn(gaugeIn.getAccessor()),_shiftx(shiftx),_shifty(shifty)
    {
    }

    /// This is the operator that is called inside the Kernel. We set the type to GCOMPLEX(floatT) because the
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
                GSU3<floatT> su3Temp = _gaugeIn.getLinkDagger(GInd::getSiteMu(GInd::getSiteFull(coords.x,coords.y, coords.z, (size_t)(it+tt)),0));

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