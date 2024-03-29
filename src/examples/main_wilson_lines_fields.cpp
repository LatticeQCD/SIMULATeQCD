/*
 * main_wilson_lines_fields.cpp
 *
 * Rasmus Larsen, 16 Dec 2020
 *
 */

#include "../simulateqcd.h"
#include "../modules/rhmc/rhmcParameters.h"
#include "../modules/gaugeFixing/gfix.h"

#include <iostream>
using namespace std;

#define PREC double
#define MY_BLOCKSIZE 256


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
    }
    else{
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


// Not used, as DotAlongXYInterval has taken over the job
template<class floatT,size_t HaloDepth,Layout LatLayout>
struct DotAlongXY{

    /// Gauge accessor to access the gauge field.
    SU3Accessor<floatT> _gaugeIn;
    int _shiftx;
    int _shifty;

    /// Constructor to initialize all necessary members.
    DotAlongXY(Gaugefield<floatT,true,HaloDepth> &gaugeIn,int shiftx,int shifty) :
            _gaugeIn(gaugeIn.getAccessor()),_shiftx(shiftx),_shifty(shifty)
   {
    }

    /// This is the operator that is called inside the Kernel. We set the type to COMPLEX(floatT) because the
    /// Polyakov loop is complex valued.
    __device__ __host__ auto operator()(gSite site) {

        typedef GIndexer<All,HaloDepth> GInd;

        /// Get coordinates.
        sitexyzt coords=site.coordFull;
        int ix=(int)coords.x+_shiftx;
        int iy=(int)coords.y+_shifty;
        int iz=(int)coords.z;
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

//        for(int tt = 0; tt < (int)GInd::getLatData().ltFull; tt += 1){
//                results = results +  tr_c(_gaugeIn.getLinkDagger(GInd::getSiteMu(GInd::getSiteFull(coords.x,coords.y, coords.z, (size_t)(it+tt)),0))
//                                        *_gaugeIn.getLink(GInd::getSiteMu(GInd::getSiteFull((size_t)ix, (size_t)iy, coords.z, (size_t)(it+tt)),1)))/3.0;
//        }
//        return results;


//        return  conj(_spinorIn1.getElement(site).getElement0())*_spinorIn2.getElement(GInd::getSiteFull((size_t)ix, (size_t)iy, (size_t)iz, (size_t)it)).getElement0();
        return  tr_c(_gaugeIn.getLinkDagger(GInd::getSiteMu(site,0))*_gaugeIn.getLink(GInd::getSiteMu(GInd::getSiteFull((size_t)ix, (size_t)iy, (size_t)iz, (size_t)it),1)))/3.0;
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
            _gaugeIn(gaugeIn.getAccessor()),_shiftx(shiftx),_shifty(shifty)
   {
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
//        int iz=(int)coords.z;
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
//        	results = results + tr_c(_gaugeIn.getLinkDagger(GInd::getSiteMu(site,0))*_gaugeIn.getLink(GInd::getSiteMu(GInd::getSiteFull((size_t)ix, (size_t)iy, (size_t)iz, (size_t)(it+tt)),1)));
                results = results +  tr_c(_gaugeIn.getLinkDagger(GInd::getSiteMu(GInd::getSiteFull(coords.x,coords.y, coords.z, (size_t)(it+tt)),0))
                                         *_gaugeIn.getLink(GInd::getSiteMu(GInd::getSiteFull((size_t)ix, (size_t)iy, coords.z, (size_t)(it+tt)),1)));
	}

	return results/3.0;


//        return  conj(_spinorIn1.getElement(site).getElement0())*_spinorIn2.getElement(GInd::getSiteFull((size_t)ix, (size_t)iy, (size_t)iz, (size_t)it)).getElement0();
//        return  tr_c(_gaugeIn.getLinkDagger(GInd::getSiteMu(site,0))*_gaugeIn.getLink(GInd::getSiteMu(GInd::getSiteFull((size_t)ix, (size_t)iy, (size_t)iz, (size_t)it),1)))/3.0;

	}
	else{
		return COMPLEX(floatT) (0.0,0.0);
	}

    }

};


template<class floatT, size_t HaloDepth>
COMPLEX(floatT) gDotAlongXY( Gaugefield<floatT,true,HaloDepth> &gauge ,int shiftx, int shifty,  LatticeContainer<true,COMPLEX(floatT)> &redBase){

    typedef GIndexer<All,HaloDepth> GInd;
    /// Since we run the kernel on the spacelike volume only, elems need only be size d_vol3.
    const size_t elems = GInd::getLatData().vol3;
    redBase.adjustSize(elems);

    /// main call in this function
    redBase.template iterateOverSpatialBulk<All, HaloDepth>(DotAlongXYInterval<floatT,HaloDepth,All>(gauge,shiftx,shifty));

    /// Do the final reduction.
    COMPLEX(floatT) val;
    redBase.reduce(val, elems);

    /// This construction ensures you obtain the spacelike volume of the entire lattice, rather than just a sublattice.
    floatT vol=GInd::getLatData().globvol4;

    // normalize
    val /= (vol);

    return val;
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

//        gSite site = GInd::getSite(siteMu.isite);


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
   //     temp=SU3Accessor.getLink(GInd::getSiteMu(site, 3));
        temp=SU3Accessor.getLink(GInd::getSiteMu(GInd::getSite(ix, iy, iz, it), 3));

        /// Loop over N_tau direction.
        for (size_t itp = 1; itp < _length; itp++) {
          size_t itau=it+itp;
          if(itau >= Ntau){
             itau-=Ntau;
          }
          temp*=SU3Accessor.getLink(GInd::getSiteMu(GInd::getSite(ix, iy, iz, itau), 3));
        }

        /// tr_c is the complex trace.
//        result = tr_c(temp);

//        SU3Accessor.setLink(GInd::getSiteMu(GInd::getSite(ix, iy, iz, it), 0));


        return temp;

    }

    auto getAccessor() const
    {
        return *this;
    }

};

/// Function to compute the wilson line using the above struct CalcWilson.
template<class floatT, size_t HaloDepth>
void gWilson(Gaugefield<floatT,true,HaloDepth> &gauge , size_t length){

    gauge.template iterateOverBulkAtMu<0,256>(CalcWilson<floatT,HaloDepth>(gauge,length));
//    rootLogger.info(spinor.dotProduct(spinor));
    return;

}

template<class floatT, size_t HaloDepth>
void gMoveOne( Gaugefield<floatT,true,HaloDepth> &gauge , int direction, int up){

    /// move gauge field in mu=1 specified direction  and save it into mu=2
    if(direction == 0){
        if(up ==1){
            gauge.template iterateOverBulkAtMu<2,256>(ShiftVectorOne<floatT,HaloDepth,All,0,true>(gauge));
        }
        else{
            gauge.template iterateOverBulkAtMu<2,256>(ShiftVectorOne<floatT,HaloDepth,All,0,false>(gauge));
        }
    }

    if(direction == 1){
        if(up ==1){
            gauge.template iterateOverBulkAtMu<2,256>(ShiftVectorOne<floatT,HaloDepth,All,1,true>(gauge));
        }
        else{
            gauge.template iterateOverBulkAtMu<2,256>(ShiftVectorOne<floatT,HaloDepth,All,1,false>(gauge));
        }
    }

    if(direction == 2){
        if(up ==1){
            gauge.template iterateOverBulkAtMu<2,256>(ShiftVectorOne<floatT,HaloDepth,All,2,true>(gauge));
        }
        else{
            gauge.template iterateOverBulkAtMu<2,256>(ShiftVectorOne<floatT,HaloDepth,All,2,false>(gauge));
        }
    }

    // copy back from mu=2 to mu=1
    gauge.template iterateOverBulkAtMu<1,256>(CopyFromMu<floatT,HaloDepth,All,2>(gauge));


    return;

}


int main(int argc, char *argv[]) {

    /// Controls whether DEBUG statements are shown as it runs; could also set to INFO, which is less verbose.
    stdLogger.setVerbosity(INFO);

    /// Initialize parameter class.
//    LatticeParameters param;
    RhmcParameters param;
    /// Initialize the Lattice dimension.
//    const int LatDim[] = {96,96,96, 32}; // {Ns,Ns,Ns,Ntau}

    /// Number of sublattices in each direction.
//    const int NodeDim[] = {1, 1, 8, 1};
//    if(NodeDim[3] != 1){
 //       rootLogger.error("WilsonLines does not allow partitions in time direction");
 //       exit(1);
 //   }

    /// Pass these dimensions to the parameter class.
//    param.latDim.set(LatDim);
//    param.nodeDim.set(NodeDim);


    /// Initialize a timer.
    StopWatch<true> timer;

    /// Initialize the CommunicationBase.
    CommunicationBase commBase(&argc, &argv);

    param.readfile(commBase, "../parameter/test.param", argc, argv);


    commBase.init(param.nodeDim());

    cout << param.nodeDim[0] << " param 0 " <<  param.nodeDim[1] << " param 1 " << param.nodeDim[2] << " param 2 " << param.nodeDim[3] << " param 3 " <<endl;

    /// Set the HaloDepth.
    const size_t HaloDepth = 2;

    rootLogger.info("Initialize Lattice");

    /// Initialize the Lattice class.
    initIndexer(HaloDepth,param,commBase);

    /// Initialize the Gaugefield.
    rootLogger.info("Initialize Gaugefield");
    Gaugefield<PREC,true,HaloDepth> gauge(commBase);

    /// Initialize gaugefield with unit-matrices.
    gauge.one();

    std::string gauge_file;

    // load gauge file, 0 start from 1, 1 and 2 load file, 2 will also gauge fix
    if (param.load_conf() == 0)
    {
        rootLogger.info("Starting from unit configuration");
        gauge.one();
    }
    else if(param.load_conf() == 2 || param.load_conf() == 1)
    {
//        gauge_file = param.gauge_file() + std::to_string(param.confnumber());
        rootLogger.info("Starting from configuration: " ,  gauge_file);
//        gauge.readconf_nersc(gauge_file);
//	gauge.readconf_nersc("../test/l328f21b6285m0009875m0790a_019.995");
	cout << param.gauge_file() << endl;
	gauge.readconf_nersc(param.gauge_file());

    }

    /// Exchange Halos
    gauge.updateAll();


    /// Initialize ReductionBase.
//    ReductionBase<true,COMPLEX(PREC)> redBase(commBase);
    LatticeContainer<true,COMPLEX(PREC)> redBase(commBase);

    /// We need to tell the Reductionbase how large our array will be. Again it runs on the spacelike volume only,
    /// so make sure you adjust this parameter accordingly, so that you don't waste memory.
    typedef GIndexer<All,HaloDepth> GInd;
    redBase.adjustSize(GInd::getLatData().vol4);
    rootLogger.info("volume size " ,  GInd::getLatData().globvol4);
    /// Read a configuration from hard drive. For the given configuration you should find
 //   rootLogger.info("Read configuration");
//    gauge.readconf_nersc("../test_conf/l328f21b6285m0009875m0790a_019.995");


//    std::string gauge_file;
//    gauge_file = param.gauge_file() + std::to_string(param.confnumber());
//    rootLogger.info("Starting from configuration: " ,  gauge_file);
//    gauge.readconf_nersc(gauge_file);


///////////// gauge fixing

    if(param.load_conf() ==2){
    GaugeFixing<PREC,true,HaloDepth>    GFixing(gauge);
    int ngfstep=0;
    PREC gftheta=1e10;
    const PREC gtol=1e-6;          /// When theta falls below this number, stop...
    const int ngfstepMAX=9000;     /// ...or stop after a fixed number of steps; this way the program doesn't get stuck.
    const int nunit=20;            /// Re-unitarize every 20 steps.
    while ( (ngfstep<ngfstepMAX) && (gftheta>gtol) ) {
    /// Compute starting GF functional and update the lattice.
        GFixing.gaugefixOR();
    /// Due to the nature of the update, we have to re-unitarize every so often.
        if ( (ngfstep%nunit) == 0 ) {
            gauge.su3latunitarize();
        }
    /// Re-calculate theta to determine whether we are sufficiently fixed.
        gftheta=GFixing.getTheta();
        ngfstep+=1;
    }
    gauge.su3latunitarize(); /// One final re-unitarization.


    }


    /// Start timer.
    timer.start();

    /// Exchange Halos
    gauge.updateAll();

    COMPLEX(PREC) dot;

///////////////// Structures needed for comparison

        CorrelatorTools<PREC,true,HaloDepth> corrTools;

        COMPLEX(PREC) corrComplex;
        COMPLEX(PREC) corrComplex2;
        COMPLEX(PREC) corrComplex3;

        Gaugefield<PREC,false,HaloDepth>  gaugeCPU(commBase);

        SU3Accessor<PREC> _gaugeCPU(gaugeCPU.getAccessor());

        CorrField<false,SU3<PREC>> CPUfield3(commBase, corrTools.vol4);
        CorrField<false,SU3<PREC>> CPUfield4(commBase, corrTools.vol4);
        Correlator<false,PREC> CPUnorm(commBase, corrTools.UAr2max);
        Correlator<false,COMPLEX(PREC)> CPUcorrComplex(commBase, corrTools.UAr2max);
        Correlator<false,COMPLEX(PREC)> CPUcorrComplexTemp(commBase, corrTools.UAr2max);

        LatticeContainerAccessor _CPUfield3(CPUfield3.getAccessor());
        LatticeContainerAccessor _CPUfield4(CPUfield4.getAccessor());

        LatticeContainerAccessor _CPUcorrComplex(CPUcorrComplex.getAccessor());
        LatticeContainerAccessor _CPUcorrComplexTemp(CPUcorrComplexTemp.getAccessor());

        Correlator<false,COMPLEX(PREC)> CPUresults(commBase, corrTools.UAr2max);
        LatticeContainerAccessor _CPUresults(CPUresults.getAccessor());
        Correlator<false,COMPLEX(PREC)> CPUnormR(commBase, corrTools.UAr2max);
        LatticeContainerAccessor _CPUnormR(CPUnormR.getAccessor());


///////////////



    ///
    timer.start();
    //// loop over length of wilson lines
    for(int length = 1; length < 2;length++){
    //for(int length = 1; length<GInd::getLatData().globLT+1;length++){

        /// calculate the wilson line starting from any spacetime point save in mu=0 direction
        gWilson<PREC,HaloDepth>(gauge, length);

        /// copy from mu=0 to mu=1
        gauge.template iterateOverBulkAtMu<1,256>(CopyFromMu<PREC,HaloDepth,All,0>(gauge));
        gauge.updateAll();
        //    dot = lines.dotProduct(shifted)/3.0/GInd::getLatData().globvol4;
        /// check that dot product is with conjugate, hack

        //    rootLogger.info(0 ,  " " ,  0 ,  " ",  0 ,  " " ,  length ,  " " ,  dot);

        // initial position x0=-1 due to adding dx in first line
        int x0 = -1;
        int y0 = 0;
        int z0 = 0;

        int dx = 1;
        int dy = 1;
        int dz = 1;

        for(size_t i = 0; i<GInd::getLatData().globvol3/2+GInd::getLatData().globLX*GInd::getLatData().globLY;i++){
            x0 += dx;

            if(x0 >= (int)GInd::getLatData().globLX || x0 <0){
                dx *= -1;
                x0 += dx;
                y0 += dy;
                if(y0 >= (int)GInd::getLatData().globLY|| y0 <0){
                    dy *= -1;
                    y0 += dy;
                    z0 += dz;
                    /// move mu=1 direction by dz
                    gMoveOne(gauge,2,dz);
                    gauge.updateAll();
                }
                else if(param.nodeDim[1]>1){
                    /// mode mu=1 direction by dy
                    gMoveOne(gauge,1,dy);
                    gauge.updateAll();
                }
            }

    // A(x).A(x+r)^dag along direction x and also y if not split on different GPU's
            if(param.nodeDim[1] == 1){
                 dot = gDotAlongXY(gauge,x0,y0,redBase);
            }
            else{
                 dot = gDotAlongXY(gauge,x0,0,redBase);
            }

            rootLogger.info(x0 ,  " " ,  y0 ,  " ",  z0 ,  " " ,  length ,  " " ,  dot);


            // save results
            if(length == 1){

	        int ir2 = 0;
	        if(x0 > (int)GInd::getLatData().globLX/2){
	            ir2 += (x0-(int)GInd::getLatData().globLX)*(x0-(int)GInd::getLatData().globLX);
	        }
	        else{
	            ir2 += x0*x0;
	        }

                if(y0 > (int)GInd::getLatData().globLY/2){
                    ir2 += (y0-(int)GInd::getLatData().globLY)*(y0-(int)GInd::getLatData().globLY);
                }
                else{
                    ir2 += y0*y0;
                }

                if(z0 > (int)GInd::getLatData().globLZ/2){
                    ir2 += (z0-(int)GInd::getLatData().globLZ)*(z0-(int)GInd::getLatData().globLZ);
                }
                else{
                    ir2 += z0*z0;
                }

                // factor for counting contributions
                // Initial factor 2 for symmetry between z and -z
                // double the factor if x or y = l/2 due to periodicity
                double factor = 1.0;

                if(z0 == (int)GInd::getLatData().globLZ/2 || z0 == 0){
                    factor = 0.5*factor;
                    if((y0 == (int)GInd::getLatData().globLY/2 || y0 == 0) && (x0 == (int)GInd::getLatData().globLX/2 || x0==0) ){
                        factor = 2.0*factor;
                    }
		}

                if( (z0 == (int)GInd::getLatData().globLZ/2 || z0==0) && (y0 == (int)GInd::getLatData().globLY/2 || y0 == 0) && (x0 == (int)GInd::getLatData().globLX/2 || x0==0) ){
		    factor = 0.5*factor;
                }

                _CPUresults.getValue<COMPLEX(PREC)>(ir2,corrComplex);
                corrComplex += factor*dot;
	        _CPUresults.setValue<COMPLEX(PREC)>(ir2,corrComplex);

//                rootLogger.info(x0 ,  " " ,  y0 ,  " ",  z0 ,  " " ,  ir2 ,  " " ,  dot ,  " " ,  corrComplex);

                _CPUnormR.getValue<COMPLEX(PREC)>(ir2,corrComplex);
                corrComplex += factor;
                _CPUnormR.setValue<COMPLEX(PREC)>(ir2, corrComplex);

                rootLogger.info(x0 ,  " " ,  y0 ,  " ",  z0 ,  " " ,  length ,  " " ,  dot ,  " " ,  corrComplex ,  " " ,  factor ,  " " ,  i ,  " r2 " ,  ir2);

            }

        }

    }

    timer.stop();
    rootLogger.info("Time for operators: " ,  timer);
    /// stop timer and print time
    timer.stop();

    /////// Comparison with correlator class


    gaugeCPU = gauge;
    corrTools.createNorm("spatial",commBase);

//    for(int m=0; m<corrTools.vol4; m++) {
//        _CPUfield3.setValue(m, _gaugeCPU.getLink(GInd::getSiteMu(m,0)));
//        _CPUfield4.setValue(m, _gaugeCPU.getLink(GInd::getSiteMu(m,0)));
//    }

    for(int ir2=0; ir2<corrTools.UAr2max+1; ir2++) {
        _CPUcorrComplex.setValue<COMPLEX(PREC)>(ir2,0.0);
    }


    for(size_t itau = 0; itau<GInd::getLatData().globLT;itau++){

    for(size_t it = 0; it<GInd::getLatData().globLT;it++){
        for(size_t ix = 0; ix<GInd::getLatData().globLX;ix++){
            for(size_t iy = 0; iy<GInd::getLatData().globLY;iy++){
                for(size_t iz = 0; iz<GInd::getLatData().globLZ;iz++){
                _CPUfield3.setValue(GInd::getSite(ix,iy,iz,it).isite, _gaugeCPU.getLink(GInd::getSiteMu(GInd::getSite(ix,iy,iz,itau).isite,0)));
                _CPUfield4.setValue(GInd::getSite(ix,iy,iz,it).isite, _gaugeCPU.getLink(GInd::getSiteMu(GInd::getSite(ix,iy,iz,itau).isite,0)));
                }
            }
        }
    }


    corrTools.correlateAt<SU3<PREC>,COMPLEX(PREC),trAxBt<PREC>>("spatial", CPUfield3, CPUfield4, CPUnorm, CPUcorrComplexTemp, true);

    for(int ir2=0; ir2<corrTools.UAr2max+1; ir2++) {
        _CPUcorrComplex.getValue<COMPLEX(PREC)>(ir2,corrComplex);
        _CPUcorrComplexTemp.getValue<COMPLEX(PREC)>(ir2,corrComplex2);
        corrComplex = corrComplex + corrComplex2;
        _CPUcorrComplex.setValue<COMPLEX(PREC)>(ir2,corrComplex);
    }

    }

    double difference = 0.0;
    for(int ir2=0; ir2<corrTools.UAr2max+1; ir2++) {

        _CPUcorrComplex.getValue<COMPLEX(PREC)>(ir2,corrComplex);
        _CPUresults.getValue<COMPLEX(PREC)>(ir2,corrComplex2);
        _CPUnormR.getValue<COMPLEX(PREC)>(ir2,corrComplex3);
	if(real(corrComplex3) > 0.1){
	    corrComplex2 = corrComplex2/real(corrComplex3);
	}

    	rootLogger.info(ir2 ,  " " ,  corrComplex/3.0/GInd::getLatData().globLT ,  " , " ,  corrComplex2 ,  "    " ,  real(corrComplex/3.0/GInd::getLatData().globLT - corrComplex2) ,  "   Norm " ,  real(corrComplex3));
    difference += abs(real(corrComplex/3.0/GInd::getLatData().globLT - corrComplex2));
    if(abs(real(corrComplex/3.0/GInd::getLatData().globLT - corrComplex2)) > 1e-10){
	   rootLogger.info(" Error, large difference");
	}
    }

   rootLogger.info("Total difference between all resuls are "  ,  difference);

    ///////



    return 0;
}

