/* 
 * main_flt.cu
 *
 * v1.0: Sodbileg Ch., Battogtokh P. & Enkhtuya G.
 *
 * Version History:
 *	v1.0--Working on 1
 *
 * About the program:
 *	The code can compute flux tube profiles produced by 
 *	quark–antiquark pair with reference point method reading a configuration. 
 *	It can create 9 output files automatically. Output files: 
 *	- flt_beta_fields --> 
 *		correlation of the Polyakov loops (L(0), L+(R)) 
 *		with plaquette (P(mu,nu)) --> <L(0) P(mu,nu) L+(R)>
 *	- flt_beta_fields_ref -->
 *		correlation of the reference plaquette (P_{ref})
 *		with Polyakov loops (L(0), L+(R)) --> <L(0) P_{ref} L+(R)>
 *	- flt_beta_plc_sum -->
 *		correlation of the Polyakov loops --> <L(0) L+(R)>
 *	Then we can compute fluxtube profiles using these data via
 *	f_{mu,nu} = (beta/a^4) *
 *	  * [(<L(0) P(mu,nu) L+(R)> - <L(0) P_{ref} L+(R)>) / <L(0) L+(R)>].
 *
 * How to use the code:
 *	- to change separation distances R in the source code
 *	  before compile the program... (R --> $dist, dist = dist + $1). 
 *	  (maybe it should be defined in parameter file or not!!!)
 *	- you should change parameters in parameter/flt.parameter.
 *	- you should use bash code to measure flux tube as follows:
for((j=101; j<=125; j++)); do for ((i=50;i<=2300;i=i+50)); do mpiexec -np 1 ./fluxtube Gaugefile="/home/battogtokh/configurations/flowed_l328f21_$((j))/flowed_$((j))_$((i))FT00250_s032t08_b06423"; done; done
 *
 */

#include "../SIMULATeQCD.h"
#include "../modules/observables/PolyakovLoop.h"

#define PREC double 
#define MY_BLOCKSIZE 256

template<class floatT,size_t HaloDepth>
struct CalcPloop{

    /// Gauge accessor to access the gauge field.
    gaugeAccessor<floatT> gaugeAccessor;
    MemoryAccessor _ploop;
    /// Constructor to initialize all necessary members.
    CalcPloop(Gaugefield<floatT,true,HaloDepth> &gauge, MemoryAccessor ploop) : gaugeAccessor(gauge.getAccessor()), _ploop(ploop) {} 
    /// This is the operator that is called inside the Kernel. We set the type to GCOMPLEX(floatT) because the
    /// Polyakov loop is complex valued.
    __device__ __host__ GCOMPLEX(floatT) operator()(gSite site) {
	
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
	
	size_t pind = site.isite;
        /// Start off at this site, pointing in N_tau direction.
        temp=gsu3_one<floatT>();

        /// Loop over N_tau direction.
        for (size_t itp = 0; itp < Ntau; itp++) {
          size_t itau=it+itp;
          temp*=gaugeAccessor.getLink(GInd::getSiteMu(GInd::getSite(ix, iy, iz, itau), 3));
        }
        /// tr_c is the complex trace.
        result = tr_c(temp) / (floatT) 3.0;
	_ploop.setValue<GCOMPLEX(floatT)>(pind,result);
//        printf("%u %f %f \n", pind, result.cREAL, result.cIMAG );
        return result;
    }
};

template<class floatT, size_t HaloDepth>
struct plaq4 {
    //Gauge accessor to access the gauge field
    gaugeAccessor<floatT> acc;

    plaq4(Gaugefield<floatT,true,HaloDepth> &gauge) : acc(gauge.getAccessor()) {}

    __device__ __host__  inline GCOMPLEX(floatT) operator()(gSite site, int mu, int nu) {
       
	/// Define an SU(3) matrix and initialize result variable.
        typedef GIndexer<All,HaloDepth> GInd;

        /// Define an SU(3) matrix and initialize result variable.
        GSU3<floatT> temp;
        GCOMPLEX(floatT) result = 0.0;

        temp = Plaq_P<floatT, HaloDepth>(acc, site, mu, nu);
        result += tr_c(temp);
        temp = Plaq_Q<floatT, HaloDepth>(acc, site, mu, nu);
        result += tr_c(temp);
        temp = Plaq_R<floatT, HaloDepth>(acc, site, mu, nu);
        result += tr_c(temp);
        temp = Plaq_S<floatT, HaloDepth>(acc, site, mu, nu);
        result += tr_c(temp);

        return result/(floatT)(12.0);
    }
};


template<class floatT,size_t HaloDepth>
struct calcTmpPL {

/// Gauge accessor to access the gauge field.
    gaugeAccessor<floatT> acc;  
    MemoryAccessor _tmp_pl01;
    MemoryAccessor _tmp_pl02;
    MemoryAccessor _tmp_pl03;
    MemoryAccessor _tmp_pl12;
    MemoryAccessor _tmp_pl13;
    MemoryAccessor _tmp_pl23;
    plaq4<floatT,HaloDepth> plq4;    
    
/// Constructor to initialize all necessary members.
    calcTmpPL(Gaugefield<floatT,true,HaloDepth> &gauge, MemoryAccessor tmp_pl01,
      MemoryAccessor tmp_pl02,
      MemoryAccessor tmp_pl03, 
      MemoryAccessor tmp_pl12, 
      MemoryAccessor tmp_pl13, 
      MemoryAccessor tmp_pl23) : 
    acc(gauge.getAccessor()), 
    _tmp_pl01(tmp_pl01), 
    _tmp_pl02(tmp_pl02), 
    _tmp_pl03(tmp_pl03), 
    _tmp_pl12(tmp_pl12), 
    _tmp_pl13(tmp_pl13), 
    _tmp_pl23(tmp_pl23), plq4(gauge) {}
    
/// This is the operator that is called inside the Kernel. We set the type to GCOMPLEX(floatT) because the
    /// Polyakov loop is complex valued.
	__device__ __host__ GCOMPLEX(floatT) operator()(gSite site) {
        
	typedef GIndexer<All,HaloDepth> GInd;

        /// Extension in timelike direction. In general unsigned declarations reduce compiler warnings.
        const size_t Ntau=GInd::getLatData().lt;

        /// Get coordinates.
        sitexyzt coords=site.coord;
        size_t ix=coords.x;
        size_t iy=coords.y;
        size_t iz=coords.z;
        size_t it=coords.t;

        size_t pind = site.isite;

	GCOMPLEX(floatT) result01 = 0.0, result02 = 0.0, result03 = 0.0, result12 = 0.0, result13 = 0.0, result23 = 0.0;


        for (size_t itp = 0; itp < Ntau; itp++) {

            size_t itau=it+itp;
            gSite sitel = GInd::getSite(ix,iy,iz,itau);
            
	        result01 += plq4(sitel, 0, 1);
            result02 += plq4(sitel, 0, 2);
            result03 += plq4(sitel, 0, 3);
            result12 += plq4(sitel, 1, 2);
            result13 += plq4(sitel, 1, 3);
            result23 += plq4(sitel, 2, 3);
        }

        result01/=(floatT)(Ntau);
        result02/=(floatT)(Ntau);
        result03/=(floatT)(Ntau);
        result12/=(floatT)(Ntau);
        result13/=(floatT)(Ntau);
        result23/=(floatT)(Ntau);

        _tmp_pl01.setValue<GCOMPLEX(floatT)>(pind,result01);
        _tmp_pl02.setValue<GCOMPLEX(floatT)>(pind,result02);
        _tmp_pl03.setValue<GCOMPLEX(floatT)>(pind,result03);
        _tmp_pl12.setValue<GCOMPLEX(floatT)>(pind,result12);
        _tmp_pl13.setValue<GCOMPLEX(floatT)>(pind,result13);
        _tmp_pl23.setValue<GCOMPLEX(floatT)>(pind,result23);

       // printf("%f %f %f %f %f %f \n", result01.cREAL, result02.cREAL, result03.cREAL, result12.cREAL, result13.cREAL, result23.cREAL);
		
	return 0;
    }
};


template<class floatT,size_t HaloDepth>
struct calcPPCORR
{
    MemoryAccessor _ploop;
    MemoryAccessor _tmp_pl01;
    MemoryAccessor _tmp_pl02;
    MemoryAccessor _tmp_pl03;
    MemoryAccessor _tmp_pl12;
    MemoryAccessor _tmp_pl13;
    MemoryAccessor _tmp_pl23;
    MemoryAccessor _tmp_plc01;
    MemoryAccessor _tmp_plc02;
    MemoryAccessor _tmp_plc03;
    MemoryAccessor _tmp_plc12;
    MemoryAccessor _tmp_plc13;
    MemoryAccessor _tmp_plc23;
    size_t dist;
    calcPPCORR(MemoryAccessor ploop,
                MemoryAccessor tmp_pl01,
                MemoryAccessor tmp_pl02,
                MemoryAccessor tmp_pl03,
                MemoryAccessor tmp_pl12,
                MemoryAccessor tmp_pl13,
                MemoryAccessor tmp_pl23,
                MemoryAccessor tmp_plc01,
                MemoryAccessor tmp_plc02,
                MemoryAccessor tmp_plc03,
                MemoryAccessor tmp_plc12,
                MemoryAccessor tmp_plc13,
                MemoryAccessor tmp_plc23,
                size_t dist) : _ploop(ploop), _tmp_pl01(tmp_pl01), _tmp_pl02(tmp_pl02), _tmp_pl03(tmp_pl03), _tmp_pl12(tmp_pl12), _tmp_pl13(tmp_pl13), _tmp_pl23(tmp_pl23), _tmp_plc01(tmp_plc01), _tmp_plc02(tmp_plc02), _tmp_plc03(tmp_plc03), _tmp_plc12(tmp_plc12), _tmp_plc13(tmp_plc13), _tmp_plc23(tmp_plc23), dist(dist) {}

    __device__ __host__ GCOMPLEX(floatT) operator()(gSite site) {

        typedef GIndexer<All,HaloDepth> GInd;

        size_t Nx       = (int)GInd::getLatData().globLX; /// globL% is a size_t object, but in practice N% will never be
        size_t Ny       = (int)GInd::getLatData().globLY; /// large enough for that to matter. Furthermore N% variables are
        size_t Nz       = (int)GInd::getLatData().globLZ; /// used in statements involving subtraction.
        size_t Nt       = (int)GInd::getLatData().globLT;
        size_t vol3  = GInd::getLatData().vol3;
        size_t dxmax    = Nx/4+1; /// N%/4 is set as the maximum distance in the off-axis direction because diagonal
        size_t dymax    = Ny/4+1; /// correlations become noisy for large distances. N%/4 is somehow large enough.
        size_t dzmax    = Nz/4+1;
        size_t distmax = dist + 2*dxmax - 3;
        size_t pindex, site1, site2, site3, site31, site32, site33;
    
        GCOMPLEX(floatT) pol_1, pol_2, val; 
        GCOMPLEX(floatT) pl_01, pl1_01, pl2_01, pl3_01;
        GCOMPLEX(floatT) pl_02, pl1_02, pl2_02, pl3_02;
        GCOMPLEX(floatT) pl_03, pl1_03, pl2_03, pl3_03;
        GCOMPLEX(floatT) pl_12, pl1_12, pl2_12, pl3_12;
        GCOMPLEX(floatT) pl_13, pl1_13, pl2_13, pl3_13;
        GCOMPLEX(floatT) pl_23, pl1_23, pl2_23, pl3_23;
    
        /// Get coordinates.
        sitexyzt coords=site.coord;
        size_t ix=coords.x;
        size_t iy=coords.y;
        size_t iz=coords.z;
        //size_t it=coords.t;
	
        GCOMPLEX(floatT) result01, result02, result03, result12, result13, result23;

        GCOMPLEX(floatT) sum = 0.0;

    if (ix <= distmax && iy <= dymax && iz <= dzmax){

        pindex =ix + iy*Nx + iz*Nx*(dymax+1);

        result01 = 0.0; result02 = 0.0; result03 = 0.0; result12 = 0.0; result13 = 0.0; result23 = 0.0;

	//pol_1=0.0; pol_2=0.0; val=0.0;
        
	for(size_t tx=0;tx<Nx;tx++)
            for(size_t ty=0;ty<Ny;ty++)
                for(size_t tz=0;tz<Nz;tz++){

                    site1 = GInd::getSiteSpatial( tx                    , ty        , tz        , 0).isite;
                    site2 = GInd::getSiteSpatial((tx+dist)%Nx           , ty        , tz        , 0).isite;
                    site3 = GInd::getSiteSpatial((tx-(dxmax-1)+Nx+ix)%Nx, (ty+iy)%Ny, (tz+iz)%Nz, 0).isite;

                    _ploop.getValue<GCOMPLEX(floatT)>(site1, pol_1);
                    _ploop.getValue<GCOMPLEX(floatT)>(site2, pol_2);
                    
                    val = pol_1 * conj(pol_2);

                    _tmp_pl01.getValue<GCOMPLEX(floatT)>(site3, pl_01);
                    _tmp_pl02.getValue<GCOMPLEX(floatT)>(site3, pl_02);
                    _tmp_pl03.getValue<GCOMPLEX(floatT)>(site3, pl_03);
                    _tmp_pl12.getValue<GCOMPLEX(floatT)>(site3, pl_12);
                    _tmp_pl13.getValue<GCOMPLEX(floatT)>(site3, pl_13);
                    _tmp_pl23.getValue<GCOMPLEX(floatT)>(site3, pl_23);

                    result01 += val * pl_01;
                    result02 += val * pl_02;
                    result03 += val * pl_03;
                    result12 += val * pl_12;
                    result13 += val * pl_13;
                    result23 += val * pl_23;

                    site31 = GInd::getSiteSpatial((tx-(dxmax-1)+Nx+ix)%Nx, (ty+iy)%Ny, (tz-iz+Nz)%Nz, 0).isite;

                    _tmp_pl01.getValue<GCOMPLEX(floatT)>(site31, pl1_01);
                    _tmp_pl02.getValue<GCOMPLEX(floatT)>(site31, pl1_02);
                    _tmp_pl03.getValue<GCOMPLEX(floatT)>(site31, pl1_03);
                    _tmp_pl12.getValue<GCOMPLEX(floatT)>(site31, pl1_12);
                    _tmp_pl13.getValue<GCOMPLEX(floatT)>(site31, pl1_13);
                    _tmp_pl23.getValue<GCOMPLEX(floatT)>(site31, pl1_23);

                    result01 += val * pl1_01;
                    result02 += val * pl1_02;
                    result03 += val * pl1_03;
                    result12 += val * pl1_12;
                    result13 += val * pl1_13;
                    result23 += val * pl1_23;

                    site32 = GInd::getSiteSpatial((tx-(dxmax-1)+Nx+ix)%Nx, (ty-iy+Ny)%Ny, (tz+iz)%Nz, 0).isite;

                    _tmp_pl01.getValue<GCOMPLEX(floatT)>(site32, pl2_01);
                    _tmp_pl02.getValue<GCOMPLEX(floatT)>(site32, pl2_02);
                    _tmp_pl03.getValue<GCOMPLEX(floatT)>(site32, pl2_03);
                    _tmp_pl12.getValue<GCOMPLEX(floatT)>(site32, pl2_12);
                    _tmp_pl13.getValue<GCOMPLEX(floatT)>(site32, pl2_13);
                    _tmp_pl23.getValue<GCOMPLEX(floatT)>(site32, pl2_23);

                    result01 += val * pl2_01;
                    result02 += val * pl2_02;
                    result03 += val * pl2_03;
                    result12 += val * pl2_12;
                    result13 += val * pl2_13;
                    result23 += val * pl2_23;

                    site33 = GInd::getSiteSpatial((tx-(dxmax-1)+Nx+ix)%Nx, (ty-iy+Ny)%Ny, (tz-iz+Nz)%Nz, 0).isite;

                    _tmp_pl01.getValue<GCOMPLEX(floatT)>(site33, pl3_01);
                    _tmp_pl02.getValue<GCOMPLEX(floatT)>(site33, pl3_02);
                    _tmp_pl03.getValue<GCOMPLEX(floatT)>(site33, pl3_03);
                    _tmp_pl12.getValue<GCOMPLEX(floatT)>(site33, pl3_12);
                    _tmp_pl13.getValue<GCOMPLEX(floatT)>(site33, pl3_13);
                    _tmp_pl23.getValue<GCOMPLEX(floatT)>(site33, pl3_23);

                    result01 += val * pl3_01;
                    result02 += val * pl3_02;
                    result03 += val * pl3_03;
                    result12 += val * pl3_12;
                    result13 += val * pl3_13;
                    result23 += val * pl3_23;

                    sum += pol_1 * conj(pol_2);;

                }

                result01 /= (floatT)(4.0 * vol3);
                result02 /= (floatT)(4.0 * vol3);
                result03 /= (floatT)(4.0 * vol3);
                result12 /= (floatT)(4.0 * vol3);
                result13 /= (floatT)(4.0 * vol3);
                result23 /= (floatT)(4.0 * vol3);
	
                _tmp_plc01.setValue<GCOMPLEX(floatT)>(pindex,result01);
                _tmp_plc02.setValue<GCOMPLEX(floatT)>(pindex,result02);
                _tmp_plc03.setValue<GCOMPLEX(floatT)>(pindex,result03);
                _tmp_plc12.setValue<GCOMPLEX(floatT)>(pindex,result12);
                _tmp_plc13.setValue<GCOMPLEX(floatT)>(pindex,result13);
                _tmp_plc23.setValue<GCOMPLEX(floatT)>(pindex,result23);

                //printf("%u %f %f %f \n", pindex, result01.cREAL, result02.cREAL, sum.cREAL);
        }

	return sum;

    }

};

template<class floatT,size_t HaloDepth>
struct CalcReferencePoint{

    /// Gauge accessor to access the gauge field.
    MemoryAccessor _ploop;
    MemoryAccessor _tmp_pl01;
    int dist;
    /// Constructor to initialize all necessary members.
    CalcReferencePoint(MemoryAccessor ploop,
                MemoryAccessor tmp_pl01, int dist) :  _ploop(ploop), _tmp_pl01(tmp_pl01), dist(dist) {} 
    /// This is the operator that is called inside the Kernel. We set the type to GCOMPLEX(floatT) because the
    /// Polyakov loop is complex valued.
   
    __device__ __host__ GCOMPLEX(floatT) operator()(gSite site) {
   
        typedef GIndexer<All,HaloDepth> GInd;

        size_t site1, site2, site3;

        GCOMPLEX(floatT) pol_1, pol_2, val;
  
        GCOMPLEX(floatT) pl_01;
             
        int Nx       = (int)GInd::getLatData().globLX; /// globL% is a size_t object, but in practice N% will never be
        int Ny       = (int)GInd::getLatData().globLY; /// large enough for that to matter. Furthermore N% variables are
        int Nz       = (int)GInd::getLatData().globLZ; /// used in statements involving subtraction.
        
        size_t vol3  = GInd::getLatData().globvol3;
 
        GCOMPLEX(floatT) result01;

        //GCOMPLEX(floatT) pl_01, pl_02, pl_03, pl_12, pl_13, pl_23;
        /// Extension in timelike direction. In general unsigned declarations reduce compiler warnings.
        const size_t Ntau=GInd::getLatData().lt;

                /// Get coordinates.
        sitexyzt coords=site.coord;
        size_t tx=coords.x;
        size_t ty=coords.y;
        size_t tz=coords.z;

        site1 = GInd::getSiteSpatial( tx                    , ty        , tz        , 0).isite;
        site2 = GInd::getSiteSpatial((tx+dist)%Nx           , ty        , tz        , 0).isite;
        site3 = GInd::getSiteSpatial((tx+dist+(Nx/4))%Nx, (ty+(Ny/2))%Ny, (tz+(Nz/2))%Nz, 0).isite;

        _ploop.getValue<GCOMPLEX(floatT)>(site1, pol_1);
        _ploop.getValue<GCOMPLEX(floatT)>(site2, pol_2);
                    
        val = pol_1 * conj(pol_2);

        _tmp_pl01.getValue<GCOMPLEX(floatT)>(site3, pl_01);

        result01 = val * pl_01;

        //printf("%u %f %f %f \n", dx, result01.cREAL, result02.cREAL, result23.cREAL);

    return result01;
    
    }
};

template<class floatT,size_t HaloDepth>
struct averOS
{
    MemoryAccessor _tmp_plc01;
    MemoryAccessor _tmp_plc02;
    MemoryAccessor _tmp_plc03;
    MemoryAccessor _tmp_plc12;
    MemoryAccessor _tmp_plc13;
    MemoryAccessor _tmp_plc23;
    MemoryAccessor _old_tmp_plc01;
    MemoryAccessor _old_tmp_plc02;
    MemoryAccessor _old_tmp_plc03;
    MemoryAccessor _old_tmp_plc12;
    MemoryAccessor _old_tmp_plc13;
    MemoryAccessor _old_tmp_plc23;
    int dist;
               
    averOS(     MemoryAccessor tmp_plc01,
                MemoryAccessor tmp_plc02,
                MemoryAccessor tmp_plc03,
                MemoryAccessor tmp_plc12,
                MemoryAccessor tmp_plc13,
                MemoryAccessor tmp_plc23,
                MemoryAccessor old_tmp_plc01,
                MemoryAccessor old_tmp_plc02,
                MemoryAccessor old_tmp_plc03,
                MemoryAccessor old_tmp_plc12,
                MemoryAccessor old_tmp_plc13,
                MemoryAccessor old_tmp_plc23,
                int dist) : _tmp_plc01(tmp_plc01), _tmp_plc02(tmp_plc02), _tmp_plc03(tmp_plc03), _tmp_plc12(tmp_plc12), _tmp_plc13(tmp_plc13), _tmp_plc23(tmp_plc23), _old_tmp_plc01(old_tmp_plc01), _old_tmp_plc02(old_tmp_plc02), _old_tmp_plc03(old_tmp_plc03), _old_tmp_plc12(old_tmp_plc12), _old_tmp_plc13(old_tmp_plc13), _old_tmp_plc23(old_tmp_plc23), dist(dist) {}

    __device__ __host__ GCOMPLEX(floatT) operator()(gSite site) {

        typedef GIndexer<All,HaloDepth> GInd;
        int Nx       = (int)GInd::getLatData().globLX; /// globL% is a size_t object, but in practice N% will never be
        int Ny       = (int)GInd::getLatData().globLY; /// large enough for that to matter. Furthermore N% variables are
        int Nz       = (int)GInd::getLatData().globLZ; /// used in statements involving subtraction.
        int Nt       = (int)GInd::getLatData().globLT;
        size_t vol3  = GInd::getLatData().vol3;
        size_t dxmax    = Nx/4+1; /// N%/4 is set as the maximum distance in the off-axis direction because diagonal
        size_t dymax    = Ny/4+1; /// correlations become noisy for large distances. N%/4 is somehow large enough.
        size_t dzmax    = Nz/4+1;
        size_t distmax = dist + 2*dxmax - 3;
        size_t pindex, site1;

        GCOMPLEX(floatT) pl1_01, pl1_02, pl1_03, pl1_12, pl1_13, pl1_23;      
        GCOMPLEX(floatT) pl2_01, pl2_02, pl2_03, pl2_12, pl2_13, pl2_23;
    
        /// Get coordinates.
        sitexyzt coords=site.coord;
        size_t ix=coords.x;
        size_t iy=coords.y;
        size_t iz=coords.z;
        //size_t it=coords.t;
    
        GCOMPLEX(floatT) result01, result02, result03, result12, result13, result23;

        if (ix <= distmax && iy <= dymax && iz <= dzmax){

            pindex = ix + iy * Nx + iz * Nx * (dymax + 1);
            site1 = ix + iz * Nx + iy * Nx * (dymax + 1);

            _old_tmp_plc01.getValue<GCOMPLEX(floatT)>(site1, pl2_01);
            _old_tmp_plc02.getValue<GCOMPLEX(floatT)>(site1, pl2_02);
            _old_tmp_plc03.getValue<GCOMPLEX(floatT)>(site1, pl2_03);
            _old_tmp_plc12.getValue<GCOMPLEX(floatT)>(site1, pl2_12);
            _old_tmp_plc13.getValue<GCOMPLEX(floatT)>(site1, pl2_13);
            _old_tmp_plc23.getValue<GCOMPLEX(floatT)>(site1, pl2_23);

            _old_tmp_plc01.getValue<GCOMPLEX(floatT)>(pindex, pl1_01);
            _old_tmp_plc02.getValue<GCOMPLEX(floatT)>(pindex, pl1_02);
            _old_tmp_plc03.getValue<GCOMPLEX(floatT)>(pindex, pl1_03);
            _old_tmp_plc12.getValue<GCOMPLEX(floatT)>(pindex, pl1_12);
            _old_tmp_plc13.getValue<GCOMPLEX(floatT)>(pindex, pl1_13);
            _old_tmp_plc23.getValue<GCOMPLEX(floatT)>(pindex, pl1_23);

            result01 = pl1_01 + pl2_01;
            result02 = pl1_02 + pl2_02;
            result03 = pl1_03 + pl2_03;
            result12 = pl1_12 + pl2_12;
            result13 = pl1_13 + pl2_13;
            result23 = pl1_23 + pl2_23;

            result01 /= 2.0;
            result02 /= 2.0;
            result03 /= 2.0;
            result12 /= 2.0;
            result13 /= 2.0;
            result23 /= 2.0;

            _tmp_plc01.setValue<GCOMPLEX(floatT)>(pindex,result01);
            _tmp_plc02.setValue<GCOMPLEX(floatT)>(pindex,result02);
            _tmp_plc03.setValue<GCOMPLEX(floatT)>(pindex,result03);
            _tmp_plc12.setValue<GCOMPLEX(floatT)>(pindex,result12);
            _tmp_plc13.setValue<GCOMPLEX(floatT)>(pindex,result13);
            _tmp_plc23.setValue<GCOMPLEX(floatT)>(pindex,result23);
  
            //_tmp_plc01.setValue<GCOMPLEX(floatT)>(site1,result01);
            //_tmp_plc02.setValue<GCOMPLEX(floatT)>(site1,result02);
            //_tmp_plc03.setValue<GCOMPLEX(floatT)>(site1,result03);
            //_tmp_plc12.setValue<GCOMPLEX(floatT)>(site1,result12);
            //_tmp_plc13.setValue<GCOMPLEX(floatT)>(site1,result13);
            //_tmp_plc23.setValue<GCOMPLEX(floatT)>(site1,result23);
    
            }

    return 0;

    }

};

template<class floatT, size_t HaloDepth>
GCOMPLEX(floatT) flt_meas(Gaugefield<floatT,true,HaloDepth> &gauge, LatticeContainer<true,GCOMPLEX(floatT)> &redBase, CommunicationBase &commBase, double beta){

    typedef GIndexer<All,HaloDepth> GInd;

    char            flt_name_2b_ort[200];
    char            flt_name_e_par[200];
    char            flt_name_b_par[200];
    char            flt_name_2e_ort[200];
    char            flt_name_plc_sum[200];
    char            flt_name_2b_ort_ref[200];
    char            flt_name_e_par_ref[200];
    char            flt_name_b_par_ref[200];
    char            flt_name_2e_ort_ref[200];

    FILE *flt_out, *flt_out1, *flt_out2, *flt_out3, *flt_out4;

    sprintf(flt_name_2b_ort,"flt_%0.3f_2b_ort", beta);
    sprintf(flt_name_e_par,"flt_%0.3f_e_par", beta);
    sprintf(flt_name_b_par,"flt_%0.3f_b_par", beta);  
    sprintf(flt_name_2e_ort,"flt_%0.3f_2e_ort", beta);
    sprintf(flt_name_plc_sum,"flt_%0.3f_plc_sum", beta);
    sprintf(flt_name_2b_ort_ref,"flt_%0.3f_2b_ort_ref", beta);
    sprintf(flt_name_e_par_ref,"flt_%0.3f_e_par_ref", beta);
    sprintf(flt_name_b_par_ref,"flt_%0.3f_b_par_ref", beta);
    sprintf(flt_name_2e_ort_ref,"flt_%0.3f_2e_ort_ref", beta);

    flt_out1 = fopen(flt_name_2b_ort, "a");
    flt_out2 = fopen(flt_name_e_par, "a");
    flt_out3 = fopen(flt_name_b_par, "a");
    flt_out4 = fopen(flt_name_2e_ort, "a");

    fprintf(flt_out1, "%d \n", 1);
    fprintf(flt_out2, "%d \n", 1);
    fprintf(flt_out3, "%d \n", 1);
    fprintf(flt_out4, "%d \n", 1);

    fclose(flt_out1);
    fclose(flt_out2);
    fclose(flt_out3);
    fclose(flt_out4);

    flt_out1 = fopen(flt_name_2b_ort_ref, "a");
    flt_out2 = fopen(flt_name_e_par_ref, "a");
    flt_out3 = fopen(flt_name_b_par_ref, "a");
    flt_out4 = fopen(flt_name_2e_ort_ref, "a");
    flt_out = fopen(flt_name_plc_sum, "a");

    fprintf(flt_out1, "%d \n", 1);
    fprintf(flt_out2, "%d \n", 1);
    fprintf(flt_out3, "%d \n", 1);
    fprintf(flt_out4, "%d \n", 1);
    fprintf(flt_out, "%d \n", 1);

    fclose(flt_out1);
    fclose(flt_out2);
    fclose(flt_out3);
    fclose(flt_out4);
    fclose(flt_out);

    const int Nx       = (int)GInd::getLatData().globLX; /// globL% is a size_t object, but in practice N% will never be
    const int Ny       = (int)GInd::getLatData().globLY; /// large enough for that to matter. Furthermore N% variables are
    const int Nz       = (int)GInd::getLatData().globLZ; /// used in statements involving subtraction.
    const int Nt       = (int)GInd::getLatData().globLT;

    const int vol3  = (int)GInd::getLatData().globvol3;

    const int dxmax    = Nx/4 + 1; /// N%/4 is set as the maximum distance in the off-axis direction because diagonal
    const int dymax    = Ny/4 + 1; /// correlations become noisy for large distances. N%/4 is somehow large enough.
    const int dzmax    = Nz/4 + 1;

    LatticeContainer<true,GCOMPLEX(floatT)> redBase7(commBase);
    LatticeContainer<true,GCOMPLEX(floatT)> redBase1(commBase);
    LatticeContainer<true,GCOMPLEX(floatT)> redBase2(commBase);
    LatticeContainer<true,GCOMPLEX(floatT)> redBase3(commBase);
    LatticeContainer<true,GCOMPLEX(floatT)> redBase4(commBase);
    LatticeContainer<true,GCOMPLEX(floatT)> redBase5(commBase);
    LatticeContainer<true,GCOMPLEX(floatT)> redBase6(commBase);

    const size_t elems = (int)GInd::getLatData().vol3;

    redBase.adjustSize(elems);
    redBase7.adjustSize(elems);
    redBase1.adjustSize(elems);
    redBase2.adjustSize(elems);
    redBase3.adjustSize(elems);
    redBase4.adjustSize(elems);
    redBase5.adjustSize(elems);
    redBase6.adjustSize(elems);

    /// Now we create memory accessors for some auxiliary arrays that will aid us with the correlation function
    /// calculations. The memory accessors are the things that refer to the arrays. Some arrays will be accessed on the
    /// GPU, while others will be accessed on the CPU. Therefore we need a MemType for each. 
    typedef gMemoryPtr<true>  MemTypeGPU;
    typedef gMemoryPtr<false> MemTypeCPU;

    /// These next three lines create a memory accessor for the untraced Polyakov loop array. The getMemAt template
    /// parameter must be true for GPU and false for CPU.
    MemTypeGPU mem20 = MemoryManagement::getMemAt<true>("PPCorrs");
   
    mem20->template adjustSize<GCOMPLEX(PREC)>(GInd::getLatData().globvol3);

    MemoryAccessor _ploopGPU(mem20->getPointer());

    //GPU arrays for plaquette
    MemTypeGPU mem01 = MemoryManagement::getMemAt<true>("PPCorrs");
    MemTypeGPU mem02 = MemoryManagement::getMemAt<true>("PPCorrs");
    MemTypeGPU mem03 = MemoryManagement::getMemAt<true>("PPCorrs");
    MemTypeGPU mem12 = MemoryManagement::getMemAt<true>("PPCorrs");
    MemTypeGPU mem13 = MemoryManagement::getMemAt<true>("PPCorrs");
    MemTypeGPU mem23 = MemoryManagement::getMemAt<true>("PPCorrs");

    mem01->template adjustSize<GCOMPLEX(PREC)>(GInd::getLatData().globvol3);
    mem02->template adjustSize<GCOMPLEX(PREC)>(GInd::getLatData().globvol3);
    mem03->template adjustSize<GCOMPLEX(PREC)>(GInd::getLatData().globvol3);
    mem12->template adjustSize<GCOMPLEX(PREC)>(GInd::getLatData().globvol3);
    mem13->template adjustSize<GCOMPLEX(PREC)>(GInd::getLatData().globvol3);
    mem23->template adjustSize<GCOMPLEX(PREC)>(GInd::getLatData().globvol3);

    MemoryAccessor _tmp_pl01GPU(mem01->getPointer());
    MemoryAccessor _tmp_pl02GPU(mem02->getPointer());
    MemoryAccessor _tmp_pl03GPU(mem03->getPointer());
    MemoryAccessor _tmp_pl12GPU(mem12->getPointer());
    MemoryAccessor _tmp_pl13GPU(mem13->getPointer());
    MemoryAccessor _tmp_pl23GPU(mem23->getPointer());

    //GPU arrays for Plaquette-Polyakov loop correlation
    MemTypeGPU memc01 = MemoryManagement::getMemAt<true>("PPCorrs");
    MemTypeGPU memc02 = MemoryManagement::getMemAt<true>("PPCorrs");
    MemTypeGPU memc03 = MemoryManagement::getMemAt<true>("PPCorrs");
    MemTypeGPU memc12 = MemoryManagement::getMemAt<true>("PPCorrs");
    MemTypeGPU memc13 = MemoryManagement::getMemAt<true>("PPCorrs");
    MemTypeGPU memc23 = MemoryManagement::getMemAt<true>("PPCorrs");
   
    memc01->template adjustSize<GCOMPLEX(PREC)>(GInd::getLatData().globvol3);
    memc02->template adjustSize<GCOMPLEX(PREC)>(GInd::getLatData().globvol3);
    memc03->template adjustSize<GCOMPLEX(PREC)>(GInd::getLatData().globvol3);
    memc12->template adjustSize<GCOMPLEX(PREC)>(GInd::getLatData().globvol3);
    memc13->template adjustSize<GCOMPLEX(PREC)>(GInd::getLatData().globvol3);
    memc23->template adjustSize<GCOMPLEX(PREC)>(GInd::getLatData().globvol3);

    MemoryAccessor _tmp_plc01GPU(memc01->getPointer());
    MemoryAccessor _tmp_plc02GPU(memc02->getPointer());
    MemoryAccessor _tmp_plc03GPU(memc03->getPointer());
    MemoryAccessor _tmp_plc12GPU(memc12->getPointer());
    MemoryAccessor _tmp_plc13GPU(memc13->getPointer());
    MemoryAccessor _tmp_plc23GPU(memc23->getPointer());

    //GPU arrays for average over symmetry points
    MemTypeGPU memas01 = MemoryManagement::getMemAt<true>("PPCorrs");
    MemTypeGPU memas02 = MemoryManagement::getMemAt<true>("PPCorrs");
    MemTypeGPU memas03 = MemoryManagement::getMemAt<true>("PPCorrs");
    MemTypeGPU memas12 = MemoryManagement::getMemAt<true>("PPCorrs");
    MemTypeGPU memas13 = MemoryManagement::getMemAt<true>("PPCorrs");
    MemTypeGPU memas23 = MemoryManagement::getMemAt<true>("PPCorrs");
   
    memas01->template adjustSize<GCOMPLEX(PREC)>(GInd::getLatData().globvol3);
    memas02->template adjustSize<GCOMPLEX(PREC)>(GInd::getLatData().globvol3);
    memas03->template adjustSize<GCOMPLEX(PREC)>(GInd::getLatData().globvol3);
    memas12->template adjustSize<GCOMPLEX(PREC)>(GInd::getLatData().globvol3);
    memas13->template adjustSize<GCOMPLEX(PREC)>(GInd::getLatData().globvol3);
    memas23->template adjustSize<GCOMPLEX(PREC)>(GInd::getLatData().globvol3);

    MemoryAccessor _tmp_plc_pl01GPU(memas01->getPointer());
    MemoryAccessor _tmp_plc_pl02GPU(memas02->getPointer());
    MemoryAccessor _tmp_plc_pl03GPU(memas03->getPointer());
    MemoryAccessor _tmp_plc_pl12GPU(memas12->getPointer());
    MemoryAccessor _tmp_plc_pl13GPU(memas13->getPointer());
    MemoryAccessor _tmp_plc_pl23GPU(memas23->getPointer());
    
    /// The ploop is an operator that is defined on spacelike points; therefore the kernel should run only over
    /// spacelike sites. If this is what you require, use iterateOverSpatialBulk instead of iterateOverBulk.
    redBase.template iterateOverSpatialBulk<All, HaloDepth>(CalcPloop<floatT,HaloDepth>(gauge, _ploopGPU));

    /// This construction ensures you obtain the spacelike volume of the entire lattice, rather than just a sublattice.
    floatT spacelikevol=GInd::getLatData().globvol3;
    
    GCOMPLEX(floatT) ploop;
    redBase.reduce(ploop, elems);
 
    /// Normalize. 
    ploop /= (spacelikevol);
 
    ReadIndexSpatial<HaloDepth> calcReadIndexSpatial;
    
    iterateFunctorNoReturn<true>(calcTmpPL<floatT,HaloDepth>(gauge, _tmp_pl01GPU, _tmp_pl02GPU, _tmp_pl03GPU, _tmp_pl12GPU, _tmp_pl13GPU, _tmp_pl23GPU),calcReadIndexSpatial,elems);

    for(register int dist = 4; dist <= 10; dist=dist+2){

    	redBase7.template iterateOverSpatialBulk<All, HaloDepth>(calcPPCORR<floatT,HaloDepth>(_ploopGPU, _tmp_pl01GPU, _tmp_pl02GPU, _tmp_pl03GPU, _tmp_pl12GPU, _tmp_pl13GPU, _tmp_pl23GPU, _tmp_plc01GPU, _tmp_plc02GPU, _tmp_plc03GPU, _tmp_plc12GPU, _tmp_plc13GPU, _tmp_plc23GPU, dist));
    
    	GCOMPLEX(floatT) plc_sum;
    	redBase7.reduce(plc_sum, elems);
    	//plc_sum /= ((dist + 2*dxmax - 2)*(dymax + 1)*(dzmax + 1)*(spacelikevol));

        iterateFunctorNoReturn<true>(averOS<floatT,HaloDepth>(_tmp_plc_pl01GPU, _tmp_plc_pl02GPU, _tmp_plc_pl03GPU, _tmp_plc_pl12GPU, _tmp_plc_pl13GPU, _tmp_plc_pl23GPU, _tmp_plc01GPU, _tmp_plc02GPU, _tmp_plc03GPU, _tmp_plc12GPU, _tmp_plc13GPU, _tmp_plc23GPU, dist),calcReadIndexSpatial,elems);
 
   	redBase1.template iterateOverSpatialBulk<All, HaloDepth>(CalcReferencePoint<floatT,HaloDepth>(_ploopGPU, _tmp_pl01GPU, dist));
   
    	GCOMPLEX(floatT) tmp_plc01_ref;
    	redBase1.reduce(tmp_plc01_ref, elems);
    
   	redBase2.template iterateOverSpatialBulk<All, HaloDepth>(CalcReferencePoint<floatT,HaloDepth>(_ploopGPU, _tmp_pl02GPU, dist));

    	GCOMPLEX(floatT) tmp_plc02_ref;
    	redBase2.reduce(tmp_plc02_ref, elems);

   	redBase3.template iterateOverSpatialBulk<All, HaloDepth>(CalcReferencePoint<floatT,HaloDepth>(_ploopGPU, _tmp_pl03GPU, dist));

    	GCOMPLEX(floatT) tmp_plc03_ref;
    	redBase3.reduce(tmp_plc03_ref, elems);

   	redBase4.template iterateOverSpatialBulk<All, HaloDepth>(CalcReferencePoint<floatT,HaloDepth>(_ploopGPU, _tmp_pl12GPU, dist));

    	GCOMPLEX(floatT) tmp_plc12_ref;
    	redBase4.reduce(tmp_plc12_ref, elems);

   	redBase5.template iterateOverSpatialBulk<All, HaloDepth>(CalcReferencePoint<floatT,HaloDepth>(_ploopGPU, _tmp_pl13GPU, dist));

    	GCOMPLEX(floatT) tmp_plc13_ref;
    	redBase5.reduce(tmp_plc13_ref, elems);

   	redBase6.template iterateOverSpatialBulk<All, HaloDepth>(CalcReferencePoint<floatT,HaloDepth>(_ploopGPU, _tmp_pl23GPU, dist));

    	GCOMPLEX(floatT) tmp_plc23_ref;
    	redBase6.reduce(tmp_plc23_ref, elems);

    	//printf("%f\n", tmp_plc01_ref.cREAL / elems + tmp_plc02_ref.cREAL / elems);
   
    	//CPU allocate memory for Plaquette-Polyakov loop correlation
    	MemTypeCPU memu01 = MemoryManagement:: getMemAt<false>("plcCorrs");
    	MemTypeCPU memu02 = MemoryManagement:: getMemAt<false>("plcCorrs");
    	MemTypeCPU memu03 = MemoryManagement:: getMemAt<false>("plcCorrs");
    	MemTypeCPU memu12 = MemoryManagement:: getMemAt<false>("plcCorrs");
    	MemTypeCPU memu13 = MemoryManagement:: getMemAt<false>("plcCorrs");
    	MemTypeCPU memu23 = MemoryManagement:: getMemAt<false>("plcCorrs");

    	memu01->template adjustSize<floatT>(GInd::getLatData().globvol3);
    	memu02->template adjustSize<floatT>(GInd::getLatData().globvol3);
    	memu03->template adjustSize<floatT>(GInd::getLatData().globvol3);
   	    memu12->template adjustSize<floatT>(GInd::getLatData().globvol3);
    	memu13->template adjustSize<floatT>(GInd::getLatData().globvol3);
    	memu23->template adjustSize<floatT>(GInd::getLatData().globvol3);
    /// This time, we have to copy data from the GPU array to the CPU array. The template parameter for copyFrom should
    /// match the target array from which we copy. In this case it's true, because we copy from an array on the GPU.
    /// The second argument is the size of the array in bytes.
  
        memu01->template copyFrom<true>(memas01,GInd::getLatData().globvol3*sizeof(GCOMPLEX(floatT)));
        memu02->template copyFrom<true>(memas02,GInd::getLatData().globvol3*sizeof(GCOMPLEX(floatT)));
        memu03->template copyFrom<true>(memas03,GInd::getLatData().globvol3*sizeof(GCOMPLEX(floatT)));
        memu12->template copyFrom<true>(memas12,GInd::getLatData().globvol3*sizeof(GCOMPLEX(floatT)));
        memu13->template copyFrom<true>(memas13,GInd::getLatData().globvol3*sizeof(GCOMPLEX(floatT)));
        memu23->template copyFrom<true>(memas23,GInd::getLatData().globvol3*sizeof(GCOMPLEX(floatT)));
  
    	MemoryAccessor _tmp_plc01CPU (memu01->getPointer());
    	MemoryAccessor _tmp_plc02CPU (memu02->getPointer());
    	MemoryAccessor _tmp_plc03CPU (memu03->getPointer());
    	MemoryAccessor _tmp_plc12CPU (memu12->getPointer());
    	MemoryAccessor _tmp_plc13CPU (memu13->getPointer());
    	MemoryAccessor _tmp_plc23CPU (memu23->getPointer());
 
    	GCOMPLEX(floatT) polm1, polm2, polm3, polm4, polm5, polm6;

    	int indx;

	//printing Plaquette-Polyakov loop correlation data
    	flt_out1 = fopen(flt_name_2b_ort, "a");
    	flt_out2 = fopen(flt_name_e_par, "a");
    	flt_out3 = fopen(flt_name_b_par, "a");
    	flt_out4 = fopen(flt_name_2e_ort, "a");

    	for(register int dx=0; dx<=dist+2*dxmax-3;dx++)
        	for(register int dz=0;dz<=dzmax;dz++)
            		for(register int dy=0;dy<=dz;dy++){
                
		indx = dx + dy * Nx + dz * Nx * (dymax + 1);

                _tmp_plc01CPU.getValue<GCOMPLEX(floatT)>(indx, polm1);
                _tmp_plc02CPU.getValue<GCOMPLEX(floatT)>(indx, polm2);
                _tmp_plc03CPU.getValue<GCOMPLEX(floatT)>(indx, polm3);
                _tmp_plc12CPU.getValue<GCOMPLEX(floatT)>(indx, polm4);
                _tmp_plc13CPU.getValue<GCOMPLEX(floatT)>(indx, polm5);
                _tmp_plc23CPU.getValue<GCOMPLEX(floatT)>(indx, polm6);

                fprintf(flt_out1, "%d %.10le\n", dist, polm1.cREAL + polm2.cREAL);
                fprintf(flt_out2, "%d %.10le\n", dist, polm3.cREAL);
                fprintf(flt_out3, "%d %.10le\n", dist, polm4.cREAL);
                fprintf(flt_out4, "%d %.10le\n", dist, polm5.cREAL + polm6.cREAL);

        }

        fclose(flt_out1);
        fclose(flt_out2);
        fclose(flt_out3);
        fclose(flt_out4);

	//printing Reference point method data         
        flt_out = fopen(flt_name_plc_sum, "a");
        flt_out1 = fopen(flt_name_2b_ort_ref, "a");
        flt_out2 = fopen(flt_name_e_par_ref, "a");
        flt_out3 = fopen(flt_name_b_par_ref, "a");
        flt_out4 = fopen(flt_name_2e_ort_ref, "a");

        fprintf(flt_out, "%d %.10le\n", dist, plc_sum.cREAL / ((dist + 2*dxmax - 2)*(dymax + 1)*(dzmax + 1)*(vol3)) );
        
        fprintf(flt_out1, "%d %.10le\n", dist, tmp_plc01_ref.cREAL / elems + tmp_plc02_ref.cREAL / elems);
        fprintf(flt_out2, "%d %.10le\n", dist, tmp_plc03_ref.cREAL / elems);
        fprintf(flt_out3, "%d %.10le\n", dist, tmp_plc12_ref.cREAL / elems );
        fprintf(flt_out4, "%d %.10le\n", dist, tmp_plc13_ref.cREAL / elems + tmp_plc23_ref.cREAL / elems);

        fclose(flt_out1);
        fclose(flt_out2);
        fclose(flt_out3);
        fclose(flt_out4);
        fclose(flt_out);
    }

    return ploop;
}

int main(int argc, char *argv[]) {

    /// Controls whether DEBUG statements are shown as it runs; could also set to INFO, which is less verbose.
    stdLogger.setVerbosity(INFO);

    /// Initialize a timer.
    StopWatch<true> timer;

    /// Initialize the CommunicationBase.
    CommunicationBase commBase(&argc, &argv);
    /// Initialize parameter class.
    LatticeParameters param;

    param.readfile(commBase, "../parameter/applications/flt.param", argc, argv);

    commBase.init(param.nodeDim());

    /// Set the HaloDepth.
    const size_t HaloDepth = 0;

    rootLogger.info("Initialize Lattice");

    /// Initialize the Lattice class.
    initIndexer(HaloDepth,param,commBase);

    /// Initialize the Gaugefield.
    rootLogger.info("Initialize Gaugefield");
    Gaugefield<PREC,true,HaloDepth> gauge(commBase);

    /// Initialize gaugefield with unit-matrices.
    gauge.one();

    /// Initialize LatticeContainer.
    LatticeContainer<true,GCOMPLEX(PREC)> redBase(commBase);

    /// We need to tell the Reductionbase how large our array will be. Again it runs on the spacelike volume only,
    /// so make sure you adjust this parameter accordingly, so that you don't waste memory.
    typedef GIndexer<All,HaloDepth> GInd;
    redBase.adjustSize(GInd::getLatData().vol3);

    /// Read a configuration from hard drive. For the given configuration you should find
    rootLogger.info("Read configuration");
    gauge.readconf_nersc(param.GaugefileName());

    /// Ploop variable
    GCOMPLEX(PREC) ploop;

    /// Start timer.
    timer.start();

    /// Exchange Halos
    gauge.updateAll();

    /// Calculate Plaquette-Ploop correlation and report polyakov loop.

    ploop = flt_meas<PREC,HaloDepth>(gauge, redBase, commBase, (double)(param.beta()));
 
    rootLogger.info(std::setprecision(20) ,  "Reduced RE(ploop) = " ,  ploop.cREAL);
    rootLogger.info(std::setprecision(20) ,  "Reduced IM(ploop) = " ,  ploop.cIMAG);
    
    /// stop timer and print time
    timer.stop();

    rootLogger.info("Time for operators: " ,  timer);
   
    return 0;
}

// Created by Battogtokh P.

