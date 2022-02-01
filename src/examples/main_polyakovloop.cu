/* 
 * main_ploop.cu                                                               
 * 
 * David Clarke, 30 Oct 2018
 * 
 * Measure Polyakov loops using the multi-GPU framework. Initialization copied from main_plaquette.cu. This is a good
 * example of a simple operator calculated on spatial sites only.
 * 
 */

#include "../SIMULATeQCD.h"

#define PREC double 
#define MY_BLOCKSIZE 256

template<class floatT,size_t HaloDepth>
struct CalcPloop{

    /// Gauge accessor to access the gauge field.
    gaugeAccessor<floatT> gaugeAccessor;

    /// Constructor to initialize all necessary members.
    CalcPloop(Gaugefield<floatT,true,HaloDepth> &gauge) : gaugeAccessor(gauge.getAccessor()){
    }

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

        /// Start off at this site, pointing in N_tau direction.
        temp=gaugeAccessor.getLink(GInd::getSiteMu(site, 3));

        /// Loop over N_tau direction.
        for (size_t itp = 1; itp < Ntau; itp++) {
          size_t itau=it+itp;
          temp*=gaugeAccessor.getLink(GInd::getSiteMu(GInd::getSite(ix, iy, iz, itau), 3));
        }

        /// tr_c is the complex trace.
        result = tr_c(temp);
        return result;
    }
};

/// Function to compute the polyakov loop using the above struct CalcPloop.
template<class floatT, size_t HaloDepth>
GCOMPLEX(floatT) gPloop(Gaugefield<floatT,true,HaloDepth> &gauge, LatticeContainer<true,GCOMPLEX(floatT)> &redBase){

    typedef GIndexer<All,HaloDepth> GInd;
    /// Since we run the kernel on the spacelike volume only, elems need only be size d_vol3.
    const size_t elems = GInd::getLatData().vol3;
    redBase.adjustSize(elems);

    /// The ploop is an operator that is defined on spacelike points; therefore the kernel should run only over
    /// spacelike sites. If this is what you require, use iterateOverSpatialBulk instead of iterateOverBulk.
    redBase.template iterateOverSpatialBulk<All, HaloDepth>(CalcPloop<floatT,HaloDepth>(gauge));

    /// Do the final reduction.
    GCOMPLEX(floatT) ploop;
    redBase.reduce(ploop, elems);

    /// This construction ensures you obtain the spacelike volume of the entire lattice, rather than just a sublattice.
    floatT spacelikevol=GInd::getLatData().globvol3;

    /// Normalize. Factor 3 because tr 1= 1+1+1= 3.
    ploop /= (3.*spacelikevol);

    return ploop;
}



int main(int argc, char *argv[]) {

    try {
        /// Controls whether DEBUG statements are shown as it runs; could also set to INFO, which is less verbose.
        stdLogger.setVerbosity(INFO);

        /// Initialize parameter class.
        LatticeParameters param;

        /// Initialize the Lattice dimension.
        const int LatDim[] = {32, 32, 32, 8}; // {Ns,Ns,Ns,Ntau}

        /// Number of sublattices in each direction.
        const int NodeDim[] = {2, 1, 2, 1};

        /// Pass these dimensions to the parameter class.
        param.latDim.set(LatDim);
        param.nodeDim.set(NodeDim);

        /// Initialize a timer.
        StopWatch<true> timer;

        /// Initialize the CommunicationBase.
        CommunicationBase commBase(&argc, &argv);
        commBase.init(param.nodeDim());

        /// Set the HaloDepth.
        const size_t HaloDepth = 1;

        rootLogger.info("Initialize Lattice");

        /// Initialize the Lattice class.
        initIndexer(HaloDepth, param, commBase);

        /// Initialize the Gaugefield.
        rootLogger.info("Initialize Gaugefield");
        Gaugefield<PREC, true, HaloDepth> gauge(commBase);

        /// Initialize gaugefield with unit-matrices.
        gauge.one();

        /// Initialize LatticeContainer.
        LatticeContainer<true, GCOMPLEX(PREC) > redBase(commBase);

        /// We need to tell the Reductionbase how large our array will be. Again it runs on the spacelike volume only,
        /// so make sure you adjust this parameter accordingly, so that you don't waste memory.
        typedef GIndexer<All, HaloDepth> GInd;
        redBase.adjustSize(GInd::getLatData().vol3);

        /// Read a configuration from hard drive. For the given configuration you should find
        ///   Reduced RE(ploop) =  0.00358613
        ///   Reduced IM(ploop) = -0.000869849
        rootLogger.info("Read configuration");
        gauge.readconf_nersc("../test_conf/l328f21b6285m0009875m0790a_019.995");

        /// Start timer.
        timer.start();

        /// Ploop variable
        GCOMPLEX(PREC) ploop;

        /// Exchange Halos
        gauge.updateAll();

        /// Calculate and report Ploop.
        timer.start();
        ploop = gPloop<PREC, HaloDepth>(gauge, redBase);
        timer.stop();
        rootLogger.info("Time for operators: ", timer);
        rootLogger.info(std::setprecision(20), "Reduced RE(ploop) = ", ploop.cREAL);
        rootLogger.info(std::setprecision(20), "Reduced IM(ploop) = ", ploop.cIMAG);

        /// stop timer and print time
        timer.stop();
    }
    catch (const std::runtime_error &error) {
        return 1;
    }
    return 0;
}

