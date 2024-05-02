/*
 * main_plaquette.cpp
 *
 * Lukas Mazur, 10 Apr 2018
 *
 * This is just an example how a very very basic program works. Look at src/testing/main_GeneralOperatorTest.cpp to see
 * how to write more advanced GPU code.
 *
 */

#include "../simulateqcd.h"

#define PREC double

/* A quick implementation of the plaquette. This object must be called in member function
   iterateOverBulk of LatticeContainer. This will initiate a kernel, that runs over all lattice sites
   and performs operator() at this sites
   */
template<class floatT,size_t HaloDepth>
struct CalcPlaq{

    //Gauge accessor to access the gauge field
    SU3Accessor<floatT> SU3Accessor;

    //Constructor to initialize all necessary members.
    CalcPlaq(Gaugefield<floatT,true,HaloDepth> &gauge) : SU3Accessor(gauge.getAccessor()){
    }

    //This is the operator that is called inside the Kernel
    __device__ __host__ floatT operator()(gSite site) {

        /// We need to choose the type of indexer. The first template is the layout of the lattice.
        typedef GIndexer<All, HaloDepth> GInd;

        /// Define a SU(3) matrix
        SU3<floatT> temp;

        floatT result = 0;
        /// loop through all directions which are needed by the four link variables in the plaquette
        for (int nu = 1; nu < 4; nu++) {
            for (int mu = 0; mu < nu; mu++) {
                /// The SU3Accessor class provides access to the link variables (SU(3)-matrices).
                /// The SU3Accessor.getLink(index) method takes the index of the Link (not site! which means that
                /// that index involves also a direction mu) and returns the SU(3) matrix at this position.
                /// With a given gSite object and a direction mu, the link index can be computed with
                /// GInd::index(site, nu).
                /// If an Link one step further in mu direction is needed, its index in nu direction can be computed with
                /// GInd::index(GInd::site_up(site, nu), mu))

                // However, a simple path of links, like in the plaquette, may be defined
                // Using the getLinkPath statement of the SU3Accessor.
                // Here you can pass an arbitrary number of directions mu or nu. In the case of the plaquette only for
                // Site is changed. It ends up at the last point of the path. In this case, this is the origin again
                result += tr_d(SU3Accessor.template getLinkPath<All, HaloDepth>(site, mu, nu, Back(mu), Back(nu)));

                // You can also use gSiteMu objects. In that case, the first step is done in direction mu of gSiteMu
                //gSiteMu siteMu = GInd::indexGSiteMu(site, mu);
                //result += tr_d(SU3Accessor.template getLinkPath<All, HaloDepth>(siteMu, nu, Back(mu), Back(nu)));

                // This is a bit faster, as tr(A*B) is less expensive, when only computing diagonal elements of A*B
                //SU3<floatT> tmp = SU3Accessor.template getLinkPath<All, HaloDepth>(site, nu, mu, Back(nu));
                //result += tr_d(SU3Accessor.template getLinkPath<All, HaloDepth>(site, Back(mu)), tmp);

                // equivalent but way less intuitive
                //SU3<floatT> temp;
                //temp = SU3Accessor.getLink(GInd::getSiteMu(GInd::site_up(site, mu), nu))
                //       * dagger(SU3Accessor.getLink(GInd::getSiteMu(GInd::site_up(site, nu), mu)))
                //       * dagger(SU3Accessor.getLink(GInd::getSiteMu(site, nu)));
                //result += tr_d(SU3Accessor.getLink(GInd::getSiteMu(site, mu)), temp);
            }
        }

        //Return the result
        //The return value will be stored in the array of the reductionbase at index site.isite.
        return result;
    }
};


//Function to compute the plaquette using the above struct CalcPlaq.
template<class floatT, size_t HaloDepth>
floatT gPlaq(Gaugefield<floatT,true, HaloDepth> &gauge, LatticeContainer<true,floatT> &redBase){

    typedef GIndexer<All,HaloDepth> GInd;
    const size_t elems = GInd::getLatData().vol4;
    //Make sure, redBase is large enough
    redBase.adjustSize(elems);

    //Perform the Plaquette computation. Simply pass a valid instance of CalcPlaq to
    //iterateOverBulk. This will call a kernel, which runs over all lattice sites.
    //At each lattice site, CalcPlaq() is called. The result is stored at that site
    redBase.template iterateOverBulk<All, HaloDepth>(CalcPlaq<floatT, HaloDepth>(gauge));

    //Do the final reduction
    floatT plaq;
    redBase.reduce(plaq, elems);

    //Normalize the result
    plaq /= (GInd::getLatData().globalLattice().mult()*18);
    return plaq;
}


int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(DEBUG);

    /// Initialize parameter class. This class can also read parameter from textfiles!
    LatticeParameters param;

    /// Initialize the Lattice dimension
    const int LatDim[] = {20, 20, 20, 20};

    /// How the lattice should be distributed on different GPU's.
    /// In this case we don't split the lattice into sub-lattices.
    /// If we would set it to {1,2,1,1}, then the lattice will be split in y-dimension. This would mean that
    /// we have two sub-lattices, which are stored on two different GPU's
    const int NodeDim[] = {1, 1, 1, 1};

    /// Just pass these dimensions to the parameter class
    param.latDim.set(LatDim);
    param.nodeDim.set(NodeDim);

    /// Initialize a timer
    StopWatch<true> timer;

    /// Initialize the CommunicationBase. This class handles the communitation between different Cores/GPU's.
    CommunicationBase commBase(&argc, &argv, true);
    commBase.init(param.nodeDim());

    /// Set the HaloDepth. It should be a constant values, since this value should be passed as an non-type template
    /// parameter to each kernel.
    const size_t HaloDepth = 1;


    /// rootLogger.info() is a method which prints messages. It can be used as std::cout, but it involves always a newline.
    /// This rootLogger class makes sure that only the root Core/GPU prints something.
    /// Alternatively, one can use stdLogger.info() where each Core/GPU will print something.
    /// Apart from the method info() there is also alloc() trace() debug() info() warn() error() fatal() which
    /// highlight the output differently.
    rootLogger.info("Initialize Lattice");
    /// Initialize the Indexer on GPU and CPU.
    initIndexer(HaloDepth,param,commBase);
    typedef GIndexer<All,HaloDepth> GInd;


    /// Initialize the Gaugefield. Basically, this object holds all SU(3)-matrices of the gaugefield.
    /// The second template parameter determines whether the gaugefield should be stored on GPU or CPU but this will
    /// be changed in the future ...
    rootLogger.info("Initialize Gaugefield");
    Gaugefield<PREC, true,HaloDepth> gauge(commBase);

    /// Initialize gaugefield with unity-matrices.
    gauge.one();

    /// Initialize LatticeContainer. This is in principle the "array", where the values of the plaquette are
    /// stored which are summed up in the end
    LatticeContainer<true,PREC> redBase(commBase);
    /// We need to tell the Reductionbase how large our Array will be
    redBase.adjustSize(GInd::getLatData().vol4);
    grnd_state<false> h_rand;
    grnd_state<true> d_rand;

    h_rand.make_rng_state(1337);
    d_rand = h_rand;


    /// Read a configuration from hard drive.
    rootLogger.info("Read configuration");
    //    gauge.readconf_nersc("../test_conf/l20t20b06498a_nersc.302500");
    gauge.random(d_rand.state);
    /// start timer...
    timer.start();
    /// define variable where the plaquette should be stored.
    PREC plaq = 0;

    /// Exchange Halos before calculating the plaquette!
    gauge.updateAll();

    timer.start();
    /// compute plaquette
    plaq = gPlaq<PREC,HaloDepth>(gauge, redBase);
    /// stop timer and print time
    timer.stop();
    rootLogger.info("Time for operators " ,  timer);
    rootLogger.info("Reduced Plaquette is: " ,  plaq);

    return 0;
}
