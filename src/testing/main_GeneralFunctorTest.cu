/* 
 * main_GeneralFunctorTest.cu                                                               
 * 
 * This file includes a lot of examples for how to our coding paradigm, which we call "general functor syntax".
 * Please start reading at run_func below.
 * 
 */

#include "../SIMULATeQCD.h"
#include "../modules/dslash/dslash.h"
#include "testing.h"


/*! A short implementation of the "Dslash". It is a fully working Dslash, including even to odd, odd to even and all to
 * all. If applied to a spinor, it multiplies the Dirac matrix from the left to the spinor. An even spinor is multiplied
 * by D_oe, and an odd spinor is multiplied by D_eo. This object must be used in the member function iterateOverBulk
 * of a spinor. The result will be written into that spinor. (See below)
 */
template<class floatT, Layout LatLayout, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
struct QuickDslash {

    //! The functor has to know about all the elements that it needs for computation.
    //! However, it does not need the Spinor, where the result should go (SpinorOut).
    gVect3arrayAcc<floatT> spinorIn;
    gaugeAccessor<floatT, R18> gAcc;
    gaugeAccessor<floatT, U3R14> gAccN;

    //! Use the constructor to initialize the members
    QuickDslash(
            Spinorfield<floatT, true, LatLayoutRHS, HaloDepthSpin> &spinorIn,
            Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
            Gaugefield<floatT, true, HaloDepthGauge, U3R14> &gaugeN) :
            spinorIn(spinorIn.getAccessor()),
            gAcc(gauge.getAccessor()),
            gAccN(gaugeN.getAccessor()) {}

    /*! This is the operator() overload that is called to perform the Dslash. This has to have the following design: It
     * takes a gSite, and it returns the object that we want to write. In this case, we want to return a gVect3<floatT>
     * to store it in another spinor.
     */
    __device__ __host__ inline auto operator()(gSiteStack site) const
    {
        //! We need an indexer to access elements. As the indexer knows about the lattice layout, we do not have to
        //! care about even/odd here explicitly. All that is done by the indexer.
        typedef GIndexer<LatLayout, HaloDepthSpin > GInd;

        /// Define temporary spinor that's 0 everywhere
        gVect3<floatT> Stmp(0.0);

        /// loop through all 4 directions and add result to current site
        for (int mu = 0; mu < 4; mu++) {

            //! transport spinor psi(x+mu) to psi(x) with link
            Stmp += static_cast<floatT>(C_1000)
                    * gAcc.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu)))
                    * spinorIn.getElement(GInd::site_up(site, mu));
            //! transport spinor psi(x-mu) to psi(x) with link dagger
            Stmp -= static_cast<floatT>(C_1000)
                    * gAcc.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu)))
                    * spinorIn.getElement(GInd::site_dn(site, mu));

#define C_3000 (-1./48.0)
            
            Stmp += static_cast<floatT>(C_3000)
                    * gAccN.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, mu), mu)))
                    * spinorIn.getElement(GInd::site_up_up_up(site, mu, mu, mu));
            Stmp -= static_cast<floatT>(C_3000)
                    * gAccN.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, mu, mu), mu)))
                    * spinorIn.getElement(GInd::site_dn_dn_dn(site, mu, mu, mu));
        }
        return Stmp;
    }

    //! If the Dslash shall be part of a math operation (see last line of run_func), it has to have a getAccessor() method.
    auto getAccessor() const
    {
        return *this;
    }
};

//! Functor to compute the plaquette given a gSite. It is called in a Kernel that is already defined by templates.
//! You do not need to write a custom Kernel. See below.
template<class floatT, size_t HaloDepth>
struct SimplePlaq {

    //! Functors can have member variables.
    gaugeAccessor<floatT> gAcc;     //! Here we just need the accessor from the gaugefield reference and nothing else

    //! and constructors that initialize those members:
    explicit SimplePlaq(Gaugefield<floatT, true, HaloDepth> &gaugeIn) :
            gAcc(gaugeIn.getAccessor()) {}

    /*! This is the operator() overload that is called to perform the local plaquette computation. This function has to
     * have the following design: It takes a gSite, and it returns the computed object corresponding to that site.
     * In this case, we want to return a float and store it in a LatticeContainer object. */
    __host__ __device__ floatT operator()(gSite site) {
        typedef GIndexer<All, HaloDepth> GInd;

        GSU3<floatT> temp;
        floatT result = 0;

        //! Compute the plaquette
        for (int nu = 1; nu < 4; nu++) {
            for (int mu = 0; mu < nu; mu++) {
                //! This is the manual way to do it:
                //!   temp = gAcc.getLink(GInd::getSiteMu(GInd::site_up(site, mu), nu))
                //!          * gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up(site, nu), mu))
                //!          * gAcc.getLinkDagger(GInd::getSiteMu(site, nu));
                //!   result += tr_d(gAcc.getLink(GInd::getSiteMu(site, mu)), temp);

                //! Instead we can use the equivalent but way more convenient function getLinkPath:
                result += tr_d(gAcc.template getLinkPath<All, HaloDepth>(site, mu, nu, Back(mu), Back(nu)));
                //! Notice here that site is changed implicitly by this (!), that is, it ends up at the last point of the
                //! path (in this case, it is the origin again)

                //! You can also a gSiteMu object in the first argument. In that case, the first step is done in
                //! direction mu of gSiteMu:
                //!   gSiteMu siteMu = GInd::indexGSiteMu(site, mu);
                //!   result += tr_d(gAcc.template getLinkPath<All, HaloDepth>(siteMu, nu, Back(mu), Back(nu)));

                //! Another possibility:
                //!   GSU3<floatT> tmp = gAcc.template getLinkPath<All, HaloDepth>(site, nu, mu, Back(nu));
                //!   result += tr_d(gAcc.template getLinkPath<All, HaloDepth>(site, Back(mu)), tmp);
                //! This is a bit faster, as tr(A*B) is less expensive, when only computing diagonal elements of A*B
            }
        }
        return result;
    }
};


//! Wrapper to compute the plaquette using the functor "SimplePlaq" (see above)
template<class floatT, size_t HaloDepth>
floatT compute_plaquette(Gaugefield<floatT, true, HaloDepth> &gauge, LatticeContainer<true,floatT> &latContainer) {

    typedef GIndexer<All, HaloDepth> GInd;
    const size_t elems = GInd::getLatData().vol4;

    latContainer.adjustSize(elems); //! Make sure container is large enough to hold one float for each site

    //! Perform the Plaquette computation. This is done by passing an instance of the functor "SimplePlaq" to the
    //! iterateOverBulk method of the LatticeContainer object. The functor "SimplePlaq" is called on each lattice site,
    //! which calculates the local contribution of each site to the plaquette. The "iterateOver..." member function of
    //! the LatticeContainer (and not gaugefield) is used here, since we want to store the results from the computation
    //! inside of this LatticeContainer and not inside of the gaugefield (the result at each site is simply a float and
    //! not an SU3 matrix). The Gaugefield is passed to the functor by reference.
    latContainer.template iterateOverBulk<All, HaloDepth>(SimplePlaq<floatT, HaloDepth>(gauge));

    floatT plaq;
    latContainer.reduce(plaq, elems); //! Sum up all contributions
    plaq /= (GInd::getLatData().globalLattice().mult() * 18); //! Normalize
    return plaq;
}


//! Simple functor to multiply links by two.
template<class floatT, size_t HaloDepth>
struct MultbyTwo {

    gaugeAccessor<floatT> gAcc;

    explicit MultbyTwo(Gaugefield<floatT, true, HaloDepth> &gaugeIn) :
            gAcc(gaugeIn.getAccessor()) {}

    //! operator() overload. It takes a gSiteMu object because we work on gauge fields.
    __host__ __device__ GSU3<floatT> operator()(gSiteMu site) {
        typedef GIndexer<All, HaloDepth> GInd;
        GSU3<floatT> temp;

        //! Multiply link by two and return it.
        temp = 2.0 * gAcc.getLink(site);

        //! returns to the gaugefield that we want to change
        return temp;
    }
};


//! This is an object that can be used in loop constructions
template<class floatT, size_t HaloDepth>
struct SillyMultiplicationInsideMuLoop{

    //! Member variable to read from a gaugefield
    gaugeAccessor<floatT> gAcc;

    //! A matrix that we will use later
    GSU3<floatT> my_mat;

    //! Constructor
    explicit SillyMultiplicationInsideMuLoop(Gaugefield<floatT, true, HaloDepth> &gaugeIn) :
            gAcc(gaugeIn.getAccessor()) {}

    //! This function is called at each lattice site, before the loop is performed. Here you can put some preparation
    //! that does not need to be inside the loop. In this specific example we do not use the gSite, so we mark it with
    //! __attribute__((unused)).
    __host__ __device__ void initialize(__attribute__((unused)) gSite site){
        //! Initialize our member matrix to one
        my_mat = gsu3_one<floatT>();
    }

    //! This is called inside the mu loop. We're just doing some silly multiplications here.
    __host__ __device__ GSU3<floatT> operator()(__attribute__((unused)) gSite site, size_t mu) {
        return static_cast<floatT>(mu) * my_mat;
    }
};


template<class floatT, size_t HaloDepth>
struct MultbyTwoNoReturn {

    gaugeAccessor<floatT> gAcc;

    explicit MultbyTwoNoReturn(Gaugefield<floatT, true, HaloDepth> &gaugeIn) :
            gAcc(gaugeIn.getAccessor()) {}

    //! operator() overload. It takes a gSiteMu object because we work on gauge fields.
    __host__ __device__ void operator()(gSiteMu site) {
        typedef GIndexer<All, HaloDepth> GInd;
        GSU3<floatT> temp;

        //! Multiply link by two and set it without using return
        temp = 2.0 * gAcc.getLink(site);
        gAcc.setLink(site, temp);
    }
};

//! Simple functor which can be used together with CalcReadIndex, CalcWriteIndex and iterateFunctor to copy a link
//! from one gSiteMu to another.
template<class floatT, size_t HaloDepth>
struct ReturnLink {

    //! Member variable to access the links stored in the gaugefield
    gaugeAccessor<floatT> gAcc;

    explicit ReturnLink(Gaugefield<floatT, true, HaloDepth> &gaugeIn) :
            gAcc(gaugeIn.getAccessor()) {}

    //! Functor() that simply returns the link at a given gSiteMu.
    __host__ __device__ GSU3<floatT> operator()(gSiteMu site) {
        return gAcc.getLink(site);
    }
};

//! Functor to compute a write index at fixed mu=0. This functor calculates the write index where the result will be
//! stored. It takes a gSite object as an argument, which, when called in the right order in the iterateFunctor method,
//! it gets from "ReadAtMu1". (Note that the gSiteMu class is a child of gSite).
template<Layout LatticeLayout, size_t HaloDepth>
struct CalcWriteIndex {
    inline __host__ __device__ gSiteMu operator()(const gSite &site) {
        return GIndexer<LatticeLayout, HaloDepth>::getSiteMu(site, 0);
    }
};

//! Functor to compute a gSiteMu object with mu=1. This takes the thread index of the gpu-thread as argument.
//! We use variadic templates here, as the thread index may either be encoded in a single int or in the dim3 variables.
//! (See BulkIndexer.h) This is an example how a functor looks, which computes one of the two indices that are used in
//! the iterateFunctor method.
template<Layout LatticeLayout, size_t HaloDepth>
struct ReadAtMu1 {
    template<typename... Args> inline __host__ __device__ gSiteMu operator()(Args... args) {
        return GIndexer<LatticeLayout, HaloDepth>::getSiteMuFull(args..., 1);
    }
};

//! A custom implementation of the dot product using functor syntax
template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth>
GCOMPLEX(floatT) myDotProd(Spinorfield<floatT, onDevice, LatLayout, HaloDepth> &lhs,
                           Spinorfield<floatT, onDevice, LatLayout, HaloDepth> &rhs,
                           LatticeContainer<onDevice,GCOMPLEX(floatT)> &latContainer) {

    typedef GIndexer<LatLayout, HaloDepth> GInd;
    size_t elems;
    if (LatLayout == All) {
        elems = GInd::getLatData().vol4;
    } else {
        elems = GInd::getLatData().vol4 / 2;
    }

    GCOMPLEX(floatT) res;

    //! Make sure we have enough space to store the result
    latContainer.adjustSize(elems);

    //! Compute the result. lhs*rhs returns a functor (operator* of the Spinorfield is overloaded that way)
    latContainer.template iterateOverBulk<LatLayout, HaloDepth>(lhs * rhs);

    //! Do the final reduction
    latContainer.reduce(res, elems);
    return res;
}

/// --------------------------------------------------------------------------- RUN FUNCTION. PLEASE START READING HERE.
template<class floatT, Layout LatLayout, Layout LatLayoutRHS>
void run_func(CommunicationBase &commBase) {

    /*! Here is a first example for how the functor syntax works. All important classes that store memory (Spinorfield,
     * Gaugefield and LatticeContainer) have special functions to initialize these objects. These functions have names
     * like iterateOverBulk and iterateOverFull. The former, for instance, will only affect objects at the bulk, while
     * the latter will modify all objects, including those at the halos.
     *
     * These functions take a functor as an argument. (A functor is a struct or a class that has the operator()
     * implemented). This functor has to have the following design: Its operator() needs to be a
     * __device__ __host__ function and it has to take a gSite object as an argument. (In the case of a gaugefield, it
     * may also take a gSiteMu object.) This gSite object holds all the information about the lattice site. It has to
     * return the result that we want to store in our class, e.g. a gVect3<floatT> if you work on a Spinor object.
     *
     * It is up to you how to compute this result. Use the gSite object to get the current index. The rest of the
     * functor may look as you want, but it must not contain host objects or functions. Above you find some examples of
     * these functors. When one of the iterateOver functions is called, a kernel is started, which runs over all
     * lattice sites, and calls the functor and stores the result. Here is an example of how such a Kernel could look.
     * (The actual implementation is more complex, as it is more general)
     *
     *  template <typename Functor, class floatT, Layout LatLayout, size_t HaloDepths>
     *  __global__ void iterateOverBulk( gVect3arrayAcc<floatT> spinorOut, Functor opc, const size_t size) {
     *
     *    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
     *    if (i >= size) {
     *      return;
     *    }
     *    gSite site = getSite<LatLayout, HaloDepth>(i);
     *
     *    spinorOut.setElement(site, functor(site));
     *  }
     */

    //! Initialization
    const int HaloDepth = 2;
    const int HaloDepthSpin = 4;
    typedef GIndexer<LatLayout, HaloDepth> GInd;

    LatticeContainer<true,floatT> latContainer(commBase);
    Gaugefield<floatT, true, HaloDepth> gauge(commBase);


    //! -------------------------------------------------------------------------------
    //! -------------------------- EXAMPLES WITH GAUGEFIELDS --------------------------
    //! -------------------------------------------------------------------------------


    //! This is the first example how the syntax can be used in the end. We want to initialize gauge from a constant
    //! object. In this case the constant object is simply an identity matrix.
    gauge.iterateWithConst(gsu3_one<floatT>());

    //! If everything went correctly, the plaquette should be equal to 1. plaq_ref will be our reference
    floatT plaq_ref = 1.0;
    floatT plaq;

    //! Compute the plaquette using the functor syntax. Note that we pass the gaugefield AND a LatticeContainer object
    //! to this wrapper function. The purpose of the LatticeContainer is to hold the local contributions of the
    //! plaquette computation for each site. Please look at compute_plaquette (it's just a wrapper function)
    //! and read through the comments there before continuing.
    plaq = compute_plaquette(gauge, latContainer);

    //! Compare the result to the reference
    compare_relative(plaq, plaq_ref, 1e-8, 1e-8, "Initialize to one");

    //! Now an example that directly modifies the gaugefield: We want to multiply all links with 2. To do so, we have
    //! created a functor (MultbyTwo, see above). This functor is executed for all links (including Halos) if we call
    //! iterateOverFullAllMu.
    gauge.iterateOverFullAllMu(MultbyTwo<floatT, HaloDepth>(gauge));

    //! Another possibility is to use the global function iterateOverFullAllMu(...) (not a member function of
    //! Gaugefield). With this function the passed functor should have return type void! (See definition of
    //! MultbyTwoNoReturn.) The gaugefield that should be iterated over is simply passed to it by reference.
    iterateOverFullAllMu<true, All, HaloDepth >(MultbyTwoNoReturn<floatT, HaloDepth>(gauge));

    //! The plaquette should now be (2*2)**4=256
    plaq_ref = 256.0;
    plaq = compute_plaquette(gauge, latContainer);
    compare_relative(plaq, plaq_ref, 1e-8, 1e-8, "Multiply by two");

    //! Instead of looping over all gaugelinks directly, we can also iterate over the sites and loop over mu at each
    //! lattice site. Here, on each site the functor SillyMultiplicationInsideMuLoop is called, which first calls its
    //! "initialize" member function and then calls the operator() overload inside of the mu loop.
    gauge.iterateOverFullLoopMu(SillyMultiplicationInsideMuLoop<floatT, HaloDepth>(gauge));

    //! Now, the result should be: (0 + 0 + 0 + 1*2*1*2 + 1*3*1*3 + 2*3*2*3)/6.
    plaq_ref = 8.166666666666666;
    plaq = compute_plaquette(gauge, latContainer);
    compare_relative(plaq, plaq_ref, 1e-8, 1e-8, "Initialize via loop");

    //! Set the gaugefield to one again.
    gauge.one();

    //! Same as above again. But this time only at the bulk
    gauge.iterateOverBulkLoopMu(SillyMultiplicationInsideMuLoop<floatT, HaloDepth>(gauge));

    //! After update we should get the same result
    gauge.updateAll();

    plaq_ref = 8.166666666666666;
    plaq = compute_plaquette(gauge, latContainer);
    compare_relative(plaq, plaq_ref, 1e-8, 1e-8, "Initialize via loop after update");

    //! Set the gaugefield to one again.
    gauge.one();

    //! Creating a custom object just for a multiplication is a bit much. There is an easier way:
    gauge = 2.0 * gauge;
    /*! So, how did that work? The "*" operator of the Gaugefield, that is, Gaugefield::operator*(type), is overloaded
     * using templates (here it uses the template for type=double). It returns a functor that can compute the result.
     * And the operator= is also overloaded! It is just a wrapper for iterateOverFullAllMu(). So actually what is
     * happening here is this:
     *   gauge.iterateOverFullAllMu(GeneralOperator<double, GSU3<floatT>, mult>(2.0, gauge))
     * The template parameter "mult" here controls that this is a multiplication.
     *
     * Note that if you try this syntax and you find a compiler error saying that some operator overload is not defined,
     * this is likely because the element-wise operator overload is not defined. For instance, it is impossible to use
     * gauge/2, because operator GSU3<floatT> / int is not defined.
     */

    plaq_ref = 16.0;
    plaq = compute_plaquette(gauge, latContainer);
    compare_relative(plaq, plaq_ref, 1e-8, 1e-8, "Multiply by two using operators");

    //! This works also for other stuff. These general functors that are returned are defined in
    //! src/math/operators.h and are templated.
    gauge = gauge * 0.5 + gsu3_one<floatT>();
    plaq_ref = 16.0;
    plaq = compute_plaquette(gauge, latContainer);
    compare_relative(plaq, plaq_ref, 1e-8, 1e-8, "Gaugefield + constant");

    gauge.one();

    //! But what if we do not want to work on the full lattice, but only at the bulk? In that case, we can not use
    //! the = operator any more, since it is hardcoded to perform operations on the bulk.
    gauge.iterateOverBulkAllMu(2.0 * gauge);

    //! The last tests only work with one GPU, as the results depend on the number of involved processes. Therefore, we
    //! print the result only when we are running on a single GPU
    if (commBase.single()){
        //! Now, the plaquette is incomplete, as halos have not changed. Some birdy told me its value.
        plaq_ref = 15.21;
        plaq = compute_plaquette(gauge, latContainer);
        compare_relative(plaq, plaq_ref, 1e-8, 1e-8, "Functors at Bulk");
    }

    //! After communication the plaquette should be 16 again.
    gauge.updateAll();

    plaq_ref = 16.0;
    plaq = compute_plaquette(gauge, latContainer);
    compare_relative(plaq, plaq_ref, 1e-8, 1e-8, "Plaquette after update");

    gauge.one();

    //! So far, we have only worked on *all* links, (mu = 0 to mu = 3). What if we want to work only for a certain mu?
    //! Let's multiply all links in mu=3 direction by 2.
    gauge.template iterateOverFullAtMu<3>(2.0 * gauge);

    //! If links in one particular mu direction are 2 and all others are 1, the plaquette is 2.5.
    plaq_ref = 2.5;
    plaq = compute_plaquette(gauge, latContainer);
    compare_relative(plaq, plaq_ref, 1e-8, 1e-8, "Special mu direction");

    //! If we do the same for the rest, the plaquette will be 16 again.
    gauge.template iterateOverFullAtMu<0>(2.0 * gauge);
    gauge.template iterateOverFullAtMu<1>(2.0 * gauge);

    //! Wait, what is the 512 here? That is an optional template parameter to define the blocksize of the GPU kernel call
    gauge.template iterateOverFullAtMu<2, 512>(2.0 * gauge);

    plaq_ref = 16.0;
    plaq = compute_plaquette(gauge, latContainer);
    compare_relative(plaq, plaq_ref, 1e-8, 1e-8, "All mu directions");

    //! The same is possible if we only work at the bulk.
    gauge.one();
    gauge.template iterateOverBulkAtMu<1>(GCOMPLEX(floatT)(2) * gauge);

    //! The last tests only work with one GPU, as the results depend on the number of involved processes. Therefore, we
    //! print the result only when we are running on a single GPU.
    if (commBase.single()){

        //! In this case, the plaquette is something skewed again.
        plaq_ref = 2.45;
        plaq = compute_plaquette(gauge, latContainer);
        compare_relative(plaq, plaq_ref, 1e-8, 1e-8, "Special mu at bulk");

        //! After update, plaquette should be 2.5 again.
    }
    gauge.updateAll();

    plaq_ref = 2.5;
    plaq = compute_plaquette(gauge, latContainer);
    compare_relative(plaq, plaq_ref, 1e-8, 1e-8, "Plaquette after update");


    /*! Now let us do something more complicated. Let us copy the links from mu=1 to mu=0 at each site. For that we need
     * three functors that we will pass to another iterate method, called "iterateFunctor", which makes use of those three.
     *   1. We need a functor that calculates the read index of the data at mu=1 based on the thread index from the GPU
     *      kernel call.
     *   2. We need a functor that calculates the write index (same site at mu=2) based on the read index.
     *   3. We need a functor that simply returns a copy of a link at a given site.
     * The iterateFunctor method first calls functor 1 to obtain the read indices, then functor 3 to return a copy of
     * the link, which is then written to the index that functor 2 provides.
     */

    //! This functor computes the site index with mu=1 based on the thread index by the GPU.
    ReadAtMu1<All, HaloDepth> calcReadIndex;

    //! Taking the index which the above functor returns, this functor creates a index at mu=0.
    CalcWriteIndex<All, HaloDepth> calcWriteIndex;

    //! This is a functor that simply returns a copy of a link at a given gSiteMu.
    ReturnLink<floatT, HaloDepth> returnLink(gauge);

    //! Now we put everything together and pass these functors to the iterateFunctor method. It takes the two functors
    //! that each compute one index, and the functor that does something with the object at the read index, and also the
    //! number of elements over which the kernel should run.
    gauge.iterateFunctor(returnLink, calcReadIndex, calcWriteIndex, GInd::getLatData().vol4Full);
    //! Behind the scenes this method does something like this for all elements:
    //!   auto site = calcReadIndex(blockDim, blockIdx, threadIdx);
    //!   gaugeAccessor.setElement(calcWriteIndex(site), returnLink(site));

    //! You may notice that using this function, it is possible to perform any custom math operation you want.
    //! There is no need to create a custom kernel.

    //! Plaquette should 5.5
    plaq_ref = 5.5;
    plaq = compute_plaquette(gauge, latContainer);
    compare_relative(plaq, plaq_ref, 1e-8, 1e-8, "Fully custom operator");


    //! -------------------------------------------------------------------------------
    //! -------------------------- EXAMPLES WITH SPINORFIELDS--------------------------
    //! -------------------------------------------------------------------------------


    //! Enough with Gaugefields. Let us continue with Spinors. To have a full physical example, we read in a test
    //! configuration.
    gauge.readconf_nersc("../test_conf/l20t20b06498a_nersc.302500");
    gauge.updateAll();

    //! Check if plaquette also agrees for a true configuration.
    plaq = compute_plaquette(gauge, latContainer);
    GaugeAction<floatT, true ,HaloDepth,R18> gAction(gauge);

    plaq_ref = gAction.plaquette();
    compare_relative(plaq, plaq_ref, 1e-8, 1e-8, "Plaquette after readin");

    //! The configuration for the naik term.
    Gaugefield<floatT, true, HaloDepth, U3R14> gauge_Naik(commBase);
    gauge_Naik.readconf_nersc("../test_conf/l20t20b06498a_nersc.302500");
    gauge_Naik.updateAll();

    rootLogger.info("Initialize Spinorfield");
    //! Two data spinors that we will use for further tests.
    Spinorfield<floatT, true, LatLayout, HaloDepthSpin> spinor1(commBase);
    Spinorfield<floatT, true, LatLayout, HaloDepthSpin> spinor2(commBase);

    //! Spinor to hold the result.
    Spinorfield<floatT, true, LatLayout, HaloDepthSpin> res_spinor(commBase);

    //! Spinor to enter the Dslash from the right. This needs another parity, therefore LatLayoutRHS.
    Spinorfield<floatT, true, LatLayoutRHS, HaloDepthSpin> rhs_spinor(commBase);

    //! Spinor to hold the result from the reference.
    Spinorfield<floatT, true, LatLayout, HaloDepthSpin> ref_spinor(commBase);

    //! We compare the results of the spinor operations by projecting the result onto this spinor. This gives a scalar,
    //! which is easy to compare. Similar to the plaquette above.
    Spinorfield<floatT, true, LatLayout, HaloDepthSpin> project_to_spinor(commBase);

    rootLogger.info("Fill with random numbers");

    //! Let us fill the spinors with some random numbers. This means we need a random number generator state
    grnd_state<true> d_rand;

    initialize_rng(1337, d_rand);

    spinor1.gauss(d_rand.state);
    spinor2.gauss(d_rand.state);
    rhs_spinor.gauss(d_rand.state);
    project_to_spinor.gauss(d_rand.state);

    rootLogger.info("Update Halos");
    spinor1.updateAll();
    spinor2.updateAll();
    project_to_spinor.updateAll();

    //! This will hold the result of the projection.
    GCOMPLEX(floatT) res_projected;

    //! For a reference which we use for testing.
    GCOMPLEX(floatT) ref_projected;

    LatticeContainer<true,GCOMPLEX(floatT)> redBase2(commBase);


    //! ---------- Start of actual functor syntax examples using spinorfields -----------


    //! First let us test the DotProduct that is defined using operator syntax. See above.
    res_projected = myDotProd(spinor1, spinor2, redBase2);

    //! Test against an existing version.
    ref_projected = spinor1.dotProduct(spinor2);
    //! Compare the result.
    compare_relative(res_projected, ref_projected, 1e-8, 1e2, "DotProduct test");

    //! Construct a spinor from a constant
    spinor2.iterateWithConst(gvect3_one<floatT>());

    res_projected = spinor2.dotProduct(spinor2);
    //! If the whole spinor consists out of 1 we can compute the result by hand:
    if (LatLayout == All) {
        ref_projected = GInd::getLatData().globvol4 * 3;
    } else {
        ref_projected = GInd::getLatData().globvol4 / 2 * 3;
    }
    //! Check if our computation is the same as by hand.
    compare_relative(res_projected, ref_projected, 1e-8, 1e2, "Construct from const");

    //! Perform an axpy operation using functor syntax. Here we measure the time to compare if the functor syntax is
    //! slower, which should not be the case.
    rootLogger.info("Iterate and perform functors");
    StopWatch<true> timer;
    timer.start();
    res_spinor = 1.234 * spinor1 + spinor2;
    timer.stop();
    timer.print("axpy by functors");
    timer.reset();
    res_projected = project_to_spinor.dotProduct(res_spinor);

    //! Use an existing implementation as reference.
    rootLogger.info("Perform reference calculation");
    ref_spinor = spinor2;
    timer.start();
    ref_spinor.axpyThis(1.234, spinor1);
    timer.stop();
    timer.print("axpy by explicit implementation");

    ref_projected = project_to_spinor.dotProduct(ref_spinor);
    //! Compare once more.
    compare_relative(res_projected, ref_projected, 1e-8, 1e2, "Axpy test");

    //! Finally, we perform a whole Dslash using a custom functor called "QuickDslash" (see above)
    timer.reset();
    timer.start();
    //! Template parameter is block size.
    res_spinor.template iterateOverBulk<256>(
            QuickDslash<floatT, LatLayout, LatLayoutRHS, HaloDepth, HaloDepthSpin>(rhs_spinor, gauge, gauge_Naik));
    timer.stop();
    timer.print("Dslash by custom functor");
    timer.reset();
    res_projected = project_to_spinor.dotProduct(res_spinor);

    //! Reference computation using an existing functor.
    HisqDSlash<floatT, true, LatLayoutRHS, HaloDepth, HaloDepthSpin> DD(gauge, gauge_Naik, 0.1);

    timer.start();
    DD.Dslash(ref_spinor, rhs_spinor, false);
    timer.stop();
    timer.print("Dslash by explicit implementation");
    timer.reset();

    ref_projected = project_to_spinor.dotProduct(ref_spinor);
    compare_relative(res_projected, ref_projected, 1e-8, 1e2, "Dslash test");

    //! This does not makes sense, but it is a final demonstration of what is possible.
    res_spinor.iterateOverBulk(spinor2 * 2.0 + spinor1 + (spinor1 -
                                                          QuickDslash<floatT, LatLayout, LatLayoutRHS, HaloDepth, HaloDepthSpin>(
                                                                  rhs_spinor, gauge, gauge_Naik)));

    //! If you encounter compiler errors that a certain operator is not defined, try general_mult, general_divide,
    //! general_add or general_subtract instead. As long as both participants of that operator have a getAccessor()
    //! method, this will work. For instance:
    res_spinor.iterateOverBulk(general_add(spinor1 * 2.0, QuickDslash<floatT, LatLayout, LatLayoutRHS, HaloDepth, HaloDepthSpin>(
                                                                  rhs_spinor, gauge, gauge_Naik)));
}

int main(int argc, char **argv) {


    stdLogger.setVerbosity(INFO);

    LatticeParameters param;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/tests/GeneralFunctorTest.param", argc, argv);
    commBase.init(param.nodeDim());

    const int HaloDepth = 2;

    /// Let's force Halos in all directions; otherwise the test doesn't work... (last parameter)
    initIndexer(HaloDepth,param, commBase, true);
    const int HaloDepthSpin = 4;
    initIndexer(HaloDepthSpin,param, commBase, true);

    rootLogger.info("------------------");
    rootLogger.info("Testing All - All");
    rootLogger.info("------------------");
    run_func<double, All, All>(commBase);
    rootLogger.info("------------------");
    rootLogger.info("Testing Even - Odd");
    rootLogger.info("------------------");
    run_func<double, Even, Odd>(commBase);
    rootLogger.info("------------------");
    rootLogger.info("Testing Odd - Even");
    rootLogger.info("------------------");
    run_func<double, Odd, Even>(commBase);
}

