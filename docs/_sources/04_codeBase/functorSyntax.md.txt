# Functor syntax

In SIMULATeQCD we try to abstract away highly complex parallelization, which
depends on the API, whether GPUs or CPUs are used, the number of processes,
the node layout being used, and so on.
In order to accomplish this for the general case, we have implemented a system where one can
iterate an arbitrary operation that depends on arbitrary arguments over an arbitrary set.
For example a common task needed in lattice calculations is to iterate a plaquette
calculation, which depends on the `Gaugefield`, over all space-time points.
This operation is a _functor_ and the iterating method is an _iterator_,
which together comprise _functor syntax_.

Each functor is implemented as a `struct`. One passes the arguments of the functor
when initializing the `struct`. The argument over which the functor should
be iterated is implemented by overloading the syntax `operator()` of the `struct`.
The `RunFunctors` class contains all methods that iterate functors over
the desired target set. We choose to associate
`gSite` and `gSiteMu` objects to the coordinates of sites and links, respectively;
hence one usually passes a `gSite` or `gSiteMu` object
to the `operator()`. The class also contains several
`CalcGSite` methods that tell the iterator how to translate from
these objects to GPU thread indices.

There are many thorough examples on how to use functor syntax in
`src/testing/main_GeneralFunctorTest.cpp`. One example functor implementation
for the plaquette is shown below:
```C++
//! Functor to compute the plaquette given a gSite. It is called in a Kernel that is already defined by templates.
//! You do not need to write a custom Kernel. See below.
template<class floatT, size_t HaloDepth>
struct SimplePlaq {

    //! Functors can have member variables. Here we just need the accessor from the gaugefield reference.
    gaugeAccessor<floatT> gAcc;

    //! Constructor that initializes those members:
    explicit SimplePlaq(Gaugefield<floatT, true, HaloDepth> &gaugeIn) : gAcc(gaugeIn.getAccessor()) {}

    /* This is the operator() overload that is called to perform the local plaquette computation. This function has to
     * have the following design: It takes a gSite, and it returns the computed object corresponding to that site.
     * In this case, we want to return a float and store it in a LatticeContainer object. */
    __host__ __device__ floatT operator()(gSite site) {
        typedef GIndexer<All, HaloDepth> GInd;

        GSU3<floatT> temp;
        floatT result = 0;

        for (int nu = 1; nu < 4; nu++) {
            for (int mu = 0; mu < nu; mu++) {
                //! Notice here that site is changed implicitly by this; that is, it ends up at the last point of the
                //! path (in this case, it is the origin again)
                result += tr_d(gAcc.template getLinkPath<All, HaloDepth>(site, mu, nu, Back(mu), Back(nu)));
            }
        }
        return result;
    }
};


//! Wrapper to compute the plaquette using the functor "SimplePlaq"
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
    latContainer.reduce(plaq, elems);                         //! Sum up all contributions
    plaq /= (GInd::getLatData().globalLattice().mult() * 18); //! Normalize
    return plaq;
}
```
