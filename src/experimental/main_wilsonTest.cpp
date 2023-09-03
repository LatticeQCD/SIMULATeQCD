#include "../simulateqcd.h"
#include "fullSpinor.h"
#include "fullSpinorfield.h"
#include "gammaMatrix.h"


template<class floatT, size_t HaloDepth>
struct TestKernel{

    //Gauge accessor to access the gauge field
    SU3Accessor<floatT> _SU3Accessor;
    SpinorColorAcc<floatT> _SpinorColorAccessor;

    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    TestKernel(Gaugefield<floatT,true,HaloDepth> &gauge, FullSpinorfield<floatT,true,HaloDepth> &spinorIn) 
                : _SU3Accessor(gauge.getAccessor()), 
                  _SpinorColorAccessor(spinorIn.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ Vect12<floatT> operator()(gSite site) {

        SU3<floatT> link;

        for (int nu = 1; nu < 4; nu++) {
            for (int mu = 0; mu < nu; mu++) {
                link += _SU3Accessor.template getLinkPath<All, HaloDepth>(site, mu, nu, Back(mu), Back(nu));
            }
        }

        ColorVect<floatT> spinCol = _SpinorColorAccessor.getColorVect(site);
        
        for (auto& s : spinCol){
            s = link*s;
        }

        return convertColorVectToVect12(spinCol);
    }
};


int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(DEBUG);

    LatticeParameters param;

    const int LatDim[] = {20, 20, 20, 20};
    const int NodeDim[] = {1, 1, 1, 1};

    param.latDim.set(LatDim);
    param.nodeDim.set(NodeDim);

    CommunicationBase commBase(&argc, &argv);
    commBase.init(param.nodeDim());

    const size_t HaloDepth = 0;


    rootLogger.info("Initialize Lattice");
    /// Initialize the Indexer on GPU and CPU.
    initIndexer(HaloDepth,param,commBase);
    
    using PREC = double;
    
    Gaugefield<PREC, true,HaloDepth> gauge(commBase);
    FullSpinorfield<PREC, true,HaloDepth> spinor_res(commBase);
    FullSpinorfield<PREC, true,HaloDepth> spinor_in(commBase);

    grnd_state<true> d_rand;
    initialize_rng(1337, d_rand);
    
    gauge.gauss(d_rand.state);
    spinor_res.gauss(d_rand.state);
    spinor_in.gauss(d_rand.state);



    StopWatch<true> timer;
    timer.start();
    spinor_res.template iterateOverBulk(TestKernel<PREC, HaloDepth>(gauge, spinor_in));
    timer.stop();
    timer.print("Test Kernel runtime");

    return 0;
}
