#include "../simulateqcd.h"


template<class floatT, Layout LatLayout, size_t HaloDepth, typename const_T ,size_t NStacks>
struct SpinorPlusConstTTimesSpinor{

    Vect3arrayAcc<floatT> spinor1;
    Vect3arrayAcc<floatT> spinor2;
    const_T val;
    SpinorPlusConstTTimesSpinor(Vect3arrayAcc<floatT> spinor1,
                                  Vect3arrayAcc<floatT> spinor2,
                                  const_T val) : spinor1(spinor1), spinor2(spinor2), val(val){}

    __device__ __host__ Vect3<floatT> operator()(gSiteStack& site){
        Vect3<floatT> Stmp;
        Stmp = spinor1.getElement(site);
        Stmp += val(site) * spinor2.getElement(site);

        return Stmp;
    }
};


struct triad_op{
    LatticeContainerAccessor b_acc;
    LatticeContainerAccessor c_acc;
    double scalar;

    triad_op(LatticeContainer<true,double> &b, LatticeContainer<true,double> &c, double s) : b_acc(b.getAccessor()), c_acc(c.getAccessor()), scalar(s) {}

    __host__ __device__ double operator()(gSite site) {
       double result;
        result = b_acc.template getElement<double>(site)+scalar*c_acc.template getElement<double>(site);
        return result;
    }
};
template<class spin>
struct triad_op_vector{
    Vect3arrayAcc<double> b_acc;
    Vect3arrayAcc<double> c_acc;
    SimpleArray<double,1> scalar;

    triad_op_vector<spin>(spin &b, spin &c, SimpleArray<double,1> s) : b_acc(b.getAccessor()), c_acc(c.getAccessor()), scalar(s) {}

    __host__ __device__ Vect3<double> operator()(gSiteStack site) {
        Vect3 result = b_acc.template getElement(site)+scalar(site)*c_acc.template getElement(site);
        return result;
    }
};

struct triad_op_fill{
    LatticeContainerAccessor a_acc;
    LatticeContainerAccessor b_acc;
    LatticeContainerAccessor c_acc;
    double scalar;

     triad_op_fill(LatticeContainer<true,double> &a, LatticeContainer<true,double> &b, LatticeContainer<true,double> &c, double s) : a_acc(a.getAccessor()), b_acc(b.getAccessor()), c_acc(c.getAccessor()), scalar(s) {}

    __host__ __device__ void operator()(gSite site) {
        double result;
        result = b_acc.template getElement<double>(site)+scalar*c_acc.template getElement<double>(site);
        a_acc.template setElement<double>(site, result);
    }
};

struct fill_with_double{
    double d;

    fill_with_double(double s) : d(s) {} 

    __host__ __device__ double operator()(__attribute__((unused)) gSite site) {
        return d;
    }
};

/*template<typename CalcReadInd>
__global__ void fillHaloSites(CalcReadInd calcReadInd, gSite* HalSites, const size_t size_x) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
     if (index >= size_x) {
        return;
    }
    auto site = calcReadInd(dim3(blockDim), GetUint3(dim3(blockIdx)), GetUint3(dim3(threadIdx)));
    HalSites[index] = site;
}

template<size_t BlockSize = DEFAULT_NBLOCKS, typename CalcReadInd>
void iterateFillHaloSites(CalcReadInd calcReadInd, gSite* HalSites, const size_t elems_x, const size_t elems_y = 1, 
    const size_t elems_z =1 , __attribute__((unused)) gpuStream_t stream = (gpuStream_t)nullptr) {
        dim3 blockDim;

    blockDim.x = BlockSize/(elems_y * elems_z);
    blockDim.y = elems_y;
    blockDim.z = elems_z;

    //Grid only in x direction!
    const dim3 gridDim = static_cast<int> (ceilf(static_cast<double> (elems_x)
                / static_cast<double> (blockDim.x)));

    hipLaunchKernelGGL((fillHaloSites), dim3(gridDim), dim3(blockDim), 0, stream , calcReadInd, HalSites, elems_x);

    };*/

int main(int argc, char *argv[]) {
    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);
    LatticeParameters param;

    param.readfile(commBase, "../parameter/profiling/mrhsDSlashProf.param", argc, argv);

    commBase.init(param.nodeDim());
    // commBase.forceHalos(true);

    const size_t HaloDepth = 2;
    initIndexer(HaloDepth, param,commBase);
    grnd_state<true> rand;
    initialize_rng(1337,rand);
    
    StopWatch<true> timer;
    Spinorfield<double, true, Even, HaloDepth, 1> a(commBase);
    SimpleArray<double,1> scal = 2.5;
    double sc = 2.5;
    Spinorfield<double, true, Even, HaloDepth, 1> b(commBase);
    Spinorfield<double, true, Even, HaloDepth, 1> c(commBase);
    

    const int LatDim[] = {param.latDim[0],param.latDim[1],param.latDim[2],param.latDim[3]};
    const int volume = LatDim[0]*LatDim[1]*LatDim[2]*LatDim[3];

    LatticeContainer<true, double> ac(commBase, "NOTSHARED_AC");
    LatticeContainer<true, double> bc(commBase, "NOTSHARED_BC");
    LatticeContainer<true, double> cc(commBase, "NOTSHARED_CC");
    ac.adjustSize(volume);
    bc.adjustSize(volume);
    cc.adjustSize(volume);
    a.gauss(rand.state);
    b.gauss(rand.state);
    c.gauss(rand.state);
    size_t triad_bytes = 3.0*3.0*2*sizeof(double)*volume/2.0;

    timer.setBytes(50*triad_bytes);

    timer.start();
    for (int i = 0; i < 50; i++) {
        a = b + scal * c;
    }
    timer.stop();
    auto dot = a.realdotProduct(a);
    rootLogger.info("50x Spinor triad timing: ", timer);
    rootLogger.info("Achieved Memory Bandwidth = ", timer.mbs(), " MB/s");
    rootLogger.info("Check result: ", dot/(0.5*volume));


    timer.reset();
    timer.start();
    for (int i = 0; i < 50; i++) {
        a.iterateOverCenter(SpinorPlusConstTTimesSpinor<double,Even,HaloDepth,SimpleArray<double,1>,1>(b.getAccessor(),c.getAccessor(),scal));
    }
    timer.stop();
    rootLogger.info("50x Spinor triad inner bulk timing: ", timer);
    timer.reset();
    if (param.nodeDim[0]*param.nodeDim[1]*param.nodeDim[2]*param.nodeDim[3] > 1) {
    timer.start();
    for (int i = 0; i < 50; i++) {
        a.iterateOverHalo(SpinorPlusConstTTimesSpinor<double,Even,HaloDepth,SimpleArray<double,1>,1>(b.getAccessor(),c.getAccessor(),scal));
    }
    timer.stop();
    rootLogger.info("50x Spinor triad inner halo timing: ", timer);
    rootLogger.info("Check result: ", dot/(0.5*volume));
    dot = a.realdotProduct(a);

    }

    triad_bytes = 3.0*sizeof(double)*volume;
    bc.iterateOverBulk<All, HaloDepth>(fill_with_double(1.0));
    cc.iterateOverBulk<All, HaloDepth>(fill_with_double(1.0));
    ac.iterateOverBulk<All, HaloDepth>(fill_with_double(0.0));

    timer.setBytes(50*triad_bytes);
    timer.start();
    for (int i = 0; i < 50; i++) {
        ac.iterateOverBulk<All,HaloDepth>(triad_op(bc,cc,sc));
    }
    
    timer.stop();
    double ac_red = 0;
    
    ac.reduce(ac_red,volume);

    // gSite* HaloSites;
    // gpuMalloc(&HaloSites,sizeof(gSite)*HaloIndexer<All,HaloDepth>::getInnerHaloSize());
    // CalcGSiteHalo<All, HaloDepth> calcHaloSites;
    // iterateFillHaloSites(calcHaloSites,HaloSites,HaloIndexer<All,HaloDepth>::getInnerHaloSize());

    rootLogger.info("50x LatticeContainer triad timing: ", timer);
    rootLogger.info("Achieved Memory Bandwidth = ", timer.mbs(), " MB/s");
    rootLogger.info("Check result: ", ac_red/volume);
    timer.reset();

    bc.iterateOverBulk<All, HaloDepth>(fill_with_double(1.0));
    cc.iterateOverBulk<All, HaloDepth>(fill_with_double(1.0));
    ac.iterateOverBulk<All, HaloDepth>(fill_with_double(0.0));

    timer.start();
    for (int i = 0; i < 50; i++) {
        // CalcGSiteInnerBulk<All,HaloDepth> calcsite;
        // iterateFunctorNoReturn<true,DEFAULT_NBLOCKS>(triad_op_fill(ac,bc,cc,sc),calcsite,HaloIndexer<All,HaloDepth>::getCenterSize());
        ac.iterateOverCenter<All,HaloDepth>(triad_op(bc,cc,sc));
    }
    timer.stop();
    ac.reduce(ac_red,volume);
    rootLogger.info("50x LatticeContainer triad inner Bulk timing: ", timer);
    rootLogger.info("Check result: ", ac_red/volume);
    timer.reset();
    if (param.nodeDim[0]*param.nodeDim[1]*param.nodeDim[2]*param.nodeDim[3] > 1) {
    timer.start();
    for (int i = 0; i < 50; i++) {
        // CalcGSiteHalo<All,HaloDepth> calcsite;
        // iterateFunctorNoReturn<true,DEFAULT_NBLOCKS>(triad_op_fill(ac,bc,cc,sc),calcsite,HaloIndexer<All,HaloDepth>::getInnerHaloSize());
        // ac.iterateOverHaloLookup<All,HaloDepth>(HaloSites,triad_op(bc,cc,sc));
        ac.iterateOverHalo<All,HaloDepth>(triad_op(bc,cc,sc));
    }
    timer.stop();
    ac.reduce(ac_red,volume);
    rootLogger.info("50x LatticeContainer triad inner Halo timing: ", timer);
    rootLogger.info("Check result: ", ac_red/volume);
    rootLogger.info("Inner Bulk Size: ", HaloIndexer<All, HaloDepth>::getCenterSize());
    rootLogger.info("Inner Halo Size: ", HaloIndexer<All,HaloDepth>::getInnerHaloSize());
    rootLogger.info("Lattice Size: ", GIndexer<All, HaloDepth>::getLatData().vol4);
    }
    return 0;
}
