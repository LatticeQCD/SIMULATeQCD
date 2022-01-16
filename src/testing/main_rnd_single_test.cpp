/* 
 * main_rnd_single_test.cpp                                                               
 * 
 * Philipp Scior, 12 Nov 2018
 * 
 */

#include "../SIMULATeQCD.h"
#include <iostream>
#include <string>
#include <stdlib.h>
#include "../modules/rhmc/rhmcParameters.h"


std::ostream& operator<< (std::ostream &out, const uint4 &rand){
    out << rand.x << ", " << rand.y << ", " << rand.z << ", " << rand.w << std::endl;
    return out;
}


template<Layout LatLayout, size_t HaloDepth>
struct ReadIndex {
    inline __host__ __device__ gSite operator()(const dim3& blockDim, const uint3& blockIdx, const uint3& threadIdx) {
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        typedef GIndexer<LatLayout, HaloDepth> GInd;
        gSite site = GInd::getSite(i);
        return site;
    }
};


template <class floatT>
struct draw_rand
{
    uint4 * _state;
    floatT* _rand_array;

    draw_rand(uint4* state, floatT* rand_array) : _state(state), _rand_array(rand_array){}

    __host__ __device__ void initialize(__attribute__((unused)) gSite& site){}

     __device__ __host__ void operator()(gSite& site){

        floatT rand;
        rand = get_rand<floatT>(&_state[site.isite]);
        _rand_array[site.isite] = rand;   
    }
};


template <class floatT>
struct is_rand_zero
{
    uint4* _state;

    is_rand_zero(uint4* state) : _state(state) {}


    __host__ __device__ void initialize(__attribute__((unused)) gSite& site){}

    __device__ __host__ void operator()(gSite& site){

        floatT rand;
        rand = get_rand<floatT>(&_state[site.isite]);

        if (fabs(rand) <= minVal<floatT>()) {
            printf("WARN, WARN, WARN there is somewhere a random number = 0!\n");
        }
    }
};


template< class floatT, size_t HaloDepth>
bool test_old_new_host_dev(CommunicationBase &commBase){

    int seed = 1337;

    typedef GIndexer<All,HaloDepth> GInd;

    grnd_state<false> host_state;
    grnd_state<true> dev_state;

    initialize_rng(seed, host_state);
    initialize_rng(seed, dev_state);

    floatT * d_rand;
    floatT h_rand[GInd::getLatData().vol4];
    floatT h_rand2[GInd::getLatData().vol4];

    gpuMalloc(&d_rand, GInd::getLatData().vol4*sizeof(floatT));

    ReadIndex<All,HaloDepth> index;
    const size_t elems=GInd::getLatData().vol4;

    rootLogger.info("Draw random numbers twice:");

    iterateFunctorNoReturn<true>(draw_rand<floatT>(dev_state.state, d_rand), index, elems);
    iterateFunctorNoReturn<true>(draw_rand<floatT>(dev_state.state, d_rand), index, elems);

    iterateFunctorNoReturn<false>(draw_rand<floatT>(host_state.state, h_rand), index, elems);
    iterateFunctorNoReturn<false>(draw_rand<floatT>(host_state.state, h_rand), index, elems);

    gpuMemcpy(h_rand2, d_rand, GInd::getLatData().vol4*sizeof(floatT), gpuMemcpyDeviceToHost);

    bool host_dev = true;
    bool new_old = true;


    for (size_t i = 0; i < GInd::getLatData().vol4; ++i) {

        if ((h_rand[i]-h_rand2[i]) != 0.0) {
            rootLogger.error("At index " ,  i ,  ": host neq device rnd number!");
            host_dev = host_dev && false;
        
        }
    }

    if ((h_rand[0]-0.351445) > 1e-6){
        rootLogger.error("At index 0: old neq new rnd number!");
        new_old = new_old && false;
    }

    if ((h_rand[189]-0.82627) > 1e-6){
        rootLogger.error("At index 189: old neq new rnd number!");
        new_old = new_old && false;
    }

    if ((h_rand[1024]-0.670597) > 1e-6){
        rootLogger.error("At index 0: old neq new rnd number!");
        new_old = new_old && false;
    }

    if ((h_rand[2045]-0.511756) > 1e-6){
        rootLogger.error("At index 0: old neq new rnd number!");
        new_old = new_old && false;
    }

    std::string filename;

    if (typeid(floatT) == typeid(float))
        filename = "output_one_gpu_float.out";
    else
        filename = "output_one_gpu_double.out";
                                            
    host_state.write_to_file(filename, commBase, ENDIAN_LITTLE);

    if (host_dev) {
        rootLogger.info(CoutColors::green ,  "Host and device random numbers match",  CoutColors::reset);
    } else { 
        rootLogger.error("Host and device random numbers do not match");
        return true;
    }

    if (new_old) {
        rootLogger.info(CoutColors::green ,  "Old code and new code random numbers match",  CoutColors::reset);
    } else {
        rootLogger.error("Old code and new code random numbers do not match");
        return true;
    }

    return false;
}

template <class floatT, size_t HaloDepth>
void test_for_zeros(){

    int seed = 1337;

    typedef GIndexer<All,HaloDepth> GInd;
    grnd_state<true> dev_state;
    initialize_rng(seed, dev_state);

    ReadIndex<All,HaloDepth> index;


    const size_t elems=GInd::getLatData().vol4;

    rootLogger.info("Draw random numbers twice:");

    for (int i = 0; i < 25000; ++i) {
        printf("\033[0;31m"); 
        iterateFunctorNoReturn<true>(is_rand_zero<floatT>(dev_state.state), index, elems);
        printf("\033[0m");
        rootLogger.info("completed # " ,  i ,  "sweeps"); 
    }
}


int main(int argc, char *argv[]) {
    stdLogger.setVerbosity(DEBUG);

    RhmcParameters param;
    int LatDim[] = {8, 8, 8, 4};
    int NodeDim[] = {1, 1, 1, 1};
    const int HaloDepth = 0;
    bool lerror=false;
    
    param.latDim.set(LatDim);
    param.nodeDim.set(NodeDim);

    CommunicationBase commBase(&argc, &argv);
    commBase.init(param.nodeDim());

    initIndexer(HaloDepth,param, commBase);

    rootLogger.info("Testig RNG for single prec:");
    lerror = (lerror || test_old_new_host_dev<float, HaloDepth>(commBase));

    rootLogger.info("Testing RNG for double prec:");
    lerror = (lerror || test_old_new_host_dev<double, HaloDepth>(commBase));

    if(lerror) {
        rootLogger.error("At least one test failed!");
        return -1;
    } else {
        rootLogger.info("All tests " ,  CoutColors::green ,  "passed!" ,  CoutColors::reset);
    }

    return 0;
}