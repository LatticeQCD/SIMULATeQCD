/* 
 * main_rnd_multiple_test.cu                                                               
 * 
 * Philipp Scior, 14 May 2019
 * 
 */

#include "../SIMULATeQCD.h"
#include <iostream>
#include <stdlib.h>
#include <string>
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
struct draw_rand {
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

template <class floatT, size_t HaloDepth>
void test_single_multiple(CommunicationBase &commBase){

int seed = 1337;

    typedef GIndexer<All,HaloDepth> GInd;

    grnd_state<false> host_state;
    grnd_state<true> dev_state;

    initialize_rng(seed, dev_state);

    std::string filename;
    if(typeid(floatT)==typeid(float))
        filename = "output_one_gpu_float.out";
    else
        filename = "output_one_gpu_double.out";

    host_state.read_from_file(filename, commBase);

    floatT * d_rand;
    floatT h_rand[GInd::getLatData().vol4];
    floatT h_rand2[GInd::getLatData().vol4];

    gpuMalloc(&d_rand, GInd::getLatData().vol4*sizeof(floatT));

    ReadIndex<All,HaloDepth> index;
    const size_t elems=GInd::getLatData().vol4;

    rootLogger.info() << "Draw random numbers and compare:";

    iterateFunctorNoReturn<true>(draw_rand<floatT>(dev_state.state, d_rand), index, elems);
    iterateFunctorNoReturn<true>(draw_rand<floatT>(dev_state.state, d_rand), index, elems);
    iterateFunctorNoReturn<true>(draw_rand<floatT>(dev_state.state, d_rand), index, elems);

    iterateFunctorNoReturn<false>(draw_rand<floatT>(host_state.state, h_rand), index, elems);

    gpuMemcpy(h_rand2, d_rand, GInd::getLatData().vol4*sizeof(floatT), gpuMemcpyDeviceToHost);

    bool host_dev = true;

    for (size_t i = 0; i < GInd::getLatData().vol4; ++i) {

        if ((h_rand[i]-h_rand2[i]) != 0.0) {
            rootLogger.error() << "At index " << i << ": host neq device rnd number!";
            host_dev = host_dev && false;
        }
    }

    if (host_dev)
        rootLogger.info() << CoutColors::green << "Random numbers on one and multiple GPUs match" << CoutColors::reset;
    else
        rootLogger.info() <<  CoutColors::red << "Random numbers on one and multiple GPU do not match" << CoutColors::reset;
}



int main(int argc, char *argv[]) {
    stdLogger.setVerbosity(INFO);

    RhmcParameters param;
    int LatDim[] = {8, 8, 8, 4};
    int NodeDim[] = {2, 1, 1, 1};
    const int HaloDepth = 0;
    
    param.latDim.set(LatDim);
    param.nodeDim.set(NodeDim);

    CommunicationBase commBase(&argc, &argv);
    commBase.init(param.nodeDim());

    initIndexer(HaloDepth,param, commBase);

    rootLogger.warn() << "Before running this test you have to run RndSingeTest!";

    rootLogger.info() << "Testing RNG for single prec:";
    test_single_multiple<float, HaloDepth>(commBase);

    rootLogger.info() << "Testing RNG for double prec:";
    test_single_multiple<float, HaloDepth>(commBase);

    return 0;
}