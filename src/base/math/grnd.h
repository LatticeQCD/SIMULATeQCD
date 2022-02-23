
// Created by Philipp Scior
// 9.11.2018

#ifndef GRAND_STATE
#define GRAND_STATE

#ifdef USE_CUDA
#include <curand_kernel.h>
#endif

#include "../../define.h"
#include "../gutils.h"
#include "../IO/misc.h"
#include "../IO/parameterManagement.h"
#include "../memoryManagement.h"
#include "../indexer/BulkIndexer.h"
#include "../communication/communicationBase.h"
#include <iostream>
#include "../IO/fileWriter.h"
#include <stdlib.h>
#include <float.h>
#include <functional>


/// functions for generating random numbers using the hybrid tausworthe generator from
/// https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch37.html


/// WARNING this RNG generates floats on [0,1] ! please be sure to remove 0 in a save(!) way
/// to circumvent NaNs in the Box-Muller transformation



template <class floatT> __host__ __device__ inline floatT minVal();
template<> __host__ __device__ inline float minVal<float>(){ return FLT_MIN; }
template<> __host__ __device__ inline double minVal<double>(){ return DBL_MIN; }
/**
* internal functions, should only be called from get_rand!
*/
__device__ __host__ inline unsigned taus_step( unsigned &z, int S1, int S2, int S3, unsigned M)
{
    unsigned b=((((z<<S1) &0xffffffffUL)^z)>>S2);
    return z=((((z &M)<<S3) &0xffffffffUL)^b);
}


__device__ __host__ inline unsigned LCG_step( unsigned &z, unsigned A, unsigned C)
{
    return z=(((A*z) & 0xffffffffUL) +C);
}
//////////////////////////////

/// A random variable in [0,1].
template<class floatT>
__device__ __host__ inline floatT get_rand(uint4* state)
{
    return 2.3283064365386963e-10*( taus_step( state->x, 13, 19, 12, 4294967294ul)^
                                    taus_step( state->y, 2, 25, 4,   4294967288ul)^
                                    taus_step( state->z, 3, 11, 17,  4294967280ul)^
                                    LCG_step(  state->w, 1664525,    1013904223ul) );
}

/// A random variable in (0,1].
template<class floatT>
__device__ __host__ inline floatT get_rand_excl0(uint4* state)
{
    floatT xR = get_rand<floatT>(state);
    return xR + (1.0-xR)*minVal<floatT>();
}


template<Layout LatLayout, size_t HaloDepth>
class GIndexer;
struct gSite;


class RNDHeader : virtual private ParameterList {

private:
    int header_size;

    Parameter<std::string> dattype;
    Parameter<int> dim[4];
    Parameter<std::string> checksum;
    Parameter<std::string> endian;

    bool read(std::istream &in, std::string &content);

    RNDHeader(){

        header_size = 0;

        addDefault(dattype, "DATATYPE", std::string("RNGSTATE"));
        add(dim[0], "DIMENSION_1");
        add(dim[1], "DIMENSION_2");
        add(dim[2], "DIMENSION_3");
        add(dim[3], "DIMENSION_4");
        add(checksum, "CHECKSUM");
        addDefault(endian, "ENDIANESS", std::string("BIG"));

    };
    ~RNDHeader(){};
    
    template<bool onDevice>
    friend class grnd_state;

public:
    size_t size() const {
        return header_size;
    }

    bool read(std::istream &in, CommunicationBase &comm);
    bool write(std::ostream &out, CommunicationBase &comm);

};


/// The class for the random number generator state
template<bool onDevice>
class grnd_state
{
private:
    //second template argument is just a dummy
    typedef GIndexer<All, 2> GInd;
    int elems;
    gMemoryPtr<onDevice> memory;
    RNDHeader header;
    bool switch_endian;
    uint32_t stored_checksum, computed_checksum;
    std::vector<uint4> buffer;
    std::hash<size_t> hash_f;

    void init();

    void byte_swap() {
        for (long i = 0; i < buffer.size(); i++)
            Byte_swap(buffer[i]);
    }

    bool read_header(std::istream &in, CommunicationBase &comm);
    bool write_header(Endianness en, std::ostream &out, CommunicationBase &comm);
    size_t header_size();

    void process_read_data() {
        if (switch_endian)
            byte_swap();
    }

    void process_write_data() {
        if (switch_endian)
            byte_swap();
    }

    bool checksums_match(CommunicationBase &comm);


public:
    uint4 * state;


    grnd_state() : header() {
        init();

    }

    ~grnd_state(){}

    void make_rng_state(unsigned int seed);
    __host__ __device__ uint4* getElement(gSite site);

    gMemoryPtr<onDevice>& getMemPtr(){
        return memory;
    }

    template<bool onDevice2>
    grnd_state<onDevice> &
    operator=(grnd_state<onDevice2> & src){
        memory->copyFrom(src.getMemPtr(),src.getMemPtr()->getSize());
        return *this;
    }

    void write_to_file(const std::string &fname, CommunicationBase &comm, Endianness en=ENDIAN_BIG); 
    void read_from_file(const std::string &fname, CommunicationBase &comm);

    template<class floatT, Layout LatticeLayout, size_t HaloDepth>
    void spam_to_file(const CommunicationBase &comm, const LatticeParameters &lp);
 };


template<bool onDevice> void initialize_rng(int seed, grnd_state<onDevice> &state){
    if (onDevice)
    {
        grnd_state<false> h_state;
        h_state.make_rng_state(seed);
        state = h_state;
    }else{
        state.make_rng_state(seed);
    }
}

#endif


