//
// Created by Lukas Mazur
//

#pragma once

#include "../../define.h"
#include "complex.h"
#include "random.h"
#include "vect.h"

// forward declaration
template <class floatT> class SU3;

template <class floatT> 
using Vect3 = Vect<floatT,3>;

template<class floatT>
__device__ __host__ inline floatT get_rand(uint4* state);

template <class floatT> __device__ __host__ Vect3<floatT> operator*(const SU3<floatT> &,const Vect3<floatT> &);


template <class floatT> __host__ __device__ inline floatT minVal();

template <class floatT>
__device__ __host__ void gauss(Vect3<floatT> &vec, uint4 * state )
    {
#ifndef USE_HIP_AMD
   	if constexpr (!std::is_same<floatT,__half>::value) {
#endif
        floatT radius0,radius1,radius2,phi0,phi1,phi2;

        phi0 = 2.0*M_PI * get_rand<floatT>(state);
        phi1 = 2.0*M_PI * get_rand<floatT>(state);
        phi2 = 2.0*M_PI * get_rand<floatT>(state);

        radius0 = get_rand<floatT>(state);
        radius0 = radius0 + (1.0 - radius0) * minVal<floatT>(); // exclude 0 from random numbers!
        radius0 = sqrt(-1.0 * log(radius0));
        radius1 = get_rand<floatT>(state);
        radius1 = radius1 + (1.0 - radius1) * minVal<floatT>(); // exclude 0 from random numbers!
        radius1 = sqrt(-1.0 * log(radius1));
        radius2 = get_rand<floatT>(state);
        radius2 = radius2 + (1.0 - radius2) * minVal<floatT>(); // exclude 0 from random numbers!
        radius2 = sqrt(-1.0 * log(radius2));

        vec.template setElement<0>( GPUcomplex<floatT>(radius0 * cos(phi0), radius0 * sin(phi0)));
        vec.template setElement<1>( GPUcomplex<floatT>(radius1 * cos(phi1), radius1 * sin(phi1)));
        vec.template setElement<2>( GPUcomplex<floatT>(radius2 * cos(phi2), radius2 * sin(phi2)));
#ifndef USE_HIP_AMD
	    }
        else {
            #ifdef __GPU_ARCH__
            float radius0,radius1,radius2,phi0,phi1,phi2;
            phi0 = 2.0*M_PI * get_rand<float>(state);
        phi1 = 2.0*M_PI * get_rand<float>(state);
        phi2 = 2.0*M_PI * get_rand<float>(state);

        radius0 = get_rand<float>(state);
        radius0 = radius0 + (1.0 - radius0) * minVal<float>(); // exclude 0 from random numbers!
        radius0 = sqrt(-1.0 * log(radius0));
        radius1 = get_rand<float>(state);
        radius1 = radius1 + (1.0 - radius1) * minVal<float>(); // exclude 0 from random numbers!
        radius1 = sqrt(-1.0 * log(radius1));
        radius2 = get_rand<float>(state);
        radius2 = radius2 + (1.0 - radius2) * minVal<float>(); // exclude 0 from random numbers!
        radius2 = sqrt(-1.0 * log(radius2));

        vec.template setElement<0>(GPUcomplex<__half>(__float2half(radius0 * cos(phi0)), __float2half(radius0 * sin(phi0))));
        vec.template setElement<1>(GPUcomplex<__half>(__float2half(radius1 * cos(phi1)), __float2half(radius1 * sin(phi1))));
        vec.template setElement<2>(GPUcomplex<__half>(__float2half(radius2 * cos(phi2)), __float2half(radius2 * sin(phi2))));
        #endif
        }
#endif
}


// vect3 = (1,0,0)  or (0,1,0)  or  (0,0,1)
template <class floatT>
__device__ __host__ inline Vect3<floatT> vect3_unity(const int& i)
{
    switch ( i )
    {
    case 1:
        return Vect3<floatT> (0.0, 1.0, 0.0);
    case 2:
        return Vect3<floatT> (0.0, 0.0, 1.0);
    }
// default value
    return Vect3<floatT> (1.0, 0.0, 0.0);
}

template <class floatT>
__device__ __host__ inline Vect3<floatT> vect3_zero() {
    return vect_zero<floatT,3>();
}

template <class floatT>
__device__ __host__ inline Vect3<floatT> vect3_one(){
    return vect_one<floatT,3>();
}

#ifndef USE_HIP_AMD
template <>
__device__ inline Vect3<__half> vect3_unity(const int& i)
{

    switch ( i )
    {
    case 1:
return Vect3<__half> (__float2half(0), __float2half(1), __float2half(0));
    case 2:
return Vect3<__half> (__float2half(0), __float2half(0), __float2half(1));
    }
// default value
return Vect3<__half> (__float2half(1), __float2half(0), __float2half(0));

}

template<>
__device__ inline Vect3<__half> vect3_zero()
{
    return Vect3<__half> (__float2half(0), __float2half(0), __float2half(0));
}
#endif

template <class floatT>
__device__ __host__ GPUcomplex<floatT> complex_product(const Vect3<floatT> &x,const Vect3<floatT> &y)
{
//  GPUcomplex<floatT> res = x.data[0] *(y.data[0]);
//  res += x.data[1] * (y.data[1]);
//  res += x.data[2] * (y.data[2]);
//  return res;

  return  fma(x.data[0] ,y.data[0],
            fma(x.data[1] , (y.data[1]), x.data[2] * (y.data[2])));


}

template <class floatT>
__device__ __host__ GPUcomplex<floatT> complex_product_add(const Vect3<floatT> &x,const Vect3<floatT> &y, const GPUcomplex<floatT> &d)
{
    //GPUcomplex<floatT> res = x.data[0] *(y.data[0]);
    //res += x.data[1] * (y.data[1]);
    //res += x.data[2] * (y.data[2]);
    //return res;

    return  fma(x.data[0] ,y.data[0],
                fma(x.data[1] , (y.data[1]),fma(x.data[2] ,(y.data[2]),d)));
}






