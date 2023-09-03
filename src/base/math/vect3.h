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

template <class floatT> __device__ __host__ Vect3<floatT> operator*(const SU3<floatT> &,const Vect3<floatT> &);


template <class floatT>
__device__ __host__ inline Vect3<floatT> unit_basis_vect3(const int& i) {
    return unit_basis_vect<floatT,3>(i);
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
__device__ inline Vect3<__half> unit_basis_vect3(const int& i)
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






