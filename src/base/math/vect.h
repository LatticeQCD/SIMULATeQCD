//
// Created by Lukas Mazur
//

#pragma once
#include <array>
#include <type_traits>


template<typename inputType>
using isAllowedVecElemType = typename std::enable_if_t< std::is_same_v<inputType, GPUcomplex<__half>>
                                              || std::is_same_v<inputType, GPUcomplex<float>>
                                              || std::is_same_v<inputType, GPUcomplex<double>>
                                              || std::is_same_v<inputType, __half>
                                              || std::is_same_v<inputType, float>
                                              || std::is_same_v<inputType, double>
                                              , bool>;

template<typename... inputTypes>
using areAllowedVecElemTypes = typename std::enable_if_t< (std::is_same_v<GPUcomplex<__half> ,inputTypes> && ...)
                                                   || (std::is_same_v<GPUcomplex<float>      ,inputTypes> && ...)
                                                   || (std::is_same_v<GPUcomplex<double>     ,inputTypes> && ...)
                                                   || (std::is_same_v<__half                 ,inputTypes> && ...)
                                                   || (std::is_same_v<float                  ,inputTypes> && ...)
                                                   || (std::is_same_v<double                 ,inputTypes> && ...)
                                                   , bool>; 

template <class floatT> __device__ __host__ inline floatT minVal();

template<class floatT>
__device__ __host__ inline floatT get_rand(uint4* state);


template <class floatT, uint8_t elems>
struct Vect{

    std::array<GPUcomplex<floatT>, elems> data;

    __host__ __device__ constexpr Vect() {
        for(auto &elem: data)
        {
            elem = 0.0;
        }
    };

    template <typename... Args, areAllowedVecElemTypes<Args...> = true, typename std::enable_if_t<sizeof...(Args) == elems, int> = 0 >  
    __host__ __device__ Vect(Args... elements) 
            : data{elements...}
    {};

   
    template <typename Arg, isAllowedVecElemType<Arg> = true>
        __host__ __device__ Vect(Arg v0) 
        {
            for(auto &elem: data)
            {
                elem = v0;
            }
        };
    
    __device__ __host__ void gauss( uint4 * state )
    {
#ifndef USE_HIP_AMD
   	if constexpr (!std::is_same<floatT,__half>::value) {
#endif

        for(int i = 0; i<elems; ++i){
            floatT radius0,phi0;

            phi0 = 2.0*M_PI * get_rand<floatT>(state);

            radius0 = get_rand<floatT>(state);
            radius0 = radius0 + (1.0 - radius0) * minVal<floatT>(); // exclude 0 from random numbers!
            radius0 = sqrt(-1.0 * log(radius0));

            data[i] = COMPLEX(floatT)(radius0 * cos(phi0), radius0 * sin(phi0));
        }
#ifndef USE_HIP_AMD
	    }
    else {
        #ifdef __GPU_ARCH__
        for(int i = 0; i<elems; ++i){
            float radius0,phi0;
            phi0 = 2.0*M_PI * get_rand<float>(state);

            radius0 = get_rand<float>(state);
            radius0 = radius0 + (1.0 - radius0) * minVal<float>(); // exclude 0 from random numbers!
            radius0 = sqrt(-1.0 * log(radius0));

            data[i] = COMPLEX(__half)(__float2half(radius0 * cos(phi0)), __float2half(radius0 * sin(phi0)));
        }
        #endif
    }
#endif
    }


    __device__ __host__ Vect<floatT,elems> &operator=(const Vect<floatT,elems> &y);
    __device__ __host__ Vect<floatT,elems> &operator-=(const Vect<floatT,elems> &y);
    __device__ __host__ Vect<floatT,elems> &operator+=(const Vect<floatT,elems> &y);
    __device__ __host__ Vect<floatT,elems> &operator*=(const floatT &y);
    __device__ __host__ Vect<floatT,elems> &operator*=(const GPUcomplex<floatT> &y);

    // cast operations single <-> double precision
    template <class T>
    __device__ __host__ operator Vect<T,elems> () const {
        Vect<T,elems> res;
        for (auto i = 0; i < elems; ++i)
        {
            //res.data[i] = static_cast<GPUcomplex<T>>(data[i]);
            res.data[i] = GPUcomplex<T>(static_cast<T>(data[i].cREAL), static_cast<T>(data[i].cIMAG));
        }
        return res;
    }

    template<uint8_t elem, typename std::enable_if_t<elem < elems, int> = 0 >
        __device__ __host__ inline GPUcomplex<floatT> getElement() const  {
            return data[elem];
        }


    template<uint8_t elem, typename std::enable_if_t<elem < elems, int> = 0 >
        __device__ __host__ inline void addtoElement(const GPUcomplex<floatT> a){
            data[elem] += a;
        }

    template<uint8_t elem, typename std::enable_if_t<elem < elems, int> = 0 >
        __device__ __host__ inline void setElement(const GPUcomplex<floatT>& a){
            data[elem] = a;
        }

    template<uint8_t elem, typename std::enable_if_t<elem < elems, int> = 0 >
        __device__ __host__ inline void subfromElement(const GPUcomplex<floatT> a){
            data[elem] -= a;
        }

    __device__ __host__ inline GPUcomplex<floatT>& operator() (int i) {
        return data[i];
    }


    __host__ __device__ Vect<floatT,elems> getAccessor() const{
        return *this;
    }
};

    template <class floatT, uint8_t elems>
__device__ __host__ Vect<floatT,elems> &Vect<floatT,elems>::operator=(const Vect<floatT,elems> &y)
{   
    for(auto i = 0; i < elems; i++){
        data[i] = y.data[i]; 
    }
    return (*this);
}

    template <class floatT, uint8_t elems>
__device__ __host__ Vect<floatT,elems> &Vect<floatT,elems>::operator-=(const Vect<floatT,elems> &y)
{
    for(auto i = 0; i < elems; i++){
        data[i] -= y.data[i];
    }
    return (*this);
}

    template <class floatT, uint8_t elems>
__device__ __host__ Vect<floatT,elems> &Vect<floatT,elems>::operator+=(const Vect<floatT,elems> &y)
{
    for(auto i = 0; i < elems; i++){
        data[i] += y.data[i];
    }
    return (*this);
}

    template <class floatT, uint8_t elems>
__device__ __host__ Vect<floatT,elems> &Vect<floatT,elems>::operator*=(const floatT &y)
{
    for(auto i = 0; i < elems; i++){
        data[i] *= y;
    }
    return (*this);
}

    template <class floatT, uint8_t elems>
__device__ __host__ Vect<floatT,elems> &Vect<floatT,elems>::operator*=(const GPUcomplex<floatT> &y)
{
    for(auto i = 0; i < elems; i++){
        data[i] *= y;
    }
    return (*this);
}

    template <class floatT, uint8_t elems>
__device__ __host__ GPUcomplex<floatT> operator*(const Vect<floatT,elems> &x,const Vect<floatT,elems> &y)
{
    GPUcomplex<floatT> res = conj(x.data[0]) * y.data[0];
    for(auto i = 1; i < elems; i++){
        res += conj(x.data[i]) * y.data[i];
    }
    return res;
}

    template <class floatT, uint8_t elems>
__device__ __host__ Vect<floatT,elems> operator+(const Vect<floatT,elems> &x,const Vect<floatT,elems> &y)
{
    Vect<floatT,elems> z;
    for(auto i = 0; i < elems; i++){
        z.data[i] = x.data[i] + y.data[i];
    }
    return z;
}

    template <class floatT, uint8_t elems>
__device__ __host__ Vect<floatT,elems> operator-(const Vect<floatT,elems> &x,const Vect<floatT,elems> &y)
{
    Vect<floatT,elems> z;
    for(auto i = 0; i < elems; i++){
        z.data[i] = x.data[i] - y.data[i];
    }
    return z;
}

    template <class floatT, uint8_t elems>
__device__ __host__ Vect<floatT,elems> operator*(const GPUcomplex<floatT>& x,const Vect<floatT,elems>& y)
{
    Vect<floatT,elems> z;
    for(auto i = 0; i < elems; i++){
        z.data[i] = x * y.data[i];
    }
    return z;
}

    template <class floatT, uint8_t elems>
__device__ __host__ Vect<floatT,elems> operator*(const floatT & x,const Vect<floatT,elems>& y)
{
    Vect<floatT,elems> z;
    for(auto i = 0; i < elems; i++){
        z.data[i] = x * y.data[i];
    }
    return z;
}

    template <class floatT, uint8_t elems>
__device__ __host__ Vect<floatT,elems> operator*(const Vect<floatT,elems>& x,const GPUcomplex<floatT>& y)
{
    Vect<floatT,elems> z;
    for(auto i = 0; i < elems; i++){
        z.data[i] = x.data[i] * y;
    }
    return z;
}

    template <class floatT, uint8_t elems>
__device__ __host__ Vect<floatT,elems> operator*(const Vect<floatT,elems>& x,const floatT & y)
{
    Vect<floatT,elems> z;
    for(auto i = 0; i < elems; i++){
        z.data[i] = x.data[i] * y;
    }
    return z;
}


//! complex dot product x*y = sum_i(v_i conj(w_i))
    template <class floatT, uint8_t elems>
__device__ __host__ GPUcomplex<floatT> dot_prod(const Vect<floatT,elems> &x,const Vect<floatT,elems> &y)
{

    floatT real = x.data[0].cREAL*y.data[0].cREAL + x.data[0].cIMAG*y.data[0].cIMAG;
    floatT imag = x.data[0].cIMAG*y.data[0].cREAL - x.data[0].cREAL*y.data[0].cIMAG;

    for(auto i = 1; i < elems; i++){
        real += x.data[i].cREAL*y.data[i].cREAL + x.data[i].cIMAG*y.data[i].cIMAG;
        imag += x.data[i].cIMAG*y.data[i].cREAL - x.data[i].cREAL*y.data[i].cIMAG;
    }
    return GPUcomplex<floatT>(real,imag);
}

//! real part of dot product (no conjugation for y)
    template <class floatT, uint8_t elems>
__device__ __host__ floatT re_dot_prod(const Vect<floatT,elems> &x,const Vect<floatT,elems> &y)
{
    floatT res = x.data[0].cREAL*y.data[0].cREAL + x.data[0].cIMAG*y.data[0].cIMAG;
    for(auto i = 1; i < elems; i++){
        res += x.data[i].cREAL*y.data[i].cREAL + x.data[i].cIMAG*y.data[i].cIMAG;
    }
    return res;
}

// norm2 of vector
    template <class floatT, uint8_t elems>
__device__ __host__ floatT norm2(const Vect<floatT,elems> &x)
{
    floatT res = x.data[0].cREAL*x.data[0].cREAL + x.data[0].cIMAG*x.data[0].cIMAG;
    for(auto i = 1; i < elems; i++){
        res += x.data[i].cREAL*x.data[i].cREAL + x.data[i].cIMAG*x.data[i].cIMAG;
    }
    return res;
}

// complex conjugate
    template <class floatT, uint8_t elems>
__device__ __host__ Vect<floatT,elems> conj(const Vect<floatT,elems> &x)
{
    Vect<floatT,elems> z;
    for(auto i = 0; i < elems; i++){
        z.data[i] = conj(x.data[i]);
    }
    return z;
}


template <class floatT, uint8_t elems>
__device__ __host__ inline Vect<floatT,elems> vect_zero()
{
    return Vect<floatT,elems>(GPUcomplex<floatT>(0.0));
}

template <class floatT, uint8_t elems>
__device__ __host__ inline Vect<floatT,elems> vect_one()
{
    return Vect<floatT,elems>(GPUcomplex<floatT>(1.0));
}


template <class floatT, uint8_t elems>
__device__ __host__ inline Vect<floatT,elems> unit_basis_vect(const int& i)
{
    Vect<floatT,elems> vec = vect_zero<floatT,elems>();
    vec.data[i] = static_cast<floatT>(1.0);
    return vec;
}



#ifdef __GPUCC__

    template <class floatT, uint8_t elems>
__host__ std::ostream &operator << (std::ostream &s, const Vect<floatT,elems> &x)
{ 
    for(auto i = 0; i < elems; i++){
        s << x.data[i];
    }
    return s;
}

    template <class floatT, uint8_t elems>
__host__ std::istream &operator >> (std::istream &s, Vect<floatT,elems> &x)
{
    for(auto i = 0; i < elems; i++){
        s >> x.data[i];
    }
    return s;
}


#endif
