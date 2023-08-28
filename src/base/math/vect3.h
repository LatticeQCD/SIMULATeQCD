/*-----------------------------------------------------------------------------------------------
 * /
 * / cvect3.h
 * /
 * / $Id: cvect3.h,v 1.1 2002/09/10 11:56:54 okacz Exp $
 * /
 * /---------------------------------------------------------------------------------------------*/

#ifndef _vect3_h_
#define _vect3_h_

#include "../../define.h"
#include "complex.h"
#include "random.h"


// forward declaration
template <class floatT> class SU3;
template <class floatT> class Vect3;
template <class floatT> class cVect3;
template <class floatT, bool onDevice> class Vect3array;
template <class floatT> __host__ std::ostream & operator<<(std::ostream &, const Vect3<floatT> &);
template <class floatT> __host__ std::istream & operator>>(std::istream &, Vect3<floatT> &);
template <class floatT>  __device__ __host__ COMPLEX(floatT) operator*(const Vect3<floatT> &,const Vect3<floatT> &);
template <class floatT>  __device__ __host__ COMPLEX(floatT) complex_product(const Vect3<floatT> &,const Vect3<floatT> &);
template <class floatT>  __device__ __host__ COMPLEX(floatT) complex_product_add(const Vect3<floatT> &,const Vect3<floatT> &, const COMPLEX(floatT) &);


template <class floatT>  __device__ __host__ Vect3<floatT> operator+(const Vect3<floatT> &,const Vect3<floatT> &);
template <class floatT>  __device__ __host__ Vect3<floatT> operator-(const Vect3<floatT> &,const Vect3<floatT> &);
template <class floatT>  __device__ __host__ Vect3<floatT> operator*(const floatT &,const Vect3<floatT> &);
template <class floatT>  __device__ __host__ Vect3<floatT> operator*(const COMPLEX(floatT) &,const Vect3<floatT> &);
template <class floatT>  __device__ __host__ Vect3<floatT> operator*(const Vect3<floatT> &,const floatT &);
template <class floatT>  __device__ __host__ Vect3<floatT> operator*(const Vect3<floatT> &,const COMPLEX(floatT) &);
template <class floatT> __device__ __host__ Vect3<floatT> conj(const Vect3<floatT> &);
template <class floatT> __device__ __host__ floatT norm2(const Vect3<floatT> &);
template <class floatT> __device__ __host__ COMPLEX(floatT) dot_prod(const Vect3<floatT> &,const Vect3<floatT> &);
template <class floatT> __device__ __host__ floatT re_dot_prod(const Vect3<floatT> &,const Vect3<floatT> &);
template <class floatT> __device__ __host__ Vect3<floatT> operator*(const SU3<floatT> &,const Vect3<floatT> &);
template <class floatT> __device__ __host__ SU3<floatT> tensor_prod(const Vect3<floatT> &,const Vect3<floatT> &);
template <class floatT> __device__ __host__ inline floatT minVal();

template<class floatT>
__device__ __host__ inline floatT get_rand(uint4* state);

template <class floatT>
class Vect3
{
private:

    COMPLEX(floatT) _v0, _v1, _v2;

public:

    __host__ __device__ Vect3() {};
    __host__ __device__ Vect3(COMPLEX(floatT) v0) : _v0(v0), _v1(v0), _v2(v0) {};
    __host__ __device__ Vect3(floatT v0) : _v0(v0), _v1(v0), _v2(v0) {};
    __host__ __device__ Vect3(COMPLEX(floatT) v0, COMPLEX(floatT) v1, COMPLEX(floatT) v2) : _v0(v0), _v1(v1), _v2(v2) {};

#if (!defined __GPUCC__)
    __host__ friend std::ostream &operator << <> (std::ostream &, const Vect3<floatT> &);
#endif
    __host__ friend std::istream &operator >> <> (std::istream &, Vect3<floatT> &);

    friend class Vect3array<floatT,true>;
    friend class Vect3array<floatT,false>;


    // vector operations
    __device__ __host__ Vect3<floatT> &operator =(const Vect3<floatT> &);
    __device__ __host__ Vect3<floatT> &operator-=(const Vect3<floatT> &);
    __device__ __host__ Vect3<floatT> &operator+=(const Vect3<floatT> &);
    __device__ __host__ Vect3<floatT> &operator*=(const floatT &);
    __device__ __host__ Vect3<floatT> &operator*=(const COMPLEX(floatT) &);
    __device__ __host__ friend COMPLEX(floatT) operator* <> (const Vect3<floatT> &,const Vect3<floatT> &);
    __device__ __host__ friend COMPLEX(floatT) complex_product <> (const Vect3<floatT> &,const Vect3<floatT> &);
    __device__ __host__ friend COMPLEX(floatT) complex_product_add <> (const Vect3<floatT> &,const Vect3<floatT> &, const COMPLEX(floatT) & );
    __device__ __host__ friend Vect3<floatT> operator+  <> (const Vect3<floatT> &,const Vect3<floatT> &);
    __device__ __host__ friend Vect3<floatT> operator-  <> (const Vect3<floatT> &,const Vect3<floatT> &);
    __device__ __host__ friend Vect3<floatT> operator*  <> (const floatT &,const Vect3<floatT> &);
    __device__ __host__ friend Vect3<floatT> operator*  <> (const COMPLEX(floatT) &,const Vect3<floatT> &);
    __device__ __host__ friend Vect3<floatT> operator*  <> (const Vect3<floatT> &,const floatT &);
    __device__ __host__ friend Vect3<floatT> operator*  <> (const Vect3<floatT> &,const COMPLEX(floatT) &);

    __device__ __host__ friend Vect3<floatT> conj <> (const Vect3<floatT> &);  // complex conjugate
    __device__ __host__ friend floatT norm2 <> (const Vect3<floatT> &);  // norm2
    __device__ __host__ friend COMPLEX(floatT) dot_prod <> (const Vect3<floatT>&, const Vect3<floatT>&); // true complex dot product
    __device__ __host__ friend floatT re_dot_prod <> (const Vect3<floatT> &,const Vect3<floatT> &);  // real part of dot product
    template<class rndstateT>
    __device__ __host__ void random( rndstateT * const);   // set vect3 randomly
    __device__ __host__ void gauss( uint4 * state )
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

        _v0 = COMPLEX(floatT)(radius0 * cos(phi0), radius0 * sin(phi0));
        _v1 = COMPLEX(floatT)(radius1 * cos(phi1), radius1 * sin(phi1));
        _v2 = COMPLEX(floatT)(radius2 * cos(phi2), radius2 * sin(phi2));
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

        _v0 = COMPLEX(__half)(__float2half(radius0 * cos(phi0)), __float2half(radius0 * sin(phi0)));
        _v1 = COMPLEX(__half)(__float2half(radius1 * cos(phi1)), __float2half(radius1 * sin(phi1)));
        _v2 = COMPLEX(__half)(__float2half(radius2 * cos(phi2)), __float2half(radius2 * sin(phi2)));
        #endif
        }
#endif
    };

    // cast operations single <-> double precision
    template <class T>
    __device__ __host__ operator Vect3<T> () const {
        return Vect3<T>( COMPLEX(T)(_v0.cREAL, _v0.cIMAG), COMPLEX(T)(_v1.cREAL, _v1.cIMAG), COMPLEX(T)(_v2.cREAL, _v2.cIMAG) );
    }


    __device__ __host__ friend Vect3<floatT> operator* <> (const SU3<floatT> &,const Vect3<floatT> &);   // su3 * vect3 multiplication
    __device__ __host__ friend SU3<floatT> tensor_prod <> (const Vect3<floatT> &,const Vect3<floatT> &); // tensor product of two vect3


    __device__ __host__ inline COMPLEX(floatT) getElement0() const  {
        return _v0;
    };

    __device__ __host__ inline COMPLEX(floatT) getElement1()const  {
        return _v1;
    };

    __device__ __host__  inline COMPLEX(floatT) getElement2() const {
        return _v2;
    };

    __device__ __host__ inline void addtoElement0(const COMPLEX(floatT) a){
        _v0 += a;
    }
    __device__ __host__ inline void addtoElement1(const COMPLEX(floatT) a){
        _v1 += a;
    }
    __device__ __host__ inline void addtoElement2(const COMPLEX(floatT) a){
        _v2 += a;
    }

    __device__ __host__ inline void setElement0(const COMPLEX(floatT)& a){
        _v0 = a;
    }
    __device__ __host__ inline void setElement1(const COMPLEX(floatT)& a){
        _v1 = a;
    }
    __device__ __host__ inline void setElement2(const COMPLEX(floatT)& a){
        _v2 = a;
    }

    __device__ __host__ inline void subfromElement0(const COMPLEX(floatT) a){
        _v0 -= a;
    }
    __device__ __host__ inline void subfromElement1(const COMPLEX(floatT) a){
        _v1 -= a;
    }
    __device__ __host__ inline void subfromElement2(const COMPLEX(floatT) a){
        _v2 -= a;
    }

    __device__ __host__ inline COMPLEX(floatT)& operator() (int i) {
        switch (i) {
            case 0:
                return _v0;
            case 1:
                return _v1;
            case 2:
                return _v2;
            default:
                return _v0;
        }
    }


    __host__ __device__ Vect3<floatT> getAccessor() const{
        return *this;
    }

    template <typename Index>
    __host__ __device__ Vect3<floatT> operator()(const Index) const {
        return *this;
    }
};


// vect3 = (1,0,0)  or (0,1,0)  or  (0,0,1)
template <class floatT>
__device__ __host__ inline Vect3<floatT> vect3_unity(const int& i)
{
    switch ( i )
    {
    case 1:
        return Vect3<floatT> (0, 1, 0);
    case 2:
        return Vect3<floatT> (0, 0, 1);
    }
// default value
    return Vect3<floatT> (1, 0, 0);
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
#endif
// cvect3 = (1,1,1)
template <class floatT>
__device__ __host__ inline Vect3<floatT> vect3_one()
{
    return Vect3<floatT> (1, 1, 1);
}



// cvect3 = (0,0,0)
template <class floatT>
__device__ __host__ inline Vect3<floatT> vect3_zero()
{
    return Vect3<floatT> (0, 0, 0);
}
#ifndef USE_HIP_AMD
template<>
__device__ inline Vect3<__half> vect3_zero()
{
    return Vect3<__half> (__float2half(0), __float2half(0), __float2half(0));
}
#endif
template <class floatT>
__device__ __host__ Vect3<floatT> &Vect3<floatT>::operator=(const Vect3<floatT> &y)
{
    _v0 = y._v0;
    _v1 = y._v1;
    _v2 = y._v2;
    return (*this);
}

template <class floatT>
__device__ __host__ Vect3<floatT> &Vect3<floatT>::operator-=(const Vect3<floatT> &y)
{
    _v0-= y._v0;
    _v1-= y._v1;
    _v2-= y._v2;
    return (*this);
}

template <class floatT>
__device__ __host__ Vect3<floatT> &Vect3<floatT>::operator+=(const Vect3<floatT> &y)
{
    _v0+= y._v0;
    _v1+= y._v1;
    _v2+= y._v2;
    return (*this);
}

template <class floatT>
__device__ __host__ Vect3<floatT> &Vect3<floatT>::operator*=(const floatT &y)
{
    _v0*= y;
    _v1*= y;
    _v2*= y;
    return (*this);
}

template <class floatT>
__device__ __host__ Vect3<floatT> &Vect3<floatT>::operator*=(const COMPLEX(floatT) &y)
{
    _v0*= y;
    _v1*= y;
    _v2*= y;
    return (*this);
}

template <class floatT>
__device__ __host__ COMPLEX(floatT) operator*(const Vect3<floatT> &x,const Vect3<floatT> &y)
{
    COMPLEX(floatT) res = conj(x._v0) * y._v0;
    res += conj(x._v1) * y._v1;
    res += conj(x._v2) * y._v2;
    return res;
}

template <class floatT>
__device__ __host__ COMPLEX(floatT) complex_product(const Vect3<floatT> &x,const Vect3<floatT> &y)
{
//  COMPLEX(floatT) res = x._v0 *(y._v0);
//  res += x._v1 * (y._v1);
//  res += x._v2 * (y._v2);
//  return res;

  return  fma(x._v0 ,y._v0,
            fma(x._v1 , (y._v1), x._v2 * (y._v2)));


}

template <class floatT>
__device__ __host__ COMPLEX(floatT) complex_product_add(const Vect3<floatT> &x,const Vect3<floatT> &y, const COMPLEX(floatT) &d)
{
    //COMPLEX(floatT) res = x._v0 *(y._v0);
    //res += x._v1 * (y._v1);
    //res += x._v2 * (y._v2);
    //return res;

    return  fma(x._v0 ,y._v0,
                fma(x._v1 , (y._v1),fma(x._v2 ,(y._v2),d)));
}

template <class floatT>
__device__ __host__ Vect3<floatT> operator+(const Vect3<floatT> &x,const Vect3<floatT> &y)
{
    Vect3<floatT> z;
    z._v0 = x._v0 + y._v0;
    z._v1 = x._v1 + y._v1;
    z._v2 = x._v2 + y._v2;
    return z;
}

template <class floatT>
__device__ __host__ Vect3<floatT> operator-(const Vect3<floatT> &x,const Vect3<floatT> &y)
{
    Vect3<floatT> z;
    z._v0 = x._v0 - y._v0;
    z._v1 = x._v1 - y._v1;
    z._v2 = x._v2 - y._v2;
    return z;
}

template <class floatT>
__device__ __host__ Vect3<floatT> operator*(const COMPLEX(floatT)& x,const Vect3<floatT>& y)
{
    Vect3<floatT> z;
    z._v0 = x * y._v0;
    z._v1 = x * y._v1;
    z._v2 = x * y._v2;
    return z;
}

template <class floatT>
__device__ __host__ Vect3<floatT> operator*(const floatT & x,const Vect3<floatT>& y)
{
    Vect3<floatT> z;
    z._v0 = x * y._v0;
    z._v1 = x * y._v1;
    z._v2 = x * y._v2;
    return z;
}

template <class floatT>
__device__ __host__ Vect3<floatT> operator*(const Vect3<floatT>& x,const COMPLEX(floatT)& y)
{
    Vect3<floatT> z;
    z._v0 = x._v0 * y;
    z._v1 = x._v1 * y;
    z._v2 = x._v2 * y;
    return z;
}

template <class floatT>
__device__ __host__ Vect3<floatT> operator*(const Vect3<floatT>& x,const floatT & y)
{
    Vect3<floatT> z;
    z._v0 = x._v0 * y;
    z._v1 = x._v1 * y;
    z._v2 = x._v2 * y;
    return z;
}

//! complex dot product x*y = sum_i(v_i conj(w_i))
template <class floatT>
__device__ __host__ COMPLEX(floatT) dot_prod(const Vect3<floatT> &x,const Vect3<floatT> &y)
{
    floatT real = x._v0.cREAL*y._v0.cREAL + x._v0.cIMAG*y._v0.cIMAG;
    real       += x._v1.cREAL*y._v1.cREAL + x._v1.cIMAG*y._v1.cIMAG;
    real       += x._v2.cREAL*y._v2.cREAL + x._v2.cIMAG*y._v2.cIMAG;
    floatT imag = x._v0.cIMAG*y._v0.cREAL - x._v0.cREAL*y._v0.cIMAG;
    imag       += x._v1.cIMAG*y._v1.cREAL - x._v1.cREAL*y._v1.cIMAG;
    imag       += x._v2.cIMAG*y._v2.cREAL - x._v2.cREAL*y._v2.cIMAG;
    return COMPLEX(floatT)(real,imag);
}

//! real part of dot product (no conjugation for y)
template <class floatT>
__device__ __host__ floatT re_dot_prod(const Vect3<floatT> &x,const Vect3<floatT> &y)
{
  floatT res = x._v0.cREAL*y._v0.cREAL + x._v0.cIMAG*y._v0.cIMAG;
  res       += x._v1.cREAL*y._v1.cREAL + x._v1.cIMAG*y._v1.cIMAG;
  res       += x._v2.cREAL*y._v2.cREAL + x._v2.cIMAG*y._v2.cIMAG;
  return (res);
}

// norm2 of vector
template <class floatT>
__device__ __host__ floatT norm2(const Vect3<floatT> &x)
{
  floatT res = x._v0.cREAL*x._v0.cREAL + x._v0.cIMAG*x._v0.cIMAG;
  res       += x._v1.cREAL*x._v1.cREAL + x._v1.cIMAG*x._v1.cIMAG;
  res       += x._v2.cREAL*x._v2.cREAL + x._v2.cIMAG*x._v2.cIMAG;
  return (res);
}

// complex conjugate
template <class floatT>
__device__ __host__ Vect3<floatT> conj(const Vect3<floatT> &x)
{
    Vect3<floatT> z;
    z._v0 = conj(x._v0);
    z._v1 = conj(x._v1);
    z._v2 = conj(x._v2);
    return z;
}


#ifdef __GPUCC__

template <class floatT>
__host__ std::ostream &operator << (std::ostream &s, const Vect3<floatT> &x)
{
    return s << x.getElement0() << x.getElement1() << x.getElement2();
}

template <class floatT>
__host__ std::istream &operator >> (std::istream &s, Vect3<floatT> &x)
{
    return s >> x._v0.cREAL >> x._v0.cIMAG >> x._v1.cREAL >> x._v1.cIMAG >> x._v2.cREAL >> x._v2.cIMAG;
}



#endif


#endif


