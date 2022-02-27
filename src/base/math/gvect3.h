/*-----------------------------------------------------------------------------------------------
 * /
 * / cvect3.h
 * /
 * / $Id: cvect3.h,v 1.1 2002/09/10 11:56:54 okacz Exp $
 * /
 * /---------------------------------------------------------------------------------------------*/

#ifndef _gvect3_h_
#define _gvect3_h_

#include "../../define.h"
#include "gcomplex.h"
#include "grnd.h"


// forward declaration
template <class floatT> class GSU3;
template <class floatT> class gVect3;
template <class floatT> class cVect3;
template <class floatT, bool onDevice> class gVect3array;
template <class floatT> __host__ std::ostream & operator<<(std::ostream &, const gVect3<floatT> &);
template <class floatT> __host__ std::istream & operator>>(std::istream &, gVect3<floatT> &);
template <class floatT>  __device__ __host__ GCOMPLEX(floatT) operator*(const gVect3<floatT> &,const gVect3<floatT> &);
template <class floatT>  __device__ __host__ GCOMPLEX(floatT) complex_product(const gVect3<floatT> &,const gVect3<floatT> &);
template <class floatT>  __device__ __host__ GCOMPLEX(floatT) complex_product_add(const gVect3<floatT> &,const gVect3<floatT> &, const GCOMPLEX(floatT) &);


template <class floatT>  __device__ __host__ gVect3<floatT> operator+(const gVect3<floatT> &,const gVect3<floatT> &);
template <class floatT>  __device__ __host__ gVect3<floatT> operator-(const gVect3<floatT> &,const gVect3<floatT> &);
template <class floatT>  __device__ __host__ gVect3<floatT> operator*(const floatT &,const gVect3<floatT> &);
template <class floatT>  __device__ __host__ gVect3<floatT> operator*(const GCOMPLEX(floatT) &,const gVect3<floatT> &);
template <class floatT>  __device__ __host__ gVect3<floatT> operator*(const gVect3<floatT> &,const floatT &);
template <class floatT>  __device__ __host__ gVect3<floatT> operator*(const gVect3<floatT> &,const GCOMPLEX(floatT) &);
template <class floatT> __device__ __host__ gVect3<floatT> conj(const gVect3<floatT> &);
template <class floatT> __device__ __host__ floatT norm2(const gVect3<floatT> &);
template <class floatT> __device__ __host__ GCOMPLEX(floatT) dot_prod(const gVect3<floatT> &,const gVect3<floatT> &);
template <class floatT> __device__ __host__ floatT re_dot_prod(const gVect3<floatT> &,const gVect3<floatT> &);
template <class floatT> __device__ __host__ gVect3<floatT> operator*(const GSU3<floatT> &,const gVect3<floatT> &);
template <class floatT> __device__ __host__ GSU3<floatT> tensor_prod(const gVect3<floatT> &,const gVect3<floatT> &);
template <class floatT> __device__ __host__ inline floatT minVal();

template<class floatT>
__device__ __host__ inline floatT get_rand(uint4* state);

template <class floatT>
class gVect3
{
private:

    GCOMPLEX(floatT) _v0, _v1, _v2;

public:

    __host__ __device__ gVect3() {};
    __host__ __device__ gVect3(GCOMPLEX(floatT) v0) : _v0(v0), _v1(v0), _v2(v0) {};
    __host__ __device__ gVect3(floatT v0) : _v0(v0), _v1(v0), _v2(v0) {};
    __host__ __device__ gVect3(GCOMPLEX(floatT) v0, GCOMPLEX(floatT) v1, GCOMPLEX(floatT) v2) : _v0(v0), _v1(v1), _v2(v2) {};

#if (!defined __GPUCC__)
    __host__ friend std::ostream &operator << <> (std::ostream &, const gVect3<floatT> &);
#endif
    __host__ friend std::istream &operator >> <> (std::istream &, gVect3<floatT> &);

    friend class gVect3array<floatT,true>;
    friend class gVect3array<floatT,false>;


    // vector operations
    __device__ __host__ gVect3<floatT> &operator =(const gVect3<floatT> &);
    __device__ __host__ gVect3<floatT> &operator-=(const gVect3<floatT> &);
    __device__ __host__ gVect3<floatT> &operator+=(const gVect3<floatT> &);
    __device__ __host__ gVect3<floatT> &operator*=(const floatT &);
    __device__ __host__ gVect3<floatT> &operator*=(const GCOMPLEX(floatT) &);
    __device__ __host__ friend GCOMPLEX(floatT) operator* <> (const gVect3<floatT> &,const gVect3<floatT> &);
    __device__ __host__ friend GCOMPLEX(floatT) complex_product <> (const gVect3<floatT> &,const gVect3<floatT> &);
    __device__ __host__ friend GCOMPLEX(floatT) complex_product_add <> (const gVect3<floatT> &,const gVect3<floatT> &, const GCOMPLEX(floatT) & );
    __device__ __host__ friend gVect3<floatT> operator+  <> (const gVect3<floatT> &,const gVect3<floatT> &);
    __device__ __host__ friend gVect3<floatT> operator-  <> (const gVect3<floatT> &,const gVect3<floatT> &);
    __device__ __host__ friend gVect3<floatT> operator*  <> (const floatT &,const gVect3<floatT> &);
    __device__ __host__ friend gVect3<floatT> operator*  <> (const GCOMPLEX(floatT) &,const gVect3<floatT> &);
    __device__ __host__ friend gVect3<floatT> operator*  <> (const gVect3<floatT> &,const floatT &);
    __device__ __host__ friend gVect3<floatT> operator*  <> (const gVect3<floatT> &,const GCOMPLEX(floatT) &);

    __device__ __host__ friend gVect3<floatT> conj <> (const gVect3<floatT> &);  // complex conjugate
    __device__ __host__ friend floatT norm2 <> (const gVect3<floatT> &);  // norm2
    __device__ __host__ friend GCOMPLEX(floatT) dot_prod <> (const gVect3<floatT>&, const gVect3<floatT>&); // true complex dot product
    __device__ __host__ friend floatT re_dot_prod <> (const gVect3<floatT> &,const gVect3<floatT> &);  // real part of dot product
    template<class rndstateT>
    __device__ __host__ void random( rndstateT * const);   // set gvect3 randomly
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

        _v0 = GCOMPLEX(floatT)(radius0 * cos(phi0), radius0 * sin(phi0));
        _v1 = GCOMPLEX(floatT)(radius1 * cos(phi1), radius1 * sin(phi1));
        _v2 = GCOMPLEX(floatT)(radius2 * cos(phi2), radius2 * sin(phi2));
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

        _v0 = GCOMPLEX(__half)(__float2half(radius0 * cos(phi0)), __float2half(radius0 * sin(phi0)));
        _v1 = GCOMPLEX(__half)(__float2half(radius1 * cos(phi1)), __float2half(radius1 * sin(phi1)));
        _v2 = GCOMPLEX(__half)(__float2half(radius2 * cos(phi2)), __float2half(radius2 * sin(phi2)));
        #endif
        }
#endif
    };

    // cast operations single <-> double precision
    template <class T>
    __device__ __host__ operator gVect3<T> () const {
        return gVect3<T>( GCOMPLEX(T)(_v0.cREAL, _v0.cIMAG), GCOMPLEX(T)(_v1.cREAL, _v1.cIMAG), GCOMPLEX(T)(_v2.cREAL, _v2.cIMAG) );
    }


    __device__ __host__ friend gVect3<floatT> operator* <> (const GSU3<floatT> &,const gVect3<floatT> &);   // gsu3 * gvect3 multiplication
    __device__ __host__ friend GSU3<floatT> tensor_prod <> (const gVect3<floatT> &,const gVect3<floatT> &); // tensor product of two gvect3


    __device__ __host__ inline GCOMPLEX(floatT) getElement0() const  {
        return _v0;
    };

    __device__ __host__ inline GCOMPLEX(floatT) getElement1()const  {
        return _v1;
    };

    __device__ __host__  inline GCOMPLEX(floatT) getElement2() const {
        return _v2;
    };

    __device__ __host__ inline void addtoElement0(const GCOMPLEX(floatT) a){
        _v0 += a;
    }
    __device__ __host__ inline void addtoElement1(const GCOMPLEX(floatT) a){
        _v1 += a;
    }
    __device__ __host__ inline void addtoElement2(const GCOMPLEX(floatT) a){
        _v2 += a;
    }

    __device__ __host__ inline void setElement0(const GCOMPLEX(floatT)& a){
        _v0 = a;
    }
    __device__ __host__ inline void setElement1(const GCOMPLEX(floatT)& a){
        _v1 = a;
    }
    __device__ __host__ inline void setElement2(const GCOMPLEX(floatT)& a){
        _v2 = a;
    }

    __device__ __host__ inline void subfromElement0(const GCOMPLEX(floatT) a){
        _v0 -= a;
    }
    __device__ __host__ inline void subfromElement1(const GCOMPLEX(floatT) a){
        _v1 -= a;
    }
    __device__ __host__ inline void subfromElement2(const GCOMPLEX(floatT) a){
        _v2 -= a;
    }

    __device__ __host__ inline GCOMPLEX(floatT)& operator() (int i) {
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


    __host__ __device__ gVect3<floatT> getAccessor() const{
        return *this;
    }

    template <typename Index>
    __host__ __device__ gVect3<floatT> operator()(const Index) const {
        return *this;
    }
};


// gvect3 = (1,0,0)  or (0,1,0)  or  (0,0,1)
template <class floatT>
__device__ __host__ inline gVect3<floatT> gvect3_unity(const int& i)
{
    switch ( i )
    {
    case 1:
        return gVect3<floatT> (0, 1, 0);
    case 2:
        return gVect3<floatT> (0, 0, 1);
    }
// default value
    return gVect3<floatT> (1, 0, 0);
}
#ifndef USE_HIP_AMD
template <>
__device__ inline gVect3<__half> gvect3_unity(const int& i)
{

    switch ( i )
    {
    case 1:
return gVect3<__half> (__float2half(0), __float2half(1), __float2half(0));
    case 2:
return gVect3<__half> (__float2half(0), __float2half(0), __float2half(1));
    }
// default value
return gVect3<__half> (__float2half(1), __float2half(0), __float2half(0));

}
#endif
// cvect3 = (1,1,1)
template <class floatT>
__device__ __host__ inline gVect3<floatT> gvect3_one()
{
    return gVect3<floatT> (1, 1, 1);
}



// cvect3 = (0,0,0)
template <class floatT>
__device__ __host__ inline gVect3<floatT> gvect3_zero()
{
    return gVect3<floatT> (0, 0, 0);
}
#ifndef USE_HIP_AMD
template<>
__device__ inline gVect3<__half> gvect3_zero()
{
    return gVect3<__half> (__float2half(0), __float2half(0), __float2half(0));
}
#endif
template <class floatT>
__device__ __host__ gVect3<floatT> &gVect3<floatT>::operator=(const gVect3<floatT> &y)
{
    _v0 = y._v0;
    _v1 = y._v1;
    _v2 = y._v2;
    return (*this);
}

template <class floatT>
__device__ __host__ gVect3<floatT> &gVect3<floatT>::operator-=(const gVect3<floatT> &y)
{
    _v0-= y._v0;
    _v1-= y._v1;
    _v2-= y._v2;
    return (*this);
}

template <class floatT>
__device__ __host__ gVect3<floatT> &gVect3<floatT>::operator+=(const gVect3<floatT> &y)
{
    _v0+= y._v0;
    _v1+= y._v1;
    _v2+= y._v2;
    return (*this);
}

template <class floatT>
__device__ __host__ gVect3<floatT> &gVect3<floatT>::operator*=(const floatT &y)
{
    _v0*= y;
    _v1*= y;
    _v2*= y;
    return (*this);
}

template <class floatT>
__device__ __host__ gVect3<floatT> &gVect3<floatT>::operator*=(const GCOMPLEX(floatT) &y)
{
    _v0*= y;
    _v1*= y;
    _v2*= y;
    return (*this);
}

template <class floatT>
__device__ __host__ GCOMPLEX(floatT) operator*(const gVect3<floatT> &x,const gVect3<floatT> &y)
{
    GCOMPLEX(floatT) res = conj(x._v0) * y._v0;
    res += conj(x._v1) * y._v1;
    res += conj(x._v2) * y._v2;
    return res;
}

template <class floatT>
__device__ __host__ GCOMPLEX(floatT) complex_product(const gVect3<floatT> &x,const gVect3<floatT> &y)
{
//  GCOMPLEX(floatT) res = x._v0 *(y._v0);
//  res += x._v1 * (y._v1);
//  res += x._v2 * (y._v2);
//  return res;

  return  fma(x._v0 ,y._v0,
            fma(x._v1 , (y._v1), x._v2 * (y._v2)));


}

template <class floatT>
__device__ __host__ GCOMPLEX(floatT) complex_product_add(const gVect3<floatT> &x,const gVect3<floatT> &y, const GCOMPLEX(floatT) &d)
{
    //GCOMPLEX(floatT) res = x._v0 *(y._v0);
    //res += x._v1 * (y._v1);
    //res += x._v2 * (y._v2);
    //return res;

    return  fma(x._v0 ,y._v0,
                fma(x._v1 , (y._v1),fma(x._v2 ,(y._v2),d)));
}

template <class floatT>
__device__ __host__ gVect3<floatT> operator+(const gVect3<floatT> &x,const gVect3<floatT> &y)
{
    gVect3<floatT> z;
    z._v0 = x._v0 + y._v0;
    z._v1 = x._v1 + y._v1;
    z._v2 = x._v2 + y._v2;
    return (z);
}

template <class floatT>
__device__ __host__ gVect3<floatT> operator-(const gVect3<floatT> &x,const gVect3<floatT> &y)
{
    gVect3<floatT> z;
    z._v0 = x._v0 - y._v0;
    z._v1 = x._v1 - y._v1;
    z._v2 = x._v2 - y._v2;
    return (z);
}

template <class floatT>
__device__ __host__ gVect3<floatT> operator*(const GCOMPLEX(floatT)& x,const gVect3<floatT>& y)
{
    gVect3<floatT> z;
    z._v0 = x * y._v0;
    z._v1 = x * y._v1;
    z._v2 = x * y._v2;
    return (z);
}

template <class floatT>
__device__ __host__ gVect3<floatT> operator*(const floatT & x,const gVect3<floatT>& y)
{
    gVect3<floatT> z;
    z._v0 = x * y._v0;
    z._v1 = x * y._v1;
    z._v2 = x * y._v2;
    return (z);
}

template <class floatT>
__device__ __host__ gVect3<floatT> operator*(const gVect3<floatT>& x,const GCOMPLEX(floatT)& y)
{
    gVect3<floatT> z;
    z._v0 = x._v0 * y;
    z._v1 = x._v1 * y;
    z._v2 = x._v2 * y;
    return (z);
}

template <class floatT>
__device__ __host__ gVect3<floatT> operator*(const gVect3<floatT>& x,const floatT & y)
{
    gVect3<floatT> z;
    z._v0 = x._v0 * y;
    z._v1 = x._v1 * y;
    z._v2 = x._v2 * y;
    return (z);
}

//! complex dot product x*y = sum_i(v_i conj(w_i))
template <class floatT>
__device__ __host__ GCOMPLEX(floatT) dot_prod(const gVect3<floatT> &x,const gVect3<floatT> &y)
{
    floatT real = x._v0.cREAL*y._v0.cREAL + x._v0.cIMAG*y._v0.cIMAG;
    real       += x._v1.cREAL*y._v1.cREAL + x._v1.cIMAG*y._v1.cIMAG;
    real       += x._v2.cREAL*y._v2.cREAL + x._v2.cIMAG*y._v2.cIMAG;
    floatT imag = x._v0.cIMAG*y._v0.cREAL - x._v0.cREAL*y._v0.cIMAG;
    imag       += x._v1.cIMAG*y._v1.cREAL - x._v1.cREAL*y._v1.cIMAG;
    imag       += x._v2.cIMAG*y._v2.cREAL - x._v2.cREAL*y._v2.cIMAG;
    return GCOMPLEX(floatT)(real,imag);
}

//! real part of dot product (no conjugation for y)
template <class floatT>
__device__ __host__ floatT re_dot_prod(const gVect3<floatT> &x,const gVect3<floatT> &y)
{
  floatT res = x._v0.cREAL*y._v0.cREAL + x._v0.cIMAG*y._v0.cIMAG;
  res       += x._v1.cREAL*y._v1.cREAL + x._v1.cIMAG*y._v1.cIMAG;
  res       += x._v2.cREAL*y._v2.cREAL + x._v2.cIMAG*y._v2.cIMAG;
  return (res);
}

// norm2 of vector
template <class floatT>
__device__ __host__ floatT norm2(const gVect3<floatT> &x)
{
  floatT res = x._v0.cREAL*x._v0.cREAL + x._v0.cIMAG*x._v0.cIMAG;
  res       += x._v1.cREAL*x._v1.cREAL + x._v1.cIMAG*x._v1.cIMAG;
  res       += x._v2.cREAL*x._v2.cREAL + x._v2.cIMAG*x._v2.cIMAG;
  return (res);
}

// complex conjugate
template <class floatT>
__device__ __host__ gVect3<floatT> conj(const gVect3<floatT> &x)
{
    gVect3<floatT> z;
    z._v0 = conj(x._v0);
    z._v1 = conj(x._v1);
    z._v2 = conj(x._v2);
    return (z);
}


#ifdef __GPUCC__

template <class floatT>
__host__ std::ostream &operator << (std::ostream &s, const gVect3<floatT> &x)
{
    return s << x.getElement0() << x.getElement1() << x.getElement2();
}

template <class floatT>
__host__ std::istream &operator >> (std::istream &s, gVect3<floatT> &x)
{
    return s >> x._v0.cREAL >> x._v0.cIMAG >> x._v1.cREAL >> x._v1.cIMAG >> x._v2.cREAL >> x._v2.cIMAG;
}



#endif


#endif


