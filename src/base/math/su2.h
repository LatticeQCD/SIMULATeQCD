/*
 * su2.h
 *
 * Header file for algebra involving SU(2) subgoups embedded in SU(3).
 *
 *
 *
 *  Representation:
 *
 *
 *
 *  SU(2) =  ( _e11              _e12   )
 *           ( -conj(_e12)   conj(_e11) )
 *
 */

#ifndef _su2_h_
#define _su2_h_

#include <complex>
#include "complex.h"
#include "su3.h"

template<typename floatT>
class SU2 {
public:

  __device__ __host__ SU2() { };
  COMPLEX(floatT) _e11,_e12;
  __device__ __host__ SU2(COMPLEX(floatT) e11, COMPLEX(floatT) e12) : _e11(e11), _e12(e12) {}


  __device__ __host__  friend SU2 operator+(const SU2 &x,const SU2 &y) {
    return SU2 (x._e11+y._e11,x._e12+y._e12);
  }

  __device__ __host__  friend SU2 operator-(const SU2 &x,const SU2 &y) {
    return SU2 (x._e11-y._e11,x._e12-y._e12);
  }

  __device__ __host__  friend SU2 operator*(const SU2 &x,const COMPLEX(floatT)  &y) {
    return SU2 (x._e11*y,x._e12*y);
  }

  __device__ __host__  friend SU2 operator*(const COMPLEX(floatT)  &x,const SU2 &y) {
    return SU2 (x*y._e11,x*y._e12);
  }

  __device__ __host__  friend SU2 operator*(const SU2 &x,const floatT  &y) {
    return SU2 (x._e11*y,x._e12*y);
  }

  __device__ __host__  friend SU2 operator*(const floatT  &x,const SU2 &y) {
    return SU2 (x*y._e11,x*y._e12);
  }

  __device__ __host__  friend SU2 operator/(const SU2 &x,const floatT  &y) {
    return SU2 (x._e11/y,x._e12/y);
  }

  __device__ __host__  friend SU2 operator*(const SU2 &x,const SU2 &y) {
    COMPLEX(floatT) tmp1,tmp2;
    tmp1=y._e12;
    tmp2=y._e11;
    tmp1=x._e11*y._e11-x._e12*conj(tmp1);
    tmp2=x._e11*y._e12+x._e12*conj(tmp2);
    return SU2 (tmp1,tmp2);
  }

  __device__ __host__  SU2 &operator =(const SU2 &y) {
    _e11=y._e11;
    _e12=y._e12;
    return *this;
  }
  __device__ __host__  SU2 &operator+=(const SU2 &y) {
    _e11+=y._e11;
    _e12+=y._e12;
    return *this;
  }
  __device__ __host__  SU2 &operator-=(const SU2 &y) {
    _e11-=y._e11;
    _e12-=y._e12;
    return *this;
  }
  __device__ __host__  SU2 &operator*=(const SU2 &y) {
    *this=*this*y;
    return *this;
  }
  __device__ __host__  SU2 &operator*=(const COMPLEX(floatT) &y) {
    _e11*=y;
    _e12*=y;
    return *this;
  }
  __device__ __host__  SU2 &operator*=(const floatT &y) {
    *this=*this*y;
    return *this;
  }
  __device__ __host__  SU2 &operator/=(const floatT &y) {
    *this=*this/y;
    return *this;
  }

  __device__ __host__  floatT tr2() {
    return( real(_e11) );
  }

  __device__ __host__  COMPLEX(floatT) det() {
    return( real(_e11) );
  }

  __device__ __host__  void unitarize() {
    floatT res;

    res = real(_e11)*real(_e11) + imag(_e11)*imag(_e11) +
      real(_e12)*real(_e12) + imag(_e12)*imag(_e12);
    res=1.0/sqrt(res);

    _e11=_e11*res;
    _e12=_e12*res;
  }

  __device__ __host__  SU2 dagger() const {
    SU2 tmp;

    tmp._e11 = conj(_e11);
    tmp._e12 = - _e12;

    return tmp;
  }

  __device__ __host__  floatT norm2() const {
    return (real(_e11)*real(_e11) + real(_e12)*real(_e12)
	    +   imag(_e11)*imag(_e11) + imag(_e12)*imag(_e12));
  }

private:

  friend std::ostream &operator << (std::ostream &s, const SU2 &x) {
    return s << x._e11  << x._e12;
  }

};

template<typename floatT>
__device__ __host__ inline SU2<floatT> dagger(const SU2<floatT> &x) {
    SU2<floatT> tmp;
    tmp._e11 = conj(x._e11);
    tmp._e12 = - x._e12;
    return tmp;
}

template<typename floatT>
__device__ __host__  inline floatT norm2(const SU2<floatT> &x) {
  return (  real(x._e11)*real(x._e11) + real(x._e12)*real(x._e12)
	        + imag(x._e11)*imag(x._e11) + imag(x._e12)*imag(x._e12) );
}

template<typename floatT>
__device__ __host__ inline  SU2<floatT> sub12 (const SU3<floatT> &u, const SU3<floatT> &v) {
  COMPLEX(floatT)  r00,r01,r10,r11;

  r00   = u.getLink00()*v.getLink00() + u.getLink01()*v.getLink10() + u.getLink02()*v.getLink20();
  r01   = u.getLink00()*v.getLink01() + u.getLink01()*v.getLink11() + u.getLink02()*v.getLink21();
  r10   = u.getLink10()*v.getLink00() + u.getLink11()*v.getLink10() + u.getLink12()*v.getLink20();
  r11   = u.getLink10()*v.getLink01() + u.getLink11()*v.getLink11() + u.getLink12()*v.getLink21();

  return SU2<floatT>( COMPLEX(floatT)(0.5 * (r00 + conj(r11))), COMPLEX(floatT)(0.5 * (r01 - conj(r10))) );
}

template<typename floatT>
__device__ __host__ inline SU2<floatT> sub13(const SU3<floatT> &u, const SU3<floatT> &v) {
  COMPLEX(floatT)  r00,r01,r10,r11;

  r00   = u.getLink00()*v.getLink00() + u.getLink01()*v.getLink10() + u.getLink02()*v.getLink20();
  r01   = u.getLink00()*v.getLink02() + u.getLink01()*v.getLink12() + u.getLink02()*v.getLink22();
  r10   = u.getLink20()*v.getLink00() + u.getLink21()*v.getLink10() + u.getLink22()*v.getLink20();
  r11   = u.getLink20()*v.getLink02() + u.getLink21()*v.getLink12() + u.getLink22()*v.getLink22();

  return SU2<floatT> ( COMPLEX(floatT)(0.5 * (r00 + conj(r11))), COMPLEX(floatT)(0.5 * (r01 - conj(r10))) );
}

template<typename floatT>
__device__ __host__ inline SU2<floatT> sub23(const SU3<floatT> &u, const SU3<floatT> &v) {
  COMPLEX(floatT)  r00,r01,r10,r11;

  r00   = u.getLink10()*v.getLink01() + u.getLink11()*v.getLink11() + u.getLink12()*v.getLink21();
  r01   = u.getLink10()*v.getLink02() + u.getLink11()*v.getLink12() + u.getLink12()*v.getLink22();
  r10   = u.getLink20()*v.getLink01() + u.getLink21()*v.getLink11() + u.getLink22()*v.getLink21();
  r11   = u.getLink20()*v.getLink02() + u.getLink21()*v.getLink12() + u.getLink22()*v.getLink22();

  return SU2<floatT> ( COMPLEX(floatT)(0.5 * (r00 + conj(r11))), COMPLEX(floatT)(0.5 * (r01 - conj(r10))) );
}

template<typename floatT>
__device__ __host__ inline SU3<floatT> sub12(const SU2<floatT> &u,
					      const SU3<floatT> &v) {
  return SU3<floatT> (u._e11 *v.getLink00() + u._e12 *v.getLink10(),
		                   u._e11 *v.getLink01() + u._e12 *v.getLink11(),
		                   u._e11 *v.getLink02() + u._e12 *v.getLink12(),
		                   -conj(u._e12)*v.getLink00() + conj(u._e11)*v.getLink10(),
		                   -conj(u._e12)*v.getLink01() + conj(u._e11)*v.getLink11(),
		                   -conj(u._e12)*v.getLink02() + conj(u._e11)*v.getLink12(),
		                   v.getLink20(),
		                   v.getLink21(),
		                   v.getLink22());
}

template<typename floatT>
__device__ __host__ inline SU3<floatT> sub13(const SU2<floatT> &u, const SU3<floatT> &v) {
  return SU3<floatT> (u._e11 *v.getLink00() + u._e12 *v.getLink20(),
		                   u._e11 *v.getLink01() + u._e12 *v.getLink21(),
		                   u._e11 *v.getLink02() + u._e12 *v.getLink22(),
		                   v.getLink10(),
		                   v.getLink11(),
		                   v.getLink12(),
		                   -conj(u._e12)*v.getLink00() + conj(u._e11)*v.getLink20(),
		                   -conj(u._e12)*v.getLink01() + conj(u._e11)*v.getLink21(),
		                   -conj(u._e12)*v.getLink02() + conj(u._e11)*v.getLink22());
}

template<typename floatT>
__device__ __host__ inline SU3<floatT> sub23(const SU2<floatT> &u, const SU3<floatT> &v) {
  return SU3<floatT> (v.getLink00(),
                       v.getLink01(),
		                   v.getLink02(),
		                   u._e11 *v.getLink10() + u._e12 *v.getLink20(),
		                   u._e11 *v.getLink11() + u._e12 *v.getLink21(),
	            	       u._e11 *v.getLink12() + u._e12 *v.getLink22(),
	            	       -conj(u._e12)*v.getLink10() + conj(u._e11)*v.getLink20(),
		                   -conj(u._e12)*v.getLink11() + conj(u._e11)*v.getLink21(),
		                   -conj(u._e12)*v.getLink12() + conj(u._e11)*v.getLink22());
}

template<typename floatT>
__device__ __host__  inline floatT realtrace(const SU3<floatT> &x) {
  return ( real(x.getLink00() + x.getLink11() +  x.getLink22()) );
}


#endif
