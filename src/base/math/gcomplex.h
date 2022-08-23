/*
   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Library General Public
   License version 2 as published by the Free Software Foundation.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Library General Public License for more details.

   You should have received a copy of the GNU Library General Public License
   along with this library; see the file COPYING.LIB.  If not, write to
   the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.
*/

#ifndef SP_COMPLEX_HCU
#define SP_COMPLEX_HCU


#include "../wrapper/gpu_wrapper.h"
#include "floatComparison.h"
#include <complex>
#include <iostream>
#include <type_traits>

template <typename T>
class Selector; // implement this one as well, if you want to have a default...

template <> class Selector<float> {
public:
  using Type = float2;
};
template <> class Selector<double> {
public:
  using Type = double2;
};
#ifndef USE_CPU_ONLY
template <> class Selector<__half> {
public:
  using Type = __half2;
};
#endif

/**
 * A utility class to provide complex numbers for operation
 * on the GPU
 */
template <class floatT, typename floatT2 = typename Selector<floatT>::Type>
class GPUcomplex {
public:
    floatT2 c;
#define cREAL c.x
#define cIMAG c.y

  /**
   * Default constructor, leave values uninitialized.
   */
  HOST_DEVICE GPUcomplex(){};
  constexpr GPUcomplex(const GPUcomplex<floatT, floatT2> &) = default;

  /**
   * Utility constructor, creates class from given real and imaginary value
   */
  HOST_DEVICE GPUcomplex(const floatT &real, const floatT &imag) {
    cREAL = real;
    cIMAG = imag;
  };

  /**
   * Utility constructor, creates class from real value, assumes imaginary value
   * to be zero.
   */
  HOST_DEVICE GPUcomplex(const floatT &real) {
    cREAL = real;
    cIMAG = 0.0f;
  };

  SQCD_HOST GPUcomplex(const std::complex<float> &orig) {
    cREAL = std::real<floatT>(orig);
    cIMAG = std::imag<floatT>(orig);
  }
  SQCD_HOST GPUcomplex(const std::complex<double> &orig) {
    cREAL = std::real<floatT>(orig);
    cIMAG = std::imag<floatT>(orig);
  }

  HOST_DEVICE GPUcomplex &operator=(const GPUcomplex<float> &orig) {
    this->c = static_cast<floatT2>(orig.c);
    return *this;
  }

  HOST_DEVICE GPUcomplex &operator=(const GPUcomplex<double> &orig) {
    this->c = static_cast<floatT2>(orig.c);
    return *this;
  }

  HOST_DEVICE GPUcomplex &operator=(const floatT &orig) {
    this->cREAL = orig;
    this->cIMAG = 0.0f;
    return *this;
  }

  HOST_DEVICE GPUcomplex &operator+=(const GPUcomplex &op) {
    this->cREAL += op.cREAL;
    this->cIMAG += op.cIMAG;
    return *this;
  }

  HOST_DEVICE GPUcomplex &operator+=(const floatT &op) {
    this->cREAL += op;
    return *this;
  }

  HOST_DEVICE GPUcomplex &operator-=(const GPUcomplex &op) {
    this->cREAL -= op.cREAL;
    this->cIMAG -= op.cIMAG;
    return *this;
  }

  HOST_DEVICE GPUcomplex &operator-=(const floatT &op) {
    this->cREAL -= op;
    return *this;
  }

  HOST_DEVICE GPUcomplex &operator*=(const GPUcomplex &op) {
    floatT newReal = this->cREAL * op.cREAL - this->cIMAG * op.cIMAG;
    this->cIMAG = this->cREAL * op.cIMAG + this->cIMAG * op.cREAL;
    this->cREAL = newReal;
    return *this;
  }

  HOST_DEVICE GPUcomplex &operator*=(const floatT &op) {
    this->cREAL *= op;
    this->cIMAG *= op;
    return *this;
  }

  HOST_DEVICE GPUcomplex &operator/=(const floatT &op) {
    this->cREAL /= op;
    this->cIMAG /= op;
    return *this;
  }

  /// Note: You should not use this operator to compare with zero, because
  /// cmp_rel breaks down in that case.
  HOST_DEVICE bool operator==(const GPUcomplex &op) {
    ////TODO:: THAT PRECISION HAS TO BE CHANGED!!
    return (cmp_rel<floatT>(this->cREAL, op.cREAL, 1.e-6, 1.e-6) &&
            cmp_rel<floatT>(this->cIMAG, op.cIMAG, 1.e-6, 1.e-6));
    //	return (isApproximatelyEqual<floatT>(this->cREAL, op.cREAL, 1.e-14) &&
    //isApproximatelyEqual<floatT>(this->cIMAG, op.cIMAG, 1.e-14));
  }

  HOST_DEVICE friend GPUcomplex operator+(const GPUcomplex &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(left.cREAL + right.cREAL, left.cIMAG + right.cIMAG);
  }

  HOST_DEVICE friend GPUcomplex operator+(const GPUcomplex &left,
                                                  const floatT &right) {
    return GPUcomplex(left.cREAL + right, left.cIMAG);
  }

  HOST_DEVICE friend GPUcomplex operator+(const floatT &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(left + right.cREAL, right.cIMAG);
  }

  HOST_DEVICE friend GPUcomplex operator-(const GPUcomplex &op) {
    return GPUcomplex(-op.cREAL, -op.cIMAG);
  }

  HOST_DEVICE friend GPUcomplex operator-(const GPUcomplex &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(left.cREAL - right.cREAL, left.cIMAG - right.cIMAG);
  }

  HOST_DEVICE friend GPUcomplex operator-(const GPUcomplex &left,
                                                  const floatT &right) {
    return GPUcomplex(left.cREAL - right, left.cIMAG);
  }

  HOST_DEVICE friend GPUcomplex operator-(const floatT &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(left - right.cREAL, -right.cIMAG);
  }

  HOST_DEVICE friend GPUcomplex operator*(const GPUcomplex &left,
                                                  const GPUcomplex &right) {
    floatT newReal = left.cREAL * right.cREAL - left.cIMAG * right.cIMAG;
    floatT newImag = left.cREAL * right.cIMAG + left.cIMAG * right.cREAL;
    return GPUcomplex(newReal, newImag);
  }

  HOST_DEVICE friend GPUcomplex operator*(const GPUcomplex &left,
                                                  const floatT &right) {
    return GPUcomplex(left.cREAL * right, left.cIMAG * right);
  }

  HOST_DEVICE friend GPUcomplex operator*(const floatT &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(left * right.cREAL, left * right.cIMAG);
  }

  HOST_DEVICE friend GPUcomplex
  fma(const GPUcomplex &x, const GPUcomplex &y, const GPUcomplex &d) {
    floatT real_res;
    floatT imag_res;

    real_res = (x.cREAL * y.cREAL) + d.cREAL;
    imag_res = (x.cREAL * y.cIMAG) + d.cIMAG;

    real_res = -(x.cIMAG * y.cIMAG) + real_res;
    imag_res = (x.cIMAG * y.cREAL) + imag_res;

    return GPUcomplex(real_res, imag_res);
  }

  HOST_DEVICE friend GPUcomplex fma(const floatT x, const GPUcomplex &y,
                                            const GPUcomplex &d) {
    floatT real_res;
    floatT imag_res;

    real_res = (x * y.cREAL) + d.cREAL;
    imag_res = (x * y.cIMAG) + d.cIMAG;

    return GPUcomplex(real_res, imag_res);
  }

  HOST_DEVICE void addProduct(const GPUcomplex &x,
                                      const GPUcomplex &y) {
    this->cREAL = (x.cREAL * y.cREAL) + this->cREAL;
    this->cIMAG = (x.cREAL * y.cIMAG) + this->cIMAG;

    this->cREAL = -(x.cIMAG * y.cIMAG) + this->cREAL;
    this->cIMAG = (x.cIMAG * y.cREAL) + this->cIMAG;

    return;
  }

  HOST_DEVICE void addProduct(const floatT &x, const GPUcomplex &y) {
    this->cREAL = (x * y.cREAL) + this->cREAL;
    this->cIMAG = (x * y.cIMAG) + this->cIMAG;

    return;
  }

  template <typename T>
  HOST_DEVICE friend GPUcomplex operator/(const GPUcomplex &left,
                                                  const T &right) {
    return GPUcomplex(left.cREAL / right, left.cIMAG / right);
  }

  template <typename T>
  HOST_DEVICE friend GPUcomplex operator/(const T &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(
        left * right.cREAL / (right.cREAL * right.cREAL + right.cIMAG * right.cIMAG),
        -left * right.cIMAG / (right.cREAL * right.cREAL + right.cIMAG * right.cIMAG));
  }

  HOST_DEVICE inline static GPUcomplex invalid();

  // These are needed to make sure that dp_complex may be part in general
  // operators src/math/operators.h
  HOST_DEVICE GPUcomplex getAccessor() const { return *this; }

  template <typename Index>
  HOST_DEVICE GPUcomplex operator()(const Index) const {
    return *this;
  }
};

#ifdef __GPU_ARCH__
#ifndef USE_HIP
template <> class GPUcomplex<__half> {
public:
  __half2 c;
  HOST_DEVICE GPUcomplex(){};

  HOST_DEVICE GPUcomplex(const __half &real, const __half &imag) {
    cREAL = real;
    cIMAG = imag;
  };

  HOST_DEVICE GPUcomplex(const __half &real) {
    cREAL = real;
    cIMAG = __float2half(0.0f);
  };

  HOST_DEVICE GPUcomplex(const __half2 &vec_type) { c = vec_type; };

  HOST_DEVICE GPUcomplex &operator=(const GPUcomplex<float> &orig) {
    this->c = __float22half2_rn(orig.c);
    return *this;
  }

  HOST_DEVICE GPUcomplex &operator=(const GPUcomplex<double> &orig) {
    __half realpart = __double2half(orig.cREAL);
    __half imagpart = __double2half(orig.cIMAG);
    this->c = __halves2half2(realpart, imagpart);
    return *this;
  }

  HOST_DEVICE GPUcomplex &operator=(const GPUcomplex<__half> orig) {
    this->c = static_cast<__half2>(orig.c);
    return *this;
  }
  HOST_DEVICE GPUcomplex &operator=(const __half &orig) {
    this->cREAL = orig;
    this->cIMAG = 0.0f;
    return *this;
  }
  HOST_DEVICE GPUcomplex &operator+=(const __half &op) {
    this->cREAL += op;
    return *this;
  }
  HOST_DEVICE GPUcomplex &operator+=(const GPUcomplex &op) {
    this->c += op.c;
    return *this;
  }

  HOST_DEVICE GPUcomplex &operator-=(const GPUcomplex &op) {
    this->c -= op.c;
    return *this;
  }

  HOST_DEVICE GPUcomplex &operator*=(const GPUcomplex &op) {

    const __half2 a_re = __half2half2(this->cREAL);
    __half2 acc = __hfma2(a_re, op.c, __float2half2_rn(0.0));
    const __half2 a_im = __half2half2(this->cIMAG);
    const __half2 ib = __halves2half2(__hneg(op.cIMAG), op.cREAL);
    acc = __hfma2(a_im, ib, acc);
    //            __half2 result = __hcmadd( this->c , op.c , __float2half2_rn (
    //            0.0 ) );
    this->c = acc;
    return *this;
  }

  HOST_DEVICE GPUcomplex &operator*=(const __half &op) {
    __half2 temp = __half2half2(op);
    this->c *= temp;
    return *this;
  }

  HOST_DEVICE GPUcomplex &operator/=(const __half &op) {
    __half2 temp = __half2half2(op);
    this->c /= temp;
    return *this;
  }

  HOST_DEVICE friend GPUcomplex operator+(const GPUcomplex left,
                                                  const GPUcomplex right) {
    return GPUcomplex(left.c + right.c);
  }

  HOST_DEVICE friend GPUcomplex operator+(const GPUcomplex &left,
                                                  const __half &right) {
    return GPUcomplex(left.cREAL + right, left.cIMAG);
  }

  HOST_DEVICE friend GPUcomplex operator+(const __half &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(left + right.cREAL, right.cIMAG);
  }

  HOST_DEVICE friend GPUcomplex operator-(const GPUcomplex &op) {
    return GPUcomplex(-op.c);
  }

  HOST_DEVICE friend GPUcomplex operator-(const GPUcomplex &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(left.c - right.c);
  }

  HOST_DEVICE friend GPUcomplex operator-(const GPUcomplex &left,
                                                  const __half &right) {
    return GPUcomplex(left.cREAL - right, left.cIMAG);
  }

  HOST_DEVICE friend GPUcomplex operator-(const __half &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(left - right.cREAL, -right.cIMAG);
  }

  HOST_DEVICE friend GPUcomplex operator*(const GPUcomplex &left,
                                                  const GPUcomplex &right) {

    const __half2 a_re = __half2half2(left.cREAL);
    __half2 acc = __hfma2(a_re, right.c, __float2half2_rn(0.0));
    const __half2 a_im = __half2half2(left.cIMAG);
    const __half2 ib = __halves2half2(__hneg(right.cIMAG), right.cREAL);
    acc = __hfma2(a_im, ib, acc);
    //            __half2 result = __hcmadd( left.c , right.c , __float2half2_rn
    //            ( 0.0 ) );

    return GPUcomplex(acc);
  }

  HOST_DEVICE friend GPUcomplex operator*(const GPUcomplex &left,
                                                  const __half &right) {
    return GPUcomplex(left.cREAL * right, left.cIMAG * right);
  }

  HOST_DEVICE friend GPUcomplex operator*(const __half &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(left * right.cREAL, left * right.cIMAG);
  }

  HOST_DEVICE friend GPUcomplex
  fma(const GPUcomplex &a, const GPUcomplex &b, const GPUcomplex &d) {
    const __half2 a_re = __half2half2(a.cREAL);
    __half2 acc = __hfma2(a_re, b.c, d.c);
    const __half2 a_im = __half2half2(a.cIMAG);
    const __half2 ib = __halves2half2(__hneg(b.cIMAG), b.cREAL);
    acc = __hfma2(a_im, ib, acc);
    //            return GPUcomplex( __hcmadd( x.c, y.c, d.c ) );
    return GPUcomplex(acc);
  }

  HOST_DEVICE friend GPUcomplex fma(const __half x, const GPUcomplex &y,
                                            const GPUcomplex &d) {
    __half2 xh2 = __half2half2(x);
    return GPUcomplex(__hfma2(xh2, y.c, d.c));
  }

  HOST_DEVICE void addProduct(const GPUcomplex &a,
                                      const GPUcomplex &b) {
    const __half2 a_re = __half2half2(a.cREAL);
    __half2 acc = __hfma2(a_re, b.c, this->c);
    const __half2 a_im = __half2half2(a.cIMAG);
    const __half2 ib = __halves2half2(__hneg(b.cIMAG), b.cREAL);
    acc = __hfma2(a_im, ib, acc);
    this->c = acc;
    // this->c = __hcmadd( x.c, y.c, this->c );
    return;
  }

  HOST_DEVICE void addProduct(const __half &x, const GPUcomplex &y) {
    __half2 xh2 = __half2half2(x);
    this->c = __hfma2(xh2, y.c, this->c);
    return;
  }

  template <typename T>
  HOST_DEVICE friend GPUcomplex operator/(const GPUcomplex &left,
                                                  const T &right) {
    return GPUcomplex(left.cREAL / right, left.cIMAG / right);
  }

  HOST_DEVICE friend GPUcomplex operator/(const GPUcomplex &left,
                                                  const __half &right) {
    __half2 right2 = __half2half2(right);

    return GPUcomplex(left.c / right2);
  }

  template <typename T>
  HOST_DEVICE friend GPUcomplex operator/(const T &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(
        left * right.cREAL / (right.cREAL * right.cREAL + right.cIMAG * right.cIMAG),
        -left * right.cIMAG / (right.cREAL * right.cREAL + right.cIMAG * right.cIMAG));
  }

  HOST_DEVICE inline static GPUcomplex invalid();

  HOST_DEVICE GPUcomplex getAccessor() const { return *this; }

  template <typename Index>
  HOST_DEVICE GPUcomplex operator()(const Index) const {
    return *this;
  }
};
#endif
#endif

template <class floatT>
HOST_DEVICE inline floatT real(const GPUcomplex<floatT> &op) {
  return op.cREAL;
}

template <class floatT>
HOST_DEVICE inline floatT imag(const GPUcomplex<floatT> &op) {
  return op.cIMAG;
}

template <class floatT>
HOST_DEVICE inline floatT abs(const GPUcomplex<floatT> &op) {
  floatT square = op.cREAL * op.cREAL + op.cIMAG * op.cIMAG;
  return sqrtf(square);
}

template <class floatT>
HOST_DEVICE inline floatT abs2(const GPUcomplex<floatT> &op) {
  return op.cREAL * op.cREAL + op.cIMAG * op.cIMAG;
}

template <class floatT>
HOST_DEVICE inline GPUcomplex<floatT>
conj(const GPUcomplex<floatT> &op) {
  return GPUcomplex<floatT>(op.cREAL, -op.cIMAG);
}

template <class floatT>
HOST_DEVICE inline floatT arg(const GPUcomplex<floatT> &op) {
  return atan2(op.cIMAG, op.cREAL);
}

template <class floatT>
HOST_DEVICE inline GPUcomplex<floatT>
cupow(const GPUcomplex<floatT> &base, const floatT &exp) {
  return GPUcomplex<floatT>(pow(abs(base), exp) * cos(arg(base) * exp),
                            pow(abs(base), exp) * sin(arg(base) * exp));
}

template <class floatT>
HOST_DEVICE inline GPUcomplex<floatT>
cusqrt(const GPUcomplex<floatT> &base) {
  return GPUcomplex<floatT>(sqrt(abs(base)) * cos(arg(base) * 0.5),
                            sqrt(abs(base)) * sin(arg(base) * 0.5));
}

template <class floatT>
const GPUcomplex<floatT> GPUcomplex_invalid(nanf(" "), nanf(" "));

template <class floatT>
SQCD_HOST inline std::ostream &operator<<(std::ostream &s,
                                         GPUcomplex<floatT> z) {
  return s << '(' << real(z) << ',' << imag(z) << ')';
}

template <class floatT, typename floatT2>
HOST_DEVICE inline GPUcomplex<floatT, floatT2>
GPUcomplex<floatT, floatT2>::invalid() {
  return GPUcomplex_invalid<floatT>;
}

template <class floatT>
HOST_DEVICE inline bool
compareGCOMPLEX(GPUcomplex<floatT> a, GPUcomplex<floatT> b, floatT tol) {
  floatT diffRe = abs(real(a) - real(b));
  floatT diffIm = abs(imag(a) - imag(b));
  if (diffRe > tol || diffIm > tol)
    return false;
  return true;
}

#define GCOMPLEX(floatT) GPUcomplex<floatT>

#endif // SP_COMPLEX_HCU
