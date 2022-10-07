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
template <> class Selector<__half> {
public:
  using Type = __half2;
};

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
  __host__ __device__ GPUcomplex(){};
  constexpr GPUcomplex(const GPUcomplex<floatT, floatT2> &) = default;

  /**
   * Utility constructor, creates class from given real and imaginary value
   */
  __host__ __device__ GPUcomplex(const floatT &real, const floatT &imag) {
    c.x = real;
    c.y = imag;
  };

  /**
   * Utility constructor, creates class from real value, assumes imaginary value
   * to be zero.
   */
  __host__ __device__ GPUcomplex(const floatT &real) {
    c.x = real;
    c.y = 0.0f;
  };

  __host__ GPUcomplex(const std::complex<float> &orig) {
    c.x = std::real<floatT>(orig);
    c.y = std::imag<floatT>(orig);
  }
  __host__ GPUcomplex(const std::complex<double> &orig) {
    c.x = std::real<floatT>(orig);
    c.y = std::imag<floatT>(orig);
  }

  __host__ __device__ GPUcomplex &operator=(const GPUcomplex<float> &orig) {
    this->c = static_cast<floatT2>(orig.c);
    return *this;
  }

  __host__ __device__ GPUcomplex &operator=(const GPUcomplex<double> &orig) {
    this->c = static_cast<floatT2>(orig.c);
    return *this;
  }

  __host__ __device__ GPUcomplex &operator=(const floatT &orig) {
    this->c.x = orig;
    this->c.y = 0.0f;
    return *this;
  }

  __host__ __device__ GPUcomplex &operator+=(const GPUcomplex &op) {
    this->c.x += op.c.x;
    this->c.y += op.c.y;
    return *this;
  }

  __host__ __device__ GPUcomplex &operator+=(const floatT &op) {
    this->c.x += op;
    return *this;
  }

  __host__ __device__ GPUcomplex &operator-=(const GPUcomplex &op) {
    this->c.x -= op.c.x;
    this->c.y -= op.c.y;
    return *this;
  }

  __host__ __device__ GPUcomplex &operator-=(const floatT &op) {
    this->c.x -= op;
    return *this;
  }

  __host__ __device__ GPUcomplex &operator*=(const GPUcomplex &op) {
    floatT newReal = this->c.x * op.c.x - this->c.y * op.c.y;
    this->c.y = this->c.x * op.c.y + this->c.y * op.c.x;
    this->c.x = newReal;
    return *this;
  }

  __host__ __device__ GPUcomplex &operator*=(const floatT &op) {
    this->c.x *= op;
    this->c.y *= op;
    return *this;
  }

  __host__ __device__ GPUcomplex &operator/=(const floatT &op) {
    this->c.x /= op;
    this->c.y /= op;
    return *this;
  }

  /// Note: You should not use this operator to compare with zero, because
  /// cmp_rel breaks down in that case.
  __host__ __device__ bool operator==(const GPUcomplex &op) {
    ////TODO:: THAT PRECISION HAS TO BE CHANGED!!
    return (cmp_rel<floatT>(this->c.x, op.c.x, 1.e-6, 1.e-6) &&
            cmp_rel<floatT>(this->c.y, op.c.y, 1.e-6, 1.e-6));  
    //	return (isApproximatelyEqual<floatT>(this->c.x, op.c.x, 1.e-14) &&
    //isApproximatelyEqual<floatT>(this->c.y, op.c.y, 1.e-14));
  }

  __host__ __device__ friend GPUcomplex operator+(const GPUcomplex &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(left.c.x + right.c.x, left.c.y + right.c.y);
  }

  __host__ __device__ friend GPUcomplex operator+(const GPUcomplex &left,
                                                  const floatT &right) {
    return GPUcomplex(left.c.x + right, left.c.y);
  }

  __host__ __device__ friend GPUcomplex operator+(const floatT &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(left + right.c.x, right.c.y);
  }

  __host__ __device__ friend GPUcomplex operator-(const GPUcomplex &op) {
    return GPUcomplex(-op.c.x, -op.c.y);
  }

  __host__ __device__ friend GPUcomplex operator-(const GPUcomplex &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(left.c.x - right.c.x, left.c.y - right.c.y);
  }

  __host__ __device__ friend GPUcomplex operator-(const GPUcomplex &left,
                                                  const floatT &right) {
    return GPUcomplex(left.c.x - right, left.c.y);
  }

  __host__ __device__ friend GPUcomplex operator-(const floatT &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(left - right.c.x, -right.c.y);
  }

  __host__ __device__ friend GPUcomplex operator*(const GPUcomplex &left,
                                                  const GPUcomplex &right) {
    floatT newReal = left.c.x * right.c.x - left.c.y * right.c.y;
    floatT newImag = left.c.x * right.c.y + left.c.y * right.c.x;
    return GPUcomplex(newReal, newImag);
  }

  __host__ __device__ friend GPUcomplex operator*(const GPUcomplex &left,
                                                  const floatT &right) {
    return GPUcomplex(left.c.x * right, left.c.y * right);
  }

  __host__ __device__ friend GPUcomplex operator*(const floatT &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(left * right.c.x, left * right.c.y);
  }

  __host__ __device__ friend GPUcomplex
  fma(const GPUcomplex &x, const GPUcomplex &y, const GPUcomplex &d) {
    floatT real_res;
    floatT imag_res;

    real_res = (x.c.x * y.c.x) + d.c.x;
    imag_res = (x.c.x * y.c.y) + d.c.y;

    real_res = -(x.c.y * y.c.y) + real_res;
    imag_res = (x.c.y * y.c.x) + imag_res;

    return GPUcomplex(real_res, imag_res);
  }

  __host__ __device__ friend GPUcomplex fma(const floatT x, const GPUcomplex &y,
                                            const GPUcomplex &d) {
    floatT real_res;
    floatT imag_res;

    real_res = (x * y.c.x) + d.c.x;
    imag_res = (x * y.c.y) + d.c.y;

    return GPUcomplex(real_res, imag_res);
  }

  __host__ __device__ void addProduct(const GPUcomplex &x,
                                      const GPUcomplex &y) {
    this->c.x = (x.c.x * y.c.x) + this->c.x;
    this->c.y = (x.c.x * y.c.y) + this->c.y;

    this->c.x = -(x.c.y * y.c.y) + this->c.x;
    this->c.y = (x.c.y * y.c.x) + this->c.y;

    return;
  }

  __host__ __device__ void addProduct(const floatT &x, const GPUcomplex &y) {
    this->c.x = (x * y.c.x) + this->c.x;
    this->c.y = (x * y.c.y) + this->c.y;

    return;
  }

  template <typename T>
  __host__ __device__ friend GPUcomplex operator/(const GPUcomplex &left,
                                                  const T &right) {
    return GPUcomplex(left.c.x / right, left.c.y / right);
  }

  template <typename T>
  __host__ __device__ friend GPUcomplex operator/(const T &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(
        left * right.c.x / (right.c.x * right.c.x + right.c.y * right.c.y),
        -left * right.c.y / (right.c.x * right.c.x + right.c.y * right.c.y));
  }

  __host__ __device__ inline static GPUcomplex invalid();

  // These are needed to make sure that dp_complex may be part in general
  // operators src/math/operators.h
  __host__ __device__ GPUcomplex getAccessor() const { return *this; }

  template <typename Index>
  __host__ __device__ GPUcomplex operator()(const Index) const {
    return *this;
  }
};

#ifdef __GPU_ARCH__
#ifndef USE_HIP
template <> class GPUcomplex<__half> {
public:
  __half2 c;
  __host__ __device__ GPUcomplex(){};

  __host__ __device__ GPUcomplex(const __half &real, const __half &imag) {
    c.x = real;
    c.y = imag;
  };

  __host__ __device__ GPUcomplex(const __half &real) {
    c.x = real;
    c.y = __float2half(0.0f);
  };

  __host__ __device__ GPUcomplex(const __half2 &vec_type) { c = vec_type; };

  __host__ __device__ GPUcomplex &operator=(const GPUcomplex<float> &orig) {
    this->c = __float22half2_rn(orig.c);
    return *this;
  }

  __host__ __device__ GPUcomplex &operator=(const GPUcomplex<double> &orig) {
    __half realpart = __double2half(orig.c.x);
    __half imagpart = __double2half(orig.c.y);
    this->c = __halves2half2(realpart, imagpart);
    return *this;
  }

  __host__ __device__ GPUcomplex &operator=(const GPUcomplex<__half> orig) {
    this->c = static_cast<__half2>(orig.c);
    return *this;
  }
  __host__ __device__ GPUcomplex &operator=(const __half &orig) {
    this->c.x = orig;
    this->c.y = 0.0f;
    return *this;
  }
  __host__ __device__ GPUcomplex &operator+=(const __half &op) {
    this->c.x += op;
    return *this;
  }
  __host__ __device__ GPUcomplex &operator+=(const GPUcomplex &op) {
    this->c += op.c;
    return *this;
  }

  __host__ __device__ GPUcomplex &operator-=(const GPUcomplex &op) {
    this->c -= op.c;
    return *this;
  }

  __host__ __device__ GPUcomplex &operator*=(const GPUcomplex &op) {

    const __half2 a_re = __half2half2(this->c.x);
    __half2 acc = __hfma2(a_re, op.c, __float2half2_rn(0.0));
    const __half2 a_im = __half2half2(this->c.y);
    const __half2 ib = __halves2half2(__hneg(op.c.y), op.c.x);
    acc = __hfma2(a_im, ib, acc);
    //            __half2 result = __hcmadd( this->c , op.c , __float2half2_rn (
    //            0.0 ) );
    this->c = acc;
    return *this;
  }

  __host__ __device__ GPUcomplex &operator*=(const __half &op) {
    __half2 temp = __half2half2(op);
    this->c *= temp;
    return *this;
  }

  __host__ __device__ GPUcomplex &operator/=(const __half &op) {
    __half2 temp = __half2half2(op);
    this->c /= temp;
    return *this;
  }

  __host__ __device__ friend GPUcomplex operator+(const GPUcomplex left,
                                                  const GPUcomplex right) {
    return GPUcomplex(left.c + right.c);
  }

  __host__ __device__ friend GPUcomplex operator+(const GPUcomplex &left,
                                                  const __half &right) {
    return GPUcomplex(left.c.x + right, left.c.y);
  }

  __host__ __device__ friend GPUcomplex operator+(const __half &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(left + right.c.x, right.c.y);
  }

  __host__ __device__ friend GPUcomplex operator-(const GPUcomplex &op) {
    return GPUcomplex(-op.c);
  }

  __host__ __device__ friend GPUcomplex operator-(const GPUcomplex &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(left.c - right.c);
  }

  __host__ __device__ friend GPUcomplex operator-(const GPUcomplex &left,
                                                  const __half &right) {
    return GPUcomplex(left.c.x - right, left.c.y);
  }

  __host__ __device__ friend GPUcomplex operator-(const __half &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(left - right.c.x, -right.c.y);
  }

  __host__ __device__ friend GPUcomplex operator*(const GPUcomplex &left,
                                                  const GPUcomplex &right) {

    const __half2 a_re = __half2half2(left.c.x);
    __half2 acc = __hfma2(a_re, right.c, __float2half2_rn(0.0));
    const __half2 a_im = __half2half2(left.c.y);
    const __half2 ib = __halves2half2(__hneg(right.c.y), right.c.x);
    acc = __hfma2(a_im, ib, acc);
    //            __half2 result = __hcmadd( left.c , right.c , __float2half2_rn
    //            ( 0.0 ) );

    return GPUcomplex(acc);
  }

  __host__ __device__ friend GPUcomplex operator*(const GPUcomplex &left,
                                                  const __half &right) {
    return GPUcomplex(left.c.x * right, left.c.y * right);
  }

  __host__ __device__ friend GPUcomplex operator*(const __half &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(left * right.c.x, left * right.c.y);
  }

  __host__ __device__ friend GPUcomplex
  fma(const GPUcomplex &a, const GPUcomplex &b, const GPUcomplex &d) {
    const __half2 a_re = __half2half2(a.c.x);
    __half2 acc = __hfma2(a_re, b.c, d.c);
    const __half2 a_im = __half2half2(a.c.y);
    const __half2 ib = __halves2half2(__hneg(b.c.y), b.c.x);
    acc = __hfma2(a_im, ib, acc);
    //            return GPUcomplex( __hcmadd( x.c, y.c, d.c ) );
    return GPUcomplex(acc);
  }

  __host__ __device__ friend GPUcomplex fma(const __half x, const GPUcomplex &y,
                                            const GPUcomplex &d) {
    __half2 xh2 = __half2half2(x);
    return GPUcomplex(__hfma2(xh2, y.c, d.c));
  }

  __host__ __device__ void addProduct(const GPUcomplex &a,
                                      const GPUcomplex &b) {
    const __half2 a_re = __half2half2(a.c.x);
    __half2 acc = __hfma2(a_re, b.c, this->c);
    const __half2 a_im = __half2half2(a.c.y);
    const __half2 ib = __halves2half2(__hneg(b.c.y), b.c.x);
    acc = __hfma2(a_im, ib, acc);
    this->c = acc;
    // this->c = __hcmadd( x.c, y.c, this->c );
    return;
  }

  __host__ __device__ void addProduct(const __half &x, const GPUcomplex &y) {
    __half2 xh2 = __half2half2(x);
    this->c = __hfma2(xh2, y.c, this->c);
    return;
  }

  template <typename T>
  __host__ __device__ friend GPUcomplex operator/(const GPUcomplex &left,
                                                  const T &right) {
    return GPUcomplex(left.c.x / right, left.c.y / right);
  }

  __host__ __device__ friend GPUcomplex operator/(const GPUcomplex &left,
                                                  const __half &right) {
    __half2 right2 = __half2half2(right);

    return GPUcomplex(left.c / right2);
  }

  template <typename T>
  __host__ __device__ friend GPUcomplex operator/(const T &left,
                                                  const GPUcomplex &right) {
    return GPUcomplex(
        left * right.c.x / (right.c.x * right.c.x + right.c.y * right.c.y),
        -left * right.c.y / (right.c.x * right.c.x + right.c.y * right.c.y));
  }

  __host__ __device__ inline static GPUcomplex invalid();

  __host__ __device__ GPUcomplex getAccessor() const { return *this; }

  template <typename Index>
  __host__ __device__ GPUcomplex operator()(const Index) const {
    return *this;
  }
};
#endif
#endif

template <class floatT>
__host__ __device__ inline floatT real(const GPUcomplex<floatT> &op) {
  return op.c.x;
}

template <class floatT>
__host__ __device__ inline floatT imag(const GPUcomplex<floatT> &op) {
  return op.c.y;
}

template <class floatT>
__host__ __device__ inline floatT abs(const GPUcomplex<floatT> &op) {
  floatT square = op.c.x * op.c.x + op.c.y * op.c.y;
  return sqrtf(square);
}

template <class floatT>
__host__ __device__ inline floatT abs2(const GPUcomplex<floatT> &op) {
  return op.c.x * op.c.x + op.c.y * op.c.y;
}

template <class floatT>
__host__ __device__ inline GPUcomplex<floatT>
conj(const GPUcomplex<floatT> &op) {
  return GPUcomplex<floatT>(op.c.x, -op.c.y);
}

template <class floatT>
__host__ __device__ inline floatT arg(const GPUcomplex<floatT> &op) {
  return atan2(op.c.y, op.c.x);
}

template <class floatT>
__host__ __device__ inline GPUcomplex<floatT>
cupow(const GPUcomplex<floatT> &base, const floatT &exp) {
  return GPUcomplex<floatT>(pow(abs(base), exp) * cos(arg(base) * exp),
                            pow(abs(base), exp) * sin(arg(base) * exp));
}

template <class floatT>
__host__ __device__ inline GPUcomplex<floatT>
cusqrt(const GPUcomplex<floatT> &base) {
  return GPUcomplex<floatT>(sqrt(abs(base)) * cos(arg(base) * 0.5),
                            sqrt(abs(base)) * sin(arg(base) * 0.5));
}

template <class floatT>
const GPUcomplex<floatT> GPUcomplex_invalid(nanf(" "), nanf(" "));

template <class floatT>
__host__ inline std::ostream &operator<<(std::ostream &s,
                                         GPUcomplex<floatT> z) {
  return s << '(' << real(z) << ',' << imag(z) << ')';
}

template <class floatT, typename floatT2>
__host__ __device__ inline GPUcomplex<floatT, floatT2>
GPUcomplex<floatT, floatT2>::invalid() {
  return GPUcomplex_invalid<floatT>;
}

template <class floatT>
__device__ __host__ inline bool
compareGCOMPLEX(GPUcomplex<floatT> a, GPUcomplex<floatT> b, floatT tol) {
  floatT diffRe = abs(real(a) - real(b));
  floatT diffIm = abs(imag(a) - imag(b));
  if (diffRe > tol || diffIm > tol)
    return false;
  return true;
}

#define GCOMPLEX(floatT) GPUcomplex<floatT>

#endif // SP_COMPLEX_HCU
