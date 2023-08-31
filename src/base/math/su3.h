/*-----------------------------------------------------------------------------------------------
* /
* / su3.h
* /
* / $Id: su3.h,v 1.1 2002/09/10 11:56:54 okacz Exp $
* /
* /---------------------------------------------------------------------------------------------*/
#ifndef _su3_h_
#define _su3_h_

#include "../../define.h"
#include "complex.h"
#include "vect3.h"
#include <random>
#include <limits>
#include "random.h"
#include <float.h>
#include <type_traits>
#include "su2.h"



template<class floatT>
class SU3;

template<class floatT>
__host__ std::ostream &operator<<(std::ostream &, const SU3<floatT> &);

template<class floatT>
__host__ std::istream &operator>>(std::istream &, SU3<floatT> &);

template<class floatT>
__device__ __host__ inline SU3<floatT> operator+(const SU3<floatT> &, const SU3<floatT> &);

template<class floatT>
__device__ __host__ inline SU3<floatT> operator-(const SU3<floatT> &, const SU3<floatT> &);

template<class floatT>
__device__ __host__ inline SU3<floatT> operator*(const COMPLEX(floatT) &, const SU3<floatT> &);

template<class floatT>
__device__ __host__ inline SU3<floatT> operator*(const SU3<floatT> &, const COMPLEX(floatT) &);

template<class floatT>
__device__ __host__ inline SU3<floatT> operator*(const floatT &, const SU3<floatT> &);

template<class floatT>
__device__ __host__ inline SU3<floatT> operator*(const SU3<floatT> &, const floatT &);

template<class floatT>
__device__ __host__ inline SU3<floatT> operator*(const SU3<floatT> &, const SU3<floatT> &);

template<class floatT>
__device__ __host__ inline SU3<floatT> operator/(const SU3<floatT> &, const floatT &);

template<class floatT>
__device__ __host__ floatT tr_d(const SU3<floatT> &);

template<class floatT>
__device__ __host__ floatT tr_i(const SU3<floatT> &);

template<class floatT>
__device__ __host__ floatT tr_d(const SU3<floatT> &, const SU3<floatT> &);

template<class floatT>
__device__ __host__ COMPLEX(floatT) tr_c(const SU3<floatT> &);

template<class floatT>
__device__ __host__ COMPLEX(floatT) tr_c(const SU3<floatT> &, const SU3<floatT> &);

template<class floatT>
__device__ __host__ SU3<floatT> dagger(const SU3<floatT> &);

template<class floatT>
__device__ __host__ COMPLEX(floatT) det(const SU3<floatT> &X);

template<class floatT>
__device__ __host__ floatT realdet(const SU3<floatT> &X);

template<class floatT>
__device__ __host__ floatT infnorm(const SU3<floatT> &X);

template<class floatT>
__device__ __host__ SU3<floatT> su3_exp(SU3<floatT>);

template<class floatT>
__device__ __host__ Vect3<floatT> operator*(const SU3<floatT> &, const Vect3<floatT> &);

template<class floatT>
__device__ __host__ SU3<floatT> tensor_prod(const Vect3<floatT> &, const Vect3<floatT> &);

template<class floatT>
__device__ __host__ inline bool compareSU3(SU3<floatT> a, SU3<floatT> b, floatT tol=1e-13);

template<class floatT>
class SU3 {
private:
    COMPLEX(floatT) _e00, _e01, _e02,
                     _e10, _e11, _e12,
                     _e20, _e21, _e22;

public:

    constexpr SU3(const SU3<floatT>&) = default;
    __host__ __device__ SU3() {};

    __host__ __device__ SU3(const floatT x) {
        _e00 = x;
        _e01 = x;
        _e02 = x;
        _e10 = x;
        _e11 = x;
        _e12 = x;
        _e20 = x;
        _e21 = x;
        _e22 = x;
    };

    __host__ __device__ SU3(COMPLEX(floatT) e00, COMPLEX(floatT) e01, COMPLEX(floatT) e02,
                             COMPLEX(floatT) e10, COMPLEX(floatT) e11, COMPLEX(floatT) e12,
                             COMPLEX(floatT) e20, COMPLEX(floatT) e21, COMPLEX(floatT) e22) :
                                 _e00(e00), _e01(e01), _e02(e02),
                                 _e10(e10), _e11(e11), _e12(e12),
                                 _e20(e20), _e21(e21), _e22(e22) {};



#if (!defined __GPUCC__)
    __host__ friend std::ostream& operator<< <> (std::ostream&, const SU3<floatT> &);
#endif

    __host__ friend std::istream &operator>><>(std::istream &, SU3<floatT> &);


    // matrix operations
    __device__ __host__ friend SU3<floatT> operator+<>(const SU3<floatT> &, const SU3<floatT> &);

    __device__ __host__ friend SU3<floatT> operator-<>(const SU3<floatT> &, const SU3<floatT> &);

    __device__ __host__ friend SU3<floatT> operator*<>(const COMPLEX(floatT) &x, const SU3<floatT> &y);

    __device__ __host__ friend SU3<floatT> operator*<>(const SU3<floatT> &x, const COMPLEX(floatT) &y);

    __device__ __host__ friend SU3<floatT> operator*<>(const floatT &x, const SU3<floatT> &y);

    __device__ __host__ friend SU3<floatT> operator*<>(const SU3<floatT> &x, const floatT &y);

    __device__ __host__ friend SU3<floatT> operator*<>(const SU3<floatT> &, const SU3<floatT> &);

    __device__ __host__ friend SU3<floatT> operator/<>(const SU3<floatT> &x, const floatT &y);

    __device__ __host__ bool operator==(const SU3<floatT> &);

    __device__ __host__ SU3<floatT> &operator=(const SU3<floatT> &);

    __device__ __host__ SU3<floatT> &operator+=(const SU3<floatT> &);

    __device__ __host__ SU3<floatT> &operator-=(const SU3<floatT> &);

    __device__ __host__ SU3<floatT> &operator*=(const floatT &);

    __device__ __host__ SU3<floatT> &operator*=(const COMPLEX(floatT) &);

    __device__ __host__ SU3<floatT> &operator*=(const SU3<floatT> &);

    __device__ __host__ SU3<floatT> &operator/=(const floatT &);

    // cast operations single <-> double precision
    template<class T>
    __device__ __host__ inline operator SU3<T>() const {
        return SU3<T>(COMPLEX(T)(_e00.cREAL, _e00.cIMAG), COMPLEX(T)(_e01.cREAL, _e01.cIMAG),
                       COMPLEX(T)(_e02.cREAL, _e02.cIMAG),
                       COMPLEX(T)(_e10.cREAL, _e10.cIMAG), COMPLEX(T)(_e11.cREAL, _e11.cIMAG),
                       COMPLEX(T)(_e12.cREAL, _e12.cIMAG),
                       COMPLEX(T)(_e20.cREAL, _e20.cIMAG), COMPLEX(T)(_e21.cREAL, _e21.cIMAG),
                       COMPLEX(T)(_e22.cREAL, _e22.cIMAG));
    }


    __device__ __host__ friend Vect3<floatT>
    operator*<>(const SU3<floatT> &, const Vect3<floatT> &);     // SU3 * cvect3 multiplication
    __device__ __host__ friend SU3<floatT>
    tensor_prod<>(const Vect3<floatT> &, const Vect3<floatT> &); // tensor product of two cvect3

    __device__ __host__ friend bool
    compareSU3<>(SU3<floatT> a, SU3<floatT> b, floatT tol);

    __device__ __host__ void random(uint4 *state);                 // set links randomly
    __device__ __host__ void gauss(uint4 *state);                  // set links gauss
    __device__ __host__ void su3unitarize();                       // project to su3 using first two rows of link
    __device__ __host__ int su3unitarize_hits(const int Nhit, floatT tol); // project to su3 by maximizing Re(Tr(guess*(toproj)))
    __device__ __host__ void su3reconstruct12()                    // project to su3 using first two rows of link
    {
        _e20 = COMPLEX(floatT)((_e01.cREAL * _e12.cREAL - _e01.cIMAG * _e12.cIMAG
                                 - (_e02.cREAL * _e11.cREAL - _e02.cIMAG * _e11.cIMAG)),
                                (-(_e01.cIMAG * _e12.cREAL + _e01.cREAL * _e12.cIMAG)
                                 + (_e02.cIMAG * _e11.cREAL + _e02.cREAL * _e11.cIMAG)));

        _e21 = COMPLEX(floatT)((_e02.cREAL * _e10.cREAL - _e02.cIMAG * _e10.cIMAG
                                 - (_e00.cREAL * _e12.cREAL - _e00.cIMAG * _e12.cIMAG)),
                                (-(_e02.cIMAG * _e10.cREAL + _e02.cREAL * _e10.cIMAG)
                                 + (_e00.cIMAG * _e12.cREAL + _e00.cREAL * _e12.cIMAG)));

        _e22 = COMPLEX(floatT)((_e00.cREAL * _e11.cREAL - _e00.cIMAG * _e11.cIMAG
                                 - (_e01.cREAL * _e10.cREAL - _e01.cIMAG * _e10.cIMAG)),
                                (-(_e00.cIMAG * _e11.cREAL + _e00.cREAL * _e11.cIMAG)
                                 + (_e01.cIMAG * _e10.cREAL + _e01.cREAL * _e10.cIMAG)));
    }

    __device__ __host__ void su3reconstruct12Dagger()   // project to su3 using first two rows of link
    {
        _e02 = COMPLEX(floatT)((_e10.cREAL * _e21.cREAL - _e10.cIMAG * _e21.cIMAG
                                 - (_e20.cREAL * _e11.cREAL - _e20.cIMAG * _e11.cIMAG)),
                                (-(_e10.cIMAG * _e21.cREAL + _e10.cREAL * _e21.cIMAG)
                                 + (_e20.cIMAG * _e11.cREAL + _e20.cREAL * _e11.cIMAG)));

        _e12 = COMPLEX(floatT)((_e20.cREAL * _e01.cREAL - _e20.cIMAG * _e01.cIMAG
                                 - (_e00.cREAL * _e21.cREAL - _e00.cIMAG * _e21.cIMAG)),
                                (-(_e20.cIMAG * _e01.cREAL + _e20.cREAL * _e01.cIMAG)
                                 + (_e00.cIMAG * _e21.cREAL + _e00.cREAL * _e21.cIMAG)));

        _e22 = COMPLEX(floatT)((_e00.cREAL * _e11.cREAL - _e00.cIMAG * _e11.cIMAG
                                 - (_e10.cREAL * _e01.cREAL - _e10.cIMAG * _e01.cIMAG)),
                                (-(_e00.cIMAG * _e11.cREAL + _e00.cREAL * _e11.cIMAG)
                                 + (_e10.cIMAG * _e01.cREAL + _e10.cREAL * _e01.cIMAG)));
    }

    __device__ __host__ void u3reconstruct(const COMPLEX(floatT) phase)   // project to u3 using first two rows of link
    {

        _e20 = COMPLEX(floatT)((_e01.cREAL * _e12.cREAL - _e01.cIMAG * _e12.cIMAG
                                 - (_e02.cREAL * _e11.cREAL - _e02.cIMAG * _e11.cIMAG)),
                                (-(_e01.cIMAG * _e12.cREAL + _e01.cREAL * _e12.cIMAG)
                                 + (_e02.cIMAG * _e11.cREAL + _e02.cREAL * _e11.cIMAG)));

        _e21 = COMPLEX(floatT)((_e02.cREAL * _e10.cREAL - _e02.cIMAG * _e10.cIMAG
                                 - (_e00.cREAL * _e12.cREAL - _e00.cIMAG * _e12.cIMAG)),
                                (-(_e02.cIMAG * _e10.cREAL + _e02.cREAL * _e10.cIMAG)
                                 + (_e00.cIMAG * _e12.cREAL + _e00.cREAL * _e12.cIMAG)));

        _e22 = COMPLEX(floatT)((_e00.cREAL * _e11.cREAL - _e00.cIMAG * _e11.cIMAG
                                 - (_e01.cREAL * _e10.cREAL - _e01.cIMAG * _e10.cIMAG)),
                                (-(_e00.cIMAG * _e11.cREAL + _e00.cREAL * _e11.cIMAG)
                                 + (_e01.cIMAG * _e10.cREAL + _e01.cREAL * _e10.cIMAG)));



        //For half prec, this block is faster than the above because COMPLEX(__half)*COMPLEX(__half) uses __half2 intrinsics. The above version would end up using __half intrinsics.
        /*_e20 = _e01*_e12 - _e02*_e11;
        _e12 = _e02*_e10 - _e00*_e12;
        _e22 = _e00*_e11 - _e01*_e10;
        */
        _e20 *= phase;
        _e21 *= phase;
        _e22 *= phase;
    }

    __device__ __host__ void u3reconstructDagger(const COMPLEX(floatT) phase)   // project to u3 using first two rows of link
    {

        _e02 = COMPLEX(floatT)((_e10.cREAL * _e21.cREAL - _e10.cIMAG * _e21.cIMAG
                                 - (_e20.cREAL * _e11.cREAL - _e20.cIMAG * _e11.cIMAG)),
                                (-(_e10.cIMAG * _e21.cREAL + _e10.cREAL * _e21.cIMAG)
                                 + (_e20.cIMAG * _e11.cREAL + _e20.cREAL * _e11.cIMAG)));

        _e12 = COMPLEX(floatT)((_e20.cREAL * _e01.cREAL - _e20.cIMAG * _e01.cIMAG
                                 - (_e00.cREAL * _e21.cREAL - _e00.cIMAG * _e21.cIMAG)),
                                (-(_e20.cIMAG * _e01.cREAL + _e20.cREAL * _e01.cIMAG)
                                 + (_e00.cIMAG * _e21.cREAL + _e00.cREAL * _e21.cIMAG)));

        _e22 = COMPLEX(floatT)((_e00.cREAL * _e11.cREAL - _e00.cIMAG * _e11.cIMAG
                                 - (_e10.cREAL * _e01.cREAL - _e10.cIMAG * _e01.cIMAG)),
                                (-(_e00.cIMAG * _e11.cREAL + _e00.cREAL * _e11.cIMAG)
                                 + (_e10.cIMAG * _e01.cREAL + _e10.cREAL * _e01.cIMAG)));

        //For half prec, this block is faster than the above because COMPLEX(__half)*COMPLEX(__half) uses __half2 intrinsics. The above version would end up using __half intrinsics.
        /*_e02 = _e10*_e21 - _e20*_e11;
        _e12 = _e20*_e01 - _e00*_e21;
        _e22 = _e00*_e11 - _e10*_e01;
        */
        _e02 *= phase;
        _e12 *= phase;
        _e22 *= phase;
    }

    __device__ __host__ void reconstruct14(const COMPLEX(floatT) det)
    {
        floatT amp = pow(abs(det), 1.0/3.0);
        COMPLEX(floatT) phase = det / abs(det);

        _e20 = COMPLEX(floatT)((_e01.cREAL * _e12.cREAL - _e01.cIMAG * _e12.cIMAG
                                 - (_e02.cREAL * _e11.cREAL - _e02.cIMAG * _e11.cIMAG)),
                                (-(_e01.cIMAG * _e12.cREAL + _e01.cREAL * _e12.cIMAG)
                                 + (_e02.cIMAG * _e11.cREAL + _e02.cREAL * _e11.cIMAG)));

        _e21 = COMPLEX(floatT)((_e02.cREAL * _e10.cREAL - _e02.cIMAG * _e10.cIMAG
                                 - (_e00.cREAL * _e12.cREAL - _e00.cIMAG * _e12.cIMAG)),
                                (-(_e02.cIMAG * _e10.cREAL + _e02.cREAL * _e10.cIMAG)
                                 + (_e00.cIMAG * _e12.cREAL + _e00.cREAL * _e12.cIMAG)));

        _e22 = COMPLEX(floatT)((_e00.cREAL * _e11.cREAL - _e00.cIMAG * _e11.cIMAG
                                 - (_e01.cREAL * _e10.cREAL - _e01.cIMAG * _e10.cIMAG)),
                                (-(_e00.cIMAG * _e11.cREAL + _e00.cREAL * _e11.cIMAG)
                                 + (_e01.cIMAG * _e10.cREAL + _e01.cREAL * _e10.cIMAG)));

        _e20 *= phase/amp;
        _e21 *= phase/amp;
        _e22 *= phase/amp;
    }

    __device__ __host__ void reconstruct14Dagger(const COMPLEX(floatT) det)
    {

        floatT amp = pow(abs(det), 1.0/3.0);
        COMPLEX(floatT) phase = det / abs(det);

        _e02 = COMPLEX(floatT)((_e10.cREAL * _e21.cREAL - _e10.cIMAG * _e21.cIMAG
                                 - (_e20.cREAL * _e11.cREAL - _e20.cIMAG * _e11.cIMAG)),
                                (-(_e10.cIMAG * _e21.cREAL + _e10.cREAL * _e21.cIMAG)
                                 + (_e20.cIMAG * _e11.cREAL + _e20.cREAL * _e11.cIMAG)));

        _e12 = COMPLEX(floatT)((_e20.cREAL * _e01.cREAL - _e20.cIMAG * _e01.cIMAG
                                 - (_e00.cREAL * _e21.cREAL - _e00.cIMAG * _e21.cIMAG)),
                                (-(_e20.cIMAG * _e01.cREAL + _e20.cREAL * _e01.cIMAG)
                                 + (_e00.cIMAG * _e21.cREAL + _e00.cREAL * _e21.cIMAG)));

        _e22 = COMPLEX(floatT)((_e00.cREAL * _e11.cREAL - _e00.cIMAG * _e11.cIMAG
                                 - (_e10.cREAL * _e01.cREAL - _e10.cIMAG * _e01.cIMAG)),
                                (-(_e00.cIMAG * _e11.cREAL + _e00.cREAL * _e11.cIMAG)
                                 + (_e10.cIMAG * _e01.cREAL + _e10.cREAL * _e01.cIMAG)));
        _e02 *= phase/amp;
        _e12 *= phase/amp;
        _e22 *= phase/amp;
    }
    __device__ __host__ void TA();                                               // traceless anti-hermitian of link
    __device__ __host__ friend floatT tr_d<>(const SU3<floatT> &);              // real part of trace of link
    __device__ __host__ friend floatT tr_i<>(const SU3<floatT> &);              // imaginary part of trace of link
    __device__ __host__ friend floatT
    tr_d<>(const SU3<floatT> &, const SU3<floatT> &);                          // real part of trace of link*link
    __device__ __host__ friend COMPLEX(floatT) tr_c<>(const SU3<floatT> &);    // trace of link
    __device__ __host__ friend COMPLEX(floatT) tr_c<>(const SU3<floatT> &,
                                                       const SU3<floatT> &);    // trace of link*link
    __device__ __host__ friend SU3<floatT>
    dagger<>(const SU3<floatT> &);                                              // hermitian conjugate
    __device__ __host__ friend SU3<floatT> su3_exp<>(SU3<floatT>);             // exp( link )
    __device__ __host__ friend COMPLEX(floatT) det<>(const SU3<floatT> &);
    __device__ __host__ friend floatT infnorm<>(const SU3<floatT> &);

    // accessors
    __host__ __device__ inline COMPLEX(floatT) getLink00() const;
    __host__ __device__ inline COMPLEX(floatT) getLink01() const;
    __host__ __device__ inline COMPLEX(floatT) getLink02() const;
    __host__ __device__ inline COMPLEX(floatT) getLink10() const;
    __host__ __device__ inline COMPLEX(floatT) getLink11() const;
    __host__ __device__ inline COMPLEX(floatT) getLink12() const;
    __host__ __device__ inline COMPLEX(floatT) getLink20() const;
    __host__ __device__ inline COMPLEX(floatT) getLink21() const;
    __host__ __device__ inline COMPLEX(floatT) getLink22() const;

    // setters
    __host__ __device__ inline void setLink00(COMPLEX(floatT) x);
    __host__ __device__ inline void setLink01(COMPLEX(floatT) x);
    __host__ __device__ inline void setLink02(COMPLEX(floatT) x);
    __host__ __device__ inline void setLink10(COMPLEX(floatT) x);
    __host__ __device__ inline void setLink11(COMPLEX(floatT) x);
    __host__ __device__ inline void setLink12(COMPLEX(floatT) x);
    __host__ __device__ inline void setLink20(COMPLEX(floatT) x);
    __host__ __device__ inline void setLink21(COMPLEX(floatT) x);
    __host__ __device__ inline void setLink22(COMPLEX(floatT) x);

    __host__ __device__ inline COMPLEX(floatT) &operator()(int i, int j) {
        switch (i * 3 + j) {
            case 0:
                return _e00;
            case 1:
                return _e01;
            case 2:
                return _e02;
            case 3:
                return _e10;
            case 4:
                return _e11;
            case 5:
                return _e12;
            case 6:
                return _e20;
            case 7:
                return _e21;
            case 8:
                return _e22;
        }
        _e00 = COMPLEX(floatT)(nan(""), nan(""));
        return _e00;
    }

    __host__ inline const COMPLEX(floatT) &operator()(int i, int j) const {
        switch (i * 3 + j) {
            case 0:
                return _e00;
            case 1:
                return _e01;
            case 2:
                return _e02;
            case 3:
                return _e10;
            case 4:
                return _e11;
            case 5:
                return _e12;
            case 6:
                return _e20;
            case 7:
                return _e21;
            case 8:
                return _e22;
        }
        throw std::runtime_error(stdLogger.fatal("SU3 access to element (", i, ",", j, ") not possible!"));
    }

    __host__ __device__ SU3<floatT> getAccessor() const {
        return *this;
    }

    template<typename Index>
    __host__ __device__ SU3<floatT> operator()(const Index) const {
        return *this;
    }
};

// accessors
template<class floatT>
__host__ __device__ inline COMPLEX(floatT) SU3<floatT>::getLink00() const {
    return _e00;
}

template<class floatT>
__host__ __device__ inline COMPLEX(floatT) SU3<floatT>::getLink01() const {
    return _e01;
}

template<class floatT>
__host__ __device__ inline COMPLEX(floatT) SU3<floatT>::getLink02() const {
    return _e02;
}

template<class floatT>
__host__ __device__ inline COMPLEX(floatT) SU3<floatT>::getLink10() const {
    return _e10;
}

template<class floatT>
__host__ __device__ inline COMPLEX(floatT) SU3<floatT>::getLink11() const {
    return _e11;
}

template<class floatT>
__host__ __device__ inline COMPLEX(floatT) SU3<floatT>::getLink12() const {
    return _e12;
}

template<class floatT>
__host__ __device__ inline COMPLEX(floatT) SU3<floatT>::getLink20() const {
    return _e20;
}

template<class floatT>
__host__ __device__ inline COMPLEX(floatT) SU3<floatT>::getLink21() const {
    return _e21;
}

template<class floatT>
__host__ __device__ inline COMPLEX(floatT) SU3<floatT>::getLink22() const {
    return _e22;
}


// setters
template<class floatT>
__host__ __device__ inline void SU3<floatT>::setLink00(COMPLEX(floatT) x) {
    _e00 = x;
}

template<class floatT>
__host__ __device__ inline void SU3<floatT>::setLink01(COMPLEX(floatT) x) {
    _e01 = x;
}

template<class floatT>
__host__ __device__ inline void SU3<floatT>::setLink02(COMPLEX(floatT) x) {
    _e02 = x;
}

template<class floatT>
__host__ __device__ inline void SU3<floatT>::setLink10(COMPLEX(floatT) x) {
    _e10 = x;
}

template<class floatT>
__host__ __device__ inline void SU3<floatT>::setLink11(COMPLEX(floatT) x) {
    _e11 = x;
}

template<class floatT>
__host__ __device__ inline void SU3<floatT>::setLink12(COMPLEX(floatT) x) {
    _e12 = x;
}

template<class floatT>
__host__ __device__ inline void SU3<floatT>::setLink20(COMPLEX(floatT) x) {
    _e20 = x;
}

template<class floatT>
__host__ __device__ inline void SU3<floatT>::setLink21(COMPLEX(floatT) x) {
    _e21 = x;
}

template<class floatT>
__host__ __device__ inline void SU3<floatT>::setLink22(COMPLEX(floatT) x) {
    _e22 = x;
}

// some constant su3 matrices
template<class floatT>
__device__ __host__ inline SU3<floatT> su3_one() {
    return SU3<floatT>(1, 0, 0,
                        0, 1, 0,
                        0, 0, 1);
}

template <>
__device__ __host__ inline SU3<__half> su3_one() {
    GPUcomplex<__half> g_one(__float2half(1.0));
    GPUcomplex<__half> g_zero(__float2half(0.0));

    return SU3<__half>( g_one, g_zero, g_zero,
                         g_zero, g_one, g_zero,
                         g_zero, g_zero, g_one);
}


template<class floatT>
__device__ __host__ inline SU3<floatT> su3_zero() {
    return SU3<floatT>(0, 0, 0,
                        0, 0, 0,
                        0, 0, 0);
}

template<class floatT>
__device__ __host__ inline SU3<floatT> glambda_1() {
    return SU3<floatT>(0, 1, 0,
                        1, 0, 0,
                        0, 0, 0);
}

template<class floatT>
__device__ __host__ inline SU3<floatT> glambda_2() {
    return SU3<floatT>(0                     , -COMPLEX(floatT)(0, 1), 0,
                        COMPLEX(floatT)(0, 1), 0                      , 0,
                        0                     , 0                      , 0);
}

template<class floatT>
__device__ __host__ inline SU3<floatT> glambda_3() {
    return SU3<floatT>(1, 0 , 0,
                        0, -1, 0,
                        0, 0 , 0);
}

template<class floatT>
__device__ __host__ inline SU3<floatT> glambda_4() {
    return SU3<floatT>(0, 0, 1,
                        0, 0, 0,
                        1, 0, 0);
}

template<class floatT>
__device__ __host__ inline SU3<floatT> glambda_5() {
    return SU3<floatT>(0                     , 0, -COMPLEX(floatT)(0, 1),
                        0                     , 0, 0,
                        COMPLEX(floatT)(0, 1), 0, 0);
}

template<class floatT>
__device__ __host__ inline SU3<floatT> glambda_6() {
    return SU3<floatT>(0, 0, 0,
                        0, 0, 1,
                        0, 1, 0);
}

template<class floatT>
__device__ __host__ inline SU3<floatT> glambda_7() {
    return SU3<floatT>(0, 0                     , 0,
                        0, 0                     , -COMPLEX(floatT)(0, 1),
                        0, COMPLEX(floatT)(0, 1), 0);
}

template<class floatT>
__device__ __host__ inline SU3<floatT> glambda_8() {
    return SU3<floatT>(1 / sqrt(3), 0          , 0,
                        0          , 1 / sqrt(3), 0,
                        0          , 0          , -2 / sqrt(3));
}




// matrix operations
template<class floatT>
__device__ __host__ SU3<floatT> operator+(const SU3<floatT> &x, const SU3<floatT> &y) {
    return SU3<floatT>(
            x._e00 + y._e00, x._e01 + y._e01, x._e02 + y._e02,
            x._e10 + y._e10, x._e11 + y._e11, x._e12 + y._e12,
            x._e20 + y._e20, x._e21 + y._e21, x._e22 + y._e22);
}

template<class floatT>
__device__ __host__ SU3<floatT> operator-(const SU3<floatT> &x, const SU3<floatT> &y) {
    return SU3<floatT>(
            x._e00 - y._e00, x._e01 - y._e01, x._e02 - y._e02,
            x._e10 - y._e10, x._e11 - y._e11, x._e12 - y._e12,
            x._e20 - y._e20, x._e21 - y._e21, x._e22 - y._e22);
}


template<class floatT>
__device__ __host__ SU3<floatT> operator*(const COMPLEX(floatT) &x, const SU3<floatT> &y) {
    return SU3<floatT>(
            x * y._e00, x * y._e01, x * y._e02,
            x * y._e10, x * y._e11, x * y._e12,
            x * y._e20, x * y._e21, x * y._e22);
}

template<class floatT>
__device__ __host__ SU3<floatT> operator*(const SU3<floatT> &x, const COMPLEX(floatT) &y) {
    return SU3<floatT>(
            x._e00 * y, x._e01 * y, x._e02 * y,
            x._e10 * y, x._e11 * y, x._e12 * y,
            x._e20 * y, x._e21 * y, x._e22 * y);
}

template<class floatT>
__device__ __host__ SU3<floatT> operator*(const floatT &x, const SU3<floatT> &y) {
    return SU3<floatT>(
            x * y._e00, x * y._e01, x * y._e02,
            x * y._e10, x * y._e11, x * y._e12,
            x * y._e20, x * y._e21, x * y._e22);
}

template<class floatT>
__device__ __host__ SU3<floatT> operator*(const SU3<floatT> &x, const floatT &y) {
    return SU3<floatT>(
            x._e00 * y, x._e01 * y, x._e02 * y,
            x._e10 * y, x._e11 * y, x._e12 * y,
            x._e20 * y, x._e21 * y, x._e22 * y);
}

template<class floatT>
__device__ __host__ SU3<floatT> operator/(const SU3<floatT> &x, const floatT &y) {
    return SU3<floatT>(
            x._e00 / y, x._e01 / y, x._e02 / y,
            x._e10 / y, x._e11 / y, x._e12 / y,
            x._e20 / y, x._e21 / y, x._e22 / y);
}


template<class floatT>
__device__ __host__ SU3<floatT> operator*(const SU3<floatT> &x, const SU3<floatT> &y) {
    COMPLEX(floatT) tmp00, tmp01, tmp02,
            tmp10, tmp11, tmp12,
            tmp20, tmp21, tmp22;

    tmp00 = fma(x._e00, y._e00, fma(x._e01, (y._e10), x._e02 * (y._e20)));
    tmp01 = fma(x._e00, y._e01, fma(x._e01, (y._e11), x._e02 * (y._e21)));
    tmp02 = fma(x._e00, y._e02, fma(x._e01, (y._e12), x._e02 * (y._e22)));
    tmp10 = fma(x._e10, y._e00, fma(x._e11, (y._e10), x._e12 * (y._e20)));
    tmp11 = fma(x._e10, y._e01, fma(x._e11, (y._e11), x._e12 * (y._e21)));
    tmp12 = fma(x._e10, y._e02, fma(x._e11, (y._e12), x._e12 * (y._e22)));
    tmp20 = fma(x._e20, y._e00, fma(x._e21, (y._e10), x._e22 * (y._e20)));
    tmp21 = fma(x._e20, y._e01, fma(x._e21, (y._e11), x._e22 * (y._e21)));
    tmp22 = fma(x._e20, y._e02, fma(x._e21, (y._e12), x._e22 * (y._e22)));

    return SU3<floatT>(tmp00, tmp01, tmp02,
                        tmp10, tmp11, tmp12,
                        tmp20, tmp21, tmp22);
}


// su3 * cvect3 multiplication
template<class floatT>
__device__ __host__ Vect3<floatT> operator*(const SU3<floatT> &x, const Vect3<floatT> &y) {
    COMPLEX(floatT) tmp0, tmp1, tmp2;

    tmp0 = x._e00 * y.template getElement<0>() + x._e01 * y.template getElement<1>() + x._e02 * y.template getElement<2>();
    tmp1 = x._e10 * y.template getElement<0>() + x._e11 * y.template getElement<1>() + x._e12 * y.template getElement<2>();
    tmp2 = x._e20 * y.template getElement<0>() + x._e21 * y.template getElement<1>() + x._e22 * y.template getElement<2>();

    return Vect3<floatT>(tmp0, tmp1, tmp2);
}


template<class floatT>
__device__ __host__ inline SU3<floatT> &SU3<floatT>::operator=(const SU3<floatT> &y) {
    _e00 = y._e00;
    _e01 = y._e01;
    _e02 = y._e02;
    _e10 = y._e10;
    _e11 = y._e11;
    _e12 = y._e12;
    _e20 = y._e20;
    _e21 = y._e21;
    _e22 = y._e22;
    return *this;
}

template<class floatT>
__device__ __host__ SU3<floatT> &SU3<floatT>::operator+=(const SU3<floatT> &y) {
    _e00 += y._e00;
    _e01 += y._e01;
    _e02 += y._e02;
    _e10 += y._e10;
    _e11 += y._e11;
    _e12 += y._e12;
    _e20 += y._e20;
    _e21 += y._e21;
    _e22 += y._e22;
    return *this;
}

template<class floatT>
__device__ __host__ SU3<floatT> &SU3<floatT>::operator-=(const SU3<floatT> &y) {
    _e00 -= y._e00;
    _e01 -= y._e01;
    _e02 -= y._e02;
    _e10 -= y._e10;
    _e11 -= y._e11;
    _e12 -= y._e12;
    _e20 -= y._e20;
    _e21 -= y._e21;
    _e22 -= y._e22;
    return *this;
}

template<class floatT>
__device__ __host__ SU3<floatT> &SU3<floatT>::operator*=(const floatT &y) {
    *this = *this * y;
    return *this;
}

template<class floatT>
__device__ __host__ SU3<floatT> &SU3<floatT>::operator*=(const COMPLEX(floatT) &y) {
    _e00 *= y;
    _e01 *= y;
    _e02 *= y;
    _e10 *= y;
    _e11 *= y;
    _e12 *= y;
    _e20 *= y;
    _e21 *= y;
    _e22 *= y;
    return *this;
}

template<class floatT>
__device__ __host__ SU3<floatT> &SU3<floatT>::operator*=(const SU3<floatT> &y) {
    *this = *this * y;
    return *this;
}

template<class floatT>
__device__ __host__ SU3<floatT> &SU3<floatT>::operator/=(const floatT &y) {
    *this = *this / y;
    return *this;
}

/// This method is a straightforward way to compare two SU(3) matrices. You may decide that you want to control the
/// tolerance for comparison. In that case please look to the compareSU3 method. In case you are comparing with the
/// zero matrix, you should use compareSU3, as the present method seems not to work for that case.
template<class floatT>
__device__ __host__ bool SU3<floatT>::operator==(const SU3<floatT> &y) {
    if (_e00 == y._e00 &&
        _e01 == y._e01 &&
        _e02 == y._e02 &&
        _e10 == y._e10 &&
        _e11 == y._e11 &&
        _e12 == y._e12 &&
        _e20 == y._e20 &&
        _e21 == y._e21 &&
        _e22 == y._e22) {
        return true;
    } else return false;
}

template<class floatT>
__host__ inline std::ostream &operator<<(std::ostream &s, const SU3<floatT> &x) {
return s << "\n" << x.getLink00() << x.getLink01() << x.getLink02() << "\n"
                 << x.getLink10() << x.getLink11() << x.getLink12() << "\n"
                 << x.getLink20() << x.getLink21() << x.getLink22() << "\n";
}

/// TODO: This presumably doesn't work
template<class floatT>
__host__ inline std::istream &operator>>(std::istream &s, SU3<floatT> &x) {
    return s >> x._e00.cREAL >> x._e00.cIMAG >> x._e01.cREAL >> x._e01.cIMAG >> x._e02.cREAL >> x._e02.cIMAG
             >> x._e10.cREAL >> x._e10.cIMAG >> x._e11.cREAL >> x._e11.cIMAG >> x._e12.cREAL >> x._e12.cIMAG
             >> x._e20.cREAL >> x._e20.cIMAG >> x._e21.cREAL >> x._e21.cIMAG >> x._e22.cREAL >> x._e22.cIMAG;
}


template<class floatT>
__device__ __host__ void SU3<floatT>::random(uint4 *state) {

    COMPLEX(floatT)
            rnd;

    rnd = COMPLEX(floatT)(get_rand<floatT>(state), get_rand<floatT>(state));
    _e00 = (floatT) 1.0 - (floatT) 2.0 * rnd;
    rnd = COMPLEX(floatT)(get_rand<floatT>(state), get_rand<floatT>(state));
    _e01 = (floatT) 1.0 - (floatT) 2.0 * rnd;
    rnd = COMPLEX(floatT)(get_rand<floatT>(state), get_rand<floatT>(state));
    _e02 = (floatT) 1.0 - (floatT) 2.0 * rnd;
    rnd = COMPLEX(floatT)(get_rand<floatT>(state), get_rand<floatT>(state));
    _e10 = (floatT) 1.0 - (floatT) 2.0 * rnd;
    rnd = COMPLEX(floatT)(get_rand<floatT>(state), get_rand<floatT>(state));
    _e11 = (floatT) 1.0 - (floatT) 2.0 * rnd;
    rnd = COMPLEX(floatT)(get_rand<floatT>(state), get_rand<floatT>(state));
    _e12 = (floatT) 1.0 - (floatT) 2.0 * rnd;
    rnd = COMPLEX(floatT)(get_rand<floatT>(state), get_rand<floatT>(state));
    _e20 = (floatT) 1.0 - (floatT) 2.0 * rnd;
    rnd = COMPLEX(floatT)(get_rand<floatT>(state), get_rand<floatT>(state));
    _e21 = (floatT) 1.0 - (floatT) 2.0 * rnd;
    rnd = COMPLEX(floatT)(get_rand<floatT>(state), get_rand<floatT>(state));
    _e22 = (floatT) 1.0 - (floatT) 2.0 * rnd;

    (*this).su3unitarize();
}


template<class floatT>
__device__ __host__ void SU3<floatT>::gauss(uint4 *state) {
    if constexpr (!std::is_same<floatT,__half>::value) {
            floatT rand1[4], rand2[4], phi[4], radius[4], temp1[4], temp2[4];

            for (int i = 0; i < 4; ++i) {
                rand1[i] = get_rand<floatT>(state);
                rand2[i] = get_rand<floatT>(state);
            }

            for (int i = 0; i < 4; ++i) {
                phi[i] = 2.0 * M_PI * rand1[i];
                rand2[i] = rand2[i]+(1.0 - rand2[i])*minVal<floatT>(); // exclude 0 from random numbers!
                radius[i] = sqrt(-log(rand2[i]));

                temp1[i] = radius[i] * cos(phi[i]);
                temp2[i] = radius[i] * sin(phi[i]);
            }

            _e00 = COMPLEX(floatT)(temp1[2] + 1. / sqrt(3.0) * temp2[3], 0.0);
            _e01 = COMPLEX(floatT)(temp1[0], -temp1[1]);
            _e02 = COMPLEX(floatT)(temp1[3], -temp2[0]);
            _e10 = COMPLEX(floatT)(temp1[0], temp1[1]);
            _e11 = COMPLEX(floatT)(-temp1[2] + 1. / sqrt(3.0) * temp2[3], 0.0);
            _e12 = COMPLEX(floatT)(temp2[1], -temp2[2]);
            _e20 = COMPLEX(floatT)(temp1[3], temp2[0]);
            _e21 = COMPLEX(floatT)(temp2[1], temp2[2]);
            _e22 = COMPLEX(floatT)(-2. / sqrt(3.0) * temp2[3], 0.0);
        }
    else {
#ifdef __GPU_ARCH__
        float rand1[4], rand2[4], phi[4], radius[4], temp1[4], temp2[4];

    for (int i = 0; i < 4; ++i) {
        rand1[i] = get_rand<float>(state);
        rand2[i] = get_rand<float>(state);
    }

    for (int i = 0; i < 4; ++i) {
        phi[i] = 2.0 * M_PI * rand1[i];
        rand2[i] = rand2[i]+(1.0 - rand2[i])*FLT_MIN; // exclude 0 from random numbers!
        radius[i] = sqrt(-log(rand2[i]));

        temp1[i] = radius[i] * cos(phi[i]);
        temp2[i] = radius[i] * sin(phi[i]);
    }

    _e00 = COMPLEX(__half)(__float2half(temp1[2] + 1. / sqrt(3.0) * temp2[3]), __float2half( 0.0));
    _e01 = COMPLEX(__half)(__float2half(temp1[0]), __float2half( -temp1[1]));
    _e02 = COMPLEX(__half)(__float2half(temp1[3]), __float2half( -temp2[0]));
    _e10 = COMPLEX(__half)(__float2half(temp1[0]), __float2half( temp1[1]));
    _e11 = COMPLEX(__half)(__float2half(-temp1[2] + 1. / sqrt(3.0) * temp2[3]), __float2half( 0.0));
    _e12 = COMPLEX(__half)(__float2half(temp2[1]), __float2half( -temp2[2]));
    _e20 = COMPLEX(__half)(__float2half(temp1[3]), __float2half( temp2[0]));
    _e21 = COMPLEX(__half)(__float2half(temp2[1]), __float2half( temp2[2]));
    _e22 = COMPLEX(__half)(__float2half(-2. / sqrt(3.0) * temp2[3]), __float2half( 0.0));
#endif
    }
}



// project to su3 by maximizing Re(Tr(guess*(toproj)))
// ported from Milc by Dan Hoying, 2022
template<class floatT>
__device__ __host__ int su3unitarize_hits(
	SU3<floatT> *w,         /* input initial guess. output resulting SU(3) matrix */
	SU3<floatT> *q,         /* 3 x 3 complex matrix to be projected */
   	int Nhit,              /* number of SU(2) hits. 0 for no projection */
   	floatT tol              /* tolerance for SU(3) projection.
			     If nonzero, treat Nhit as a maximum
			     number of hits.  If zero, treat Nhit
			     as a prescribed number of hits. */
		) {

   int index1, ina, inb,ii;
   floatT v0,v1,v2,v3, vsq;
   floatT z;
   SU3<floatT> action(0);
   const int Nc = 3;
   double conver, old_tr = 0, new_tr;

   if(tol > 0)
     old_tr = tr_d(*w,*q)/3.0;
   conver = 1.0;
   assert(!std::isnan(old_tr));

   /* Do SU(2) hits */
   for(index1=0;index1<Nhit && conver > tol; index1++)
   {
      /*  pick out an SU(2) subgroup */
      ina =  index1    % Nc;
      inb = (index1+1) % Nc;
      if(ina > inb){ ii=ina; ina=inb; inb=ii; }

      //mult_su3_na( w, q, &action );
      assert(!std::isnan(real(q->operator()(ina,ina))));
      action = (*w) * dagger(*q);

      /* decompose the action into SU(2) subgroups using
         Pauli matrix expansion */
      /* The SU(2) hit matrix is represented as v0 + i *
         Sum j (sigma j * vj)*/
      v0 =  real(action(ina,ina)) + real(action(inb,inb));
      assert(!std::isnan(v0));
      v3 =  imag(action(ina,ina)) - imag(action(inb,inb));
      v1 =  imag(action(ina,inb)) + imag(action(inb,ina));
      v2 =  real(action(ina,inb)) - real(action(inb,ina));

      /* Normalize v */
      vsq = v0*v0 + v1*v1 + v2*v2 + v3*v3;
      z = sqrt((double)vsq );
      if(z == 0.){z = 1.;v0 = 1.;}
      else {v0 = v0/z; v1 = v1/z; v2 = v2/z; v3 = v3/z;}

      COMPLEX(floatT) x00, x01;
      SU2<floatT> zz;
      x00=COMPLEX(floatT)(v0,-v3);
      x01=COMPLEX(floatT)(-v2,-v1);
      zz=SU2<floatT>(x00,x01);
      if( ina==0 && inb ==1){
          *w=sub12(zz,*w);
      }
      else if( ina==0 && inb ==2){
          *w=sub13(zz,*w);
      }
      else if( ina==1 && inb ==2){
          *w=sub23(zz,*w);
      }



      /* convergence measure every third hit */
      if(tol>0 && (index1 % 3) == 2){
        new_tr = tr_d(*w,*q)/3.0;
	conver = (new_tr-old_tr)/old_tr; /* trace always increases */
	old_tr = new_tr;
      }

   } /* hits */

   int status = 0;
   if( Nhit > 0 && tol > 0 && conver > tol )
     status = 1;
   return status;

}


// project to su3 using first two rows of link
template<class floatT>
__device__ __host__ void SU3<floatT>::su3unitarize() {
    if constexpr (!std::is_same<floatT,__half>::value) {
    double quadnorm, invnorm;
    double Cre, Cim;

    quadnorm = _e00.cREAL * _e00.cREAL + _e00.cIMAG * _e00.cIMAG
               + _e01.cREAL * _e01.cREAL + _e01.cIMAG * _e01.cIMAG
               + _e02.cREAL * _e02.cREAL + _e02.cIMAG * _e02.cIMAG;

    invnorm = 1.0 / sqrt(quadnorm);

    _e00 *= invnorm;
    _e01 *= invnorm;
    _e02 *= invnorm;


// 2.Zeile ist die ON-Projektion auf die 1.Zeile

    Cre = _e10.cREAL * _e00.cREAL + _e10.cIMAG * _e00.cIMAG
          + _e11.cREAL * _e01.cREAL + _e11.cIMAG * _e01.cIMAG
          + _e12.cREAL * _e02.cREAL + _e12.cIMAG * _e02.cIMAG;

    Cim = _e10.cIMAG * _e00.cREAL - _e10.cREAL * _e00.cIMAG
          + _e11.cIMAG * _e01.cREAL - _e11.cREAL * _e01.cIMAG
          + _e12.cIMAG * _e02.cREAL - _e12.cREAL * _e02.cIMAG;

    _e10 -= COMPLEX(floatT)((Cre * _e00.cREAL - Cim * _e00.cIMAG), (Cre * _e00.cIMAG + Cim * _e00.cREAL));
    _e11 -= COMPLEX(floatT)((Cre * _e01.cREAL - Cim * _e01.cIMAG), (Cre * _e01.cIMAG + Cim * _e01.cREAL));
    _e12 -= COMPLEX(floatT)((Cre * _e02.cREAL - Cim * _e02.cIMAG), (Cre * _e02.cIMAG + Cim * _e02.cREAL));


// Normierung der 2.Zeile

    quadnorm = _e10.cREAL * _e10.cREAL + _e10.cIMAG * _e10.cIMAG
               + _e11.cREAL * _e11.cREAL + _e11.cIMAG * _e11.cIMAG
               + _e12.cREAL * _e12.cREAL + _e12.cIMAG * _e12.cIMAG;

    invnorm = 1.0 / sqrt(quadnorm);

    _e10 *= invnorm;
    _e11 *= invnorm;
    _e12 *= invnorm;


// 3.Zeile ist das Vektorprodukt von 1* und 2*

    _e20 = COMPLEX(floatT)((_e01.cREAL * _e12.cREAL - _e01.cIMAG * _e12.cIMAG
                             - (_e02.cREAL * _e11.cREAL - _e02.cIMAG * _e11.cIMAG)),
                            (-(_e01.cIMAG * _e12.cREAL + _e01.cREAL * _e12.cIMAG)
                             + (_e02.cIMAG * _e11.cREAL + _e02.cREAL * _e11.cIMAG)));

    _e21 = COMPLEX(floatT)((_e02.cREAL * _e10.cREAL - _e02.cIMAG * _e10.cIMAG
                             - (_e00.cREAL * _e12.cREAL - _e00.cIMAG * _e12.cIMAG)),
                            (-(_e02.cIMAG * _e10.cREAL + _e02.cREAL * _e10.cIMAG)
                             + (_e00.cIMAG * _e12.cREAL + _e00.cREAL * _e12.cIMAG)));

    _e22 = COMPLEX(floatT)((_e00.cREAL * _e11.cREAL - _e00.cIMAG * _e11.cIMAG
                             - (_e01.cREAL * _e10.cREAL - _e01.cIMAG * _e10.cIMAG)),
                            (-(_e00.cIMAG * _e11.cREAL + _e00.cREAL * _e11.cIMAG)
                             + (_e01.cIMAG * _e10.cREAL + _e01.cREAL * _e10.cIMAG)));
        }
    else {
 #ifdef __GPU_ARCH__
    double quadnorm, invnorm;
    double Cre, Cim;

    quadnorm = __half2float( _e00.cREAL * _e00.cREAL + _e00.cIMAG * _e00.cIMAG
                             + _e01.cREAL * _e01.cREAL + _e01.cIMAG * _e01.cIMAG
                             + _e02.cREAL * _e02.cREAL + _e02.cIMAG * _e02.cIMAG);

    invnorm = 1.0 / sqrt(quadnorm);

    _e00 *= __float2half(invnorm);
    _e01 *= __float2half(invnorm);
    _e02 *= __float2half(invnorm);


// 2.Zeile ist die ON-Projektion auf die 1.Zeile

    Cre = __half2float(_e10.cREAL * _e00.cREAL + _e10.cIMAG * _e00.cIMAG
                       + _e11.cREAL * _e01.cREAL + _e11.cIMAG * _e01.cIMAG
                       + _e12.cREAL * _e02.cREAL + _e12.cIMAG * _e02.cIMAG);

    Cim = __half2float(_e10.cIMAG * _e00.cREAL - _e10.cREAL * _e00.cIMAG
                       + _e11.cIMAG * _e01.cREAL - _e11.cREAL * _e01.cIMAG
                       + _e12.cIMAG * _e02.cREAL - _e12.cREAL * _e02.cIMAG);

    _e10 -= COMPLEX(__half)((Cre * __half2float(_e00.cREAL) - Cim * __half2float(_e00.cIMAG)), (Cre * __half2float(_e00.cIMAG) + Cim * __half2float(_e00.cREAL)));
    _e11 -= COMPLEX(__half)((Cre * __half2float(_e01.cREAL) - Cim * __half2float(_e01.cIMAG)), (Cre * __half2float(_e01.cIMAG) + Cim * __half2float(_e01.cREAL)));
    _e12 -= COMPLEX(__half)((Cre * __half2float(_e02.cREAL) - Cim * __half2float(_e02.cIMAG)), (Cre * __half2float(_e02.cIMAG) + Cim * __half2float(_e02.cREAL)));


// Normierung der 2.Zeile

    quadnorm =  __half2float(_e10.cREAL * _e10.cREAL + _e10.cIMAG * _e10.cIMAG
                             + _e11.cREAL * _e11.cREAL + _e11.cIMAG * _e11.cIMAG
                             + _e12.cREAL * _e12.cREAL + _e12.cIMAG * _e12.cIMAG);

    invnorm = 1.0 / sqrt(quadnorm);

    _e10 *=  __float2half(invnorm);
    _e11 *= __float2half(invnorm);
    _e12 *= __float2half(invnorm);


// 3.Zeile ist das Vektorprodukt von 1* und 2*

    _e20 = COMPLEX(__half)((_e01.cREAL * _e12.cREAL - _e01.cIMAG * _e12.cIMAG
                             - (_e02.cREAL * _e11.cREAL - _e02.cIMAG * _e11.cIMAG)),
                            (-(_e01.cIMAG * _e12.cREAL + _e01.cREAL * _e12.cIMAG)
                             + (_e02.cIMAG * _e11.cREAL + _e02.cREAL * _e11.cIMAG)));

    _e21 = COMPLEX(__half)((_e02.cREAL * _e10.cREAL - _e02.cIMAG * _e10.cIMAG
                             - (_e00.cREAL * _e12.cREAL - _e00.cIMAG * _e12.cIMAG)),
                            (-(_e02.cIMAG * _e10.cREAL + _e02.cREAL * _e10.cIMAG)
                             + (_e00.cIMAG * _e12.cREAL + _e00.cREAL * _e12.cIMAG)));

    _e22 = COMPLEX(__half)((_e00.cREAL * _e11.cREAL - _e00.cIMAG * _e11.cIMAG
                             - (_e01.cREAL * _e10.cREAL - _e01.cIMAG * _e10.cIMAG)),
                            (-(_e00.cIMAG * _e11.cREAL + _e00.cREAL * _e11.cIMAG)
                             + (_e01.cIMAG * _e10.cREAL + _e01.cREAL * _e10.cIMAG)));
    #endif
    }

}

template<class floatT>
__device__ __host__ COMPLEX(floatT) det(const SU3<floatT> &x) {
    COMPLEX(floatT)
            res;

    res = x._e00 * (x._e11 * x._e22 - x._e12 * x._e21)
          + x._e01 * (x._e12 * x._e20 - x._e10 * x._e22)
          + x._e02 * (x._e10 * x._e21 - x._e11 * x._e20);

    return (res);
}

template<class floatT>
__device__ __host__ floatT realdet(const SU3<floatT> &x) {
    return det(x).cREAL;
}

template<class floatT>
__device__ __host__ floatT infnorm(const SU3<floatT> &x) {
    floatT res = x._e00.cREAL * x._e00.cREAL;
    res = x._e00.cIMAG * x._e00.cIMAG + res;
    res = x._e01.cREAL * x._e01.cREAL + res;
    res = x._e01.cIMAG * x._e01.cIMAG + res;
    res = x._e02.cREAL * x._e02.cREAL + res;
    res = x._e02.cIMAG * x._e02.cIMAG + res;

    floatT tmp = x._e10.cREAL * x._e10.cREAL;
    tmp = x._e10.cIMAG * x._e10.cIMAG + tmp;
    tmp = x._e11.cREAL * x._e11.cREAL + tmp;
    tmp = x._e11.cIMAG * x._e11.cIMAG + tmp;
    tmp = x._e12.cREAL * x._e12.cREAL + tmp;
    tmp = x._e12.cIMAG * x._e12.cIMAG + tmp;

    if (tmp > res) {
        res = tmp;
    }

    tmp = x._e20.cREAL * x._e20.cREAL;
    tmp = x._e20.cIMAG * x._e20.cIMAG + tmp;
    tmp = x._e21.cREAL * x._e21.cREAL + tmp;
    tmp = x._e21.cIMAG * x._e21.cIMAG + tmp;
    tmp = x._e22.cREAL * x._e22.cREAL + tmp;
    tmp = x._e22.cIMAG * x._e22.cIMAG + tmp;
    if (tmp > res) {
        res = tmp;
    }
    return sqrt(res);
}

// traceless anti-hermitian of link
template<class floatT>
__device__ __host__ void SU3<floatT>::TA() {
    SU3 <floatT> tmp;

    tmp._e00 = COMPLEX(floatT)(0, 0.6666666666666666 * _e00.cIMAG - 0.3333333333333333 * (_e11.cIMAG + _e22.cIMAG));
    tmp._e01 = 0.5 * (_e01 - conj(_e10));
    tmp._e02 = 0.5 * (_e02 - conj(_e20));
    tmp._e10 = 0.5 * (_e10 - conj(_e01));
    tmp._e11 = COMPLEX(floatT)(0, 0.6666666666666666 * _e11.cIMAG - 0.3333333333333333 * (_e00.cIMAG + _e22.cIMAG));
    tmp._e12 = 0.5 * (_e12 - conj(_e21));
    tmp._e20 = 0.5 * (_e20 - conj(_e02));
    tmp._e21 = 0.5 * (_e21 - conj(_e12));
    tmp._e22 = COMPLEX(floatT)(0, 0.6666666666666666 * _e22.cIMAG - 0.3333333333333333 * (_e00.cIMAG + _e11.cIMAG));

    (*this) = tmp;
}

// real part of trace of link
template<class floatT>
__device__ __host__ floatT tr_d(const SU3<floatT> &x) {
    return floatT(x._e00.cREAL + x._e11.cREAL + x._e22.cREAL);
}

// imaginary part of trace of link
template<class floatT>
__device__ __host__ floatT tr_i(const SU3<floatT> &x) {
    return floatT(x._e00.cIMAG + x._e11.cIMAG + x._e22.cIMAG);
}

// real part of trace of link*link
template<class floatT>
__device__ __host__ floatT tr_d(const SU3<floatT> &x, const SU3<floatT> &y) {
    floatT res;
    res = (x._e00 * y._e00).cREAL + (x._e01 * y._e10).cREAL + (x._e02 * y._e20).cREAL
          + (x._e10 * y._e01).cREAL + (x._e11 * y._e11).cREAL + (x._e12 * y._e21).cREAL
          + (x._e20 * y._e02).cREAL + (x._e21 * y._e12).cREAL + (x._e22 * y._e22).cREAL;

    return (res);
}

// trace of link
template<class floatT>
__device__ __host__ COMPLEX(floatT) tr_c(const SU3<floatT> &x) {
    return COMPLEX(floatT)(x._e00 + x._e11 + x._e22);
}

// trace of link*link
template<class floatT>
__device__ __host__ COMPLEX(floatT) tr_c(const SU3<floatT> &x, const SU3<floatT> &y) {
    COMPLEX(floatT)
            res;

    res = x._e00 * y._e00 + x._e01 * y._e10 + x._e02 * y._e20
          + x._e10 * y._e01 + x._e11 * y._e11 + x._e12 * y._e21
          + x._e20 * y._e02 + x._e21 * y._e12 + x._e22 * y._e22;

    return (res);
}

// hermitian conjugate
template<class floatT>
__device__ __host__ SU3<floatT> dagger(const SU3<floatT> &x) {
    SU3 <floatT> tmp;

    tmp._e00 = conj(x._e00);
    tmp._e01 = conj(x._e10);
    tmp._e02 = conj(x._e20);
    tmp._e10 = conj(x._e01);
    tmp._e11 = conj(x._e11);
    tmp._e12 = conj(x._e21);
    tmp._e20 = conj(x._e02);
    tmp._e21 = conj(x._e12);
    tmp._e22 = conj(x._e22);

    return (tmp);
}

// exp( link )
template<class floatT>
__device__ __host__ SU3<floatT> su3_exp(SU3<floatT> u) {
    SU3 <floatT> res;

    res = su3_one<floatT>()
          + u * (su3_one<floatT>()
                 + u * ((floatT) 0.5 * su3_one<floatT>()
                        + u * ((floatT) 0.1666666666666666 * su3_one<floatT>()
                               + u * ((floatT) 0.0416666666666666 * su3_one<floatT>()
                                      + u * ((floatT) 0.0083333333333333 * su3_one<floatT>()
                                             + (floatT) 0.0013888888888888 * u)))));
    return (res);
}

// tensor product of two cvect3
template<class floatT>
__device__ __host__ SU3<floatT> tensor_prod(const Vect3<floatT> &x, const Vect3<floatT> &y) {
    SU3 <floatT> res;

    res._e00 = x.template getElement<0>() * y.template getElement<0>();
    res._e01 = x.template getElement<0>() * y.template getElement<1>();
    res._e02 = x.template getElement<0>() * y.template getElement<2>();
    res._e10 = x.template getElement<1>() * y.template getElement<0>();
    res._e11 = x.template getElement<1>() * y.template getElement<1>();
    res._e12 = x.template getElement<1>() * y.template getElement<2>();
    res._e20 = x.template getElement<2>() * y.template getElement<0>();
    res._e21 = x.template getElement<2>() * y.template getElement<1>();
    res._e22 = x.template getElement<2>() * y.template getElement<2>();

    return (res);
}

template<class floatT>
__device__ __host__ inline bool compareSU3(SU3<floatT> a, SU3<floatT> b, floatT tol) {

    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
        COMPLEX(floatT) diff = a(i, j) - b(i, j);
        if (fabs(diff.cREAL) > tol) return false;
    }
    return true;
}

#endif
