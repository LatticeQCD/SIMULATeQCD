/*-----------------------------------------------------------------------------------------------
* /
* / gsu3.h
* /
* / $Id: gsu3.h,v 1.1 2002/09/10 11:56:54 okacz Exp $
* /
* /---------------------------------------------------------------------------------------------*/
#ifndef _gsu3_h_
#define _gsu3_h_

#include "../../define.h"
#include "gcomplex.h"
#include "gvect3.h"
#include <random>
#include <limits>
#include "grnd.h"
#include <float.h>
#include <type_traits>
//<<<<<<< HEAD
#include "gsu2.h"
//=======
//>>>>>>> main



template<class floatT>
class GSU3;

template<class floatT>
__host__ std::ostream &operator<<(std::ostream &, const GSU3<floatT> &);

template<class floatT>
__host__ std::istream &operator>>(std::istream &, GSU3<floatT> &);

template<class floatT>
__device__ __host__ inline GSU3<floatT> operator+(const GSU3<floatT> &, const GSU3<floatT> &);

template<class floatT>
__device__ __host__ inline GSU3<floatT> operator-(const GSU3<floatT> &, const GSU3<floatT> &);

template<class floatT>
__device__ __host__ inline GSU3<floatT> operator*(const GCOMPLEX(floatT) &, const GSU3<floatT> &);

template<class floatT>
__device__ __host__ inline GSU3<floatT> operator*(const GSU3<floatT> &, const GCOMPLEX(floatT) &);

template<class floatT>
__device__ __host__ inline GSU3<floatT> operator*(const floatT &, const GSU3<floatT> &);

template<class floatT>
__device__ __host__ inline GSU3<floatT> operator*(const GSU3<floatT> &, const floatT &);

template<class floatT>
__device__ __host__ inline GSU3<floatT> operator*(const GSU3<floatT> &, const GSU3<floatT> &);

template<class floatT>
__device__ __host__ inline GSU3<floatT> operator/(const GSU3<floatT> &, const floatT &);

template<class floatT>
__device__ __host__ floatT tr_d(const GSU3<floatT> &);

template<class floatT>
__device__ __host__ floatT tr_i(const GSU3<floatT> &);

template<class floatT>
__device__ __host__ floatT tr_d(const GSU3<floatT> &, const GSU3<floatT> &);

template<class floatT>
__device__ __host__ GCOMPLEX(floatT) tr_c(const GSU3<floatT> &);

template<class floatT>
__device__ __host__ GCOMPLEX(floatT) tr_c(const GSU3<floatT> &, const GSU3<floatT> &);

template<class floatT>
__device__ __host__ GSU3<floatT> dagger(const GSU3<floatT> &);

template<class floatT>
__device__ __host__ GCOMPLEX(floatT) det(const GSU3<floatT> &X);

template<class floatT>
__device__ __host__ floatT realdet(const GSU3<floatT> &X);

template<class floatT>
__device__ __host__ floatT infnorm(const GSU3<floatT> &X);

template<class floatT>
__device__ __host__ GSU3<floatT> su3_exp(GSU3<floatT>);

template<class floatT>
__device__ __host__ void dumpmat();

template<class floatT>
__device__ __host__ gVect3<floatT> operator*(const GSU3<floatT> &, const gVect3<floatT> &);

template<class floatT>
__device__ __host__ GSU3<floatT> tensor_prod(const gVect3<floatT> &, const gVect3<floatT> &);

template<class floatT>
__device__ __host__ inline bool compareGSU3(GSU3<floatT> a, GSU3<floatT> b, floatT tol=1e-13);

template<class floatT>
class GSU3 {
private:
    GCOMPLEX(floatT) _e00, _e01, _e02,
                     _e10, _e11, _e12,
                     _e20, _e21, _e22;

public:

    constexpr GSU3(const GSU3<floatT>&) = default;
    __host__ __device__ GSU3() {};

    __host__ __device__ GSU3(const floatT x) {
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

    __host__ __device__ GSU3(GCOMPLEX(floatT) e00, GCOMPLEX(floatT) e01, GCOMPLEX(floatT) e02,
                             GCOMPLEX(floatT) e10, GCOMPLEX(floatT) e11, GCOMPLEX(floatT) e12,
                             GCOMPLEX(floatT) e20, GCOMPLEX(floatT) e21, GCOMPLEX(floatT) e22) :
                                 _e00(e00), _e01(e01), _e02(e02),
                                 _e10(e10), _e11(e11), _e12(e12),
                                 _e20(e20), _e21(e21), _e22(e22) {};



#if (!defined __GPUCC__)
    __host__ friend std::ostream& operator<< <> (std::ostream&, const GSU3<floatT> &);
#endif

    __host__ friend std::istream &operator>><>(std::istream &, GSU3<floatT> &);


    // matrix operations
    __device__ __host__ friend GSU3<floatT> operator+<>(const GSU3<floatT> &, const GSU3<floatT> &);

    __device__ __host__ friend GSU3<floatT> operator-<>(const GSU3<floatT> &, const GSU3<floatT> &);

    __device__ __host__ friend GSU3<floatT> operator*<>(const GCOMPLEX(floatT) &x, const GSU3<floatT> &y);

    __device__ __host__ friend GSU3<floatT> operator*<>(const GSU3<floatT> &x, const GCOMPLEX(floatT) &y);

    __device__ __host__ friend GSU3<floatT> operator*<>(const floatT &x, const GSU3<floatT> &y);

    __device__ __host__ friend GSU3<floatT> operator*<>(const GSU3<floatT> &x, const floatT &y);

    __device__ __host__ friend GSU3<floatT> operator*<>(const GSU3<floatT> &, const GSU3<floatT> &);

    __device__ __host__ friend GSU3<floatT> operator/<>(const GSU3<floatT> &x, const floatT &y);

    __device__ __host__ bool operator==(const GSU3<floatT> &);

    __device__ __host__ GSU3<floatT> &operator=(const GSU3<floatT> &);

    __device__ __host__ GSU3<floatT> &operator+=(const GSU3<floatT> &);

    __device__ __host__ GSU3<floatT> &operator-=(const GSU3<floatT> &);

    __device__ __host__ GSU3<floatT> &operator*=(const floatT &);

    __device__ __host__ GSU3<floatT> &operator*=(const GCOMPLEX(floatT) &);

    __device__ __host__ GSU3<floatT> &operator*=(const GSU3<floatT> &);

    __device__ __host__ GSU3<floatT> &operator/=(const floatT &);

    // cast operations single <-> double precision
    template<class T>
    __device__ __host__ inline operator GSU3<T>() const {
        return GSU3<T>(GCOMPLEX(T)(_e00.cREAL, _e00.cIMAG), GCOMPLEX(T)(_e01.cREAL, _e01.cIMAG),
                       GCOMPLEX(T)(_e02.cREAL, _e02.cIMAG),
                       GCOMPLEX(T)(_e10.cREAL, _e10.cIMAG), GCOMPLEX(T)(_e11.cREAL, _e11.cIMAG),
                       GCOMPLEX(T)(_e12.cREAL, _e12.cIMAG),
                       GCOMPLEX(T)(_e20.cREAL, _e20.cIMAG), GCOMPLEX(T)(_e21.cREAL, _e21.cIMAG),
                       GCOMPLEX(T)(_e22.cREAL, _e22.cIMAG));
    }


    __device__ __host__ friend gVect3<floatT>
    operator*<>(const GSU3<floatT> &, const gVect3<floatT> &);     // GSU3 * cvect3 multiplication
    __device__ __host__ friend GSU3<floatT>
    tensor_prod<>(const gVect3<floatT> &, const gVect3<floatT> &); // tensor product of two cvect3

    __device__ __host__ friend bool
    compareGSU3<>(GSU3<floatT> a, GSU3<floatT> b, floatT tol);

    __device__ __host__ void random(uint4 *state);                 // set links randomly
    __device__ __host__ void gauss(uint4 *state);                  // set links gauss
    __device__ __host__ void su3unitarize();                       // project to su3 using first two rows of link
    __device__ __host__ int su3unitarize_hits(const int Nhit, floatT tol); // project to su3 using first two rows of link
    __device__ __host__ void su3reconstruct12()                    // project to su3 using first two rows of link
    {
        _e20 = GCOMPLEX(floatT)((_e01.cREAL * _e12.cREAL - _e01.cIMAG * _e12.cIMAG
                                 - (_e02.cREAL * _e11.cREAL - _e02.cIMAG * _e11.cIMAG)),
                                (-(_e01.cIMAG * _e12.cREAL + _e01.cREAL * _e12.cIMAG)
                                 + (_e02.cIMAG * _e11.cREAL + _e02.cREAL * _e11.cIMAG)));

        _e21 = GCOMPLEX(floatT)((_e02.cREAL * _e10.cREAL - _e02.cIMAG * _e10.cIMAG
                                 - (_e00.cREAL * _e12.cREAL - _e00.cIMAG * _e12.cIMAG)),
                                (-(_e02.cIMAG * _e10.cREAL + _e02.cREAL * _e10.cIMAG)
                                 + (_e00.cIMAG * _e12.cREAL + _e00.cREAL * _e12.cIMAG)));

        _e22 = GCOMPLEX(floatT)((_e00.cREAL * _e11.cREAL - _e00.cIMAG * _e11.cIMAG
                                 - (_e01.cREAL * _e10.cREAL - _e01.cIMAG * _e10.cIMAG)),
                                (-(_e00.cIMAG * _e11.cREAL + _e00.cREAL * _e11.cIMAG)
                                 + (_e01.cIMAG * _e10.cREAL + _e01.cREAL * _e10.cIMAG)));
    }

    __device__ __host__ void su3reconstruct12Dagger()   // project to su3 using first two rows of link
    {
        _e02 = GCOMPLEX(floatT)((_e10.cREAL * _e21.cREAL - _e10.cIMAG * _e21.cIMAG
                                 - (_e20.cREAL * _e11.cREAL - _e20.cIMAG * _e11.cIMAG)),
                                (-(_e10.cIMAG * _e21.cREAL + _e10.cREAL * _e21.cIMAG)
                                 + (_e20.cIMAG * _e11.cREAL + _e20.cREAL * _e11.cIMAG)));

        _e12 = GCOMPLEX(floatT)((_e20.cREAL * _e01.cREAL - _e20.cIMAG * _e01.cIMAG
                                 - (_e00.cREAL * _e21.cREAL - _e00.cIMAG * _e21.cIMAG)),
                                (-(_e20.cIMAG * _e01.cREAL + _e20.cREAL * _e01.cIMAG)
                                 + (_e00.cIMAG * _e21.cREAL + _e00.cREAL * _e21.cIMAG)));

        _e22 = GCOMPLEX(floatT)((_e00.cREAL * _e11.cREAL - _e00.cIMAG * _e11.cIMAG
                                 - (_e10.cREAL * _e01.cREAL - _e10.cIMAG * _e01.cIMAG)),
                                (-(_e00.cIMAG * _e11.cREAL + _e00.cREAL * _e11.cIMAG)
                                 + (_e10.cIMAG * _e01.cREAL + _e10.cREAL * _e01.cIMAG)));
    }

    __device__ __host__ void u3reconstruct(const GCOMPLEX(floatT) phase)   // project to u3 using first two rows of link
    {
        
        _e20 = GCOMPLEX(floatT)((_e01.cREAL * _e12.cREAL - _e01.cIMAG * _e12.cIMAG
                                 - (_e02.cREAL * _e11.cREAL - _e02.cIMAG * _e11.cIMAG)),
                                (-(_e01.cIMAG * _e12.cREAL + _e01.cREAL * _e12.cIMAG)
                                 + (_e02.cIMAG * _e11.cREAL + _e02.cREAL * _e11.cIMAG)));

        _e21 = GCOMPLEX(floatT)((_e02.cREAL * _e10.cREAL - _e02.cIMAG * _e10.cIMAG
                                 - (_e00.cREAL * _e12.cREAL - _e00.cIMAG * _e12.cIMAG)),
                                (-(_e02.cIMAG * _e10.cREAL + _e02.cREAL * _e10.cIMAG)
                                 + (_e00.cIMAG * _e12.cREAL + _e00.cREAL * _e12.cIMAG)));

        _e22 = GCOMPLEX(floatT)((_e00.cREAL * _e11.cREAL - _e00.cIMAG * _e11.cIMAG
                                 - (_e01.cREAL * _e10.cREAL - _e01.cIMAG * _e10.cIMAG)),
                                (-(_e00.cIMAG * _e11.cREAL + _e00.cREAL * _e11.cIMAG)
                                 + (_e01.cIMAG * _e10.cREAL + _e01.cREAL * _e10.cIMAG)));
        
        
        
        //For half prec, this block is faster than the above because GCOMPLEX(__half)*GCOMPLEX(__half) uses __half2 intrinsics. The above version would end up using __half intrinsics.
        /*_e20 = _e01*_e12 - _e02*_e11;
        _e12 = _e02*_e10 - _e00*_e12;
        _e22 = _e00*_e11 - _e01*_e10;
        */
        _e20 *= phase;
        _e21 *= phase;
        _e22 *= phase;
    }

    __device__ __host__ void u3reconstructDagger(const GCOMPLEX(floatT) phase)   // project to u3 using first two rows of link
    {
        
        _e02 = GCOMPLEX(floatT)((_e10.cREAL * _e21.cREAL - _e10.cIMAG * _e21.cIMAG
                                 - (_e20.cREAL * _e11.cREAL - _e20.cIMAG * _e11.cIMAG)),
                                (-(_e10.cIMAG * _e21.cREAL + _e10.cREAL * _e21.cIMAG)
                                 + (_e20.cIMAG * _e11.cREAL + _e20.cREAL * _e11.cIMAG)));

        _e12 = GCOMPLEX(floatT)((_e20.cREAL * _e01.cREAL - _e20.cIMAG * _e01.cIMAG
                                 - (_e00.cREAL * _e21.cREAL - _e00.cIMAG * _e21.cIMAG)),
                                (-(_e20.cIMAG * _e01.cREAL + _e20.cREAL * _e01.cIMAG)
                                 + (_e00.cIMAG * _e21.cREAL + _e00.cREAL * _e21.cIMAG)));

        _e22 = GCOMPLEX(floatT)((_e00.cREAL * _e11.cREAL - _e00.cIMAG * _e11.cIMAG
                                 - (_e10.cREAL * _e01.cREAL - _e10.cIMAG * _e01.cIMAG)),
                                (-(_e00.cIMAG * _e11.cREAL + _e00.cREAL * _e11.cIMAG)
                                 + (_e10.cIMAG * _e01.cREAL + _e10.cREAL * _e01.cIMAG)));
        
        //For half prec, this block is faster than the above because GCOMPLEX(__half)*GCOMPLEX(__half) uses __half2 intrinsics. The above version would end up using __half intrinsics.
        /*_e02 = _e10*_e21 - _e20*_e11;
        _e12 = _e20*_e01 - _e00*_e21;
        _e22 = _e00*_e11 - _e10*_e01;
        */
        _e02 *= phase;
        _e12 *= phase;
        _e22 *= phase;
    }

    __device__ __host__ void reconstruct14(const GCOMPLEX(floatT) det)
    {
        floatT amp = pow(abs(det), 1.0/3.0);
        GCOMPLEX(floatT) phase = det / abs(det);

        _e20 = GCOMPLEX(floatT)((_e01.cREAL * _e12.cREAL - _e01.cIMAG * _e12.cIMAG
                                 - (_e02.cREAL * _e11.cREAL - _e02.cIMAG * _e11.cIMAG)),
                                (-(_e01.cIMAG * _e12.cREAL + _e01.cREAL * _e12.cIMAG)
                                 + (_e02.cIMAG * _e11.cREAL + _e02.cREAL * _e11.cIMAG)));

        _e21 = GCOMPLEX(floatT)((_e02.cREAL * _e10.cREAL - _e02.cIMAG * _e10.cIMAG
                                 - (_e00.cREAL * _e12.cREAL - _e00.cIMAG * _e12.cIMAG)),
                                (-(_e02.cIMAG * _e10.cREAL + _e02.cREAL * _e10.cIMAG)
                                 + (_e00.cIMAG * _e12.cREAL + _e00.cREAL * _e12.cIMAG)));

        _e22 = GCOMPLEX(floatT)((_e00.cREAL * _e11.cREAL - _e00.cIMAG * _e11.cIMAG
                                 - (_e01.cREAL * _e10.cREAL - _e01.cIMAG * _e10.cIMAG)),
                                (-(_e00.cIMAG * _e11.cREAL + _e00.cREAL * _e11.cIMAG)
                                 + (_e01.cIMAG * _e10.cREAL + _e01.cREAL * _e10.cIMAG)));

        _e20 *= phase/amp;
        _e21 *= phase/amp;
        _e22 *= phase/amp;
    }

    __device__ __host__ void reconstruct14Dagger(const GCOMPLEX(floatT) det)
    {

        floatT amp = pow(abs(det), 1.0/3.0);
        GCOMPLEX(floatT) phase = det / abs(det);

        _e02 = GCOMPLEX(floatT)((_e10.cREAL * _e21.cREAL - _e10.cIMAG * _e21.cIMAG
                                 - (_e20.cREAL * _e11.cREAL - _e20.cIMAG * _e11.cIMAG)),
                                (-(_e10.cIMAG * _e21.cREAL + _e10.cREAL * _e21.cIMAG)
                                 + (_e20.cIMAG * _e11.cREAL + _e20.cREAL * _e11.cIMAG)));

        _e12 = GCOMPLEX(floatT)((_e20.cREAL * _e01.cREAL - _e20.cIMAG * _e01.cIMAG
                                 - (_e00.cREAL * _e21.cREAL - _e00.cIMAG * _e21.cIMAG)),
                                (-(_e20.cIMAG * _e01.cREAL + _e20.cREAL * _e01.cIMAG)
                                 + (_e00.cIMAG * _e21.cREAL + _e00.cREAL * _e21.cIMAG)));

        _e22 = GCOMPLEX(floatT)((_e00.cREAL * _e11.cREAL - _e00.cIMAG * _e11.cIMAG
                                 - (_e10.cREAL * _e01.cREAL - _e10.cIMAG * _e01.cIMAG)),
                                (-(_e00.cIMAG * _e11.cREAL + _e00.cREAL * _e11.cIMAG)
                                 + (_e10.cIMAG * _e01.cREAL + _e10.cREAL * _e01.cIMAG)));
        _e02 *= phase/amp;
        _e12 *= phase/amp;
        _e22 *= phase/amp;
    }
    __device__ __host__ void TA();                                               // traceless anti-hermitian of link
    __device__ __host__ friend floatT tr_d<>(const GSU3<floatT> &);              // real part of trace of link
    __device__ __host__ friend floatT tr_i<>(const GSU3<floatT> &);              // imaginary part of trace of link
    __device__ __host__ friend floatT
    tr_d<>(const GSU3<floatT> &, const GSU3<floatT> &);                          // real part of trace of link*link
    __device__ __host__ friend GCOMPLEX(floatT) tr_c<>(const GSU3<floatT> &);    // trace of link
    __device__ __host__ friend GCOMPLEX(floatT) tr_c<>(const GSU3<floatT> &,
                                                       const GSU3<floatT> &);    // trace of link*link
    __device__ __host__ friend GSU3<floatT>
    dagger<>(const GSU3<floatT> &);                                              // hermitian conjugate
    __device__ __host__ friend GSU3<floatT> su3_exp<>(GSU3<floatT>);             // exp( link )
    __device__ __host__ friend GCOMPLEX(floatT) det<>(const GSU3<floatT> &);
    __device__ __host__ friend floatT infnorm<>(const GSU3<floatT> &);

    // accessors
    __host__ __device__ inline GCOMPLEX(floatT) getLink00() const;
    __host__ __device__ inline GCOMPLEX(floatT) getLink01() const;
    __host__ __device__ inline GCOMPLEX(floatT) getLink02() const;
    __host__ __device__ inline GCOMPLEX(floatT) getLink10() const;
    __host__ __device__ inline GCOMPLEX(floatT) getLink11() const;
    __host__ __device__ inline GCOMPLEX(floatT) getLink12() const;
    __host__ __device__ inline GCOMPLEX(floatT) getLink20() const;
    __host__ __device__ inline GCOMPLEX(floatT) getLink21() const;
    __host__ __device__ inline GCOMPLEX(floatT) getLink22() const;

    // setters
    __host__ __device__ inline void setLink00(GCOMPLEX(floatT) x);
    __host__ __device__ inline void setLink01(GCOMPLEX(floatT) x);
    __host__ __device__ inline void setLink02(GCOMPLEX(floatT) x);
    __host__ __device__ inline void setLink10(GCOMPLEX(floatT) x);
    __host__ __device__ inline void setLink11(GCOMPLEX(floatT) x);
    __host__ __device__ inline void setLink12(GCOMPLEX(floatT) x);
    __host__ __device__ inline void setLink20(GCOMPLEX(floatT) x);
    __host__ __device__ inline void setLink21(GCOMPLEX(floatT) x);
    __host__ __device__ inline void setLink22(GCOMPLEX(floatT) x);


    __host__ __device__ inline GCOMPLEX(floatT) &operator()(int i, int j) {
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
        _e00 = GCOMPLEX(floatT)(nan(""), nan(""));
        return _e00;
    }

//<<<<<<< HEAD
    //ported from Milc ~dsh
    __device__ __host__ void dumpmat(){
    int i,j;
    for(i=0;i<3;i++){
        for(j=0;j<3;j++)printf("(%.2e,%.2e)\t",
            real((*this)(i,j)),imag((*this)(i,j)));
        printf("\n");
    }
    	printf("\n");
    }

//    __host__ __device__ inline const GCOMPLEX(floatT) &operator()(int i, int j) const {
//=======
    __host__ inline const GCOMPLEX(floatT) &operator()(int i, int j) const {
//>>>>>>> main
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
        throw std::runtime_error(stdLogger.fatal("GSU3 access to element (", i, ",", j, ") not possible!"));
    }

    __host__ __device__ GSU3<floatT> getAccessor() const {
        return *this;
    }

    template<typename Index>
    __host__ __device__ GSU3<floatT> operator()(const Index) const {
        return *this;
    }
};

// accessors
template<class floatT>
__host__ __device__ inline GCOMPLEX(floatT) GSU3<floatT>::getLink00() const {
    return _e00;
}

template<class floatT>
__host__ __device__ inline GCOMPLEX(floatT) GSU3<floatT>::getLink01() const {
    return _e01;
}

template<class floatT>
__host__ __device__ inline GCOMPLEX(floatT) GSU3<floatT>::getLink02() const {
    return _e02;
}

template<class floatT>
__host__ __device__ inline GCOMPLEX(floatT) GSU3<floatT>::getLink10() const {
    return _e10;
}

template<class floatT>
__host__ __device__ inline GCOMPLEX(floatT) GSU3<floatT>::getLink11() const {
    return _e11;
}

template<class floatT>
__host__ __device__ inline GCOMPLEX(floatT) GSU3<floatT>::getLink12() const {
    return _e12;
}

template<class floatT>
__host__ __device__ inline GCOMPLEX(floatT) GSU3<floatT>::getLink20() const {
    return _e20;
}

template<class floatT>
__host__ __device__ inline GCOMPLEX(floatT) GSU3<floatT>::getLink21() const {
    return _e21;
}

template<class floatT>
__host__ __device__ inline GCOMPLEX(floatT) GSU3<floatT>::getLink22() const {
    return _e22;
}


// setters
template<class floatT>
__host__ __device__ inline void GSU3<floatT>::setLink00(GCOMPLEX(floatT) x) {
    _e00 = x;
}

template<class floatT>
__host__ __device__ inline void GSU3<floatT>::setLink01(GCOMPLEX(floatT) x) {
    _e01 = x;
}

template<class floatT>
__host__ __device__ inline void GSU3<floatT>::setLink02(GCOMPLEX(floatT) x) {
    _e02 = x;
}

template<class floatT>
__host__ __device__ inline void GSU3<floatT>::setLink10(GCOMPLEX(floatT) x) {
    _e10 = x;
}

template<class floatT>
__host__ __device__ inline void GSU3<floatT>::setLink11(GCOMPLEX(floatT) x) {
    _e11 = x;
}

template<class floatT>
__host__ __device__ inline void GSU3<floatT>::setLink12(GCOMPLEX(floatT) x) {
    _e12 = x;
}

template<class floatT>
__host__ __device__ inline void GSU3<floatT>::setLink20(GCOMPLEX(floatT) x) {
    _e20 = x;
}

template<class floatT>
__host__ __device__ inline void GSU3<floatT>::setLink21(GCOMPLEX(floatT) x) {
    _e21 = x;
}

template<class floatT>
__host__ __device__ inline void GSU3<floatT>::setLink22(GCOMPLEX(floatT) x) {
    _e22 = x;
}

// some constant su3 matrices
template<class floatT>
__device__ __host__ inline GSU3<floatT> gsu3_one() {
    return GSU3<floatT>(1, 0, 0,
                        0, 1, 0,
                        0, 0, 1);
}

template <>
__device__ __host__ inline GSU3<__half> gsu3_one() {
    GPUcomplex<__half> g_one(__float2half(1.0));
    GPUcomplex<__half> g_zero(__float2half(0.0));
    
    return GSU3<__half>( g_one, g_zero, g_zero,
                         g_zero, g_one, g_zero,
                         g_zero, g_zero, g_one);
}


template<class floatT>
__device__ __host__ inline GSU3<floatT> gsu3_zero() {
    return GSU3<floatT>(0, 0, 0,
                        0, 0, 0,
                        0, 0, 0);
}

template<class floatT>
__device__ __host__ inline GSU3<floatT> glambda_1() {
    return GSU3<floatT>(0, 1, 0,
                        1, 0, 0,
                        0, 0, 0);
}

template<class floatT>
__device__ __host__ inline GSU3<floatT> glambda_2() {
    return GSU3<floatT>(0                     , -GCOMPLEX(floatT)(0, 1), 0,
                        GCOMPLEX(floatT)(0, 1), 0                      , 0,
                        0                     , 0                      , 0);
}

template<class floatT>
__device__ __host__ inline GSU3<floatT> glambda_3() {
    return GSU3<floatT>(1, 0 , 0,
                        0, -1, 0,
                        0, 0 , 0);
}

template<class floatT>
__device__ __host__ inline GSU3<floatT> glambda_4() {
    return GSU3<floatT>(0, 0, 1,
                        0, 0, 0,
                        1, 0, 0);
}

template<class floatT>
__device__ __host__ inline GSU3<floatT> glambda_5() {
    return GSU3<floatT>(0                     , 0, -GCOMPLEX(floatT)(0, 1),
                        0                     , 0, 0,
                        GCOMPLEX(floatT)(0, 1), 0, 0);
}

template<class floatT>
__device__ __host__ inline GSU3<floatT> glambda_6() {
    return GSU3<floatT>(0, 0, 0,
                        0, 0, 1,
                        0, 1, 0);
}

template<class floatT>
__device__ __host__ inline GSU3<floatT> glambda_7() {
    return GSU3<floatT>(0, 0                     , 0,
                        0, 0                     , -GCOMPLEX(floatT)(0, 1),
                        0, GCOMPLEX(floatT)(0, 1), 0);
}

template<class floatT>
__device__ __host__ inline GSU3<floatT> glambda_8() {
    return GSU3<floatT>(1 / sqrt(3), 0          , 0,
                        0          , 1 / sqrt(3), 0,
                        0          , 0          , -2 / sqrt(3));
}




// matrix operations
template<class floatT>
__device__ __host__ GSU3<floatT> operator+(const GSU3<floatT> &x, const GSU3<floatT> &y) {
    return GSU3<floatT>(
            x._e00 + y._e00, x._e01 + y._e01, x._e02 + y._e02,
            x._e10 + y._e10, x._e11 + y._e11, x._e12 + y._e12,
            x._e20 + y._e20, x._e21 + y._e21, x._e22 + y._e22);
}

template<class floatT>
__device__ __host__ GSU3<floatT> operator-(const GSU3<floatT> &x, const GSU3<floatT> &y) {
    return GSU3<floatT>(
            x._e00 - y._e00, x._e01 - y._e01, x._e02 - y._e02,
            x._e10 - y._e10, x._e11 - y._e11, x._e12 - y._e12,
            x._e20 - y._e20, x._e21 - y._e21, x._e22 - y._e22);
}


template<class floatT>
__device__ __host__ GSU3<floatT> operator*(const GCOMPLEX(floatT) &x, const GSU3<floatT> &y) {
    return GSU3<floatT>(
            x * y._e00, x * y._e01, x * y._e02,
            x * y._e10, x * y._e11, x * y._e12,
            x * y._e20, x * y._e21, x * y._e22);
}

template<class floatT>
__device__ __host__ GSU3<floatT> operator*(const GSU3<floatT> &x, const GCOMPLEX(floatT) &y) {
    return GSU3<floatT>(
            x._e00 * y, x._e01 * y, x._e02 * y,
            x._e10 * y, x._e11 * y, x._e12 * y,
            x._e20 * y, x._e21 * y, x._e22 * y);
}

template<class floatT>
__device__ __host__ GSU3<floatT> operator*(const floatT &x, const GSU3<floatT> &y) {
    return GSU3<floatT>(
            x * y._e00, x * y._e01, x * y._e02,
            x * y._e10, x * y._e11, x * y._e12,
            x * y._e20, x * y._e21, x * y._e22);
}

template<class floatT>
__device__ __host__ GSU3<floatT> operator*(const GSU3<floatT> &x, const floatT &y) {
    return GSU3<floatT>(
            x._e00 * y, x._e01 * y, x._e02 * y,
            x._e10 * y, x._e11 * y, x._e12 * y,
            x._e20 * y, x._e21 * y, x._e22 * y);
}

template<class floatT>
__device__ __host__ GSU3<floatT> operator/(const GSU3<floatT> &x, const floatT &y) {
    return GSU3<floatT>(
            x._e00 / y, x._e01 / y, x._e02 / y,
            x._e10 / y, x._e11 / y, x._e12 / y,
            x._e20 / y, x._e21 / y, x._e22 / y);
}


template<class floatT>
__device__ __host__ GSU3<floatT> operator*(const GSU3<floatT> &x, const GSU3<floatT> &y) {
    GCOMPLEX(floatT) tmp00, tmp01, tmp02,
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

    return GSU3<floatT>(tmp00, tmp01, tmp02,
                        tmp10, tmp11, tmp12,
                        tmp20, tmp21, tmp22);
}


// su3 * cvect3 multiplication
template<class floatT>
__device__ __host__ gVect3<floatT> operator*(const GSU3<floatT> &x, const gVect3<floatT> &y) {
    GCOMPLEX(floatT) tmp0, tmp1, tmp2;

    tmp0 = x._e00 * y._v0 + x._e01 * y._v1 + x._e02 * y._v2;
    tmp1 = x._e10 * y._v0 + x._e11 * y._v1 + x._e12 * y._v2;
    tmp2 = x._e20 * y._v0 + x._e21 * y._v1 + x._e22 * y._v2;

    return gVect3<floatT>(tmp0, tmp1, tmp2);
}


template<class floatT>
__device__ __host__ inline GSU3<floatT> &GSU3<floatT>::operator=(const GSU3<floatT> &y) {
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
__device__ __host__ GSU3<floatT> &GSU3<floatT>::operator+=(const GSU3<floatT> &y) {
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
__device__ __host__ GSU3<floatT> &GSU3<floatT>::operator-=(const GSU3<floatT> &y) {
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
__device__ __host__ GSU3<floatT> &GSU3<floatT>::operator*=(const floatT &y) {
    *this = *this * y;
    return *this;
}

template<class floatT>
__device__ __host__ GSU3<floatT> &GSU3<floatT>::operator*=(const GCOMPLEX(floatT) &y) {
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
__device__ __host__ GSU3<floatT> &GSU3<floatT>::operator*=(const GSU3<floatT> &y) {
    *this = *this * y;
    return *this;
}

template<class floatT>
__device__ __host__ GSU3<floatT> &GSU3<floatT>::operator/=(const floatT &y) {
    *this = *this / y;
    return *this;
}

/// This method is a straightforward way to compare two SU(3) matrices. You may decide that you want to control the
/// tolerance for comparison. In that case please look to the compareGSU3 method. In case you are comparing with the
/// zero matrix, you should use compareGSU3, as the present method seems not to work for that case.
template<class floatT>
__device__ __host__ bool GSU3<floatT>::operator==(const GSU3<floatT> &y) {
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
__host__ inline std::ostream &operator<<(std::ostream &s, const GSU3<floatT> &x) {
return s << "\n" << x.getLink00() << x.getLink01() << x.getLink02() << "\n"
                 << x.getLink10() << x.getLink11() << x.getLink12() << "\n"
                 << x.getLink20() << x.getLink21() << x.getLink22() << "\n";
}

/// TODO: This presumably doesn't work
template<class floatT>
__host__ inline std::istream &operator>>(std::istream &s, GSU3<floatT> &x) {
    return s >> x._e00.cREAL >> x._e00.cIMAG >> x._e01.cREAL >> x._e01.cIMAG >> x._e02.cREAL >> x._e02.cIMAG
             >> x._e10.cREAL >> x._e10.cIMAG >> x._e11.cREAL >> x._e11.cIMAG >> x._e12.cREAL >> x._e12.cIMAG
             >> x._e20.cREAL >> x._e20.cIMAG >> x._e21.cREAL >> x._e21.cIMAG >> x._e22.cREAL >> x._e22.cIMAG;
}


template<class floatT>
__device__ __host__ void GSU3<floatT>::random(uint4 *state) {

    GCOMPLEX(floatT)
            rnd;

    rnd = GCOMPLEX(floatT)(get_rand<floatT>(state), get_rand<floatT>(state));
    _e00 = (floatT) 1.0 - (floatT) 2.0 * rnd;
    rnd = GCOMPLEX(floatT)(get_rand<floatT>(state), get_rand<floatT>(state));
    _e01 = (floatT) 1.0 - (floatT) 2.0 * rnd;
    rnd = GCOMPLEX(floatT)(get_rand<floatT>(state), get_rand<floatT>(state));
    _e02 = (floatT) 1.0 - (floatT) 2.0 * rnd;
    rnd = GCOMPLEX(floatT)(get_rand<floatT>(state), get_rand<floatT>(state));
    _e10 = (floatT) 1.0 - (floatT) 2.0 * rnd;
    rnd = GCOMPLEX(floatT)(get_rand<floatT>(state), get_rand<floatT>(state));
    _e11 = (floatT) 1.0 - (floatT) 2.0 * rnd;
    rnd = GCOMPLEX(floatT)(get_rand<floatT>(state), get_rand<floatT>(state));
    _e12 = (floatT) 1.0 - (floatT) 2.0 * rnd;
    rnd = GCOMPLEX(floatT)(get_rand<floatT>(state), get_rand<floatT>(state));
    _e20 = (floatT) 1.0 - (floatT) 2.0 * rnd;
    rnd = GCOMPLEX(floatT)(get_rand<floatT>(state), get_rand<floatT>(state));
    _e21 = (floatT) 1.0 - (floatT) 2.0 * rnd;
    rnd = GCOMPLEX(floatT)(get_rand<floatT>(state), get_rand<floatT>(state));
    _e22 = (floatT) 1.0 - (floatT) 2.0 * rnd;

    (*this).su3unitarize();
}


template<class floatT>
__device__ __host__ void GSU3<floatT>::gauss(uint4 *state) {
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
            
            _e00 = GCOMPLEX(floatT)(temp1[2] + 1. / sqrt(3.0) * temp2[3], 0.0);
            _e01 = GCOMPLEX(floatT)(temp1[0], -temp1[1]);
            _e02 = GCOMPLEX(floatT)(temp1[3], -temp2[0]);
            _e10 = GCOMPLEX(floatT)(temp1[0], temp1[1]);
            _e11 = GCOMPLEX(floatT)(-temp1[2] + 1. / sqrt(3.0) * temp2[3], 0.0);
            _e12 = GCOMPLEX(floatT)(temp2[1], -temp2[2]);
            _e20 = GCOMPLEX(floatT)(temp1[3], temp2[0]);
            _e21 = GCOMPLEX(floatT)(temp2[1], temp2[2]);
            _e22 = GCOMPLEX(floatT)(-2. / sqrt(3.0) * temp2[3], 0.0);
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

    _e00 = GCOMPLEX(__half)(__float2half(temp1[2] + 1. / sqrt(3.0) * temp2[3]), __float2half( 0.0));
    _e01 = GCOMPLEX(__half)(__float2half(temp1[0]), __float2half( -temp1[1]));
    _e02 = GCOMPLEX(__half)(__float2half(temp1[3]), __float2half( -temp2[0]));
    _e10 = GCOMPLEX(__half)(__float2half(temp1[0]), __float2half( temp1[1]));
    _e11 = GCOMPLEX(__half)(__float2half(-temp1[2] + 1. / sqrt(3.0) * temp2[3]), __float2half( 0.0));
    _e12 = GCOMPLEX(__half)(__float2half(temp2[1]), __float2half( -temp2[2]));
    _e20 = GCOMPLEX(__half)(__float2half(temp1[3]), __float2half( temp2[0]));
    _e21 = GCOMPLEX(__half)(__float2half(temp2[1]), __float2half( temp2[2]));
    _e22 = GCOMPLEX(__half)(__float2half(-2. / sqrt(3.0) * temp2[3]), __float2half( 0.0));
#endif
    }
}

/******************  realtr.c  (in su3.a) *******************************
*									*
* Real realtrace_su3( su3_matrix *a,*b)				*
* return Re( Tr( A_adjoint*B )  					*
*/
// ported from Milc by Dan Hoying, 2022
template<class floatT>
__device__ __host__ floatT realtrace_su3( GSU3<floatT> a, GSU3<floatT> b ){
int i,j;
floatT sum;
    for(sum=0.0,i=0;i<3;i++)for(j=0;j<3;j++)
	sum+= real(a(i,j))*real(b(i,j)) + imag(a(i,j))*imag(b(i,j));
    return(sum);
}

// ported from Milc by Dan Hoying, 2022
template<class floatT>
__device__ __host__ void left_su2_hit_n(GSU2_mat<floatT> *u, int p, int q, GSU3<floatT> *link)
{
  /* link <- u * link */
  /* The 0 row of the SU(2) matrix u matches row p of the SU(3) matrix */
  /* The 1 row of the SU(2) matrix u matches row q of the SU(3) matrix */
  /* C. DeTar 18 Oct 1990 */

  int m;

  for (m = 0; m < 3; m++)
    mult_su2_mat_vec_elem_n(u, &((*link)(p,m)), &((*link)(q,m)));

} /* l_su2_hit_n.c */

// ported from Milc by Dan Hoying, 2022
template<class floatT>
__device__ __host__ void mult_su2_mat_vec_elem_n(GSU2_mat<floatT> *u, GCOMPLEX(floatT) *x0,GCOMPLEX(floatT) *x1)
{
  /* Multiplies the complex column spinor (x0, x1) by the SU(2) matrix u */
  /* and puts the result in (x0,x1).  */
  /* Thus x <- u * x          */
  /* C. DeTar 3 Oct 1990 */
  
  GCOMPLEX(floatT) z0, z1, t0, t1;

  t0 = *x0; t1 = *x1;

  z0 = u->operator()(0,0) * t0;
  z1 = u->operator()(0,1) * t1;
  *x0 = z0 + z1;
  z0 = u->operator()(1,0) * t0;
  z1 = u->operator()(1,1) * t1;
  *x1 = z0 + z1;

} /* m_su2_mat_vec_elem_n.c */



// project to su3 by maximizing Re(Tr(guess*(toproj)))
// ported from Milc by Dan Hoying, 2022
template<class floatT>
__device__ __host__ int su3unitarize_hits(
	GSU3<floatT> *w,         /* input initial guess. output resulting SU(3) matrix */
	GSU3<floatT> *q,         /* 3 x 3 complex matrix to be projected */
   	int Nhit,              /* number of SU(2) hits. 0 for no projection */
   	floatT tol              /* tolerance for SU(3) projection.
			     If nonzero, treat Nhit as a maximum
			     number of hits.  If zero, treat Nhit
			     as a prescribed number of hits. */ 
		) {

   int index1, ina, inb,ii;
   floatT v0,v1,v2,v3, vsq;
   floatT z;
   GSU3<floatT> action(0);
   GSU2_mat<floatT> h;
   const int Nc = 3;
   double conver, old_tr = 0, new_tr;

   if(tol > 0)
     old_tr = realtrace_su3(*w,*q)/3.0;
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

      /* Elements of SU(2) matrix */

      h(0,0) = GCOMPLEX(floatT)( v0,-v3);
      h(0,1) = GCOMPLEX(floatT)(-v2,-v1);
      h(1,0) = GCOMPLEX(floatT)( v2,-v1);
      h(1,1) = GCOMPLEX(floatT)( v0, v3);

      /* update the link */
      left_su2_hit_n(&h,ina,inb,w);

      /* convergence measure every third hit */
      if(tol>0 && (index1 % 3) == 2){
	new_tr = realtrace_su3(*w,*q)/3.;
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
__device__ __host__ void GSU3<floatT>::su3unitarize() {
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

    _e10 -= GCOMPLEX(floatT)((Cre * _e00.cREAL - Cim * _e00.cIMAG), (Cre * _e00.cIMAG + Cim * _e00.cREAL));
    _e11 -= GCOMPLEX(floatT)((Cre * _e01.cREAL - Cim * _e01.cIMAG), (Cre * _e01.cIMAG + Cim * _e01.cREAL));
    _e12 -= GCOMPLEX(floatT)((Cre * _e02.cREAL - Cim * _e02.cIMAG), (Cre * _e02.cIMAG + Cim * _e02.cREAL));


// Normierung der 2.Zeile

    quadnorm = _e10.cREAL * _e10.cREAL + _e10.cIMAG * _e10.cIMAG
               + _e11.cREAL * _e11.cREAL + _e11.cIMAG * _e11.cIMAG
               + _e12.cREAL * _e12.cREAL + _e12.cIMAG * _e12.cIMAG;

    invnorm = 1.0 / sqrt(quadnorm);

    _e10 *= invnorm;
    _e11 *= invnorm;
    _e12 *= invnorm;


// 3.Zeile ist das Vektorprodukt von 1* und 2*

    _e20 = GCOMPLEX(floatT)((_e01.cREAL * _e12.cREAL - _e01.cIMAG * _e12.cIMAG
                             - (_e02.cREAL * _e11.cREAL - _e02.cIMAG * _e11.cIMAG)),
                            (-(_e01.cIMAG * _e12.cREAL + _e01.cREAL * _e12.cIMAG)
                             + (_e02.cIMAG * _e11.cREAL + _e02.cREAL * _e11.cIMAG)));

    _e21 = GCOMPLEX(floatT)((_e02.cREAL * _e10.cREAL - _e02.cIMAG * _e10.cIMAG
                             - (_e00.cREAL * _e12.cREAL - _e00.cIMAG * _e12.cIMAG)),
                            (-(_e02.cIMAG * _e10.cREAL + _e02.cREAL * _e10.cIMAG)
                             + (_e00.cIMAG * _e12.cREAL + _e00.cREAL * _e12.cIMAG)));

    _e22 = GCOMPLEX(floatT)((_e00.cREAL * _e11.cREAL - _e00.cIMAG * _e11.cIMAG
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

    _e10 -= GCOMPLEX(__half)((Cre * __half2float(_e00.cREAL) - Cim * __half2float(_e00.cIMAG)), (Cre * __half2float(_e00.cIMAG) + Cim * __half2float(_e00.cREAL)));
    _e11 -= GCOMPLEX(__half)((Cre * __half2float(_e01.cREAL) - Cim * __half2float(_e01.cIMAG)), (Cre * __half2float(_e01.cIMAG) + Cim * __half2float(_e01.cREAL)));
    _e12 -= GCOMPLEX(__half)((Cre * __half2float(_e02.cREAL) - Cim * __half2float(_e02.cIMAG)), (Cre * __half2float(_e02.cIMAG) + Cim * __half2float(_e02.cREAL)));


// Normierung der 2.Zeile

    quadnorm =  __half2float(_e10.cREAL * _e10.cREAL + _e10.cIMAG * _e10.cIMAG
                             + _e11.cREAL * _e11.cREAL + _e11.cIMAG * _e11.cIMAG
                             + _e12.cREAL * _e12.cREAL + _e12.cIMAG * _e12.cIMAG);

    invnorm = 1.0 / sqrt(quadnorm);

    _e10 *=  __float2half(invnorm);
    _e11 *= __float2half(invnorm);
    _e12 *= __float2half(invnorm);


// 3.Zeile ist das Vektorprodukt von 1* und 2*

    _e20 = GCOMPLEX(__half)((_e01.cREAL * _e12.cREAL - _e01.cIMAG * _e12.cIMAG
                             - (_e02.cREAL * _e11.cREAL - _e02.cIMAG * _e11.cIMAG)),
                            (-(_e01.cIMAG * _e12.cREAL + _e01.cREAL * _e12.cIMAG)
                             + (_e02.cIMAG * _e11.cREAL + _e02.cREAL * _e11.cIMAG)));

    _e21 = GCOMPLEX(__half)((_e02.cREAL * _e10.cREAL - _e02.cIMAG * _e10.cIMAG
                             - (_e00.cREAL * _e12.cREAL - _e00.cIMAG * _e12.cIMAG)),
                            (-(_e02.cIMAG * _e10.cREAL + _e02.cREAL * _e10.cIMAG)
                             + (_e00.cIMAG * _e12.cREAL + _e00.cREAL * _e12.cIMAG)));

    _e22 = GCOMPLEX(__half)((_e00.cREAL * _e11.cREAL - _e00.cIMAG * _e11.cIMAG
                             - (_e01.cREAL * _e10.cREAL - _e01.cIMAG * _e10.cIMAG)),
                            (-(_e00.cIMAG * _e11.cREAL + _e00.cREAL * _e11.cIMAG)
                             + (_e01.cIMAG * _e10.cREAL + _e01.cREAL * _e10.cIMAG)));
    #endif
    }
    
}

template<class floatT>
__device__ __host__ GCOMPLEX(floatT) det(const GSU3<floatT> &x) {
    GCOMPLEX(floatT)
            res;

    res = x._e00 * (x._e11 * x._e22 - x._e12 * x._e21)
          + x._e01 * (x._e12 * x._e20 - x._e10 * x._e22)
          + x._e02 * (x._e10 * x._e21 - x._e11 * x._e20);

    return (res);
}

template<class floatT>
__device__ __host__ floatT realdet(const GSU3<floatT> &x) {
    return det(x).cREAL;
}

template<class floatT>
__device__ __host__ floatT infnorm(const GSU3<floatT> &x) {
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
__device__ __host__ void GSU3<floatT>::TA() {
    GSU3 <floatT> tmp;

    tmp._e00 = GCOMPLEX(floatT)(0, 0.6666666666666666 * _e00.cIMAG - 0.3333333333333333 * (_e11.cIMAG + _e22.cIMAG));
    tmp._e01 = 0.5 * (_e01 - conj(_e10));
    tmp._e02 = 0.5 * (_e02 - conj(_e20));
    tmp._e10 = 0.5 * (_e10 - conj(_e01));
    tmp._e11 = GCOMPLEX(floatT)(0, 0.6666666666666666 * _e11.cIMAG - 0.3333333333333333 * (_e00.cIMAG + _e22.cIMAG));
    tmp._e12 = 0.5 * (_e12 - conj(_e21));
    tmp._e20 = 0.5 * (_e20 - conj(_e02));
    tmp._e21 = 0.5 * (_e21 - conj(_e12));
    tmp._e22 = GCOMPLEX(floatT)(0, 0.6666666666666666 * _e22.cIMAG - 0.3333333333333333 * (_e00.cIMAG + _e11.cIMAG));

    (*this) = tmp;
}

// real part of trace of link
template<class floatT>
__device__ __host__ floatT tr_d(const GSU3<floatT> &x) {
    return floatT(x._e00.cREAL + x._e11.cREAL + x._e22.cREAL);
}

// imaginary part of trace of link
template<class floatT>
__device__ __host__ floatT tr_i(const GSU3<floatT> &x) {
    return floatT(x._e00.cIMAG + x._e11.cIMAG + x._e22.cIMAG);
}

// real part of trace of link*link
template<class floatT>
__device__ __host__ floatT tr_d(const GSU3<floatT> &x, const GSU3<floatT> &y) {
    floatT res;
    res = (x._e00 * y._e00).cREAL + (x._e01 * y._e10).cREAL + (x._e02 * y._e20).cREAL
          + (x._e10 * y._e01).cREAL + (x._e11 * y._e11).cREAL + (x._e12 * y._e21).cREAL
          + (x._e20 * y._e02).cREAL + (x._e21 * y._e12).cREAL + (x._e22 * y._e22).cREAL;

    return (res);
}

// trace of link
template<class floatT>
__device__ __host__ GCOMPLEX(floatT) tr_c(const GSU3<floatT> &x) {
    return GCOMPLEX(floatT)(x._e00 + x._e11 + x._e22);
}

// trace of link*link
template<class floatT>
__device__ __host__ GCOMPLEX(floatT) tr_c(const GSU3<floatT> &x, const GSU3<floatT> &y) {
    GCOMPLEX(floatT)
            res;

    res = x._e00 * y._e00 + x._e01 * y._e10 + x._e02 * y._e20
          + x._e10 * y._e01 + x._e11 * y._e11 + x._e12 * y._e21
          + x._e20 * y._e02 + x._e21 * y._e12 + x._e22 * y._e22;

    return (res);
}

// hermitian conjugate
template<class floatT>
__device__ __host__ GSU3<floatT> dagger(const GSU3<floatT> &x) {
    GSU3 <floatT> tmp;

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
__device__ __host__ GSU3<floatT> su3_exp(GSU3<floatT> u) {
    GSU3 <floatT> res;

    res = gsu3_one<floatT>()
          + u * (gsu3_one<floatT>()
                 + u * ((floatT) 0.5 * gsu3_one<floatT>()
                        + u * ((floatT) 0.1666666666666666 * gsu3_one<floatT>()
                               + u * ((floatT) 0.0416666666666666 * gsu3_one<floatT>()
                                      + u * ((floatT) 0.0083333333333333 * gsu3_one<floatT>()
                                             + (floatT) 0.0013888888888888 * u)))));
    return (res);
}

// tensor product of two cvect3
template<class floatT>
__device__ __host__ GSU3<floatT> tensor_prod(const gVect3<floatT> &x, const gVect3<floatT> &y) {
    GSU3 <floatT> res;

    res._e00 = x._v0 * y._v0;
    res._e01 = x._v0 * y._v1;
    res._e02 = x._v0 * y._v2;
    res._e10 = x._v1 * y._v0;
    res._e11 = x._v1 * y._v1;
    res._e12 = x._v1 * y._v2;
    res._e20 = x._v2 * y._v0;
    res._e21 = x._v2 * y._v1;
    res._e22 = x._v2 * y._v2;

    return (res);
}

template<class floatT>
__device__ __host__ inline bool compareGSU3(GSU3<floatT> a, GSU3<floatT> b, floatT tol) {

    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
        GCOMPLEX(floatT) diff = a(i, j) - b(i, j);
        if (fabs(diff.cREAL) > tol) return false;
    }
    return true;
}

#endif
