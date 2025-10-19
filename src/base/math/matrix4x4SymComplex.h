//
// Created by Lukas Mazur on 29.11.18.
//

#pragma once
#include "../../define.h"
#include "complex.h"
#include "indices4x4Sym.h"


// Symmetric COMPLEX 4x4 Matrix
template<class floatT>
struct Matrix4x4SymComplex {
    enum entry {
        e00, e11, e22, e33, e01, e02, e03, e12, e13, e23
    };
    COMPLEX(floatT) elems[10];


    constexpr Matrix4x4SymComplex(const Matrix4x4SymComplex<floatT>&) = default;
    __device__ __host__ Matrix4x4SymComplex(floatT a) : elems{a, a, a, a, a, a, a, a, a, a} {}
    __device__ __host__ Matrix4x4SymComplex() : elems{0, 0, 0, 0, 0, 0, 0, 0, 0, 0} {}

    __device__ __host__ Matrix4x4SymComplex(
        floatT e00, floatT e11, floatT e22, floatT e33,
        floatT e01, floatT e02, floatT e03,
        floatT e12, floatT e13, floatT e23
    ) : elems{e00, e11, e22, e33, e01, e02, e03, e12, e13, e23} {}

    __device__ __host__ Matrix4x4SymComplex(
        COMPLEX(floatT) e00, COMPLEX(floatT) e11, COMPLEX(floatT) e22, COMPLEX(floatT) e33,
        COMPLEX(floatT) e01, COMPLEX(floatT) e02, COMPLEX(floatT) e03,
        COMPLEX(floatT) e12, COMPLEX(floatT) e13, COMPLEX(floatT) e23
    ) : elems{e00, e11, e22, e33, e01, e02, e03, e12, e13, e23} {}

    __device__ __host__ inline floatT operator()(int mu, int nu) {
        if (mu == 0 && nu == 0) return elems[entry::e00]; // 0 & 0 = 0
        else if (mu == 1 && nu == 1) return elems[entry::e11]; // 1 & 1 = 1
        else if (mu == 2 && nu == 2) return elems[entry::e22]; // 2 & 2 = 2
        else if (mu == 3 && nu == 3) return elems[entry::e33]; // 3 & 3 = 3 // if mu == nu then mu
        
        else if (mu == 0 && nu == 1) return elems[entry::e01]; // 0 & 1 = 4
        else if (mu == 0 && nu == 2) return elems[entry::e02]; // 0 & 2 = 5
        else if (mu == 0 && nu == 3) return elems[entry::e03]; // 0 & 3 = 6 // if mu == 0 or nu == 0, then the other one +3
        else if (mu == 1 && nu == 2) return elems[entry::e12]; // 1 & 2 = 7
        else if (mu == 1 && nu == 3) return elems[entry::e13]; // 1 & 3 = 8 
        else if (mu == 2 && nu == 3) return elems[entry::e23]; // 2 & 3 = 9 // mu + nu + 4
        
        else if (nu == 0 && mu == 1) return elems[entry::e01]; // 1 & 0 = 4
        else if (nu == 0 && mu == 2) return elems[entry::e02]; // 2 & 0 = 5
        else if (nu == 0 && mu == 3) return elems[entry::e03]; // 3 & 0 = 6 // if mu == 0 or nu == 0, then the other one +3
        else if (nu == 1 && mu == 2) return elems[entry::e12]; // 2 & 1 = 7
        else if (nu == 1 && mu == 3) return elems[entry::e13]; // 3 & 1 = 8
        else if (nu == 2 && mu == 3) return elems[entry::e23]; // 3 & 2 = 9 // mu + nu + 4
        return 0;
    }

    __device__ __host__ inline void operator()(int mu, int nu, floatT value) {
        if (mu == 0 && nu == 0) elems[entry::e00] = value;
        else if (mu == 1 && nu == 1) elems[entry::e11] = value;
        else if (mu == 2 && nu == 2) elems[entry::e22] = value;
        else if (mu == 3 && nu == 3) elems[entry::e33] = value;

        else if (mu == 0 && nu == 1) elems[entry::e01] = value;
        else if (mu == 0 && nu == 2) elems[entry::e02] = value;
        else if (mu == 0 && nu == 3) elems[entry::e03] = value;
        else if (mu == 1 && nu == 2) elems[entry::e12] = value;
        else if (mu == 1 && nu == 3) elems[entry::e13] = value;
        else if (mu == 2 && nu == 3) elems[entry::e23] = value;

        else if (nu == 0 && mu == 1) elems[entry::e01] = value;
        else if (nu == 0 && mu == 2) elems[entry::e02] = value;
        else if (nu == 0 && mu == 3) elems[entry::e03] = value;
        else if (nu == 1 && mu == 2) elems[entry::e12] = value;
        else if (nu == 1 && mu == 3) elems[entry::e13] = value;
        else if (nu == 2 && mu == 3) elems[entry::e23] = value;
    }

   /* __device__ __host__ inline Matrix4x4SymComplex<floatT>& operator=(const floatT &y)
    {
        for(int i = 0; i<10;i++){
            elems[i]=y;
        }
        return *this;
    }*/
    __device__ __host__ inline Matrix4x4SymComplex<floatT>& operator=(const Matrix4x4SymComplex<floatT> &y)
    {
        for(int i = 0; i<10;i++){
            elems[i]=y.elems[i];
        }
        return *this;
    }

    __device__ __host__ inline Matrix4x4SymComplex<floatT>& operator=(const Matrix4x4Sym<floatT> &y)
    {
        for(int i = 0; i<10; i++) {
            COMPLEX(floatT) tmp = y.elems[i]; // not needed
            elems[i] = tmp;
            // elems[i] = y.elems[i];
        }
        return *this;
    }

    __device__ __host__ inline Matrix4x4SymComplex<floatT>& operator+=(const Matrix4x4SymComplex<floatT> &y)
    {
        for(int i = 0; i<10;i++){
            elems[i]+=y.elems[i];
        }
        return *this;
    }

    __device__ __host__ inline Matrix4x4SymComplex<floatT>& operator-=(const Matrix4x4SymComplex<floatT> &y)
    {
        for(int i = 0; i<10;i++){
            elems[i]-=y.elems[i];
        }
        return *this;
    }

    __device__ __host__ inline Matrix4x4SymComplex<floatT>& operator/=(floatT y)
    {
        for(int i = 0; i<10;i++){
            elems[i]/=y;
        }
        return *this;
    }

    __device__ __host__ inline Matrix4x4SymComplex<floatT>& operator*=(floatT y)
    {
        for(int i = 0; i<10;i++){
            elems[i]*=y;
        }
        return *this;
    }

    __device__ __host__ friend Matrix4x4SymComplex<floatT> operator-(
        const Matrix4x4SymComplex<floatT> &left,
        const Matrix4x4SymComplex<floatT> &right
    ) {
        Matrix4x4SymComplex<floatT> result;
        for (int i = 0; i < 10; i++) {
            result.elems[i] = left.elems[i] - right.elems[i];
        }
        return result;
    }

    __host__ inline void printIndexPair() {
        std::cout << std::scientific << std::showpos << std::setprecision(8);
        std::cout << "Components of Matrix4x4SymComplex:" << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << "tensor[" << i << "]=" << elems[i] << std::endl;
        }
    }

    __host__ inline void printMatrix4x4Triangle() {
        std::cout << std::scientific << std::showpos << std::setprecision(4);
        std::cout << "Components of Matrix4x4SymComplex:" << std::endl;
        for (int mu = 0; mu <= 3; mu++) {
            for (int nu = 0; nu <= mu; nu++) {
                int indexPair = twoIndicesToIndexPairIndex(mu, nu);
                std::cout << elems[indexPair];
            }
            std::cout << std::endl;
        }
    }

    __host__ inline void printMatrix4x4Full() {
        std::cout << std::scientific << std::showpos << std::setprecision(4);
        std::cout << "Components of Matrix4x4SymComplex:" << std::endl;
        for (int mu = 0; mu <= 3; mu++) {
            for (int nu = 0; nu <= 3; nu++) {
                int indexPair = twoIndicesToIndexPairIndex(mu, nu);
                std::cout << elems[indexPair];
            }
            std::cout << std::endl;
        }
    }

};


template<class floatT>
__device__ __host__ inline Matrix4x4SymComplex<floatT> operator+(const Matrix4x4SymComplex<floatT> &x, const Matrix4x4SymComplex<floatT> &y) {
    return Matrix4x4SymComplex<floatT>(
        x.elems[0]+ y.elems[0], x.elems[1]+y.elems[1], x.elems[2]+y.elems[2], x.elems[3]+y.elems[3],
        x.elems[4]+y.elems[4], x.elems[5]+y.elems[5], x.elems[6]+y.elems[6],
        x.elems[7]+y.elems[7], x.elems[8]+y.elems[8], x.elems[9]+y.elems[9]
    );
}


template<class floatT>
__device__ __host__ inline floatT abs(const Matrix4x4SymComplex<floatT> &matrix) {
    floatT square;
    for (int i = 0; i < 10; i++) {
        square += abs2(matrix.elems[i]);
    }
    return sqrtf(square);
}


template<class floatT>
__host__ inline std::ostream &operator<<(std::ostream &s, Matrix4x4SymComplex<floatT> matrix) {
    for (int i = 0; i < 10; i++) {
        s << matrix.elems[i];
    }
    return s;
}


template<class floatT>
__device__ __host__ inline Matrix4x4SymComplex<floatT> elementwise_division(
    const Matrix4x4SymComplex<floatT> &x,
    const Matrix4x4SymComplex<floatT> &y
) {
    Matrix4x4SymComplex<floatT> result;
    for (int i = 0; i < 10; i++) {
        result.elems[i] = x.elems[i]/y.elems[i];
    }
    return result;
}

template<class floatT>
__device__ __host__ inline bool cmp_all_elements_prec(
    const Matrix4x4SymComplex<floatT> &x,
    const Matrix4x4SymComplex<floatT> &y,
    const floatT prec
) {
    for (int i = 0; i < 10; i++) {
        if (!compareCOMPLEX(x.elems[i], y.elems[i], prec)) return false;
    }
    return true;
}


template<class floatT>
__device__ __host__ inline bool compareMatrix4x4SymComplex(Matrix4x4SymComplex<floatT> a, Matrix4x4SymComplex<floatT> b, floatT tol) {
    for (int i = 0; i < 10; i++) {
        if (!compareCOMPLEX(a.elems[i], b.elems[i], tol)) return false;
    }
    return true;
}
