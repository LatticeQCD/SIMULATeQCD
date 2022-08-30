//
// Created by Lukas Mazur on 29.11.18.
//

#ifndef MATRIX4X4_H
#define MATRIX4X4_H

#include "../../define.h"


// Symmetric 4x4 Matrix
template<class floatT>
struct Matrix4x4Sym {
    enum entry {
        e00, e11, e22, e33, e01, e02, e03, e12, e13, e23
    };
    floatT elems[10];


    constexpr Matrix4x4Sym(const Matrix4x4Sym<floatT>&) = default;
    __host__ __device__ Matrix4x4Sym(floatT a) : elems{a, a, a, a, a, a, a, a, a, a} {}
    __host__ __device__ Matrix4x4Sym() : elems{0, 0, 0, 0, 0, 0, 0, 0, 0, 0} {}

    __host__ __device__ Matrix4x4Sym(floatT e00, floatT e11, floatT e22, floatT e33, floatT e01, floatT e02, floatT e03, floatT e12,
                                     floatT e13, floatT e23) :
            elems{e00, e11, e22, e33, e01, e02, e03, e12, e13, e23} {}

    __host__ __device__ inline floatT operator()(int mu, int nu) {
        if (mu == 0 && nu == 0) return elems[entry::e00];
        if (mu == 1 && nu == 1) return elems[entry::e11];
        if (mu == 2 && nu == 2) return elems[entry::e22];
        if (mu == 3 && nu == 3) return elems[entry::e33];

        if (mu == 0 && nu == 1) return elems[entry::e01];
        if (mu == 0 && nu == 2) return elems[entry::e02];
        if (mu == 0 && nu == 3) return elems[entry::e03];
        if (mu == 1 && nu == 2) return elems[entry::e12];
        if (mu == 1 && nu == 3) return elems[entry::e13];
        if (mu == 2 && nu == 3) return elems[entry::e23];

        if (nu == 0 && mu == 1) return elems[entry::e01];
        if (nu == 0 && mu == 2) return elems[entry::e02];
        if (nu == 0 && mu == 3) return elems[entry::e03];
        if (nu == 1 && mu == 2) return elems[entry::e12];
        if (nu == 1 && mu == 3) return elems[entry::e13];
        if (nu == 2 && mu == 3) return elems[entry::e23];
        return 0;
    }

    __host__ __device__ inline void operator()(int mu, int nu, floatT value) {
        if (mu == 0 && nu == 0) elems[entry::e00] = value;
        if (mu == 1 && nu == 1) elems[entry::e11] = value;
        if (mu == 2 && nu == 2) elems[entry::e22] = value;
        if (mu == 3 && nu == 3) elems[entry::e33] = value;

        if (mu == 0 && nu == 1) elems[entry::e01] = value;
        if (mu == 0 && nu == 2) elems[entry::e02] = value;
        if (mu == 0 && nu == 3) elems[entry::e03] = value;
        if (mu == 1 && nu == 2) elems[entry::e12] = value;
        if (mu == 1 && nu == 3) elems[entry::e13] = value;
        if (mu == 2 && nu == 3) elems[entry::e23] = value;

        if (nu == 0 && mu == 1) elems[entry::e01] = value;
        if (nu == 0 && mu == 2) elems[entry::e02] = value;
        if (nu == 0 && mu == 3) elems[entry::e03] = value;
        if (nu == 1 && mu == 2) elems[entry::e12] = value;
        if (nu == 1 && mu == 3) elems[entry::e13] = value;
        if (nu == 2 && mu == 3) elems[entry::e23] = value;
    }

   /* __host__ __device__ inline Matrix4x4Sym<floatT>& operator=(const floatT &y)
    {
        for(int i = 0; i<10;i++){
            elems[i]=y;
        }
        return *this;
    }*/
    __host__ __device__ inline Matrix4x4Sym<floatT>& operator=(const Matrix4x4Sym<floatT> &y)
    {
        for(int i = 0; i<10;i++){
            elems[i]=y.elems[i];
        }
        return *this;
    }
    __host__ __device__ inline Matrix4x4Sym<floatT>& operator+=(const Matrix4x4Sym<floatT> &y)
    {

        for(int i = 0; i<10;i++){
            elems[i]+=y.elems[i];
        }
        return *this;
    }

    __host__ __device__ inline Matrix4x4Sym<floatT>& operator/=(floatT y)
    {
        for(int i = 0; i<10;i++){
            elems[i]/=y;
        }
        return *this;
    }

    __host__ __device__ inline Matrix4x4Sym<floatT>& operator*=(floatT y)
    {
        for(int i = 0; i<10;i++){
            elems[i]*=y;
        }
        return *this;
    }

};


template<class floatT>
__host__ __device__ inline Matrix4x4Sym<floatT> operator+(const Matrix4x4Sym<floatT> &x, const Matrix4x4Sym<floatT> &y) {
    return Matrix4x4Sym<floatT>(x.elems[0]+ y.elems[0], x.elems[1]+y.elems[1], x.elems[2]+y.elems[2], x.elems[3]+y.elems[3],
                                x.elems[4]+y.elems[4], x.elems[5]+y.elems[5], x.elems[6]+y.elems[6],
                                x.elems[7]+y.elems[7], x.elems[8]+y.elems[8], x.elems[9]+y.elems[9]);
}
#endif //MATRIX4X4_H
