#pragma once
#include "../define.h"
#include "../base/math/complex.h"

template <typename floatT>
struct FourMatrix {

    GPUcomplex<floatT> A[4][4];

    // Default constructor
    __host__ __device__ FourMatrix<floatT>() : A{} {}

    // Parameterized constructor
    __host__ __device__ FourMatrix<floatT>(GPUcomplex<floatT> a00, GPUcomplex<floatT> a01,
                       GPUcomplex<floatT> a02, GPUcomplex<floatT> a03,
                       GPUcomplex<floatT> a10, GPUcomplex<floatT> a11,
                       GPUcomplex<floatT> a12, GPUcomplex<floatT> a13,
                       GPUcomplex<floatT> a20, GPUcomplex<floatT> a21,
                       GPUcomplex<floatT> a22, GPUcomplex<floatT> a23,
                       GPUcomplex<floatT> a30, GPUcomplex<floatT> a31,
                       GPUcomplex<floatT> a32, GPUcomplex<floatT> a33) 
    {
      A[0][0]=a00;
      A[0][1]=a01;
      A[0][2]=a02;
      A[0][3]=a03;

      A[1][0]=a10;
      A[1][1]=a11;
      A[1][2]=a12;
      A[1][3]=a13;
    
      A[2][0]=a20;
      A[2][1]=a21;
      A[2][2]=a22;
      A[2][3]=a23;
      
      A[3][0]=a30;
      A[3][1]=a31;
      A[3][2]=a32;
      A[3][3]=a33;
    }

    // Constructor from an array of complex numbers
    __host__ __device__ FourMatrix<floatT>(GPUcomplex<floatT> ar[16]) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                A[j][i] = ar[j * 4 + i];
            }
        }
    }

    // Static method to create a zero matrix
    __host__ __device__ inline static constexpr FourMatrix<floatT> zero() {
        return {};
    }

    // Static method to create an identity matrix
    __host__ __device__ inline static constexpr FourMatrix<floatT> identity() {
      FourMatrix<floatT> I;
      I.A[0][0]=1;
      I.A[1][1]=1;
      I.A[2][2]=1;
      I.A[3][3]=1;
      return I; 
    }

    // Static method to create a gamma matrix
    __host__ __device__ static const FourMatrix<floatT> gamma(int mu) {
        const GPUcomplex<floatT> i(0,1);
        switch (mu) {
            case 0:
                return FourMatrix<floatT>(
                        0, 0, 0, i,
                        0, 0, i, 0,
                        0,-i, 0, 0,
                        -i, 0, 0, 0);
            case 1:
                return FourMatrix<floatT>(
                        0, 0, 0, 1,
                        0, 0,-1, 0,
                        0,-1, 0, 0,
                        1, 0, 0, 0);
            case 2:
                return FourMatrix<floatT>(
                        0, 0, i, 0,
                        0, 0, 0,-i,
                        -i, 0, 0, 0,
                        0, i, 0, 0);
            case 3:
                return FourMatrix<floatT>(
                        1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0,-1, 0,
                        0, 0, 0,-1);
            case 5:
                return FourMatrix<floatT>(
                        0, 0, 1, 0,
                        0, 0, 0, 1,
                        1, 0, 0, 0,
                        0, 1, 0, 0);
            default:
                return zero();
        }
    }


    // Addition assignment operator
    __host__ __device__ FourMatrix<floatT>& operator+=(const FourMatrix<floatT>& rhs) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                this->A[i][j] += rhs.A[i][j];
            }
        }
        return *this;
    }

    // Subtraction assignment operator
    __host__ __device__ FourMatrix<floatT>& operator-=(const FourMatrix<floatT>& rhs) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                this->A[i][j] -= rhs.A[i][j];
            }
        }
        return *this;
    }

    // Addition operator
    __host__ __device__ const FourMatrix<floatT> operator+(const FourMatrix<floatT>& rhs) const {
        FourMatrix<floatT> acu = *this;
        return acu += rhs;
    }

    // Subtraction operator
    __host__ __device__ const FourMatrix<floatT> operator-(const FourMatrix<floatT>& rhs) const {
        FourMatrix<floatT> acu = *this;
        return acu -= rhs;
    }

    // Unary minus operator
    __host__ __device__ const FourMatrix<floatT> operator-() const {
        FourMatrix<floatT> acu = FourMatrix<floatT>::zero();
        return (acu - (*this));
    }

    // Multiplication operator
    __host__ __device__ const FourMatrix<floatT> operator*(const FourMatrix<floatT>& b) const {
        FourMatrix<floatT> result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    result.A[i][j] += A[i][k] * b.A[k][j];
                }
            }
        }
        return result;
    }
    __host__ __device__ const GPUcomplex<floatT> get(const int i, const int j) const {
      return A[i][j];
    }
};

