#pragma once
#include "../define.h"
#include "../base/math/complex.h"
#include "../base/math/su3.h"

template <typename floatT>
struct WilsonPropagator {
    //12 column vectors, each column is 12 vector
    //12x12xvol matrix.
//    Spinorfield<floatT, true, All, HaloDepth,12> A[12];
    ColorVect<floatT> A[12];

    // Default constructor
    __host__ __device__ WilsonPropagator<floatT>() : A{} {}

    // Parameterized constructor
    __host__ __device__ WilsonPropagator<floatT>(
                       ColorVect<floatT> a00, ColorVect<floatT> a01,
                       ColorVect<floatT> a02, ColorVect<floatT> a03,
                       ColorVect<floatT> a04, ColorVect<floatT> a05,
                       ColorVect<floatT> a06, ColorVect<floatT> a07,
                       ColorVect<floatT> a08, ColorVect<floatT> a09,
                       ColorVect<floatT> a10, ColorVect<floatT> a11
                       ) 
    {
      A[0]=a00;
      A[1]=a01;
      A[2]=a02;
      A[3]=a03;
      A[4]=a04;
      A[5]=a05;
      A[6]=a06;
      A[7]=a07;
      A[8]=a08;
      A[9]=a09;
      A[10]=a10;
      A[11]=a11;
    }
/*
    // Constructor from an array of SU(3) matrices
    __host__ __device__ WilsonPropagator<floatT>(SU3<floatT> ar[16]) {

        for (int i = 0; i < 12; i++) { //column of A
            for (int j = 0; j < 4; j++) { //row of column vector A[i]
                A[i][j] = ar[j * 4 + i];
            }
        }
    }
*/
    //insert ColorVect to a column of WilsonPropagator
    __host__ __device__ void insert(ColorVect<floatT> cv, int col) const {
      A[col]=cv;
    }
    //get ColorVect to a column of WilsonPropagator
    __host__ __device__ const ColorVect<floatT> get(const int col) const {
      return A[col];
    }

    __host__ __device__ const ColorVect<floatT> point_source(const sitexyzt pointsource) const {
      //if site == pointsource
      for(int ispin = 0 ; ispin < 4 ; ++ispin){
        for(int icolor = 0 ; icolor < 3 ; ++icolor){
          //A[icolor+3*ispin][ispin][icolor]=1;
          Vect3<floatT> one = Vect3<floatT>(1.0,0.0,0.0);
          A[icolor+3*ispin][ispin]=one;//Vect3<floatT>(1.0,0.0,0.0);
        }
      }
       
      return ;
    }


/*
    // Static method to create a zero matrix --> This must be checked!
    __host__ __device__ inline static constexpr WilsonPropagator<floatT> zero() {
        return WilsonPropagator<floatT>(
          su3_zero(), su3_zero(), su3_zero(), su3_zero(),
          su3_zero(), su3_zero(), su3_zero(), su3_zero(),
          su3_zero(), su3_zero(), su3_zero(), su3_zero(),
          su3_zero(), su3_zero(), su3_zero(), su3_zero()
          );
    }

    // Static method to create an identity matrix
    __host__ __device__ inline static constexpr WilsonPropagator<floatT> identity() {
        return WilsonPropagator<floatT>(
          su3_one(), su3_zero(), su3_zero(), su3_zero(),
          su3_zero(), su3_one(), su3_zero(), su3_zero(),
          su3_zero(), su3_zero(), su3_one(), su3_zero(),
          su3_zero(), su3_zero(), su3_zero(), su3_one()
          );
; 
    }
*/
    // Addition assignment operator
    __host__ __device__ WilsonPropagator<floatT>& operator+=(const WilsonPropagator<floatT>& rhs) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                this->A[i][j] += rhs.A[i][j];
            }
        }
        return *this;
    }

    // Subtraction assignment operator
    __host__ __device__ WilsonPropagator<floatT>& operator-=(const WilsonPropagator<floatT>& rhs) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                this->A[i][j] -= rhs.A[i][j];
            }
        }
        return *this;
    }

    // Addition operator
    __host__ __device__ const WilsonPropagator<floatT> operator+(const WilsonPropagator<floatT>& rhs) const {
        WilsonPropagator<floatT> acu = *this;
        return acu += rhs;
    }

    // Subtraction operator
    __host__ __device__ const WilsonPropagator<floatT> operator-(const WilsonPropagator<floatT>& rhs) const {
        WilsonPropagator<floatT> acu = *this;
        return acu -= rhs;
    }

    // Unary minus operator
    __host__ __device__ const WilsonPropagator<floatT> operator-() const {
        WilsonPropagator<floatT> acu = WilsonPropagator<floatT>::zero();
        return (acu - (*this));
    }

    // Multiplication operator
    __host__ __device__ const WilsonPropagator<floatT> operator*(const WilsonPropagator<floatT>& b) const {
        WilsonPropagator<floatT> result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
              result.A[i][j]=0.0;
            }
        }
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

