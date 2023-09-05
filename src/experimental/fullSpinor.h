#pragma once

#include "../base/math/vect3.h"
#include "../base/math/vectArray.h"
#include "gammaMatrix.h"
#include <array>

template <class floatT> 
using Vect12 = Vect<floatT,12>;

template <class floatT> 
using ColorVect = std::array<Vect3<floatT>,4>;

template<class floatT>
using Vect12ArrayAcc = VectArrayAcc<floatT,12>;

template<class floatT, bool onDevice>
using Vect12Array = VectArray<floatT,12,onDevice>;


template <class floatT>
__host__ __device__ auto operator+(const ColorVect<floatT>& ob1, const ColorVect<floatT>& ob2)->ColorVect<floatT> {
    ColorVect<floatT> res;
    for (int i = 0; i < 4; ++i)
        res[i] = ob1[i] + ob2[i];
    return res; 
}

template <class floatT>
__host__ __device__ auto operator*(const SU3<floatT>& ob1, const ColorVect<floatT>& ob2)->ColorVect<floatT> {
    ColorVect<floatT> res;
    for (int i = 0; i < 4; ++i)
        res[i] = ob1 * ob2[i];
    return res; 
}

template <class floatT>
__host__ __device__ auto operator*(const FourMatrix<floatT>& matrix, const ColorVect<floatT>& colVec)->ColorVect<floatT> {
  ColorVect<floatT> res;
  for (int i = 0; i < 4; ++i){
    for (int j = 0 ; j < 4 ; ++j){ 
        GPUcomplex<floatT> matrix_element=matrix.get(i,j);
        res[i] += matrix_element * colVec[j];
      }
  }
  return res; 
}

template <class floatT>
__host__ __device__ auto operator*(const floatT& scalar, const ColorVect<floatT>& colVec)->ColorVect<floatT> {
  ColorVect<floatT> res;
  for (int i = 0; i < 4; ++i){
    res[i] += scalar * colVec[i];
  }
  return res; 
}

template <class floatT>
__host__ __device__ auto operator*(const GPUcomplex<floatT>& scalar, const ColorVect<floatT>& colVec)->ColorVect<floatT> {
  ColorVect<floatT> res;
  for (int i = 0; i < 4; ++i){
    res[i] += scalar * colVec[i];
  }
  return res; 
}


template<class floatT>
__host__ __device__ inline auto convertVect12ToColorVect(Vect12<floatT> &vec){
    ColorVect<floatT> res = {Vect3<floatT>(vec.template getElement<0>(),
                           vec.template getElement<1>(),
                           vec.template getElement<2>()),
                                                 
                     Vect3<floatT>(vec.template getElement<3>(),
                           vec.template getElement<4>(),
                           vec.template getElement<5>()),
                                                 
                     Vect3<floatT>(vec.template getElement<6>(),
                           vec.template getElement<7>(),
                           vec.template getElement<8>()),
                                                 
                     Vect3<floatT>(vec.template getElement<9>(),
                           vec.template getElement<10>(),
                           vec.template getElement<11>())};
    return res;
}

template<class floatT>
__host__ __device__ inline auto convertColorVectToVect12(ColorVect<floatT> &vec){
    return Vect12<floatT>(vec[0].template getElement<0>(),
                  vec[0].template getElement<1>(),
                  vec[0].template getElement<2>(),
                                           
                  vec[1].template getElement<0>(),
                  vec[1].template getElement<1>(),
                  vec[1].template getElement<2>(),
                                           
                  vec[2].template getElement<0>(),
                  vec[2].template getElement<1>(),
                  vec[2].template getElement<2>(),
                                           
                  vec[3].template getElement<0>(),
                  vec[3].template getElement<1>(),
                  vec[3].template getElement<2>());
}


template<class floatT>
struct SpinorColorAcc : public VectArrayAcc<floatT,12> {

    SpinorColorAcc(VectArrayAcc<floatT,12> vecAcc )
            : VectArrayAcc<floatT, 12>(vecAcc) { }



    __host__ __device__ inline ColorVect<floatT> getColorVect(const gSite &site) const {
        auto res = this->template getElement(site);
        return convertVect12ToColorVect(res);
    }

    __host__ __device__ inline void setColorVect(const gSite &site, const ColorVect<floatT> &vec) {
        auto res = convertColorVectToVect12(vec);
        this->template setElement(site, res);

    }

};



