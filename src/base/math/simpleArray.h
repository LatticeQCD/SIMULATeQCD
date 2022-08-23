#ifndef SIMPLE_ARRAY_H
#define SIMPLE_ARRAY_H

#include<vector>
#include "operators.h"

//Simple array class to work with the general operators and can interact with std::vector
template<typename T, size_t N>
class SimpleArray{

    T values[N];

    public:

    HOST_DEVICE T& operator[](size_t i){
        return values[i];
    }


    HOST_DEVICE inline auto operator()(gSiteStack site) const
    {
        return values[site.stack];
    }


    HOST_DEVICE inline auto operator()(gSiteMu site) const
    {
        return values[site.mu];
    }


    SimpleArray() = default;

    HOST_DEVICE SimpleArray(const T& init){
        for(size_t i = 0; i < N; i++){
            values[i] = init;
        }
    }
    template<class floatT>
        HOST_DEVICE SimpleArray(SimpleArray<floatT, N> s_array) {
        for(size_t i = 0; i < N; i++) {
            values[i] = s_array[i];
        }
    }

    HOST_DEVICE void operator=(SimpleArray<T,N> vec){
        for(size_t i = 0; i < N; i++){
            values[i] = vec[i];
        }
    }


    SQCD_HOST void operator=(std::vector<T> vec){
        for(size_t i = 0; i < N; i++){
            values[i] = vec.at(i);
        }
    }

    HOST_DEVICE SimpleArray getAccessor() const {
        return *this;
    }

};



template<typename floatT, size_t N>
HOST_DEVICE SimpleArray<floatT, N> operator/(SimpleArray<floatT, N> a, SimpleArray<floatT, N> b){
    SimpleArray<floatT, N> ret;
    for(size_t i = 0; i < N; i++){
        ret[i] = a[i] / b[i];
    }
    return ret;
}

template<typename floatT, size_t N>
HOST_DEVICE SimpleArray<floatT, N> operator*(SimpleArray<floatT, N> a, SimpleArray<floatT, N> b){
    SimpleArray<floatT, N> ret;
    for(size_t i = 0; i < N; i++){
        ret[i] = a * b;
    }
    return ret;
}

template<typename floatT, size_t N>
HOST_DEVICE SimpleArray<floatT, N> operator-(SimpleArray<floatT, N> a, SimpleArray<floatT, N> b){
    SimpleArray<floatT, N> ret;
    for(size_t i = 0; i < N; i++){
        ret[i] = a[i] - b[i];
    }
    return ret;
}

template<typename floatT, size_t N>
HOST_DEVICE SimpleArray<floatT, N> operator+(SimpleArray<floatT, N> a, SimpleArray<floatT, N> b){
    SimpleArray<floatT, N> ret;
    for(size_t i = 0; i < N; i++){
        ret[i] = a[i] + b[i];
    }
    return ret;
}

template<typename floatT, size_t N>
HOST_DEVICE SimpleArray<floatT, N> operator*(floatT a, SimpleArray<floatT, N> b){
    SimpleArray<floatT, N> ret;
    for(size_t i = 0; i < N; i++){
        ret[i] = a * b[i];
    }
    return ret;
}

template<typename floatT, size_t N>
    HOST_DEVICE SimpleArray<floatT, N> operator/(SimpleArray<floatT, N> a, floatT b){
    SimpleArray<floatT, N> ret;
    for (size_t i = 0; i < N; i++) {
        ret[i] = a[i]/b;
    }
    return ret;
}

template<typename floatT, size_t N>
HOST_DEVICE floatT max(SimpleArray<floatT, N> a){
    floatT ret = a[0];
    for(size_t i = 1; i < N; i++){
        if (a[i] > ret){
            ret = a[i];
        }
    }
    return ret;
}

template<typename floatTret, typename floatT, size_t N>
HOST_DEVICE SimpleArray<floatTret, N> real(SimpleArray<floatT, N> c){
    SimpleArray<floatTret, N> ret;
    for(size_t i = 0; i < N; i++){
        ret[i] = c[i].cREAL;
    }
    return ret;
}



template<typename typeLHS1, typename typeRHS1, Operation op1, typename T, size_t N>
GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>,
        SimpleArray<T, N>,
        add>
operator+(const GeneralOperator<typeLHS1, typeRHS1, op1> &lhs,
          const SimpleArray<T, N> rhs) {
    return GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>, SimpleArray<T, N>, add>(lhs, rhs);
}


template<typename typeLHS1, typename typeRHS1, Operation op1, typename T, size_t N>
GeneralOperator<SimpleArray<T, N>,
        GeneralOperator<typeLHS1, typeRHS1, op1>,
        add>
operator+(const SimpleArray<T, N> lhs, const GeneralOperator<typeLHS1, typeRHS1, op1> &rhs) {
    return GeneralOperator<SimpleArray<T,N>, GeneralOperator<typeLHS1, typeRHS1, op1>, add>(lhs, rhs);
}

template<typename typeLHS1, typename typeRHS1, Operation op1, typename T, size_t N>
GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>,
        SimpleArray<T, N>,
        subtract>
operator-(const GeneralOperator<typeLHS1, typeRHS1, op1> &lhs,
          const SimpleArray<T, N> rhs) {
    return GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>, SimpleArray<T, N>, subtract>(lhs, rhs);
}


template<typename typeLHS1, typename typeRHS1, Operation op1, typename T, size_t N>
GeneralOperator<SimpleArray<T, N>,
        GeneralOperator<typeLHS1, typeRHS1, op1>,
        subtract>
operator-(const SimpleArray<T, N> lhs, const GeneralOperator<typeLHS1, typeRHS1, op1> &rhs) {
    return GeneralOperator<SimpleArray<T,N>, GeneralOperator<typeLHS1, typeRHS1, op1>, subtract>(lhs, rhs);
}

template<typename typeLHS1, typename typeRHS1, Operation op1, typename T, size_t N>
GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>,
        SimpleArray<T, N>,
        divide>
operator/(const GeneralOperator<typeLHS1, typeRHS1, op1> &lhs,
          const SimpleArray<T, N> rhs) {
    return GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>, SimpleArray<T, N>, divide>(lhs, rhs);
}


template<typename typeLHS1, typename typeRHS1, Operation op1, typename T, size_t N>
GeneralOperator<SimpleArray<T, N>,
        GeneralOperator<typeLHS1, typeRHS1, op1>,
        divide>
operator/(const SimpleArray<T, N> lhs, const GeneralOperator<typeLHS1, typeRHS1, op1> &rhs) {
    return GeneralOperator<SimpleArray<T,N>, GeneralOperator<typeLHS1, typeRHS1, op1>, divide>(lhs, rhs);
}



template<typename typeLHS1, typename typeRHS1, Operation op1, typename T, size_t N>
GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>,
        SimpleArray<T, N>,
        mult>
operator*(const GeneralOperator<typeLHS1, typeRHS1, op1> &lhs,
          const SimpleArray<T, N> rhs) {
    return GeneralOperator<GeneralOperator<typeLHS1, typeRHS1, op1>, SimpleArray<T, N>, mult>(lhs, rhs);
}


template<typename typeLHS1, typename typeRHS1, Operation op1, typename T, size_t N>
GeneralOperator<SimpleArray<T, N>,
        GeneralOperator<typeLHS1, typeRHS1, op1>,
        mult>
operator*(const SimpleArray<T, N> lhs, const GeneralOperator<typeLHS1, typeRHS1, op1> &rhs) {
    return GeneralOperator<SimpleArray<T,N>, GeneralOperator<typeLHS1, typeRHS1, op1>, mult>(lhs, rhs);
}


#endif
