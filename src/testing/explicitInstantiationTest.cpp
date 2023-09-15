#include "explicitInstantiationTest.h"


template<typename floatT, size_t HaloDepth>
floatT testFunc(floatT num) {
    return HaloDepth * num * num * num;
}



template<typename floatT, size_t HaloDepth>
floatT TestClass1<floatT,HaloDepth>::add(floatT left, floatT right) {
    return (left + right)*HaloDepth;
}

template<typename floatT>
floatT TestClass2::sub(floatT left, floatT right) {
    return left - right;
}


struct CustomInstantiation {
    
    template<typename floatT, typename HaloType>
    std::any operator()(floatT , HaloType halo_object) const { 
    //std::any operator()(auto prec, auto halo, auto halo2) const { // works only with C++20
    
        constexpr auto HaloDepth = halo_object.value;
        
        // Check if floatT is float
        if constexpr (std::is_same_v<floatT,float>) {
            constexpr auto ptr1  = &testFunc<floatT,HaloDepth>;
            constexpr auto ptr2  = &TestClass1<floatT,HaloDepth>::add;
            constexpr auto ptr3  = &TestClass2::sub<floatT>;
            return std::make_tuple(ptr1,ptr2,ptr3);
        }
        else{
            // else return empty tuple
            return std::tuple();
        }
    }

};

template struct Instantiate<CustomInstantiation,PRECISION_VARIANTS,HALO_VARIANTS>; 

