#pragma once
#include "../define.h"
#include "../explicit_instantiation_helper.h"

template<typename floatT, size_t HaloDepth>
class TestClass1 {
  public:
    floatT add(floatT left, floatT right);  
};

class TestClass2 {
  public:
    template<typename floatT>
    floatT sub(floatT left, floatT right);  
};



template<typename floatT, size_t HaloDepth>
floatT testFunc(floatT num);

