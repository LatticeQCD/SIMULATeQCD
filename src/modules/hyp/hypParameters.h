//
// Created by D. Hoying on 09/20/21.
//

#pragma once

#include "../../define.h"
template <class floatT>
struct HypSmearingParameters{
  //HypSmearingParameters(){}
    floatT alpha_1;
    floatT alpha_2;
    floatT alpha_3;
    HypSmearingParameters() :
        alpha_1(0.75),
        alpha_2(0.6),
        alpha_3(0.3) {}
};