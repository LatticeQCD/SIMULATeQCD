#pragma once

#include "explicit_instantiation_macros.h"
#include <utility>
#include <tuple>
#include <cstdint>
#include <type_traits>
#include <any>
#include <variant>

template <int value_> struct HALO_TYPE{ static const int value = value_; }; 

#define ENABLED_HALODEPTHS REMOVE_TRAILING_COMMAS( \
       IF(BOOL(HALODEPTH_0)) (HALO_TYPE<0>,) \
       IF(BOOL(HALODEPTH_1)) (HALO_TYPE<1>,) \
       IF(BOOL(HALODEPTH_2)) (HALO_TYPE<2>,) \
       IF(BOOL(HALODEPTH_3)) (HALO_TYPE<3>,) \
       IF(BOOL(HALODEPTH_4)) (HALO_TYPE<4>,) \
       )


#define ENABLED_PRECISIONS REMOVE_TRAILING_COMMAS( \
       IF(BOOL(SINGLEPREC)) (float,) \
       IF(BOOL(DOUBLEPREC)) (double,) \
       )


using HALO_VARIANTS = std::variant< ENABLED_HALODEPTHS >;
//using HALO_VARIANTS = std::variant< HALO_TYPE<0>, HALO_TYPE<1> >;


using PRECISION_VARIANTS = std::variant< ENABLED_PRECISIONS >;
//using PRECISION_VARIANTS = std::variant< float, double >;


template<class FUNCTOR, typename... Args>
constexpr auto Instantiate( Args&& ... args  ) {

    return std::visit(FUNCTOR{}, args...);
}
