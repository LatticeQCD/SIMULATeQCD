/*
 * MDWF 5D spinor scaffold.
 *
 * This header gives the MDWF code path a named 5D spinor representation
 * without introducing a new storage layout.  The existing Spinorfield stack
 * dimension is interpreted as the physical fifth dimension Ls only through
 * these MDWF aliases.
 */

#pragma once

#include "../../spinor/spinorfield.h"

template<size_t Ls>
struct MDWFSpinorLayout {
    static_assert(Ls > 0, "MDWF fifth dimension Ls must be positive");

    // Wilson spinors use 12 complex components per 4D site: 4 spin x 3 color.
    static constexpr size_t spin_color_components = 12;

    // In MDWF code, this is the physical fifth dimension, not independent RHS.
    static constexpr size_t fifth_dimension = Ls;
};

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t Ls>
using MDWFSpinor = Spinorfield<floatT,
                               onDevice,
                               LatLayout,
                               HaloDepth,
                               MDWFSpinorLayout<Ls>::spin_color_components,
                               MDWFSpinorLayout<Ls>::fifth_dimension>;

template<class floatT, bool onDevice, size_t HaloDepth, size_t Ls>
using MDWFSpinorAll = SpinorfieldAll<floatT,
                                     onDevice,
                                     HaloDepth,
                                     MDWFSpinorLayout<Ls>::spin_color_components,
                                     MDWFSpinorLayout<Ls>::fifth_dimension>;

template<class floatT, bool onDevice, Layout LatLayout, size_t HaloDepth, size_t Ls>
struct MDWFSpinorTypes {
    using Field = MDWFSpinor<floatT, onDevice, LatLayout, HaloDepth, Ls>;
    using EvenOddField = MDWFSpinorAll<floatT, onDevice, HaloDepth, Ls>;

    static constexpr size_t spin_color_components = MDWFSpinorLayout<Ls>::spin_color_components;
    static constexpr size_t fifth_dimension = MDWFSpinorLayout<Ls>::fifth_dimension;
};
