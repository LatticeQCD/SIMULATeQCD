#pragma once
#include "../spinor/spinorfield.h"

template<class floatT, bool onDevice, size_t HaloDepth>
using FullSpinorfield = Spinorfield<floatT, onDevice, All, HaloDepth, 12, 1>;
