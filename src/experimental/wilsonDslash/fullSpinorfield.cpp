#include "fullSpinorfield.h"
#include "../../spinor/spinorfield.cpp"

#define INIT_FULLSPINOR(floatT,HALO) \
template class Spinorfield<floatT,true,All,HALO,12,1>;\
template class Spinorfield<floatT,false,All,HALO,12,1>;\
template class Spinorfield<floatT,true,Even,HALO,12,1>;\
template class Spinorfield<floatT,false,Even,HALO,12,1>;\
template class Spinorfield<floatT,true,Odd,HALO,12,1>;\
template class Spinorfield<floatT,false,Odd,HALO,12,1>;\

INIT_PH(INIT_FULLSPINOR)
