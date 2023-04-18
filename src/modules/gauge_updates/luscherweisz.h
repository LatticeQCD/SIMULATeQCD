//
// Created by Hai-Tao Shu on 06.05.19
//

#pragma once
#include "../../gauge/gaugefield.h"


template<class floatT, bool onDevice, size_t HaloDepth>
class LuscherWeisz {

protected:

    Gaugefield<floatT, onDevice, HaloDepth> &_gauge;

private:
    typedef GIndexer<All, HaloDepth> GInd;
    const size_t elems=GInd::getLatData().sizeh;

public:
    LuscherWeisz(Gaugefield<floatT, true, HaloDepth> &gaugefield) :
            _gauge(gaugefield){}

    void subUpdateOR(int sub_lt, int local_pos_t);
    void subUpdateHB(uint4* state, floatT beta, int sub_lt, int local_pos_t, bool ltest=false);
    // release the object (destructor)
    ~LuscherWeisz() {}
};

