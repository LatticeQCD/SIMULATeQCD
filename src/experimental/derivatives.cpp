#include "derivatives.h"

#ifdef USE_HIP_AMD
#define BLOCKSIZE 64
#else
#define BLOCKSIZE 32
#endif


template<typename floatT, bool onDevice, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void applyDmu(Gaugefield<floatT, onDevice, HaloDepthGauge, R18> & _gauge, Spinorfield<floatT, true, All, HaloDepthSpin, 12, NStacks>& spinorOut, Spinorfield<floatT, true, All, HaloDepthSpin, 12, NStacks>& spinorIn,COMPLEX(floatT) phase){

    spinorOut.template iterateOverBulk<BLOCKSIZE>(Dmu<floatT,HaloDepthGauge,HaloDepthSpin,NStacks>(_gauge, spinorIn,phase));

    bool update = true;
    if(update)
        spinorOut.updateAll();
}


template<typename floatT, bool onDevice, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks,int dir>
void applyDmu_Individual(Gaugefield<floatT, onDevice, HaloDepthGauge, R18> & _gauge, Spinorfield<floatT, true, All, HaloDepthSpin, 12, NStacks>& spinorOut, Spinorfield<floatT, true, All, HaloDepthSpin, 12, NStacks>& spinorIn,
	                 COMPLEX(floatT) phase){

    spinorOut.template iterateOverBulk<BLOCKSIZE>(Dmu_Individual<floatT,HaloDepthGauge,HaloDepthSpin,NStacks,dir>(_gauge, spinorIn,phase));

    bool update = true;
    if(update)
        spinorOut.updateAll();
}


template<typename floatT, bool onDevice, size_t HaloDepthGauge>
COMPLEX(double) calc_emt_Fmunu(Gaugefield<floatT, onDevice, HaloDepthGauge, R18> & _gauge){

        LatticeContainer<onDevice,COMPLEX(double)> _redBase(_gauge.getComm());

        typedef GIndexer<All, HaloDepthGauge> GInd;        

        COMPLEX(double) result = 0;

        size_t elems_ = GInd::getLatData().vol4;
        
        _redBase.adjustSize(elems_);

        _redBase.template iterateOverBulk<All, HaloDepthGauge>(emt_Fmunu<floatT,HaloDepthGauge>(_gauge));

        _redBase.reduce(result, elems_);
        return result;
}

template<typename floatT, bool onDevice, size_t HaloDepthGauge,int dir>
COMPLEX(double) calc_emt_Fmunu_Individual(Gaugefield<floatT, onDevice, HaloDepthGauge, R18> & _gauge){

        LatticeContainer<onDevice,COMPLEX(double)> _redBase(_gauge.getComm());

        typedef GIndexer<All, HaloDepthGauge> GInd;

        COMPLEX(double) result = 0;

        size_t elems_ = GInd::getLatData().vol4;

        _redBase.adjustSize(elems_);

        _redBase.template iterateOverBulk<All, HaloDepthGauge>(emt_Fmunu_Individual<floatT,HaloDepthGauge,dir>(_gauge));

        _redBase.reduce(result, elems_);
        return result;
}





template void applyDmu<double,true,2,2,12>(Gaugefield<double, true, 2, R18> & _gauge,
	                 Spinorfield<double, true, All, 2, 12, 12>& spinorOut, Spinorfield<double, true, All, 2, 12, 12>& spinorIn,COMPLEX(double) phase);

template void applyDmu_Individual<double,true,2,2,12,0>(Gaugefield<double, true, 2, R18> & _gauge,
                         Spinorfield<double, true, All, 2, 12, 12>& spinorOut, Spinorfield<double, true, All, 2, 12, 12>& spinorIn,COMPLEX(double) phase);
template void applyDmu_Individual<double,true,2,2,12,1>(Gaugefield<double, true, 2, R18> & _gauge,
                         Spinorfield<double, true, All, 2, 12, 12>& spinorOut, Spinorfield<double, true, All, 2, 12, 12>& spinorIn,COMPLEX(double) phase);
template void applyDmu_Individual<double,true,2,2,12,2>(Gaugefield<double, true, 2, R18> & _gauge,
                         Spinorfield<double, true, All, 2, 12, 12>& spinorOut, Spinorfield<double, true, All, 2, 12, 12>& spinorIn,COMPLEX(double) phase);
template void applyDmu_Individual<double,true,2,2,12,3>(Gaugefield<double, true, 2, R18> & _gauge,
                         Spinorfield<double, true, All, 2, 12, 12>& spinorOut, Spinorfield<double, true, All, 2, 12, 12>& spinorIn,COMPLEX(double) phase);


template COMPLEX(double) calc_emt_Fmunu(Gaugefield<double, true, 2, R18> & _gauge);

template COMPLEX(double) calc_emt_Fmunu_Individual<double,true,2,0>(Gaugefield<double, true, 2, R18> & _gauge);
template COMPLEX(double) calc_emt_Fmunu_Individual<double,true,2,1>(Gaugefield<double, true, 2, R18> & _gauge);
