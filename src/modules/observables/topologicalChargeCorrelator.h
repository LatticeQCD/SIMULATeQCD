//
// created by Jonas Winter on 09.10.2025
//

#pragma once
#include "../../define.h"
#include "../../base/latticeContainer.h"
#include "../../gauge/gaugefield.h"
#include "../../experimental/fourierNon2.h"
#include "../tensor_decomposition/tensorDecomposition.h"


template<class floatT>
struct XTimesXStar {

    LatticeContainerAccessor _firstAccessor;
    LatticeContainerAccessor _secondAccessor;
    typedef GIndexer<All> GInd;

    XTimesXStar(LatticeContainerAccessor firstAccessor, LatticeContainerAccessor secondAccessor) :
        _firstAccessor(firstAccessor), _secondAccessor(secondAccessor) {}

    __device__ __host__ inline COMPLEX(floatT) operator()(gSite site) {

        COMPLEX(floatT) firstElement(_firstAccessor.getElement<COMPLEX(floatT)>(site));
        COMPLEX(floatT) secondElement(_secondAccessor.getElement<COMPLEX(floatT)>(site));

        COMPLEX(floatT) result = firstElement * conj(secondElement);

        return result;
    }

};


// template<class floatT>
// struct DivideBySqrtVolume {

//     LatticeContainerAccessor accessor;
//     typedef GIndexer<All> GInd;
//     int globvol4 = GInd::getLatData().globvol4;

//     DivideBySqrtVolume(LatticeContainerAccessor firstAccessor) : accessor(firstAccessor) {}

//     __device__ __host__ inline COMPLEX(floatT) operator()(gSite site) {

//         COMPLEX(floatT) element(accessor.getElement<COMPLEX(floatT)>(site));

//         return element / sqrt(globvol4);
//     }

// };


template<class floatT, size_t HaloDepth>
class TopologicalChargeCorrelator {
    protected:
        FourierClass<floatT> fourierClass;
        TensorDecomposition<floatT, HaloDepth> tensorDecomposition;
    private:
        typedef GIndexer<All, HaloDepth> GInd;
    public:
        TopologicalChargeCorrelator(CommunicationBase& commBase) :
            fourierClass(commBase), tensorDecomposition(commBase) {}
        
        ~TopologicalChargeCorrelator() {}

        void checkLatticeExtents() {
            fourierClass.checkLatticeExtents();
        }

        int getR2max() {
            return tensorDecomposition.getR2max();
        }

        int getTauMax() {
            return tensorDecomposition.getTauMax();
        }

        void getR2Counts(
            std::vector<int>& counts
        );

        template<bool improved, bool improved_O6>
        void topChargeCorrGeneralTau(
            Gaugefield<floatT, true, HaloDepth>& gaugefield,
            std::vector<std::vector<COMPLEX(floatT)>>& array,
            bool matsubara
        );
};

template<class floatT, size_t HaloDepth>
template<bool improved, bool improved_O6>
void TopologicalChargeCorrelator<floatT, HaloDepth>::topChargeCorrGeneralTau(
    Gaugefield<floatT, true, HaloDepth>& gaugefield,
    std::vector<std::vector<COMPLEX(floatT)>>& array,
    bool matsubara
) {
    StopWatch<true> timer;

    // create helper lattice containers
    LatticeContainer<true, COMPLEX(floatT)> topChargeFouriered(gaugefield.getComm(), "topChargeFT", "topChargeFT", "topChargeFT", "topChargeFT");
    LatticeContainer<true, COMPLEX(floatT)> GFieldDevice(gaugefield.getComm(), "GFieldDevice", "GFieldDevice", "GFieldDevice", "GFieldDevice");
    LatticeContainer<false, COMPLEX(floatT)> GFieldHost(gaugefield.getComm(), "GFieldHost", "GFieldHost", "GFieldHost", "GFieldHost");
    topChargeFouriered.adjustSize(GInd::getLatData().vol4);
    GFieldDevice.adjustSize(GInd::getLatData().vol4);
    GFieldHost.adjustSize(GInd::getLatData().vol4);

    // calculate Q-Top
    topChargeFouriered.template iterateOverBulk<All, HaloDepth>(topChargeDensKernelComplex<floatT, HaloDepth, true, improved, improved_O6>(gaugefield));
    
    // FFT it, store it in the same Container
    timer.start();
    fourierClass.template performFourierTransformPolymorph<COMPLEX(floatT), SpatialTemporal::Both>(topChargeFouriered, topChargeFouriered, 1.0);
    timer.stop();
    rootLogger.debug("   Q Fourier took            ", timer.seconds(), "s.");
    
    timer.reset();
    
    // create product out of the two FFTed EMTs, store it in G
    timer.start();
    GFieldDevice.template iterateOverBulk<All, HaloDepth>(XTimesXStar<floatT>(topChargeFouriered.getAccessor(), topChargeFouriered.getAccessor()));
    timer.stop();
    rootLogger.debug("   Q times Q took            ", timer.seconds(), "s.");
    
    timer.reset();
    
    // FFT the product back, store it in G again
    timer.start();
    if (matsubara) {
        fourierClass.template performFourierTransformPolymorph<COMPLEX(floatT), SpatialTemporal::Spatial>(GFieldDevice, GFieldDevice, -1.0);
    } else {
        fourierClass.template performFourierTransformPolymorph<COMPLEX(floatT), SpatialTemporal::Both>(GFieldDevice, GFieldDevice, -1.0);
    }
    timer.stop();
    rootLogger.debug("   G Fourier took            ", timer.seconds(), "s.");
    
    timer.reset();
    
    // divide by sqrt of the 4D global volume as
    timer.start();
    floatT divisor;
    if (matsubara) {
        divisor = sqrt(GInd::getLatData().vol3);
    } else {
        divisor = sqrt(GInd::getLatData().vol4);
    }
    GFieldDevice.template iterateOverBulk<All, HaloDepth>(DivideByValue<COMPLEX(floatT), floatT>(GFieldDevice.getAccessor(), divisor));
    timer.stop();
    rootLogger.debug("   G normalization took      ", timer.seconds(), "s.");
    
    // copy to host for upcoming reduction
    GFieldHost.copyFromLatticeContainer(GFieldDevice);
    
    // reduce to r^2-dependent function and store in array
    GFieldHost.reduceR2generalTau(array, getTauMax(), getR2max());

}
