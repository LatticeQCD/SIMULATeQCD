//
// created by Jonas Winter on 22.10.2025
//

#pragma once
#include "../../define.h"
#include "../../base/stopWatch.h"
#include "../../base/latticeContainer.h"
#include "../../gauge/gaugefield.h"
#include "../../experimental/fourierNon2.h"
#include "../../base/math/matrix4x4SymComplex.h"
#include "../../base/math/tensor4x4Symx4x4SymComplex.h"
#include "../tensor_decomposition/tensorDecomposition.h"


template<class floatT, size_t HaloDepth>
class EnergyMomentumTensorCorrelator {
    protected:
        FourierClass<floatT> fourierClass;
        TensorDecomposition<floatT, HaloDepth> tensorDecomposition;
    private:
        typedef GIndexer<All, HaloDepth> GInd;
    public:
        EnergyMomentumTensorCorrelator(CommunicationBase& commBase) :
            fourierClass(commBase), tensorDecomposition(commBase) {}

        ~EnergyMomentumTensorCorrelator() {}

        int getR2max() {
            return tensorDecomposition.getR2max();
        }

        void EMTCorrGTensor(
            Gaugefield<floatT, true, HaloDepth>& gaugefield,
            LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& field
        );

        void EMTCorrGFunctions(
            Gaugefield<floatT, true, HaloDepth>& gaugefield,
            std::vector<std::vector<COMPLEX(floatT)>>& array
        );

        void getR2Counts(
            std::vector<int>& counts
        );

};


template<class floatT>
struct EMTtimesEMTStar {

    LatticeContainerAccessor _firstAccessor;
    LatticeContainerAccessor _secondAccessor;
    typedef GIndexer<All> GInd;

    EMTtimesEMTStar(LatticeContainerAccessor firstAccessor, LatticeContainerAccessor secondAccessor) :
        _firstAccessor(firstAccessor), _secondAccessor(secondAccessor) {}

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT> operator()(gSite site) {

        Matrix4x4SymComplex<floatT> firstElement(_firstAccessor.getElement<Matrix4x4SymComplex<floatT>>(site));
        Matrix4x4SymComplex<floatT> secondElement(_secondAccessor.getElement<Matrix4x4SymComplex<floatT>>(site));

        Tensor4x4Symx4x4SymComplex<floatT> result(firstElement, conj(secondElement));

        return result;
    }

};


template<class floatT, size_t HaloDepth>
void EnergyMomentumTensorCorrelator<floatT, HaloDepth>::EMTCorrGTensor(
    Gaugefield<floatT, true, HaloDepth>& gaugefield,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& G
) {
    StopWatch<true> emtFourierTimer;
    StopWatch<true> gFourierTimer;

    // create helper lattice containers
    LatticeContainer<true, Matrix4x4SymComplex<floatT>> emtFouriered(gaugefield.getComm(), "emtFT", "emtFT", "emtFT", "emtFT");
    emtFouriered.adjustSize(GInd::getLatData().vol4);

    // calculate EMT
    emtFouriered.template iterateOverBulk<All, HaloDepth>(EMTFullComplex<floatT, true, HaloDepth>(gaugefield.getAccessor()));
    
    // FFT it, store it in the same Container
    emtFourierTimer.start();
    fourierClass.template performFourierTransformMatrix4x4SymComponentwise<SpatialTemporal::Both>(emtFouriered, emtFouriered, 1.0);
    // fourierClass.template performFourierTransformPolymorph<Matrix4x4SymComplex<floatT>, SpatialTemporal::Both>(emtFouriered, emtFouriered, 1.0);
    emtFourierTimer.stop();
    rootLogger.debug("   EMT Fourier took          ", emtFourierTimer.seconds(), "s.");
    
    // create product out of the two FFTed EMTs, store it in G
    G.template iterateOverBulk<All, HaloDepth>(EMTtimesEMTStar<floatT>(emtFouriered.getAccessor(), emtFouriered.getAccessor()));
    
    // FFT the product back, store it in G again
    gFourierTimer.start();
    // fourierClass.template performFourierTransformationTensor4x4Symx4x4SymComplexComponentwise<SpatialTemporal::Both>(G, G, -1.0);
    fourierClass.template performFourierTransformTensor4x4Symx4x4SymHalfPolymorph<SpatialTemporal::Both>(G, G, -1.0);
    // fourierClass.template performFourierTransformPolymorph<Tensor4x4Symx4x4SymComplex<floatT>, SpatialTemporal::Both>(G, G, -1.0);
    gFourierTimer.stop();
    rootLogger.debug("   G Fourier took            ", gFourierTimer.seconds(), "s.");

}


template<class floatT, size_t HaloDepth>
void EnergyMomentumTensorCorrelator<floatT, HaloDepth>::EMTCorrGFunctions(
    Gaugefield<floatT, true, HaloDepth>& gaugefield,
    std::vector<std::vector<COMPLEX(floatT)>>& array
) {

    StopWatch<true> tensorDecompositionTimer;

    // create lattice container for G tensor
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> GTensor(gaugefield.getComm(), "GTensor", "GTensor", "GTensor", "GTensor");
    GTensor.adjustSize(GInd::getLatData().vol4);

    // calculate EMT correlator into G tensor
    this->EMTCorrGTensor(gaugefield, GTensor);

    tensorDecompositionTimer.start();
    tensorDecomposition.template getAllTensorFunctions<true>(GTensor, array);
    tensorDecompositionTimer.stop();

    rootLogger.debug("   Tensor Decomposition took ", tensorDecompositionTimer.seconds(), "s.");

}


template<class floatT, size_t HaloDepth>
void EnergyMomentumTensorCorrelator<floatT, HaloDepth>::getR2Counts(
    std::vector<int>& counts
) {
    tensorDecomposition.getR2Counts(counts);
}
