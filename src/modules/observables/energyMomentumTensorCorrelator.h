//
// created by Jonas Winter on 22.10.2025
//

#pragma once
#include "../../define.h"
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
        TensorDecomposition<floatT, HaloDepth> tensor_decomposition;
    private:
        typedef GIndexer<All, HaloDepth> GInd;
    public:
        EnergyMomentumTensorCorrelator(CommunicationBase& commBase) : fourierClass(commBase), tensor_decomposition(commBase) {}

        ~EnergyMomentumTensorCorrelator() {}

        int get_r2max() {
            return tensor_decomposition.get_r2max();
        }

        void EMTU_Corr(
            Gaugefield<floatT, true, HaloDepth>& gaugefield,
            LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& field
        );

        void EMTU_Corr_Gs(
            Gaugefield<floatT, true, HaloDepth>& gaugefield,
            std::vector<std::vector<COMPLEX(floatT)>>& array
        );

        void get_r2Counts(
            std::vector<int>& counts
        );

};


template<class floatT>
struct EMTtimesEMTStar {

    LatticeContainerAccessor _firstAccessor;
    LatticeContainerAccessor _secondAccessor;
    typedef GIndexer<All> GInd;

    EMTtimesEMTStar(LatticeContainerAccessor firstAccessor, LatticeContainerAccessor secondAccessor) : _firstAccessor(firstAccessor), _secondAccessor(secondAccessor) {}

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT> operator()(gSite site) {

        Matrix4x4SymComplex<floatT> firstElement(_firstAccessor.getElement<Matrix4x4SymComplex<floatT>>(site));
        Matrix4x4SymComplex<floatT> secondElement(_secondAccessor.getElement<Matrix4x4SymComplex<floatT>>(site));

        Tensor4x4Symx4x4SymComplex<floatT> result(firstElement, conj(secondElement));

        return result;
    }

};


template<class floatT, size_t HaloDepth>
void EnergyMomentumTensorCorrelator<floatT, HaloDepth>::EMTU_Corr(
    Gaugefield<floatT, true, HaloDepth>& gaugefield,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& G
) {
    // create helper lattice containers
    LatticeContainer<true, Matrix4x4SymComplex<floatT>> EMTU_FT(gaugefield.getComm(), "EMTU_FT", "EMTU_FT", "EMTU_FT", "EMTU_FT");
    // LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> G_FT(gaugefield.getComm(), "G_FT", "G_FT", "G_FT", "G_FT");

    EMTU_FT.adjustSize(GInd::getLatData().vol4);
    // G_FT.adjustSize(GInd::getLatData().vol4);
    G.adjustSize(GInd::getLatData().vol4);

    // calculate EMT
    EMTU_FT.template iterateOverBulk<All, HaloDepth>(EMTtracelessComplex<floatT, true, HaloDepth>(gaugefield.getAccessor()));

    // FFT it, store it in the same Container
    fourierClass.template performFourier3DEMT<SpatialTemporal::Both>(EMTU_FT, EMTU_FT, 1.0);

    // create product out of the two FFTed EMTs, store it in G
    G.template iterateOverBulk<All, HaloDepth>(EMTtimesEMTStar<floatT>(EMTU_FT.getAccessor(), EMTU_FT.getAccessor()));

    // FFT the product back, store it in G again
    fourierClass.template performFourier3DTensor4x4Symx4x4SymComplex<SpatialTemporal::Both>(G, G, -1.0);

}

template<class floatT, size_t HaloDepth>
void EnergyMomentumTensorCorrelator<floatT, HaloDepth>::EMTU_Corr_Gs(
    Gaugefield<floatT, true, HaloDepth>& gaugefield,
    std::vector<std::vector<COMPLEX(floatT)>>& array
) {

    // create lattice container for G tensor
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> G_tensor(gaugefield.getComm(), "G_tensor", "G_tensor", "G_tensor", "G_tensor");
    G_tensor.adjustSize(GInd::getLatData().vol4);

    // calculate EMT correlator into G tensor
    this->EMTU_Corr(gaugefield, G_tensor);

    tensor_decomposition.template getAllTensorFunctions<true>(G_tensor, array);

}


template<class floatT, size_t HaloDepth>
void EnergyMomentumTensorCorrelator<floatT, HaloDepth>::get_r2Counts(
    std::vector<int>& counts
) {
    tensor_decomposition.get_r2Counts(counts);
}
