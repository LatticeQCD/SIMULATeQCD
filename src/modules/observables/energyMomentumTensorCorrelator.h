//
// created by Jonas Winter on 22.10.2025
//

#pragma once
#include "../../base/latticeContainer.h"
#include "../../gauge/gaugefield.h"
#include "../../experimental/fourierNon2.h"
#include "../../base/math/matrix4x4SymComplex.h"
#include "../../base/math/tensor4x4Symx4x4SymComplex.h"


template<class floatT, size_t HaloDepth>
class EnergyMomentumTensorCorrelator {
    protected:
        FourierClass<floatT> fourierClass;
    private:
        typedef GIndexer<All, HaloDepth> GInd;
    public:
        EnergyMomentumTensorCorrelator(CommunicationBase& commBase) : fourierClass(commBase) {}

        ~EnergyMomentumTensorCorrelator() {}

        void EMTU_Corr(
            Gaugefield<floatT, true, HaloDepth>& gaugefield,
            LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& field
        );
};


template<class floatT>
struct EMTtimesEMT {

    LatticeContainerAccessor _firstAccessor;
    LatticeContainerAccessor _secondAccessor;
    typedef GIndexer<All> GInd;

    EMTtimesEMT(LatticeContainerAccessor firstAccessor, LatticeContainerAccessor secondAccessor) : _firstAccessor(firstAccessor), _secondAccessor(secondAccessor) {}

    __device__ __host__ inline Tensor4x4Symx4x4SymComplex<floatT> operator()(gSite site) {

        Matrix4x4SymComplex<floatT> firstElement(_firstAccessor.getElement<Matrix4x4SymComplex<floatT>>(site));
        Matrix4x4SymComplex<floatT> secondElement(_secondAccessor.getElement<Matrix4x4SymComplex<floatT>>(site));

        Tensor4x4Symx4x4SymComplex<floatT> result(firstElement, secondElement);

        return result;
    }

};


template<class floatT, size_t HaloDepth>
void EnergyMomentumTensorCorrelator<floatT, HaloDepth>::EMTU_Corr(
    Gaugefield<floatT, true, HaloDepth>& gaugefield,
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>>& G
) {
    // create helper lattice containers
    LatticeContainer<true, Matrix4x4SymComplex<floatT>> EMTU(gaugefield.getComm(), "EMTU", "EMTU", "EMTU", "EMTU");
    LatticeContainer<true, Matrix4x4SymComplex<floatT>> EMTU_FT(gaugefield.getComm(), "EMTU_FT", "EMTU_FT", "EMTU_FT", "EMTU_FT");
    LatticeContainer<true, Matrix4x4SymComplex<floatT>> EMTU_FTB(gaugefield.getComm(), "EMTU_FTB", "EMTU_FTB", "EMTU_FTB", "EMTU_FTB");
    LatticeContainer<true, Tensor4x4Symx4x4SymComplex<floatT>> G_FT(gaugefield.getComm(), "G_FT", "G_FT", "G_FT", "G_FT");

    EMTU.adjustSize(GInd::getLatData().vol4);
    EMTU_FT.adjustSize(GInd::getLatData().vol4);
    EMTU_FTB.adjustSize(GInd::getLatData().vol4);
    G_FT.adjustSize(GInd::getLatData().vol4);
    G.adjustSize(GInd::getLatData().vol4);

    // calculate EMT
    EMTU.template iterateOverBulk<All, HaloDepth>(EMTtracelessComplex<floatT, true, HaloDepth>(gaugefield.getAccessor()));

    // FFT it, and FFT it with opposite sign
    fourierClass.performFourier3DEMT(EMTU, EMTU_FT, 1.0);
    fourierClass.performFourier3DEMT(EMTU, EMTU_FTB, -1.0);

    // create product out of the two FFTed EMTs
    G_FT.template iterateOverBulk<All, HaloDepth>(EMTtimesEMT<floatT>(EMTU_FTB.getAccessor(), EMTU_FT.getAccessor()));

    // FFT the product back
    fourierClass.performFourier3DTensor4x4Symx4x4SymComplex(G_FT, G, -1.0);

}

