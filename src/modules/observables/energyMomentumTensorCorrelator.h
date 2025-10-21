//
// Created by Jonas Winter on 21.10.2025
//

#pragma once


// this class is used for calculating EMT correlators by using the Fourier method
class EnergyMomentumTensorCorrelator {
    protected:
        LatticeContainer<true, Matrix4x4SymComplex<PREC>>
    public:
        EnergyMomentumTensorCorrelator(const Gaugefield<floatT, onDevice, HaloDepth> &gaugefield)
}