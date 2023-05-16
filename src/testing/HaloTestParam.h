//
// Created by Lukas Mazur on 12.04.19.
//

#pragma once
class HaloTestParam : public LatticeParameters{
public:
    Parameter<int, 4> SimulatedNodeDim;
    Parameter<bool> forceHalos;
    HaloTestParam() {
        add(SimulatedNodeDim, "SimulatedNodes");
        add(forceHalos, "forceHalos");
    }
};

