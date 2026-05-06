//
// Created by Marius Neumann on 04.01.2022.
//

#ifndef SPINORFIELD_CONTAINER
#define SPINORFIELD_CONTAINER

#include "../../spinor/spinorfield.h"
#include <vector>
#include <string>

template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t, elems, size_t NStacks = 1>
class Spinorfield_container
{
public:

    explicit Spinorfield_container(CommunicationBase &comm, int no_pf=1) :
        phi_container()
    {
        rootLogger.info("Constructing spiorfields with ", no_pf, " pseudofermions");
        for(int i = 0; i < no_pf; i++) {
            rootLogger.info("Initializing pseudofermion No: ", i);
            phi_container.emplace_back(std::move(Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, elems>(comm, "Spinorfield_" + std::to_string(i) )));
        }
    }

    std::vector<Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, elems, NStacks>> phi_container;
};

#endif //SPINORFIELD_CONTAINER
