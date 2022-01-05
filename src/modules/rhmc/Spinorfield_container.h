// 
// Created by Marius Neumann on 04.01.2022.
//

#ifndef SPINORFIELD_CONTAINER
#define SPINORFIELD_CONTAINER

#include "../../spinor/spinorfield.h"
#include <vector>


template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks = 1>
class Spinorfield_container
{
public:
    
    explicit Spinorfield_container(CommunicationBase &comm, int no_pf=1) : 
        phi_container()
    {
        
    rootLogger.info("Constructing spiorfields with ", no_pf, " pseudofermions");

    for(int i = 0; i < no_pf; i++) {
        rootLogger.info("Initializing ", i, "th pseudofermion");
        phi_container.emplace_back(std::move(Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth>(comm, "Spinorfield_" + i )));
    }
        
    }
    
//     Spinorfield_container<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &
//     operator=(const Spinorfield_container<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &_phi_container) {
//         for (int i=0; i<_phi_container.length(); ++i){
//             phi_container.emplace_back(_phi_container.at(i));
//         }
//         return *this;
//     }
    
/*    
    Spinorfield_container(Spinorfield_container<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &_phi_container) : 
        phi_container()
    {
        //phi_container.resize(_phi_container.phi_container.size());
        //*phi_container.data() = &_phi_container.phi_container.data();
        _phi_container.phi_container = phi_container.swap(_phi_container.phi_container);
    }*/
    
    
//     //! copy assignment: host to host / device to device
//     Spinorfield_container<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &
//     operator=(const Spinorfield_container<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &spinorRHS) {
//         _lattice.copyFrom(spinorRHS.getArray());
//         return *this;
//     }
    
    
//     Spinorfield_container(Spinorfield_container<floatT,onDevice,LatticeLayout,HaloDepth,NStacks>&& _source) noexcept :
//             phi_container(std::move(_source.phi_container)){}
    
    std::vector<Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>> phi_container;
};




#endif //SPINORFIELD_CONTAINER








// template<class floatT, bool onDevice, Layout LatticeLayout, size_t HaloDepth, size_t NStacks>
// class Spinorfield_container
// {
// public:
//     
//     Spinorfield_container(CommunicationBase &comm, std::string spinorfieldName="Spinorfield", int no_pf) :
//             std::vector<Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth>> phi_container()
//     {
//         
//     rootLogger.info("Constructing spiorfields with ", _rhmc_param.no_pf(), " pseudofermions");
// 
//     for(int i = 0; i < no_pf; i++) {
//         rootLogger.info("Initializing ", i, "th pseudofermion");
//         phi_container.emplace_back(std::move(Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth>(_gaugeField.getComm(), "Spinorfield_lf_" + i )));
//         }
//         
//     }
//         
//     //! destructor
//     ~Spinorfield_container() = default;
//     
// };
