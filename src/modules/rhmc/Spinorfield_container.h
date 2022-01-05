// 
// Created by Marius Neumann on 04.01.2022.
//

#ifndef SPINORFIELD_CONTAINER
#define SPINORFIELD_CONTAINER

#include "../../spinor/spinorfield.h"
#include <vector>
#include <string>

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
            phi_container.emplace_back(std::move(Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth>(comm, "Spinorfield_" + std::to_string(i) )));
        }
        
    }
    
//     Spinorfield_container(Spinorfield_container<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &_phi_container) {
//         //phi_container.resize(_phi_container.phi_container.size());
// //         for (int i=0; i<_phi_container.phi_container.size(); ++i){
// //             phi_container[i] = *_phi_container.phi_container[i];
// //         }
//         * phi_container = * _phi_container.phi_container;
//     }
    
    
//         Spinorfield_container(Spinorfield_container<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &_phi_container) {
//         //phi_container.resize(_phi_container.phi_container.size());
//         for (int i=0; i<_phi_container.phi_container.size(); ++i){
//             phi_container.emplace_back(Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth>(_phi_container.phi_container.at(i)));
//         }
       // * phi_container = * _phi_container.phi_container;
        
        
   // end_ptr = &_phi_container.phi_container[size - 1];


        
        
  //  }
    
    
    
//     Spinorfield_container(Spinorfield_container<floatT, onDevice, LatticeLayout, HaloDepth, NStacks> &_phi_container) : 
//         phi_container()
//     {
//         //phi_container.resize(_phi_container.phi_container.size());
//         //*phi_container.data() = &_phi_container.phi_container.data();
//         _phi_container.phi_container = phi_container.swap(_phi_container.phi_container);
//     }
    
    
//         Spinorfield_container() = default;
    
    
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
