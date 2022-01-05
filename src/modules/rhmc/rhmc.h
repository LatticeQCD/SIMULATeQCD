// THIS A RHMC CLASS
// Created by Philipp Scior on 10.12.18

#ifndef RHMC
#define RHMC

#include "../../base/math/gsu3.h"
#include "../../base/math/grnd.h"
#include "../../gauge/gaugefield.h"
#include "rhmcParameters.h"
#include "../../base/memoryManagement.h"
#include "../../base/stopWatch.h"
#include "../../base/LatticeContainer.h"
#include "../../gauge/GaugeAction.h"
#include "../../gauge/gaugeActionDeriv.h"
#include "integrator.h"
#include "../../spinor/spinorfield.h"
#include "../dslash/dslash.h"
#include <random>
#include <math.h>
#include <iostream>
#include <vector>
#include "../HISQ/hisqSmearing.h"
#include "Spinorfield_container.h"


template <class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin=4>
class rhmc
{

public:

    rhmc(RhmcParameters rhmc_param, RationalCoeff rat, Gaugefield<floatT,onDevice,HaloDepth> &gaugeField, uint4* rand_state) : 
        _rhmc_param(rhmc_param), _rat(rat), _gaugeField(gaugeField),
        gAcc(gaugeField.getAccessor()), _savedField(gaugeField.getComm()),
        _p(gaugeField.getComm()), _rand_state(rand_state), _smeared_W(gaugeField.getComm()),
        _smeared_X(gaugeField.getComm()),
        //phi_1f(gaugeField.getComm()), phi_2f(gaugeField.getComm()),
        //phi_lf_container(), phi_sf_container(),
        phi_lf_container(gaugeField.getComm(), rhmc_param.no_pf()),
        phi_sf_container(gaugeField.getComm(), rhmc_param.no_pf()),
        chi(gaugeField.getComm()),
        dslash(_smeared_W, _smeared_X, 0.0), 
        integrator(_rhmc_param, _gaugeField, _p, _smeared_X, _smeared_W, phi_lf_container, phi_sf_container, dslash, _rat, _smearing),
        _smearing(_gaugeField, _smeared_W, _smeared_X), elems_full(GInd::getLatData().vol4),
        energy_dens_old(gaugeField.getComm(), "old_energy_density"), energy_dens_new(gaugeField.getComm(), "new_energy_density"), 
        dens_delta(gaugeField.getComm(), "energy_density_difference")
    {           
        energy_dens_old.adjustSize(elems_full);
        energy_dens_new.adjustSize(elems_full);
        dens_delta.adjustSize(elems_full);
        
        
        
//         rootLogger.info("Constructing spiorfields with ", _rhmc_param.no_pf(), " pseudofermions");
// 
//         for(int i = 0; i < _rhmc_param.no_pf(); i++) {
//             rootLogger.info("Initializing ", i, "th pseudofermion");
//             phi_lf_container.emplace_back(std::move(Spinorfield<floatT, onDevice, Even, HaloDepthSpin>(_gaugeField.getComm(), "Spinorfield_lf_" + i )));
//             phi_sf_container.emplace_back(std::move(Spinorfield<floatT, onDevice, Even, HaloDepthSpin>(_gaugeField.getComm(), "Spinorfield_sf_" + i )));
//         }
        
        //_no_pf = _rhmc_param.no_pf();
//         for(int i = 0; i < _rhmc_param.no_pf(); i++) {
//             rootLogger.info("Initializing ", i, "th pseudofermion");
//             //for(int j = 0; j < i; j++) {
//                 phi_lf_container.emplace_back(std::move(Spinorfield<floatT, onDevice, Even, HaloDepthSpin>(_gaugeField.getComm(), "Spinorfield_lf_" + i )));
//                 phi_sf_container.emplace_back(std::move(Spinorfield<floatT, onDevice, Even, HaloDepthSpin>(_gaugeField.getComm(), "Spinorfield_sf_" + i )));
//             //}
//         }
    };
    
     
    //_vol4 = GInd::getLatData().vol4;
    
//     //! set up _contracted_propagators
//     for(size_t i = 0; i < _no_pf; i++) {
//         for(size_t j = 0; j <= i; j++) {
//             //! Here we need to give names without the "SHARED_" prefix to the MemoryManagement, otherwise all
//             //! of these will point to the same memory.
//             _contracted_propagators.emplace_back(std::move(LatticeContainer<false, GPUcomplex<floatT>>(_commBase,
//                     "propmemA", "propmemB", "propmemC", "propmemD")));
//             _contracted_propagators.back().adjustSize(_vol4);
//         }
//     }
    
    //! setup memory for multiple pf
    
//     for(size_t i = 0; i < _no_pf; i++) {
//         phi_lf_container.emplace_back(std::move(Spinorfield<floatT, onDevice, Even, HaloDepthSpin>(gaugeField.getComm())));
//         phi_sf_container.emplace_back(std::move(Spinorfield<floatT, onDevice, Even, HaloDepthSpin>(gaugeField.getComm())));
//     }
    
//     //! is getComm() neccessary?
//     for(size_t i = 0; i < _no_pf; i++) {
//         phi_lf_container[i](gaugeField.getComm());
//         phi_sf_container[i](gaugeField.getComm());
//     }

    int update(bool metro=true, bool reverse=false);

    void init_ratapprox();
    
    // only to use in tests!
    int update_test();

private:
    RhmcParameters _rhmc_param;
    RationalCoeff _rat;

    typedef GIndexer<All,HaloDepth> GInd;
    const size_t elems_full;
    const int _no_pf = _rhmc_param.no_pf();
    //const size_t _vol4;

    // We need the gauge field, two smeared fields and a copy of the gauge field
    Gaugefield<floatT,onDevice,HaloDepth> &_gaugeField;
    Gaugefield<floatT,onDevice,HaloDepth> _smeared_W;
    Gaugefield<floatT,onDevice,HaloDepth, U3R14> _smeared_X;
    Gaugefield<floatT,onDevice,HaloDepth> _savedField;
    HisqSmearing<floatT,onDevice,HaloDepth,R18,R18,R18,U3R14> _smearing;
    // The conjugate momentum field
    Gaugefield<floatT,onDevice,HaloDepth> _p;

    //! In the end this contains the *contracted* propagators for each mass combination and spacetime point (vol4)
    //std::vector<LatticeContainer<false,GPUcomplex<floatT>>> _contracted_propagators;
    // Fields containing energy densities for the Metropolis step
    LatticeContainer<onDevice,double> energy_dens_old;
    LatticeContainer<onDevice,double> energy_dens_new;

    LatticeContainer<onDevice,double> dens_delta;

    // Pseudo-spinor fields for both flavors and another field for calculating the hamiltonian
     Spinorfield_container<floatT, onDevice, Even, HaloDepthSpin> phi_sf_container;
     Spinorfield_container<floatT, onDevice, Even, HaloDepthSpin> phi_lf_container;
//     std::vector<Spinorfield<floatT, onDevice, Even, HaloDepthSpin>> phi_sf_container;
//     std::vector<Spinorfield<floatT, onDevice, Even, HaloDepthSpin>> phi_lf_container;
    //Spinorfield<floatT, onDevice, Even, HaloDepthSpin> phi_1f;
    //Spinorfield<floatT, onDevice, Even, HaloDepthSpin> phi_2f;
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin> chi;

    integrator<floatT,onDevice,All,HaloDepth,HaloDepthSpin> integrator;

    // the rng state
    uint4* _rand_state;

    // vectors holding the coeff. for the rational approximations used
    std::vector<floatT> rat_sf;
    std::vector<floatT> rat_lf;
    std::vector<floatT> rat_inv_sf;
    std::vector<floatT> rat_inv_lf;
    std::vector<floatT> rat_bar_sf;
    std::vector<floatT> rat_bar_lf;

    AdvancedMultiShiftCG<floatT, 14> cgM;
    HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, 1> dslash;

    gaugeAccessor<floatT, R18> gAcc;

    void generate_momenta();

    void check_unitarity();

    void make_phi(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &phi, std::vector<floatT> rat_coeff);
    void make_chi(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &chi, Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &phi, 
        std::vector<floatT> rat_coeff);

    double get_Hamiltonian(LatticeContainer<onDevice,double> &energy_dens);

    bool Metropolis();

    //use this only for testing
    void generate_const_momenta();
    void make_const_phi(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &phi, std::vector<floatT> rat_coeff);
    
    
//     //! set up _contracted_propagators
//     for(size_t i = 0; i < _no_pf; i++) {
//         for(size_t j = 0; j <= i; j++) {
//             //! Here we need to give names without the "SHARED_" prefix to the MemoryManagement, otherwise all
//             //! of these will point to the same memory.
//             _contracted_propagators.emplace_back(std::move(LatticeContainer<false, GPUcomplex<floatT>>(_commBase,
//                     "propmemA", "propmemB", "propmemC", "propmemD")));
//             _contracted_propagators.back().adjustSize(_vol4);
//         }
//     }
//     
//     //! setup memory for multiple pf
//     for(size_t i = 0; i < _no_pf; i++) {
//         phi_lf_container.emplace_back(std::move(Spinorfield<floatT, onDevice, Even, HaloDepthSpin>(_commBase, "Spinorfield_lf"[i])));
//         phi_sf_container.emplace_back(std::move(Spinorfield<floatT, onDevice, Even, HaloDepthSpin>(_commBase, "Spinorfield_sf"[i])));
//     }
//     
//     //! is getComm() neccessary?
//     for(size_t i = 0; i < _no_pf; i++) {
//         phi_lf_container[i](gaugeField.getComm());
//         phi_sf_container[i](gaugeField.getComm());
//     }
    

};

#endif
