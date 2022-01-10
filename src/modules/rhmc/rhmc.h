/*
 * rhmc.h
 *
 * P. Scior
 * 
 */

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


template <class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin=4>
class rhmc
{

public:

    rhmc(RhmcParameters rhmc_param, RationalCoeff rat, Gaugefield<floatT,onDevice,HaloDepth> &gaugeField, uint4* rand_state) : 
        _rhmc_param(rhmc_param), _rat(rat), _gaugeField(gaugeField),
        gAcc(gaugeField.getAccessor()), _savedField(gaugeField.getComm()),
        _p(gaugeField.getComm()), _rand_state(rand_state), _smeared_W(gaugeField.getComm()),
        _smeared_X(gaugeField.getComm()), phi_1f(gaugeField.getComm()), 
        phi_2f(gaugeField.getComm()), chi(gaugeField.getComm()),
        dslash(_smeared_W, _smeared_X, 0.0), 
        integrator(_rhmc_param, _gaugeField, _p, _smeared_X, _smeared_W, phi_1f, phi_2f, dslash, _rat,_smearing),
        _smearing(_gaugeField, _smeared_W, _smeared_X), elems_full(GInd::getLatData().vol4),
        energy_dens_old(gaugeField.getComm(), "old_energy_density"), energy_dens_new(gaugeField.getComm(), "new_energy_density"), 
        dens_delta(gaugeField.getComm(), "energy_density_difference")
    {           
        energy_dens_old.adjustSize(elems_full);
        energy_dens_new.adjustSize(elems_full);
        dens_delta.adjustSize(elems_full);
    };

    int update(bool metro=true, bool reverse=false);

    void init_ratapprox();
    
    // only to use in tests!
    int update_test();

private:
    RhmcParameters _rhmc_param;
    RationalCoeff _rat;

    typedef GIndexer<All,HaloDepth> GInd;
    const size_t elems_full;

    // We need the gauge field, two smeared fields and a copy of the gauge field
    Gaugefield<floatT,onDevice,HaloDepth> &_gaugeField;
    Gaugefield<floatT,onDevice,HaloDepth> _smeared_W;
    Gaugefield<floatT,onDevice,HaloDepth, U3R14> _smeared_X;
    Gaugefield<floatT,onDevice,HaloDepth> _savedField;
    HisqSmearing<floatT,onDevice,HaloDepth,R18,R18,R18,U3R14> _smearing;
    // The conjugate momentum field
    Gaugefield<floatT,onDevice,HaloDepth> _p;


    // Fields containing energy densities for the Metropolis step
    LatticeContainer<onDevice,double> energy_dens_old;
    LatticeContainer<onDevice,double> energy_dens_new;

    LatticeContainer<onDevice,double> dens_delta;

    // Pseudo-spinor fields for both flavors and another field for calculating the hamiltonian
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin> phi_1f;
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin> phi_2f;
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin> chi;

    integrator<floatT,onDevice,All,HaloDepth,HaloDepthSpin> integrator;

    // the rng state
    uint4* _rand_state;

    // vectors holding the coeff. for the rational approximations used
    std::vector<floatT> rat_1f;
    std::vector<floatT> rat_2f;
    std::vector<floatT> rat_inv_1f;
    std::vector<floatT> rat_inv_2f;
    std::vector<floatT> rat_bar_1f;
    std::vector<floatT> rat_bar_2f;

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
};

#endif
