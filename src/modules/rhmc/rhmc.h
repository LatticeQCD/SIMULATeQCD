/*
 * rhmc.h
 *
 * P. Scior
 *
 */

#ifndef RHMC
#define RHMC

#include "../../base/math/su3.h"
#include "../../base/math/random.h"
#include "../../gauge/gaugefield.h"
#include "rhmcParameters.h"
#include "../../base/memoryManagement.h"
#include "../../base/stopWatch.h"
#include "../../base/latticeContainer.h"
#include "../../gauge/gaugeAction.h"
#include "../../gauge/gaugeActionDeriv.h"
#include "integrator.h"
#include "../../spinor/spinorfield.h"
#include "../dslash/dslash.h"
#include <random>
#include <math.h>
#include <iostream>
#include <vector>
#include "../hisq/hisqSmearing.h"
#include "spinorfield_container.h"


template <class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin=4>
class rhmc
{

public:

    rhmc(RhmcParameters rhmc_param, RationalCoeff rat, Gaugefield<floatT,onDevice,HaloDepth> &gaugeField, uint4* rand_state)
        : _rhmc_param(rhmc_param),
          _rat(rat),
          elems_full(GInd::getLatData().vol4),
          _gaugeField(gaugeField),
           _smeared_W(gaugeField.getComm()),
           _smeared_X(gaugeField.getComm()),
           _savedField(gaugeField.getComm()),
           _smearing(_gaugeField, _smeared_W, _smeared_X),
          _p(gaugeField.getComm()),
          energy_dens_old(gaugeField.getComm(), "old_energy_density"),
          energy_dens_new(gaugeField.getComm(), "new_energy_density"),
          dens_delta(gaugeField.getComm(), "energy_density_difference"),
          phi_sf_container(gaugeField.getComm(), rhmc_param.no_pf()),
          phi_lf_container(gaugeField.getComm(), rhmc_param.no_pf()),
          chi(gaugeField.getComm()),
          dslash(_smeared_W, _smeared_X, 0.0),
          integrator(_rhmc_param, _gaugeField, _p, _smeared_X, _smeared_W, dslash, _rat, _smearing),
          _rand_state(rand_state),
          gAcc(gaugeField.getAccessor())          
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
    const int _no_pf = _rhmc_param.no_pf();

    // We use five gauge fields altogether:
    Gaugefield<floatT,onDevice,HaloDepth> &_gaugeField;                    // The to-be-updated field
    Gaugefield<floatT,onDevice,HaloDepth> _smeared_W;
    Gaugefield<floatT,onDevice,HaloDepth, U3R14> _smeared_X;
    Gaugefield<floatT,onDevice,HaloDepth> _savedField;                     // A saved copy. If we reject the update, we go back to savedField.
    HisqSmearing<floatT,onDevice,HaloDepth,R18,R18,R18,U3R14> _smearing;
    Gaugefield<floatT,onDevice,HaloDepth> _p;                              // The conjugate momentum field

    //! In the end this contains the *contracted* propagators for each mass combination and spacetime point (vol4)
    //std::vector<LatticeContainer<false,GPUcomplex<floatT>>> _contracted_propagators;
    // Fields containing energy densities for the Metropolis step
    LatticeContainer<onDevice,double> energy_dens_old;
    LatticeContainer<onDevice,double> energy_dens_new;

    LatticeContainer<onDevice,double> dens_delta;

    // Pseudo-spinor fields for both flavors and another field for calculating the hamiltonian
    Spinorfield_container<floatT, onDevice, Even, HaloDepthSpin> phi_sf_container;
    Spinorfield_container<floatT, onDevice, Even, HaloDepthSpin> phi_lf_container;
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin> chi;

    HisqDSlash<floatT, onDevice, Even, HaloDepth, HaloDepthSpin, 1> dslash;
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

    SU3Accessor<floatT, R18> gAcc;

    void generate_momenta();

    void check_unitarity();

    void make_phi(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &phi, std::vector<floatT> rat_coeff);
    void make_chi(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &chi, Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &phi, std::vector<floatT> rat_coeff);

    double get_Hamiltonian(LatticeContainer<onDevice,double> &energy_dens);

    bool Metropolis();

    //use this only for testing
    void generate_const_momenta();
    void make_const_phi(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &phi, std::vector<floatT> rat_coeff);
};

#endif //RHMC
