/*
 * pure_gauge_hmc.h
 *
 * P. Scior
 *
 */

#ifndef PURE_GAUGE_HMC
#define PURE_GAUGE_HMC

#include "../../base/math/gsu3.h"
#include "../../base/math/grnd.h"
#include "../../gauge/gaugefield.h"
#include "../../base/latticeParameters.h"
#include "../../base/memoryManagement.h"
#include "../../base/stopWatch.h"
#include "../../base/LatticeContainer.h"
#include "../../gauge/GaugeAction.h"
#include "../../gauge/gaugeActionDeriv.h"
#include "integrator.h"
#include <random>
#include <math.h>
#include <iostream>
#include <string>

template<class floatT, Layout LatticeLayout, size_t HaloDepth, CompressionType comp = R18>
class pure_gauge_hmc {

public:
    pure_gauge_hmc(RhmcParameters rhmc_param, Gaugefield<floatT, true, HaloDepth, comp> &gaugeField, uint4 *rand_state)
            : _rhmc_param(rhmc_param), _gaugeField(gaugeField),
              gAcc(gaugeField.getAccessor()), _savedField(gaugeField.getComm()), elems(GInd::getLatData().vol4),
              _p(gaugeField.getComm()), _rand_state(rand_state), energy_dens_old(gaugeField.getComm(), "old_energy_density"),
              energy_dens_new(gaugeField.getComm(), "new_energy_density"), dens_delta(gaugeField.getComm(), "energy_density_difference")
              {
                energy_dens_old.adjustSize(elems);
                energy_dens_new.adjustSize(elems);
                dens_delta.adjustSize(elems);
              };

    int update(bool metro = true, bool reverse = false);

    // only to use in tests!
    int update_test();

private:
    RhmcParameters _rhmc_param;

    typedef GIndexer<All,HaloDepth> GInd;
    const size_t elems;

    LatticeContainer<true,floatT> energy_dens_old;
    LatticeContainer<true,floatT> energy_dens_new;

    LatticeContainer<true,floatT> dens_delta;

    Gaugefield<floatT, true, HaloDepth, comp> &_gaugeField;
    Gaugefield<floatT, true, HaloDepth, comp> _savedField;
    Gaugefield<floatT, true, HaloDepth> _p;

    uint4 *_rand_state;

    gaugeAccessor<floatT, comp> gAcc;

    void generate_momenta();

    floatT get_Hamiltonian(LatticeContainer<true,floatT> &energy_dens);

    bool Metropolis();

    //use this only for testing
    void generate_const_momenta();
};

#endif
