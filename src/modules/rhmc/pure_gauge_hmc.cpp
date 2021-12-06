// Created by Philipp Scior on 22.10.18

#include "pure_gauge_hmc.h"
#include "../../gauge/gauge_kernels.cpp"

template <bool onDevice, class floatT>
struct add_f_r_f_r
{
    LatticeContainerAccessor acc_a;
    LatticeContainerAccessor acc_b;

    floatT _aa, _bb;

    floatT ret;

    add_f_r_f_r(LatticeContainer<onDevice, floatT> &a, LatticeContainer<onDevice, floatT> &b, floatT aa, floatT bb) :
    acc_a(a.getAccessor()), acc_b(b.getAccessor()), _aa(aa), _bb(bb) {}

    __device__ __host__ floatT operator()(gSite site) {


    ret = _aa * acc_a.template getElement<floatT>(site) + _bb * acc_b.template getElement<floatT>(site);

    return ret;

    }


};



template <class floatT, Layout LatticeLayout, size_t HaloDepth, CompressionType comp>
int pure_gauge_hmc<floatT, LatticeLayout, HaloDepth, comp>::update(bool metro, bool reverse){

    //check unitarity of lattice

    //copy gaugefield to savedfield

    _savedField = _gaugeField;

    //make momenta

    generate_momenta();

    //get oldaction
    floatT old_hamiltonian = get_Hamiltonian(energy_dens_old);
    
    pure_gauge_integrator<floatT, true, HaloDepth, comp> integrator(_rhmc_param, _gaugeField, _p);

    //do leapfrog
    integrator.integrate();

    //possible reversibility check
    if (reverse)
    {
        rootLogger.warn("Checking if integration is reversible");

        _p = -1.0 * _p;

    integrator.integrate();
    }

    //get newaction
    floatT new_hamiltonian = get_Hamiltonian(energy_dens_new);

    // rootLogger.info("Simple Delta H = " ,  new_hamiltonian- old_hamiltonian);

    int ret;


        //make Metropolis step
        bool accept = Metropolis();

        if (metro)
    {
        if (accept){
            ret=1;
            rootLogger.info("Update acepted!");
    }
        else{
            _gaugeField=_savedField;
            _gaugeField.updateAll();
            ret=0;
            rootLogger.info("Update declined!");
        }
    }
    else{

        //skip Metropolis step
        ret=1;
        rootLogger.warn("Skipped Metropolis step!");
    }

    return ret;
}

template<class floatT, Layout LatticeLayout, size_t HaloDepth, CompressionType comp>
void pure_gauge_hmc<floatT, LatticeLayout, HaloDepth, comp>::generate_momenta(){

    _p.gauss(_rand_state);
}

template<class floatT, Layout LatticeLayout, size_t HaloDepth, CompressionType comp>
void pure_gauge_hmc<floatT, LatticeLayout, HaloDepth, comp>::generate_const_momenta(){

    _p.iterateWithConst(glambda_1<floatT>());
}

template<class floatT, size_t HaloDepth>
struct get_momenta
{
    gaugeAccessor<floatT> pAccessor;

    get_momenta(Gaugefield<floatT,true,HaloDepth> &p) : pAccessor(p.getAccessor()) {
    }

    __device__ __host__ floatT operator()(gSite site) {

        typedef GIndexer<All,HaloDepth> GInd;

        floatT result = 0;

        for (int mu = 0; mu < 4; mu++) {
            {
                result += tr_d(pAccessor.getLink(GInd::getSiteMu(site, mu)), pAccessor.getLink(GInd::getSiteMu(site, mu)));
            }
        }
        return result;
    }
};

template<class floatT, Layout LatticeLayout, size_t HaloDepth, CompressionType comp>
floatT pure_gauge_hmc<floatT, LatticeLayout, HaloDepth, comp>::get_Hamiltonian(LatticeContainer<true,floatT> &energy_dens){


    LatticeContainer<true,floatT> redBase(_p.getComm(), "momenta");
    LatticeContainer<true,floatT> redBase2(_p.getComm(), "helper field for energy_dens");
    LatticeContainer<true,floatT> redBase3(_p.getComm(), "second helper field for energy_dens");

    redBase.adjustSize(elems);
    redBase2.adjustSize(elems);
    redBase3.adjustSize(elems);


    floatT momenta;
    floatT gaugeact;

    redBase.template iterateOverBulk<All, HaloDepth>(get_momenta<floatT, HaloDepth>(_p));
    redBase2.template iterateOverBulk<All, HaloDepth>(plaquetteKernel<floatT, true, HaloDepth,comp>(_gaugeField));
    redBase3.template iterateOverBulk<All, HaloDepth>(rectangleKernel<floatT, true, HaloDepth,comp>(_gaugeField));
    // rootLogger.info("constructed momentum, plaquette and rectangle dens");
    redBase2.template iterateOverBulk<All, HaloDepth>(add_f_r_f_r<true, floatT>(redBase2, redBase3, 5.0/3.0 , -1.0/12.0)); 
    // rootLogger.info("added plaquette and rectangle dens to symanzik dens");

    floatT beta = _rhmc_param.beta() *3.0/5.0;

    energy_dens.template iterateOverBulk<All, HaloDepth>(add_f_r_f_r<true, floatT>(redBase, redBase2, 0.5, -beta/3.0));
    // rootLogger.info("added momentumm and symanzik dens to energy_dens");

    redBase.reduce(momenta, elems);


    // floatT H;


    // energy_dens.reduce(H, elems);

    // rootLogger.info("reduced momentum dens");
    

    GaugeAction<floatT, true, HaloDepth, comp> gaugeaction(_gaugeField);
    gaugeact = - _rhmc_param.beta() * gaugeaction.symanzik(); 

    floatT hamiltonian = 0.5 *momenta;

    hamiltonian+= gaugeact; 

    rootLogger.info("momenta = " ,  0.5 *momenta ,  " glue = " ,  gaugeact);// << "H = " << H;

    return hamiltonian;
}

template<class floatT, Layout LatticeLayout, size_t HaloDepth, CompressionType comp>
bool pure_gauge_hmc<floatT, LatticeLayout, HaloDepth, comp>::Metropolis(){

    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_real_distribution<> dis(0.0, 1.0);

    dens_delta.template iterateOverBulk<All, HaloDepth>(add_f_r_f_r<true, floatT>(energy_dens_new, energy_dens_old, 1.0, -1.0));

    double delta_E=0.0;

    dens_delta.reduce(delta_E, elems);

    rootLogger.info("Delta H = " ,  delta_E);

    uint4 state;

    gpuMemcpy(&state, _rand_state, sizeof(uint4), gpuMemcpyDeviceToHost);

    if (delta_E < 0.0)
        return true;
    else{
        // double rand = dis(gen);
        double rand = get_rand<double>(&state);
        _p.getComm().root2all(rand); // Is is important so sync the random numbers between processes!
        if(rand < exp(-delta_E))
            return true;
        else
            return false;
    }
}

template <class floatT, Layout LatticeLayout, size_t HaloDepth, CompressionType comp>
int pure_gauge_hmc<floatT, LatticeLayout, HaloDepth, comp>::update_test(){

    //check unitarity of lattice

    //make momenta

    generate_const_momenta();

    //get oldaction
    floatT old_hamiltonian = get_Hamiltonian(energy_dens_old);
    
    //do leapfrog
    pure_gauge_integrator<floatT,true,HaloDepth, comp> integrator(_rhmc_param, _gaugeField, _p);

    integrator.integrate();

    //get newaction
    floatT new_hamiltonian = get_Hamiltonian(energy_dens_new);

    // rootLogger.info("Delta H =" ,  new_hamiltonian - old_hamiltonian);

    int ret;

    //make Metropolis step
    bool accept = Metropolis();

    if (accept){
        ret=1;
        rootLogger.info("Update acepted!");
    }
    else{
        _gaugeField=_savedField;
        _gaugeField.updateAll();
        ret=0;
        rootLogger.info("Update declined!");
    }

    return ret;
}

// explicit instantiation
#define CLASS_INIT(floatT,HALO, comp) \
template class pure_gauge_hmc<floatT, All, HALO, comp>;

INIT_PHC(CLASS_INIT)

