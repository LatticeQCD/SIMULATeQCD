/*
 * rhmc.cu
 *
 * P. Scior
 *
 */

#include "rhmc.h"
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


template<class floatT,bool onDevice, size_t HaloDepthSpin>
struct get_fermion_act
{
    Vect3arrayAcc<floatT> spin_acc;

    double ret;

    get_fermion_act(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &chi) : spin_acc(chi.getAccessor()) {};

    __device__ __host__ double operator()(gSite site){


        typedef GIndexer<All,HaloDepthSpin> GInd;

        if (site.isite < GInd::getLatData().sizeh)
            ret = norm2(spin_acc.getElement(site));
        else
            ret = 0.0;

        return ret;
    }
};


template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp=R18>
struct do_check_unitarity
{
    do_check_unitarity(Gaugefield<floatT,onDevice,HaloDepth,comp> &gauge) : gAcc(gauge.getAccessor()) {};

    SU3Accessor<floatT, comp> gAcc;

    __device__ __host__ floatT operator()(gSite site){

        typedef GIndexer<All,HaloDepth> GInd;

        floatT ret=0.0;

        for (size_t mu = 0; mu < 4; ++mu)
        {
            gSiteMu siteM = GInd::getSiteMu(site, mu);
            ret += tr_d(gAcc.getLinkDagger(siteM)*gAcc.getLink(siteM));
        }

        return ret/4.0;
    }
};


template <class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin>
void rhmc<floatT,onDevice,HaloDepth,HaloDepthSpin>::check_unitarity()
{
    LatticeContainer<true,floatT> unitarity(_p.getComm());
    unitarity.adjustSize(elems_full);

    unitarity.template iterateOverBulk<All, HaloDepth>(do_check_unitarity<floatT, onDevice, HaloDepth>(_gaugeField));

    floatT unit_norm;
    unitarity.reduce(unit_norm, elems_full);

    rootLogger.info(std::setprecision(10) ,  "Unitarity norm <Tr(U^+U)> = " ,  unit_norm/floatT(GInd::getLatData().globvol4));
}


// constructing the vectors with the coeff. for the rational approx. from the ones in the parameter file
// This assumes that all approx without bar have that same degree, same for bar!
template <class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin>
void rhmc<floatT,onDevice,HaloDepth,HaloDepthSpin>::init_ratapprox()
{
    int length = _rat.r_inv_sf_num.get().size();

    floatT ml2 = _rhmc_param.m_ud() * _rhmc_param.m_ud();
    floatT ms2 = _rhmc_param.m_s() * _rhmc_param.m_s();

    rat_sf.push_back(_rat.r_sf_const());
    rat_lf.push_back(_rat.r_lf_const());
    rat_inv_sf.push_back(_rat.r_inv_sf_const());
    rat_inv_lf.push_back(_rat.r_inv_lf_const());
    rat_bar_sf.push_back(_rat.r_bar_sf_const());
    rat_bar_lf.push_back(_rat.r_bar_lf_const());

    for (int i = 0; i < length; ++i)
    {
        rat_sf.push_back(_rat.r_sf_num[i]);
        rat_lf.push_back(_rat.r_lf_num[i]);
        rat_inv_sf.push_back(_rat.r_inv_sf_num[i]);
        rat_inv_lf.push_back(_rat.r_inv_lf_num[i]);
    }

    rat_sf.push_back(_rat.r_sf_den[0]+ms2);
    rat_lf.push_back(_rat.r_lf_den[0]+ml2);
    rat_inv_sf.push_back(_rat.r_inv_sf_den[0]+ms2);
    rat_inv_lf.push_back(_rat.r_inv_lf_den[0]+ml2);

    for (int i = 1; i < length; ++i)
    {
        rat_sf.push_back(_rat.r_sf_den[i]-_rat.r_sf_den[0]);
        rat_lf.push_back(_rat.r_lf_den[i]-_rat.r_lf_den[0]);
        rat_inv_sf.push_back(_rat.r_inv_sf_den[i]-_rat.r_inv_sf_den[0]);
        rat_inv_lf.push_back(_rat.r_inv_lf_den[i]-_rat.r_inv_lf_den[0]);
    }

    length = _rat.r_bar_sf_num.get().size();


    for (int i = 0; i < length; ++i)
    {
        rat_bar_sf.push_back(_rat.r_bar_sf_num[i]);
        rat_bar_lf.push_back(_rat.r_bar_lf_num[i]);
    }

    rat_bar_sf.push_back(_rat.r_bar_sf_den[0]+ms2);
    rat_bar_lf.push_back(_rat.r_bar_lf_den[0]+ml2);

    for (int i = 0; i < length; ++i)
    {
        rat_bar_sf.push_back(_rat.r_bar_sf_den[i]-_rat.r_bar_sf_den[0]);
        rat_bar_lf.push_back(_rat.r_bar_lf_den[i]-_rat.r_bar_lf_den[0]);
    }
}

// Method to be called from outside, does the rhmc update
template <class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin>
int rhmc<floatT, onDevice, HaloDepth, HaloDepthSpin>::update(bool metro, bool reverse){

    check_unitarity();

    //copy gaugefield to savedfield
    _savedField = _gaugeField;

    rootLogger.info("Smearing gauge fields");
    _smearing.SmearAll(_rhmc_param.mu_f());

    rootLogger.info("generating momenta");
    generate_momenta();

    rootLogger.info("Constructing peudo-fermion fields");

    for(int i = 0; i < _no_pf; i++) {
        make_phi(phi_sf_container.phi_container[i], rat_inv_sf);
    }
    rootLogger.info("phi_sf: done");

    for(int i = 0; i < _no_pf; i++) {
        make_phi(phi_lf_container.phi_container[i], rat_inv_lf);
    }
    rootLogger.info("phi_lf: done");

    //get oldaction
    __attribute__((unused))double old_hamiltonian = get_Hamiltonian(energy_dens_old);

    //do the integration
    integrator.integrate(phi_lf_container, phi_sf_container);

    //possible reversibility check
    if (reverse)
    {
        rootLogger.warn("Checking if integration is reversible");

        _p = -floatT(1.0) * _p;

        integrator.integrate(phi_lf_container, phi_sf_container);

        Gaugefield<floatT,false,HaloDepth> saved_h(_p.getComm());
        Gaugefield<floatT,false,HaloDepth> gauge_h(_p.getComm());

        saved_h = _savedField;
        gauge_h = _gaugeField;

        for (int x = 0; x < (int) GInd::getLatData().lx; x++)
        for (int y = 0; y < (int) GInd::getLatData().ly; y++)
        for (int z = 0; z < (int) GInd::getLatData().lz; z++)
        for (int t = 0; t < (int) GInd::getLatData().lt; t++)
            for (int mu = 0; mu < 4; mu++) {
                gSite site = GInd::getSite(x, y, z, t);

                SU3<double> tmpA = saved_h.getAccessor().template getLink<double>(GInd::getSiteMu(site, mu));

                SU3<double> tmpB = gauge_h.getAccessor().template getLink<double>(GInd::getSiteMu(site, mu));

                    if (!compareSU3(tmpA, tmpB, 1e-4)) {
                        rootLogger.error("Difference in saved and evolved Gaugefields at " ,  LatticeDimensions(x, y, z, t) , ", mu = " ,  mu);
                        rootLogger.error("|| S - G ||_inf = " ,  infnorm(tmpA-tmpB));
                    }
        }
    }

    //get newaction
    __attribute__((unused)) double new_hamiltonian = get_Hamiltonian(energy_dens_new);

    int ret;

    bool accept = Metropolis();

    if (metro)
    {
        //make Metropolis step
        if (accept){
            ret=1;
            rootLogger.info("Update accepted!");
        } else {
            _gaugeField=_savedField;
            _gaugeField.updateAll();
            ret=0;
            rootLogger.info("Update declined!");
        }
    } else {

        //skip Metropolis step
        ret=1;
        rootLogger.warn("Skipped Metropolis step!");
    }

    return ret;
}

// Only use this in tests!
template <class floatT,bool onDevice, size_t HaloDepth, size_t HaloDepthSpin>
int rhmc<floatT, onDevice, HaloDepth, HaloDepthSpin>::update_test(){

    //check unitarity of lattice, not implemented yet.
    //copy gaugefield to savedfield
    _savedField = _gaugeField;

    rootLogger.info("Smearing gauge fields");

    _smearing.SmearAll(_rhmc_param.mu_f());

    rootLogger.info("Constructing peudo-fermion fields");

    for(int i = 0; i < _no_pf; i++) {
        make_const_phi(phi_sf_container.phi_container[i], rat_inv_sf);
    }

    rootLogger.info("phi_sf: done");

    for(int i = 0; i < _no_pf; i++) {
        make_const_phi(phi_lf_container.phi_container[i], rat_inv_lf);
    }

    rootLogger.info("phi_lf: done");

    //make momenta
    generate_const_momenta();

    //get oldaction
    __attribute__((unused))floatT old_hamiltonian = get_Hamiltonian(energy_dens_old);

    //do the integration
    integrator.integrate(phi_lf_container, phi_sf_container);

    //get newaction
    __attribute__((unused))floatT new_hamiltonian = get_Hamiltonian(energy_dens_new);

    int ret;

    //make Metropolis step
    bool accept = Metropolis();

    if (accept){
        ret=1;
        rootLogger.info("Update accepted!");
    } else {
        _gaugeField=_savedField;
        _gaugeField.updateAll();
        ret=0;
        rootLogger.info("Update declined!");
    }

    return ret;
}

// method for generating the gaussian conjugate momenta of the gauge field
template<class floatT,bool onDevice, size_t HaloDepth, size_t HaloDepthSpin>
void rhmc<floatT, onDevice, HaloDepth, HaloDepthSpin>::generate_momenta() {

    _p.gauss(_rand_state);
}

// generating constant conjugate momentum field. Only for testing
template<class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin>
void rhmc<floatT, onDevice, HaloDepth, HaloDepthSpin>::generate_const_momenta() {

    _p.iterateWithConst(su3_zero<floatT>());
}


// struct for reducing the momentum part of the Hamiltonian
template<class floatT, bool onDevice, size_t HaloDepth>
struct get_momenta
{
    SU3Accessor<floatT> pAccessor;

    get_momenta(Gaugefield<floatT,onDevice,HaloDepth> &p) : pAccessor(p.getAccessor()) {
    }

    __device__ __host__ double operator()(gSite site) {

        typedef GIndexer<All,HaloDepth> GInd;

        double result = 0.0;

        for (int mu = 0; mu < 4; mu++) {
            result += tr_d(pAccessor.getLink(GInd::getSiteMu(site, mu)), pAccessor.getLink(GInd::getSiteMu(site, mu)));
        }
        return result;
    }
};

// get the Hamiltonian
template<class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin>
double rhmc<floatT, onDevice, HaloDepth, HaloDepthSpin>::get_Hamiltonian(LatticeContainer<onDevice,double> &energy_dens){

    LatticeContainer<true,double> momentum(_p.getComm(), "momenta");
    LatticeContainer<true,double> redBase2(_p.getComm(), "helper field for energy_dens");
    LatticeContainer<true,double> redBase3(_p.getComm(), "second helper field for energy_dens");
    LatticeContainer<true,double> redBase4(_p.getComm(), "third helper field for energy_dens");

    momentum.adjustSize(elems_full);
    redBase2.adjustSize(elems_full);
    redBase3.adjustSize(elems_full);
    redBase4.adjustSize(elems_full);

    momentum.template iterateOverBulk<All, HaloDepth>(get_momenta<floatT, onDevice, HaloDepth>(_p));

    double momenta;
    double gaugeact;
    double act_sf=0.0;

    // strange fermion action
    make_chi(chi, phi_sf_container.phi_container[0], rat_sf);
    redBase2.template iterateOverBulk<All, HaloDepthSpin>(get_fermion_act<floatT,onDevice,HaloDepthSpin>(chi));
    act_sf = chi.realdotProduct(chi);

    for(int i = 1; i < _no_pf; i++) {
        make_chi(chi, phi_sf_container.phi_container[i], rat_sf);
        redBase3.template iterateOverBulk<All, HaloDepthSpin>(get_fermion_act<floatT,onDevice,HaloDepthSpin>(chi));
        redBase2.template iterateOverBulk<All, HaloDepth>(add_f_r_f_r<onDevice, double>(redBase2, redBase3, 1.0, 1.0));
        act_sf += chi.realdotProduct(chi);
    }

    // momentum part
    redBase3.template iterateOverBulk<All, HaloDepth>(add_f_r_f_r<onDevice, double>(momentum, redBase2, 0.5, 1.0));
    momentum.reduce(momenta, elems_full);

    rootLogger.info(std::fixed ,  std::setprecision(12) ,  "momentum part = " ,  0.5*momenta);
    rootLogger.info("fermion action sf by   dotp = " ,  act_sf);
    double fermionaction1 = 0.0;
    redBase2.reduce(fermionaction1, elems_full);

    rootLogger.info("fermion action sf by reduce = " ,  fermionaction1);

    double act_lf=0.0;
    double mom_ferms=0.0;
    redBase3.reduce(mom_ferms, elems_full);

    rootLogger.info("mom + ferm sf = " ,  mom_ferms);

    // light fermion action
    for(int i = 0; i < _no_pf; i++) {
        make_chi(chi, phi_lf_container.phi_container[i], rat_lf);
        redBase2.template iterateOverBulk<All, HaloDepthSpin>(get_fermion_act<floatT,onDevice,HaloDepthSpin>(chi));
        redBase3.template iterateOverBulk<All, HaloDepth>(add_f_r_f_r<onDevice, double>(redBase3, redBase2, 1.0, 1.0));
        act_lf += chi.realdotProduct(chi);
    }

    rootLogger.info("fermion action lf by   dotp = " ,  act_lf);
    double fermionactionl = 0.0;
    redBase2.reduce(fermionactionl, elems_full);

    rootLogger.info("fermion action lf by reduce = " ,  fermionactionl);

    double mom_ferms_ferml = 0.0;
    redBase3.reduce(mom_ferms_ferml , elems_full);

    rootLogger.info("mom  +  ferm sf  +  ferm lf = " ,  mom_ferms_ferml);

    rootLogger.info("fermion action sf by   dotp = " ,  act_sf);

    // gauge action
    redBase2.template iterateOverBulk<All, HaloDepth>(plaquetteKernel_double<floatT, onDevice, HaloDepth, R18>(_gaugeField));
    redBase4.template iterateOverBulk<All, HaloDepth>(rectangleKernel_double<floatT, onDevice, HaloDepth, R18>(_gaugeField));

    double plaq;
    double rect;

    redBase2.reduce(plaq, elems_full);
    redBase4.reduce(rect, elems_full);

    rootLogger.info("reduced plaq = " ,  plaq);
    rootLogger.info("reduced rect = " ,  rect);
    redBase2.template iterateOverBulk<All, HaloDepth>(add_f_r_f_r<onDevice, double>(redBase2, redBase4, 5.0/3.0 , -1.0/12.0));

    double beta = _rhmc_param.beta() *3.0/5.0;

    double gauge = 0.0;
    redBase2.reduce(gauge, elems_full);
    gauge *= -beta/3.0;

    GaugeAction<floatT, onDevice, HaloDepth, R18> gaugeaction(_gaugeField);
    gaugeact = - beta * gaugeaction.symanzik();

    rootLogger.info("gauge act. by reduce = " ,  gauge);


    //CAVE: In contrast to std. textbook definitions of the gauge action: Here we find an additional factor of 3/5!
    //      This is inherited from MILC!

    energy_dens.template iterateOverBulk<All, HaloDepth>(add_f_r_f_r<onDevice, double>(redBase3, redBase2, 1.0, -beta/3.0));

    double hamiltonian = 0.5 *momenta;

    hamiltonian += act_sf + act_lf;

    hamiltonian += gaugeact;

    double H=0.0;

    energy_dens.reduce(H, elems_full);

    rootLogger.info("reduced energy_dens - individual parts = " ,  H - hamiltonian);

    rootLogger.info(std::setprecision(10) ,  "momenta = " ,  0.5 *momenta ,  " fermion sf = " ,  act_sf , " fermion lf = " ,  act_lf , " glue = " ,  gaugeact);

    return hamiltonian;
}

// The Metropolis step
template<class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin>
bool rhmc<floatT, onDevice, HaloDepth, HaloDepthSpin>::Metropolis(){

    dens_delta.template iterateOverBulk<All, HaloDepth>(add_f_r_f_r<true, double>(energy_dens_new, energy_dens_old, 1.0, -1.0));

    double delta_E=0.0;

    dens_delta.reduce(delta_E, elems_full);

    rootLogger.info("Delta H = " ,  delta_E);

    uint4 state;

    gpuError_t gpuErr;
    gpuErr = gpuMemcpy(&state, _rand_state, sizeof(uint4), gpuMemcpyDeviceToHost);
    if (gpuErr) GpuError("rhmc::Metropolis: gpuMemcpy", gpuErr);

    if (delta_E < 0.0) {
        return true;
    } else {
        double rand = get_rand<double>(&state);
        _p.getComm().root2all(rand); // Is is important so sync the random numbers between processes!
        if(rand < exp(-delta_E))
            return true;
        else
            return false;
    }
}

// Make the pseudo-spinor field
template <class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin>
void rhmc<floatT, onDevice, HaloDepth, HaloDepthSpin>::make_phi(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &phi, std::vector<floatT> rat_coeff)
{
    // generate a gaussian vector
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin> eta(phi.getComm());
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin, 14> spinorOutMulti(phi.getComm());
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin> spinortmp(phi.getComm());
    eta.gauss(_rand_state);

    int length = rat_coeff.size();

    SimpleArray<floatT, 15> rat_num(0.0);
    SimpleArray<floatT, 14> rat_den(0.0);

    // break up vector with rat. coeff. into two arrays used in multishift inverter
    for (int i = 0; i < length/2+1; ++i)
    {
        rat_num[i] = rat_coeff[i];
    }

    for (int i = 0; i < length/2; ++i)
    {
        rat_den[i] = rat_coeff[length/2+1+i];
    }

    // using the multishift inverter

    cgM.invert(dslash, spinorOutMulti, eta, rat_den, _rhmc_param.cgMax(), _rhmc_param.residue());

    phi = rat_num[0] * eta;

    for (size_t i = 0; i < 14; ++i)
    {
        spinortmp.copyFromStackToStack(spinorOutMulti, 0, i);
        phi = phi + rat_num[i+1]*spinortmp;
    }
}

// make the chi field for constructing the Hamiltonian
template <class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin>
void rhmc<floatT, onDevice, HaloDepth, HaloDepthSpin>::make_chi(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &chi,
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &phi, std::vector<floatT> rat_coeff)
{
    int length = rat_coeff.size();

    SimpleArray<floatT, 15> rat_num(0.0);
    SimpleArray<floatT, 14> rat_den(0.0);

    Spinorfield<floatT, onDevice, Even, HaloDepthSpin, 14> spinorOutMulti(phi.getComm());
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin> spinortmp(phi.getComm());


    for (int i = 0; i < length/2+1; ++i)
    {
        rat_num[i] = rat_coeff[i];
    }
    for (int i = 0; i < length/2; ++i)
    {
        rat_den[i] = rat_coeff[length/2+1+i];
    }

    // using the multishift inverter
    cgM.invert(dslash, spinorOutMulti, phi, rat_den, _rhmc_param.cgMax(), _rhmc_param.residue());

    chi = rat_num[0] * phi;

    for (size_t i = 0; i < 14; ++i)
    {
        spinortmp.copyFromStackToStack(spinorOutMulti, 0, i);
        chi = chi + rat_num[i+1]*spinortmp;
    }
}


// Make const pseudo-spinor field, only use for testing!
template <class floatT, bool onDevice, size_t HaloDepth, size_t HaloDepthSpin>
void rhmc<floatT, onDevice, HaloDepth, HaloDepthSpin>::make_const_phi(Spinorfield<floatT, onDevice, Even, HaloDepthSpin> &phi, std::vector<floatT> rat_coeff)
{
    // generate a gaussian vector
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin> eta(phi.getComm());
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin, 14> spinorOutMulti(phi.getComm());
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin> spinortmp(phi.getComm());
    eta.iterateWithConst(vect3_unity<floatT>(0));

    int length = rat_coeff.size();

    SimpleArray<floatT, 15> rat_num(0.0);
    SimpleArray<floatT, 14> rat_den(0.0);

    // break up vector with rat. coeff. into two arrays used in multishift inverter
    for (int i = 0; i < length/2+1; ++i)
    {
        rat_num[i] = rat_coeff[i];
    }

    for (int i = 0; i < length/2; ++i)
    {
        rat_den[i] = rat_coeff[length/2+1+i];
    }

    // using the multishift inverter

    cgM.invert(dslash, spinorOutMulti, eta, rat_den, _rhmc_param.cgMax(), _rhmc_param.residue());

    phi = rat_num[0] * eta;

    for (size_t i = 0; i < 14; ++i)
    {
        spinortmp.copyFromStackToStack(spinorOutMulti, 0, i);
        phi = phi + rat_num[i+1]*spinortmp;
    }
}

// explicit instantiation
#define CLASS_INIT(floatT,HALO,HALOSPIN)			\
template class rhmc<floatT,true, HALO, HALOSPIN>;

INIT_PHHS(CLASS_INIT)

