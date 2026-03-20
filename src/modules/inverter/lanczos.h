# pragma once

#include "../../spinor/spinorfield.h"



class oldAlgTRLan {
private:
    std::string *cname;
    std::string *fname;
    int _devId;
    bool _EVCalculated;

    // The Lattice Size
    LatticeSize latticeSize;

    // The argument structure for the trlan algorithm
    TRLanArg *alg_trlan_arg;
 
    // The argument structure for the inversion parameters
    CgArg *cg_arg;

    //The argument structure for the polynomial filter
    FilterArg *filter_arg;
    int tschebyscheff_filter;
    int tschebyscheff_order;
    double tschebyscheff_alpha;
    double tschebyscheff_beta;
    double tschebyscheff_gamma;
    int exponential_filter;
    int exponential_order;
    double exponential_alpha;
    double exponential_beta;
    int filtertype;
    int ordertype;

    // Node checkerboard size of the fermion field 
    int vsize;
    int vsizeh;
    int fsize;
    int fsizeh;

    int m_lan, k_lan;
    int ev_write, ev_read;
    int restarts;

    // Spionor todo
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> q;
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> q_m1;

    double ktoll;
    float mass;

    double **T_lan;
    double *T_lan_h;
    double *T_lan_d;
    double *a_lan_h;
    double *b_lan_h;
    double beta_m;
    double alpha_k1;
    double beta_k1;

    
}

class lanczos {
private:
    spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> R_lan;


public:
    Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> lanczos(const int &num_vec_in);
    int FLan(double **T,
			GPUcvect3array<SPcomplex> *q_k2,
			double *beta_k1, int m1, int g1);

};