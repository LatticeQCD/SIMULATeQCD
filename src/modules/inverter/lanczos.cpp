#include "lanczos.h"

Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> lanczos::lanczos(const int &num_vec_in) {
    // Setup
    lambda_vec.clear();
    spinor_vec.clear();
    spinor_count = num_vec_in;
    spinor_vec.reserve(spinor_count);
    lambda_vec.reserve(spinor_count);
    CommunicationBase &commBase = this->getComm();

    // Allocate spinorfields
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> vec(commBase);
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> q_m1(commBase);

    // Allocate scalars
    double normsq;
    int m1, k1, diff;

    // old Alg class variables 
    int m_lan, k_lan;

    // Initialize startvector with random Gaussian
    grnd_state<false> h_rand;
    h_rand.make_rng_state(1234);
    vec.gauss(h_rand.state);

    normsq = vec.realdotProduct(vec);
    vec *= static_cast<floatT>(1.0) / normsq;
    m1 = 0;

    diff = m_lan - k_lan;
    k1 = m1 - diff;
}


int lanczos::FLan(double **T,
			GPUcvect3array<SPcomplex> *q_k2,
			double *beta_k1, int m1, int g1) {
    CommunicationBase &commBase = this->getComm();

    int   i, p, m1_mod;
    double s, u;

    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> R(commBase);

    R = R_lan;

    int vsizeh;

    vsizeh = latticeSize.sizeh();

    s = 0;

    T[g1+2][g1+2]=s;
    T[g1+2][g1+1]=*beta_k1;
    T[g1+1][g1+2]=*beta_k1;

    R->copyFrom(*q_k2,vsizeh,0,(g1+1)*vsizeh);

    m1_mod=m1;
    i=0;
    //start main loop over i
    //----------------------
    while(i<m1-g1-2){

        // d_loc_r->LanVec(d_loc_s->getAccessor(),T[g1+2+i][g1+2+i],R->getAccessor((g1+1+i)*vsizeh),T[g1+2+i][g1+1+i],R->getAccessor((g1+i)*vsizeh),0,vsizeh);
        // MGS(R,d_loc_r,g1+2+i,vsizeh,0);
        // s=d_loc_r->ReDotProdSumD(d_loc_r, vsizeh);
        s=sqrt(s);
        T[g1+3+i][g1+2+i]=s;

        if(s>1e-7){
            T[g1+2+i][g1+3+i]=s;
            u = 1.0/s;
            // R->VecTimesFloat(d_loc_r->getAccessor(),u,(g1+2+i)*vsizeh,vsizeh);
            p=g1+2+i;
            
            s = 0;
            T[g1+3+i][g1+3+i]=s;
            i++;
        } else {
            m1_mod=g1+3+i-1;
            i=m1-g1-2;
        }
    } // end of main loop

    return m1_mod;

}