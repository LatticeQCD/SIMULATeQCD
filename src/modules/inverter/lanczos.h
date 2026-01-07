# pragma once

#include "../../spinor/spinorfield.h"


class lanczos {
private:
    spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> R_lan;


public:
    Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> lanczos(const int &num_vec_in);
    int FLan(double **T,
			GPUcvect3array<SPcomplex> *q_k2,
			double *beta_k1, int m1, int g1);

};