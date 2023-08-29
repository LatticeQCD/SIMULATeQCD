/*
 * projectU3.h
 *
 * J. Goswami
 *
 */

#pragma once
#include "../../define.h"
#include "../../base/math/complex.h"
#include "../../base/gutils.h"
#include "../../base/math/su3array.h"
#include "../../base/math/su3.h"
#include "../gaugefield.h"
#include "gsvd.h"
template<class floatT,size_t HaloDepth>
__host__ __device__ SU3<floatT> inline projectU3(SU3Accessor<floatT> gAcc, gSite site, int mu) {
    typedef GIndexer<All, HaloDepth> GInd;

    SU3<double> V;
    SU3<double> Q;
    V = gAcc.template getLink<double>(GInd::getSiteMu(site,mu));
    Q = dagger(V) * V;
    //Calculate Coefficients of characteristic equation
    double c0 = tr_d(Q);
    double c1 = tr_d(Q*Q)*0.5;
    double c2 = tr_d(Q*Q*Q)*(1./3.);

    double g0;
    double g1;
    double g2;

    double S_coeff = c1*(1./3.)-(c0*c0/18.);

    if (fabs(S_coeff) < 1e-7) {

        g0=c0/3.0;
        g1=g0;
        g2=g0;

    } else {

        double R_coeff = c2*0.5-(c0*c1/3.)+(c0*c0*c0/27.);
        S_coeff = sqrt(S_coeff);
        double R_cOverS_ccubed = R_coeff/(S_coeff*S_coeff*S_coeff);
        double Theta;

        if (fabs(R_cOverS_ccubed) < 1.0) {
        Theta = acos(R_cOverS_ccubed);

        } else {
            if (R_coeff > 0) {
              Theta = 0.0;
            } else {
              Theta = M_PI;
            }
        }

        //Solutions to characteristic equation
        g0 = c0*(1./3.)+2.0*S_coeff*cos(Theta/3.0-2*M_PI/3.0);
        g1 = c0*(1./3.)+2.0*S_coeff*cos(Theta/3.0);
        g2 = c0*(1./3.)+2.0*S_coeff*cos(Theta/3.0+2*M_PI/3.0);
    }

    double detQ = realdet(Q);
    double sv[3];

    SU3<double> temp_svd;
    if (fabs(detQ/(g0*g1*g2)-1.0) > 1e-5) {
        printf("Using SVD in smearing\n");
        temp_svd=svd3x3core<double,double>(V,sv);
        return temp_svd;
    } else {
        double u   = sqrt(g0) + sqrt(g1) + sqrt(g2);
        double v   = sqrt(g0*g1) + sqrt(g0*g2) + sqrt(g1*g2);
        double w   = sqrt(g0*g1*g2);
        double den = w*(u*v-w);
        double f0  = (-w*(u*u+v)+u*v*v)/den;
        double f1  = (-w-u*u*u+2*u*v)/den;
        double f2  = u/den;

        SU3<double> Qinvsq = f0*su3_one<double>() + f1*Q + f2*Q*Q;

        return V*Qinvsq;
    }
}

