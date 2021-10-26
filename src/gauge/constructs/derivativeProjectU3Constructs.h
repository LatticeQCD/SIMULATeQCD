#ifndef DERIVATIVE_PROJECTU3CONSTRUCTS_H
#define DERIVATIVE_PROJECTU3CONSTRUCTS_H

#include "../../define.h"
#include "../../base/math/gcomplex.h"
#include "../../base/gutils.h"
#include "../../base/math/gsu3array.h"
#include "../../base/math/gsu3.h"
#include "../gaugefield.h"

#include "gsvd.h"

template<class floatT, size_t HaloDepth, CompressionType compIn = R18, CompressionType compForce = R18>
__host__ __device__ GSU3<floatT> derivativeProjectU3(gaugeAccessor<floatT,compIn> gAcc, gaugeAccessor<floatT,compForce> fAcc, gSite site, int mu) {

    typedef GIndexer<All, HaloDepth> GInd;

    GSU3<double> V;
    GSU3<double> Q;
    GSU3<double> f;
    GSU3<double> fd;
    V = gAcc.template getLink<double>(GInd::getSiteMu(site,mu));
    Q = dagger(V) * V;

    f = fAcc.template getLink<double>(GInd::getSiteMu(site,mu));
    fd = fAcc.template getLinkDagger<double>(GInd::getSiteMu(site,mu));

    double delta = 5e-5;

    //Calculate Coefficients of characteristic equation
    double c0 = tr_d(Q);
    double c1 = tr_d(Q*Q)*0.5;
    double c2 = tr_d(Q*Q*Q)*(1./3.);

    double g0;
    double g1;
    double g2;

    //Some Abbreviations
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
    GSU3<double> temp_svd;

    if (fabs(detQ/(g0*g1*g2)-1.0) > 1e-5) {
        printf("Using SVD!\n");
        temp_svd=svd3x3core<double,double>(V,sv);
        g0 = sv[0];
        g1 = sv[1];
        g2 = sv[2];
    }

    //force cut-off
    if (g0 < delta || g1 < delta || g2 < delta) {
        printf("HISQ FORCE filter active\n");
        g0 = g0 + delta;
        g1 = g1 + delta;
        g2 = g2 + delta;
        Q = Q + delta*gsu3_one<double>();
    }

    //symmetric polynomials of roots of eigenvalues
    double u = sqrt(g0) + sqrt(g1) + sqrt(g2);
    double v = sqrt(g0*g1) + sqrt(g0*g2) + sqrt(g1*g2);
    double w = sqrt(g0*g1*g2);

    double den = w*(u*v-w);

    double f0 = (-w*(u*u+v)+u*v*v)/den;
    double f1 = (-w-u*u*u+2*u*v)/den;
    double f2 = u/den;

    GSU3<double> Qinvsq = f0*gsu3_one<double>() + f1*Q + f2*Q*Q;

    double u2 = u * u;
    double u3 = u2 * u;
    double u4 = u3 * u;
    double u5 = u4 * u;
    double u6 = u5 * u;
    double u7 = u6 * u;
    double u8 = u7 * u;

    double v2 = v * v;
    double v3 = v2 * v;
    double v4 = v3 * v;
    double v5 = v4 * v;
    double v6 = v5 * v;

    double w2 = w * w;
    double w3 = w2 * w;
    double w4 = w3 * w;
    double w5 = w4 * w;

    //def Denominator to obtain Bij = Cij/den3
    double den3 = 2*w3*(u*v-w)*(u*v-w)*(u*v-w);

    double C00 = -w3*u6 + 3*v*w3*u4 + 3*v4*w*u4 - v6*u3 - 4*w4*u3 - 12*v3*w2*u3 + 16*v2*w3*u2 + 3*v5*w*u2 - 8*v*w4*u - 3*v4*w2*u + w5 + v3*w3;

    C00 = C00/den3;

    double C01 = -w2*u7 - v2*w*u6 + v4*u5 + 6*v*w2*u5 - 5*w3*u4 - v3*w*u4 - 2*v5*u3 - 6*v2*w2*u3 + 10*v*w3*u2 + 6*v4*w*u2 - 3*w4*u - 6*v3*w2*u + 2*v2*w3;

    C01 = C01/den3;

    double C02 = w2*u5 + v2*w*u4 - v4*u3 - 4*v*w2*u3 + 4*w3*u2 +3*v3*w*u2 - 3*v2*w2*u + v*w3;

    C02 = C02/den3;

    double C11 = -w*u8 - v2*u7 + 7*v*w*u6 + 4*v3*u5 - 5*w2*u5 - 16*v2*w*u4 - 4*v4*u3 + 16*v*w2*u3 - 3*w3*u2 + 12*v3*w*u2 - 12*v2*w2*u + 3*v*w3;

    C11 = C11/den3;

    double C12 = w*u6 + v2*u5 - 5*v*w*u4 - 2*v3*u3 + 4*w2*u3 + 6*v2*w*u2 - 6*v*w2*u + w3;

    C12 = C12/den3;

    double C22 = -w*u4 - v2*u3 + 3*v*w*u2 - 3*w2*u;

    C22 = C22/den3;

    //calculate P = B00*id + B01*Q + B02*Q*Q
    GSU3<double> P = C00*gsu3_one<double>() + C01*Q + C02*Q*Q;

    //calculate R = B10*id + B11*Q + B12*Q*Q
    GSU3<double> R = C01*gsu3_one<double>() + C11*Q + C12*Q*Q;

    //calculate S = B20*id + B21*Q + B22*Q*Q
    GSU3<double> S = C02*gsu3_one<double>() + C12*Q + C22*Q*Q;

    GSU3<double> temp = gsu3_zero<double>();
    GSU3<double> Vd = dagger(V);
    GSU3<double> VVd = V*dagger(V);
    GSU3<double> VQVd = V*Q*dagger(V);
    GSU3<double> PVd = P*dagger(V);
    GSU3<double> VQ = V*Q;
    GSU3<double> RVd = R*dagger(V);
    GSU3<double> VQ2 = V*Q*Q;
    GSU3<double> SVd = S*dagger(V);
    GSU3<double> QVd = Q*dagger(V);
    GSU3<double> Q2Vd = Q*Q*dagger(V);

    GCOMPLEX(double) der;

    for (int k = 0; k < 3; k++)
    for (int l = 0; l < 3; l++)
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
        //first half: dWij/dVkl
        der = 0;

        if (k == i) {
          der+=Qinvsq(l,j);
        }

        if (l == j) {
          der+=f1*VVd(i,k)+f2*VQVd(i,k);
        }

        der += f2*VVd(i,k)*Q(l,j) + V(i,j)*PVd(l,k) + VQ(i,j)*RVd(l,k) + VQ2(i,j)*SVd(l,k);
        temp(l,k) = temp(l,k) + der * f(j,i);

        //second half: dWij^+/dVkl
        der = (f1*Vd(i,k)+f2*QVd(i,k))*Vd(l,j) + f2*Vd(i,k)*QVd(l,j) + Vd(i,j)*PVd(l,k) + QVd(i,j)*RVd(l,k)+Q2Vd(i,j)*SVd(l,k);

        temp(l,k) = temp(l,k) + der * fd(j,i);
  	}

    return temp;
}

#endif // DERIVATIVE_PROJECTU3CONSTRUCTS_H