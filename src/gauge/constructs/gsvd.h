/*
 * gsvd.h
 *
 *  Created on: May 31, 2011
 *      Author: mathias-wagner
 */

#ifndef GSVD_HCU_
#define GSVD_HCU_
#include "../../base/math/su3.h"
#include "../../base/math/su3array.h"

/* define precision for chopping small values */
#define SVD3x3_PREC 5e-16

/* defines that allow to remap input arrays easily,
   in the routine internal arrays in double precision are used */
#define A00re real(AA.getLink00())
#define A00im imag(AA.getLink00())
#define A01re real(AA.getLink01())
#define A01im imag(AA.getLink01())
#define A02re real(AA.getLink02())
#define A02im imag(AA.getLink02())
#define A10re real(AA.getLink10())
#define A10im imag(AA.getLink10())
#define A11re real(AA.getLink11())
#define A11im imag(AA.getLink11())
#define A12re real(AA.getLink12())
#define A12im imag(AA.getLink12())
#define A20re real(AA.getLink20())
#define A20im imag(AA.getLink20())
#define A21re real(AA.getLink21())
#define A21im imag(AA.getLink21())
#define A22re real(AA.getLink22())
#define A22im imag(AA.getLink22())
#define U00re U[0][0][0]
#define U00im U[0][0][1]
#define U01re U[0][1][0]
#define U01im U[0][1][1]
#define U02re U[0][2][0]
#define U02im U[0][2][1]
#define U10re U[1][0][0]
#define U10im U[1][0][1]
#define U11re U[1][1][0]
#define U11im U[1][1][1]
#define U12re U[1][2][0]
#define U12im U[1][2][1]
#define U20re U[2][0][0]
#define U20im U[2][0][1]
#define U21re U[2][1][0]
#define U21im U[2][1][1]
#define U22re U[2][2][0]
#define U22im U[2][2][1]
#define V00re V[0][0][0]
#define V00im V[0][0][1]
#define V01re V[0][1][0]
#define V01im V[0][1][1]
#define V02re V[0][2][0]
#define V02im V[0][2][1]
#define V10re V[1][0][0]
#define V10im V[1][0][1]
#define V11re V[1][1][0]
#define V11im V[1][1][1]
#define V12re V[1][2][0]
#define V12im V[1][2][1]
#define V20re V[2][0][0]
#define V20im V[2][0][1]
#define V21re V[2][1][0]
#define V21im V[2][1][1]
#define V22re V[2][2][0]
#define V22im V[2][2][1]
#define b00 P[0][0][0]
#define b01 P[0][1][0]
#define b02 P[0][2][0]
#define b10 P[1][0][0]
#define b11 P[1][1][0]
#define b12 P[1][2][0]
#define b20 P[2][0][0]
#define b21 P[2][1][0]
#define b22 P[2][2][0]

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wregister"
/************************************************************************
 * SVD of 2x2 real matrix brought to the form:                          *
 *  [ a00 a01]                                                          *
 *  [   0 a11]                                                          *
 * This routine eliminates off-diagonal element, handling special cases *
 ************************************************************************/
template<class svdfloatT>
__device__ __host__ inline int svd2x2bidiag(svdfloatT *a00, svdfloatT *a01, svdfloatT *a11, svdfloatT U2[2][2], svdfloatT V2[2][2])
{
    register svdfloatT sinphi, cosphi, tanphi, cotphi;
    register svdfloatT a, b, min, max, abs00, abs01, abs11;
    register svdfloatT lna01a11, lna00, ln_num, tau, t;
    register svdfloatT P00, P01, P10, P11;
    register int isign;

    U2[0][0]=1.0f; U2[0][1]=0.0f;
    U2[1][0]=0.0f; U2[1][1]=1.0f;
    V2[0][0]=1.0f; V2[0][1]=0.0f;
    V2[1][0]=0.0f; V2[1][1]=1.0f;

    if( *a00==0 ) {
        if( *a11==0 ) {
            cosphi=1.0f;
            sinphi=0.0f;
        } else {
            if( fabs(*a11)>fabs(*a01) ) {
                cotphi=-(*a01)/(*a11);
                sinphi=1/sqrt(1+cotphi*cotphi);
                cosphi=cotphi*sinphi;
            } else {
                tanphi=-(*a11)/(*a01);
                cosphi=1/sqrt(1+tanphi*tanphi);
                sinphi=tanphi*cosphi;
            }
        }
        /* multiply matrix A */
        (*a00)=cosphi*(*a01)-sinphi*(*a11);
        (*a01)=0.0f; (*a11)=0.0f;
        /* exchange columns in matrix V */
        V2[0][0]=0.0f; V2[0][1]=1.0f;
        V2[1][0]=1.0f; V2[1][1]=0.0f;
        /* U is just Givens rotation */
        U2[0][0]= cosphi; U2[0][1]= sinphi;
        U2[1][0]=-sinphi; U2[1][1]= cosphi;

    } else if( *a11==0 ) {
        if( *a01==0 ) {
            cosphi=1.0f;
            sinphi=0.0f;
        } else {
            if( fabs(*a01)>fabs(*a00) ) {
                cotphi=-(*a00)/(*a01);
                sinphi=1/sqrt(1+cotphi*cotphi);
                cosphi=cotphi*sinphi;
            }
            else {
                tanphi=-(*a01)/(*a00);
                cosphi=1/sqrt(1+tanphi*tanphi);
                sinphi=tanphi*cosphi;
            }
        }
        /* multiply matrix A */
        (*a00)=cosphi*(*a00)-sinphi*(*a01);
        (*a01)=0.0f; (*a11)=0.0f;
        /* V is just Givens rotation */
        V2[0][0]= cosphi; V2[0][1]= sinphi;
        V2[1][0]=-sinphi; V2[1][1]= cosphi;

    } else if( *a01==0 ) { /* nothing to be done */
      ;
    } else {
        /* need to calculate ( a11^2+a01^2-a00^2 )/( 2*a00*a01 )
           avoiding overflow/underflow,
           use logarithmic coding */
        abs01=fabs(*a01); abs11=fabs(*a11);
        if(abs01>abs11) {
            min=abs11; max=abs01;
        } else {
            min=abs01; max=abs11;
        }
        a=min/max;
        lna01a11=2*log(max)+log(1+a*a);

        abs00=fabs(*a00);
        lna00=2*log(abs00);
        if( lna01a11>lna00 ) {
            /* subtract smaller from larger, overall "+" */
            isign=1;
            ln_num=lna01a11+log(1.-exp(lna00-lna01a11));
        } else {
            /* subtract larger from smaller, need to change order, overall "-" */
            isign=-1;
            ln_num=lna00+log(1.-exp(lna01a11-lna00));
        }
        a=ln_num-log(2.0f)-log(abs00)-log(abs01);
        tau=exp(a);
        tau*=isign;
        if(*a00<0.) {
            tau*=-1;
          }
        if(*a01<0.) {
            tau*=-1;
          }

        /* calculate b=sqrt(1+tau^2) */
        a=fabs(tau);
        if( a>1. ) {
            max=a; min=1.0f;
        } else {
            max=1.0f; min=a;
        }
        if( min==0 ) {
            b = max;
        } else {
            a = min/max;
            b = max*sqrt(1+a*a);
        }
        if(tau>=0.) {
            t = 1.0/(tau + b);
        } else {
            t = 1.0/(tau - b);
        }

        /* calculate b=sqrt(1+t^2) */
        a=fabs(t);
        if( a>1. ) {
            max=a; min=1.0f;
        } else {
            max=1.0f; min=a;
        }
        if( min==0 ) {
            b = max;
        } else {
            a = min/max;
            b = max*sqrt(1+a*a);
        }
        cosphi=1./b;
        sinphi=t*cosphi;

        /* transform matrix A so it has othogonal columns */
        P00= cosphi*(*a00)-sinphi*(*a01);
        P10=-sinphi*(*a11);
        P01= sinphi*(*a00)+cosphi*(*a01);
        P11= cosphi*(*a11);

        /* prepare V  */
        V2[0][0]= cosphi; V2[0][1]= sinphi;
        V2[1][0]=-sinphi; V2[1][1]= cosphi;

        /* make column with the largest norm first column */
        if( sqrt(P00*P00+P10*P10)<sqrt(P01*P01+P11*P11) ) {
            a=P00; P00=P01; P01=a;
            a=P10; P10=P11; P11=a;
            /* exchange columns in matrix V2 */
            a=V2[0][0]; V2[0][0]=V2[0][1]; V2[0][1]=a;
            a=V2[1][0]; V2[1][0]=V2[1][1]; V2[1][1]=a;
        }

        /* calculate left Givens rotation and diagonalize */
        if( P10==0 ) {
            cosphi=1.0f;
            sinphi=0.0f;
        } else {
            if( fabs(P10)>fabs(P00) ) {
              cotphi=-P00/P10;
              sinphi=1/sqrt(1+cotphi*cotphi);
              cosphi=cotphi*sinphi;
            } else {
              tanphi=-P10/P00;
              cosphi=1/sqrt(1+tanphi*tanphi);
              sinphi=tanphi*cosphi;
            }
        }
        *a00=P00*cosphi-P10*sinphi;
        *a01=0.0f;
        *a11=P01*sinphi+P11*cosphi;

        /* U is just Givens rotation */
        U2[0][0]= cosphi; U2[0][1]= sinphi;
        U2[1][0]=-sinphi; U2[1][1]= cosphi;
    }

    return 0;
}

/* Input:  A -- 3x3 complex matrix,   */
/* Output: W -- Projectet U(3) matrix */



template<class floatT, class svdfloatT>
__device__ __host__ SU3<floatT> svd3x3core(const SU3<floatT>& AA, floatT* sv){

  /******************************************
   *  sigma[3] -- singular values,          *
   *       U,V -- U(3) matrices such, that  *
   *       A=U Sigma V^+                    *
   ******************************************/
  svdfloatT U[3][3][2], V[3][3][2]; // double sigma[3];

  svdfloatT Ad[3][3][2], P[3][3][2], Q[3][3][2];
  svdfloatT U1[3][3][2], U2[3][3][2], U3[3][3][2], V1[3][3][2], V2[3][3][2];
  svdfloatT UO3[3][3], VO3[3][3], v[3][2];
  svdfloatT UO2[2][2], VO2[2][2];
  register svdfloatT a, b, c, d, factor, norm, min, max, taure, tauim, beta;
  register svdfloatT m11, m12, m22, dm, lambdamax, cosphi, sinphi, tanphi, cotphi;
  register int i, j, iter;

  /* format of external matrices A, U and V can be arbitrary,
     therefore this routine uses defines (above) to access them
     and never reads A, U and V directly */

  /* original matrix can be in single precision,
     so copy it into double */
  Ad[0][0][0]=(svdfloatT)A00re; Ad[0][0][1]=(svdfloatT)A00im;
  Ad[0][1][0]=(svdfloatT)A01re; Ad[0][1][1]=(svdfloatT)A01im;
  Ad[0][2][0]=(svdfloatT)A02re; Ad[0][2][1]=(svdfloatT)A02im;
  Ad[1][0][0]=(svdfloatT)A10re; Ad[1][0][1]=(svdfloatT)A10im;
  Ad[1][1][0]=(svdfloatT)A11re; Ad[1][1][1]=(svdfloatT)A11im;
  Ad[1][2][0]=(svdfloatT)A12re; Ad[1][2][1]=(svdfloatT)A12im;
  Ad[2][0][0]=(svdfloatT)A20re; Ad[2][0][1]=(svdfloatT)A20im;
  Ad[2][1][0]=(svdfloatT)A21re; Ad[2][1][1]=(svdfloatT)A21im;
  Ad[2][2][0]=(svdfloatT)A22re; Ad[2][2][1]=(svdfloatT)A22im;

  i=0; j=0;

  /* *** Step 1: build first left reflector v,
                 calculate first left rotation U1,
                 apply to the original matrix A *** */
  /* calculate norm of ( A[10] )
                       ( A[20] ) vector
     with minimal loss of accuracy (similar to BLAS) */
  c = 1.0f;
  factor = fabs( Ad[1][0][0] );
  a = fabs( Ad[1][0][1] );
  if( a!=0 ) {
    if( factor < a ) {
      c = 1.0f + (factor/a)*(factor/a);
      factor = a;
    }
    else {
      c = 1.0f + (a/factor)*(a/factor);
    }
  }
  a = fabs( Ad[2][0][0] );
  if( a!=0 ) {
    if( factor < a ) {
      c = 1.0f + c*(factor/a)*(factor/a);
      factor = a;
    }
    else {
      c += (a/factor)*(a/factor);
    }
  }
  a = fabs( Ad[2][0][1] );
  if( a!=0 ) {
    if( factor < a ) {
      c = 1.0f + c*(factor/a)*(factor/a);
      factor = a;
    }
    else {
      c += (a/factor)*(a/factor);
    }
  }
  norm = factor*sqrt(c);

  if( norm==0 && Ad[0][0][1]==0 ) { /* no rotation needed */
#ifdef SVD3x3_DEBUG
    printf("Step 1: no rotation needed\n");
#endif /* SVD3x3_DEBUG */
    U1[0][0][0]=1.0f; U1[0][0][1]=0.0f;
    U1[0][1][0]=0.0f; U1[0][1][1]=0.0f;
    U1[0][2][0]=0.0f; U1[0][2][1]=0.0f;
    U1[1][0][0]=0.0f; U1[1][0][1]=0.0f;
    U1[1][1][0]=1.0f; U1[1][1][1]=0.0f;
    U1[1][2][0]=0.0f; U1[1][2][1]=0.0f;
    U1[2][0][0]=0.0f; U1[2][0][1]=0.0f;
    U1[2][1][0]=0.0f; U1[2][1][1]=0.0f;
    U1[2][2][0]=1.0f; U1[2][2][1]=0.0f;
    P[0][0][0]=Ad[0][0][0]; P[0][0][1]=Ad[0][0][1];
    P[1][0][0]=Ad[1][0][0]; P[1][0][1]=Ad[1][0][1];
    P[2][0][0]=Ad[2][0][0]; P[2][0][1]=Ad[2][0][1];
    P[0][1][0]=Ad[0][1][0]; P[0][1][1]=Ad[0][1][1];
    P[1][1][0]=Ad[1][1][0]; P[1][1][1]=Ad[1][1][1];
    P[2][1][0]=Ad[2][1][0]; P[2][1][1]=Ad[2][1][1];
    P[0][2][0]=Ad[0][2][0]; P[0][2][1]=Ad[0][2][1];
    P[1][2][0]=Ad[1][2][0]; P[1][2][1]=Ad[1][2][1];
    P[2][2][0]=Ad[2][2][0]; P[2][2][1]=Ad[2][2][1];
  }
  else {


    /* get the norm of full first column of A matrix */
    c=1.0f;
    factor = norm;
    a = fabs( Ad[0][0][0] );
    if( a!=0 ) {
      if( factor < a ) {
        c = 1.0f + (factor/a)*(factor/a);
        factor = a;
      }
      else {
        c += (a/factor)*(a/factor);
      }
    }
    a = fabs( Ad[0][0][1] );
    if( a!=0 ) {
      if( factor < a ) {
        c = 1.0f + c*(factor/a)*(factor/a);
        factor = a;
      }
      else {
        c += (a/factor)*(a/factor);
      }
    }
    beta = factor*sqrt(c); /* norm of first column */
    if( Ad[0][0][0]>0 ) {
      beta = -beta;
    }

#ifdef SVD3x3_DEBUG
    printf("beta=%28.18e\n",beta);
#endif /* SVD3x3_DEBUG */


    /* a=Re(A_00-beta), b=Im(A_00-beta) */
    a=Ad[0][0][0]-beta; b=Ad[0][0][1];
    /* norm=sqrt(a^2+b^2) */
    c=fabs(a); d=fabs(b);
    if( c>d ) {
      max=c; min=d;
    }
    else {
      max=d; min=c;
    }
    if( min==0 ) {
      norm = max;
    }
    else {
      c = min/max;
      norm = max*sqrt(1+c*c);
    }
    /* c=a/norm, d=b/norm */
    c=a/norm; d=b/norm;


    /* construct reflector (vector "v" for Householder transformation)
       v_0=1 */
    v[0][0]=1.0f; v[0][1]=0.0f;
    /* v_1=A_10/(A_00-beta)=A_10/(a+ib)=(A_10*(a-ib))/norm^2=(A_10/norm)*((a-ib)/norm)
          =(A_10/norm)*(c-id)=|a=Re(A_10)/norm,b=Im(A_10)/norm|=(a+ib)*(c-id)
          =(a*c+b*d)+i(b*c-a*d) */
    a=Ad[1][0][0]/norm; b=Ad[1][0][1]/norm;
    v[1][0]=a*c+b*d;
    v[1][1]=b*c-a*d;
    /* v_2=A_20/(A_00-beta)=A_20/(a+ib)=(A_20*(a-ib))/norm^2=(A_20/norm)*((a-ib)/norm)
          =(A_20/norm)*(c-id)=|a=Re(A_20)/norm,b=Im(A_20)/norm|=(a+ib)*(c-id)
          =(a*c+b*d)+i(b*c-a*d) */
    a=Ad[2][0][0]/norm; b=Ad[2][0][1]/norm;
    v[2][0]=a*c+b*d;
    v[2][1]=b*c-a*d;
#ifdef SVD3x3_DEBUG
for(i=0;i<3;i++) {
  printf("v[%d].re=%28.18e  v[%d].im=%28.18e\n",i,v[i][0],i,v[i][1]);
}
#endif /* SVD3x3_DEBUG */

    /* calcualate tau (coefficient for reflector) */
    taure=(beta-Ad[0][0][0])/beta;
    tauim=(Ad[0][0][1])/beta;


    /* assemble left unitary matrix U1=I-tau^+*v*v^+ (store in U1[3][3][2])
       U1_00=A_00/beta */
    U1[0][0][0]=(Ad[0][0][0])/beta;
    U1[0][0][1]=(Ad[0][0][1])/beta;
    /* U1_10=A_10/beta */
    U1[1][0][0]=(Ad[1][0][0])/beta;
    U1[1][0][1]=(Ad[1][0][1])/beta;
    /* U1_20=A_20/beta */
    U1[2][0][0]=(Ad[2][0][0])/beta;
    U1[2][0][1]=(Ad[2][0][1])/beta;
    /* U1_01=-tau^+*v_1^+=-(tau*v_1)^+ */
    U1[0][1][0]=-(taure*v[1][0]-tauim*v[1][1]);
    U1[0][1][1]=taure*v[1][1]+tauim*v[1][0];
    /* U1_11=1-tau^+*v_1*v_1^+ */
    a=v[1][0]*v[1][0]+v[1][1]*v[1][1];
    U1[1][1][0]=1-taure*a;
    U1[1][1][1]=tauim*a;
    /* U1_21=-tau^+*v_2*v_1^+ */
      /* v_2*v_1^+ */
      a=v[2][0]*v[1][0]+v[2][1]*v[1][1];
      b=-v[2][0]*v[1][1]+v[2][1]*v[1][0];
    U1[2][1][0]=-(taure*a+tauim*b);
    U1[2][1][1]=-(taure*b-tauim*a);
    /* U1_02=-tau^+*v_2^+=-(tau*v_2)^+ */
    U1[0][2][0]=-(taure*v[2][0]-tauim*v[2][1]);
    U1[0][2][1]=taure*v[2][1]+tauim*v[2][0];
    /* U1_12=-tau^+*v_1*v_2^+ */
      /* v_1*v_2^+ */
      a=v[1][0]*v[2][0]+v[1][1]*v[2][1];
      b=-v[1][0]*v[2][1]+v[1][1]*v[2][0];
    U1[1][2][0]=-(taure*a+tauim*b);
    U1[1][2][1]=-(taure*b-tauim*a);
    /* U1_22=1-tau^+*v_2*v_2^+ */
    a=v[2][0]*v[2][0]+v[2][1]*v[2][1];
    U1[2][2][0]=1-taure*a;
    U1[2][2][1]=tauim*a;


    /* apply the transformation to A matrix and store the result in P
       P=U^+A */
    P[0][0][0]=beta;
    P[0][0][1]=0;
    P[1][0][0]=0;
    P[1][0][1]=0;
    P[2][0][0]=0;
    P[2][0][1]=0;
    /* P_01=U1_00^+*A_01+U1_10^+*A_11+U1_20^+*A_21 */
    P[0][1][0]=U1[0][0][0]*Ad[0][1][0]+U1[0][0][1]*Ad[0][1][1]
              +U1[1][0][0]*Ad[1][1][0]+U1[1][0][1]*Ad[1][1][1]
              +U1[2][0][0]*Ad[2][1][0]+U1[2][0][1]*Ad[2][1][1];
    P[0][1][1]=U1[0][0][0]*Ad[0][1][1]-U1[0][0][1]*Ad[0][1][0]
              +U1[1][0][0]*Ad[1][1][1]-U1[1][0][1]*Ad[1][1][0]
              +U1[2][0][0]*Ad[2][1][1]-U1[2][0][1]*Ad[2][1][0];
    /* P_02=U1_00^+*A_02+U1_10^+*A_12+U1_20^+*A_22 */
    P[0][2][0]=U1[0][0][0]*Ad[0][2][0]+U1[0][0][1]*Ad[0][2][1]
              +U1[1][0][0]*Ad[1][2][0]+U1[1][0][1]*Ad[1][2][1]
              +U1[2][0][0]*Ad[2][2][0]+U1[2][0][1]*Ad[2][2][1];
    P[0][2][1]=U1[0][0][0]*Ad[0][2][1]-U1[0][0][1]*Ad[0][2][0]
              +U1[1][0][0]*Ad[1][2][1]-U1[1][0][1]*Ad[1][2][0]
              +U1[2][0][0]*Ad[2][2][1]-U1[2][0][1]*Ad[2][2][0];
    /* P_11=U1_01^+*A_01+U1_11^+*A_11+U1_21^+*A_21 */
    P[1][1][0]=U1[0][1][0]*Ad[0][1][0]+U1[0][1][1]*Ad[0][1][1]
              +U1[1][1][0]*Ad[1][1][0]+U1[1][1][1]*Ad[1][1][1]
              +U1[2][1][0]*Ad[2][1][0]+U1[2][1][1]*Ad[2][1][1];
    P[1][1][1]=U1[0][1][0]*Ad[0][1][1]-U1[0][1][1]*Ad[0][1][0]
              +U1[1][1][0]*Ad[1][1][1]-U1[1][1][1]*Ad[1][1][0]
              +U1[2][1][0]*Ad[2][1][1]-U1[2][1][1]*Ad[2][1][0];
    /* P_12=U1_01^+*A_02+U1_11^+*A_12+U1_21^+*A_22 */
    P[1][2][0]=U1[0][1][0]*Ad[0][2][0]+U1[0][1][1]*Ad[0][2][1]
              +U1[1][1][0]*Ad[1][2][0]+U1[1][1][1]*Ad[1][2][1]
              +U1[2][1][0]*Ad[2][2][0]+U1[2][1][1]*Ad[2][2][1];
    P[1][2][1]=U1[0][1][0]*Ad[0][2][1]-U1[0][1][1]*Ad[0][2][0]
              +U1[1][1][0]*Ad[1][2][1]-U1[1][1][1]*Ad[1][2][0]
              +U1[2][1][0]*Ad[2][2][1]-U1[2][1][1]*Ad[2][2][0];
    /* P_21=U1_02^+*A_01+U1_12^+*A_11+U1_22^+*A_21 */
    P[2][1][0]=U1[0][2][0]*Ad[0][1][0]+U1[0][2][1]*Ad[0][1][1]
              +U1[1][2][0]*Ad[1][1][0]+U1[1][2][1]*Ad[1][1][1]
              +U1[2][2][0]*Ad[2][1][0]+U1[2][2][1]*Ad[2][1][1];
    P[2][1][1]=U1[0][2][0]*Ad[0][1][1]-U1[0][2][1]*Ad[0][1][0]
              +U1[1][2][0]*Ad[1][1][1]-U1[1][2][1]*Ad[1][1][0]
              +U1[2][2][0]*Ad[2][1][1]-U1[2][2][1]*Ad[2][1][0];
    /* P_22=U1_02^+*A_02+U1_12^+*A_12+U1_22^+*A_22 */
    P[2][2][0]=U1[0][2][0]*Ad[0][2][0]+U1[0][2][1]*Ad[0][2][1]
              +U1[1][2][0]*Ad[1][2][0]+U1[1][2][1]*Ad[1][2][1]
              +U1[2][2][0]*Ad[2][2][0]+U1[2][2][1]*Ad[2][2][1];
    P[2][2][1]=U1[0][2][0]*Ad[0][2][1]-U1[0][2][1]*Ad[0][2][0]
              +U1[1][2][0]*Ad[1][2][1]-U1[1][2][1]*Ad[1][2][0]
              +U1[2][2][0]*Ad[2][2][1]-U1[2][2][1]*Ad[2][2][0];


  }
#ifdef SVD3x3_DEBUG
printf("Left unitary matrix U1:\n");
    for(i=0;i<3;i++)for(j=0;j<3;j++) {
      printf( "U1[%d][%d].re=%26.18e  U1[%d][%d].im=%26.18e\n",
              i, j, U1[i][j][0], i, j, U1[i][j][1] );
    }
#endif /* SVD3x3_DEBUG */



  /* *** Step 2: build first right reflector v,
                 calculate first right rotation V1,
                 apply to the matrix P from step 1 *** */
  /* calculate norm of ( P[02] )
     with minimal loss of accuracy */
  a=fabs( P[0][2][0] ); b=fabs( P[0][2][1] );
  /* norm=sqrt(a^2+b^2) */
  if( a>b ) {
    max=a; min=b;
  }
  else {
    max=b; min=a;
  }
  if( min==0 ) {
    norm = max;
  }
  else {
    c = min/max;
    norm = max*sqrt(1+c*c);
  }

  if( norm==0 && P[0][1][1]==0 ) { /* no rotation needed */
#ifdef SVD3x3_DEBUG
    printf("Step 2: no rotation needed\n");
#endif /* SVD3x3_DEBUG */
    V1[0][0][0]=1.0f; V1[0][0][1]=0.0f;
    V1[0][1][0]=0.0f; V1[0][1][1]=0.0f;
    V1[0][2][0]=0.0f; V1[0][2][1]=0.0f;
    V1[1][0][0]=0.0f; V1[1][0][1]=0.0f;
    V1[1][1][0]=1.0f; V1[1][1][1]=0.0f;
    V1[1][2][0]=0.0f; V1[1][2][1]=0.0f;
    V1[2][0][0]=0.0f; V1[2][0][1]=0.0f;
    V1[2][1][0]=0.0f; V1[2][1][1]=0.0f;
    V1[2][2][0]=1.0f; V1[2][2][1]=0.0f;
    Q[0][0][0]=P[0][0][0]; Q[0][0][1]=P[0][0][1];
    Q[1][0][0]=P[1][0][0]; Q[1][0][1]=P[1][0][1];
    Q[2][0][0]=P[2][0][0]; Q[2][0][1]=P[2][0][1];
    Q[0][1][0]=P[0][1][0]; Q[0][1][1]=P[0][1][1];
    Q[1][1][0]=P[1][1][0]; Q[1][1][1]=P[1][1][1];
    Q[2][1][0]=P[2][1][0]; Q[2][1][1]=P[2][1][1];
    Q[0][2][0]=P[0][2][0]; Q[0][2][1]=P[0][2][1];
    Q[1][2][0]=P[1][2][0]; Q[1][2][1]=P[1][2][1];
    Q[2][2][0]=P[2][2][0]; Q[2][2][1]=P[2][2][1];
  }
  else {
    /* get the norm of (P_01 P_02) row vector */
    c=1.0f;
    factor = norm;
    a = fabs( P[0][1][0] );
    if( a!=0 ) {
      if( factor < a ) {
        c = 1.0f + (factor/a)*(factor/a);
        factor = a;
      }
      else {
        c += (a/factor)*(a/factor);
      }
    }
    a = fabs( P[0][1][1] );
    if( a!=0 ) {
      if( factor < a ) {
        c = 1.0f + c*(factor/a)*(factor/a);
        factor = a;
      }
      else {
        c += (a/factor)*(a/factor);
      }
    }
    beta = factor*sqrt(c); /* norm of (P_01 P_02) row vector */
    if( P[0][1][0]>0 ) {
      beta = -beta;
    }

#ifdef SVD3x3_DEBUG
    printf("beta=%28.18e\n",beta);
#endif /* SVD3x3_DEBUG */


    /* a=Re(P_01^+-beta), b=Im(P_01^+-beta) */
    a=P[0][1][0]-beta; b=-P[0][1][1];
    /* norm=sqrt(a^2+b^2) */
    c=fabs(a); d=fabs(b);
    if( c>d ) {
      max=c; min=d;
    }
    else {
      max=d; min=c;
    }
    if( min==0 ) {
      norm = max;
    }
    else {
      c = min/max;
      norm = max*sqrt(1+c*c);
    }
    /* c=a/norm, d=b/norm */
    c=a/norm; d=b/norm;


    /* construct reflector (vector "v" for Householder transformation) */
    /* v_0=0 */
    v[0][0]=0.0f; v[0][1]=0.0f;
    /* v_1=1 */
    v[1][0]=1.0f; v[1][1]=0.0f;
    /* v_2=P_02^+/(P_01^+-beta)=P_02^+/(a+ib)=(P_02^+*(a-ib))/norm^2=(P_02^+/norm)*((a-ib)/norm)
          =(P_02^+/norm)*(c-id)=|a=Re(P_02^+)/norm,b=Im(P_02^+)/norm|=(a+ib)*(c-id)
          =(a*c+b*d)+i(b*c-a*d) */
    a=P[0][2][0]/norm; b=-P[0][2][1]/norm;
    v[2][0]=a*c+b*d;
    v[2][1]=b*c-a*d;
#ifdef SVD3x3_DEBUG
for(i=0;i<3;i++) {
  printf("v[%d].re=%28.18e  v[%d].im=%28.18e\n",i,v[i][0],i,v[i][1]);
}
#endif /* SVD3x3_DEBUG */

    /* calcualate tau (coefficient for reflector) */
    taure=(beta-P[0][1][0])/beta;
    tauim=-P[0][1][1]/beta;

    /* assemble right unitary matrix V1=I-tau^+*v*v^+ (store in V1[3][3][2]) */
    V1[0][0][0]=1.0f;
    V1[0][0][1]=0.0f;
    V1[1][0][0]=0.0f;
    V1[1][0][1]=0.0f;
    V1[2][0][0]=0.0f;
    V1[2][0][1]=0.0f;
    V1[0][1][0]=0.0f;
    V1[0][1][1]=0.0f;
    V1[0][2][0]=0.0f;
    V1[0][2][1]=0.0f;
    /* V1_11=P_01^+/beta */
    V1[1][1][0]=P[0][1][0]/beta;
    V1[1][1][1]=-P[0][1][1]/beta;
    /* V1_21=P_02^+/beta */
    V1[2][1][0]=P[0][2][0]/beta;
    V1[2][1][1]=-P[0][2][1]/beta;
    /* V1_12=-tau^+*v_2^+=-(tau*v_2)^+ */
    V1[1][2][0]=-(taure*v[2][0]-tauim*v[2][1]);
    V1[1][2][1]=taure*v[2][1]+tauim*v[2][0];
    /* V1_22=1-tau^+*v_2*v_2^+ */
    a=v[2][0]*v[2][0]+v[2][1]*v[2][1];
    V1[2][2][0]=1-taure*a;
    V1[2][2][1]=tauim*a;


    /* apply the transformation to P matrix and store the result in Q
       Q=PV */
    Q[0][0][0]=P[0][0][0];
    Q[0][0][1]=0.0f;
    Q[1][0][0]=0.0f;
    Q[1][0][1]=0.0f;
    Q[2][0][0]=0.0f;
    Q[2][0][1]=0.0f;
    Q[0][1][0]=beta;
    Q[0][1][1]=0.0f;
    Q[0][2][0]=0.0f;
    Q[0][2][1]=0.0f;
    /* Q_11=P_11*V1_11+P_12*V_21 */
    Q[1][1][0]=P[1][1][0]*V1[1][1][0]-P[1][1][1]*V1[1][1][1]
              +P[1][2][0]*V1[2][1][0]-P[1][2][1]*V1[2][1][1];
    Q[1][1][1]=P[1][1][0]*V1[1][1][1]+P[1][1][1]*V1[1][1][0]
              +P[1][2][0]*V1[2][1][1]+P[1][2][1]*V1[2][1][0];
    /* Q_12=P_11*V1_12+P_12*V_22 */
    Q[1][2][0]=P[1][1][0]*V1[1][2][0]-P[1][1][1]*V1[1][2][1]
              +P[1][2][0]*V1[2][2][0]-P[1][2][1]*V1[2][2][1];
    Q[1][2][1]=P[1][1][0]*V1[1][2][1]+P[1][1][1]*V1[1][2][0]
              +P[1][2][0]*V1[2][2][1]+P[1][2][1]*V1[2][2][0];
    /* Q_21=P_21*V1_11+P_22*V_21 */
    Q[2][1][0]=P[2][1][0]*V1[1][1][0]-P[2][1][1]*V1[1][1][1]
              +P[2][2][0]*V1[2][1][0]-P[2][2][1]*V1[2][1][1];
    Q[2][1][1]=P[2][1][0]*V1[1][1][1]+P[2][1][1]*V1[1][1][0]
              +P[2][2][0]*V1[2][1][1]+P[2][2][1]*V1[2][1][0];
    /* Q_22=P_21*V1_12+P_22*V_22 */
    Q[2][2][0]=P[2][1][0]*V1[1][2][0]-P[2][1][1]*V1[1][2][1]
              +P[2][2][0]*V1[2][2][0]-P[2][2][1]*V1[2][2][1];
    Q[2][2][1]=P[2][1][0]*V1[1][2][1]+P[2][1][1]*V1[1][2][0]
              +P[2][2][0]*V1[2][2][1]+P[2][2][1]*V1[2][2][0];

  }
#ifdef SVD3x3_DEBUG
printf("Right unitary matrix V1:\n");
    for(i=0;i<3;i++)for(j=0;j<3;j++) {
      printf( "V1[%d][%d].re=%26.18e  V1[%d][%d].im=%26.18e\n",
              i, j, V1[i][j][0], i, j, V1[i][j][1] );
    }
#endif /* SVD3x3_DEBUG */



  /* *** Step 3: build second left reflector v,
                 calculate second left rotation U2,
                 apply to the matrix Q *** */
  /* calculate norm of ( Q[21] )
     with minimal loss of accuracy (similar to BLAS) */
  c=fabs(Q[2][1][0]); d=fabs(Q[2][1][1]);
  if( c>d ) {
    max=c; min=d;
  }
  else {
    max=d; min=c;
  }
  if( min==0 ) {
    norm = max;
  }
  else {
    c = min/max;
    norm = max*sqrt(1+c*c);
  }

  if( norm==0 && Q[1][1][1]==0 ) { /* no rotation needed */
#ifdef SVD3x3_DEBUG
    printf("Step 3: no rotation needed\n");
#endif /* SVD3x3_DEBUG */
    U2[0][0][0]=1.0f; U2[0][0][1]=0.0f;
    U2[0][1][0]=0.0f; U2[0][1][1]=0.0f;
    U2[0][2][0]=0.0f; U2[0][2][1]=0.0f;
    U2[1][0][0]=0.0f; U2[1][0][1]=0.0f;
    U2[1][1][0]=1.0f; U2[1][1][1]=0.0f;
    U2[1][2][0]=0.0f; U2[1][2][1]=0.0f;
    U2[2][0][0]=0.0f; U2[2][0][1]=0.0f;
    U2[2][1][0]=0.0f; U2[2][1][1]=0.0f;
    U2[2][2][0]=1.0f; U2[2][2][1]=0.0f;
    P[0][0][0]=Q[0][0][0]; P[0][0][1]=Q[0][0][1];
    P[1][0][0]=Q[1][0][0]; P[1][0][1]=Q[1][0][1];
    P[2][0][0]=Q[2][0][0]; P[2][0][1]=Q[2][0][1];
    P[0][1][0]=Q[0][1][0]; P[0][1][1]=Q[0][1][1];
    P[1][1][0]=Q[1][1][0]; P[1][1][1]=Q[1][1][1];
    P[2][1][0]=Q[2][1][0]; P[2][1][1]=Q[2][1][1];
    P[0][2][0]=Q[0][2][0]; P[0][2][1]=Q[0][2][1];
    P[1][2][0]=Q[1][2][0]; P[1][2][1]=Q[1][2][1];
    P[2][2][0]=Q[2][2][0]; P[2][2][1]=Q[2][2][1];
  }
  else {
    /* get the norm of (Q_11 Q_21) column vector */
    c=1.0f;
    factor = norm;
    a = fabs( Q[1][1][0] );
    if( a!=0 ) {
      if( factor < a ) {
        c = 1.0f + (factor/a)*(factor/a);
        factor = a;
      }
      else {
        c += (a/factor)*(a/factor);
      }
    }
    a = fabs( Q[1][1][1] );
    if( a!=0 ) {
      if( factor < a ) {
        c = 1.0f + c*(factor/a)*(factor/a);
        factor = a;
      }
      else {
        c += (a/factor)*(a/factor);
      }
    }
    beta = factor*sqrt(c); /* norm of (Q_11 Q_21) column vector */
    if( Q[1][1][0]>0 ) {
      beta = -beta;
    }

#ifdef SVD3x3_DEBUG
    printf("beta=%28.18e\n",beta);
#endif /* SVD3x3_DEBUG */


    /* a=Re(Q_11-beta), b=Im(Q_11-beta) */
    a=Q[1][1][0]-beta; b=Q[1][1][1];
    /* norm=sqrt(a^2+b^2) */
    c=fabs(a); d=fabs(b);
    if( c>d ) {
      max=c; min=d;
    }
    else {
      max=d; min=c;
    }
    if( min==0 ) {
      norm = max;
    }
    else {
      c = min/max;
      norm = max*sqrt(1+c*c);
    }
    /* c=a/norm, d=b/norm */
    c=a/norm; d=b/norm;

    /* construct reflector (vector "v" for Householder transformation) */
    /* v_0=0 */
    v[0][0]=0.0f; v[0][1]=0.0f;
    /* v_1=1 */
    v[1][0]=1.0f; v[1][1]=0.0f;
    /* v_2=Q_21/(Q_11-beta)=Q_21/(a+ib)=(Q_21*(a-ib))/norm^2=(Q_21/norm)*((a-ib)/norm)
          =(Q_21/norm)*(c-id)=|a=Re(Q_21)/norm,b=Im(Q_21)/norm|=(a+ib)*(c-id)
          =(a*c+b*d)+i(b*c-a*d) */
    a=Q[2][1][0]/norm; b=Q[2][1][1]/norm;
    v[2][0]=a*c+b*d;
    v[2][1]=b*c-a*d;
#ifdef SVD3x3_DEBUG
for(i=0;i<3;i++) {
  printf("v[%d].re=%28.18e  v[%d].im=%28.18e\n",i,v[i][0],i,v[i][1]);
}
#endif /* SVD3x3_DEBUG */


    /* calcualate tau (coefficient for reflector) */
    taure=(beta-Q[1][1][0])/beta;
    tauim=Q[1][1][1]/beta;


    /* assemble right unitary matrix U2=I-tau^+*v*v^+ (store in U2[3][3][2]) */
    U2[0][0][0]=1.0f;
    U2[0][0][1]=0.0f;
    U2[1][0][0]=0.0f;
    U2[1][0][1]=0.0f;
    U2[2][0][0]=0.0f;
    U2[2][0][1]=0.0f;
    U2[0][1][0]=0.0f;
    U2[0][1][1]=0.0f;
    U2[0][2][0]=0.0f;
    U2[0][2][1]=0.0f;
    /* U2_11=Q_11/beta */
    U2[1][1][0]=Q[1][1][0]/beta;
    U2[1][1][1]=Q[1][1][1]/beta;
    /* U2_21=Q_21/beta */
    U2[2][1][0]=Q[2][1][0]/beta;
    U2[2][1][1]=Q[2][1][1]/beta;
    /* U2_12=-tau^+*v_2^+=-(tau*v_2)^+ */
    U2[1][2][0]=-(taure*v[2][0]-tauim*v[2][1]);
    U2[1][2][1]=taure*v[2][1]+tauim*v[2][0];
    /* U2_22=1-tau^+*v_2*v_2^+ */
    a=v[2][0]*v[2][0]+v[2][1]*v[2][1];
    U2[2][2][0]=1-taure*a;
    U2[2][2][1]=tauim*a;
#ifdef SVD3x3_DEBUG
printf("Left unitary matrix U2:\n");
    for(i=0;i<3;i++)for(j=0;j<3;j++) {
      printf( "U2[%d][%d].re=%26.18e  U2[%d][%d].im=%26.18e\n",
              i, j, U2[i][j][0], i, j, U2[i][j][1] );
    }
#endif /* SVD3x3_DEBUG */


    /* apply the transformation to Q matrix and store the result in P
       P=U^+Q */
    P[0][0][0]=Q[0][0][0];
    P[0][0][1]=0.0f;
    P[1][0][0]=0.0f;
    P[1][0][1]=0.0f;
    P[2][0][0]=0.0f;
    P[2][0][1]=0.0f;
    P[0][1][0]=Q[0][1][0];
    P[0][1][1]=0.0f;
    P[0][2][0]=0.0f;
    P[0][2][1]=0.0f;
    P[1][1][0]=beta;
    P[1][1][1]=0.0f;
    P[2][1][0]=0.0f;
    P[2][1][1]=0.0f;
    /* P_12=U2_11^+*Q_12+U2_21^+*Q_22 */
    P[1][2][0]=U2[1][1][0]*Q[1][2][0]+U2[1][1][1]*Q[1][2][1]
              +U2[2][1][0]*Q[2][2][0]+U2[2][1][1]*Q[2][2][1];
    P[1][2][1]=U2[1][1][0]*Q[1][2][1]-U2[1][1][1]*Q[1][2][0]
              +U2[2][1][0]*Q[2][2][1]-U2[2][1][1]*Q[2][2][0];
    /* P_22=U2_12^+*Q_12+U2_22^+*Q_22 */
    P[2][2][0]=U2[1][2][0]*Q[1][2][0]+U2[1][2][1]*Q[1][2][1]
              +U2[2][2][0]*Q[2][2][0]+U2[2][2][1]*Q[2][2][1];
    P[2][2][1]=U2[1][2][0]*Q[1][2][1]-U2[1][2][1]*Q[1][2][0]
              +U2[2][2][0]*Q[2][2][1]-U2[2][2][1]*Q[2][2][0];

  }



  /* *** Step 4: build second right reflector v,
                 calculate second right rotation V2,
                 apply to the matrix P *** */
  if( P[1][2][1]==0 ) { /* no rotation needed */
#ifdef SVD3x3_DEBUG
    printf("Step 4: no rotation needed\n");
#endif /* SVD3x3_DEBUG */
    V2[0][0][0]=1.0f; V2[0][0][1]=0.0f;
    V2[0][1][0]=0.0f; V2[0][1][1]=0.0f;
    V2[0][2][0]=0.0f; V2[0][2][1]=0.0f;
    V2[1][0][0]=0.0f; V2[1][0][1]=0.0f;
    V2[1][1][0]=1.0f; V2[1][1][1]=0.0f;
    V2[1][2][0]=0.0f; V2[1][2][1]=0.0f;
    V2[2][0][0]=0.0f; V2[2][0][1]=0.0f;
    V2[2][1][0]=0.0f; V2[2][1][1]=0.0f;
    V2[2][2][0]=1.0f; V2[2][2][1]=0.0f;
    Q[0][0][0]=P[0][0][0]; Q[0][0][1]=P[0][0][1];
    Q[1][0][0]=P[1][0][0]; Q[1][0][1]=P[1][0][1];
    Q[2][0][0]=P[2][0][0]; Q[2][0][1]=P[2][0][1];
    Q[0][1][0]=P[0][1][0]; Q[0][1][1]=P[0][1][1];
    Q[1][1][0]=P[1][1][0]; Q[1][1][1]=P[1][1][1];
    Q[2][1][0]=P[2][1][0]; Q[2][1][1]=P[2][1][1];
    Q[0][2][0]=P[0][2][0]; Q[0][2][1]=P[0][2][1];
    Q[1][2][0]=P[1][2][0]; Q[1][2][1]=P[1][2][1];
    Q[2][2][0]=P[2][2][0]; Q[2][2][1]=P[2][2][1];
  }
  else {
    /* calculate norm of ( P[12] ) */
    c=fabs(P[1][2][0]); d=fabs(P[1][2][1]);
    if( c>d ) {
      max=c; min=d;
    }
    else {
      max=d; min=c;
    }
    if( min==0 ) {
      beta = max;
    }
    else {
      c = min/max;
      beta = max*sqrt(1+c*c);
    }

    if( P[1][2][0]>0 ) {
      beta = -beta;
    }

#ifdef SVD3x3_DEBUG
    printf("beta=%28.18e\n",beta);
#endif /* SVD3x3_DEBUG */

    /* assemble right unitary matrix V1=I-tau^+*v*v^+ (store in V1[3][3][2]) */
    V2[0][0][0]=1.0f;
    V2[0][0][1]=0.0f;
    V2[1][0][0]=0.0f;
    V2[1][0][1]=0.0f;
    V2[2][0][0]=0.0f;
    V2[2][0][1]=0.0f;
    V2[0][1][0]=0.0f;
    V2[0][1][1]=0.0f;
    V2[0][2][0]=0.0f;
    V2[0][2][1]=0.0f;
    V2[1][1][0]=1.0f;
    V2[1][1][1]=0.0f;
    V2[2][1][0]=0.0f;
    V2[2][1][1]=0.0f;
    V2[1][2][0]=0.0f;
    V2[1][2][1]=0.0f;
    /* V2_22=1-tau^+*v_2*v_2^+=1-tau^+ */
    V2[2][2][0]=P[1][2][0]/beta;
    V2[2][2][1]=-P[1][2][1]/beta;
#ifdef SVD3x3_DEBUG
printf("Right unitary matrix V2:\n");
    for(i=0;i<3;i++)for(j=0;j<3;j++) {
      printf( "V2[%d][%d].re=%26.18e  V2[%d][%d].im=%26.18e\n",
              i, j, V2[i][j][0], i, j, V2[i][j][1] );
    }
#endif /* SVD3x3_DEBUG */


    /* apply the transformation to P matrix and store the result in Q
       Q=PV */
    Q[0][0][0]=P[0][0][0];
    Q[0][0][1]=0.0f;
    Q[1][0][0]=0.0f;
    Q[1][0][1]=0.0f;
    Q[2][0][0]=0.0f;
    Q[2][0][1]=0.0f;
    Q[0][1][0]=P[0][1][0];
    Q[0][1][1]=0.0f;
    Q[0][2][0]=0.0f;
    Q[0][2][1]=0.0f;
    Q[1][1][0]=P[1][1][0];
    Q[1][1][1]=0.0f;
    Q[1][2][0]=beta;
    Q[1][2][1]=0.0f;
    Q[2][1][0]=0.0f;
    Q[2][1][1]=0.0f;
    /* Q_22=P_22*V2_22 */
    Q[2][2][0]=P[2][2][0]*V2[2][2][0]-P[2][2][1]*V2[2][2][1];
    Q[2][2][1]=P[2][2][0]*V2[2][2][1]+P[2][2][1]*V2[2][2][0];


  }



  /* *** Step 5: build third left reflector v,
                 calculate third left rotation U3,
                 apply to the matrix P *** */
  if( Q[2][2][1]==0 ) { /* no rotation needed */
#ifdef SVD3x3_DEBUG
    printf("Step 5: no rotation needed\n");
#endif /* SVD3x3_DEBUG */
    U3[0][0][0]=1.0f; U3[0][0][1]=0.0f;
    U3[0][1][0]=0.0f; U3[0][1][1]=0.0f;
    U3[0][2][0]=0.0f; U3[0][2][1]=0.0f;
    U3[1][0][0]=0.0f; U3[1][0][1]=0.0f;
    U3[1][1][0]=1.0f; U3[1][1][1]=0.0f;
    U3[1][2][0]=0.0f; U3[1][2][1]=0.0f;
    U3[2][0][0]=0.0f; U3[2][0][1]=0.0f;
    U3[2][1][0]=0.0f; U3[2][1][1]=0.0f;
    U3[2][2][0]=1.0f; U3[2][2][1]=0.0f;
    P[0][0][0]=Q[0][0][0]; P[0][0][1]=Q[0][0][1];
    P[1][0][0]=Q[1][0][0]; P[1][0][1]=Q[1][0][1];
    P[2][0][0]=Q[2][0][0]; P[2][0][1]=Q[2][0][1];
    P[0][1][0]=Q[0][1][0]; P[0][1][1]=Q[0][1][1];
    P[1][1][0]=Q[1][1][0]; P[1][1][1]=Q[1][1][1];
    P[2][1][0]=Q[2][1][0]; P[2][1][1]=Q[2][1][1];
    P[0][2][0]=Q[0][2][0]; P[0][2][1]=Q[0][2][1];
    P[1][2][0]=Q[1][2][0]; P[1][2][1]=Q[1][2][1];
    P[2][2][0]=Q[2][2][0]; P[2][2][1]=Q[2][2][1];
  }
  else {
    /* calculate norm of ( Q[22] ) */
    c=fabs(Q[2][2][0]); d=fabs(Q[2][2][1]);
    if( c>d ) {
      max=c; min=d;
    }
    else {
      max=d; min=c;
    }
    if( min==0 ) {
      beta = max;
    }
    else {
      c = min/max;
      beta = max*sqrt(1+c*c);
    }

    if( Q[2][2][0]>0 ) {
      beta = -beta;
    }

#ifdef SVD3x3_DEBUG
    printf("beta=%28.18e\n",beta);
#endif /* SVD3x3_DEBUG */

    /* assemble left unitary matrix U3=I-tau^+*v*v^+ (store in U3[3][3][2]) */
    U3[0][0][0]=1.0f;
    U3[0][0][1]=0.0f;
    U3[1][0][0]=0.0f;
    U3[1][0][1]=0.0f;
    U3[2][0][0]=0.0f;
    U3[2][0][1]=0.0f;
    U3[0][1][0]=0.0f;
    U3[0][1][1]=0.0f;
    U3[0][2][0]=0.0f;
    U3[0][2][1]=0.0f;
    U3[1][1][0]=1.0f;
    U3[1][1][1]=0.0f;
    U3[2][1][0]=0.0f;
    U3[2][1][1]=0.0f;
    U3[1][2][0]=0.0f;
    U3[1][2][1]=0.0f;
    /* U3_22=1-tau^+*v_2*v_2^+=1-tau^+ */
    U3[2][2][0]=Q[2][2][0]/beta;
    U3[2][2][1]=Q[2][2][1]/beta;
#ifdef SVD3x3_DEBUG
printf("Left unitary matrix U3:\n");
    for(i=0;i<3;i++)for(j=0;j<3;j++) {
      printf( "U3[%d][%d].re=%26.18e  U3[%d][%d].im=%26.18e\n",
              i, j, U3[i][j][0], i, j, U3[i][j][1] );
    }
#endif /* SVD3x3_DEBUG */


    /* apply the transformation to Q matrix and store the result in P
       P=U^+Q */
    P[0][0][0]=Q[0][0][0];
    P[0][0][1]=0.0f;
    P[1][0][0]=0.0f;
    P[1][0][1]=0.0f;
    P[2][0][0]=0.0f;
    P[2][0][1]=0.0f;
    P[0][1][0]=Q[0][1][0];
    P[0][1][1]=0.0f;
    P[0][2][0]=0.0f;
    P[0][2][1]=0.0f;
    P[1][1][0]=Q[1][1][0];
    P[1][1][1]=0.0f;
    P[1][2][0]=Q[1][2][0];
    P[1][2][1]=0.0f;
    P[2][1][0]=0.0f;
    P[2][1][1]=0.0f;
    P[2][2][0]=beta;
    P[2][2][1]=0.0f;

  }




  /* *** This part starts with a bidiagonal matrix and uses
         QR algorithm with shifts to eliminate the superdiagonal *** */
  /* prepare left and right real orthogonal matrices that
     accumulate Givens rotations from QR algorithm */
  UO3[0][0]=1.0f; UO3[0][1]=0.0f; UO3[0][2]=0.0f;
  UO3[1][0]=0.0f; UO3[1][1]=1.0f; UO3[1][2]=0.0f;
  UO3[2][0]=0.0f; UO3[2][1]=0.0f; UO3[2][2]=1.0f;
  VO3[0][0]=1.0f; VO3[0][1]=0.0f; VO3[0][2]=0.0f;
  VO3[1][0]=0.0f; VO3[1][1]=1.0f; VO3[1][2]=0.0f;
  VO3[2][0]=0.0f; VO3[2][1]=0.0f; VO3[2][2]=1.0f;

  iter=0;

#ifdef SVD3x3_DEBUG
printf( "QR iteration: %d\n", iter );
printf( "%+20.16e %+20.16e %+20.16e\n", b00, b01, b02 );
printf( "%+20.16e %+20.16e %+20.16e\n", b10, b11, b12 );
printf( "%+20.16e %+20.16e %+20.16e\n", b20, b21, b22 );
#endif /* SVD3x3_DEBUG */

  do {

    iter++;
    if(iter>300) break;

    /* chop small superdiagonal elements */
    if( fabs(b01) < SVD3x3_PREC*(fabs(b00)+fabs(b11)) ) {
      b01=0;
    }
    if( fabs(b12) < SVD3x3_PREC*(fabs(b00)+fabs(b22)) ) {
      b12=0;
    }

    /* Cases:
       b01=b12=0 -- matrix is already diagonalized,
       b01=0 -- need to work with 2x2 lower block,
       b12=0 -- need to work with 2x2 upper block,
       else -- normal iteration */
    if( !(b01==0 && b12==0) ) {
      if( b01==0 ) {
#ifdef SVD3x3_DEBUG
printf( "Entering case b01==0\n" );
#endif /* SVD3x3_DEBUG */
        /* need to diagonalize 2x2 lower block */
        svd2x2bidiag( &b11, &b12, &b22, UO2, VO2 );

        /* multiply left UO3 matrix */
        for(i=0;i<3;i++) {
          a=UO3[i][1]; b=UO3[i][2];
          UO3[i][1]=a*UO2[0][0]+b*UO2[1][0];
          UO3[i][2]=a*UO2[0][1]+b*UO2[1][1];
        }
        /* multiply right VO3 matrix */
        for(i=0;i<3;i++) {
          a=VO3[i][1]; b=VO3[i][2];
          VO3[i][1]=a*VO2[0][0]+b*VO2[1][0];
          VO3[i][2]=a*VO2[0][1]+b*VO2[1][1];
        }

      }
      else {
        if( b12==0 ) {
#ifdef SVD3x3_DEBUG
printf( "Entering case b12==0\n" );
#endif /* SVD3x3_DEBUG */
          /* need to diagonalize 2x2 upper block */
          svd2x2bidiag( &b00, &b01, &b11, UO2, VO2 );

          /* multiply left UO3 matrix */
          for(i=0;i<3;i++) {
            a=UO3[i][0]; b=UO3[i][1];
            UO3[i][0]=a*UO2[0][0]+b*UO2[1][0];
            UO3[i][1]=a*UO2[0][1]+b*UO2[1][1];
          }
          /* multiply right VO3 matrix */
          for(i=0;i<3;i++) {
            a=VO3[i][0]; b=VO3[i][1];
            VO3[i][0]=a*VO2[0][0]+b*VO2[1][0];
            VO3[i][1]=a*VO2[0][1]+b*VO2[1][1];
          }

        }
        else {
          /* normal 3x3 iteration */

          /* QR shift does not work if there are zeros
             on the diagonal, therefore first check
             for special cases: b00==0 or b11==0 or b22==0 */

          if( b00==0 ) {
#ifdef SVD3x3_DEBUG
printf( "Entering case b00==0\n" );
#endif /* SVD3x3_DEBUG */
            /* b01 can be rotated away to create b02,
               and then b02 can be rotated away
               (both are left rotations) */
            if( fabs(b01)>fabs(b11) ) {
              cotphi=b11/b01;
              sinphi=1/sqrt(1+cotphi*cotphi);
              cosphi=cotphi*sinphi;
            }
            else {
              tanphi=b01/b11;
              cosphi=1/sqrt(1+tanphi*tanphi);
              sinphi=tanphi*cosphi;
            }
            /* multiply left UO3 matrix */
            for(i=0;i<3;i++) {
              a=UO3[i][0]; b=UO3[i][1];
              UO3[i][0]=a*cosphi-b*sinphi;
              UO3[i][1]=a*sinphi+b*cosphi;
            }
            /* apply to bidiagonal matrix, this generates b02 */
            b11=b01*sinphi+b11*cosphi;
            b02=-b12*sinphi;
            b12=b12*cosphi;
            b01=0.0f;
            if( fabs(b02)>fabs(b22) ) {
              cotphi=b22/b02;
              sinphi=1/sqrt(1+cotphi*cotphi);
              cosphi=cotphi*sinphi;
            }
            else {
              tanphi=b02/b22;
              cosphi=1/sqrt(1+tanphi*tanphi);
              sinphi=tanphi*cosphi;
            }
            /* multiply left UO3 matrix */
            for(i=0;i<3;i++) {
              a=UO3[i][0]; b=UO3[i][2];
              UO3[i][0]=a*cosphi-b*sinphi;
              UO3[i][2]=a*sinphi+b*cosphi;
            }
            /* apply to bidiagonal matrix */
            b22=b02*sinphi+b22*cosphi;
            b02=0.0f;
          }
          else if( b11==0 ) {
#ifdef SVD3x3_DEBUG
printf( "Entering case b11==0\n" );
#endif /* SVD3x3_DEBUG */
            /* b12 is rotated away with left rotation */
            if( fabs(b12)>fabs(b22) ) {
              cotphi=b22/b12;
              sinphi=1/sqrt(1+cotphi*cotphi);
              cosphi=cotphi*sinphi;
            }
            else {
              tanphi=b12/b22;
              cosphi=1/sqrt(1+tanphi*tanphi);
              sinphi=tanphi*cosphi;
            }
            /* multiply left UO3 matrix */
            for(i=0;i<3;i++) {
              a=UO3[i][1]; b=UO3[i][2];
              UO3[i][1]=a*cosphi-b*sinphi;
              UO3[i][2]=a*sinphi+b*cosphi;
            }
            /* apply to bidiagonal matrix */
            b22=b12*sinphi+b22*cosphi;
            b12=0.0f;
          }
          else if( b22==0 ) {
#ifdef SVD3x3_DEBUG
printf( "Entering case b22==0\n" );
#endif /* SVD3x3_DEBUG */
            /* b12 is rotated away and b02 appears,
               then b02 is rotated away, both are
               right rotations */
            if( fabs(b12)>fabs(b11) ) {
              cotphi=b11/b12;
              sinphi=1/sqrt(1+cotphi*cotphi);
              cosphi=cotphi*sinphi;
            }
            else {
              tanphi=b12/b11;
              cosphi=1/sqrt(1+tanphi*tanphi);
              sinphi=tanphi*cosphi;
            }
            /* multiply right VO3 matrix */
            for(i=0;i<3;i++) {
              a=VO3[i][1]; b=VO3[i][2];
              VO3[i][1]= a*cosphi+b*sinphi;
              VO3[i][2]=-a*sinphi+b*cosphi;
            }
            /* apply to bidiagonal matrix */
            b02=-b01*sinphi;
            b01=b01*cosphi;
            b11=b11*cosphi+b12*sinphi;
            b12=0.0f;
            /* second rotation removes b02 */
            if( fabs(b02)>fabs(b00) ) {
              cotphi=b00/b02;
              sinphi=1/sqrt(1+cotphi*cotphi);
              cosphi=cotphi*sinphi;
            }
            else {
              tanphi=b02/b00;
              cosphi=1/sqrt(1+tanphi*tanphi);
              sinphi=tanphi*cosphi;
            }
            /* multiply right VO3 matrix */
            for(i=0;i<3;i++) {
              a=VO3[i][0]; b=VO3[i][2];
              VO3[i][0]= a*cosphi+b*sinphi;
              VO3[i][2]=-a*sinphi+b*cosphi;
            }
            /* apply to bidiagonal matrix */
            b00=b00*cosphi+b02*sinphi;
            b02=0.0f;
          }
          else {
            /* full iteration with QR shift */
#ifdef SVD3x3_DEBUG
printf( "Entering case of normal QR iteration\n" );
#endif /* SVD3x3_DEBUG */

            /* find max eigenvalue of bottom 2x2 minor */
            m11=b11*b11+b01*b01;
            m22=b22*b22+b12*b12;
            m12=b11*b12;
            dm=(m11-m22)/2;

            /* safely calculate sqrt */
            c=fabs(dm); d=fabs(m12);
            if( c>d ) {
              max=c; min=d;
            }
            else {
              max=d; min=c;
            }
            if( min==0 ) {
              norm = max;
            }
            else {
              c = min/max;
              norm = max*sqrt(1+c*c);
            }

            if( dm>=0 ) {
              lambdamax=m22-(m12*m12)/(dm+norm);
            }
            else {
              lambdamax=m22+(m12*m12)/(norm-dm);
            }

            /* calculate first Givens rotation (on the right) */
            a=b00*b00-lambdamax;
            b=b00*b01;
            if( 0==b ) {
              cosphi=1.0f;
              sinphi=0.0f;
            }
            else {
              if( fabs(b)>fabs(a) ) {
                cotphi=-a/b;
                sinphi=1./sqrt(1+cotphi*cotphi);
                cosphi=cotphi*sinphi;
              }
              else {
                tanphi=-b/a;
                cosphi=1./sqrt(1+tanphi*tanphi);
                sinphi=tanphi*cosphi;
              }
            }
            /* multiply right VO3 matrix */
            for(i=0;i<3;i++) {
              a=VO3[i][0]; b=VO3[i][1];
              VO3[i][0]=a*cosphi-b*sinphi;
              VO3[i][1]=a*sinphi+b*cosphi;
            }
            /* apply to bidiagonal matrix, this generate b10 */
            a=b00; b=b01;
            b00=a*cosphi-b*sinphi;
            b01=a*sinphi+b*cosphi;
            b10=-b11*sinphi;
            b11=b11*cosphi;

            /* calculate second Givens rotation (on the left) */
            if(0==b10) {
              cosphi=1.0f;
              sinphi=0.0f;
            }
            else {
              if( fabs(b10)>fabs(b00) ) {
                cotphi=-b00/b10;
                sinphi=1/sqrt(1+cotphi*cotphi);
                cosphi=cotphi*sinphi;
              }
              else {
                tanphi=-b10/b00;
                cosphi=1/sqrt(1+tanphi*tanphi);
                sinphi=tanphi*cosphi;
              }
            }
            /* multiply left UO3 matrix */
            for(i=0;i<3;i++) {
              a=UO3[i][0]; b=UO3[i][1];
              UO3[i][0]= a*cosphi-b*sinphi;
              UO3[i][1]= a*sinphi+b*cosphi;
            }
            /* apply to bidiagonal matrix, this generates b02 */
            b00=b00*cosphi-b10*sinphi;
            a=b01; b=b11;
            b01=a*cosphi-b*sinphi;
            b11=a*sinphi+b*cosphi;
            b02=-b12*sinphi;
            b12=b12*cosphi;
            b10=0.0f;

            /* calculate third Givens rotation (on the right) */
            if(0==b02) {
              cosphi=1.0f;
              sinphi=0.0f;
            }
            else {
              if( fabs(b02)>fabs(b01) ) {
                cotphi=-b01/b02;
                sinphi=1/sqrt(1+cotphi*cotphi);
                cosphi=cotphi*sinphi;
              }
              else {
                tanphi=-b02/b01;
                cosphi=1/sqrt(1+tanphi*tanphi);
                sinphi=tanphi*cosphi;
              }
            }
            /* multiply right VO3 matrix */
            for(i=0;i<3;i++) {
              a=VO3[i][1]; b=VO3[i][2];
              VO3[i][1]=a*cosphi-b*sinphi;
              VO3[i][2]=a*sinphi+b*cosphi;
            }
            /* apply to bidiagonal matrix, this generates b21 */
            b01=b01*cosphi-b02*sinphi;
            a=b11; b=b12;
            b11=a*cosphi-b*sinphi;
            b12=a*sinphi+b*cosphi;
            b21=-b22*sinphi;
            b22=b22*cosphi;
            b02=0.0f;

            /* calculate fourth Givens rotation (on the left) */
            if(0==b21) {
              cosphi=1.0f;
              sinphi=0.0f;
            }
            else {
              if( fabs(b21)>fabs(b11) ) {
                cotphi=-b11/b21;
                sinphi=1/sqrt(1+cotphi*cotphi);
                cosphi=cotphi*sinphi;
              }
              else {
                tanphi=-b21/b11;
                cosphi=1/sqrt(1+tanphi*tanphi);
                sinphi=tanphi*cosphi;
              }
            }
            /* multiply left UO3 matrix */
            for(i=0;i<3;i++) {
              a=UO3[i][1]; b=UO3[i][2];
              UO3[i][1]= a*cosphi-b*sinphi;
              UO3[i][2]= a*sinphi+b*cosphi;
            }
            /* apply to bidiagonal matrix, this eliminates b21 */
            b11=b11*cosphi-b21*sinphi;
            a=b12; b=b22;
            b12=a*cosphi-b*sinphi;
            b22=a*sinphi+b*cosphi;
            b21=0.0f;
          }
        } /* end of normal 3x3 iteration */
      }
    }


#ifdef SVD3x3_DEBUG
printf( "QR iteration: %d\n", iter );
printf( "%+20.16e %+20.16e %+20.16e\n", b00, b01, b02 );
printf( "%+20.16e %+20.16e %+20.16e\n", b10, b11, b12 );
printf( "%+20.16e %+20.16e %+20.16e\n", b20, b21, b22 );
#endif /* SVD3x3_DEBUG */
  }



  while( b01!=0 || b12!=0 );


  /* make singular values positive */
  if(b00<0) {
    b00=-b00;
    VO3[0][0]=-VO3[0][0];
    VO3[1][0]=-VO3[1][0];
    VO3[2][0]=-VO3[2][0];
  }
  if(b11<0) {
    b11=-b11;
    VO3[0][1]=-VO3[0][1];
    VO3[1][1]=-VO3[1][1];
    VO3[2][1]=-VO3[2][1];
  }
  if(b22<0) {
    b22=-b22;
    VO3[0][2]=-VO3[0][2];
    VO3[1][2]=-VO3[1][2];
    VO3[2][2]=-VO3[2][2];
  }



  /* Q=U1*U2 (U2 is block diagonal with U2_00=1) */
  Q[0][0][0]=U1[0][0][0]; Q[0][0][1]=U1[0][0][1];
  Q[1][0][0]=U1[1][0][0]; Q[1][0][1]=U1[1][0][1];
  Q[2][0][0]=U1[2][0][0]; Q[2][0][1]=U1[2][0][1];
  /* Q_01=U1_01*U2_11+U1_02*U2_21 */
  Q[0][1][0]=U1[0][1][0]*U2[1][1][0]-U1[0][1][1]*U2[1][1][1]
            +U1[0][2][0]*U2[2][1][0]-U1[0][2][1]*U2[2][1][1];
  Q[0][1][1]=U1[0][1][0]*U2[1][1][1]+U1[0][1][1]*U2[1][1][0]
            +U1[0][2][0]*U2[2][1][1]+U1[0][2][1]*U2[2][1][0];
  /* Q_02=U1_01*U2_12+U1_02*U2_22 */
  Q[0][2][0]=U1[0][1][0]*U2[1][2][0]-U1[0][1][1]*U2[1][2][1]
            +U1[0][2][0]*U2[2][2][0]-U1[0][2][1]*U2[2][2][1];
  Q[0][2][1]=U1[0][1][0]*U2[1][2][1]+U1[0][1][1]*U2[1][2][0]
            +U1[0][2][0]*U2[2][2][1]+U1[0][2][1]*U2[2][2][0];
  /* Q_11=U1_11*U2_11+U1_12*U2_21 */
  Q[1][1][0]=U1[1][1][0]*U2[1][1][0]-U1[1][1][1]*U2[1][1][1]
            +U1[1][2][0]*U2[2][1][0]-U1[1][2][1]*U2[2][1][1];
  Q[1][1][1]=U1[1][1][0]*U2[1][1][1]+U1[1][1][1]*U2[1][1][0]
            +U1[1][2][0]*U2[2][1][1]+U1[1][2][1]*U2[2][1][0];
  /* Q_12=U1_11*U2_12+U1_12*U2_22 */
  Q[1][2][0]=U1[1][1][0]*U2[1][2][0]-U1[1][1][1]*U2[1][2][1]
            +U1[1][2][0]*U2[2][2][0]-U1[1][2][1]*U2[2][2][1];
  Q[1][2][1]=U1[1][1][0]*U2[1][2][1]+U1[1][1][1]*U2[1][2][0]
            +U1[1][2][0]*U2[2][2][1]+U1[1][2][1]*U2[2][2][0];
  /* Q_21=U1_21*U2_11+U1_22*U2_21 */
  Q[2][1][0]=U1[2][1][0]*U2[1][1][0]-U1[2][1][1]*U2[1][1][1]
            +U1[2][2][0]*U2[2][1][0]-U1[2][2][1]*U2[2][1][1];
  Q[2][1][1]=U1[2][1][0]*U2[1][1][1]+U1[2][1][1]*U2[1][1][0]
            +U1[2][2][0]*U2[2][1][1]+U1[2][2][1]*U2[2][1][0];
  /* Q_22=U1_21*U2_12+U1_22*U2_22 */
  Q[2][2][0]=U1[2][1][0]*U2[1][2][0]-U1[2][1][1]*U2[1][2][1]
            +U1[2][2][0]*U2[2][2][0]-U1[2][2][1]*U2[2][2][1];
  Q[2][2][1]=U1[2][1][0]*U2[1][2][1]+U1[2][1][1]*U2[1][2][0]
            +U1[2][2][0]*U2[2][2][1]+U1[2][2][1]*U2[2][2][0];

  /* Q=Q*U3 (U3 is block diagonal with U3_00=1, U3_11=1)
     (this changes only third column of Q */
  a=Q[0][2][0]*U3[2][2][0]-Q[0][2][1]*U3[2][2][1];
  b=Q[0][2][0]*U3[2][2][1]+Q[0][2][1]*U3[2][2][0];
  Q[0][2][0]=a; Q[0][2][1]=b;
  a=Q[1][2][0]*U3[2][2][0]-Q[1][2][1]*U3[2][2][1];
  b=Q[1][2][0]*U3[2][2][1]+Q[1][2][1]*U3[2][2][0];
  Q[1][2][0]=a; Q[1][2][1]=b;
  a=Q[2][2][0]*U3[2][2][0]-Q[2][2][1]*U3[2][2][1];
  b=Q[2][2][0]*U3[2][2][1]+Q[2][2][1]*U3[2][2][0];
  Q[2][2][0]=a; Q[2][2][1]=b;

  /* final U=Q*UO3
     (unitary times orthogonal that accumulated Givens rotations) */
  U00re=Q[0][0][0]*UO3[0][0]+Q[0][1][0]*UO3[1][0]+Q[0][2][0]*UO3[2][0];
  U00im=Q[0][0][1]*UO3[0][0]+Q[0][1][1]*UO3[1][0]+Q[0][2][1]*UO3[2][0];
  U01re=Q[0][0][0]*UO3[0][1]+Q[0][1][0]*UO3[1][1]+Q[0][2][0]*UO3[2][1];
  U01im=Q[0][0][1]*UO3[0][1]+Q[0][1][1]*UO3[1][1]+Q[0][2][1]*UO3[2][1];
  U02re=Q[0][0][0]*UO3[0][2]+Q[0][1][0]*UO3[1][2]+Q[0][2][0]*UO3[2][2];
  U02im=Q[0][0][1]*UO3[0][2]+Q[0][1][1]*UO3[1][2]+Q[0][2][1]*UO3[2][2];
  U10re=Q[1][0][0]*UO3[0][0]+Q[1][1][0]*UO3[1][0]+Q[1][2][0]*UO3[2][0];
  U10im=Q[1][0][1]*UO3[0][0]+Q[1][1][1]*UO3[1][0]+Q[1][2][1]*UO3[2][0];
  U11re=Q[1][0][0]*UO3[0][1]+Q[1][1][0]*UO3[1][1]+Q[1][2][0]*UO3[2][1];
  U11im=Q[1][0][1]*UO3[0][1]+Q[1][1][1]*UO3[1][1]+Q[1][2][1]*UO3[2][1];
  U12re=Q[1][0][0]*UO3[0][2]+Q[1][1][0]*UO3[1][2]+Q[1][2][0]*UO3[2][2];
  U12im=Q[1][0][1]*UO3[0][2]+Q[1][1][1]*UO3[1][2]+Q[1][2][1]*UO3[2][2];
  U20re=Q[2][0][0]*UO3[0][0]+Q[2][1][0]*UO3[1][0]+Q[2][2][0]*UO3[2][0];
  U20im=Q[2][0][1]*UO3[0][0]+Q[2][1][1]*UO3[1][0]+Q[2][2][1]*UO3[2][0];
  U21re=Q[2][0][0]*UO3[0][1]+Q[2][1][0]*UO3[1][1]+Q[2][2][0]*UO3[2][1];
  U21im=Q[2][0][1]*UO3[0][1]+Q[2][1][1]*UO3[1][1]+Q[2][2][1]*UO3[2][1];
  U22re=Q[2][0][0]*UO3[0][2]+Q[2][1][0]*UO3[1][2]+Q[2][2][0]*UO3[2][2];
  U22im=Q[2][0][1]*UO3[0][2]+Q[2][1][1]*UO3[1][2]+Q[2][2][1]*UO3[2][2];

  /* Q=V1*V2 (V1 is block diagonal with V2_11=1,
              V2 is block diagonal with V2_11=1, V2_22=1) */
  Q[0][0][0]=V1[0][0][0]; Q[0][0][1]=V1[0][0][1];
  Q[1][0][0]=V1[1][0][0]; Q[1][0][1]=V1[1][0][1];
  Q[2][0][0]=V1[2][0][0]; Q[2][0][1]=V1[2][0][1];
  Q[0][1][0]=V1[0][1][0]; Q[0][1][1]=V1[0][1][1];
  Q[0][2][0]=V1[0][2][0]; Q[0][2][1]=V1[0][2][1];
  Q[1][1][0]=V1[1][1][0]; Q[1][1][1]=V1[1][1][1];
  Q[2][1][0]=V1[2][1][0]; Q[2][1][1]=V1[2][1][1];
  Q[1][2][0]=V1[1][2][0]*V2[2][2][0]-V1[1][2][1]*V2[2][2][1];
  Q[1][2][1]=V1[1][2][0]*V2[2][2][1]+V1[1][2][1]*V2[2][2][0];
  Q[2][2][0]=V1[2][2][0]*V2[2][2][0]-V1[2][2][1]*V2[2][2][1];
  Q[2][2][1]=V1[2][2][0]*V2[2][2][1]+V1[2][2][1]*V2[2][2][0];

  /* final V=Q*VO3
     (unitary times orthogonal that accumulated Givens rotations) */
  V00re=Q[0][0][0]*VO3[0][0]+Q[0][1][0]*VO3[1][0]+Q[0][2][0]*VO3[2][0];
  V00im=Q[0][0][1]*VO3[0][0]+Q[0][1][1]*VO3[1][0]+Q[0][2][1]*VO3[2][0];
  V01re=Q[0][0][0]*VO3[0][1]+Q[0][1][0]*VO3[1][1]+Q[0][2][0]*VO3[2][1];
  V01im=Q[0][0][1]*VO3[0][1]+Q[0][1][1]*VO3[1][1]+Q[0][2][1]*VO3[2][1];
  V02re=Q[0][0][0]*VO3[0][2]+Q[0][1][0]*VO3[1][2]+Q[0][2][0]*VO3[2][2];
  V02im=Q[0][0][1]*VO3[0][2]+Q[0][1][1]*VO3[1][2]+Q[0][2][1]*VO3[2][2];
  V10re=Q[1][0][0]*VO3[0][0]+Q[1][1][0]*VO3[1][0]+Q[1][2][0]*VO3[2][0];
  V10im=Q[1][0][1]*VO3[0][0]+Q[1][1][1]*VO3[1][0]+Q[1][2][1]*VO3[2][0];
  V11re=Q[1][0][0]*VO3[0][1]+Q[1][1][0]*VO3[1][1]+Q[1][2][0]*VO3[2][1];
  V11im=Q[1][0][1]*VO3[0][1]+Q[1][1][1]*VO3[1][1]+Q[1][2][1]*VO3[2][1];
  V12re=Q[1][0][0]*VO3[0][2]+Q[1][1][0]*VO3[1][2]+Q[1][2][0]*VO3[2][2];
  V12im=Q[1][0][1]*VO3[0][2]+Q[1][1][1]*VO3[1][2]+Q[1][2][1]*VO3[2][2];
  V20re=Q[2][0][0]*VO3[0][0]+Q[2][1][0]*VO3[1][0]+Q[2][2][0]*VO3[2][0];
  V20im=Q[2][0][1]*VO3[0][0]+Q[2][1][1]*VO3[1][0]+Q[2][2][1]*VO3[2][0];
  V21re=Q[2][0][0]*VO3[0][1]+Q[2][1][0]*VO3[1][1]+Q[2][2][0]*VO3[2][1];
  V21im=Q[2][0][1]*VO3[0][1]+Q[2][1][1]*VO3[1][1]+Q[2][2][1]*VO3[2][1];
  V22re=Q[2][0][0]*VO3[0][2]+Q[2][1][0]*VO3[1][2]+Q[2][2][0]*VO3[2][2];
  V22im=Q[2][0][1]*VO3[0][2]+Q[2][1][1]*VO3[1][2]+Q[2][2][1]*VO3[2][2];

  /* singular values */
  //sigma[0]=b00; sigma[1]=b11; sigma[2]=b22;
  sv[0]=b00*b00;
  sv[1]=b11*b11;
  sv[2]=b22*b22;

  /* construct U*V^+ as a reunitarized link W */
  for(i=0;i<3;i++)for(j=0;j<3;j++){
    Ad[i][j][0] = U[i][0][0]*V[j][0][0]+U[i][0][1]*V[j][0][1];
    Ad[i][j][1] =-U[i][0][0]*V[j][0][1]+U[i][0][1]*V[j][0][0];
    Ad[i][j][0]+= U[i][1][0]*V[j][1][0]+U[i][1][1]*V[j][1][1];
    Ad[i][j][1]+=-U[i][1][0]*V[j][1][1]+U[i][1][1]*V[j][1][0];
    Ad[i][j][0]+= U[i][2][0]*V[j][2][0]+U[i][2][1]*V[j][2][1];
    Ad[i][j][1]+=-U[i][2][0]*V[j][2][1]+U[i][2][1]*V[j][2][0];
  }
//TODO convert to floatT here
  SU3<floatT> w = SU3<floatT>(
                  COMPLEX(floatT)((floatT)Ad[0][0][0],(floatT)Ad[0][0][1]),
                  COMPLEX(floatT)((floatT)Ad[0][1][0],(floatT)Ad[0][1][1]),
                  COMPLEX(floatT)((floatT)Ad[0][2][0],(floatT)Ad[0][2][1]),
                  COMPLEX(floatT)((floatT)Ad[1][0][0],(floatT)Ad[1][0][1]),
                  COMPLEX(floatT)((floatT)Ad[1][1][0],(floatT)Ad[1][1][1]),
                  COMPLEX(floatT)((floatT)Ad[1][2][0],(floatT)Ad[1][2][1]),
                  COMPLEX(floatT)((floatT)Ad[2][0][0],(floatT)Ad[2][0][1]),
                  COMPLEX(floatT)((floatT)Ad[2][1][0],(floatT)Ad[2][1][1]),
                  COMPLEX(floatT)((floatT)Ad[2][2][0],(floatT)Ad[2][2][1])
                     );

  // inner part of SVD routine ends here

  return w;


}
#pragma GCC diagnostic pop
#endif /* GSVD_HCU_ */
