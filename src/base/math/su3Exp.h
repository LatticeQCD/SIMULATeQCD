//
// Created by Lukas Mazur on 17.11.18.
//

#pragma once
#include "../../define.h"
#include "su3.h"
#include "float.h"


/* SU3Exp(SU3<floatT> & , SU3<floatT> & ,int = 10)
SU3 Matrix exponential Kernel
based on a recursive calculation of the Cayley Hamilton polynom with the
accuracy of 1/(N-1)! where N denotes the number of steps. The recursition is:

	exp(X) = q_(0,0) * I + q_(0, 1) * X + q_(0,2) * X^2

with I denoting the unity SU(3) element.
The q_(0,i) become calculated by:
	q_(N,0) = 1/N!
	q_(N,1) = q(N,2) = 0

	q_(n,0) = 1/n! + det(X) * q_(n+1,2)
	q_(n,1) = q_(n+1,0) + 0.5 * tr(X^2)
	q_(n,2) = q_(n+1,1)

	n = 0,1,...,N-1

N = 25 by default due to an estimated error of order 10^(-26)
	derived by 1/(N-1)!

*/

template<class floatT>
__device__ __host__ constexpr unsigned int countOfApproxInverseFak(){
    unsigned int N = 1;
    floatT nominator = 1.0;

    while (nominator > DBL_EPSILON) {
        N++;
        nominator /= (floatT) (N - 1);
    }
    return N;
}


// Algorithm from https://luscher.web.cern.ch/luscher/notes/su3fcts.pdf
template<class floatT>
__device__ __host__ inline void SU3Exp(const SU3<floatT> inSU3, SU3<floatT> &outSU3){

      constexpr unsigned int N = countOfApproxInverseFak<floatT>();
     floatT c_i[N+1];

    c_i[0] = 1.0;
    for (unsigned int i = 0; i < N; i++) {
        c_i[i+1] = c_i[i] / floatT(i + 1);
    }

    COMPLEX(floatT) d = det(inSU3);
    COMPLEX(floatT) t = -0.5 * tr_c(inSU3);

    COMPLEX(floatT) q_0_old = c_i[N], q_1_old = 0, q_2_old = 0;
    COMPLEX(floatT) q_0, q_1, q_2;

    for (int i = N - 1; i >= 0; i--) {
        q_0 = c_i[i] + d * q_2_old;
        q_1 = q_0_old - t * q_2_old;
        q_2 = q_1_old;

        q_0_old = q_0;
        q_1_old = q_1;
        q_2_old = q_2;
    }

    outSU3 = q_0 * su3_one<floatT>() + q_1 * inSU3 + q_2 * inSU3 * inSU3;
}

