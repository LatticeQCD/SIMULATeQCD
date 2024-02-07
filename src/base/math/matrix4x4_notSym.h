//
// Created by Rasmus Larsen 10 2022
//

#ifndef MATRIX4X4_NOTSYM_H
#define MATRIX4X4_NOTSYM_H

#include "../../define.h"

template<class floatT>
class Matrix4x4{
    public:
    floatT a[4*4];
    __host__ __device__ Matrix4x4(){
        for( int i = 0; i < 4*4; i++){
            a[i] = 0.0;
        }
    }
    __host__ __device__ Matrix4x4<floatT> mult(Matrix4x4<floatT> M2){
        Matrix4x4<floatT> Mout;
        for( int i = 0; i < 4; i++){
           for( int j = 0; j < 4; j++){
              for( int k = 0; k < 4; k++){
                 Mout.a[i+4*j] = Mout.a[i+4*j] + a[i+4*k]*M2.a[k+4*j];
              }
           }
        }
        return Mout;
    }

     __host__ __device__ floatT vecP(int i0, int j0){
        floatT sum = 0.0;
        for( int i = 0; i < 4; i++){
            sum = sum +a[i0+4*i]*a[j0+4*i];
        }
        return sum;
    }

     __host__ __device__ Matrix4x4<floatT> dagger(){
        Matrix4x4<floatT> Mout;
        for( int i = 0; i < 4; i++){
           for( int j = 0; j < 4; j++){
                 Mout.a[i+4*j] = a[j+4*i];
           }
        }
        return Mout;
    }

};


template<class floatT>
__host__ __device__  void QR(Matrix4x4<floatT> & MatOut,floatT * vecOut, Matrix4x4<floatT> & MatIn){
   for( int i = 0; i < 4*4; i++){
          MatOut.a[i] = 0.0;
   }
   for( int i = 0; i < 4; i++){
      MatOut.a[i+4*i] = 1.0;
      vecOut[i] = 0.0;
   }

   Matrix4x4<floatT> Q;

   floatT diff = 1.0;
   floatT dot;


   while( diff > 1e-20){
       Q=MatIn;
       /// orthogonalize
       for( int j = 0; j < 3; j++){
           for( int i = j+1; i < 4; i++){
               dot = (Q.vecP(i,j))*Q.vecP(j,j)/(abs(Q.vecP(j,j))*abs(Q.vecP(j,j)));
               for( int k = 0; k < 4; k++){
                    Q.a[i+4*k] = Q.a[i+4*k] - Q.a[j+4*k]*dot;
               }
           }
       }
       /// normalize
       for( int i = 0; i < 4; i++){
          dot = 1.0/sqrt(abs(Q.vecP(i,i)));
          for( int k = 0; k < 4; k++){
              Q.a[i+4*k] = Q.a[i+4*k]*dot;
          }
       }

/*       for( int i = 0; i < 4; i++){
          for( int k = 0; k < 4; k++){
              std::cout << Q.a[i+4*k] << " ";
          }
          std::cout <<  std::endl;
       }
*/


       // Apply to matrix with eigenvalues and eigenvectors
       MatIn = Q.mult(MatIn.mult(Q.dagger()));
//       MatIn = (Q.dagger()).mult(MatIn.mult(Q));
       MatOut= Q.mult(MatOut);
       diff = 0.0;
       // calculate difference compared to last step
       for( int i = 0; i < 4; i++){
          diff = diff + abs(MatIn.a[i+4*i]-vecOut[i])* abs(MatIn.a[i+4*i]-vecOut[i]);
          vecOut[i] = MatIn.a[i+4*i];
       }
//       std::cout << diff << std::endl;
   }

}


template<class floatT>
__host__ __device__ void getSU2Rotation(floatT * vecOut,floatT * vecEI, floatT * vecb, Matrix4x4<floatT> & MatIn){

    int order[4];
    order[0]=0;
    order[1]=1;
    order[2]=2;
    order[3]=3;

    for( int i = 0; i < 4; i++){
        for( int j = 0; j < 3; j++){
            if( vecEI[order[j]] > vecEI[order[j+1]]  ){
                int pp = order[j+1];
                order[j+1] = order[j];
                order[j]   = pp;
            }
        }
    }

    floatT bk[4];
    for( int i = 0; i < 4; i++){
       bk[i] = 0.0;
       for( int j = 0; j < 4; j++){
          bk[i] = bk[i] + vecb[j]*MatIn.a[i+4*j];
//          bk[i] = bk[i] + vecb[j]*MatIn.a[j+4*i];
       }
    }
/*
    for( int i = 0; i < 4; i++){
       std::cout << bk[i] << " ";
    }
    std::cout << std::endl;

    for( int i = 0; i < 4; i++){
       std::cout << order[i] << " ";
    }
    std::cout << std::endl;

    for( int i = 0; i < 4; i++){
       std::cout << vecEI[order[i]] << " ";
    }
    std::cout << std::endl;   
*/
    floatT lamb = vecEI[order[3]];
    floatT diff = 1.0;
    while(diff > 1e-20){
        floatT old = lamb;
        lamb=vecEI[order[3]] + sqrt(bk[order[3]]*bk[order[3]]
                                   +bk[order[0]]*bk[order[0]]*(vecEI[order[3]]-lamb)*(vecEI[order[3]]-lamb)/(vecEI[order[0]]-lamb)/(vecEI[order[0]]-lamb)
                                   +bk[order[1]]*bk[order[1]]*(vecEI[order[3]]-lamb)*(vecEI[order[3]]-lamb)/(vecEI[order[1]]-lamb)/(vecEI[order[1]]-lamb)
                                   +bk[order[2]]*bk[order[2]]*(vecEI[order[3]]-lamb)*(vecEI[order[3]]-lamb)/(vecEI[order[2]]-lamb)/(vecEI[order[2]]-lamb)
          );

        diff=(old-lamb)*(old-lamb);
    }

//    std::cout << lamb << std::endl;

    for( int i = 0; i < 4; i++){
       vecOut[i] = 0.0;
    }

    for( int i = 0; i < 4; i++){
        for( int j = 0; j < 4; j++){
            vecOut[j] = vecOut[j] + MatIn.a[i+4*j]*bk[i]/(vecEI[i]-lamb);
//            vecOut[j] = vecOut[j] + MatIn.a[j+4*i]*bk[i]/(vecEI[i]-lamb);
        }
    }


}




#endif
