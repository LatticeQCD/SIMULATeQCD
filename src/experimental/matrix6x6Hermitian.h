#pragma once

#include "fullSpinor.h"

/*
// Symmetric 6x6 Matrix
template<class floatT>
struct Matrix6x6Sym {
    enum entry {
        e00, e11, e22, e33, e44, e55, e01r, e01i,  e02r, e02i, e03r, e03i, e04r, e04i, e05r, e05i, e12r, e12i, e13r, e13i, e14r, e14i, e15r, e15i, e23r, e23i, e24r, e24i, e25r, e25i, e34r, e34i, e35r, e35i, e45r, e45i;
    };
    floatT elems[10];


}
*/

// Symmetric 6x6 Matrix
template<class floatT>
struct Matrix6x6 {
    public:
    COMPLEX(floatT) val[6][6];


    __host__ __device__ Matrix6x6(){
        for(int i=0; i < 6; i++){
            for(int j=0; j < 6; j++){
               val[i][j]    = 0.0;
            }
        }
    }


    __host__ __device__ Matrix6x6<floatT> operator=(COMPLEX(floatT) in){
        Matrix6x6<floatT> out;
        for(int i=0; i < 6; i++){
            for(int j=0; j < 6; j++){
                out[i][j]    =  in[i][j];

            }
        } 
    return out;
    }

    __host__ __device__ Matrix6x6<floatT>(Vect18<floatT> & in){
        for(int i = 0; i < 3;i++){
           val[i][i]     = real(in.data[i]);
           val[i+3][i+3] = imag(in.data[i]);
        }
        int count = 3;
        for(int i = 0; i < 5;i++){
           for(int j = i+1; j < 6;j++){
               val[i][j] = in.data[count];
               val[j][i] = conj(in.data[count]);
               count++;
           }
        }

   }


        // save uppper hermitian matrix to vect18
        __host__ __device__ Vect18<floatT> ConvertHermitianToVect18(){
           Vect18<floatT> out(0.0);
           COMPLEX(floatT)  ii(0.0,1.0);
           for(int i = 0; i < 3;i++){
              out.data[i]    = real(val[i][i]) +ii*real(val[i+3][i+3]);
           }
           int count = 3;
           for(int i = 0; i < 5;i++){
              for(int j = i+1; j < 6;j++){
                 out.data[count]    = val[i][j];
                 count ++;
              }
           }
           return out;
        }


   __host__ __device__ Vect12<floatT> MatrixXVect12UpDown(const Vect12<floatT> & in, int down ){
       Vect12<floatT> out(0.0);
       for(int i = 0; i < 6;i++){
           for(int j = 0; j < 6;j++){
               out.data[i+6*down] += val[i][j]*in.data[j+6*down];
           }
       }
       for(int i = 0; i < 6;i++){
           out.data[i+6*(!down)] = in.data[i+6*(!down)];
       }
       return out;
   } 

    __host__ __device__ Matrix6x6<floatT> invert(){
        
        Matrix6x6<floatT> tmp;
        for(int i=0; i < 6; i++){
            for(int j=0; j < 6; j++){
               tmp.val[i][j] = val[i][j];
            }
        }    
        Matrix6x6<floatT> out;
        for(int i=0; i < 6; i++){
            out.val[i][i] = 1.0;
        }
                           
         for(int i=0; i < 6-1; i++){
            for(int j=i+1; j < 6; j++){
                 COMPLEX(floatT) aa = (tmp.val[j][i])*(1.0/(tmp.val[i][i]));
                 for(int k=0; k < 6; k++){
                     tmp.val[j][k] -= aa*tmp.val[i][k];
                     out.val[j][k] -= aa*out.val[i][k];
                 }
             }
         }
         for(int i=5; i > 0; i--){
            for(int j=i-1; j > -1; j--){
                 COMPLEX(floatT) aa = (tmp.val[j][i])*(1.0/(tmp.val[i][i]));
                 for(int k=0; k < 6; k++){
                     tmp.val[j][k] -= aa*tmp.val[i][k];
                     out.val[j][k] -= aa*out.val[i][k];
                 }
             }
         }
         for(int i=0; i < 6; i++){
            for(int k=0; k < 6; k++){
                out.val[i][k] = out.val[i][k]*(1.0/tmp.val[i][i]);
            }
         }
         return out;
     }

};



