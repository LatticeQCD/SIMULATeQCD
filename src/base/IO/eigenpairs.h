#include <iostream>
#include <string>
#include <typeinfo>
#include <stdexcept>
#include <fstream>
#include <complex>
#include "../../SIMULATeQCD.h"
#define PREC double
bool EVAllocated=false;
bool EVDefined = false;
typedef float floatT; // Define the precision here

#define GALERKIN_K_MAX 1024

__device__ __constant__ double a_lan_c[GALERKIN_K_MAX];

void init_alan_c(double* a_lan_h, int k){
  cudaMemcpyToSymbol(a_lan_c, a_lan_h, (k+1)*sizeof(double));
}

void ReadEV(const char *s, int nvec, const int sizeh){
  rootLogger.info("Called ReadEv() with ", s, " nvec: ", nvec, " sizeh : ", sizeh);
  const int HaloDepthSpin = 4;
  const int NStacks = 8;
  std::vector<double> a_lan_h;
  
  
  std::vector<Spinorfield<PREC, true, All, HaloDepthSpin, NStacks>> * R_lan_h;

  std::ifstream inEV;
  gVect3<floatT> tmp;
  //cudaError_t cudaErr;
  std::string fname="ReadEV()";

  //open file                                                                                                                                                                  
  inEV.open(s, std::ios::in | std::ios::binary);
  if (inEV){
    rootLogger.info("Opened Eigenvalue file :", s);
    //VRB.FFlow(_devId,*cname,*fname,"Reading eigenvectors from %s",s);
    //Alocate Memory                                                                                                                                                           
    //if(!EVAllocated){
    R_lan_h = new std::vector<Spinorfield<PREC, true, All, HaloDepthSpin, NStacks>>;
      //Spinorfield<PREC, true, All, HaloDepthSpin, NStacks> * SpinorfieldIn;
      //Allocate memory for eigenvalues
      //----------------------------------------------------------------
    //   cudaErr = cudaMallocHost( reinterpret_cast< void** >( &a_lan_h ), (nvec+1)*sizeof(double) );
    //  if( cudaErr )
	//throw CudaError( "Failed to allocate memory on host for a_lan_h", cudaErr );
    //EVAllocated=true;
    //}
    // gVect3arrayAcc acc = R_lan_h->get_allocator();
    std::allocator<floatT> acc = R_lan_h->get_allocator();
    
    //read R_lan to Host Memory                                                                                                                                                
    for(int i=0;i<nvec;i++){
      inEV.read( (char*)(&a_lan_h[i]), sizeof(float) );
      rootLogger.info("ReadEv()  i:",i, " a_lan_h : ",a_lan_h[i]);
      //VRB.FFlow(_devId,*cname,*fname,"a_lan_h[%d]=%e",i,a_lan_h[i+1]);
      for(int j=0;j<sizeh;j++){
	inEV.read( (char*)(&tmp), sizeof(std::vector<floatT>) );
	rootLogger.info("ReadEv()  j:",j, " vec : ",tmp(j));
	//acc.setElement(i*sizeh+j,tmp);
      }
    }
    //copy R_lan to device
    
    //init_alan_c(a_lan_h,nvec);
    EVDefined=true;

    //close file                                                                                                                                                                 
    inEV.close();
  }
}
