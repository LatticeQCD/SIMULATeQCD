//!< Test of the k-th eigenvalue equation
float GPUsu3lattice::FeigTest(GPUcvect3array<SPcomplex> * R, double * a, int k, CgArg *cg_arg, GPUsu3latticePerformanceCounters * const counters, int index_R, cudaStream_t stream_calc){


  float norm;
  int vsizeh;
  DPcomplex dc;
  fname->assign("FeigTest()");
  //GPUcvect3array<SPcomplex> * R = R_lan;
  //double * a = a_lan_h;

  #ifdef EVSTREAM
  vsizeh = latticeSize.sizeh();
  double *ads=NULL;
  cudaMalloc((void**)&ads, sizeof(double) * vsizeh);
  double *aos=NULL;
  cudaMalloc((void**)&aos, sizeof(double) * vsizeh);

  setmass(cg_arg->mass);

  /*
    //Debug
    GPUcvect3array<SPcomplex> *Rtemp_h;
    Rtemp_h  = new GPUcvect3array<SPcomplex>(false, vsizeh); 
    //transferring result to the host
    Rtemp_h -> copyFrom(*R, vsizeh, index_R*vsizeh, 0);
    //printing
    GPUcvect3arrayAccessor<SPcomplex> acc_Rth = Rtemp_h->getAccessor();
    cout << "printing new Rtemp_h before dc in FeigTest" << endl;
    for (int i=0; i<vsizeh; i++) 
    cout << acc_Rth.getElement(i) << endl;
    cout << "fine" << endl; 
    //exit(1);
    */

  dc=R->CompDotProdSumD(R->getAccessor(index_R*vsizeh),(index_R)*vsizeh,vsizeh, stream_calc);
  VRB.FFlow(_devId,*cname,*fname,"Norm sq of eigenvector: %le",dc.real);

  
  Doe( d_loc_h->getAccessor(), R->getAccessor(index_R*vsizeh), counters, stream_calc );
  Deo( d_loc_s->getAccessor(), d_loc_h->getAccessor(), counters, stream_calc );
  CG1SumD(R->getAccessor(index_R*vsizeh), d_loc_s->getAccessor(), ads, stream_calc);

  d_loc_s->FTimesV1PlusV2(-a[k+1],R->getAccessor(index_R*vsizeh),d_loc_s->getAccessor(),0,vsizeh, stream_calc);

  /*
    //Debug
    GPUcvect3array<SPcomplex> *ds_h;
    ds_h  = new GPUcvect3array<SPcomplex>(false, vsizeh); 
    //transferring result to the host
    ds_h -> copyFrom(*d_loc_s, vsizeh, 0, 0);
    //printing
    GPUcvect3arrayAccessor<SPcomplex> acc_dsh = ds_h->getAccessor();
    cout << "printing new ds_h after DeoDoe in FeigTest" << endl;
    for (int i=0; i<vsizeh; i++) 
    cout << acc_dsh.getElement(i) << endl;
    cout << "fine" << endl; 
    //exit(1);
    */

  norm=d_loc_s->ReDotProdSumD(d_loc_s,vsizeh, stream_calc);

  //Debug
  //printf("new norm=%e \n", norm);
  //exit(1);

  cudaFree(ads);
  cudaFree(aos);

  #else

  const int kvec = k;

  vsizeh = latticeSize.sizeh();
  double *ads=NULL;
  cudaMalloc((void**)&ads, sizeof(double) * vsizeh);
  double *aos=NULL;
  cudaMalloc((void**)&aos, sizeof(double) * vsizeh);

  setmass(cg_arg->mass);

  /*
    //Debug
    GPUcvect3array<SPcomplex> *Rtemp_h;
    Rtemp_h  = new GPUcvect3array<SPcomplex>(false, vsizeh); 
    //transferring result to the host
    Rtemp_h -> copyFrom(*R, vsizeh, kvec*vsizeh, 0);
    //printing
    GPUcvect3arrayAccessor<SPcomplex> acc_Rth = Rtemp_h->getAccessor();
    cout << "printing old Rtemp_h before dc in FeigTest" << endl;
    for (int i=0; i<vsizeh; i++) 
    cout << acc_Rth.getElement(i) << endl;
    cout << "fine" << endl; 
    //exit(1);
    */

  dc=R->CompDotProdSumD(R->getAccessor(kvec*vsizeh),(kvec*vsizeh),vsizeh);
  VRB.FFlow(_devId,*cname,*fname,"Norm sq of eigenvector: %le",dc.real);


  Doe( d_loc_h->getAccessor(), R->getAccessor(kvec*vsizeh), counters );
  Deo( d_loc_s->getAccessor(), d_loc_h->getAccessor(), counters );
  CG1SumD(R->getAccessor(kvec*vsizeh), d_loc_s->getAccessor(), ads);


  d_loc_s->FTimesV1PlusV2(-a[k+1],R->getAccessor(kvec*vsizeh),d_loc_s->getAccessor(),0,vsizeh);

  /*
    //Debug
    GPUcvect3array<SPcomplex> *ds_h;
    ds_h  = new GPUcvect3array<SPcomplex>(false, vsizeh); 
    //transferring result to the host
    ds_h -> copyFrom(*d_loc_s, vsizeh, 0, 0);
    //printing
    GPUcvect3arrayAccessor<SPcomplex> acc_dsh = ds_h->getAccessor();
    cout << "printing old ds_h after DeoDoe in FeigTest" << endl;
    for (int i=0; i<vsizeh; i++) 
    cout << acc_dsh.getElement(i) << endl;
    cout << "fine" << endl; 
    //exit(1);
    */

  norm=d_loc_s->ReDotProdSumD(d_loc_s,vsizeh);

  //Debug
  //printf("old norm=%e \n", norm);
  //exit(1);

  cudaFree(ads);
  cudaFree(aos);

#endif /*EVSTREAM*/

  return norm;

}
