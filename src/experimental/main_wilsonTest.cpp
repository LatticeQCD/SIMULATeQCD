#include "../simulateqcd.h"
#include "fullSpinor.h"
//#include "fullSpinorfield.h"
//#include "gammaMatrix.h"
#include "DWilson.h"

template<class floatT>
struct wilsonParam : LatticeParameters {
    Parameter <std::string> gauge_file;

    wilsonParam() {
        addOptional(gauge_file, "gauge_file");
    }
};

template<class floatT, size_t HaloDepth>
struct MakePointSource{

    //Gauge accessor to access the gauge field
    SpinorColorAcc<floatT> _SpinorColorAccessor;
    int _color = 0;
    int _spin  = 0;



    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    MakePointSource(Spinorfield<floatT, true, All, HaloDepth, 12, 1> &spinorIn, int color, int spin)
                : _SpinorColorAccessor(spinorIn.getAccessor()), _color(color), _spin(spin)
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ Vect12<floatT> operator()(gSite site) {

        ColorVect<floatT> outSC;
        //LatticeData lat = GIndexer<All, HaloDepth>::getLatData();

        //if(site.coord[0] == 0 && site.coord[1] == 0 && site.coord[2] == 0 && site.coord[3] == 0  && lat.gPosZ == 0){

        sitexyzt coord = GIndexer<All, HaloDepth>::getLatData().globalPos(site.coord);
        if(coord[0] == 0 && coord[1] == 0 && coord[2] == 0 && coord[3] == 0 ){
            outSC[_spin].data[_color] = 1.0;
        }

        return convertColorVectToVect12(outSC);
    }
};

template<Layout LatLayout, size_t HaloDepth>
struct ReadIndex {
    inline __host__ __device__ gSite operator()(const dim3& blockDim, const uint3& blockIdx, const uint3& threadIdx) {
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        typedef GIndexer<LatLayout, HaloDepth> GInd;
        gSite site = GInd::getSite(i);
        return site;
    }
};

template<class floatT,Layout layoutRHS, size_t HaloDepth>
struct CopyAll{

    //Gauge accessor to access the gauge field
    Vect12ArrayAcc<floatT> _SpinorAll;
    Vect12ArrayAcc<floatT> _Spinor;

    typedef GIndexer<All, HaloDepth > GIndAll;
    typedef GIndexer<layoutRHS, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    CopyAll(Spinorfield<floatT, true, All, HaloDepth, 12, 12> &spinorInAll, Spinorfield<floatT, true, layoutRHS, HaloDepth, 12, 12> &spinorIn)
                : _SpinorAll(spinorInAll.getAccessor()), _Spinor(spinorIn.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ void operator()(gSite site) {

    for (size_t stack = 0; stack < 12; stack++) {
        const gSiteStack writeSite = GInd::getSiteStack(site,stack);

        Vect12<floatT> tmp = _Spinor.getElement(GInd::getSiteStack(site,stack));
  //      _SpinorAll.setElement(tmp,site);
       _SpinorAll.setElement(GIndAll::getSiteStack(GInd::template convertSite<All, HaloDepth>(site),stack),tmp );
       // return tmp;
      }
    }
};

template<class floatT, size_t HaloDepth>
struct MakePointSource12{

    //Gauge accessor to access the gauge field
    Vect12ArrayAcc<floatT> _spinorIn;

    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    MakePointSource12(Spinorfield<floatT, true, All, HaloDepth, 12, 12> & spinorIn)
                : _spinorIn(spinorIn.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ void operator()(gSite site) {

        for (size_t stack = 0; stack < 12; stack++) {
            Vect12<floatT> tmp(0.0);

      //      LatticeData lat = GIndexer<All, HaloDepth>::getLatData();
      //      if(site.coord[0] == 0 && site.coord[1] == 0 && site.coord[2] == 0 && site.coord[3] == 0 && lat.gPosZ == 0 ){


            sitexyzt coord = GIndexer<All, HaloDepth>::getLatData().globalPos(site.coord);
            if(coord[0] == 0 && coord[1] == 0 && coord[2] == 0 && coord[3] == 0 ){
                tmp.data[stack] = 1.0;
  //               const gSiteStack site2 = GInd::getSiteStack(site,stack);
  //               printf("%lu ", site2.isiteStack);
            }
            const gSiteStack writeSite = GInd::getSiteStack(site,stack);
            _spinorIn.setElement(writeSite,tmp);

        }
    }
};


template<class floatT, size_t HaloDepth, size_t NStacks>
struct MakePointSourceIdendity{

    //Gauge accessor to access the gauge field
    Vect12ArrayAcc<floatT> _spinorIn;

    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    MakePointSourceIdendity(Spinorfield<floatT, true, All, HaloDepth, 12, NStacks> & spinorIn)
                :  _spinorIn(spinorIn.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ Vect12<floatT> operator()(gSiteStack site) {

//        LatticeData lat = GIndexer<All, HaloDepth>::getLatData();

        Vect12<floatT> tmp((floatT)0.0);
//        if(site.coord[0] == 0 && site.coord[1] == 0 && site.coord[2] == 0 && site.coord[3] == 0  && lat.gPosZ == 0){
        sitexyzt coord = GIndexer<All, HaloDepth>::getLatData().globalPos(site.coord);
        if(coord[0] == 0 && coord[1] == 0 && coord[2] == 0 && coord[3] == 0 ){
            tmp.data[site.stack] = 1.0;
           // tmp.data[11] = 1.0;
            _spinorIn.setElement(site,tmp);
/*        
           printf("%lu \n", site.isiteStack);
           printf("idendity \n");
           for (size_t i2 = 0; i2 < 12; i2++) {
               printf("%f  %f %f %f \n", real(tmp.data[i2]), imag(tmp.data[i2]), 1.0*site.stack, 1.0*i2 );
           }
*/      
       }

/*
        SimpleArray<Vect12<floatT>, NStacks> Stmp((floatT)0.0);


        if(site.coord[0] == 0 && site.coord[1] == 0 && site.coord[2] == 0 && site.coord[3] == 0 ){
            for (size_t stack = 0; stack < NStacks; stack++) {
                Stmp[stack].data[stack] = 1.0; 
            }
        }
        
        
        for (size_t stack = 0; stack < NStacks; stack++) {
             const gSiteStack writeSite = GInd::getSiteStack(site,stack);
            _spinorIn.setElement(writeSite,Stmp[stack]);
        }

        if(site.coord[0] == 0 && site.coord[1] == 0 && site.coord[2] == 0 && site.coord[3] == 0 ){
             printf("idendity \n");
             for (size_t stack = 0; stack < 1; stack++) {
                 Vect12<floatT> tmp = _spinorIn.getElement(GInd::getSiteStack(site,stack));
                 printf("%f  %f \n", real(tmp.data[stack]), imag(tmp.data[stack]) );
             }
        }
*/

         
        tmp = Vect12<floatT>(0.0); 

        return tmp;
     
    }
};

int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(DEBUG);

    using PREC = double;

    //LatticeParameters param;

    //const int LatDim[] = {20, 20, 20, 20};
    //const int NodeDim[] = {1, 1, 1, 1};

    //param.latDim.set(LatDim);
    //param.nodeDim.set(NodeDim);

    wilsonParam<PREC> param;

    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/test.param", argc, argv);

    commBase.init(param.nodeDim());

    const size_t HaloDepth = 2;

    PREC mass = 2.0;

    rootLogger.info("Initialize Lattice");
    /// Initialize the Indexer on GPU and CPU.
    initIndexer(HaloDepth,param,commBase);
    
    
    Gaugefield<PREC, true,HaloDepth> gauge(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 12, 1> spinor_res(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 12, 1> spinor_in(commBase);

    grnd_state<true> d_rand;
    initialize_rng(1337, d_rand);
 
    gauge.readconf_nersc(param.GaugefileName());
    gauge.updateAll();

    spinor_in.template iterateOverBulk(MakePointSource<PREC, HaloDepth>(spinor_in,0,0));
    spinor_in.updateAll();

    SimpleArray<COMPLEX(double), 1> dot2(0); 
    dot2 = spinor_in.dotProductStacked(spinor_in);
    std::cout << "dot source " << dot2[0] << std::endl;
 
    DWilson<PREC,true,All,2,2,1> dslash(gauge,1.0,0.0); 
    dslash.applyMdaggM(spinor_res,spinor_in);

    dot2 = spinor_in.dotProductStacked(spinor_in);



    dot2 = spinor_res.dotProductStacked(spinor_res);
    std::cout << "dot dslash " << dot2[0] << std::endl;

   // timer
    StopWatch<true> timer;
    timer.start();


    // sum over all sources, will be made into 12 stack later
    COMPLEX(double) CC[20];
    for (int col=0; col<3; col++){
        for (int spin=0; spin<4; spin++){
            spinor_in.template iterateOverBulk(MakePointSource<PREC, HaloDepth>(spinor_in,col,spin));
            //spinor_in.template iterateOverBulk(MakePointSource<PREC, HaloDepth>(spinor_in,2,3));

            DWilsonInverse<PREC,true,2,2,1> dslashinverse(gauge,mass,0.0);
            dslashinverse.gamma5MultVec(spinor_in,spinor_in); 
            dslashinverse.DslashInverse(spinor_res,spinor_in,10000,1e-10);
            for (int t=0; t<20; t++){
                COMPLEX(double) val = dslashinverse.Correlator(t,spinor_res);
                CC[t] = CC[t] + val;
                std::cout << 12.0*val << std::endl;
            }
        }
    }
    timer.stop();
    timer.print("Test Kernel runtime");


    for (int t=0; t<20; t++){
        std::cout << CC[t] << std::endl;
    }


    /// stacked version
    for (int t=0; t<20; t++){
        CC[t] = 0.0;
    }
    Spinorfield<PREC, true, All, HaloDepth, 12, 12> spinor_res12(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 12, 12> spinor_in12(commBase);
    SimpleArray<COMPLEX(double), 12> dot3(0);

    spinor_res12.template iterateOverBulk(MakePointSourceIdendity<PREC, HaloDepth,12>(spinor_in12));

    dot3 = spinor_in12.dotProductStacked(spinor_in12);
    for (int t=0; t<12; t++){
        std::cout << "dot source " << dot3[t] << std::endl;
    }

    


    // test dslash by itself
//    DWilson<PREC,true,All,2,2,12> dslash12(gauge,2.0,0.0);
//    dslash12.applyMdaggM(spinor_res12,spinor_in12);



          typedef GIndexer<All, 2> GInd;
          size_t _elems = GInd::getLatData().vol4;
          CalcGSite<All, 2> calcGSite;
          ReadIndex<All,HaloDepth> index;
          iterateFunctorNoReturn<true,32>(MakePointSource12<PREC,2>( spinor_in12),index,_elems);
//          spinor_in12.template iterateOverBulk(MakePointSource12<PREC,2>( spinor_in12));

    dot3 = spinor_in12.dotProductStacked(spinor_in12);
    for (int t=0; t<12; t++){
        std::cout << "dot source2 " << dot3[t] << std::endl;
    }

          iterateFunctorNoReturn<true,32>(gamma5DiracWilsonStack<PREC,2,2,12>(gauge,spinor_res12, spinor_in12,mass,0.0),index,_elems);
    dot3 = spinor_in12.dotProductStacked(spinor_in12);
    for (int t=0; t<12; t++){
        std::cout << "dot source after " << dot3[t] << std::endl;
    }

    // timer
    timer.reset();
    timer.start();

    //iterateFunctorNoReturn<true,32>(MakePointSourceIdendity<PREC, HaloDepth,12>(spinor_in));
    DWilsonInverse<PREC,true,2,2,12> dslashinverse12(gauge,mass,0.0);
    dslashinverse12.gamma5MultVec(spinor_in12,spinor_in12);
    dslashinverse12.DslashInverse(spinor_res12,spinor_in12,10000,1e-10);
    for (int t=0; t<20; t++){
        CC[t] = CC[t] + dslashinverse12.Correlator(t,spinor_res12);
    }
    for (int t=0; t<20; t++){
        std::cout << CC[t] << std::endl;
    }
    timer.stop();
    timer.print("Test Kernel runtime");

     
//    SimpleArray<COMPLEX(double), 12> dot3(0);
    dot3 = spinor_res12.dotProductStacked(spinor_res12);
    for (int t=0; t<12; t++){
        std::cout << "dot dslash stacked " << dot3[t] << std::endl;
    }


    /// timings
    DWilsonInverse<PREC,true,2,2,1> dslashinverse1(gauge,mass,0.0);
    for (int col=0; col<3; col++){
        for (int spin=0; spin<4; spin++){
             timer.reset();
             timer.start();
             dslashinverse1.DslashInverse(spinor_res,spinor_in,10000,1e-10);
             timer.stop();
             timer.print("Test Kernel runtime 1");

        }
    }

//cut off to not compile too much
/*
    Spinorfield<PREC, true, All, HaloDepth, 12, 2> spinor_res2(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 12, 2> spinor_in2(commBase);
    spinor_res2.template iterateOverBulk(MakePointSourceIdendity<PREC, HaloDepth,2>(spinor_in2));

    DWilsonInverse<PREC,true,2,2,12> _dslashinverse12(gauge,mass,0.0);
    timer.reset();
    timer.start();
    _dslashinverse12.DslashInverse(spinor_res12,spinor_in12,10000,1e-10);
    timer.stop();
    timer.print("Test Kernel runtime 12");

    DWilsonInverse<PREC,true,2,2,2> dslashinverse2(gauge,mass,0.0);
    timer.reset();
    timer.start();
    dslashinverse2.DslashInverse(spinor_res2,spinor_in2,10000,1e-10);
    timer.stop();
    timer.print("Test Kernel runtime 2");

    Spinorfield<PREC, true, All, HaloDepth, 12, 3> spinor_res3(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 12, 3> spinor_in3(commBase);
    spinor_res3.template iterateOverBulk(MakePointSourceIdendity<PREC, HaloDepth,3>(spinor_in3));
    
    Spinorfield<PREC, true, All, HaloDepth, 12, 4> spinor_res4(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 12, 4> spinor_in4(commBase);
    spinor_res4.template iterateOverBulk(MakePointSourceIdendity<PREC, HaloDepth,4>(spinor_in4));
    
    Spinorfield<PREC, true, All, HaloDepth, 12, 5> spinor_res5(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 12, 5> spinor_in5(commBase);
    spinor_res5.template iterateOverBulk(MakePointSourceIdendity<PREC, HaloDepth,5>(spinor_in5));
    
    Spinorfield<PREC, true, All, HaloDepth, 12, 6> spinor_res6(commBase);
    Spinorfield<PREC, true, All, HaloDepth, 12, 6> spinor_in6(commBase);
    spinor_res6.template iterateOverBulk(MakePointSourceIdendity<PREC, HaloDepth,6>(spinor_in6));
    
    DWilsonInverse<PREC,true,2,2,3> _dslashinverse3(gauge,mass,0.0);
    timer.reset();
    timer.start();
    _dslashinverse3.DslashInverse(spinor_res3,spinor_in3,10000,1e-10);
    timer.stop();
    timer.print("Test Kernel runtime 3");

    DWilsonInverse<PREC,true,2,2,4> _dslashinverse4(gauge,mass,0.0);
    timer.reset();
    timer.start();
    _dslashinverse4.DslashInverse(spinor_res4,spinor_in4,10000,1e-10);
    timer.stop();
    timer.print("Test Kernel runtime 4");

    DWilsonInverse<PREC,true,2,2,5> _dslashinverse5(gauge,mass,0.0);
    timer.reset();
    timer.start();
    _dslashinverse5.DslashInverse(spinor_res5,spinor_in5,10000,1e-10);
    timer.stop();
    timer.print("Test Kernel runtime 5");

    DWilsonInverse<PREC,true,2,2,6> _dslashinverse6(gauge,mass,0.0);
    timer.reset();
    timer.start();
    _dslashinverse6.DslashInverse(spinor_res6,spinor_in6,10000,1e-10);
    timer.stop();
    timer.print("Test Kernel runtime 6");




    DWilson<PREC,true,All,2,2,1> dslash1(gauge,2.0,0.0);
    dslash1.applyMdaggM(spinor_res,spinor_in);    
    DWilson<PREC,true,All,2,2,2> dslash2(gauge,2.0,0.0);
    dslash2.applyMdaggM(spinor_res2,spinor_in2);     
    DWilson<PREC,true,All,2,2,3> dslash3(gauge,2.0,0.0);
    dslash3.applyMdaggM(spinor_res3,spinor_in3);   
    DWilson<PREC,true,All,2,2,4> dslash4(gauge,2.0,0.0);
    dslash4.applyMdaggM(spinor_res4,spinor_in4);   
    DWilson<PREC,true,All,2,2,5> dslash5(gauge,2.0,0.0);
    dslash5.applyMdaggM(spinor_res5,spinor_in5);   
    DWilson<PREC,true,All,2,2,6> dslash6(gauge,2.0,0.0);
    dslash6.applyMdaggM(spinor_res6,spinor_in6);   



    DWilson<PREC,true,All,2,2,12> dslash12(gauge,2.0,0.0);
    dslash12.applyMdaggM(spinor_res12,spinor_in12);     



    timer.reset();
    timer.start();
    for (int col=0; col<100; col++){
    dslash1.applyMdaggM(spinor_res,spinor_in);    
    }
    timer.stop();
    timer.print("Dslash runtime 1");

    timer.reset();
    timer.start();
    for (int col=0; col<100; col++){
    dslash2.applyMdaggM(spinor_res2,spinor_in2);
    }
    timer.stop();
    timer.print("Dslash runtime 2");

    timer.reset();
    timer.start();
    for (int col=0; col<100; col++){
    dslash3.applyMdaggM(spinor_res3,spinor_in3);
    }
    timer.stop();
    timer.print("Dslash runtime 3");

    timer.reset();
    timer.start();
    for (int col=0; col<100; col++){
    dslash4.applyMdaggM(spinor_res4,spinor_in4);
    }
    timer.stop();
    timer.print("Dslash runtime 4");

    timer.reset();
    timer.start();
    for (int col=0; col<100; col++){
    dslash5.applyMdaggM(spinor_res5,spinor_in5);
    }
    timer.stop();
    timer.print("Dslash runtime 5");

    timer.reset();
    timer.start();
    for (int col=0; col<100; col++){
    dslash6.applyMdaggM(spinor_res6,spinor_in6);
    }
    timer.stop();
    timer.print("Dslash runtime 6");

    timer.reset();
    timer.start();
    for (int col=0; col<100; col++){
    dslash12.applyMdaggM(spinor_res12,spinor_in12);
    }
    timer.stop();
    timer.print("Dslash runtime 12");
*/

    //spinor_res.template iterateOverBulk(MakePointSourceIdendity<PREC, HaloDepth,1>(spinor_in));
    for (int t=0; t<20; t++){
        CC[t] = 0.0;
    }

    SpinorfieldAll<PREC,true, HaloDepth, 12, 1> spinorAll_res(commBase);
    SpinorfieldAll<PREC,true, HaloDepth, 12, 1> spinorAll_in(commBase);
    for (int col=0; col<3; col++){
        for (int spin=0; spin<4; spin++){
          spinor_in.template iterateOverBulk(MakePointSource<PREC, HaloDepth>(spinor_in,col,spin));
          spinorAll_res = spinor_res;
          spinorAll_in = spinor_in;
          spinorAll_in.updateAll();

          DWilsonInverseShurComplement<PREC,true,2,2,1> _dslashinverseSC(gauge,mass,1.0);
          timer.reset();
          timer.start();
          _dslashinverseSC.DslashInverseShurComplementClover(spinorAll_res,spinorAll_in,10000,1e-10);
          timer.stop();
          timer.print("Shur test 1");
  
          spinor_res = spinorAll_res;
 
          for (int t=0; t<20; t++){
             COMPLEX(double) val = dslashinverse1.Correlator(t,spinor_res);
             std::cout << 12.0*val << std::endl;
             CC[t] = CC[t] + val;         
          }
       }
    }

    for (int t=0; t<20; t++){
        std::cout << CC[t] << std::endl;
    }

/// stacked test

    SpinorfieldAll<PREC,true, HaloDepth, 12, 12> spinorAll_res12(commBase);
    SpinorfieldAll<PREC,true, HaloDepth, 12, 12> spinorAll_in12(commBase);
    iterateFunctorNoReturn<true,32>(MakePointSource12<PREC,2>( spinor_in12),index,_elems);

    spinor_in12.template iterateOverBulk<32>(Print(spinor_in12));
    spinorAll_in12 = spinor_in12;
    spinorAll_in12.even.template iterateOverBulk<32>(Print(spinorAll_in12.even));

    DWilsonInverseShurComplement<PREC,true,2,2,12> _dslashinverseSC12(gauge,mass,1.0);
    timer.reset();
    timer.start();
    _dslashinverseSC12.DslashInverseShurComplementClover(spinorAll_res12,spinorAll_in12,10000,1e-10);
    timer.stop();
    timer.print("Shur test 12");

//    spinor_res12 = spinorAll_res12;
//     spinorAll_res12.even.template iterateOverBulk<32>(CopyAll(spinor_res12,spinorAll_res12.even));
//     spinorAll_res12.odd.template iterateOverBulk<32>(CopyAll(spinor_res12,spinorAll_res12.odd));
    ReadIndex<Even,HaloDepth> indexE;
    iterateFunctorNoReturn<true,32>(CopyAll<PREC,Even,2>(spinor_res12,spinorAll_res12.even),indexE,_elems/2);
    ReadIndex<Odd,HaloDepth> indexO;
    iterateFunctorNoReturn<true,32>(CopyAll<PREC,Odd,2>(spinor_res12,spinorAll_res12.odd),indexO,_elems/2);


    for (int t=0; t<20; t++){
        CC[t] =  dslashinverse12.Correlator(t,spinor_res12);
    }

    for (int t=0; t<20; t++){
        std::cout << CC[t] << std::endl;
    }


///
    DWilsonEvenOdd<PREC, true, Even, 2, 2, 1> dslash_eo(gauge, mass, 1.0); 
    dslash_eo.calcFmunu();




    return 0;
}
