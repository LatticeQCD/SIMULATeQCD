/*
 * Added by Rasmus Larsen 8-7-2024
 *
 */

#pragma once

#include "../modules/inverter/inverter.h"
#include "../simulateqcd.h"
#include "fullSpinor.h"
//#include "fullSpinorfield.h"
//#include "gammaMatrix.h"
#include "matrix6x6Hermitian.h"

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks = 12>
class DWilson : public LinearOperator<Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, 12, NStacks> > {

    using SpinorRHS_t = Spinorfield<floatT, true, All, HaloDepthSpin, 12, NStacks>;

    template<CompressionType comp>
    using Gauge_t = Gaugefield<floatT, onDevice, HaloDepthGauge, comp>;

    Gauge_t<R18> &_gauge;

    double _mass;
    double _csw;

    Spinorfield<floatT, true, All, HaloDepthSpin, 12, NStacks> _tmpSpin;

public:
    DWilson(Gauge_t<R18> &gaugefield, const double mass, floatT csw = 0.0) :
            _gauge(gaugefield), _mass(mass),_csw(csw), _tmpSpin(_gauge.getComm()) {}

    //! not sure if need to include this
    virtual void applyMdaggM(SpinorRHS_t &spinorOut, const SpinorRHS_t &spinorIn, bool update = false) override;


};


template<typename floatT, bool onDevice, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
class DWilsonInverse {
private:
    // operators
    ConjugateGradient<floatT, NStacks> cg;
    DWilson<floatT, onDevice, All, HaloDepthGauge, HaloDepthSpin, NStacks> dslash;

    floatT _mass;
    floatT _csw;

    LatticeContainer<onDevice,COMPLEX(double)> _redBase;
    typedef GIndexer<All, HaloDepthGauge> GInd;

public:
    DWilsonInverse(Gaugefield<floatT, onDevice, HaloDepthGauge, R18> &gauge,
                      floatT mass, floatT csw = 0.0) :
                      dslash(gauge, mass, csw), _redBase(gauge.getComm()), _mass(mass), _csw(csw) {
        _redBase.adjustSize(GIndexer<All, HaloDepthGauge>::getLatData().vol3 * NStacks);
    }

    void DslashInverse(Spinorfield<floatT, onDevice,All, HaloDepthSpin, 12, NStacks> &spinorOut,
                     Spinorfield<floatT, onDevice,All, HaloDepthSpin, 12, NStacks> &spinorIn,
                int cgMax, double residue) {
        // compute the inverse 
        cg.invert(dslash, spinorOut, spinorIn, cgMax, residue); 

        spinorOut.updateAll();
    }

    void gamma5MultVec(Spinorfield<floatT, onDevice,All, HaloDepthSpin, 12, NStacks> &spinorOut,
                       Spinorfield<floatT, onDevice,All, HaloDepthSpin, 12, NStacks> &spinorIn );
    COMPLEX(double) Correlator(int t,const  Spinorfield<floatT, onDevice,All, HaloDepthSpin, 12, NStacks> &spinorIn);
    
};

template<class floatT, size_t HaloDepth, size_t HaloDepthSpin,size_t NStacks>
struct gamma5DiracWilson{

    //Gauge accessor to access the gauge field
    SU3Accessor<floatT> _SU3Accessor;
    SpinorColorAcc<floatT> _SpinorColorAccessor;
    floatT _csw;
    floatT _mass;

    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    gamma5DiracWilson(Gaugefield<floatT,true,HaloDepth,R18> &gauge,const Spinorfield<floatT, true,All, HaloDepthSpin, 12, NStacks> &spinorIn,floatT mass, floatT csw)
                : _SU3Accessor(gauge.getAccessor()),
                  _SpinorColorAccessor(spinorIn.getAccessor()),
                  _mass(mass), _csw(csw)
    { }

    //This is the operator that is called inside the Kernel
//    __device__ __host__ Vect12<floatT> operator()(gSite site) {
    __device__ __host__ Vect12<floatT> operator()(gSiteStack site) {
        SU3<floatT> link;

//        for (int nu = 1; nu < 4; nu++) {
//            for (int mu = 0; mu < nu; mu++) { 
//                link += _SU3Accessor.template getLinkPath<All, HaloDepth>(site, mu, nu, Back(mu), Back(nu));
//            }
//        }
//        link = csw*link;

        ColorVect<floatT> spinCol = _SpinorColorAccessor.getColorVect(site);
        ColorVect<floatT> outSC   = 2.0*_mass*spinCol;
        ColorVect<floatT> temp;

        // simple implementation for (1-gamma_mu)Umu(x)psi(x+mu) + (1+gamma_mu)Umu(x-mu)^dagger*psi(x-mu)
        // might be better to combine plus and minus direction before multiplying by gamma_mu
        // x forward direction 
        temp = _SU3Accessor.getLink(GInd::getSiteMu(site,0))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 0));
        outSC = outSC - temp + GammaXMultVec(temp);
        // x backwards direction 
        temp = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 0),0))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 0));
        outSC = outSC - temp - GammaXMultVec(temp);

        // y forward direction 
        temp = _SU3Accessor.getLink(GInd::getSiteMu(site,1))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 1));
        outSC = outSC - temp + GammaYMultVec(temp);
        // y backwards direction 
        temp = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 1),1))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 1));
        outSC = outSC - temp - GammaYMultVec(temp);

        // z forward direction 
        temp = _SU3Accessor.getLink(GInd::getSiteMu(site,2))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 2));
        outSC = outSC - temp + GammaZMultVec(temp);
        // z backwards direction 
        temp = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 2),2))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 2));
        outSC = outSC - temp - GammaZMultVec(temp);

        // t forward direction 
        temp = _SU3Accessor.getLink(GInd::getSiteMu(site,3))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 3));
        outSC = outSC - temp + GammaTMultVec(temp);
        // t backwards direction 
        temp = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 3),3))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 3));
        outSC = outSC - temp - GammaTMultVec(temp);


/*         
        outSC = outSC + _SpinorColorAccessor.getColorVect(GInd::site_up(site, 0));
        outSC = outSC + _SpinorColorAccessor.getColorVect(GInd::site_dn(site, 0));
        outSC = outSC + _SpinorColorAccessor.getColorVect(GInd::site_up(site, 1));
        outSC = outSC + _SpinorColorAccessor.getColorVect(GInd::site_dn(site, 1));
        outSC = outSC + _SpinorColorAccessor.getColorVect(GInd::site_up(site, 2));
        outSC = outSC + _SpinorColorAccessor.getColorVect(GInd::site_dn(site, 2));
        outSC = outSC + _SpinorColorAccessor.getColorVect(GInd::site_up(site, 3));
        outSC = outSC + _SpinorColorAccessor.getColorVect(GInd::site_dn(site, 3));
*/
        outSC = 0.5*outSC;

/*
        if(site.coord[0] == 0 && site.coord[1] == 0 && site.coord[2] == 0 && site.coord[3] == 0 ){
             printf("x=0 \n");
             for(int i=0; i < 4; i++){
                 printf("%f  %f %f  %f %i \n", real(outSC[i].data[0]), imag((outSC[i].data[0])),real(spinCol[i].data[0]), imag((spinCol[i].data[0])), site.stack );
                 printf("%f  %f %f  %f %i \n", real(outSC[i].data[1]), imag((outSC[i].data[1])),real(spinCol[i].data[1]), imag((spinCol[i].data[1])), site.stack );
                 printf("%f  %f %f  %f %i \n", real(outSC[i].data[2]), imag((outSC[i].data[2])),real(spinCol[i].data[2]), imag((spinCol[i].data[2])), site.stack );
             }
        }
*/
/*        if(site.coord[0] == 19 && site.coord[1] == 0 && site.coord[2] == 0 && site.coord[3] == 0 ){
             printf("x=19 \n");
             for(int i=0; i < 4; i++){
                 printf("%f  %f %f  %f \n", real(outSC[i].data[0]), imag((outSC[i].data[0])),real(spinCol[i].data[0]), imag((spinCol[i].data[0])) );
                 printf("%f  %f %f  %f \n", real(outSC[i].data[1]), imag((outSC[i].data[1])),real(spinCol[i].data[1]), imag((spinCol[i].data[1])) );
                 printf("%f  %f %f  %f \n", real(outSC[i].data[2]), imag((outSC[i].data[2])),real(spinCol[i].data[2]), imag((spinCol[i].data[2])) );
             }
        }
*/


        outSC = Gamma5MultVec(outSC);

        return convertColorVectToVect12(outSC);
    }
};

template<class floatT, size_t HaloDepth, size_t HaloDepthSpin,size_t NStacks>
struct gamma5DiracWilson2{

    //Gauge accessor to access the gauge field
    SU3Accessor<floatT> _SU3Accessor;
    SpinorColorAcc<floatT> _SpinorColorAccessor;
    floatT _csw;
    floatT _mass;

    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    gamma5DiracWilson2(Gaugefield<floatT,true,HaloDepth,R18> &gauge,const Spinorfield<floatT, true,All, HaloDepthSpin, 12, NStacks> &spinorIn,floatT mass, floatT csw)
                : _SU3Accessor(gauge.getAccessor()),
                  _SpinorColorAccessor(spinorIn.getAccessor()),
                  _mass(mass), _csw(csw)
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ Vect12<floatT> operator()(gSiteStack site) {

        ColorVect<floatT> outSC   = 2.0*_mass*_SpinorColorAccessor.getColorVect(site);
        ColorVect<floatT> temp, temp2;

        // simple implementation for (1-gamma_mu)Umu(x)psi(x+mu) + (1+gamma_mu)Umu(x-mu)^dagger*psi(x-mu)
        // might be better to combine plus and minus direction before multiplying by gamma_mu
        // x forward direction 
        temp = _SU3Accessor.getLink(GInd::getSiteMu(site,0))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 0));
        temp2 = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 0),0))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 0));
        outSC = outSC - temp-temp2 + GammaXMultVec(temp-temp2);

        // y forward direction 
        temp = _SU3Accessor.getLink(GInd::getSiteMu(site,1))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 1));
        temp2 = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 1),1))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 1));
        outSC = outSC - temp-temp2 + GammaYMultVec(temp-temp2);

        // z forward direction 
        temp = _SU3Accessor.getLink(GInd::getSiteMu(site,2))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 2));
        temp2 = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 2),2))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 2));
        outSC = outSC - temp-temp2 + GammaZMultVec(temp-temp2);

        // t forward direction 
        temp = _SU3Accessor.getLink(GInd::getSiteMu(site,3))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 3));
        temp2 = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 3),3))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 3));
        outSC = outSC - temp-temp2 + GammaTMultVec(temp-temp2);

        outSC = 0.5*Gamma5MultVec(outSC);

        return convertColorVectToVect12(outSC);
    }
};


/*
template<class floatT, size_t HaloDepth, size_t HaloDepthSpin,size_t NStacks>
struct gamma5DiracWilsonStack{

    //Gauge accessor to access the gauge field
    SU3Accessor<floatT> _SU3Accessor;
    SpinorColorAcc<floatT> _spinorIn;
    Vect12ArrayAcc<floatT> _spinorIn2;
    floatT _csw;
    floatT _mass;

    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    gamma5DiracWilsonStack(Gaugefield<floatT,true,HaloDepth,R18> &gauge,const Spinorfield<floatT, true,All, HaloDepthSpin, 12, NStacks> &spinorIn,floatT mass, floatT csw)
                : _SU3Accessor(gauge.getAccessor()),
                  _spinorIn(spinorIn.getAccessor()),
                  _spinorIn2(spinorIn.getAccessor()),
                  _mass(mass), _csw(csw)
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ void operator()(gSite site) {

        SimpleArray<Vect12<floatT>, NStacks> Stmp((floatT)0.0);

        for (size_t stack = 0; stack < NStacks; stack++) {

            if(site.coord[0] == 0 && site.coord[1] == 0 && site.coord[2] == 0 && site.coord[3] == 0 ){
             printf("x=0 \n");
             ColorVect<floatT> tmp = _spinorIn.getColorVect(GInd::getSiteStack(site,stack));
             for(int i=0; i < 4; i++){
                 printf("%f  %f %lu %lu \n", real(tmp[i].data[0]), imag((tmp[i].data[0])), stack, GInd::getSiteStack(site,stack).isiteStack );
                 printf("%f  %f %lu \n", real(tmp[i].data[1]), imag((tmp[i].data[1])), stack);
                 printf("%f  %f %lu \n", real(tmp[i].data[2]), imag((tmp[i].data[2])), stack);
             }

             printf("vect12 \n");
             Vect12<floatT> tmp2 = _spinorIn2.getElement(GInd::getSiteStack(site,stack));
             for(int i=0; i < 12; i++){
                 printf("%f  %f %lu %lu \n", real(tmp2.data[i]), imag((tmp2.data[i])), stack, GInd::getSiteStack(site,stack).isiteStack );

             }
        }
        }

    }
};

*/

template<class floatT, size_t HaloDepth, size_t HaloDepthSpin,size_t NStacks>
struct gamma5DiracWilsonStack{

    //Gauge accessor to access the gauge field
    SU3Accessor<floatT> _SU3Accessor;
    SpinorColorAcc<floatT> _spinorIn;
    Vect12ArrayAcc<floatT> _spinorOut;
    floatT _csw;
    floatT _mass;

    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    gamma5DiracWilsonStack(Gaugefield<floatT,true,HaloDepth,R18> &gauge,
                                 Spinorfield<floatT, true,All, HaloDepthSpin, 12, NStacks> &spinorOut,
                           const Spinorfield<floatT, true,All, HaloDepthSpin, 12, NStacks> &spinorIn,
                           floatT mass, floatT csw)
                : _SU3Accessor(gauge.getAccessor()),
                  _spinorIn(spinorIn.getAccessor()),
                  _spinorOut(spinorOut.getAccessor()),
                  _mass(mass), _csw(csw)
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ void operator()(gSite site) {

    SimpleArray<ColorVect<floatT>, NStacks> temp;
    SimpleArray<ColorVect<floatT>, NStacks> outSC;

    #pragma unroll NStacks
    for (size_t stack = 0; stack < NStacks; stack++) {
        outSC[stack]   = 2.0*_mass*_spinorIn.getColorVect(GInd::getSiteStack(site,stack));

        // simple implementation for (1-gamma_mu)Umu(x)psi(x+mu) + (1+gamma_mu)Umu(x-mu)^dagger*psi(x-mu)
        // might be better to combine plus and minus direction before multiplying by gamma_mu
        // x forward direction 
        temp[stack] = _SU3Accessor.getLink(GInd::getSiteMu(site,0))*_spinorIn.getColorVect(GInd::site_up(GInd::getSiteStack(site,stack), 0));
        outSC[stack] = outSC[stack] - temp[stack] + GammaXMultVec(temp[stack]);
        // x backwards direction 
        temp[stack] = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 0),0))*_spinorIn.getColorVect(GInd::site_dn(GInd::getSiteStack(site,stack), 0));
        outSC[stack] = outSC[stack] - temp[stack] - GammaXMultVec(temp[stack]);

        // y forward direction 
        temp[stack] = _SU3Accessor.getLink(GInd::getSiteMu(site,1))*_spinorIn.getColorVect(GInd::site_up(GInd::getSiteStack(site,stack), 1));
        outSC[stack] = outSC[stack] - temp[stack] + GammaYMultVec(temp[stack]);
        // y backwards direction 
        temp[stack] = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 1),1))*_spinorIn.getColorVect(GInd::site_dn(GInd::getSiteStack(site,stack), 1));
        outSC[stack] = outSC[stack] - temp[stack] - GammaYMultVec(temp[stack]);

        // z forward direction 
        temp[stack] = _SU3Accessor.getLink(GInd::getSiteMu(site,2))*_spinorIn.getColorVect(GInd::site_up(GInd::getSiteStack(site,stack), 2));
        outSC[stack] = outSC[stack] - temp[stack] + GammaZMultVec(temp[stack]);
        // z backwards direction 
        temp[stack] = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 2),2))*_spinorIn.getColorVect(GInd::site_dn(GInd::getSiteStack(site,stack), 2));
        outSC[stack] = outSC[stack] - temp[stack] - GammaZMultVec(temp[stack]);

        // t forward direction 
        temp[stack] = _SU3Accessor.getLink(GInd::getSiteMu(site,3))*_spinorIn.getColorVect(GInd::site_up(GInd::getSiteStack(site,stack), 3));
        outSC[stack] = outSC[stack] - temp[stack] + GammaTMultVec(temp[stack]);
        // t backwards direction 
        temp[stack] = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 3),3))*_spinorIn.getColorVect(GInd::site_dn(GInd::getSiteStack(site,stack), 3));
        outSC[stack] = outSC[stack] - temp[stack] - GammaTMultVec(temp[stack]);

        outSC[stack] = 0.5*outSC[stack];
        outSC[stack] = Gamma5MultVec(outSC[stack]);

    }

    #pragma unroll NStacks
    for (size_t stack = 0; stack < NStacks; stack++) {
        const gSiteStack writeSite = GInd::getSiteStack(site,stack);
        _spinorOut.setElement(writeSite,convertColorVectToVect12(outSC[stack]));
    }


    }
};


template<class floatT, size_t HaloDepth, size_t HaloDepthSpin,size_t NStacks>
struct gamma5DiracWilsonStack2{

    //Gauge accessor to access the gauge field
    SU3Accessor<floatT> _SU3Accessor;
    SpinorColorAcc<floatT> _spinorIn;
    Vect12ArrayAcc<floatT> _spinorOut;
    floatT _csw;
    floatT _mass;

    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    gamma5DiracWilsonStack2(Gaugefield<floatT,true,HaloDepth,R18> &gauge,
                                 Spinorfield<floatT, true,All, HaloDepthSpin, 12, NStacks> &spinorOut,
                           const Spinorfield<floatT, true,All, HaloDepthSpin, 12, NStacks> &spinorIn,
                           floatT mass, floatT csw)
                : _SU3Accessor(gauge.getAccessor()),
                  _spinorIn(spinorIn.getAccessor()),
                  _spinorOut(spinorOut.getAccessor()),
                  _mass(mass), _csw(csw)
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ void operator()(gSite site) {

    ColorVect<floatT> temp;
    SimpleArray<ColorVect<floatT>, NStacks> outSC;
    SU3<floatT> Utmp;
    SimpleArray<gSiteStack,NStacks> siteV;

    #pragma unroll NStacks
    for (size_t stack = 0; stack < NStacks; stack++) {
        siteV[stack] = GInd::getSiteStack(site,stack);
        outSC[stack]   = 2.0*_mass*_spinorIn.getColorVect(siteV[stack]);
    }
        // simple implementation for (1-gamma_mu)Umu(x)psi(x+mu) + (1+gamma_mu)Umu(x-mu)^dagger*psi(x-mu)
        // might be better to combine plus and minus direction before multiplying by gamma_mu
        

    // x forward direction    
    Utmp = _SU3Accessor.getLink(GInd::getSiteMu(site,0)); 
    #pragma unroll NStacks
    for (size_t stack = 0; stack < NStacks; stack++) {           
        temp = Utmp*_spinorIn.getColorVect(GInd::site_up(siteV[stack], 0));
        outSC[stack] = outSC[stack] - temp + GammaXMultVec(temp);
   }

    // x backwards direction
    Utmp = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 0),0));
    #pragma unroll NStacks
    for (size_t stack = 0; stack < NStacks; stack++) {
        temp = Utmp*_spinorIn.getColorVect(GInd::site_dn(siteV[stack], 0));
        outSC[stack] = outSC[stack] - temp - GammaXMultVec(temp);
    }

    // y forward direction
    Utmp = _SU3Accessor.getLink(GInd::getSiteMu(site,1)); 
    #pragma unroll NStacks
    for (size_t stack = 0; stack < NStacks; stack++) {
        temp = Utmp*_spinorIn.getColorVect(GInd::site_up(siteV[stack], 1));
        outSC[stack] = outSC[stack] - temp + GammaYMultVec(temp);
   }

    // y backwards direction
    Utmp = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 1),1));
    #pragma unroll NStacks
    for (size_t stack = 0; stack < NStacks; stack++) {
        temp = Utmp*_spinorIn.getColorVect(GInd::site_dn(siteV[stack], 1));
        outSC[stack] = outSC[stack] - temp - GammaYMultVec(temp);
   }
   
    // z forward direction 
    Utmp = _SU3Accessor.getLink(GInd::getSiteMu(site,2));
    #pragma unroll NStacks
    for (size_t stack = 0; stack < NStacks; stack++) {
        temp = Utmp*_spinorIn.getColorVect(GInd::site_up(siteV[stack], 2));
        outSC[stack] = outSC[stack] - temp + GammaZMultVec(temp);
     }

    // z backwards direction
    Utmp = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 2),2));
    #pragma unroll NStacks
    for (size_t stack = 0; stack < NStacks; stack++) {
        temp = Utmp*_spinorIn.getColorVect(GInd::site_dn(siteV[stack], 2));
        outSC[stack] = outSC[stack] - temp - GammaZMultVec(temp);
   }

    // t forward direction
    Utmp = _SU3Accessor.getLink(GInd::getSiteMu(site,3));
    #pragma unroll NStacks
    for (size_t stack = 0; stack < NStacks; stack++) {
        temp = Utmp*_spinorIn.getColorVect(GInd::site_up(siteV[stack], 3));
        outSC[stack] = outSC[stack] - temp + GammaTMultVec(temp);
    }

    // t backwards direction
    Utmp = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 3),3));
    #pragma unroll NStacks
    for (size_t stack = 0; stack < NStacks; stack++) {
        temp = Utmp*_spinorIn.getColorVect(GInd::site_dn(siteV[stack], 3));
        outSC[stack] = outSC[stack] - temp - GammaTMultVec(temp);
    }

    #pragma unroll NStacks
    for (size_t stack = 0; stack < NStacks; stack++) {
        outSC[stack] = 0.5*outSC[stack];
        outSC[stack] = Gamma5MultVec(outSC[stack]);

    }

    #pragma unroll NStacks
    for (size_t stack = 0; stack < NStacks; stack++) {
     //   const gSiteStack writeSite = GInd::getSiteStack(site,stack);
        _spinorOut.setElement(siteV[stack],convertColorVectToVect12(outSC[stack]));
    }


    }
};

template<class floatT, size_t HaloDepth, size_t HaloDepthSpin,size_t NStacks>
struct gamma5DiracWilsonStack3{

    //Gauge accessor to access the gauge field
    SU3Accessor<floatT> _SU3Accessor;
    SpinorColorAcc<floatT> _spinorIn;
    Vect12ArrayAcc<floatT> _spinorOut;
    floatT _csw;
    floatT _mass;

    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    gamma5DiracWilsonStack3(Gaugefield<floatT,true,HaloDepth,R18> &gauge,
                                 Spinorfield<floatT, true,All, HaloDepthSpin, 12, NStacks> &spinorOut,
                           const Spinorfield<floatT, true,All, HaloDepthSpin, 12, NStacks> &spinorIn,
                           floatT mass, floatT csw)
                : _SU3Accessor(gauge.getAccessor()),
                  _spinorIn(spinorIn.getAccessor()),
                  _spinorOut(spinorOut.getAccessor()),
                  _mass(mass), _csw(csw)
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ void operator()(gSite site) {

//    SimpleArray<ColorVect<floatT>, NStacks> temp, temp2;
    SimpleArray<ColorVect<floatT>, NStacks> outSC;

    #pragma unroll NStacks
    for (size_t stack = 0; stack < NStacks; stack++) {
        outSC[stack]   = 2.0*_mass*_spinorIn.getColorVect(GInd::getSiteStack(site,stack));
         ColorVect<floatT> temp, temp2;
        // simple implementation for (1-gamma_mu)Umu(x)psi(x+mu) + (1+gamma_mu)Umu(x-mu)^dagger*psi(x-mu)
        // might be better to combine plus and minus direction before multiplying by gamma_mu
        // x forward direction 
        temp = _SU3Accessor.getLink(GInd::getSiteMu(site,0))*_spinorIn.getColorVect(GInd::site_up(GInd::getSiteStack(site,stack), 0));
        temp2 = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 0),0))*_spinorIn.getColorVect(GInd::site_dn(GInd::getSiteStack(site,stack), 0));
        outSC[stack] = outSC[stack] - temp-temp2 + GammaXMultVec(temp-temp2);

        // y forward direction 
        temp = _SU3Accessor.getLink(GInd::getSiteMu(site,1))*_spinorIn.getColorVect(GInd::site_up(GInd::getSiteStack(site,stack), 1));
        temp2 = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 1),1))*_spinorIn.getColorVect(GInd::site_dn(GInd::getSiteStack(site,stack), 1));
        outSC[stack] = outSC[stack] - temp-temp2 + GammaYMultVec(temp-temp2);

        // z forward direction 
        temp = _SU3Accessor.getLink(GInd::getSiteMu(site,2))*_spinorIn.getColorVect(GInd::site_up(GInd::getSiteStack(site,stack), 2));
        temp2 = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 2),2))*_spinorIn.getColorVect(GInd::site_dn(GInd::getSiteStack(site,stack), 2));
        outSC[stack] = outSC[stack] - temp-temp2 + GammaZMultVec(temp-temp2);

        // t forward direction 
        temp = _SU3Accessor.getLink(GInd::getSiteMu(site,3))*_spinorIn.getColorVect(GInd::site_up(GInd::getSiteStack(site,stack), 3));
        temp2 = _SU3Accessor.getLinkDagger(GInd::getSiteMu(GInd::site_dn(site, 3),3))*_spinorIn.getColorVect(GInd::site_dn(GInd::getSiteStack(site,stack), 3));
        outSC[stack] = outSC[stack] - temp-temp2 + GammaTMultVec(temp-temp2);

        outSC[stack] = 0.5*outSC[stack];
        outSC[stack] = Gamma5MultVec(outSC[stack]);

    }

    #pragma unroll NStacks
    for (size_t stack = 0; stack < NStacks; stack++) {
        const gSiteStack writeSite = GInd::getSiteStack(site,stack);
        _spinorOut.setElement(writeSite,convertColorVectToVect12(outSC[stack]));
    }


    }
};


template<class floatT,Layout LatLayout, size_t HaloDepthSpin,size_t NStacks>
struct gamma5{

    //Gauge accessor to access the gauge field
    SpinorColorAcc<floatT> _SpinorColorAccessor;

    typedef GIndexer<LatLayout, HaloDepthSpin > GInd;
    //Constructor to initialize all necessary members.
    gamma5(const Spinorfield<floatT, true,LatLayout, HaloDepthSpin, 12, NStacks> &spinorIn)
                : _SpinorColorAccessor(spinorIn.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    //__device__ __host__ Vect12<floatT> operator()(gSite site) {
    __device__ __host__ Vect12<floatT> operator()(gSiteStack site) {

        ColorVect<floatT> outSC = _SpinorColorAccessor.getColorVect(site);
        outSC = Gamma5MultVec(outSC);

        return convertColorVectToVect12(outSC);
    }
};



template<class floatT, size_t HaloDepth,size_t NStacks>
struct Contract{
    using SpinorRHS_t = Spinorfield<floatT, true, All, 2, 12, NStacks>;


    //Gauge accessor to access the gauge field
    SpinorColorAcc<floatT> _SpinorColorAccessor;
    int _t;

    // adding spinor gives compile error
    typedef GIndexer<All, HaloDepth > GInd;
    Contract(int t,const SpinorRHS_t &spinorIn)
          :  _t(t), _SpinorColorAccessor(spinorIn.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ COMPLEX(double)  operator()(gSite site){

/*
          sitexyzt coords=site.coord;
          gSite siteT = GInd::getSite(coords.x,coords.y, coords.z, _t);

          COMPLEX(double) temp  =  _SpinorColorAccessor.template getElement<double>(siteT) *  _SpinorColorAccessor.template getElement<double>(siteT);

*/

    sitexyzt coords=site.coord;
    gSite siteT = GInd::getSite(coords.x,coords.y, coords.z, _t);

    COMPLEX(double) temp(0.0,0.0);
    for (size_t stack = 0; stack < NStacks; stack++) {
        temp  = temp + _SpinorColorAccessor.template getElement<double>(GInd::getSiteStack(siteT,stack)) *
                       _SpinorColorAccessor.template getElement<double>(GInd::getSiteStack(siteT,stack));
    }

        return temp;
    }
};



///////////////////////////////  below is the functions for shur complement inversion of wilson clover dirac operator

//dslash split up into odd and even parts
template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks = 12>
class DWilsonEvenOdd : public LinearOperator<Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, 12, NStacks> > {

    using SpinorRHS_t = Spinorfield<floatT, true, LatLayoutRHS, HaloDepthSpin, 12, NStacks>;

    template<CompressionType comp>
    using Gauge_t = Gaugefield<floatT, onDevice, HaloDepthGauge, comp>;


    Gauge_t<R18> &_gauge;

    double _mass;
    double _csw;

    Spinorfield<floatT, true, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, 12, NStacks> _tmpSpin;
    Spinorfield<floatT, true, LatLayoutRHS, HaloDepthSpin, 12, NStacks> _tmpSpinEven;

    // vector to holdfmunu and its inverted
    Spinorfield<floatT, true, All, HaloDepthSpin, 18, 1> FmunuUpper;
    Spinorfield<floatT, true, All, HaloDepthSpin, 18, 1> FmunuLower;
    Spinorfield<floatT, true, All, HaloDepthSpin, 18, 1> FmunuInvUpper;
    Spinorfield<floatT, true, All, HaloDepthSpin, 18, 1> FmunuInvLower;



public:
    DWilsonEvenOdd(Gauge_t<R18> &gaugefield, const double mass, floatT csw = 0.0) :
            _gauge(gaugefield), _mass(mass),_csw(csw),
             _tmpSpin(_gauge.getComm()), _tmpSpinEven(_gauge.getComm()),
             FmunuUpper(_gauge.getComm()),
             FmunuLower(_gauge.getComm()),
             FmunuInvUpper(_gauge.getComm()),
             FmunuInvLower(_gauge.getComm()) {}

    //overwrite function for use in CG (conjugate gradient)
    virtual void applyMdaggM(SpinorRHS_t &spinorOut, const SpinorRHS_t &spinorIn, bool update = false) override;

    // calc fmunu sigmamunu and store it in 2 vectors
    void calcFmunu();

    // the part of dsalsh that takes even to even or odd to odd
    void dslashDiagonalOdd(Spinorfield<floatT, true, Odd, HaloDepthSpin, 12, NStacks> &spinorOut,const Spinorfield<floatT, true, Odd, HaloDepthSpin, 12, NStacks>  &spinorIn, bool inverse);
    void dslashDiagonalEven(Spinorfield<floatT, true, Even, HaloDepthSpin, 12, NStacks> &spinorOut,const Spinorfield<floatT, true, Even, HaloDepthSpin, 12, NStacks>  &spinorIn, bool inverse);

};


template<class floatT,Layout  LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin,size_t NStacks>
struct DiracWilsonEvenEven{

    //input 
    Vect12ArrayAcc<floatT> _spinorIn;
    Vect18ArrayAcc<floatT> _spinorUpper;
    Vect18ArrayAcc<floatT> _spinorLower;

    typedef GIndexer<LatLayoutRHS, HaloDepthSpin > GInd;
    //Constructor to initialize all necessary members.
    DiracWilsonEvenEven(const Spinorfield<floatT, true,LatLayoutRHS, HaloDepthSpin, 12, NStacks> &spinorIn,
                        const Spinorfield<floatT, true,All, HaloDepthGauge, 18,1> &spinorUpper,
                        const Spinorfield<floatT, true,All, HaloDepthGauge, 18,1> &spinorLower)
                : _spinorIn(spinorIn.getAccessor()),
                  _spinorUpper(spinorUpper.getAccessor()),
                  _spinorLower(spinorLower.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ Vect12<floatT> operator()(gSiteStack site) {
          Vect18<floatT> tmp = _spinorUpper.getElement(GInd::template convertSite<All, HaloDepthGauge>(site));
          
          Matrix6x6<floatT> upper(tmp);
          Vect12<floatT> out;
          // apply the precalculated fmunu sigmamunu on uppper 2 spin and 3 colors          
          out = upper.MatrixXVect12UpDown(_spinorIn.getElement(site),0);
          
          tmp = _spinorLower.getElement(GInd::template convertSite<All, HaloDepthGauge>(site));
          Matrix6x6<floatT> lower(tmp);
          // apply lower part
          out = lower.MatrixXVect12UpDown(out,1);   
          return out;
    }
};

//alternative version
template<class floatT,Layout  LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin,size_t NStacks>
struct DiracWilsonEvenEven2{

    //input
    Vect12ArrayAcc<floatT> _spinorIn;
    Vect18ArrayAcc<floatT> _spinorUpper;
    Vect18ArrayAcc<floatT> _spinorLower;

    typedef GIndexer<LatLayoutRHS, HaloDepthSpin > GInd;
    //Constructor to initialize all necessary members.
    DiracWilsonEvenEven2(const Spinorfield<floatT, true,LatLayoutRHS, HaloDepthSpin, 12, NStacks> &spinorIn,
                        const Spinorfield<floatT, true,All, HaloDepthGauge, 18,1> &spinorUpper,
                        const Spinorfield<floatT, true,All, HaloDepthGauge, 18,1> &spinorLower)
                : _spinorIn(spinorIn.getAccessor()),
                  _spinorUpper(spinorUpper.getAccessor()),
                  _spinorLower(spinorLower.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ Vect12<floatT> operator()(gSiteStack site) {
          Vect18<floatT> tmp = _spinorUpper.getElement(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteStack(site,0)));
          Matrix6x6<floatT> upper(tmp);
          Vect12<floatT> out;

          out = upper.MatrixXVect12UpDown(_spinorIn.getElement(site),0);

          tmp = _spinorLower.getElement(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteStack(site,0)));
          Matrix6x6<floatT> lower(tmp);
          out = lower.MatrixXVect12UpDown(out,1);
          
          return out;
    }
};

// print information
template<class floatT,Layout  LatLayoutRHS, size_t HaloDepthSpin,size_t NStacks>
struct Print{

    //Gauge accessor to access the gauge field
    Vect12ArrayAcc<floatT> _spinorIn;

    typedef GIndexer<LatLayoutRHS, HaloDepthSpin > GInd;
    //Constructor to initialize all necessary members.
    Print(Spinorfield<floatT, true,LatLayoutRHS, HaloDepthSpin, 12, NStacks> &spinorIn)
                : _spinorIn(spinorIn.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ Vect12<floatT> operator()(gSiteStack site) {
          Vect12<floatT> out = _spinorIn.getElement(site);

            sitexyzt coord = GIndexer<All, HaloDepthSpin>::getLatData().globalPos(site.coord);
            if(coord[0] == 0 && coord[1] == 0 && coord[2] == 0 && coord[3] == 0 ){
        //if(site.coord[0] == 0 && site.coord[1] == 0 && site.coord[2] == 0 && site.coord[3] == 0 ){
             printf("x=0 print \n");
             for(int i=0; i < 12; i++){
                 printf("%f  %f %u %lu \n", real(out.data[i]), imag((out.data[i])),i,  site.isiteStack );
             }
            Vect12<floatT> tmp((floatT)0.0);
            tmp.data[site.stack] = 1.0;
            out = tmp;

      }

          return out;
    }
};

// the part of dslash that takes odd to even or even to odd
template<class floatT,Layout  LatLayoutLHS,Layout  LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin,size_t NStacks,bool g5 = true>
struct DiracWilsonEvenOdd{

    //declare input variables
    SU3Accessor<floatT> _SU3Accessor;
    SpinorColorAcc<floatT> _SpinorColorAccessor;
    floatT _csw;
    floatT _mass;

    typedef GIndexer<LatLayoutLHS, HaloDepthSpin > GInd;
    //Constructor to initialize all necessary members.
    DiracWilsonEvenOdd(Gaugefield<floatT,true,HaloDepthGauge,R18> &gauge,const Spinorfield<floatT, true,LatLayoutRHS, HaloDepthSpin, 12, NStacks> &spinorIn,floatT mass, floatT csw)
                : _SU3Accessor(gauge.getAccessor()),
                  _SpinorColorAccessor(spinorIn.getAccessor()),
                  _mass(mass), _csw(csw)
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ Vect12<floatT> operator()(gSiteStack site) {

   ColorVect<floatT> outSC;

    if(LatLayoutLHS == LatLayoutRHS){
        outSC   = 2.0*_mass*_SpinorColorAccessor.getColorVect(site);
    }
    else{
        outSC = 0.0*outSC;
        ColorVect<floatT> temp, temp2;

        // simple implementation for (1-gamma_mu)Umu(x)psi(x+mu) + (1+gamma_mu)Umu(x-mu)^dagger*psi(x-mu)
        // might be better to combine plus and minus direction before multiplying by gamma_mu
        // x forward direction 

        temp = _SU3Accessor.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site,0)))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 0));
        temp2 = _SU3Accessor.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, 0),0)))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 0));
        outSC = outSC - temp-temp2 + GammaXMultVec(temp-temp2);

        // y forward direction 
        temp = _SU3Accessor.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site,1)))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 1));
        temp2 = _SU3Accessor.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, 1),1)))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 1));
        outSC = outSC - temp-temp2 + GammaYMultVec(temp-temp2);

        // z forward direction 
        temp = _SU3Accessor.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site,2)))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 2));
        temp2 = _SU3Accessor.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, 2),2)))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 2));
        outSC = outSC - temp-temp2 + GammaZMultVec(temp-temp2);

        // t forward direction 
        temp = _SU3Accessor.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site,3)))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 3));
        temp2 = _SU3Accessor.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, 3),3)))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 3));
        outSC = outSC - temp-temp2 + GammaTMultVec(temp-temp2);
    }

        outSC = 0.5*outSC;
        
        if(g5){
             outSC = Gamma5MultVec(outSC);
        }

        return convertColorVectToVect12(outSC);
    }
};

// alternative version with only odd or even part
template<class floatT,Layout  LatLayoutLHS,Layout  LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin,size_t NStacks,bool g5 = true>
struct DiracWilsonEvenOdd2{

    //Gauge accessor to access the gauge field
    SU3Accessor<floatT> _SU3Accessor;
    SpinorColorAcc<floatT> _SpinorColorAccessor;
    floatT _csw;
    floatT _mass;

    typedef GIndexer<LatLayoutLHS, HaloDepthSpin > GInd;
    //Constructor to initialize all necessary members.
    DiracWilsonEvenOdd2(Gaugefield<floatT,true,HaloDepthGauge,R18> &gauge,const Spinorfield<floatT, true,LatLayoutRHS, HaloDepthSpin, 12, NStacks> &spinorIn,floatT mass, floatT csw)
                : _SU3Accessor(gauge.getAccessor()),
                  _SpinorColorAccessor(spinorIn.getAccessor()),
                  _mass(mass), _csw(csw)
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ Vect12<floatT> operator()(gSiteStack site) {

   ColorVect<floatT> outSC;

        outSC = 0.0*outSC;
        ColorVect<floatT> temp, temp2;

        // simple implementation for (1-gamma_mu)Umu(x)psi(x+mu) + (1+gamma_mu)Umu(x-mu)^dagger*psi(x-mu)
        // x forward direction 

        temp = _SU3Accessor.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site,0)))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 0));
        temp2 = _SU3Accessor.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, 0),0)))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 0));
        outSC = outSC - temp-temp2 + GammaXMultVec(temp-temp2);

        // y forward direction 
        temp = _SU3Accessor.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site,1)))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 1));
        temp2 = _SU3Accessor.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, 1),1)))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 1));
        outSC = outSC - temp-temp2 + GammaYMultVec(temp-temp2);

        // z forward direction 
        temp = _SU3Accessor.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site,2)))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 2));
        temp2 = _SU3Accessor.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, 2),2)))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 2));
        outSC = outSC - temp-temp2 + GammaZMultVec(temp-temp2);

        // t forward direction 
        temp = _SU3Accessor.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site,3)))*_SpinorColorAccessor.getColorVect(GInd::site_up(site, 3));
        temp2 = _SU3Accessor.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, 3),3)))*_SpinorColorAccessor.getColorVect(GInd::site_dn(site, 3));
        outSC = outSC - temp-temp2 + GammaTMultVec(temp-temp2);

        outSC = 0.5*outSC;
        // multiply by gamma 5 when needed
        if(g5){
             outSC = Gamma5MultVec(outSC);
        }

        return convertColorVectToVect12(outSC);
    }
};

// calculate sigmunu fmunu that splits into 2 block matrices of size 6X6 and save to vector 18 complex
template<class floatT,size_t HaloDepthGauge>
struct preCalcFmunu{

    //Gauge accessor to access the gauge field
    SU3Accessor<floatT> _SU3Accessor;
    floatT _csw;
    floatT _mass;
    FieldStrengthTensor<floatT,HaloDepthGauge,true,R18> FT;
    Vect18ArrayAcc<floatT> _spinorOutUpper;
    Vect18ArrayAcc<floatT> _spinorOutLower;
    Vect18ArrayAcc<floatT> _spinorOutInvUpper;
    Vect18ArrayAcc<floatT> _spinorOutInvLower;


    typedef GIndexer<All, HaloDepthGauge > GInd;
    //Constructor to initialize all necessary members.
    preCalcFmunu(Gaugefield<floatT,true,HaloDepthGauge,R18> &gauge,
                 Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> & spinorOutUpper,
                 Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> & spinorOutLower,
                 Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> & spinorOutInvUpper,
                 Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1> & spinorOutInvLower,
                 floatT mass, floatT csw)
                : _SU3Accessor(gauge.getAccessor()),
                  _spinorOutUpper(spinorOutUpper.getAccessor()),
                  _spinorOutLower(spinorOutLower.getAccessor()),
                  _spinorOutInvUpper(spinorOutInvUpper.getAccessor()),
                  _spinorOutInvLower(spinorOutInvLower.getAccessor()),
                  _mass(mass), _csw(csw),
                  FT(gauge.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ void operator()(gSite site) {

        Matrix6x6<floatT> M6x6; 
    

        SU3<floatT> Fmunu ; 
        COMPLEX(floatT)  ii(0.0,1.0);
        // sigma mu nu split into first of 2 blocks 
        Fmunu = FT(site,0,1);
        for(int i=0; i < 3; i++){
            for(int j=0; j < 3; j++){
                M6x6.val[i][j]     +=  Fmunu(i,j);
                M6x6.val[i+3][j+3] += -Fmunu(i,j);
            }
        }

        Fmunu = FT(site,0,2);
        for(int i=0; i < 3; i++){
            for(int j=0; j < 3; j++){
                M6x6.val[i][j+3] += -ii*Fmunu(i,j);
                M6x6.val[i+3][j] +=  ii*Fmunu(i,j);
            }
        }

        Fmunu = FT(site,0,3);
        for(int i=0; i < 3; i++){
            for(int j=0; j < 3; j++){
                M6x6.val[i][j+3] += -Fmunu(i,j);
                M6x6.val[i+3][j] += -Fmunu(i,j);
            }
        }

        Fmunu = FT(site,1,2);
        for(int i=0; i < 3; i++){
            for(int j=0; j < 3; j++){
                M6x6.val[i][j+3] +=  Fmunu(i,j);
                M6x6.val[i+3][j] +=  Fmunu(i,j);
            }
        }

        Fmunu = FT(site,1,3);
        for(int i=0; i < 3; i++){
            for(int j=0; j < 3; j++){
                M6x6.val[i][j+3] += -ii*Fmunu(i,j);
                M6x6.val[i+3][j] +=  ii*Fmunu(i,j);
            }
        }

        Fmunu = FT(site,2,3);
        for(int i=0; i < 3; i++){
            for(int j=0; j < 3; j++){
                M6x6.val[i][j]     += -Fmunu(i,j);
                M6x6.val[i+3][j+3] +=  Fmunu(i,j);
            }
        }
/*
        // write out fmunu
        if(site.coord[0] == 0 && site.coord[1] == 0 && site.coord[2] == 0 && site.coord[3] == 0 ){
             printf("x=0 F01 \n");
             Fmunu = FT(site,0,1);
             for(int i=0; i < 3; i++){
                   for(int j=0; j < 3; j++){
                          printf("(%f  +I*%f ), ", real(Fmunu(i,j)), imag(Fmunu(i,j) ) );
                    }
                  printf(" \n");

             }
        } 
*/

        ///add mass and csw parameters
        for(int i=0; i < 6; i++){
            for(int j=0; j < 6; j++){
                M6x6.val[i][j]    = -0.5*_csw*M6x6.val[i][j];
            }
        }

/*
        // test to see iff fmunu is corret
        if(site.coord[0] == 0 && site.coord[1] == 0 && site.coord[2] == 0 && site.coord[3] == 0 ){
             printf("x=0 sigma*Fmunu \n");
             // printf("{");
             for(int i=0; i < 6; i++){
                //    printf("{");
                   for(int j=0; j < 6; j++){
                          printf("(%f  +I*%f ), ", real(M6x6.val[i][j]), imag(M6x6.val[i][j] ) );
                    }
                  printf(" \n");

             }
        }
*/


        for(int i=0; i < 6; i++){
            M6x6.val[i][i] += _mass;
        }

        // invert
        Matrix6x6<floatT> M6x6inv = M6x6.invert();

        // save uppper hermitian matrix to vect18
        Vect18<floatT> tmp = M6x6.ConvertHermitianToVect18();
        Vect18<floatT> tmpInv = M6x6inv.ConvertHermitianToVect18();

        _spinorOutUpper.setElement(site,tmp);
        _spinorOutInvUpper.setElement(site,tmpInv); 

        ///////////// lower part

        for(int i=0; i < 6; i++){
            for(int j=0; j < 6; j++){
               M6x6.val[i][j]    = 0.0;
            }
        }


        // sigma mu nu split into lower of 2 blocks 
        Fmunu = FT(site,0,1);
        for(int i=0; i < 3; i++){
            for(int j=0; j < 3; j++){
                M6x6.val[i][j]     +=  Fmunu(i,j);
                M6x6.val[i+3][j+3] += -Fmunu(i,j);
            }
        }

        Fmunu = FT(site,0,2);
        for(int i=0; i < 3; i++){
            for(int j=0; j < 3; j++){
                M6x6.val[i][j+3] += -ii*Fmunu(i,j);
                M6x6.val[i+3][j] +=  ii*Fmunu(i,j);
            }
        }

        Fmunu = FT(site,0,3);
        for(int i=0; i < 3; i++){
            for(int j=0; j < 3; j++){
                M6x6.val[i][j+3] += Fmunu(i,j);
                M6x6.val[i+3][j] += Fmunu(i,j);
            }
        }

        Fmunu = FT(site,1,2);
        for(int i=0; i < 3; i++){
            for(int j=0; j < 3; j++){
                M6x6.val[i][j+3] +=  Fmunu(i,j);
                M6x6.val[i+3][j] +=  Fmunu(i,j);
            }
        }

        Fmunu = FT(site,1,3);
        for(int i=0; i < 3; i++){
            for(int j=0; j < 3; j++){
                M6x6.val[i][j+3] +=  ii*Fmunu(i,j);
                M6x6.val[i+3][j] += -ii*Fmunu(i,j);
            }
        }

        Fmunu = FT(site,2,3);
        for(int i=0; i < 3; i++){
            for(int j=0; j < 3; j++){
                M6x6.val[i][j]     +=  Fmunu(i,j);
                M6x6.val[i+3][j+3] += -Fmunu(i,j);
            }
        }
        ///add mass and csw parameters
        for(int i=0; i < 6; i++){
            for(int j=0; j < 6; j++){
                M6x6.val[i][j]    = -0.5*_csw*M6x6.val[i][j];
            }
        }
/*
        // test to see iff fmunu is corret
        if(site.coord[0] == 0 && site.coord[1] == 0 && site.coord[2] == 0 && site.coord[3] == 0 ){
             printf("x=0 sigma*Fmunu lower \n");
             // printf("{");
             for(int i=0; i < 6; i++){
                //    printf("{");
                   for(int j=0; j < 6; j++){
                          printf("(%f  +I*%f ), ", real(M6x6.val[i][j]), imag(M6x6.val[i][j] ) );
                    }
                  printf(" \n");

             }
        }
*/

        for(int i=0; i < 6; i++){
            M6x6.val[i][i] += _mass;
        }

        // invert matrix 
        M6x6inv = M6x6.invert();

        // save lower hermitian matrix to vect18
        tmp = M6x6.ConvertHermitianToVect18();
        tmpInv = M6x6inv.ConvertHermitianToVect18();

        _spinorOutLower.setElement(site,tmp);
        _spinorOutInvLower.setElement(site,tmpInv);



    }
};

template<class floatT, size_t HaloDepth,size_t NStacks>
struct SumXYZ_TrMdaggerM{
    using SpinorRHS_t = Spinorfield<floatT, true, All, HaloDepth, 12, NStacks>;


    SpinorColorAcc<floatT> _spinorIn;
    SpinorColorAcc<floatT> _spinorInDagger;
    int _t;

    // adding spinor gives compile error
    typedef GIndexer<All, HaloDepth > GInd;
    SumXYZ_TrMdaggerM(int t,const SpinorRHS_t &spinorInDagger, const SpinorRHS_t &spinorIn)
          :  _t(t), _spinorIn(spinorIn.getAccessor()), _spinorInDagger(spinorInDagger.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ COMPLEX(double)  operator()(gSite site){

        sitexyzt coords=site.coord;
        gSite siteT = GInd::getSite(coords.x,coords.y, coords.z, _t);

        COMPLEX(double) temp(0.0,0.0);
        for (size_t stack = 0; stack < NStacks; stack++) {
            temp  = temp + _spinorInDagger.template getElement<double>(GInd::getSiteStack(siteT,stack)) *
                                 _spinorIn.template getElement<double>(GInd::getSiteStack(siteT,stack));
        }

        return temp;
    }
};


// class that does the inversion
template<typename floatT, bool onDevice, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
class DWilsonInverseShurComplement {
private:
    // operators
    ConjugateGradient<floatT, NStacks> cg;
    DWilsonEvenOdd<floatT, onDevice, Even, HaloDepthGauge, HaloDepthSpin, NStacks> dslash;
    Gaugefield<floatT, onDevice, HaloDepthGauge, R18> & _gauge;

    floatT _mass;
    floatT _csw;

    LatticeContainer<onDevice,COMPLEX(double)> _redBase;
    typedef GIndexer<All, HaloDepthGauge> GInd;

public:
    DWilsonInverseShurComplement(Gaugefield<floatT, onDevice, HaloDepthGauge, R18> &gauge,
                      floatT mass, floatT csw = 0.0) :
                      dslash(gauge, mass, csw), _redBase(gauge.getComm()), _mass(mass), _csw(csw), _gauge(gauge) {
        _redBase.adjustSize(GIndexer<All, HaloDepthGauge>::getLatData().vol3 * NStacks);
        dslash.calcFmunu();
    }

    // version without the clover term
    void DslashInverseShurComplement(SpinorfieldAll<floatT, onDevice, HaloDepthSpin, 12, NStacks> &spinorOut,
                     SpinorfieldAll<floatT, onDevice, HaloDepthSpin, 12, NStacks> &spinorIn,
                int cgMax, double residue) {
        // compute the inverse 

        (spinorOut.even).template iterateOverBulk<32>(DiracWilsonEvenOdd<floatT,Even,Odd,HaloDepthGauge,HaloDepthSpin,NStacks,false>(_gauge, spinorIn.odd,_mass,_csw));
        spinorOut.even = spinorIn.even-(1.0/_mass)*spinorOut.even;
        spinorOut.odd  = spinorIn.odd;
        (spinorOut.even).template iterateOverBulk<32>(gamma5<floatT,Even,HaloDepthSpin,NStacks>(spinorOut.even));
        spinorOut.updateAll();
       

        cg.invert(dslash, spinorIn.even, spinorOut.even, cgMax, residue);
        spinorIn.odd  = (1.0/_mass)*spinorOut.odd;
        spinorIn.updateAll();
        
        spinorIn.even = spinorIn.even;
        (spinorOut.odd).template iterateOverBulk<32>(DiracWilsonEvenOdd<floatT,Odd,Even,HaloDepthGauge,HaloDepthSpin,NStacks,false>(_gauge, spinorIn.even,_mass,_csw));
        spinorOut.odd = spinorIn.odd-(1.0/_mass)*spinorOut.odd;
        spinorOut.even  = spinorIn.even;

        spinorOut.updateAll();
    }




/// version with clover term
    void DslashInverseShurComplementClover(SpinorfieldAll<floatT, onDevice, HaloDepthSpin, 12, NStacks> &spinorOut,
                     SpinorfieldAll<floatT, onDevice, HaloDepthSpin, 12, NStacks> &spinorIn,
                int cgMax, double residue) {
        SpinorfieldAll<floatT, onDevice, HaloDepthSpin, 12, NStacks> tmp(_gauge.getComm());

   //     spinorIn.even.template iterateOverBulk<32>(Print(spinorIn.even));


        // compute the inverse
        // the chur complement splits into 3 parts for odd even
        //(A, B)^-1 = (I , -A^-1 B)(A^-1,  0            )(I   ,  0     ) odd
        //(C, D)      (0 ,  I     )(0   ,(D-C(A^-1)B)^-1)(-C (A^-1) ,I ) even

        // first part from right
        //  A^-1(odd)
        dslash.dslashDiagonalOdd(spinorOut.odd,spinorIn.odd,true);
        spinorOut.odd.updateAll();
        // C (A^-1(odd))
        (spinorOut.even).template iterateOverBulk<32>(DiracWilsonEvenOdd<floatT,Even,Odd,HaloDepthGauge,HaloDepthSpin,NStacks,false>(_gauge, spinorOut.odd,_mass,_csw));
        // (-C (A^-1) ,I )
        spinorOut.even = spinorIn.even-spinorOut.even;
        //(I   ,  0     ) odd
        spinorOut.odd  = spinorIn.odd;

        //second matrix
        // use gamma5 hermiticity of (D-C(A^-1)B), so multiply by gamma5 for even part
        (spinorOut.even).template iterateOverBulk<32>(gamma5<floatT,Even,HaloDepthSpin,NStacks>(spinorOut.even));
        spinorOut.updateAll();
        //invert even part
        cg.invert(dslash, spinorIn.even, spinorOut.even, cgMax, residue);
        // (A^-1,  0 ) Odd
        dslash.dslashDiagonalOdd(spinorIn.odd,spinorOut.odd,true);
        spinorIn.updateAll();

        //third matrix
        // B even
        (spinorOut.odd).template iterateOverBulk<32>(DiracWilsonEvenOdd<floatT,Odd,Even,HaloDepthGauge,HaloDepthSpin,NStacks,false>(_gauge, spinorIn.even,_mass,_csw));
        // A^-1 (B even)
        dslash.dslashDiagonalOdd(spinorOut.odd,spinorOut.odd,true);
        // I even
        spinorOut.even  = spinorIn.even;        
        // (I , -A^-1 B)
        spinorOut.odd = spinorIn.odd-spinorOut.odd;


        spinorOut.updateAll();
      }

      // function to contract 2 vectors
      COMPLEX(double) sumXYZ_TrMdaggerM(int t,const  Spinorfield<floatT, onDevice,All, HaloDepthSpin, 12, 12> &spinorInDagger,
                                            const  Spinorfield<floatT, onDevice,All, HaloDepthSpin, 12, 12> &spinorIn);


};



