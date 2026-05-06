#pragma once

#include "../../gauge/gaugefield.h"
#include "../../spinor/spinorfield.h"
#include "../../base/math/simpleArray.h"
#include "../../modules/inverter/inverter.h"
#include "../../base/latticeContainer.h"
#include "utilities.h"
#include "wilsonDslash.h"



template<class floatT, bool onDevice, int HaloDepth, typename Spinor_t>
class BiCGStabInverter{
public:

    void invert(WilsonDslash<floatT, onDevice, HaloDepth, Spinor_t, Spinor_t>& dslash, Spinor_t& x, Spinor_t& rhs, int max_iter, double precision)
    {
        Spinor_t r0(x.getComm());
        Spinor_t p(x.getComm());
        Spinor_t r(x.getComm());
        
        //should go to member
        Spinor_t Ap(x.getComm());
        Spinor_t As(x.getComm());
        Spinor_t s(x.getComm());
        r0 = rhs;
        p=rhs;
        r=rhs;
        
        x*=0.0;
        Ap=x;
        As=x;
        s=x;

        rootLogger.info("invert start");
        floatT rhsinvnorm=1.0/ScalarProduct(rhs,rhs).cREAL;

        rootLogger.info("rhsinvnorm: ", rhsinvnorm);
        COMPLEX(floatT) rr0 = GPUcomplex<floatT>(ScalarProduct(r,r).cREAL,0.0);
        floatT resnorm=rr0.cREAL*rhsinvnorm;

        for (int i = 0; i < max_iter && resnorm > precision; i++) {
          rootLogger.info("Iteration ", i);
          
          dslash.apply(Ap, p); // Ap:output, p:input; Dslash p = Ap
          COMPLEX(floatT) beta=ScalarProduct(Ap,r0);
          rootLogger.info("beta ", beta.cREAL, beta.cIMAG);
          COMPLEX(floatT) alpha=rr0/beta;
          rootLogger.info("alpha ", alpha.cREAL, alpha.cIMAG);

          floatT eps = abs(beta)/sqrt(ScalarProduct(Ap,Ap).cREAL * ScalarProduct(r0,r0).cREAL);
          rootLogger.info("eps ", eps);
          if(eps < 1e-8) {
                rootLogger.trace("restarting BICGSTAB. eps = " ,  eps);
                // r = r0 = p = b-Ax
                dslash.apply(r0, x);
                r0=-1.0*r0+rhs;
                r=r0;
                p=r0;
                rr0=ScalarProduct(r,r).cREAL;
                resnorm = rr0.cREAL * rhsinvnorm;
                rootLogger.info("resnorm ", resnorm, i);
                continue;
            }

            //s = r-alpha*Ap
            s = r - alpha*Ap;
            const floatT snorm = ScalarProduct(s,s).cREAL;
            rootLogger.info("snorm ", snorm);
            if (snorm < precision * precision){
                x = x + alpha*p;
                r = s;
                resnorm = snorm * rhsinvnorm;
                continue;
            }
            dslash.apply(As, s);

            //omega = (As,s)/(As,As)
            COMPLEX(floatT) omega = ScalarProduct(As,s) / ScalarProduct(As,As).cREAL;

            //x=x+alpha*p+pmega*s
            x = x + (alpha * p) + (omega * s);

            //r = s - omega*As
            r = s - omega * As;
            
            resnorm = ScalarProduct(r,r).cREAL * rhsinvnorm;

            rootLogger.info("ScalarProduct(r,r).cREAL: ", ScalarProduct(r,r).cREAL);
            if (std::isnan(resnorm)){
              rootLogger.fatal("Nan");
            }

            if (resnorm > precision || i == max_iter-1 ){// the last steps are not needed if this is the last iteration
              //beta = alpha/omega * rr'/rr
              beta = 1.0/rr0; //reuse temporary
              rr0=ScalarProduct(r,r0);
              beta = beta * (alpha/omega) * rr0;
              
              p = r + beta*(p - omega*Ap);
            }

        }
        rootLogger.info("residue " ,  resnorm);
    }
};



template<class floatT, bool onDevice, int HaloDepth, typename Spinor_t>
class BiCGStabInverterEven{
public:

    void invert(WilsonDslashEven<floatT, onDevice, HaloDepth, Spinor_t, Spinor_t>& dslash, Spinor_t& x, Spinor_t& rhs, int max_iter, double precision)
    {
        Spinor_t r0(x.getComm());
        Spinor_t p(x.getComm());
        Spinor_t r(x.getComm());
        
        //should go to member
        Spinor_t Ap(x.getComm());
        Spinor_t As(x.getComm());
        Spinor_t s(x.getComm());
        r0 = rhs;
        p=rhs;
        r=rhs;
        
        x*=0.0;
        Ap=x;
        As=x;
        s=x;

        rootLogger.info("invert start");
        floatT rhsinvnorm=1.0/ScalarProductEven(rhs,rhs).cREAL;

        rootLogger.info("rhsinvnorm: ", rhsinvnorm);
        COMPLEX(floatT) rr0 = GPUcomplex<floatT>(ScalarProductEven(r,r).cREAL,0.0);
        floatT resnorm=rr0.cREAL*rhsinvnorm;

        for (int i = 0; i < max_iter && resnorm > precision; i++) {
          rootLogger.info("Iteration ", i);
          
          dslash.apply(Ap, p); // Ap:output, p:input; Dslash p = Ap
          COMPLEX(floatT) beta=ScalarProductEven(Ap,r0);
          rootLogger.info("beta ", beta.cREAL, beta.cIMAG);
          COMPLEX(floatT) alpha=rr0/beta;
          rootLogger.info("alpha ", alpha.cREAL, alpha.cIMAG);

          floatT eps = abs(beta)/sqrt(ScalarProductEven(Ap,Ap).cREAL * ScalarProductEven(r0,r0).cREAL);
          rootLogger.info("eps ", eps);
          if(eps < 1e-8) {
                rootLogger.trace("restarting BICGSTAB. eps = " ,  eps);
                // r = r0 = p = b-Ax
                dslash.apply(r0, x);
                r0=-1.0*r0+rhs;
                r=r0;
                p=r0;
                rr0=ScalarProductEven(r,r).cREAL;
                resnorm = rr0.cREAL * rhsinvnorm;
                rootLogger.info("resnorm ", resnorm, i);
                continue;
            }

            //s = r-alpha*Ap
            s = r - alpha*Ap;
            const floatT snorm = ScalarProductEven(s,s).cREAL;
            rootLogger.info("snorm ", snorm);
            if (snorm < precision * precision){
                x = x + alpha*p;
                r = s;
                resnorm = snorm * rhsinvnorm;
                continue;
            }
            dslash.apply(As, s);

            //omega = (As,s)/(As,As)
            COMPLEX(floatT) omega = ScalarProductEven(As,s) / ScalarProductEven(As,As).cREAL;

            //x=x+alpha*p+pmega*s
            x = x + (alpha * p) + (omega * s);

            //r = s - omega*As
            r = s - omega * As;
            
            resnorm = ScalarProductEven(r,r).cREAL * rhsinvnorm;

            rootLogger.info("ScalarProductEven(r,r).cREAL: ", ScalarProductEven(r,r).cREAL);
            if (std::isnan(resnorm)){
              rootLogger.fatal("Nan");
            }

            if (resnorm > precision || i == max_iter-1 ){// the last steps are not needed if this is the last iteration
              //beta = alpha/omega * rr'/rr
              beta = 1.0/rr0; //reuse temporary
              rr0=ScalarProductEven(r,r0);
              beta = beta * (alpha/omega) * rr0;
              
              p = r + beta*(p - omega*Ap);
            }

        }
        rootLogger.info("residue " ,  resnorm);
    }
};
