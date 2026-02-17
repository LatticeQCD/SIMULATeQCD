/*
 * inverter.cpp
 *
 */
#include "inverter.h"
#define BLOCKSIZE 64

template<class floatT, size_t NStacks>
template <typename Spinor_t>
void ConjugateGradient<floatT, NStacks>::invert(LinearOperator<Spinor_t>& dslash, Spinor_t& spinorOut,
        Spinor_t& spinorIn, int max_iter, double precision)
{
    Spinor_t vr(spinorIn.getComm());

    Spinor_t vp(spinorIn.getComm());

    Spinor_t vap(spinorIn.getComm());

    SimpleArray<COMPLEX(double), NStacks> dot(0);

    vr = spinorIn;
    vp = vr;
    spinorOut = static_cast<floatT>(0.0) * vr;

    SimpleArray<floatT, NStacks> alpha(0.0);

    dot = vr.dotProductStacked(vr);

    SimpleArray<floatT, NStacks> rsold = real<floatT>(dot);
    SimpleArray<floatT, NStacks> rsnew(0.0);
    SimpleArray<floatT, NStacks> remain;

    for (int i = 0; i < max_iter; i++) {
        vp.updateAll(COMM_BOTH | Hyperplane);
        dslash.applyMdaggM(vap, vp, false);

        dot = vp.dotProductStacked(vap);
        remain = real<floatT>(dot);

        alpha  = rsold / remain;

        // spinorOut = spinorOut + ( vp * alpha);
        spinorOut.template axpyThisB<BLOCKSIZE>(alpha, vp);

        // vr = vr - (vap * alpha);
        vr.template axpyThisB<BLOCKSIZE>(((floatT)(-1.0))*alpha, vap);

        dot = vr.dotProductStacked(vr);
        rsnew = real<floatT>(dot);

        if(max(rsnew /*/ norm*/) < precision) {
            rootLogger.info("# iterations " ,  i);
            break;
        }

        // vp = vr + vp * (rsnew / rsold);
        vp.template xpayThisB<SimpleArray<floatT, NStacks>,BLOCKSIZE>((rsnew/rsold), vr);
        rsold = rsnew;

        if(i == max_iter -1) {
            rootLogger.warn("CG: Warning max iteration reached " ,  i);
        }
    }
    spinorOut.updateAll();
    rootLogger.info("residue " ,  max(rsnew /*/norm*/));
}



template<class floatT, bool onDevice, Layout LatLayout, int HaloDepth, size_t NStacks>
struct StackTimesFloatPlusFloatTimesNoStack
{
    Vect3arrayAcc<floatT> spinorIn1;
    Vect3arrayAcc<floatT> spinorIn2;
    SimpleArray<floatT, NStacks> _a;
    SimpleArray<floatT, NStacks> _b;

    typedef GIndexer<LatLayout, HaloDepth> GInd;

    StackTimesFloatPlusFloatTimesNoStack(Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &spinorIn1,
            SimpleArray<floatT, NStacks> a,
            Spinorfield<floatT, onDevice, LatLayout, HaloDepth, 1> &spinorIn2,
            SimpleArray<floatT, NStacks> b) :
        spinorIn1(spinorIn1.getAccessor()), spinorIn2(spinorIn2.getAccessor()), _a(a), _b(b) {}


    __host__ __device__ Vect3<floatT> operator()(gSiteStack& siteStack){
        gSiteStack siteUnStack = GInd::getSiteStack(siteStack, 0);
        Vect3<floatT> my_vec;

        my_vec = spinorIn1.getElement(siteStack)*_a[siteStack.stack] + spinorIn2.getElement(siteUnStack)*_b[siteStack.stack];

        return my_vec;
    }
};

template<class floatT, bool onDevice, Layout LatLayout, int HaloDepth, size_t NStacks>
struct StackMinusFloatTimeStack
{
    Vect3arrayAcc<floatT> spinorIn1;
    Vect3arrayAcc<floatT> spinorIn2;
    SimpleArray<floatT, NStacks> _a;

    typedef GIndexer<LatLayout, HaloDepth> GInd;

    StackMinusFloatTimeStack(Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &spinorIn1,
            Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &spinorIn2,
            SimpleArray<floatT,NStacks> a) :
        spinorIn1(spinorIn1.getAccessor()), spinorIn2(spinorIn2.getAccessor()), _a(a) {}

    __host__ __device__ Vect3<floatT> operator()(gSiteStack& siteStack){
        Vect3<floatT> my_vec;

        my_vec = spinorIn1.getElement(siteStack) - spinorIn2.getElement(siteStack)*_a[siteStack.stack];

        return my_vec;
    }
};


template<class floatT, size_t NStacks>
template <typename SpinorIn_t, typename SpinorOut_t>
void AdvancedMultiShiftCG<floatT, NStacks>::invert(
        LinearOperator<SpinorIn_t>& dslash, SpinorOut_t& spinorOut, const SpinorIn_t& spinorIn,
        SimpleArray<floatT, NStacks> sigma, const int max_iter, const double precision)
{
    SpinorOut_t pi(spinorIn.getComm());
    SpinorIn_t s(spinorIn.getComm());
    SpinorIn_t r(spinorIn.getComm());
    SpinorIn_t pi0(spinorIn.getComm());

    int max_term = NStacks;
    StopWatch<true> timer;
    int cg = 0;

    SimpleArray<floatT, NStacks> a(0.0);
    SimpleArray<floatT, NStacks> B(1.0);
    SimpleArray<floatT, NStacks> Z(1.0);
    SimpleArray<floatT, NStacks> Zm1(1.0);

    r = spinorIn;

    double norm_r2 = r.realdotProduct(r);

    double  pAp,lambda, lambda2, rr_1, Bm1;
    // gMemoryPtr<true> pAp_ptr(MemoryManagement::getMemAt<true>("SHARED_pAp_ptr"));
    // gMemoryPtr<true> norm_r2_ptr(MemoryManagement::getMemAt<true>("SHARED_r2_ptr"));
    // pAp_ptr->template adjustSize<double>(1);
    // norm_r2_ptr->template adjustSize<double>(1);

    // r.realDotProductNoCopy(r, norm_r2_ptr);
    // gMemoryPtr<true> a_ptr(MemoryManagement::getMemAt<true>("SHARED_CGa_ptr"));
    // gMemoryPtr<true> B_ptr(MemoryManagement::getMemAt<true>("SHARED_CGB_ptr"));
    // gMemoryPtr<true> Z_ptr(MemoryManagement::getMemAt<true>("SHARED_CGZ_ptr"));
    // gMemoryPtr<true> Zm1_ptr(MemoryManagement::getMemAt<true>("SHARED_CGZm1_ptr"));
    // a_ptr->template adjustSize<floatT>(NStacks);

    // B_ptr->template adjustSize<floatT>(NStacks);
    // Z_ptr->template adjustSize<floatT>(NStacks);
    // Zm1_ptr->template adjustSize<floatT>(NStacks);
    Bm1 = 1.0;

    for (size_t i = 0; i < NStacks; i++) {
        pi.copyFromStackToStack(spinorIn, i ,0);
    }

    spinorOut.template iterateWithConst<BLOCKSIZE>(vect3_zero<floatT>());


    do {
        cg++;

        pi0.template copyFromStackToStackDevice<NStacks,0,0>(pi);
        pi0.updateAll(COMM_BOTH | Hyperplane);

        dslash.applyMdaggM(s, pi0, false);

        s = sigma[0] * pi0 - s;

        pAp = pi0.realdotProduct(s); //Optimization: do dot product but dont copy result to host

        B[0] = - norm_r2 / pAp; //use device-resident result to do this on the gpu

        r.axpyThisB(B[0], s); //fuse with this kernel call

        // r.fusedDotProdAndaxpy(pi0, s, pAp_ptr, norm_r2_ptr);

        

        for (int j=1; j<max_term; j++) {
            rr_1   = Bm1 * Zm1[j] / ( B[0] * a[0] * (Zm1[j] - Z[j])
                       + Zm1[j] * Bm1 * (1.0 - sigma[j] * B[0]) );
            Zm1[j] = Z[j];
            Z[j]   = Z[j] * rr_1;
            B[j]   = B[0] * rr_1;
        }
        Bm1 = B[0];
        lambda2 = r.realdotProduct(r); //Optimization: do dot product but dont copy result to host
        a[0]  = lambda2 / norm_r2;
        norm_r2 = lambda2;

        // r.realDotProductNoCopy(r, norm_r2_ptr);

        spinorOut.axpyThisLoop(((floatT)(-1.0))*B, pi,max_term); // move this up
        //     spinorOut[i] = spinorOut[i] - B[i] * pi[i];


        //################################
        for (int j=1; j<max_term; j++) {
            a[j] = a[0] * Z[j] * B[j] / (Zm1[j] * B[0]); //move to gpu
        }
        //################################


        pi.template axupbyThisLoop(Z, a, r, max_term); //fuse with dot product
        //     pi[i] = Z[i] * r + a[i] * pi[i];


        //################################

        do {
            lambda = Z[max_term-1] * Z[max_term-1] * lambda2;
            if ( lambda < precision/**old_norm*/ ) {
                max_term--;
            }
        } while ( max_term > 0 && (lambda < precision/**old_norm*/) );

    } while ( (max_term>0) && (cg<max_iter) );

    if(cg >= max_iter -1) {
        rootLogger.warn("CG: Warning max iteration reached " ,  cg);
    } else {
        rootLogger.info("CG: # iterations " ,  cg);
    }

    spinorOut.updateAll();
}


template<class floatT, size_t NStacks>
template <typename Spinor_t>
void ConjugateGradient<floatT, NStacks>::invert_new(
        LinearOperator<Spinor_t>& dslash, Spinor_t& spinorOut, const Spinor_t& spinorIn,
        const int max_iter, const double precision)
{
    Spinor_t spinorSearch(spinorIn.getComm());
    Spinor_t spinorMdMx(spinorIn.getComm());
    Spinor_t spinorResidual(spinorIn.getComm());


    int cg = 0;

    SimpleArray<double, NStacks> betaScale(0.0);
    SimpleArray<double, NStacks> stepSize(1.0);
    SimpleArray<double, NStacks> betaAlt(0.0);
    SimpleArray<double, NStacks> beta(0.0);
    SimpleArray<double, NStacks> pAp(0.0);

    SimpleArray<COMPLEX(double), NStacks> dot(0.0);
    SimpleArray<COMPLEX(double), NStacks> dot2(0.0);
    SimpleArray<COMPLEX(double), NStacks> dot3(0.0);

    spinorResidual = spinorIn;


    dot3 = spinorResidual.dotProductStacked(spinorResidual);
    betaAlt = real<double>(dot3);

    SimpleArray<double, NStacks> betaStart(0.0);

    betaStart = betaAlt;

    spinorSearch = spinorIn;

    spinorOut.template iterateWithConst<BLOCKSIZE>(vect3_zero<floatT>());


    do {
        cg++;

        spinorSearch.updateAll(COMM_BOTH | Hyperplane);

        dslash.applyMdaggM(spinorMdMx, spinorSearch, false);

        dot = spinorSearch.dotProductStacked(spinorMdMx);

        pAp = real<double>(dot);

        stepSize = -1.0* betaAlt / pAp;

        spinorResidual.axpyThisLoopd(stepSize, spinorMdMx, NStacks);

        dot2 = spinorResidual.dotProductStacked(spinorResidual);

        beta = real<double>(dot2);
        betaScale = beta / betaAlt;
        betaAlt = beta;

        spinorOut.axpyThisLoopd(-1.0*stepSize, spinorSearch, NStacks);

        spinorSearch.template xpayThisBd<SimpleArray<double, NStacks>,BLOCKSIZE>(betaScale, spinorResidual);

    } while ( (max(beta/betaStart) > precision) && (cg<max_iter) );

    if(cg >= max_iter -1) {
        rootLogger.warn("CG: Warning max iteration reached " ,  cg);
    } else {
        rootLogger.info("CG: # iterations " ,  cg);
    }

    spinorOut.updateAll();
}


template<class floatT, size_t NStacks>
template <typename Spinor_t>
void ConjugateGradient<floatT, NStacks>::invert_deflation( 
        LinearOperator<Spinor_t>& dslash, Spinor_t& spinorStart, const Spinor_t& spinorRHS,
        const int max_iter, const double precision)
{
    Spinor_t spinorResidual(spinorRHS.getComm());
    Spinor_t spinorSearch(spinorRHS.getComm());
    Spinor_t spinorMdMx(spinorRHS.getComm());
    
    int cg = 0;

    SimpleArray<double, NStacks> stepSize(1.0);
    SimpleArray<double, NStacks> betaScale(0.0);
    SimpleArray<double, NStacks> betaStart(0.0);
    SimpleArray<double, NStacks> betaAlt(0.0);
    SimpleArray<double, NStacks> beta(0.0);
    SimpleArray<double, NStacks> alpha(0.0);

    SimpleArray<COMPLEX(double), NStacks> dot(0.0);

    spinorResidual = spinorRHS;
    spinorResidual.updateAll();

    spinorStart.updateAll(COMM_BOTH | Hyperplane);
    dslash.applyMdaggM(spinorMdMx, spinorStart, true);

    spinorResidual.axpyThisLoopd(-1.0 * stepSize, spinorMdMx, NStacks);
    
    dot = spinorResidual.dotProductStacked(spinorResidual);
    betaStart = real<double>(dot);

    betaAlt = betaStart;

    spinorSearch = spinorResidual;

    do {
        cg++;

        spinorSearch.updateAll(COMM_BOTH | Hyperplane);

        dslash.applyMdaggM(spinorMdMx, spinorSearch, true);

        dot = spinorSearch.dotProductStacked(spinorMdMx);
        alpha = real<double>(dot);

        stepSize = betaAlt / alpha;

        spinorStart.axpyThisLoopd(stepSize, spinorSearch, NStacks);

        spinorResidual.axpyThisLoopd(-1.0 * stepSize, spinorMdMx, NStacks);

        dot = spinorResidual.dotProductStacked(spinorResidual);
        
        beta = real<double>(dot);
        betaScale = beta / betaAlt;
        betaAlt = beta;

        spinorSearch.template xpayThisBd<SimpleArray<double, NStacks>,BLOCKSIZE>(betaScale, spinorResidual);

    } while (( sqrt(max(beta/betaStart)) > precision) && (cg<max_iter) );

    if(cg >= max_iter -1) {
        rootLogger.warn("CG: Warning max iteration reached " ,  cg);
        rootLogger.info("residual=" ,  sqrt(max(beta/betaStart)));
    } else {
        rootLogger.info("CG: # iterations " ,  cg);
    }

    spinorStart.updateAll();
}


template<class floatT, size_t NStacks>
template <typename Spinor_t>
void ConjugateGradient<floatT, NStacks>::invert_res_replace(LinearOperator<Spinor_t>& dslash, Spinor_t& spinorOut, const Spinor_t& spinorIn, const int max_iter, const double precision, double delta)
{
    Spinor_t pi(spinorIn.getComm());
    Spinor_t s(spinorIn.getComm());
    Spinor_t r(spinorIn.getComm());
    Spinor_t accum(spinorIn.getComm());


    int cg = 0;

    SimpleArray<double, NStacks> beta(0.0);
    SimpleArray<double, NStacks> alpha(1.0);
    SimpleArray<double, NStacks> norm_r2(0.0);
    SimpleArray<double, NStacks> lambda2(0.0);
    SimpleArray<double, NStacks> pAp(0.0);
    SimpleArray<double, NStacks> pdotr(0.0);
    SimpleArray<COMPLEX(double), NStacks> dot(0.0);
    SimpleArray<COMPLEX(double), NStacks> dot2(0.0);

    SimpleArray<double, NStacks> norm_restart(0.0);
    SimpleArray<double, NStacks> norm_restart_prev(0.0);
    SimpleArray<double, NStacks> norm_input(0.0);
    SimpleArray<double, NStacks> norm_comp(0.0);

    r = spinorIn;


    pi = spinorIn;


    dot = r.dotProductStacked(r);
    norm_r2 = real<double>(dot);
    norm_input = norm_r2;
    lambda2 = norm_r2;
    norm_restart = norm_r2;
    norm_comp = norm_r2;
    spinorOut.template iterateWithConst<BLOCKSIZE>(vect3_zero<floatT>());
    accum = spinorOut;

    do {
        cg++;


        pi.updateAll(COMM_BOTH | Hyperplane);
        //pAp
        dslash.applyMdaggM(s,pi,false);

        dot = pi.dotProductStacked(s);
        pAp = real<double>(dot);
        alpha = -1.0 * norm_r2 / pAp;

        //r_k+1 = r_k - |r|^2/pAp * Ap_k+1
        r.axpyThisLoopd(alpha, s, NStacks);

        dot = r.dotProductStacked(r);
        lambda2 = real<double>(dot);
        beta = lambda2 / norm_r2;

        if (max(norm_comp) < max(lambda2)) {
            norm_comp = lambda2;
        }

        //x_k+1 = x_k + |r|^2/pAp * p_k+1
        accum.axpyThisLoopd(-1.0*alpha, pi, NStacks);
        norm_r2 = lambda2;
        if ((max(lambda2) < delta*max(norm_restart)) && (max(norm_restart) <= max(norm_comp))) {
            //reliable update

            //cumulative update of solution vector
            spinorOut += accum;

            //r = b - Ax
            r = spinorIn;
            SimpleArray<double, NStacks> tmp_arr(-1.0);
            spinorOut.updateAll();
            dslash.applyMdaggM(s,spinorOut, false);
            r.axpyThisLoopd(tmp_arr,s,NStacks);

            dot = r.dotProductStacked(r);
            lambda2 = real<double>(dot);
            norm_restart_prev = norm_restart;
            norm_restart = lambda2;

            //reset acc. solution vector
            accum.template iterateWithConst<BLOCKSIZE>(vect3_zero<floatT>());

            //reproject gradient vector so that pi and r are orthogonal
            dot2 = pi.dotProductStacked(r);
            pdotr = real<double>(dot2);

            SimpleArray<double,NStacks> proj(-1.0*pdotr/norm_restart);
            //pi = pi - <p,r>/|r|^2 * r
            pi.axpyThisLoopd(proj,r,NStacks);

            pi.template xpayThisBd<SimpleArray<double, NStacks>,BLOCKSIZE>(beta,r);
            norm_r2 = lambda2;
            norm_comp = lambda2;

        } else {
            //p_k+1 = r_k - beta*p_k
            pi.template xpayThisBd<SimpleArray<double, NStacks>,BLOCKSIZE>(beta,r);

        }
    } while ( (max(lambda2/norm_input) > precision) && (cg<max_iter) );

    if(cg >= max_iter -1) {
        rootLogger.warn("CG: Warning max iteration reached " ,  cg);
    } else {
        rootLogger.info("CG: # iterations " ,  cg);
    }

    spinorOut += accum;
    spinorOut.updateAll();


}


template<class floatT, size_t NStacks>
template<typename Spinor_t, typename Spinor_t_inner>
void ConjugateGradient<floatT, NStacks>::invert_mixed(LinearOperator<Spinor_t>& dslash, LinearOperator<Spinor_t_inner>& dslash_inner, Spinor_t& spinorOut, const Spinor_t& spinorIn,
                                                     const int max_iter, const double precision, double delta)
{
    Spinor_t pi(spinorIn.getComm());
    Spinor_t r(spinorIn.getComm());
    Spinor_t accum(spinorIn.getComm());

    Spinor_t_inner r_inner(spinorIn.getComm());
    Spinor_t_inner pi_inner(spinorIn.getComm());
    Spinor_t_inner s_inner(spinorIn.getComm());

    int cg = 0;

    SimpleArray<double, NStacks> beta(0.0);
    SimpleArray<double, NStacks> alpha(1.0);
    SimpleArray<double, NStacks> norm_r2(0.0);
    SimpleArray<double, NStacks> lambda2(0.0);
    SimpleArray<double, NStacks> pAp(0.0);
    SimpleArray<double, NStacks> pdotr(0.0);
    SimpleArray<COMPLEX(double), NStacks> dot(0.0);
    SimpleArray<COMPLEX(double), NStacks> dot2(0.0);
    SimpleArray<COMPLEX(double), NStacks> dot3(0.0);
    SimpleArray<double, NStacks> norm_restart(0.0);
    SimpleArray<double, NStacks> norm_restart_prev(0.0);
    SimpleArray<double, NStacks> norm_input(0.0);
    SimpleArray<double, NStacks> norm_comp(0.0);

    r = spinorIn;
    r_inner.convert_precision(r);
    pi = spinorIn;

    pi_inner.convert_precision(pi);
    int steps_since_restart = 0;
    dot = r.dotProductStacked(r);
    norm_r2 = real<double>(dot);
    norm_input = norm_r2;
    lambda2 = norm_r2;
    norm_restart = norm_r2;
    norm_comp = norm_r2;
    spinorOut.template iterateWithConst<BLOCKSIZE>(vect3_zero<floatT>());
    accum = spinorOut;

    do {
        cg++;



        pi_inner.updateAll(COMM_BOTH | Hyperplane);
        //pAp
        dslash_inner.applyMdaggM(s_inner,pi_inner,false);


        dot = pi_inner.dotProductStacked(s_inner);
        pAp = real<double>(dot);
        alpha = -1.0 * norm_r2 / pAp;

        //r_k+1 = r_k - |r|^2/pAp * Ap_k+1
        r_inner.axpyThisLoopd(alpha, s_inner, NStacks);

        dot = r_inner.dotProductStacked(r_inner);
        lambda2 = real<double>(dot);
        beta = lambda2 / norm_r2;



        if (max(norm_comp) < max(norm_r2)) {
            norm_comp = lambda2;
        }

        //x_k+1 = x_k + |r|^2/pAp * p_k+1
        accum.axpyThisLoopd(-1.0*alpha, pi, NStacks);
        norm_r2 = lambda2;
        if ((max(lambda2) < delta*max(norm_restart)) && (max(norm_restart) <= max(norm_comp))) {
            //reliable update

            //cumulative update of solution vector
            spinorOut += accum;

            //r = b - Ax
            r = spinorIn;
            SimpleArray<double, NStacks> tmp_arr(-1.0);

            //reuse accum to save dslash result.
            spinorOut.updateAll();
            dslash.applyMdaggM(accum,spinorOut, false);
            r.axpyThisLoopd(tmp_arr,accum,NStacks);
            r_inner.convert_precision(r);

            dot = r.dotProductStacked(r);
            lambda2 = real<double>(dot);
            norm_restart_prev = norm_restart;
            norm_restart = lambda2;

            //reset acc. solution vector
            accum.template iterateWithConst<BLOCKSIZE>(vect3_zero<floatT>());

            //reproject gradient vector so that pi and r are orthogonal
            dot2 = pi.dotProductStacked(r);
            pdotr = real<double>(dot2);

            SimpleArray<double,NStacks> proj(-1.0*pdotr/norm_restart);

            //pi = pi - <p,r>/|r|^2 * r
            pi.axpyThisLoopd(proj,r,NStacks);
            //beta = norm_restart / norm_r2;
            pi.template xpayThisBd<SimpleArray<double, NStacks>,BLOCKSIZE>(beta,r);
            pi_inner.convert_precision(pi);
            norm_r2 = lambda2;
            norm_comp = lambda2;
            steps_since_restart = 0;

        } else {
            //p_k+1 = r_k - a*p_k
            pi_inner.template xpayThisBd<SimpleArray<double, NStacks>, BLOCKSIZE>(beta,r_inner);
            pi.convert_precision(pi_inner);

            steps_since_restart++;
        }

    } while ( (max(lambda2/norm_input) > precision) && (cg<max_iter) );

    if(cg >= max_iter -1) {
        rootLogger.warn("CG: Warning max iteration reached " ,  cg);
    } else {
        rootLogger.info("CG: # iterations " ,  cg ,  " residual: " ,  max(lambda2/norm_input));
    }

    spinorOut += accum;
    spinorOut.updateAll();

}


template<class floatT, size_t NStacks>
template<bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin>
void ConjugateGradient<floatT, NStacks>::startVector(double mass, 
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorStart,
    const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorIn, 
    const Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> &eigenpair) {
    CommunicationBase &commBase = spinorIn.getComm();

    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> spinorSum(commBase);
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> spinorEv(commBase);

    double lambda;
    SimpleArray<double, NStacks> factorDouble(0.0);
    SimpleArray<COMPLEX(double), NStacks> dot(0.0);
    

    if constexpr (LatticeLayout == Layout::All) {
        // TODO
    }   else {
        for (int i = 0; i < eigenpair.spinor_count; i++) {
            spinorEv = eigenpair.spinor_vec[i];
            lambda = mass*mass + eigenpair.lambda_vec[i];

            dot =  spinorEv.dotProductStacked(spinorIn);
            factorDouble = real<double>(dot);

            factorDouble = factorDouble / lambda;

            spinorSum.axpyThisLoopd(factorDouble, spinorEv, NStacks);
        }
        spinorStart = spinorSum;
        spinorStart.updateAll();
    }
}


template<class floatT, size_t NStacks>
template<bool onDevice, Layout LatticeLayout, size_t HaloDepthGauge, size_t HaloDepthSpin>
void ConjugateGradient<floatT, NStacks>::startVectorTester(double mass, LinearOperator<Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>>& dslash, 
    const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorStart, 
    const Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks>& spinorRHS, 
    const Eigenpairs<floatT, onDevice, LatticeLayout, HaloDepthGauge, HaloDepthSpin, NStacks> &eigenpair) {
    CommunicationBase &commBase = spinorRHS.getComm();


    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> spinorEv(commBase);
    Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> spinorMdMx(commBase);
    Spinorfield<floatT, false, LatticeLayout, HaloDepthSpin, NStacks> spinorHost(commBase);


    double lambda;
    double massLambda;

    SimpleArray<double, NStacks> massLambdaArray(0.0);
    SimpleArray<COMPLEX(double), NStacks> dot(0.0);

    SimpleArray<COMPLEX(double), NStacks> dot1_vec(0.0);
    SimpleArray<COMPLEX(double), NStacks> dot2_vec(0.0);
    SimpleArray<COMPLEX(double), NStacks> dot3_vec(0.0);


    if constexpr (LatticeLayout == Layout::All) {
        // TODO
    }   else {
        int steps = 5;
        double stepSize = static_cast<double>(eigenpair.spinor_count - 1) / (steps - 1);

        for (int i = 0; i < steps; i++) {
            int index = static_cast<int>(i * stepSize);

            lambda = eigenpair.lambda_vec[index];
            rootLogger.info("startVectorTester: lambda         =", lambda);
        }

        for (int i = 0; i < steps; i++) {
            int index = static_cast<int>(i * stepSize);
            
            lambda = eigenpair.lambda_vec[index];
            massLambda = - mass*mass - lambda;
            rootLogger.info("startVectorTester: - (m^2+λ) =", massLambda);
        }

        for (int i = 0; i < steps; i++) {
            int index = static_cast<int>(i * stepSize);

            spinorEv = eigenpair.spinor_vec[index];
            dslash.applyMdaggM(spinorMdMx, spinorEv, true);
            
            lambda = eigenpair.lambda_vec[index];
            massLambdaArray = - mass*mass - lambda;
            
            spinorMdMx.axpyThisLoopd(massLambdaArray, spinorEv, NStacks);

            dot = spinorMdMx.dotProductStacked(spinorMdMx);

            for (size_t j = 0; j < NStacks; j++) {
                rootLogger.info("startVectorTester: norm((m^2+D†D)v - (m^2+λ)v)**2 =",  dot[j]);
            }
        }

        for (int i = 0; i<2; i++) {
        
            spinorHost = eigenpair.spinor_vec[i];
            spinorHost.updateAll();
            Vect3arrayAcc<floatT> spinorAcc = spinorHost.getAccessor();

            typedef GIndexer<All, HaloDepthSpin> GInd;
            typedef GIndexer<Even, HaloDepthSpin> GIndEven;
            typedef GIndexer<Odd, HaloDepthSpin> GIndOdd;

            LatticeDimensions Halo = LatticeDimensions(HaloDepthSpin, HaloDepthSpin, HaloDepthSpin, HaloDepthSpin);
            for (int x = -Halo[0]; x < (int) GInd::getLatData().lx + Halo[0]; x++) {
                int y, z, t;
                y = 0;
                z = 0;
                t = 0;
                bool par = (bool) ((abs(x) + abs(y) + abs(z) + abs(t)) % 2);
                bool even = (LatticeLayout == Even) && !par;
                bool odd = (LatticeLayout == Odd) && par;

                for (size_t stack = 0; stack < 1; stack++){

                    if (LatticeLayout == All || even || odd) {
                        LatticeDimensions localCoord = LatticeDimensions(x, y, z, t);

                        Vect3<floatT> tmpB;

                        if (LatticeLayout == All) {
                            gSiteStack site = GInd::getSiteStack(x, y, z, t, stack);
                            tmpB = spinorAcc.getElement(site);
                        } else if (LatticeLayout == Even) {
                            gSiteStack site = GIndEven::getSiteStack(x, y, z, t, stack);
                            tmpB = spinorAcc.getElement(site);
                        } else if (LatticeLayout == Odd) {
                            gSiteStack site = GIndOdd::getSiteStack(x, y, z, t, stack);
                            tmpB = spinorAcc.getElement(site);
                        }
                        int globalX = (x + commBase.MyRank() * GInd::getLatData().lx) % GInd::getLatData().globLX;
                        char buffer[256];
                        sprintf(buffer, "tester:Eigenspinor %d at Rank %d(x=%2d)(%d,%2d)", i, commBase.MyRank(), x, i, globalX);
                        std::cout << buffer << tmpB.getElement0() << tmpB.getElement1() << tmpB.getElement2() << std::endl;
                    }
                }
            }
        }
    
        Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> spinorRHSLocal(commBase);
        Spinorfield<floatT, onDevice, LatticeLayout, HaloDepthSpin, NStacks> spinorStartLocal(commBase);
        
        spinorRHSLocal = spinorRHS;
        spinorStartLocal = spinorStart;
        
        spinorRHSLocal.updateAll();
        spinorStartLocal.updateAll();

        dslash.applyMdaggM(spinorMdMx, spinorStartLocal, true);

        dot1_vec = spinorRHSLocal.dotProductStacked(spinorMdMx);
        for (size_t j = 0; j < NStacks; j++) {
            rootLogger.info("startVectorTester: b*Ax             =", dot1_vec[j]);
        }
        dot2_vec = spinorMdMx.dotProductStacked(spinorMdMx);
        for (size_t j = 0; j < NStacks; j++) {
            rootLogger.info("startVectorTester: Ax*Ax            =", dot2_vec[j]);
        }

    
        for (int i =0; i < eigenpair.spinor_count; i++) {
            spinorEv  = eigenpair.spinor_vec[i];
            dot1_vec = spinorRHSLocal.dotProductStacked(spinorEv);
            dot2_vec = spinorEv.dotProductStacked(spinorRHSLocal);
            for (size_t j = 0; j < NStacks; j++) {
                dot3_vec[j] += dot1_vec[j] * dot2_vec[j];
            }
        }
        for (size_t j = 0; j < NStacks; j++) {
            rootLogger.info("startVectorTester: sum(b µ_i*µ_i b) =", dot3_vec[j]);
        }
    }
}

#define CLASSCG_INIT(floatT,STACKS) \
template class ConjugateGradient<floatT, STACKS>;

#define CLASSCG_INV_INIT(floatT,LO,HALOSPIN,STACKS) \
template void ConjugateGradient<floatT, STACKS>::invert(LinearOperator<Spinorfield<floatT, true, LO, HALOSPIN, STACKS> >& dslash, \
            Spinorfield<floatT, true, LO, HALOSPIN, STACKS>& spinorOut, Spinorfield<floatT, true, LO, HALOSPIN, STACKS>& spinorIn, int, double);\
template void ConjugateGradient<floatT, STACKS>::invert_new(LinearOperator<Spinorfield<floatT, true, LO, HALOSPIN, STACKS> >& dslash, \
                                                            Spinorfield<floatT, true, LO, HALOSPIN, STACKS>& spinorOut,const Spinorfield<floatT, true, LO, HALOSPIN, STACKS>& spinorIn, const int, const double); \
template void ConjugateGradient<floatT, STACKS>::invert_res_replace(LinearOperator<Spinorfield<floatT, true, LO, HALOSPIN, STACKS> >& dslash, \
                                                                    Spinorfield<floatT, true, LO, HALOSPIN, STACKS>& spinorOut,const Spinorfield<floatT, true, LO, HALOSPIN, STACKS>& spinorIn, const int, const double, double); \
template void ConjugateGradient<floatT, STACKS>::invert_deflation(LinearOperator<Spinorfield<floatT, true, LO, HALOSPIN, STACKS> >& dslash, \
                                                            Spinorfield<floatT, true, LO, HALOSPIN, STACKS>& spinorOut,const Spinorfield<floatT, true, LO, HALOSPIN, STACKS>& spinorIn, const int, const double); \

#define CLASSCG_STARTVECTOR_INIT(floatT,LO,HALOGAUGE,HALOSPIN,STACKS) \
template void ConjugateGradient<floatT, STACKS>::startVector<true, LO, HALOGAUGE, HALOSPIN>(double, \
            Spinorfield<floatT, true, LO, HALOSPIN, STACKS>&, const Spinorfield<floatT, true, LO, HALOSPIN, STACKS>&, \
            const Eigenpairs<floatT, true, LO, HALOGAUGE, HALOSPIN, STACKS>&); \
template void ConjugateGradient<floatT, STACKS>::startVectorTester<true, LO, HALOGAUGE, HALOSPIN>(double, \
            LinearOperator<Spinorfield<floatT, true, LO, HALOSPIN, STACKS> >& dslash, \
            const Spinorfield<floatT, true, LO, HALOSPIN, STACKS>&, const Spinorfield<floatT, true, LO, HALOSPIN, STACKS>&, \
            const Eigenpairs<floatT, true, LO, HALOGAUGE, HALOSPIN, STACKS>&);

#define CLASSCG_FLOAT_INV_INIT(floatT,LO,HALOSPIN,STACKS) \
template void ConjugateGradient<floatT,STACKS>::invert_mixed(LinearOperator<Spinorfield<floatT, true, LO, HALOSPIN, STACKS> >& dslash, LinearOperator<Spinorfield<float, true, LO, HALOSPIN,STACKS> >& dslash_inner, Spinorfield<floatT, true, LO, HALOSPIN, STACKS>& spinorOut, const Spinorfield<floatT, true, LO, HALOSPIN,STACKS>& spinorIn, const int, const double, double);

#define CLASSCG_HALF_INV_INIT(floatT,LO,HALOSPIN,STACKS)  \
template void ConjugateGradient<floatT,STACKS>::invert_mixed(LinearOperator<Spinorfield<floatT, true, LO, HALOSPIN, STACKS> >& dslash, LinearOperator<Spinorfield<__half, true, LO, HALOSPIN,STACKS> >& dslash_inner, Spinorfield<floatT, true, LO, HALOSPIN, STACKS>& spinorOut, const Spinorfield<floatT, true, LO, HALOSPIN,STACKS>& spinorIn, const int, const double, double);

#define CLASSMCG_INIT(floatT,LO,HALOSPIN,STACKS)                    \
    template class MultiShiftCG<floatT,true ,LO ,HALOSPIN, STACKS>;
#define CLASSAMCG_INIT(floatT,STACKS) \
    template class AdvancedMultiShiftCG<floatT, STACKS>;
#define CLASSAMCG_INV_INIT(floatT,LO,HALOSPIN,STACKS) \
template void AdvancedMultiShiftCG<floatT, STACKS>::invert(LinearOperator<Spinorfield<floatT, true, LO, HALOSPIN, 1> >& dslash, \
            Spinorfield<floatT, true, LO, HALOSPIN, STACKS>& spinorOut,const Spinorfield<floatT, true, LO, HALOSPIN, 1>& spinorIn, \
            SimpleArray<floatT, STACKS> sigma, const int, const double); \

INIT_PN(CLASSCG_INIT)
INIT_PLHSN(CLASSCG_INV_INIT)
INIT_PLHHSN(CLASSCG_STARTVECTOR_INIT)
#if DOUBLEPREC == 1 && SINGLEPREC ==1
INIT_PLHSN(CLASSCG_FLOAT_INV_INIT)
#endif
#if HALFPREC == 1
INIT_PLHSN_HALF(CLASSCG_HALF_INV_INIT)
#endif
INIT_PLHSN(CLASSMCG_INIT)
INIT_PN(CLASSAMCG_INIT)
INIT_PLHSN(CLASSAMCG_INV_INIT)

