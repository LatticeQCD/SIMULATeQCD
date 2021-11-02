#include "inverter.h"
#define BLOCKSIZE 32

template<class floatT, size_t NStacks>
    template <typename Spinor_t>
void ConjugateGradient<floatT, NStacks>::invert(LinearOperator<Spinor_t>& dslash, Spinor_t& spinorOut,
        Spinor_t& spinorIn, int max_iter, double precision)
{
    Spinor_t vr(spinorIn.getComm());

    Spinor_t vp(spinorIn.getComm());

    Spinor_t vap(spinorIn.getComm());

    SimpleArray<GCOMPLEX(double), NStacks> dot(0);

    vr = spinorIn;
    vp = vr;
    spinorOut = static_cast<floatT>(0.0) * vr;

    SimpleArray<floatT, NStacks> alpha(0.0);

    dot = vr.dotProductStacked(vr);

    SimpleArray<floatT, NStacks> rsold = real<floatT>(dot);
    SimpleArray<floatT, NStacks> rsnew(0.0);
    SimpleArray<floatT, NStacks> remain;

    // SimpleArray<floatT, NStacks> norm = rsold;

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
            rootLogger.info() << "# iterations " << i;
            break;
        }

        // vp = vr + vp * (rsnew / rsold);
        vp.template xpayThisB<SimpleArray<floatT, NStacks>,BLOCKSIZE>((rsnew/rsold), vr);
        rsold = rsnew;

        if(i == max_iter -1) {
            rootLogger.warn() << "CG: Warning max iteration reached " << i;
        }
    }
    spinorOut.updateAll();
    rootLogger.info() << "residue " << max(rsnew /*/norm*/);
}



template<class floatT, bool onDevice, Layout LatLayout, int HaloDepth, size_t NStacks>
struct StackTimesFloatPlusFloatTimesNoStack
{
    gVect3arrayAcc<floatT> spinorIn1;
    gVect3arrayAcc<floatT> spinorIn2;
    SimpleArray<floatT, NStacks> _a;
    SimpleArray<floatT, NStacks> _b;

    typedef GIndexer<LatLayout, HaloDepth> GInd;

    StackTimesFloatPlusFloatTimesNoStack(Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &spinorIn1,
            SimpleArray<floatT, NStacks> a,
            Spinorfield<floatT, onDevice, LatLayout, HaloDepth, 1> &spinorIn2,
            SimpleArray<floatT, NStacks> b) :
        spinorIn1(spinorIn1.getAccessor()), spinorIn2(spinorIn2.getAccessor()), _a(a), _b(b) {}


    __host__ __device__ gVect3<floatT> operator()(gSiteStack& siteStack){
        gSiteStack siteUnStack = GInd::getSiteStack(siteStack, 0);
        gVect3<floatT> my_vec;

        my_vec = spinorIn1.getElement(siteStack)*_a[siteStack.stack] + spinorIn2.getElement(siteUnStack)*_b[siteStack.stack];

        return my_vec;
    }
};

template<class floatT, bool onDevice, Layout LatLayout, int HaloDepth, size_t NStacks>
struct StackMinusFloatTimeStack
{
    gVect3arrayAcc<floatT> spinorIn1;
    gVect3arrayAcc<floatT> spinorIn2;
    SimpleArray<floatT, NStacks> _a;

    typedef GIndexer<LatLayout, HaloDepth> GInd;

    StackMinusFloatTimeStack(Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &spinorIn1,
            Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> &spinorIn2,
            SimpleArray<floatT,NStacks> a) :
        spinorIn1(spinorIn1.getAccessor()), spinorIn2(spinorIn2.getAccessor()), _a(a) {}

    __host__ __device__ gVect3<floatT> operator()(gSiteStack& siteStack){
        gVect3<floatT> my_vec;

        my_vec = spinorIn1.getElement(siteStack) - spinorIn2.getElement(siteStack)*_a[siteStack.stack];

        return my_vec;
    }
};

//     template <typename floatT, bool onDevice, Layout LatLayout, int HaloDepth, size_t NStacks>
// void MultiShiftCG<floatT, onDevice, LatLayout, HaloDepth, NStacks>::invert(
//         LinearOperator<Spinorfield<floatT, onDevice, LatLayout, HaloDepth, 1>>& dslash,
//         Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks>& spinorOut,
//         Spinorfield<floatT, onDevice, LatLayout, HaloDepth, 1>& spinorIn,
//         SimpleArray<floatT, NStacks> sigma, int max_iter, double precision)
// {
//     Spinorfield<floatT, onDevice, LatLayout, HaloDepth, NStacks> p(spinorIn.getComm());
//     Spinorfield<floatT, onDevice, LatLayout, HaloDepth, 1> Ap(spinorIn.getComm());
//     Spinorfield<floatT, onDevice, LatLayout, HaloDepth, 1> r(spinorIn.getComm());
//     Spinorfield<floatT, onDevice, LatLayout, HaloDepth, 1> p0(spinorIn.getComm());

//     SimpleArray<floatT, NStacks> alpha(0.0);
//     SimpleArray<floatT, NStacks> beta(1.0);
//     SimpleArray<floatT, NStacks> zeta(1.0);
//     SimpleArray<floatT, NStacks> z(1.0);

//     r = spinorIn;

//     double rnormsqr = real<floatT>(r.dotProduct(r));;
//     const double in_normsqr = rnormsqr;

//     for (size_t i = 0; i < NStacks; i++) {
//         p.copyFromStackToStack(spinorIn, i ,0);
//     }

//     spinorOut.template iterateWithConst<BLOCKSIZE>(gvect3_zero<floatT>());

//     int cg = 0;

//     while ((cg < max_iter) && (rnormsqr/in_normsqr > precision)) {

//         p0.copyFromStackToStack(p, 0, 0);

//         dslash.applyMdaggM(Ap, p0);

//         Ap = sigma[0] * p0 + Ap;

//         const double lastbeta = beta[0];

//         beta[0] = - rnormsqr / real<floatT>(p0.dotProduct(Ap));
//         r = r + beta[0] * Ap;

//         for (size_t i = 1; i < NStacks; i++) {
//             z[i] = lastbeta/(beta[0]*alpha[0]*(1-z[i])+lastbeta*(1-(sigma[i]-sigma[0])*beta[0]));
//             beta[i] = beta[0]*z[i];
//         }

//         alpha[0] = 1/rnormsqr;
//         rnormsqr = real<floatT>(r.dotProduct(r));
//         alpha[0] *= rnormsqr;

//         for (size_t i = 0; i < NStacks; i++) {
//             alpha[i] = alpha[0]*z[i]*beta[i]/beta[0];
//             zeta[i] *= z[i];
//         }
//         spinorOut.template iterateOverBulk<BLOCKSIZE>( StackMinusFloatTimeStack<floatT, onDevice, LatLayout, HaloDepth, NStacks>
//                 (spinorOut, p, beta));
//         p.template iterateOverBulk<BLOCKSIZE>(StackTimesFloatPlusFloatTimesNoStack<floatT, onDevice, LatLayout, HaloDepth,
//                 NStacks>(p, alpha, r, zeta));

//         spinorOut.updateAll();
//         p.updateAll();

//         cg++;
//     }

//     if(cg >= max_iter -1){
//         rootLogger.warn() << "CG: Warning max iteration reached " << cg;
//     } else {
//         rootLogger.info() << "CG: # iterations " << cg;
//         rootLogger.info() << "residue " << rnormsqr/in_normsqr;
//     }
// }
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

    int cg = 0;

    SimpleArray<floatT, NStacks> a(0.0);
    SimpleArray<floatT, NStacks> B(1.0);
    SimpleArray<floatT, NStacks> Z(1.0);
    SimpleArray<floatT, NStacks> Zm1(1.0);

    r = spinorIn;

    double norm_r2 = r.realdotProduct(r);

    // rootLogger.info() << "Norm of input = " << norm_r2;

    double  pAp,lambda, lambda2, rr_1, Bm1;

    // floatT old_norm=norm_r2;

    Bm1 = 1.0;

    for (size_t i = 0; i < NStacks; i++) {
        pi.copyFromStackToStack(spinorIn, i ,0);
    }

    spinorOut.template iterateWithConst<BLOCKSIZE>(gvect3_zero<floatT>());


    do{
        cg++;

        // rootLogger.info() << "max_term = " <<max_term;

        pi0.copyFromStackToStack(pi, 0, 0);
        pi0.updateAll(COMM_BOTH | Hyperplane);
        // pi0.updateAll();
        // rootLogger.info() << "Norm2 of pi0 = " << pi0.realdotProduct(pi0);

        dslash.applyMdaggM(s, pi0, false);

        // rootLogger.info() << "Norm2 of s after DeoDoe = " << s.realdotProduct(s);


        s = sigma[0] * pi0 - s;

        // rootLogger.info() << "Norm2 of s after lin alg = " << s.realdotProduct(s);

        //s.axpyThisB(sigma[0], pi0);

        pAp = pi0.realdotProduct(s);

        B[0] = - norm_r2 / pAp;

        // r = r + B[0] * s;
        r.template axpyThisB<64>(B[0], s);


        for (int j=1; j<max_term; j++) {
            rr_1   = Bm1 * Zm1[j] / ( B[0] * a[0] * (Zm1[j] - Z[j]) +
                    Zm1[j] * Bm1 * (1.0 - sigma[j] * B[0]) );
            Zm1[j] = Z[j];
            Z[j]   = Z[j] * rr_1;
            B[j]   = B[0] * rr_1;
        }
        Bm1 = B[0];
        lambda2 = r.realdotProduct(r);
        a[0]  = lambda2 / norm_r2;
        norm_r2 = lambda2;

        // rootLogger.info() << "residue =" << lambda2;

        spinorOut.template axpyThisLoop<32>(((floatT)(-1.0))*B, pi,max_term); 
        //     spinorOut[i] = spinorOut[i] - B[i] * pi[i];
        

        //################################
        for (int j=1; j<max_term; j++) {
            a[j] = a[0] * Z[j] * B[j] / (Zm1[j] * B[0]);
        }
        //################################        


        pi.template axupbyThisLoop<32>(Z, a, r, max_term);
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
        rootLogger.warn() << "CG: Warning max iteration reached " << cg;
    } else {
        rootLogger.info() << "CG: # iterations " << cg;
    }

    spinorOut.updateAll();
}

template<class floatT, size_t NStacks>
template <class SpinorOut_t_inner, typename SpinorIn_t, typename SpinorOut_t, typename Spinor_t_inner>
void AdvancedMultiShiftCG<floatT, NStacks>::invert_mixed(LinearOperator<SpinorIn_t>& dslash, LinearOperator<Spinor_t_inner>&  dslash_inner, SpinorOut_t& spinorOut, const SpinorIn_t& spinorIn,SimpleArray<floatT, NStacks> sigma, const int max_iter, const double precision)
{
    
    SpinorOut_t pi(spinorIn.getComm());
    SpinorOut_t r_stacked(spinorIn.getComm());
    SpinorOut_t accum(spinorIn.getComm());

    SpinorOut_t_inner pi_inner(spinorIn.getComm());
    
    SpinorIn_t s(spinorIn.getComm());
    SpinorIn_t r(spinorIn.getComm());
    SpinorIn_t pi0(spinorIn.getComm());
    SpinorIn_t tmp(spinorIn.getComm());
   

    Spinor_t_inner pi0_inner(spinorIn.getComm());
    Spinor_t_inner s_inner(spinorIn.getComm());
    Spinor_t_inner r_inner(spinorIn.getComm());
    typedef typename Spinor_t_inner::floatT_inner floatT_inner;
    SimpleArray<floatT_inner, NStacks> sigma_inner(sigma);
    int max_term = NStacks;

    int cg = 0;
    //    bool do_inner_prec = false;
    SimpleArray<double, NStacks> a(0.0);
    SimpleArray<double, NStacks> B(1.0);
    SimpleArray<double, NStacks> Z(1.0);
    SimpleArray<double, NStacks> Zm1(1.0);

    
    r = spinorIn;
    r_inner.convert_precision(r);
    
    double norm_r2 = r.realdotProduct(r);
    // double norm_restart = norm_r2;

    SimpleArray<double, NStacks> norm_relupdate(norm_r2);
    SimpleArray<double, NStacks> norm_restart(norm_r2);
    SimpleArray<double, NStacks> norm_max(norm_r2);
    bool relupdate = false;
    //  double norm_restart_prev = norm_restart;
    //    double in_norm = norm_r2;
    double  pAp,lambda, lambda2, rr_1, Bm1;
    int relupdate_shift = 0;
    Bm1 = 1.0;

    for (size_t i = 0; i < NStacks; i++) {
        pi.copyFromStackToStack(spinorIn, i ,0);
    }
    pi_inner.convert_precision(pi);
    spinorOut.template iterateWithConst<BLOCKSIZE>(gvect3_zero<floatT>());
    accum = spinorOut;
    
    do{
        cg++;

        // rootLogger.info() << "max_term = " <<max_term;

        //    pi0.copyFromStackToStack(pi, 0, 0);
        pi0_inner.copyFromStackToStack(pi_inner, 0, 0);
        // rootLogger.info() << "Norm2 of pi0 = " << pi0.realdotProduct(pi0);
        
        //        pi0_inner.convert_precision(pi0);
        pi0_inner.updateAll(COMM_BOTH | Hyperplane);

        dslash_inner.applyMdaggM(s_inner, pi0_inner, false);
        //s.convert_precision(s_inner);
        
        // rootLogger.info() << "Norm2 of s after DeoDoe = " << s.realdotProduct(s);


        // s = sigma[0] * pi0 - s;
        s_inner = pi0_inner * sigma_inner[0] - s_inner;
        // s_inner.template axpyThisB<64>(-sigma_inner[0], pi0_inner);
        
        // rootLogger.info() << "Norm2 of s after lin alg = " << s.realdotProduct(s);

        //s.axpyThisB(sigma[0], pi0);

        pAp = pi0_inner.realdotProduct(s_inner);

        B[0] = - norm_r2 / pAp;
        floatT_inner B_temp(B[0]);
        // r = r + B[0] * s;
        
        //r_inner.template axpyThisBd<64>(B[0], s_inner);
        r_inner = B_temp * s_inner + r_inner;

        for (int j=1; j<max_term; j++) {
            rr_1   = Bm1 * Zm1[j] / ( B[0] * a[0] * (Zm1[j] - Z[j]) +
                    Zm1[j] * Bm1 * (1.0 - sigma[j] * B[0]) );
            Zm1[j] = Z[j];
            Z[j]   = Z[j] * rr_1;
            B[j]   = B[0] * rr_1;
        }
        Bm1 = B[0];
        lambda2 = r_inner.realdotProduct(r_inner);
        a[0]  = lambda2 / norm_r2;
        norm_r2 = lambda2;
        for (int j=0; j>=0; j--) {
            norm_relupdate[j] = norm_r2*Z[j]*Z[j];
            if (norm_relupdate[j] > norm_restart[j]) norm_max[j] = norm_relupdate[j];
            //  rootLogger.debug() << "lambda[" << j <<"] = " << norm_relupdate[j] << ". lambda_restart[" << j <<"] = " <<norm_restart[j];
            if ((norm_relupdate[j] < 1e-1*norm_restart[j]) && (norm_restart[j] <= norm_max[j])) {
                relupdate = true;
                relupdate_shift = j;
                //       rootLogger.debug() << "new lambda_restart " << norm_restart[j];
                //rootLogger.debug() << " ";
                break;
            }
            
        }
        
        // rootLogger.info() << "residue =" << lambda2;
        accum.template axpyThisLoopd<32>((-1.0)*B, pi, max_term); //TODO this also takes way too much time
        //        spinorOut.template axpyThisLoop<32>(((floatT)(-1.0))*B, pi,max_term); 
        //     spinorOut[i] = spinorOut[i] - B[i] * pi[i];
        
        if (relupdate) {

            spinorOut += accum;
            accum.template iterateWithConst<BLOCKSIZE>(gvect3_zero<floatT>());
            r = spinorIn;
            tmp.copyFromStackToStack(spinorOut,0,0);
            dslash.applyMdaggM(s,tmp,false);
            s = sigma[0] * tmp - s;
            r = r - s;
            lambda2 = r.realdotProduct(r);

            
            norm_restart = lambda2;
            //TODO this is terribly slow!! Need to fix!
            for (int i = 0; i < max_term; i++) {
                tmp = r * (floatT)(Z[i]*Z[i]);
                r_stacked.copyFromStackToStack(tmp,i,0);
            }

            r_inner.convert_precision(r);
            
            SimpleArray<double, NStacks> pdotr(0.0);
            pdotr = r_stacked.realdotProductStacked(pi);
            pdotr = -1.0 * pdotr / lambda2;

            pi.template axpyThisLoopd<32>(pdotr,r_stacked,max_term);

            //     a[0] = norm_restart/norm_restart_prev;
            //a[0] = a[0] / norm_r2;
            //            a[0]  = lambda2 / norm_r2;
            for (int j=1; j<max_term; j++) {
                a[j] = a[0] * Z[j] * B[j] / (Zm1[j] * B[0]);
            }
            
            pi.template axupbyThisLoopd<32>(Z, a, r, max_term);
            pi_inner.convert_precision(pi);
            //     pi[i] = Z[i] * r + a[i] * pi[i];
            norm_r2 = lambda2;

            //            rootLogger.debug() << "recompute precise lambda " << norm_r2*Z[relupdate_shift]*Z[relupdate_shift] << ". Shift " << relupdate_shift << " triggered the update.";
            relupdate = false;
            for (int j = 0; j < max_term; j++) {
                norm_restart[j] = norm_relupdate[j];
            }
        }
        else {
            for (int j=1; j<max_term; j++) {
                a[j] = a[0] * Z[j] * B[j] / (Zm1[j] * B[0]);
            }
            SimpleArray<floatT_inner,NStacks> Z_inner(Z);
            SimpleArray<floatT_inner,NStacks> a_inner(a);
            pi_inner.template axupbyThisLoop<32>(Z_inner, a_inner, r_inner, max_term);
            //            pi.template axupbyThisLoop<32>(Z, a, r, max_term);
            pi.convert_precision(pi_inner); //TODO need to find alternative... this takes as much time as dslash
        }
        
                
        /*        tmp.copyFromStackToStack(spinorOut,0,0);
        dslash.applyMdaggM(s,tmp,false);
        s = sigma[0] * tmp - s;
        tmp = spinorIn;
        tmp = tmp - s;
        double dot_exact = tmp.realdotProduct(tmp);
        tmp2 = r;
        double dot_acc = tmp2.realdotProduct(tmp2);
        rootLogger.info() << "accum. residual: "<< dot_acc;
        rootLogger.info() << "exact  residual: "<< dot_exact;
        */
            do {
                lambda = Z[max_term-1] * Z[max_term-1] * lambda2;
                //          rootLogger.info() << "lambda = " << lambda << " max_term " << max_term-1;
                if ( lambda < precision/**old_norm*/ ) {
                    max_term--;
                }
            } while ( max_term > 0 && (lambda < precision/**old_norm*/) );
        
    } while ( (max_term>0) && (cg<max_iter) );

    if(cg >= max_iter -1) {
        rootLogger.warn() << "CG: Warning max iteration reached " << cg;
    } else {
        rootLogger.info() << "CG: # iterations " << cg;
    }
    spinorOut += accum;
    spinorOut.updateAll();
}


template<class floatT, size_t NStacks>
template <typename Spinor_t>
void ConjugateGradient<floatT, NStacks>::invert_new(
        LinearOperator<Spinor_t>& dslash, Spinor_t& spinorOut, const Spinor_t& spinorIn,
        const int max_iter, const double precision)
{
    Spinor_t pi(spinorIn.getComm());
    Spinor_t s(spinorIn.getComm());
    Spinor_t r(spinorIn.getComm());


    int cg = 0;

    SimpleArray<double, NStacks> a(0.0);
    SimpleArray<double, NStacks> B(1.0);
    SimpleArray<double, NStacks> norm_r2(0.0);
    SimpleArray<double, NStacks> lambda2(0.0);
    SimpleArray<double, NStacks> pAp(0.0);

    SimpleArray<GCOMPLEX(double), NStacks> dot(0.0);
    SimpleArray<GCOMPLEX(double), NStacks> dot2(0.0);
    SimpleArray<GCOMPLEX(double), NStacks> dot3(0.0);

    r = spinorIn;


    dot3 = r.dotProductStacked(r);
    norm_r2 = real<double>(dot3);

    SimpleArray<double, NStacks> in_norm(0.0);

    in_norm = norm_r2;

    pi = spinorIn;

    spinorOut.template iterateWithConst<BLOCKSIZE>(gvect3_zero<floatT>());


    do{
        cg++;

        // rootLogger.info() << "max_term = " <<max_term;

        pi.updateAll(COMM_BOTH | Hyperplane);
        // pi0.updateAll();
        // rootLogger.info() << "Norm2 of pi0 = " << pi0.realdotProduct(pi0);

        dslash.applyMdaggM(s, pi, false);

        // rootLogger.info() << "Norm2 of s after DeoDoe = " << s.realdotProduct(s);


        // s = floatT(mass*mass) * pi - s;

        // rootLogger.info() << "Norm2 of s after lin alg = " << s.realdotProduct(s);

        //s.axpyThisB(sigma[0], pi0);

        dot = pi.dotProductStacked(s);

        pAp = real<double>(dot);

        B = -1.0* norm_r2 / pAp;

        // r[i] = r[i] + B[i] * s[i];
        r.template axpyThisLoopd<32>(B, s, NStacks); 
        // r.template axpyThisB<64>(B[0], s); //WRONG


        // for (int j=1; j<max_term; j++) {
        //     rr_1   = Bm1 * Zm1[j] / ( B[0] * a[0] * (Zm1[j] - Z[j]) +
        //             Zm1[j] * Bm1 * (1.0 - sigma[j] * B[0]) );
        //     Zm1[j] = Z[j];
        //     Z[j]   = Z[j] * rr_1;
        //     B[j]   = B[0] * rr_1;
        // }
        // Bm1 = B[0];
        dot2 = r.dotProductStacked(r);

        lambda2 = real<double>(dot2);
        a = lambda2 / norm_r2;
        norm_r2 = lambda2;

        // std::cout << cg << std::endl;

        // std::cout << cg << "    " << /*lambda2[0]/in_norm[0] << "    " << lambda2[1]/in_norm[1] << "    " << lambda2[2] << "    " << */ max(lambda2/in_norm) << std::endl;


        spinorOut.template axpyThisLoopd<32>(-1.0*B, pi,NStacks); 
        //     spinorOut[i] = spinorOut[i] - B[i] * pi[i];
        

        //################################
        // for (int j=1; j<max_term; j++) {
        //     a[j] = a[0] * Z[j] * B[j] / (Zm1[j] * B[0]);
        // }
        //################################        

        pi.template xpayThisBd<SimpleArray<double, NStacks>,BLOCKSIZE>(a, r);

        // pi = r + a * pi;
        //     pi[i] = Z[i] * r + a[i] * pi[i];


        //################################

    } while ( (max(lambda2/in_norm) > precision) && (cg<max_iter) );

    if(cg >= max_iter -1) {
        rootLogger.warn() << "CG: Warning max iteration reached " << cg;
    } else {
        rootLogger.info() << "CG: # iterations " << cg;
    }

    spinorOut.updateAll();
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
    SimpleArray<GCOMPLEX(double), NStacks> dot(0.0);
    SimpleArray<GCOMPLEX(double), NStacks> dot2(0.0);
    
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
    spinorOut.template iterateWithConst<BLOCKSIZE>(gvect3_zero<floatT>());
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
        r.template axpyThisLoopd<32>(alpha, s, NStacks);

        dot = r.dotProductStacked(r);
        lambda2 = real<double>(dot);
        beta = lambda2 / norm_r2;
      
        if (max(norm_comp) < max(lambda2)) {
            norm_comp = lambda2;
        }
        
        //x_k+1 = x_k + |r|^2/pAp * p_k+1
        accum.template axpyThisLoopd<32>(-1.0*alpha, pi, NStacks);
        norm_r2 = lambda2;
        if ((max(lambda2) < delta*max(norm_restart)) && (max(norm_restart) <= max(norm_comp))) {
            //reliable update

            //cumulative update of solution vector
            spinorOut += accum;

            //r = b - Ax
            r = spinorIn;
            SimpleArray<double, NStacks> tmp_arr(-1.0);
            dslash.applyMdaggM(s,spinorOut, false);
            r.template axpyThisLoopd<32>(tmp_arr,s,NStacks);

            dot = r.dotProductStacked(r);
            lambda2 = real<double>(dot);
            norm_restart_prev = norm_restart;
            norm_restart = lambda2;
            
            //reset acc. solution vector
            accum.template iterateWithConst<BLOCKSIZE>(gvect3_zero<floatT>());

            //reproject gradient vector so that pi and r are orthogonal
            dot2 = pi.dotProductStacked(r);
            pdotr = real<double>(dot2);
            
            SimpleArray<double,NStacks> proj(-1.0*pdotr/norm_restart);
            //pi = pi - <p,r>/|r|^2 * r
            pi.template axpyThisLoopd<32>(proj,r,NStacks);
                       
            pi.template xpayThisBd<SimpleArray<double, NStacks>,BLOCKSIZE>(beta,r);       
            norm_r2 = lambda2;
            norm_comp = lambda2;
            // rootLogger.debug() << "recompute precise residual " << max(norm_restart/norm_input);
        
        }
        else {
            //p_k+1 = r_k - beta*p_k
            pi.template xpayThisBd<SimpleArray<double, NStacks>,BLOCKSIZE>(beta,r);
            
        }
        //  rootLogger.info() << "CG-Mrel: residual "<< max(lambda2/norm_input);
    } while ( (max(lambda2/norm_input) > precision) && (cg<max_iter) );

    if(cg >= max_iter -1) {
        rootLogger.warn() << "CG: Warning max iteration reached " << cg;
    } else {
        rootLogger.info() << "CG: # iterations " << cg;
    }
    
    spinorOut += accum;
    spinorOut.updateAll();
    

}
        

template<class floatT, size_t NStacks>
template<typename Spinor_t, typename Spinor_t_inner>
void ConjugateGradient<floatT, NStacks>::invert_mrel(LinearOperator<Spinor_t>& dslash, LinearOperator<Spinor_t_inner>& dslash_inner, Spinor_t& spinorOut, const Spinor_t& spinorIn,
                                                     const int max_iter, const double precision, double delta)
{
    Spinor_t pi(spinorIn.getComm());
    // Spinor_t s(spinorIn.getComm());
    Spinor_t r(spinorIn.getComm());
    Spinor_t accum(spinorIn.getComm());
   
    Spinor_t_inner r_inner(spinorIn.getComm());
    Spinor_t_inner pi_inner(spinorIn.getComm());
    Spinor_t_inner s_inner(spinorIn.getComm());
    //  Spinor_t_inner r_inner(spinorIn.getComm());
    
    int cg = 0;
  
    SimpleArray<double, NStacks> beta(0.0);
    SimpleArray<double, NStacks> alpha(1.0);
    SimpleArray<double, NStacks> norm_r2(0.0);
    SimpleArray<double, NStacks> lambda2(0.0);
    SimpleArray<double, NStacks> pAp(0.0);
    SimpleArray<double, NStacks> pdotr(0.0);
    SimpleArray<GCOMPLEX(double), NStacks> dot(0.0);
    SimpleArray<GCOMPLEX(double), NStacks> dot2(0.0);
    SimpleArray<GCOMPLEX(double), NStacks> dot3(0.0);
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
    spinorOut.template iterateWithConst<BLOCKSIZE>(gvect3_zero<floatT>());
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
        r_inner.template axpyThisLoopd<32>(alpha, s_inner, NStacks);

        dot = r_inner.dotProductStacked(r_inner);
        lambda2 = real<double>(dot);
        beta = lambda2 / norm_r2;
  
  
  
        if (max(norm_comp) < max(norm_r2)) {
            norm_comp = lambda2;
        }

        //x_k+1 = x_k + |r|^2/pAp * p_k+1
        accum.template axpyThisLoopd<32>(-1.0*alpha, pi, NStacks);
        norm_r2 = lambda2;
        if ((max(lambda2) < delta*max(norm_restart)) && (max(norm_restart) <= max(norm_comp))) {
            //reliable update
            
            //cumulative update of solution vector
            //  accum.convert_precision<__inner>(accum_inner);
            spinorOut += accum;

            //r = b - Ax
            r = spinorIn;
            SimpleArray<double, NStacks> tmp_arr(-1.0);
            
            //reuse accum to save dslash result.
            
            dslash.applyMdaggM(accum,spinorOut, false);
            r.template axpyThisLoopd<32>(tmp_arr,accum,NStacks);
            r_inner.convert_precision(r);
            
            dot = r.dotProductStacked(r);
            lambda2 = real<double>(dot);
            norm_restart_prev = norm_restart;
            norm_restart = lambda2;
            
            //reset acc. solution vector
            accum.template iterateWithConst<BLOCKSIZE>(gvect3_zero<floatT>());

            //reproject gradient vector so that pi and r are orthogonal
            dot2 = pi.dotProductStacked(r);
            pdotr = real<double>(dot2);
            
            SimpleArray<double,NStacks> proj(-1.0*pdotr/norm_restart);

            //pi = pi - <p,r>/|r|^2 * r
            pi.template axpyThisLoopd<32>(proj,r,NStacks);
            //beta = norm_restart / norm_r2;
            pi.template xpayThisBd<SimpleArray<double, NStacks>,BLOCKSIZE>(beta,r);       
            pi_inner.convert_precision(pi);
            norm_r2 = lambda2;
            norm_comp = lambda2;
            //rootLogger.debug() << "recompute precise residual " << max(norm_restart/norm_input);
            steps_since_restart = 0;
            
        }
        else {
            //p_k+1 = r_k - a*p_k
            pi_inner.template xpayThisBd<SimpleArray<double, NStacks>, BLOCKSIZE>(beta,r_inner);
            pi.convert_precision(pi_inner);
            //pi.template xpayThisBd<SimpleArray<double, NStacks>,BLOCKSIZE>(beta,r);
            
            steps_since_restart++;
        }
        
        //rootLogger.info() << "CG-Mrel: residual "<< max(lambda2/norm_input);
    } while ( (max(lambda2/norm_input) > precision) && (cg<max_iter) );

    if(cg >= max_iter -1) {
        rootLogger.warn() << "CG: Warning max iteration reached " << cg;
    } else {
        rootLogger.info() << "CG: # iterations " << cg << " residual: " << max(lambda2/norm_input);
    }
    
    spinorOut += accum;
    spinorOut.updateAll();
    
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
template void ConjugateGradient<floatT,STACKS>::invert_mrel(LinearOperator<Spinorfield<floatT, true, LO, HALOSPIN, STACKS> >& dslash, LinearOperator<Spinorfield<__half, true, LO, HALOSPIN,STACKS> >& dslash_inner, Spinorfield<floatT, true, LO, HALOSPIN, STACKS>& spinorOut, const Spinorfield<floatT, true, LO, HALOSPIN,STACKS>& spinorIn, const int, const double, double); \
template void ConjugateGradient<floatT,STACKS>::invert_mrel(LinearOperator<Spinorfield<floatT, true, LO, HALOSPIN, STACKS> >& dslash, LinearOperator<Spinorfield<float, true, LO, HALOSPIN,STACKS> >& dslash_inner, Spinorfield<floatT, true, LO, HALOSPIN, STACKS>& spinorOut, const Spinorfield<floatT, true, LO, HALOSPIN,STACKS>& spinorIn, const int, const double, double);

#define CLASSMCG_INIT(floatT,LO,HALOSPIN,STACKS) \
    template class MultiShiftCG<floatT,true ,LO ,HALOSPIN, STACKS>;
#define CLASSAMCG_INIT(floatT,STACKS) \
    template class AdvancedMultiShiftCG<floatT, STACKS>;
#define CLASSAMCG_INV_INIT(floatT,LO,HALOSPIN,STACKS) \
template void AdvancedMultiShiftCG<floatT, STACKS>::invert(LinearOperator<Spinorfield<floatT, true, LO, HALOSPIN, 1> >& dslash, \
            Spinorfield<floatT, true, LO, HALOSPIN, STACKS>& spinorOut,const Spinorfield<floatT, true, LO, HALOSPIN, 1>& spinorIn, \
            SimpleArray<floatT, STACKS> sigma, const int, const double); \
template void AdvancedMultiShiftCG<floatT,STACKS>::invert_mixed<Spinorfield<__half, true, LO, HALOSPIN, STACKS> >(LinearOperator<Spinorfield<floatT, true, LO, HALOSPIN, 1> >& dslash, LinearOperator<Spinorfield<__half, true, LO, HALOSPIN,1> >& dslash_inner, Spinorfield<floatT, true, LO, HALOSPIN, STACKS>& spinorOut, const Spinorfield<floatT, true, LO, HALOSPIN,1>& spinorIn,  SimpleArray<floatT, STACKS> sigma,const int, const double); \
template void AdvancedMultiShiftCG<floatT,STACKS>::invert_mixed<Spinorfield<float, true, LO, HALOSPIN, STACKS> >(LinearOperator<Spinorfield<floatT, true, LO, HALOSPIN, 1> >& dslash, LinearOperator<Spinorfield<float, true, LO, HALOSPIN,1> >& dslash_inner, Spinorfield<floatT, true, LO, HALOSPIN, STACKS>& spinorOut, const Spinorfield<floatT, true, LO, HALOSPIN,1>& spinorIn,  SimpleArray<floatT, STACKS> sigma,const int, const double);
INIT_PN(CLASSCG_INIT)
INIT_PLHSN(CLASSCG_INV_INIT)
INIT_PLHSN(CLASSMCG_INIT)
INIT_PN(CLASSAMCG_INIT)
INIT_PLHSN(CLASSAMCG_INV_INIT)
