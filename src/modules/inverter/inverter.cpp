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

    SimpleArray<GCOMPLEX(double), NStacks> dot(0);

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



//ranluo. reliable update
template<class floatT, size_t NStacks>
template <typename SpinorIn_t, typename SpinorOut_t>
void AdvancedMultiShiftCG<floatT, NStacks>::invert(
        LinearOperator<SpinorIn_t>& dslash, SpinorOut_t& spinorOut, const SpinorIn_t& spinorIn,
        SimpleArray<floatT, NStacks> sigma, const int max_iter, const double precision)
{
    rootLogger.info(" ");
    double delta = ( (double)(max_iter%10) ) / 10.0;
    int max_term = NStacks;
    int cg = 0;
    int cg_cor = 0;

    SpinorOut_t pi(spinorIn.getComm());
    SpinorIn_t s(spinorIn.getComm());
    SpinorIn_t r(spinorIn.getComm());
    SpinorIn_t pi0(spinorIn.getComm());
    SpinorOut_t accum(spinorIn.getComm());
    SpinorIn_t tmp(spinorIn.getComm());

    SimpleArray<double, NStacks> a(0.0);
    SimpleArray<double, NStacks> B(1.0);
    double Bm1 = 1.0;
    SimpleArray<double, NStacks> Z(1.0);
    SimpleArray<double, NStacks> Zm1(1.0);

    r = spinorIn;
    for(size_t i = 0; i < NStacks; i++)pi.copyFromStackToStack(spinorIn, i ,0);
    spinorOut.template iterateWithConst<BLOCKSIZE>(gvect3_zero<floatT>());
    accum = spinorOut;

    double pAp, lambda, lambda2, rr_1;
    double norm_r2 = r.realdotProduct(r);
    double norm_restart = norm_r2;
    do {
        cg++;

        //beta unshift
        pi0.copyFromStackToStack(pi, 0, 0);
        pi0.updateAll(COMM_BOTH | Hyperplane);
        dslash.applyMdaggM(s, pi0, false);
        //dslash.applyMdaggM_async(s, pi0, false);
        s = sigma[0] * pi0 - s; //s = A*pi0 , A = sigma0 - Mdagg*M
        pAp = pi0.realdotProduct(s);
        B[0] = - norm_r2 / pAp;

        //beta shift, zeta
        for (int j=1; j<max_term; j++) {
            rr_1   = Bm1 * Zm1[j] / ( B[0] * a[0] * (Zm1[j] - Z[j]) 
                       + Zm1[j] * Bm1 * (1.0 - sigma[j] * B[0]) );
            Zm1[j] = Z[j];
            Z[j]  *= rr_1;
            B[j]   = B[0] * rr_1;
        }
        Bm1 = B[0];

        //x
        accum.template axpyThisLoopd<32>((-1.0)*B, pi, max_term);

        //r_iter
        r.template axpyThisB<64>((floatT)B[0], s);
        lambda2 = r.realdotProduct(r);

        //alpha
        a[0]  = lambda2 / norm_r2;
        for (int j=1; j<max_term; j++) {
            a[j] = a[0] * Z[j] * B[j] / (Zm1[j] * B[0]);
        } //index???

        if( lambda2 < delta*norm_restart )
        {
            ++cg_cor;

            //x. to be optimized
            spinorOut += accum;
            accum.template iterateWithConst<BLOCKSIZE>(gvect3_zero<floatT>());

            //r_true
            r.copyFromStackToStack(spinorOut, 0, 0);//reuse r to save the x0
            r.updateAll(COMM_BOTH | Hyperplane);
            dslash.applyMdaggM(s, r, false);
            //dslash.applyMdaggM_async(s, r, false);
            s = sigma[0] * r - s; //for check
            r = spinorIn;
            r -= s;
            
            lambda2 = r.realdotProduct(r);
            norm_restart = lambda2;

            //reproject gradient vector so that p(i) and r(i+1) are orthogonal
            pAp = pi0.realdotProduct(r);
            pAp = -1.0*pAp/lambda2;
            pi0.template axpyThisB<64>((floatT)pAp,r);
            pi.copyFromStackToStack(pi0, 0, 0);

            //modifiying the other terms
            /*for(int i=1;i<max_term;++i)
            {
                tmp.copyFromStackToStack(spinorOut, 0, i);
                tmp.updateAll(COMM_BOTH | Hyperplane);
                dslash.applyMdaggM(s, tmp, false);
                s = (sigma[i] + sigma[0] ) * tmp - s;
                tmp = spinorIn;
                tmp -= s;
                lambda = tmp.realdotProduct(tmp);
                //Z[i] = sqrt( lambda / lambda2 );

                s.copyFromStackToStack(pi, 0, i);
                pAp = s.realdotProduct(tmp);
                pAp = -1.0*pAp/lambda;
                s.template axpyThisB<64>((floatT)pAp,tmp);
                pi.copyFromStackToStack(s, i, 0);
            }*/
        }        
        norm_r2 = lambda2;        

        //p
        pi.template axupbyThisLoopd<64>(Z, a, r, max_term);//pi[i] = Z[i] * r + a[i] * pi[i];

        //check if the max_term converges
        do {
            lambda = pow(Z[max_term-1],1.5) * lambda2;
            if ( lambda < precision) {
                max_term--;
                rootLogger.info("iter = " , cg , " , max_term = ", max_term , " , Z(" , max_term , ") = " , Z[max_term]);
            }
        } while ( max_term > 0 && (lambda < precision) );

    } while ( (max_term>0) && (cg<max_iter) );

    if(cg >= max_iter -1) {
        rootLogger.warn("CG: Warning max iteration reached " , cg , " (" , cg_cor , ")");
    } else {
        rootLogger.info("CG: # iterations " , cg , " (" , cg_cor , ")");
    }

    spinorOut += accum;
    spinorOut.updateAll();


    //compare 2 kinds of residues
    r.copyFromStackToStack(spinorOut, 0, 0); //reuse r to save x0
    dslash.applyMdaggM(s, r, false);
    s = sigma[0] * r - s;
    r = spinorIn;
    r -= s;
    rootLogger.info( "for the zeroth term: r_iter = " , lambda2 , " , r_true = " , r.realdotProduct(r) );
    rootLogger.info( "r_true for the rest terms: " );
    for(int i=1;i<NStacks;++i)
    {
        r.copyFromStackToStack(spinorOut, 0, i);
        dslash.applyMdaggM(s, r, false);
        s = ( sigma[i] + sigma[0] ) * r - s;
        r = spinorIn;
        r -= s;
        lambda2 = r.realdotProduct(r);
        rootLogger.info( lambda2 );
    }
    rootLogger.info(" ");
}


/*
//origin
template<class floatT, size_t NStacks>
template <typename SpinorIn_t, typename SpinorOut_t>
void AdvancedMultiShiftCG<floatT, NStacks>::invert(
        LinearOperator<SpinorIn_t>& dslash, SpinorOut_t& spinorOut, const SpinorIn_t& spinorIn,
        SimpleArray<floatT, NStacks> sigma, const int max_iter, const double precision)
{
    rootLogger.info(" ");
    
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

    double  pAp,lambda, lambda2, rr_1, Bm1;

    Bm1 = 1.0;

    for (size_t i = 0; i < NStacks; i++) {
        pi.copyFromStackToStack(spinorIn, i ,0);
    }

    spinorOut.template iterateWithConst<BLOCKSIZE>(gvect3_zero<floatT>());


    do {
        cg++;

        pi0.copyFromStackToStack(pi, 0, 0);
        
        pi0.updateAll(COMM_BOTH | Hyperplane);
        dslash.applyMdaggM(s, pi0, false);
        //dslash.applyMdaggM_async(s, pi0, false);

        s = sigma[0] * pi0 - s;

        pAp = pi0.realdotProduct(s);

        B[0] = - norm_r2 / pAp;

        r.template axpyThisB<64>(B[0], s);


        for (int j=1; j<max_term; j++) {
            rr_1   = Bm1 * Zm1[j] / ( B[0] * a[0] * (Zm1[j] - Z[j]) 
                       + Zm1[j] * Bm1 * (1.0 - sigma[j] * B[0]) );
            Zm1[j] = Z[j];
            Z[j]   = Z[j] * rr_1;
            B[j]   = B[0] * rr_1;
        }
        Bm1 = B[0];
        lambda2 = r.realdotProduct(r);
        a[0]  = lambda2 / norm_r2;
        norm_r2 = lambda2;


        spinorOut.template axpyThisLoop<64>(((floatT)(-1.0))*B, pi,max_term); 
        //     spinorOut[i] = spinorOut[i] - B[i] * pi[i];
        

        //################################
        for (int j=1; j<max_term; j++) {
            a[j] = a[0] * Z[j] * B[j] / (Zm1[j] * B[0]);
        }
        //################################        


        pi.template axupbyThisLoop<64>(Z, a, r, max_term);
        //     pi[i] = Z[i] * r + a[i] * pi[i];


        //################################

        do {
            lambda = Z[max_term-1] * Z[max_term-1] * lambda2;
            if ( lambda < precision) {
                max_term--;
                rootLogger.info("iter = " , cg , " , max_term = ", max_term);
            }
        } while ( max_term > 0 && (lambda < precision) );

    } while ( (max_term>0) && (cg<max_iter) );

    if(cg >= max_iter -1) {
        rootLogger.warn("CG: Warning max iteration reached " ,  cg);
    } else {
        rootLogger.info("CG: # iterations " ,  cg);
    }

    spinorOut.updateAll();


    //compare 2 kinds of residues
    rootLogger.info( "for the zeroth term: |p(i)|^2 = " , pi0.realdotProduct(pi0) , " , <p(i),r_iter(i+1)> = " , pi0.realdotProduct(r) );
    r.copyFromStackToStack(spinorOut, 0, 0);
    dslash.applyMdaggM_async(s, r, false);
    s = sigma[0] * r - s;
    r = spinorIn;
    r -= s; //r_true
    rootLogger.info( "<p(i),r_true(i+1)> = " , pi0.realdotProduct(r) , " , r_iter = " , lambda , " , r_true = " , r.realdotProduct(r) );
    rootLogger.info( "r_true for the rest 13 terms: " );
    for(int i=1;i<NStacks;++i)
    {
        pi0.copyFromStackToStack(spinorOut, 0, i);
        dslash.applyMdaggM_async(s, pi0, false);
        s = ( sigma[i] + sigma[0] ) * pi0 - s;
        r = spinorIn;
        r -= s;
        lambda2 = r.realdotProduct(r);
        rootLogger.info( lambda2 );
    }
    rootLogger.info(" ");
}
*/


//ranluo. previous invert_new() not retained
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

    r = spinorIn;

    dot = r.dotProductStacked(r);
    norm_r2 = real<double>(dot);

    SimpleArray<double, NStacks> in_norm(0.0);

    in_norm = norm_r2;

    pi = spinorIn;

    spinorOut.template iterateWithConst<BLOCKSIZE>(gvect3_zero<floatT>());


    do {
        cg++;

        pi.updateAll(COMM_BOTH | Hyperplane);
        dslash.applyMdaggM(s, pi, false);
        //dslash.applyMdaggM_async(s, pi, false);

        dot = pi.dotProductStacked(s);

        pAp = real<double>(dot);

        B = -1.0* norm_r2 / pAp;

        r.template axpyThisLoopd<32>(B, s, NStacks); 

        dot = r.dotProductStacked(r);

        lambda2 = real<double>(dot);
        a = lambda2 / norm_r2;
        norm_r2 = lambda2;

        spinorOut.template axpyThisLoopd<32>(-1.0*B, pi,NStacks); 

        pi.template xpayThisBd<SimpleArray<double, NStacks>,BLOCKSIZE>(a, r);

    } while ( (max(lambda2/in_norm) > precision) && (cg<max_iter) );

    if(cg >= max_iter -1) {
        rootLogger.warn("CG: Warning max iteration reached " ,  cg);
    } else {
        rootLogger.info("CG: # iterations " ,  cg);
    }

    spinorOut.updateAll();

    //compare 2 kinds of residues
    dslash.applyMdaggM(s, spinorOut, false);
    r = spinorIn;
    r -= s;
    dot = r.dotProductStacked(r);
    norm_r2 = real<double>(dot);
    rootLogger.info( "r0 = " , max(in_norm) );
    rootLogger.info( "r_iter = " , max(lambda2) , " , r_true = " , max(norm_r2) );
}


template<class floatT, size_t NStacks>
template <typename Spinor_t>
void ConjugateGradient<floatT, NStacks>::invert_res_replace(LinearOperator<Spinor_t>& dslash, Spinor_t& spinorOut, const Spinor_t& spinorIn, const int max_iter, const double precision, double delta)
{
    rootLogger.info("\n invert_res_replace starts");
    const floatT sigma = 6.6574712798406069e-9 + 0.000325 * 0.000325;//for check

    Spinor_t pi(spinorIn.getComm());
    Spinor_t s(spinorIn.getComm());
    Spinor_t r(spinorIn.getComm());
    Spinor_t accum(spinorIn.getComm());
    
    int cg = 0;
    int cg_cor = 0;

    SimpleArray<double, NStacks> beta(0.0);
    SimpleArray<double, NStacks> alpha(1.0);
    SimpleArray<double, NStacks> pAp(0.0);
    SimpleArray<double, NStacks> pdotr(0.0);
    SimpleArray<GCOMPLEX(double), NStacks> dot(0.0);
    
    spinorOut.template iterateWithConst<BLOCKSIZE>(gvect3_zero<floatT>());
    accum = spinorOut;
    r = spinorIn;
    pi = spinorIn;

    dot = r.dotProductStacked(r);
    SimpleArray<double, NStacks> norm_r2 = real<double>(dot);
    //const SimpleArray<double, NStacks> norm_input = norm_r2; //for check
    SimpleArray<double, NStacks> lambda2(1.0);
    SimpleArray<double, NStacks> norm_restart = norm_r2;
    do {
        cg++;
        
        //alpha
        pi.updateAll(COMM_BOTH | Hyperplane);
        dslash.applyMdaggM(s,pi,false);
        s = sigma * pi - s; //for check
        dot = pi.dotProductStacked(s);
        pAp = real<double>(dot);
        alpha = norm_r2 / pAp;

        //x
        accum.template axpyThisLoopd<32>(alpha, pi, NStacks);

        //r_iter
        r.template axpyThisLoopd<32>(-1.0*alpha, s, NStacks);

        //beta
        dot = r.dotProductStacked(r);
        lambda2 = real<double>(dot);
        beta = lambda2 / norm_r2; //???
        
        if ( max(lambda2) < delta*max(norm_restart) ) {
            ++cg_cor;
            rootLogger.info("cg = ", cg , " , r_iter = ", max(lambda2) ); //for check

            //x
            spinorOut += accum;
            accum.template iterateWithConst<BLOCKSIZE>(gvect3_zero<floatT>());

            //r_true
            r = spinorIn;
            spinorOut.updateAll(COMM_BOTH | Hyperplane);
            dslash.applyMdaggM(s,spinorOut, false);
            s = sigma * spinorOut - s; //for check
            r -= s;

            dot = r.dotProductStacked(r);
            lambda2 = real<double>(dot);
            norm_restart = lambda2;
            rootLogger.info("r_ture = ", max(lambda2) ); //for check

            //reproject gradient vector so that p(i) and r(i+1) are orthogonal
            dot = pi.dotProductStacked(r);
            pdotr = real<double>(dot);
            SimpleArray<double,NStacks> proj(-1.0*pdotr/norm_restart);
            pi.template axpyThisLoopd<32>(proj,r,NStacks);//pi = pi - <p,r>/|r|^2 * r
        }
        //beta = lambda2 / norm_r2; //???
        norm_r2 = lambda2;

        //p
        pi.template xpayThisBd<SimpleArray<double, NStacks>,BLOCKSIZE>(beta,r);

    } while ( (max(lambda2/*/norm_input //for check*/) > precision) && (cg<max_iter) );

    if(cg >= max_iter -1) {
        rootLogger.warn("CG: Warning max iteration reached " , cg , " (" , cg_cor , ")");
    } else {
        rootLogger.info("CG: # iterations " , cg , " (" , cg_cor , ")");
    }
    
    spinorOut += accum;
    spinorOut.updateAll();


    //compare 2 kinds of residues
    dslash.applyMdaggM_async(s, spinorOut, false);
    s = sigma * spinorOut - s; //for check
    r = spinorIn;
    r -= s;
    dot = r.dotProductStacked(r);
    norm_r2 = real<double>(dot);
    //rootLogger.info( "r0 = " , max(norm_input) );
    rootLogger.info( "r_iter = " , max(lambda2) , " , r_true = " , max(norm_r2) );
    rootLogger.info("invert_res_replace ends\n");
}
        


//ranluo
template<class floatT, size_t NStacks>
template<typename Spinor_t, typename Spinor_t_inner>
void ConjugateGradient<floatT, NStacks>::invert_mixed(LinearOperator<Spinor_t>& dslash, LinearOperator<Spinor_t_inner>& dslash_inner, Spinor_t& spinorOut, const Spinor_t& spinorIn,
                                                     const int max_iter, const double precision, double delta)
{
    rootLogger.info("\n invert_mixed starts");
    const floatT sigma = 6.6574712798406069e-9 + 0.000325 * 0.000325;//for check
    SimpleArray<double, NStacks> sigma_array((double)sigma);// for check

    Spinor_t pi(spinorIn.getComm());
    Spinor_t r(spinorIn.getComm());
    Spinor_t accum(spinorIn.getComm());
   
    Spinor_t_inner r_inner(spinorIn.getComm());
    Spinor_t_inner pi_inner(spinorIn.getComm());
    Spinor_t_inner s_inner(spinorIn.getComm());
    
    int cg = 0;
    int cg_high_prec = 0;
  
    SimpleArray<double, NStacks> beta(0.0);
    SimpleArray<double, NStacks> alpha(1.0);
    SimpleArray<double, NStacks> lambda2(0.0);
    SimpleArray<double, NStacks> pAp(0.0);
    SimpleArray<double, NStacks> pdotr(0.0);

    SimpleArray<double, NStacks> const_one(1.0);
    
    r = spinorIn;
    r_inner.convert_precision(r);
    pi = spinorIn;    
    pi_inner.convert_precision(pi);
    spinorOut.template iterateWithConst<BLOCKSIZE>(gvect3_zero<floatT>());
    accum = spinorOut;
    
    SimpleArray<GCOMPLEX(double), NStacks> dot(0.0);
    dot = r.dotProductStacked(r);
    SimpleArray<double, NStacks> norm_r2 = real<double>(dot);
    //const SimpleArray<double, NStacks> norm_input = norm_r2; //for check
    SimpleArray<double, NStacks> norm_restart = norm_r2;
    do {
        cg++;

        //alpha
        dslash_inner.applyMdaggM_async(s_inner,pi_inner,false);
        s_inner.template axpyThisLoopd<32>(-2.0*const_one, s_inner, NStacks);//for check
        s_inner.template axpyThisLoopd<32>(sigma_array, pi_inner, NStacks);//for check
        dot = pi_inner.dotProductStacked(s_inner);
        pAp = real<double>(dot);
        alpha = norm_r2 / pAp;

        //x
        accum.template axpyThisLoopd<32>(alpha, pi, NStacks);

        //r
        r_inner.template axpyThisLoopd<32>(-1.0*alpha, s_inner, NStacks);

        //beta
        dot = r_inner.dotProductStacked(r_inner);
        lambda2 = real<double>(dot);
        beta = lambda2 / norm_r2;
        
        if (max(lambda2) < delta*max(norm_restart)) {
            rootLogger.info("cg = ", cg , " , r_iter = ", max(lambda2) ); //for check
            ++cg_high_prec;
            
            //x
            spinorOut += accum;

            //r = b - Ax, reuse accum to save dslash result.
            r = spinorIn;
            dslash.applyMdaggM_async(accum,spinorOut, false);
            accum = sigma * spinorOut - accum; //for check
            r -= accum;
            r_inner.convert_precision(r);
            
            dot = r.dotProductStacked(r);
            lambda2 = real<double>(dot);
            norm_restart = lambda2;
            rootLogger.info("r_ture = ", max(lambda2) ); //for check
            
            //reset acc. solution vector
            accum.template iterateWithConst<BLOCKSIZE>(gvect3_zero<floatT>());

            //reproject gradient vector so that p(i) and r(i+1) are orthogonal
            dot = pi.dotProductStacked(r);
            pdotr = real<double>(dot);
            SimpleArray<double,NStacks> proj(-1.0*pdotr/norm_restart);
            pi.template axpyThisLoopd<32>(proj,r,NStacks);//pi = pi - <p,r>/|r|^2 * r
            
            //beta = norm_restart / norm_r2;//?????????????????????????
            
            //p
            pi.template xpayThisBd<SimpleArray<double, NStacks>,BLOCKSIZE>(beta,r);       
            pi_inner.convert_precision(pi);
            
        } else {
            //p
            pi_inner.template xpayThisBd<SimpleArray<double, NStacks>, BLOCKSIZE>(beta,r_inner);
            pi.convert_precision(pi_inner);
        }
        norm_r2 = lambda2;
        
    } while ( (max(lambda2/*/norm_input //for check*/) > precision) && (cg<max_iter) );

    if(cg >= max_iter -1) {
        rootLogger.warn("CG: Warning max iteration reached " ,  cg);
    } else {
        rootLogger.info("CG: # iterations " ,  cg , " (" , cg_high_prec , ")");
    }
    
    spinorOut += accum;
    spinorOut.updateAll();


    //compare 2 kinds of residues
    dslash.applyMdaggM_async(accum, spinorOut, false);
    accum = sigma * spinorOut - accum; //for check
    r = spinorIn;
    r -= accum;
    dot = r.dotProductStacked(r);
    norm_r2 = real<double>(dot);
    //rootLogger.info( "r0 = " , max(norm_input) );
    rootLogger.info( "r_iter = " , max(lambda2) , " , r_true = " , max(norm_r2) );
    rootLogger.info("invert_mixed ends\n");
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
#if DOUBLEPREC == 1 && SINGLEPREC ==1
INIT_PLHSN(CLASSCG_FLOAT_INV_INIT)
#endif
#if HALFPREC == 1
INIT_PLHSN_HALF(CLASSCG_HALF_INV_INIT)
#endif
INIT_PLHSN(CLASSMCG_INIT)
INIT_PN(CLASSAMCG_INIT)
INIT_PLHSN(CLASSAMCG_INV_INIT)