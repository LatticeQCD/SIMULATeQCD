/*
 * mixedMultiShiftCG.h
 *
 * Mixed-precision multi-shift CG with reliable updates. The Krylov iteration (the operator
 * applied to p_0, the residual r and the search directions p_j) runs in single precision;
 * the solutions x_j are accumulated in double precision. Whenever |r|^2 has dropped by a
 * factor delta since the last reliable update, r is replaced by the true double-precision
 * residual b - A_0 x_0 of shift 0. At the end, every shift whose true residual is still above
 * the requested precision is refined with the mixed-precision defect-correction CG
 * (ConjugateGradient::invert_mixed), starting from its multi-shift solution.
 *
 * Operators and stopping rule are those of AdvancedMultiShiftCG: stack 0 solves
 * (sigma[0] - A) x = b, stack j > 0 solves (sigma[0] + sigma[j] - A) x = b, A = applyMdaggM,
 * and a shift is done when its squared residual norm is below precision (absolute). Every
 * step is a fixed function of the input, so molecular dynamics stays reversible.
 */
#pragma once

#include "inverter.h"
#include "../../spinor/cpuSpinorKernels.h"
#include <algorithm>

//! (shift - A) with A = dslash.applyMdaggM: one shift of the multi-shift CG as a LinearOperator.
template<typename floatT, bool onDevice, Layout LatLayout, size_t HaloDepthSpin, typename Dslash_t>
class ShiftedMdaggM : public LinearOperator<Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, 1>> {
    using Spinor_t = Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, 1>;
    Dslash_t &_dslash;
    floatT _shift;

public:
    ShiftedMdaggM(Dslash_t &dslash, double shift) : _dslash(dslash), _shift(static_cast<floatT>(shift)) {}

    void applyMdaggM(Spinor_t &out, const Spinor_t &in, bool update = true) override {
        _dslash.applyMdaggM(out, in, false);
        out = _shift * in - out;
        if (update) out.updateAll();
    }
};

//! x_j += alpha_j p_j (x in double, p in single precision), then p_j = beta_j p_j + z_j r,
//! for all active shifts in one pass.
template<class floatX, class floatS, Layout LatLayout, size_t HaloDepth, size_t NStacks>
struct MixedMultiShiftUpdateNoReturn {
    Vect3arrayAcc<floatX> _x;
    Vect3arrayAcc<floatS> _p, _r;
    SimpleArray<floatX, NStacks> _alpha;
    SimpleArray<floatS, NStacks> _z, _beta;
    size_t _maxStack;
    typedef GIndexer<LatLayout, HaloDepth> GInd;

    MixedMultiShiftUpdateNoReturn(Vect3arrayAcc<floatX> x, Vect3arrayAcc<floatS> p, Vect3arrayAcc<floatS> r,
                                  SimpleArray<floatX, NStacks> alpha, SimpleArray<floatS, NStacks> z,
                                  SimpleArray<floatS, NStacks> beta, size_t maxStack)
        : _x(x), _p(p), _r(r), _alpha(alpha), _z(z), _beta(beta), _maxStack(maxStack) {}

    __host__ __device__ void operator()(gSiteStack &site) {
        const Vect3<floatS> r = _r.getElement(GInd::getSiteStack(site, 0));
        static_for<0, NStacks>::apply([&](auto stack) {
            if (stack < _maxStack) {
                const gSiteStack siteStack = GInd::getSiteStack(site, stack);
                Vect3<floatX> x = _x.getElement(siteStack);
                x += _alpha[stack] * _p.template getElement<floatX>(siteStack);
                _x.setElement(siteStack, x);
                _p.setElement(siteStack, _p.getElement(siteStack) * _beta[stack] + r * _z[stack]);
            }
        });
    }
};

template<typename floatT, bool onDevice, Layout LatLayout, size_t HaloDepthSpin, size_t NStacks>
class MixedPrecMultiShiftCG {
    using Spinor_t = Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, 1>;
    using SpinorStack_t = Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks>;
    using SpinorF_t = Spinorfield<float, onDevice, LatLayout, HaloDepthSpin, 1>;
    using SpinorStackF_t = Spinorfield<float, onDevice, LatLayout, HaloDepthSpin, NStacks>;

    ConjugateGradient<floatT, 1> _cgRefine;

public:
    //! dslash and dslashFloat must represent the same operator in double and single precision.
    template<typename Dslash_t, typename DslashF_t>
    void invert(Dslash_t &dslash, DslashF_t &dslashFloat, SpinorStack_t &spinorOut, Spinor_t &spinorIn,
                SimpleArray<floatT, NStacks> sigma, const int max_iter, const double precision, const double delta) {
        CommunicationBase &comm = spinorIn.getComm();
        SpinorStackF_t pi(comm);
        SpinorF_t pi0(comm), s(comm), r(comm);
        Spinor_t x0(comm), rTrue(comm), Ax(comm);

        // Multi-shift recurrence coefficients, in double precision.
        SimpleArray<double, NStacks> a(0.0), B(1.0), Z(1.0), Zm1(1.0);
        SimpleArray<double, NStacks> alphaX(0.0);
        SimpleArray<float, NStacks> zFloat(0.0f), aFloat(0.0f);
        const float sigma0 = static_cast<float>(sigma[0]);

        r.convert_precision(spinorIn);
        double norm_r2 = r.realdotProduct(r);
        double normReliable = norm_r2;
        double Bm1 = 1.0;
        for (size_t i = 0; i < NStacks; i++) pi.copyFromStackToStack(r, i, 0);
        spinorOut.iterateWithConst(vect3_zero<floatT>());

        // CPU: three fused passes per iteration (cpuSpinorKernels.h); the update also refreshes pi0.
#ifdef USE_CPU
        const bool fused = cpuBlas::fusedEnabled();
#else
        constexpr bool fused = false;
#endif
        int max_term = NStacks;
        int cg = 0, reliableUpdates = 0;
        if (fused) pi0.template copyFromStackToStackDevice<NStacks, 0, 0>(pi);
        do {
            cg++;
            if (!fused) pi0.template copyFromStackToStackDevice<NStacks, 0, 0>(pi);
            updateInputHalos(dslashFloat, pi0);
            dslashFloat.applyMdaggM(s, pi0, false);

            double pAp, lambda2;
#ifdef USE_CPU
            if (fused) {
                pAp = cpuBlas::shiftedRealDot(s, pi0, sigma0);
                B[0] = -norm_r2 / pAp;
                lambda2 = cpuBlas::axpyNorm(r, static_cast<float>(B[0]), s);
            } else
#endif
            {
                s = sigma0 * pi0 - s;
                pAp = pi0.realdotProduct(s);
                B[0] = -norm_r2 / pAp;
                r.axpyThisB(static_cast<float>(B[0]), s);
                lambda2 = r.realdotProduct(r);
            }

            for (int j = 1; j < max_term; j++) {
                const double rr_1 = Bm1 * Zm1[j] / (B[0] * a[0] * (Zm1[j] - Z[j])
                                     + Zm1[j] * Bm1 * (1.0 - static_cast<double>(sigma[j]) * B[0]));
                Zm1[j] = Z[j];
                Z[j] = Z[j] * rr_1;
                B[j] = B[0] * rr_1;
            }
            Bm1 = B[0];

            // The reliable update (after x_0 has received this step) once |r|^2 fell by delta.
            const bool reliable = lambda2 < delta * normReliable;
            a[0] = lambda2 / norm_r2;
            for (int j = 1; j < max_term; j++) a[j] = a[0] * Z[j] * B[j] / (Zm1[j] * B[0]);
            for (int j = 0; j < max_term; j++) {
                alphaX[j] = -B[j];
                zFloat[j] = static_cast<float>(Z[j]);
                aFloat[j] = static_cast<float>(a[j]);
            }

#ifdef USE_CPU
            if (fused) {
                cpuBlas::multiShiftUpdate(spinorOut, pi, r, alphaX, zFloat, aFloat, max_term, pi0);
            } else
#endif
            {
                CalcGSiteStackFull<LatLayout, HaloDepthSpin> calcGSite;
                iterateFunctorNoReturn<onDevice, 64>(
                    MixedMultiShiftUpdateNoReturn<floatT, float, LatLayout, HaloDepthSpin, NStacks>(
                        spinorOut.getAccessor(), pi.getAccessor(), r.getAccessor(), alphaX, zFloat, aFloat, max_term),
                    calcGSite, spinorOut.getNumberLatticePointsFull(), 1);
            }

            if (reliable) {
                // r = b - A_0 x_0 in double precision; continue the recurrence from the true residual.
                x0.copyFromStackToStack(spinorOut, 0, 0);
                x0.updateAll();
                dslash.applyMdaggM(Ax, x0, false);
                Ax = sigma[0] * x0 - Ax;
                rTrue = spinorIn - Ax;
                r.convert_precision(rTrue);
                lambda2 = rTrue.realdotProduct(rTrue);
                normReliable = lambda2;
                reliableUpdates++;
            }
            norm_r2 = lambda2;

            while (max_term > 0 && Z[max_term - 1] * Z[max_term - 1] * lambda2 < precision) max_term--;
        } while (max_term > 0 && cg < max_iter);

        if (cg >= max_iter - 1) {
            rootLogger.warn("Mixed multi-shift CG: Warning max iteration reached ", cg);
        }
        rootLogger.info("Mixed multi-shift CG: # iterations ", cg, " (", reliableUpdates, " reliable updates)");

        refine(dslash, dslashFloat, spinorOut, spinorIn, sigma, max_iter, precision, delta);
    }

private:
    //! Refine every shift whose true residual is above precision, starting from its solution.
    template<typename Dslash_t, typename DslashF_t>
    void refine(Dslash_t &dslash, DslashF_t &dslashFloat, SpinorStack_t &spinorOut, Spinor_t &spinorIn,
                SimpleArray<floatT, NStacks> sigma, const int max_iter, const double precision, const double delta) {
        CommunicationBase &comm = spinorIn.getComm();
        Spinor_t x(comm), r(comm), Ax(comm), correction(comm);
        int refined = 0;
        double worstBefore = 0.0, worst = 0.0;
        for (size_t j = 0; j < NStacks; j++) {
            const double shift = j == 0 ? static_cast<double>(sigma[0])
                                        : static_cast<double>(sigma[0]) + static_cast<double>(sigma[j]);
            ShiftedMdaggM<floatT, onDevice, LatLayout, HaloDepthSpin, Dslash_t> op(dslash, shift);
            ShiftedMdaggM<float, onDevice, LatLayout, HaloDepthSpin, DslashF_t> opFloat(dslashFloat, shift);

            x.copyFromStackToStack(spinorOut, 0, j);
            x.updateAll();
            op.applyMdaggM(Ax, x, false);
            r = spinorIn - Ax;
            const double rr = r.realdotProduct(r);
            worstBefore = std::max(worstBefore, rr);
            if (rr > precision) {
                _cgRefine.invert_mixed(op, opFloat, correction, r, max_iter, precision / rr, delta);
                x = x + correction;
                x.updateAll();
                op.applyMdaggM(Ax, x, false);
                r = spinorIn - Ax;
                spinorOut.copyFromStackToStack(x, j, 0);
                refined++;
            }
            worst = std::max(worst, r.realdotProduct(r));
        }
        spinorOut.updateAll();
        rootLogger.info("Mixed multi-shift CG: largest true |r|^2 ", worstBefore, " after the multi-shift stage; refined ",
                        refined, " of ", NStacks, " shifts; largest final |r|^2 ", worst, " (requested ", precision, ")");
    }
};
