/*
 * Even/odd (red-black) decomposition of the Mobius clover MDWF operator and
 * its Schur complement, for preconditioned solves (EVEN_ODD_DESIGN.md).
 *
 * The operator of MDWFMobiusCloverLinearOperator is
 *
 *   M = D_W Din + Shift,   D_W = A - (1/2) Hop,
 *
 * with A = mass + clover (site-local, the two 6x6 chiral blocks computed by
 * preCalcFmunu), Hop the Wilson hopping term (DiracWilsonEvenOdd2, which
 * already contains the -1/2), and Din, Shift the gauge-independent
 * fifth-direction couplings. Hop connects even and odd sites only, so
 *
 *   M_ee = A Din + Shift (even sites),   M_oo = A Din + Shift (odd sites),
 *   M_eo = Hop_eo Din,                   M_oe = Hop_oe Din,
 *
 * and, with A Hermitian and Hop^dagger = gamma5 Hop gamma5,
 *
 *   M_xx^dagger = Din^dagger A + Shift^dagger,
 *   M_eo^dagger = Din^dagger gamma5 Hop_oe gamma5,   M_oe^dagger = Din^dagger gamma5 Hop_eo gamma5.
 *
 * Diagonal-block inverse. For chirality chi (upper 6 components = P_+,
 * lower = P_-, the same split as the clover blocks), Din = b5 + c5 K and
 * Shift = 1 - K, where K is the one-directional shift in s with boundary
 * factor -mf ((K x)(s) = x(s-1), (K x)(0) = -mf x(Ls-1) for P_+; the mirror
 * image x(s+1) for P_-). So per site
 *
 *   M_xx = P + Q K,   P = b5 A + 1,   Q = c5 A - 1,
 *
 * with P, Q (6x6 Hermitian, commuting) independent of s. With R = P^-1 Q and
 * K^Ls = -mf, (P + Q K)^-1 = (1 + R K)^-1 P^-1, and (1 + R K) x = z is solved
 * by one sweep per chirality:
 *
 *   u = sum_{k=1}^{Ls-1} (-R)^(Ls-1-k) z(k)          (Horner, s = 1 ... Ls-1)
 *   x(0) = W (z(0) + mf R u),   W = (1 + (-1)^Ls mf R^Ls)^-1
 *   x(s) = z(s) - R x(s-1),     s = 1 ... Ls-1
 *
 * (mirrored in s for P_-). P^-1, R, W are precomputed per site and chirality
 * (six Hermitian 6x6 matrices per site). The adjoint M_xx^dagger = P + Q K^dagger
 * uses the same matrices with the sweep directions exchanged. The block
 * inverse is exact (to rounding) for any gauge field.
 *
 * Schur complement on the even sites (det M = det M_oo det Mhat):
 *
 *   Mhat          = M_ee - M_eo M_oo^-1 M_oe,
 *   Mhat^dagger   = M_ee^dagger - M_oe^dagger M_oo^-dagger M_eo^dagger.
 *
 * The clover field and the block-inverse matrices depend on the gauge field
 * and are recomputed by refresh(); call it after every gauge update (the
 * solver wrapper MDWFMobiusEvenOddSolver does so at the start of each solve).
 *
 * Even/odd split and merge of MDWF spinors use MDWF-local functors that keep
 * the fifth-dimension stack: SpinorfieldAll's conversion (returnSpinor) reads
 * its source through a gSite without stack index.
 *
 * Precision: floatT = double is the reference. floatT = float (the inner
 * operator of MDWFMixedPrecisionSolver.h) uses the float-safe copies of the
 * hopping term and gamma5 in MDWFFloatSafeWilson.h, and takes its clover blocks
 * and block-inverse matrices from a refreshed double operator (refreshFrom),
 * since the shared clover precompute only compiles in double.
 *
 * Correctness scaffold: single rank, no fused kernels.
 */

#pragma once

#include "MDWFSpinor.h"

#include "MDWFAdjointOperator.h"
#include "MDWFCoupledCG.h"
#include "MDWFCoupledSolverAdapter.h"
#include "MDWFFloatSafeWilson.h"
#include "MDWFMobiusMapping.h"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

template<class floatT>
__host__ __device__ inline Matrix6x6<floatT> mdwf6x6Mul(const Matrix6x6<floatT> &a, const Matrix6x6<floatT> &b) {
    Matrix6x6<floatT> out;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            COMPLEX(floatT) sum = 0.0;
            for (int k = 0; k < 6; k++) {
                sum += a.val[i][k] * b.val[k][j];
            }
            out.val[i][j] = sum;
        }
    }
    return out;
}

// scale * a + shift * 1
template<class floatT>
__host__ __device__ inline Matrix6x6<floatT> mdwf6x6Affine(const Matrix6x6<floatT> &a, floatT scale, floatT shift) {
    Matrix6x6<floatT> out;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            out.val[i][j] = scale * a.val[i][j];
        }
        out.val[i][i] += shift;
    }
    return out;
}

// log det of a Hermitian positive-definite 6x6 matrix (Gaussian elimination without pivoting).
template<class floatT>
__host__ __device__ inline double mdwf6x6HermitianLogDet(const Matrix6x6<floatT> &m) {
    COMPLEX(floatT) a[6][6];
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            a[i][j] = m.val[i][j];
        }
    }
    double logDet = 0.0;
    for (int i = 0; i < 6; i++) {
        logDet += log(static_cast<double>(real(a[i][i])));
        for (int j = i + 1; j < 6; j++) {
            const COMPLEX(floatT) factor = a[j][i] * (static_cast<floatT>(1.0) / a[i][i]);
            for (int k = i; k < 6; k++) {
                a[j][k] -= factor * a[i][k];
            }
        }
    }
    return logDet;
}

// Per odd site: log det M_oo(x) = sum_chi [Ls log det P_chi - log det W_chi] = sum_chi [-Ls log det P^-1 - log det W].
template<class floatT, size_t HaloDepthGauge, size_t Ls>
struct MDWFMooLogDetKernel {
    Vect18ArrayAcc<floatT> _pinvUpper, _pinvLower, _wUpper, _wLower;
    using Field = Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1>;

    MDWFMooLogDetKernel(const Field &pinvUpper, const Field &pinvLower, const Field &wUpper, const Field &wLower)
        : _pinvUpper(pinvUpper.getAccessor()), _pinvLower(pinvLower.getAccessor()),
          _wUpper(wUpper.getAccessor()), _wLower(wLower.getAccessor()) {}

    __host__ __device__ double operator()(gSite site) {
        const gSite allSite = GIndexer<Odd, HaloDepthGauge>::template convertSite<All, HaloDepthGauge>(site);
        Vect18<floatT> v = _pinvUpper.getElement(allSite);
        Matrix6x6<floatT> pu(v);
        v = _pinvLower.getElement(allSite);
        Matrix6x6<floatT> pl(v);
        v = _wUpper.getElement(allSite);
        Matrix6x6<floatT> wu(v);
        v = _wLower.getElement(allSite);
        Matrix6x6<floatT> wl(v);
        return -static_cast<double>(Ls) * (mdwf6x6HermitianLogDet(pu) + mdwf6x6HermitianLogDet(pl))
               - mdwf6x6HermitianLogDet(wu) - mdwf6x6HermitianLogDet(wl);
    }
};

// Per odd site: sum_chi log det W_chi (the mass-dependent part of log det M_oo).
template<class floatT, size_t HaloDepthGauge>
struct MDWFMooLogDetWKernel {
    Vect18ArrayAcc<floatT> _wUpper, _wLower;
    using Field = Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1>;

    MDWFMooLogDetWKernel(const Field &wUpper, const Field &wLower)
        : _wUpper(wUpper.getAccessor()), _wLower(wLower.getAccessor()) {}

    __host__ __device__ double operator()(gSite site) {
        const gSite allSite = GIndexer<Odd, HaloDepthGauge>::template convertSite<All, HaloDepthGauge>(site);
        Vect18<floatT> v = _wUpper.getElement(allSite);
        Matrix6x6<floatT> wu(v);
        v = _wLower.getElement(allSite);
        Matrix6x6<floatT> wl(v);
        return mdwf6x6HermitianLogDet(wu) + mdwf6x6HermitianLogDet(wl);
    }
};

/*
 * Column b of H(x) = G_W(m) - G_W(m'), G_W(m) = (-1)^Ls Ls R^(Ls-1) P^-1 (c5 - b5 R) m W(m), at s = 0
 * (other stacks zero), on the odd sites. d log det W(m)^-1 = Tr[G_W(m) dA] (R, P^-1, W commute as
 * functions of A); P and R do not depend on the boundary mass, only W and the explicit m do.
 */
template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
struct MDWFLogDetWDifferenceColumn {
    Vect18ArrayAcc<floatT> _pinvUpper, _pinvLower, _rUpper, _rLower, _wmUpper, _wmLower, _wpUpper, _wpLower;
    int _column;
    floatT _m, _mPrime, _b5, _c5;
    using Field = Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1>;

    MDWFLogDetWDifferenceColumn(int column, const Field &pinvUpper, const Field &pinvLower, const Field &rUpper,
                                const Field &rLower, const Field &wmUpper, const Field &wmLower,
                                const Field &wpUpper, const Field &wpLower, floatT m, floatT mPrime,
                                floatT b5, floatT c5)
        : _pinvUpper(pinvUpper.getAccessor()), _pinvLower(pinvLower.getAccessor()),
          _rUpper(rUpper.getAccessor()), _rLower(rLower.getAccessor()),
          _wmUpper(wmUpper.getAccessor()), _wmLower(wmLower.getAccessor()),
          _wpUpper(wpUpper.getAccessor()), _wpLower(wpLower.getAccessor()),
          _column(column), _m(m), _mPrime(mPrime), _b5(b5), _c5(c5) {}

    __host__ __device__ Matrix6x6<floatT> block(Vect18ArrayAcc<floatT> pinvAcc, Vect18ArrayAcc<floatT> rAcc,
                                                Vect18ArrayAcc<floatT> wmAcc, Vect18ArrayAcc<floatT> wpAcc,
                                                gSite allSite) {
        Vect18<floatT> v = pinvAcc.getElement(allSite);
        Matrix6x6<floatT> pinv(v);
        v = rAcc.getElement(allSite);
        Matrix6x6<floatT> R(v);
        v = wmAcc.getElement(allSite);
        Matrix6x6<floatT> wm(v);
        v = wpAcc.getElement(allSite);
        Matrix6x6<floatT> wp(v);
        Matrix6x6<floatT> rPower = mdwf6x6Affine(R, static_cast<floatT>(0.0), static_cast<floatT>(1.0));
        for (size_t n = 1; n < Ls; n++) {
            rPower = mdwf6x6Mul(rPower, R);
        }
        const floatT sign = (Ls % 2 == 0) ? static_cast<floatT>(1.0) : static_cast<floatT>(-1.0);
        Matrix6x6<floatT> left = mdwf6x6Mul(mdwf6x6Mul(rPower, pinv), mdwf6x6Affine(R, -_b5, _c5));
        Matrix6x6<floatT> diff;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                diff.val[i][j] = _m * wm.val[i][j] - _mPrime * wp.val[i][j];
            }
        }
        Matrix6x6<floatT> h = mdwf6x6Mul(left, diff);
        return mdwf6x6Affine(h, sign * static_cast<floatT>(Ls), static_cast<floatT>(0.0));
    }

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);
        if (site.stack != 0) {
            return out;
        }
        const gSite allSite = GIndexer<Odd, HaloDepthSpin>::template convertSite<All, HaloDepthGauge>(site);
        const int half = _column / 6;
        const int col = _column % 6;
        Matrix6x6<floatT> h = (half == 0) ? block(_pinvUpper, _rUpper, _wmUpper, _wpUpper, allSite)
                                          : block(_pinvLower, _rLower, _wmLower, _wpLower, allSite);
        for (int i = 0; i < 6; i++) {
            out.data[i + 6 * half] = h.val[i][col];
        }
        return out;
    }
};

// Unit vector (component, stack) on every site of the layout; component < 0 gives the zero field.
template<class floatT, Layout LatLayout, size_t HaloDepthSpin, size_t Ls>
struct MDWFUnitSliceField {
    int _component;
    size_t _stack;

    MDWFUnitSliceField(int component, size_t stack) : _component(component), _stack(stack) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        Vect12<floatT> out(0.0);
        if (_component >= 0 && site.stack == _stack) {
            out.data[_component] = COMPLEX(floatT)(1.0, 0.0);
        }
        return out;
    }
};

// Slice `stack` of the input copied to stack 0; all other stacks zero.
template<class floatT, Layout LatLayout, size_t HaloDepthSpin, size_t Ls>
struct MDWFSliceToStackZero {
    Vect12ArrayAcc<floatT> _in;
    size_t _stack;

    MDWFSliceToStackZero(const MDWFSpinor<floatT, true, LatLayout, HaloDepthSpin, Ls> &in, size_t stack)
        : _in(in.getAccessor()), _stack(stack) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        typedef GIndexer<LatLayout, HaloDepthSpin> GInd;
        if (site.stack != 0) {
            return Vect12<floatT>(0.0);
        }
        return _in.getElement(GInd::getSiteStack(site, _stack));
    }
};

// Precomputes P^-1, R = P^-1 Q, W = (1 + (-1)^Ls mf R^Ls)^-1 per site and chirality from the clover blocks A.
template<class floatT, size_t HaloDepthGauge, size_t Ls>
struct MDWFMooeeInverseSetup {
    Vect18ArrayAcc<floatT> _aUpper, _aLower;
    Vect18ArrayAcc<floatT> _pinvUpper, _pinvLower, _rUpper, _rLower, _wUpper, _wLower;
    floatT _b5, _c5, _mf;

    using Field = Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1>;

    MDWFMooeeInverseSetup(const Field &aUpper, const Field &aLower, Field &pinvUpper, Field &pinvLower,
                          Field &rUpper, Field &rLower, Field &wUpper, Field &wLower,
                          floatT b5, floatT c5, floatT mf)
        : _aUpper(aUpper.getAccessor()), _aLower(aLower.getAccessor()),
          _pinvUpper(pinvUpper.getAccessor()), _pinvLower(pinvLower.getAccessor()),
          _rUpper(rUpper.getAccessor()), _rLower(rLower.getAccessor()),
          _wUpper(wUpper.getAccessor()), _wLower(wLower.getAccessor()),
          _b5(b5), _c5(c5), _mf(mf) {}

    __host__ __device__ void block(Vect18ArrayAcc<floatT> a, Vect18ArrayAcc<floatT> pinvOut,
                                   Vect18ArrayAcc<floatT> rOut, Vect18ArrayAcc<floatT> wOut, gSite site) {
        Vect18<floatT> a18 = a.getElement(site);
        Matrix6x6<floatT> A(a18);
        Matrix6x6<floatT> P = mdwf6x6Affine(A, _b5, static_cast<floatT>(1.0));
        Matrix6x6<floatT> Q = mdwf6x6Affine(A, _c5, static_cast<floatT>(-1.0));
        Matrix6x6<floatT> Pinv = P.invert();
        Matrix6x6<floatT> R = mdwf6x6Mul(Pinv, Q);
        Matrix6x6<floatT> Rn = R;
        for (size_t n = 1; n < Ls; n++) {
            Rn = mdwf6x6Mul(Rn, R);
        }
        const floatT sign = (Ls % 2 == 0) ? static_cast<floatT>(1.0) : static_cast<floatT>(-1.0);
        Matrix6x6<floatT> Wm = mdwf6x6Affine(Rn, sign * _mf, static_cast<floatT>(1.0));
        Matrix6x6<floatT> W = Wm.invert();
        pinvOut.setElement(site, Pinv.ConvertHermitianToVect18());
        rOut.setElement(site, R.ConvertHermitianToVect18());
        wOut.setElement(site, W.ConvertHermitianToVect18());
    }

    __host__ __device__ void operator()(gSite site) {
        block(_aUpper, _pinvUpper, _rUpper, _wUpper, site);
        block(_aLower, _pinvLower, _rLower, _wLower, site);
    }
};

// out = M_xx^-1 in (Dagger = false) or M_xx^-dagger in (Dagger = true) on the sites of LatLayout.
template<class floatT, Layout LatLayout, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls, bool Dagger>
struct MDWFMooeeInverseKernel {
    Vect12ArrayAcc<floatT> _in;
    Vect12ArrayAcc<floatT> _out;
    Vect18ArrayAcc<floatT> _pinvUpper, _pinvLower, _rUpper, _rLower, _wUpper, _wLower;
    floatT _mf;

    typedef GIndexer<LatLayout, HaloDepthSpin> GInd;
    using Field = Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1>;
    using Spinor = MDWFSpinor<floatT, true, LatLayout, HaloDepthSpin, Ls>;

    MDWFMooeeInverseKernel(Spinor &out, const Spinor &in, const Field &pinvUpper, const Field &pinvLower,
                           const Field &rUpper, const Field &rLower, const Field &wUpper, const Field &wLower,
                           floatT mf)
        : _in(in.getAccessor()), _out(out.getAccessor()),
          _pinvUpper(pinvUpper.getAccessor()), _pinvLower(pinvLower.getAccessor()),
          _rUpper(rUpper.getAccessor()), _rLower(rLower.getAccessor()),
          _wUpper(wUpper.getAccessor()), _wLower(wLower.getAccessor()), _mf(mf) {}

    // Solves (1 + R K) x = z for one chirality (half = 0: upper, 1: lower); backward: (K x)(s) = x(s-1).
    __host__ __device__ void sweep(const Vect12<floatT> *z, Vect12<floatT> *x, Matrix6x6<floatT> &R,
                                   Matrix6x6<floatT> &W, int half, bool backward) {
        Vect12<floatT> zh[Ls];
        for (size_t s = 0; s < Ls; s++) {
            zh[s] = (half == 0) ? mdwfProjectPlus(z[s]) : mdwfProjectMinus(z[s]);
        }
        // Step t along the sweep direction is slice (backward ? t : Ls - 1 - t); t = 0 is the boundary slice.
        Vect12<floatT> u(0.0);
        for (size_t t = 1; t < Ls; t++) {
            u = zh[backward ? t : Ls - 1 - t] - R.MatrixXVect12UpDown(u, half);
        }
        const size_t s0 = backward ? 0 : Ls - 1;
        Vect12<floatT> first = zh[s0] + _mf * R.MatrixXVect12UpDown(u, half);
        Vect12<floatT> prev = W.MatrixXVect12UpDown(first, half);
        x[s0] += prev;
        for (size_t t = 1; t < Ls; t++) {
            const size_t st = backward ? t : Ls - 1 - t;
            Vect12<floatT> cur = zh[st] - R.MatrixXVect12UpDown(prev, half);
            x[st] += cur;
            prev = cur;
        }
    }

    __host__ __device__ void operator()(gSite site) {
        const gSite allSite = GInd::template convertSite<All, HaloDepthGauge>(site);
        Vect18<floatT> tmp = _pinvUpper.getElement(allSite);
        Matrix6x6<floatT> pinvUpper(tmp);
        tmp = _pinvLower.getElement(allSite);
        Matrix6x6<floatT> pinvLower(tmp);
        tmp = _rUpper.getElement(allSite);
        Matrix6x6<floatT> rUpper(tmp);
        tmp = _rLower.getElement(allSite);
        Matrix6x6<floatT> rLower(tmp);
        tmp = _wUpper.getElement(allSite);
        Matrix6x6<floatT> wUpper(tmp);
        tmp = _wLower.getElement(allSite);
        Matrix6x6<floatT> wLower(tmp);

        Vect12<floatT> z[Ls];
        Vect12<floatT> x[Ls];
        for (size_t s = 0; s < Ls; s++) {
            Vect12<floatT> v = _in.getElement(GInd::getSiteStack(site, s));
            v = pinvUpper.MatrixXVect12UpDown(v, 0);
            z[s] = pinvLower.MatrixXVect12UpDown(v, 1);
            x[s] = Vect12<floatT>(0.0);
        }
        // M: P_+ couples to s-1 (backward), P_- to s+1. M^dagger: exchanged.
        sweep(z, x, rUpper, wUpper, 0, !Dagger);
        sweep(z, x, rLower, wLower, 1, Dagger);
        for (size_t s = 0; s < Ls; s++) {
            _out.setElement(GInd::getSiteStack(site, s), x[s]);
        }
    }
};

// Dense c_sw = 0 block inverse (below) on unless MDWF_DENSE_MOOINV=0 (then the per-site sweep kernel, for checks).
inline bool mdwfDenseMooeeInverse() {
    static const bool on = [] {
        const char *env = std::getenv("MDWF_DENSE_MOOINV");
        return env == nullptr || std::string(env) != "0";
    }();
    return on;
}

/*
 * M_xx^-1 (or M_xx^-dagger) for c_sw = 0, where A = a 1 and the block inverse is the same Ls x Ls matrix per chirality
 * at every site: with p = b5 a + 1, r = (c5 a - 1) / p and boundary mass m of the chirality,
 * (1 + r K)^-1 = sum_k (-r)^k K^k / (1 + (-r)^Ls m), so
 *
 *   backward shape (P_+ of M, wrap mass m_+):  x(s) = sum_t c_+[(s - t) mod Ls] (t <= s ? 1 : -m_+) z(t),
 *   forward shape  (P_- of M, wrap mass m_-):  x(s) = sum_t c_-[(t - s) mod Ls] (t >= s ? 1 : -m_-) z(t),
 *
 * c_m[k] = (-r)^k / (p (1 + (-r)^Ls m)). The matrices are real, so M_xx^-dagger is the transpose: each chirality
 * keeps its coefficients and mass and takes the other shape. One thread per (site, s); the Ls input vectors of a site
 * are shared through the cache by its Ls threads. Replaces the per-site sweep kernel above (Ls-long per-thread
 * arrays, register spills) when there is no clover term; it also allows m_+ != m_- (EOFA operators).
 */
template<class floatT, Layout LatLayout, size_t HaloDepthSpin, size_t Ls, bool Dagger>
struct MDWFMooeeInverseDenseKernel {
    Vect12ArrayAcc<floatT> _in;
    Vect12ArrayAcc<floatT> _out;
    floatT _cPlus[Ls], _cMinus[Ls];
    floatT _mPlus, _mMinus;

    typedef GIndexer<LatLayout, HaloDepthSpin> GInd;
    using Spinor = MDWFSpinor<floatT, true, LatLayout, HaloDepthSpin, Ls>;

    MDWFMooeeInverseDenseKernel(Spinor &out, const Spinor &in, const double (&cPlus)[Ls], const double (&cMinus)[Ls],
                                double mPlus, double mMinus)
        : _in(in.getAccessor()), _out(out.getAccessor()), _mPlus(static_cast<floatT>(mPlus)),
          _mMinus(static_cast<floatT>(mMinus)) {
        for (size_t k = 0; k < Ls; k++) {
            _cPlus[k] = static_cast<floatT>(cPlus[k]);
            _cMinus[k] = static_cast<floatT>(cMinus[k]);
        }
    }

    __host__ __device__ void operator()(gSiteStack site) {
        const int s = static_cast<int>(site.stack);
        const int L = static_cast<int>(Ls);
        const floatT one = 1.0;
        Vect12<floatT> acc(0.0);
        for (int t = 0; t < L; t++) {
            const Vect12<floatT> z = _in.getElement(GInd::getSiteStack(static_cast<const gSite &>(site), t));
            const int back = (s - t + L) % L;   // backward shape index, factor (t <= s ? 1 : -m)
            const int fwd = (t - s + L) % L;    // forward shape index, factor (t >= s ? 1 : -m)
            const floatT up = Dagger ? _cPlus[fwd] * (t >= s ? one : -_mPlus) : _cPlus[back] * (t <= s ? one : -_mPlus);
            const floatT lo = Dagger ? _cMinus[back] * (t <= s ? one : -_mMinus)
                                     : _cMinus[fwd] * (t >= s ? one : -_mMinus);
            for (size_t i = 0; i < 6; i++) {
                acc.data[i] += up * z.data[i];
                acc.data[i + 6] += lo * z.data[i + 6];
            }
        }
        _out.setElement(site, acc);
    }
};

// Fused Schur operator for c_sw = 0 (MDWFMobiusCloverEvenOdd::schur): on unless MDWF_FUSED_SCHUR=0 (then the block
// path Mee - Meo MooInv Moe, for checks); settable at run time by tests.
inline bool &mdwfFusedSchur() {
    static bool on = [] {
        const char *env = std::getenv("MDWF_FUSED_SCHUR");
        return env == nullptr || std::string(env) != "0";
    }();
    return on;
}

/*
 * (K v)(s) and (K^T v)(s) on one site for the fifth-direction shift K of the Mobius operator (P_+ couples to s - 1
 * with wrap factor -m_+ at s = 0, P_- to s + 1 with wrap -m_- at s = Ls - 1); K^T exchanges the directions. Every
 * fifth-direction operator of the c_sw = 0 operator is a polynomial in K: Din = b5 + c5 K, M_xx = p + q K
 * (p = b5 a + 1, q = c5 a - 1), M_xx^-1 and Din M_xx^-1 (MDWFMooeeInverseDenseKernel coefficients).
 */
template<class floatT, Layout L, size_t HaloDepthSpin, size_t Ls, bool Transposed>
__host__ __device__ inline Vect12<floatT> mdwfShiftK(const Vect12ArrayAcc<floatT> &acc, const gSite &site, int s,
                                                    floatT mPlus, floatT mMinus) {
    typedef GIndexer<L, HaloDepthSpin> GInd;
    const int last = static_cast<int>(Ls) - 1;
    Vect12<floatT> out;
    if (!Transposed) {
        out = (s >= 1) ? mdwfProjectPlus(acc.getElement(GInd::getSiteStack(site, s - 1)))
                       : (-mPlus) * mdwfProjectPlus(acc.getElement(GInd::getSiteStack(site, last)));
        out += (s < last) ? mdwfProjectMinus(acc.getElement(GInd::getSiteStack(site, s + 1)))
                          : (-mMinus) * mdwfProjectMinus(acc.getElement(GInd::getSiteStack(site, 0)));
    } else {
        out = (s < last) ? mdwfProjectPlus(acc.getElement(GInd::getSiteStack(site, s + 1)))
                         : (-mPlus) * mdwfProjectPlus(acc.getElement(GInd::getSiteStack(site, 0)));
        out += (s >= 1) ? mdwfProjectMinus(acc.getElement(GInd::getSiteStack(site, s - 1)))
                        : (-mMinus) * mdwfProjectMinus(acc.getElement(GInd::getSiteStack(site, last)));
    }
    return out;
}

// dv = (b5 + c5 K) v and mv = (p + q K) v in one pass (Din v and M_xx v, c_sw = 0).
template<class floatT, Layout L, size_t HaloDepthSpin, size_t Ls>
struct MDWFFusedDinMxx {
    Vect12ArrayAcc<floatT> _v, _dv, _mv;
    floatT _b, _c, _p, _q, _mPlus, _mMinus;

    using Spinor = MDWFSpinor<floatT, true, L, HaloDepthSpin, Ls>;

    MDWFFusedDinMxx(Spinor &dv, Spinor &mv, const Spinor &v, floatT b, floatT c, floatT p, floatT q, floatT mPlus,
                    floatT mMinus)
        : _v(v.getAccessor()), _dv(dv.getAccessor()), _mv(mv.getAccessor()), _b(b), _c(c), _p(p), _q(q),
          _mPlus(mPlus), _mMinus(mMinus) {}

    __host__ __device__ void operator()(gSiteStack site) {
        const Vect12<floatT> v = _v.getElement(site);
        const Vect12<floatT> kv = mdwfShiftK<floatT, L, HaloDepthSpin, Ls, false>(
            _v, static_cast<const gSite &>(site), static_cast<int>(site.stack), _mPlus, _mMinus);
        _dv.setElement(site, _b * v + _c * kv);
        _mv.setElement(site, _p * v + _q * kv);
    }
};

// out = (p + q K^T) v - (b5 + c5 K^T) g5 w (M_xx^+ v - Din^+ g5 w, the last step of Mhat^+).
template<class floatT, Layout L, size_t HaloDepthSpin, size_t Ls>
struct MDWFFusedMxxDagMinusDinDagG5 {
    Vect12ArrayAcc<floatT> _v, _w;
    floatT _b, _c, _p, _q, _mPlus, _mMinus;

    using Spinor = MDWFSpinor<floatT, true, L, HaloDepthSpin, Ls>;

    MDWFFusedMxxDagMinusDinDagG5(const Spinor &v, const Spinor &w, floatT b, floatT c, floatT p, floatT q,
                                 floatT mPlus, floatT mMinus)
        : _v(v.getAccessor()), _w(w.getAccessor()), _b(b), _c(c), _p(p), _q(q), _mPlus(mPlus), _mMinus(mMinus) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        const gSite &s4 = static_cast<const gSite &>(site);
        const int s = static_cast<int>(site.stack);
        const Vect12<floatT> kv = mdwfShiftK<floatT, L, HaloDepthSpin, Ls, true>(_v, s4, s, _mPlus, _mMinus);
        const Vect12<floatT> kw = mdwfShiftK<floatT, L, HaloDepthSpin, Ls, true>(_w, s4, s, _mPlus, _mMinus);
        Vect12<floatT> dw = _b * _w.getElement(site) + _c * kw;   // Din^+ w; g5 commutes with it
        for (size_t i = 6; i < 12; i++) {
            dw.data[i] = -dw.data[i];
        }
        Vect12<floatT> out = _p * _v.getElement(site) + _q * kv;
        out -= dw;
        return out;
    }
};

/*
 * Wilson hopping block (the arithmetic of MDWFWilsonHopEvenOdd, -1/2 included), fused with what surrounds it in the
 * Schur operator: G5In applies gamma5 to the input spinors as they are read; Subtract returns base - Hop u.
 */
template<class floatT, Layout LOut, Layout LIn, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls, bool G5In,
         bool Subtract>
struct MDWFFusedHop {
    SU3Accessor<floatT> _U;
    SpinorColorAcc<floatT> _in;
    Vect12ArrayAcc<floatT> _base;

    typedef GIndexer<LOut, HaloDepthSpin> GInd;

    MDWFFusedHop(Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge,
                 const MDWFSpinor<floatT, true, LIn, HaloDepthSpin, Ls> &in,
                 const MDWFSpinor<floatT, true, LOut, HaloDepthSpin, Ls> &base)
        : _U(gauge.getAccessor()), _in(in.getAccessor()), _base(base.getAccessor()) {}

    __device__ __host__ ColorVect<floatT> read(const gSiteStack &site) {
        ColorVect<floatT> v = _in.getColorVect(site);
        if (G5In) {
            v = mdwfGamma5Mult(v);
        }
        return v;
    }

    __device__ __host__ Vect12<floatT> operator()(gSiteStack site) {
        ColorVect<floatT> outSC;
        outSC = static_cast<floatT>(0.0) * outSC;
        ColorVect<floatT> temp, temp2;
#define MDWF_FUSED_HOP_DIR(MU, GAMMA)                                                                                \
        temp = _U.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, MU)))                \
               * read(GInd::site_up(site, MU));                                                                       \
        temp2 = _U.getLinkDagger(                                                                                    \
                    GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, MU), MU)))   \
                * read(GInd::site_dn(site, MU));                                                                      \
        outSC = outSC - temp - temp2 + GAMMA(temp - temp2);
        MDWF_FUSED_HOP_DIR(0, mdwfGammaXMult)
        MDWF_FUSED_HOP_DIR(1, mdwfGammaYMult)
        MDWF_FUSED_HOP_DIR(2, mdwfGammaZMult)
        MDWF_FUSED_HOP_DIR(3, mdwfGammaTMult)
#undef MDWF_FUSED_HOP_DIR
        outSC = static_cast<floatT>(0.5) * outSC;
        Vect12<floatT> out = convertColorVectToVect12(outSC);
        if (Subtract) {
            Vect12<floatT> b = _base.getElement(site);
            b -= out;
            return b;
        }
        return out;
    }
};

// Even/odd part of an All-layout MDWF spinor, stack by stack.
template<class floatT, Layout LatLayout, size_t HaloDepthSpin, size_t Ls>
struct MDWFEvenOddExtract {
    Vect12ArrayAcc<floatT> _all;

    explicit MDWFEvenOddExtract(const MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls> &all)
        : _all(all.getAccessor()) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        typedef GIndexer<All, HaloDepthSpin> GIndAll;
        return _all.getElement(GIndAll::getSiteStack(GIndAll::getSite(site.coord), site.stack));
    }
};

// All-layout MDWF spinor from its even and odd parts, stack by stack.
template<class floatT, size_t HaloDepthSpin, size_t Ls>
struct MDWFEvenOddMerge {
    Vect12ArrayAcc<floatT> _even;
    Vect12ArrayAcc<floatT> _odd;

    MDWFEvenOddMerge(const MDWFSpinor<floatT, true, Even, HaloDepthSpin, Ls> &even,
                     const MDWFSpinor<floatT, true, Odd, HaloDepthSpin, Ls> &odd)
        : _even(even.getAccessor()), _odd(odd.getAccessor()) {}

    __host__ __device__ Vect12<floatT> operator()(gSiteStack site) {
        const sitexyzt c = site.coord;
        if ((c.x + c.y + c.z + c.t) % 2 == 0) {
            typedef GIndexer<Even, HaloDepthSpin> GIndE;
            return _even.getElement(GIndE::getSiteStack(GIndE::getSite(c), site.stack));
        }
        typedef GIndexer<Odd, HaloDepthSpin> GIndO;
        return _odd.getElement(GIndO::getSiteStack(GIndO::getSite(c), site.stack));
    }
};

template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
class MDWFMobiusCloverEvenOdd {
    template<class, size_t, size_t, size_t>
    friend class MDWFMobiusCloverEvenOdd;

    static constexpr bool isDouble = std::is_same<floatT, double>::value;

public:
    using Gauge = Gaugefield<floatT, true, HaloDepthGauge, R18>;
    using SpinorAll = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;
    using SpinorE = MDWFSpinor<floatT, true, Even, HaloDepthSpin, Ls>;
    using SpinorO = MDWFSpinor<floatT, true, Odd, HaloDepthSpin, Ls>;
    using Field = Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1>;

    static std::string sharedEvenName(const std::string &role) {
        return mdwfSharedSpinorName<floatT, Even, HaloDepthSpin, Ls>(role);
    }

    static std::string sharedOddName(const std::string &role) {
        return mdwfSharedSpinorName<floatT, Odd, HaloDepthSpin, Ls>(role);
    }

    static std::string sharedAllName(const std::string &role) {
        return mdwfSharedSpinorName<floatT, All, HaloDepthSpin, Ls>(role);
    }

private:
    Gauge &_gauge;
    MDWFMobiusOperatorParameters<floatT> _params;
    floatT _mf;        // boundary mass of P_+ (Grid's mass_plus); also the mass of the single-mass operator
    floatT _mfMinus;   // boundary mass of P_- (Grid's mass_minus); = _mf unless setMasses() made them differ
    floatT _csw;
    floatT _M5;
    Field _aUpper, _aLower, _aInvUpper, _aInvLower;
    Field _pinvUpper, _pinvLower, _rUpper, _rLower, _wUpper, _wLower;
    SpinorE _e1, _e2, _e3;
    SpinorO _o1, _o2, _o3;
    bool _fresh;

    // out = A Din in + Shift in (or its adjoint) on one parity; t1, t2 are same-parity work fields.
    template<Layout L>
    void diag(MDWFSpinor<floatT, true, L, HaloDepthSpin, Ls> &out,
              const MDWFSpinor<floatT, true, L, HaloDepthSpin, Ls> &in,
              MDWFSpinor<floatT, true, L, HaloDepthSpin, Ls> &t1,
              MDWFSpinor<floatT, true, L, HaloDepthSpin, Ls> &t2, bool dagger) {
        if (!dagger) {
            applyMDWFFifthDimCoupling<floatT, true, L, HaloDepthSpin, Ls>(t1, in, _params.dinCoeff);
            out.template iterateOverBulk<BLOCKSIZE>(
                DiracWilsonEvenEven2<floatT, L, HaloDepthGauge, HaloDepthSpin, Ls>(t1, _aUpper, _aLower));
            applyMDWFFifthDimCoupling<floatT, true, L, HaloDepthSpin, Ls>(t2, in, _params.shiftCoeff);
        } else {
            t1.template iterateOverBulk<BLOCKSIZE>(
                DiracWilsonEvenEven2<floatT, L, HaloDepthGauge, HaloDepthSpin, Ls>(in, _aUpper, _aLower));
            applyMDWFFifthDimAdjointCoupling<floatT, true, L, HaloDepthSpin, Ls>(out, t1, _params.dinCoeff);
            applyMDWFFifthDimAdjointCoupling<floatT, true, L, HaloDepthSpin, Ls>(t2, in, _params.shiftCoeff);
        }
        out += t2;
    }

    /*
     * Coefficients of M_xx^-1 = sum_k c[k] K^k per chirality (c_sw = 0): c[k] = (-r)^k / (p (1 + (-r)^Ls m)); with
     * withDin the coefficients of Din M_xx^-1 = (b5 + c5 K) M_xx^-1 (K^Ls = -m): e[k] = b5 c[k] + c5 c[k - 1],
     * e[0] = b5 c[0] - c5 m c[Ls - 1].
     */
    void inverseCoefficients(double (&cPlus)[Ls], double (&cMinus)[Ls], bool withDin) const {
        const double a = static_cast<double>(_params.mass);
        const double b = static_cast<double>(_params.b5);
        const double c = static_cast<double>(_params.c5);
        const double p = b * a + 1.0;
        const double r = (c * a - 1.0) / p;
        const double mPlus = static_cast<double>(_mf);
        const double mMinus = static_cast<double>(_mfMinus);
        double rk = 1.0;
        for (size_t k = 0; k < Ls; k++) {
            cPlus[k] = rk;
            cMinus[k] = rk;
            rk *= -r;
        }
        const double normPlus = 1.0 / (p * (1.0 + rk * mPlus));   // rk = (-r)^Ls
        const double normMinus = 1.0 / (p * (1.0 + rk * mMinus));
        for (size_t k = 0; k < Ls; k++) {
            cPlus[k] *= normPlus;
            cMinus[k] *= normMinus;
        }
        if (withDin) {
            double ePlus[Ls], eMinus[Ls];
            ePlus[0] = b * cPlus[0] - c * mPlus * cPlus[Ls - 1];
            eMinus[0] = b * cMinus[0] - c * mMinus * cMinus[Ls - 1];
            for (size_t k = 1; k < Ls; k++) {
                ePlus[k] = b * cPlus[k] + c * cPlus[k - 1];
                eMinus[k] = b * cMinus[k] + c * cMinus[k - 1];
            }
            for (size_t k = 0; k < Ls; k++) {
                cPlus[k] = ePlus[k];
                cMinus[k] = eMinus[k];
            }
        }
    }

    // out = sum_k c[k] K^k in per chirality (transpose with dagger), MDWFMooeeInverseDenseKernel.
    template<Layout L>
    void denseApply(MDWFSpinor<floatT, true, L, HaloDepthSpin, Ls> &out,
                    const MDWFSpinor<floatT, true, L, HaloDepthSpin, Ls> &in, const double (&cPlus)[Ls],
                    const double (&cMinus)[Ls], bool dagger) {
        const size_t elems = GIndexer<L, HaloDepthSpin>::getLatData().sizeh;
        const double mPlus = static_cast<double>(_mf);
        const double mMinus = static_cast<double>(_mfMinus);
        CalcGSiteStack<L, HaloDepthSpin> calcSiteStack;
        static_assert(Ls <= DEFAULT_NBLOCKS, "dense M_xx^-1 kernel: one block holds all Ls slices of a site");
        constexpr size_t block = (DEFAULT_NBLOCKS / Ls) * Ls;   // performFunctorNoReturn launch bound
        if (dagger) {
            iterateFunctorNoReturn<true, block>(
                MDWFMooeeInverseDenseKernel<floatT, L, HaloDepthSpin, Ls, true>(out, in, cPlus, cMinus, mPlus, mMinus),
                calcSiteStack, elems, Ls);
        } else {
            iterateFunctorNoReturn<true, block>(
                MDWFMooeeInverseDenseKernel<floatT, L, HaloDepthSpin, Ls, false>(out, in, cPlus, cMinus, mPlus,
                                                                                 mMinus),
                calcSiteStack, elems, Ls);
        }
    }

    /*
     * Mhat (or Mhat^+) for c_sw = 0 in four passes, with E = Din M_oo^-1 (one dense fifth-direction matrix) and
     * gamma5 Hop_eo gamma5 = Hop_oe^+ (g5 commutes with every fifth-direction operator):
     *   Mhat v   = M_ee v - Hop_eo E Hop_oe Din v:            (Din v, M_ee v), Hop_oe, E, M_ee v - Hop_eo (.)
     *   Mhat^+ v = M_ee^+ v - Din^+ g5 Hop_eo E^T Hop_oe g5 v:  Hop_oe g5, E^T, Hop_eo, M_ee^+ v - Din^+ g5 (.)
     * Work fields _e1, _e2 (even), _o2, _o3 (odd); out must not alias in.
     */
    void fusedSchur(SpinorE &out, const SpinorE &in, bool dagger) {
        const floatT a = _params.mass;
        const floatT b = _params.b5;
        const floatT c = _params.c5;
        const floatT p = b * a + static_cast<floatT>(1.0);
        const floatT q = c * a - static_cast<floatT>(1.0);
        double ePlus[Ls], eMinus[Ls];
        inverseCoefficients(ePlus, eMinus, true);
        const size_t elems = GIndexer<Even, HaloDepthSpin>::getLatData().sizeh;
        constexpr size_t block = (DEFAULT_NBLOCKS / Ls) * Ls;
        if (!dagger) {
            CalcGSiteStack<Even, HaloDepthSpin> calcSiteStack;
            iterateFunctorNoReturn<true, block>(
                MDWFFusedDinMxx<floatT, Even, HaloDepthSpin, Ls>(_e1, _e2, in, b, c, p, q, _mf, _mfMinus),
                calcSiteStack, elems, Ls);
            mdwfOverlapHop<BLOCKSIZE>(_o2, _e1,
                MDWFFusedHop<floatT, Odd, Even, HaloDepthGauge, HaloDepthSpin, Ls, false, false>(_gauge, _e1, _o2), MDWFHaloHalf);
            denseApply<Odd>(_o3, _o2, ePlus, eMinus, false);
            mdwfOverlapHop<BLOCKSIZE>(out, _o3,
                MDWFFusedHop<floatT, Even, Odd, HaloDepthGauge, HaloDepthSpin, Ls, false, true>(_gauge, _o3, _e2), MDWFHaloHalf);
        } else {
            mdwfOverlapHop<BLOCKSIZE>(_o2, const_cast<SpinorE &>(in),
                MDWFFusedHop<floatT, Odd, Even, HaloDepthGauge, HaloDepthSpin, Ls, true, false>(_gauge, in, _o2), MDWFHaloHalfG5);
            denseApply<Odd>(_o3, _o2, ePlus, eMinus, true);
            mdwfOverlapHop<BLOCKSIZE>(_e1, _o3,
                MDWFFusedHop<floatT, Even, Odd, HaloDepthGauge, HaloDepthSpin, Ls, false, false>(_gauge, _o3, _e1), MDWFHaloHalf);
            out.template iterateOverBulk<BLOCKSIZE>(
                MDWFFusedMxxDagMinusDinDagG5<floatT, Even, HaloDepthSpin, Ls>(in, _e1, b, c, p, q, _mf, _mfMinus));
        }
    }

    template<Layout L>
    void diagInverse(MDWFSpinor<floatT, true, L, HaloDepthSpin, Ls> &out,
                     const MDWFSpinor<floatT, true, L, HaloDepthSpin, Ls> &in, bool dagger) {
        CalcGSite<L, HaloDepthSpin> calcGSite;
        const size_t elems = GIndexer<L, HaloDepthSpin>::getLatData().sizeh;
        if (_csw == static_cast<floatT>(0.0) && mdwfDenseMooeeInverse()) {
            // No clover term: the same Ls x Ls inverse at every site (MDWFMooeeInverseDenseKernel).
            double cPlus[Ls], cMinus[Ls];
            inverseCoefficients(cPlus, cMinus, false);
            denseApply<L>(out, in, cPlus, cMinus, dagger);
            return;
        }
        if (_mf != _mfMinus) {
            throw std::runtime_error(stdLogger.fatal("MDWF even/odd: different boundary masses per chirality need "
                                                     "c_sw = 0 and the dense M_oo^-1 kernel"));
        }
        if (dagger) {
            iterateFunctorNoReturn<true, BLOCKSIZE>(
                MDWFMooeeInverseKernel<floatT, L, HaloDepthGauge, HaloDepthSpin, Ls, true>(
                    out, in, _pinvUpper, _pinvLower, _rUpper, _rLower, _wUpper, _wLower, _mf),
                calcGSite, elems);
        } else {
            iterateFunctorNoReturn<true, BLOCKSIZE>(
                MDWFMooeeInverseKernel<floatT, L, HaloDepthGauge, HaloDepthSpin, Ls, false>(
                    out, in, _pinvUpper, _pinvLower, _rUpper, _rLower, _wUpper, _wLower, _mf),
                calcGSite, elems);
        }
    }

    // Hopping block from parity LIn to parity LOut. Non-dagger: out = Hop_{LOut,LIn} Din in (= M_{LOut,LIn} in).
    // Dagger: out = Din^dagger gamma5 Hop_{LOut,LIn} gamma5 in (= (M_{LIn,LOut})^dagger in).
    // tIn / tOut are work fields of parity LIn / LOut.
    template<Layout LOut, Layout LIn>
    void hop(MDWFSpinor<floatT, true, LOut, HaloDepthSpin, Ls> &out,
             const MDWFSpinor<floatT, true, LIn, HaloDepthSpin, Ls> &in,
             MDWFSpinor<floatT, true, LIn, HaloDepthSpin, Ls> &tIn,
             MDWFSpinor<floatT, true, LOut, HaloDepthSpin, Ls> &tOut, bool dagger) {
        if (!dagger) {
            applyMDWFFifthDimCoupling<floatT, true, LIn, HaloDepthSpin, Ls>(tIn, in, _params.dinCoeff, true);
            wilsonHop<LOut, LIn>(out, tIn);
        } else {
            gamma5Into<LIn>(tIn, in, true);
            wilsonHop<LOut, LIn>(tOut, tIn);
            gamma5Into<LOut>(out, tOut, false);
            tOut = out;
            applyMDWFFifthDimAdjointCoupling<floatT, true, LOut, HaloDepthSpin, Ls>(out, tOut, _params.dinCoeff);
        }
    }

    // Wilson hopping block (DiracWilsonEvenOdd2 in double, its float-safe copy otherwise).
    template<Layout LOut, Layout LIn>
    void wilsonHop(MDWFSpinor<floatT, true, LOut, HaloDepthSpin, Ls> &out,
                   const MDWFSpinor<floatT, true, LIn, HaloDepthSpin, Ls> &in) {
        if constexpr (isDouble) {
            out.template iterateOverBulk<BLOCKSIZE>(
                DiracWilsonEvenOdd2<floatT, LOut, LIn, HaloDepthGauge, HaloDepthSpin, Ls, false>(_gauge, in, 0.0, 0.0));
        } else {
            out.template iterateOverBulk<BLOCKSIZE>(
                MDWFWilsonHopEvenOdd<floatT, LOut, LIn, HaloDepthGauge, HaloDepthSpin, Ls>(_gauge, in));
        }
    }

    template<Layout L>
    void gamma5Into(MDWFSpinor<floatT, true, L, HaloDepthSpin, Ls> &out,
                    const MDWFSpinor<floatT, true, L, HaloDepthSpin, Ls> &in, bool update) {
        if constexpr (isDouble) {
            applyMDWFGamma5<floatT, true, L, HaloDepthSpin, Ls>(out, in, update);
        } else {
            out.template iterateOverBulk<BLOCKSIZE>(MDWFGamma5Functor<floatT, L, HaloDepthSpin, Ls>(in));
            if (update) {
                mdwfUpdateHalo(out);
            }
        }
    }

    void requireFresh() const {
        if (!_fresh) {
            throw std::runtime_error(stdLogger.fatal("MDWF even/odd operator used before refresh()"));
        }
    }

public:
    MDWFMobiusCloverEvenOdd(Gauge &gauge, floatT M5, floatT mf, floatT b5, floatT csw, const std::string &name)
        : _gauge(gauge),
          _params(M5, mf, b5),
          _mf(mf),
          _mfMinus(mf),
          _csw(csw),
          _M5(M5),
          _aUpper(gauge.getComm(), name + "_a_up"),
          _aLower(gauge.getComm(), name + "_a_lo"),
          _aInvUpper(gauge.getComm(), name + "_ainv_up"),
          _aInvLower(gauge.getComm(), name + "_ainv_lo"),
          _pinvUpper(gauge.getComm(), name + "_pinv_up"),
          _pinvLower(gauge.getComm(), name + "_pinv_lo"),
          _rUpper(gauge.getComm(), name + "_r_up"),
          _rLower(gauge.getComm(), name + "_r_lo"),
          _wUpper(gauge.getComm(), name + "_w_up"),
          _wLower(gauge.getComm(), name + "_w_lo"),
          // Work fields hold nothing between calls: one set shared by all even/odd operators of this type.
          _e1(gauge.getComm(), mdwfSharedSpinorName<floatT, Even, HaloDepthSpin, Ls>("eo_work1")),
          _e2(gauge.getComm(), mdwfSharedSpinorName<floatT, Even, HaloDepthSpin, Ls>("eo_work2")),
          _e3(gauge.getComm(), mdwfSharedSpinorName<floatT, Even, HaloDepthSpin, Ls>("eo_work3")),
          _o1(gauge.getComm(), mdwfSharedSpinorName<floatT, Odd, HaloDepthSpin, Ls>("eo_work1")),
          _o2(gauge.getComm(), mdwfSharedSpinorName<floatT, Odd, HaloDepthSpin, Ls>("eo_work2")),
          _o3(gauge.getComm(), mdwfSharedSpinorName<floatT, Odd, HaloDepthSpin, Ls>("eo_work3")),
          _fresh(false) {}

    // Recomputes the clover blocks A and the block-inverse matrices from the current gauge field.
    void refresh() {
        if constexpr (isDouble) {
            typedef GIndexer<All, HaloDepthGauge> GInd;
            CalcGSite<All, HaloDepthGauge> calcGSite;
            const size_t elems = GInd::getLatData().vol4;
            iterateFunctorNoReturn<true, BLOCKSIZE>(
                preCalcFmunu<floatT, HaloDepthGauge>(_gauge, _aUpper, _aLower, _aInvUpper, _aInvLower,
                                                     _params.mass, _csw),
                calcGSite, elems);
            iterateFunctorNoReturn<true, BLOCKSIZE>(
                MDWFMooeeInverseSetup<floatT, HaloDepthGauge, Ls>(_aUpper, _aLower, _pinvUpper, _pinvLower,
                                                                  _rUpper, _rLower, _wUpper, _wLower,
                                                                  _params.b5, _params.c5, _mf),
                calcGSite, elems);
            _fresh = true;
        } else {
            throw std::runtime_error(stdLogger.fatal("MDWF even/odd: the single-precision operator is set up with "
                                                     "refreshFrom(a refreshed double operator)"));
        }
    }

    /*
     * Clover blocks and block-inverse matrices converted from a refreshed operator of the other precision; the mass
     * parameters (M5, mf, b5, c_sw) are taken over as well, so one operator can follow several source operators
     * (one single-precision operator for all masses of a Hasenbusch ladder). The gauge field of this operator must
     * hold the same links, converted.
     */
    template<class srcFloatT>
    void refreshFrom(MDWFMobiusCloverEvenOdd<srcFloatT, HaloDepthGauge, HaloDepthSpin, Ls> &src) {
        src.requireFresh();
        _M5 = static_cast<floatT>(src._M5);
        _mf = static_cast<floatT>(src._mf);
        _mfMinus = static_cast<floatT>(src._mfMinus);
        _csw = static_cast<floatT>(src._csw);
        _params = MDWFMobiusOperatorParameters<floatT>(_M5, _mf, static_cast<floatT>(src._params.b5), _mfMinus);
        _aUpper.convert_precision(src._aUpper);
        _aLower.convert_precision(src._aLower);
        _aInvUpper.convert_precision(src._aInvUpper);
        _aInvLower.convert_precision(src._aInvLower);
        _pinvUpper.convert_precision(src._pinvUpper);
        _pinvLower.convert_precision(src._pinvLower);
        _rUpper.convert_precision(src._rUpper);
        _rLower.convert_precision(src._rLower);
        _wUpper.convert_precision(src._wUpper);
        _wLower.convert_precision(src._wLower);
        _fresh = true;
    }

    const MDWFMobiusOperatorParameters<floatT> &params() const {
        return _params;
    }

    /*
     * Boundary masses m_+ (P_+, Grid's mass_plus) and m_- (P_-, mass_minus); m_+ != m_- needs c_sw = 0 (EOFA
     * operators). Call refresh() (or refreshFrom) before the next use.
     */
    void setMasses(floatT mPlus, floatT mMinus) {
        _mf = mPlus;
        _mfMinus = mMinus;
        _params = MDWFMobiusOperatorParameters<floatT>(_M5, mPlus, _params.b5, mMinus);
        _fresh = false;
    }

    floatT massPlus() const {
        return _mf;
    }

    floatT massMinus() const {
        return _mfMinus;
    }

    Gauge &gauge() const {
        return _gauge;
    }

    // Blocks (all inputs need no halo; outputs of the hopping blocks are halo-updated by the next hop).
    void Mee(SpinorE &out, const SpinorE &in, bool dagger = false) { requireFresh(); diag<Even>(out, in, _e1, _e2, dagger); }
    void Moo(SpinorO &out, const SpinorO &in, bool dagger = false) { requireFresh(); diag<Odd>(out, in, _o1, _o2, dagger); }
    void MooInv(SpinorO &out, const SpinorO &in, bool dagger = false) { requireFresh(); diagInverse<Odd>(out, in, dagger); }
    void MeeInv(SpinorE &out, const SpinorE &in, bool dagger = false) { requireFresh(); diagInverse<Even>(out, in, dagger); }

    // M_eo: odd -> even; its adjoint M_eo^dagger: even -> odd.
    void Meo(SpinorE &out, const SpinorO &in) { requireFresh(); hop<Even, Odd>(out, in, _o1, _e1, false); }
    void MeoDagger(SpinorO &out, const SpinorE &in) { requireFresh(); hop<Odd, Even>(out, in, _e1, _o1, true); }
    // M_oe: even -> odd; its adjoint M_oe^dagger: odd -> even.
    void Moe(SpinorO &out, const SpinorE &in) { requireFresh(); hop<Odd, Even>(out, in, _e1, _o1, false); }
    void MoeDagger(SpinorE &out, const SpinorO &in) { requireFresh(); hop<Even, Odd>(out, in, _o1, _e1, true); }

    // Mhat = M_ee - M_eo M_oo^-1 M_oe, or its adjoint.
    void schur(SpinorE &out, const SpinorE &in, bool dagger = false) {
        requireFresh();
        if (_csw == static_cast<floatT>(0.0) && mdwfDenseMooeeInverse() && mdwfFusedSchur()) {
            fusedSchur(out, in, dagger);
            return;
        }
        if (!dagger) {
            Moe(_o2, in);
            MooInv(_o3, _o2);
            Meo(_e3, _o3);
            Mee(out, in);
        } else {
            MeoDagger(_o2, in);
            MooInv(_o3, _o2, true);
            MoeDagger(_e3, _o3);
            Mee(out, in, true);
        }
        out.template axpyThisB<64>(static_cast<floatT>(-1.0), _e3);
    }

    // log det M_oo, summed over the odd sites (real: P, W are Hermitian positive definite).
    double logDetMoo() {
        requireFresh();
        typedef GIndexer<Odd, HaloDepthGauge> GIndO;
        const size_t elems = GIndO::getLatData().sizeh;
        LatticeContainer<true, double> reduction(_gauge.getComm());
        reduction.adjustSize(elems);
        reduction.template iterateOverBulk<Odd, HaloDepthGauge>(
            MDWFMooLogDetKernel<floatT, HaloDepthGauge, Ls>(_pinvUpper, _pinvLower, _wUpper, _wLower));
        double result = 0.0;
        reduction.reduce(result, elems);
        return result;
    }

    // sum over the odd sites of log det W (log det M_oo = sum [Ls log det P - log det W]; P is mass independent).
    double logDetW() {
        requireFresh();
        typedef GIndexer<Odd, HaloDepthGauge> GIndO;
        const size_t elems = GIndO::getLatData().sizeh;
        LatticeContainer<true, double> reduction(_gauge.getComm());
        reduction.adjustSize(elems);
        reduction.template iterateOverBulk<Odd, HaloDepthGauge>(
            MDWFMooLogDetWKernel<floatT, HaloDepthGauge>(_wUpper, _wLower));
        double result = 0.0;
        reduction.reduce(result, elems);
        return result;
    }

    /*
     * Columns h[b](x, s = 0) = H(x) e_b of H = G_W(m) - G_W(m') for two operators that differ only in the
     * boundary mass (same gauge field, M5, b5, c_sw): d[log det W(m)^-1 - log det W(m')^-1] = sum_x Tr[H dA].
     */
    static void logDetWDifferenceColumns(MDWFMobiusCloverEvenOdd &m, MDWFMobiusCloverEvenOdd &mPrime,
                                         std::vector<std::unique_ptr<SpinorO>> &h) {
        m.requireFresh();
        mPrime.requireFresh();
        if (h.size() != 12 || m._params.b5 != mPrime._params.b5 || m._params.mass != mPrime._params.mass
            || m._csw != mPrime._csw) {
            throw std::runtime_error(stdLogger.fatal("MDWF logDetWDifferenceColumns: needs 12 columns and two "
                                                     "operators differing only in the boundary mass"));
        }
        for (int b = 0; b < 12; b++) {
            h[b]->template iterateOverBulk<BLOCKSIZE>(MDWFLogDetWDifferenceColumn<floatT, HaloDepthGauge, HaloDepthSpin, Ls>(
                b, m._pinvUpper, m._pinvLower, m._rUpper, m._rLower, m._wUpper, m._wLower, mPrime._wUpper,
                mPrime._wLower, m._mf, mPrime._mf, m._params.b5, m._params.c5));
            mdwfUpdateHalo(*h[b]);
        }
    }

    /*
     * Columns of G(x) = sum_s [M_oo^-1 Din]_ss (a 12x12 matrix per odd site, block diagonal in
     * chirality): g[b](x, s = 0) = G(x) e_b, other stacks zero, so that
     * d log det M_oo = Tr[M_oo^-1 dA Din] = sum_x Tr[G(x) dA(x)] = sum_b sum_x e_b^+ dA(x) g_b(x)
     * (A and Din commute). Built from unit fields with the block inverse: 12 Ls applications.
     */
    void logDetMooColumns(std::vector<std::unique_ptr<SpinorO>> &g, SpinorO &unit, SpinorO &work1, SpinorO &work2) {
        requireFresh();
        if (g.size() != 12) {
            throw std::runtime_error(stdLogger.fatal("MDWF logDetMooColumns needs 12 column fields"));
        }
        for (int b = 0; b < 12; b++) {
            g[b]->template iterateOverBulk<BLOCKSIZE>(MDWFUnitSliceField<floatT, Odd, HaloDepthSpin, Ls>(-1, 0));
            for (size_t s0 = 0; s0 < Ls; s0++) {
                unit.template iterateOverBulk<BLOCKSIZE>(MDWFUnitSliceField<floatT, Odd, HaloDepthSpin, Ls>(b, s0));
                applyMDWFFifthDimCoupling<floatT, true, Odd, HaloDepthSpin, Ls>(work1, unit, _params.dinCoeff);
                diagInverse<Odd>(work2, work1, false);
                unit.template iterateOverBulk<BLOCKSIZE>(MDWFSliceToStackZero<floatT, Odd, HaloDepthSpin, Ls>(work2, s0));
                *g[b] += unit;
            }
            mdwfUpdateHalo(*g[b]);
        }
    }

    // All <-> even/odd.
    static void split(SpinorE &even, SpinorO &odd, const SpinorAll &all) {
        even.template iterateOverBulk<BLOCKSIZE>(MDWFEvenOddExtract<floatT, Even, HaloDepthSpin, Ls>(all));
        odd.template iterateOverBulk<BLOCKSIZE>(MDWFEvenOddExtract<floatT, Odd, HaloDepthSpin, Ls>(all));
    }

    static void merge(SpinorAll &all, const SpinorE &even, const SpinorO &odd) {
        all.template iterateOverBulk<BLOCKSIZE>(MDWFEvenOddMerge<floatT, HaloDepthSpin, Ls>(even, odd));
        mdwfUpdateHalo(all);
    }
};

// Mhat^dagger Mhat on the even sites, in the shape MDWFCoupledSolverAdapter / MDWFCoupledCG expect; with setMMdag(true)
// Mhat Mhat^dagger instead (GPT's light ratio form, MDWFHmcParameters::light_form = 1).
template<class EvenOdd>
class MDWFMobiusSchurNormalOperator {
public:
    using Spinor = typename EvenOdd::SpinorE;
    // schur() never reads the halo of its input: the hops read work fields (fused: _e1, or in through mdwfOverlapHop,
    // which exchanges it; block path: tIn written with a halo update); M_ee is site local.
    static constexpr bool readsInputHalo = false;

private:
    EvenOdd &_eo;
    Spinor _tmp;
    bool _mmdag = false;

public:
    // _tmp is shared by all Schur normal operators of this type (it holds nothing between calls); name unused.
    MDWFMobiusSchurNormalOperator(EvenOdd &eo, CommunicationBase &comm, const std::string &name)
        : _eo(eo), _tmp(comm, EvenOdd::sharedEvenName("schur_normal_tmp")) {
        (void)name;
    }

    void setMMdag(bool mmdag) {
        _mmdag = mmdag;
    }

    bool mmdag() const {
        return _mmdag;
    }

    void apply(Spinor &out, const Spinor &in, bool update = false) {
        _eo.schur(_tmp, in, _mmdag);    // Mhat x (or Mhat^+ x), then
        _eo.schur(out, _tmp, !_mmdag);  // Mhat^+ (Mhat x) (or Mhat (Mhat^+ x))
        if (update) {
            mdwfUpdateHalo(out);
        }
    }
};

/*
 * Grid's / GPT's second Schur form on the odd sites (Grid SchurDiagTwoOperator, GPT schur_complement_two with
 * parity odd, as in GPT's eo2_ne solver used for EOFA), in normal-equation form for MDWFCoupledCG:
 *   N = 1 - M_oe M_ee^-1 M_eo M_oo^-1,  apply = N^dagger N.
 * With setDagger(true) it is the same form of M^dagger: N' = 1 - M_eo^dagger M_ee^-dagger M_oe^dagger M_oo^-dagger.
 * det M = det M_ee det M_oo det N, the same determinant as Mhat up to the gauge-independent det M_oo.
 */
template<class EvenOdd>
class MDWFMobiusSchurTwoNormalOperator {
public:
    using Spinor = typename EvenOdd::SpinorO;
    using SpinorE = typename EvenOdd::SpinorE;
    // M_oo^-1 is site local and the hops read their own work fields (written with a halo update).
    static constexpr bool readsInputHalo = false;

private:
    EvenOdd &_eo;
    Spinor _o1, _o2, _n;
    SpinorE _e1, _e2;
    bool _dagger = false;

public:
    // The work fields are shared by all operators of this type (they hold nothing between calls).
    MDWFMobiusSchurTwoNormalOperator(EvenOdd &eo, CommunicationBase &comm)
        : _eo(eo), _o1(comm, EvenOdd::sharedOddName("schur2_o1")), _o2(comm, EvenOdd::sharedOddName("schur2_o2")),
          _n(comm, EvenOdd::sharedOddName("schur2_n")), _e1(comm, EvenOdd::sharedEvenName("schur2_e1")),
          _e2(comm, EvenOdd::sharedEvenName("schur2_e2")) {}

    void setDagger(bool dagger) {
        _dagger = dagger;
    }

    // out = N in, or N^dagger in (adjoint), of M (setDagger(false)) or of M^dagger (setDagger(true)); out must not be
    // a work field of this class.
    void schurTwo(Spinor &out, const Spinor &in, bool adjoint) {
        if (!_dagger && !adjoint) {         // N = 1 - M_oe M_ee^-1 M_eo M_oo^-1
            _eo.MooInv(_o1, in);
            _eo.Meo(_e1, _o1);
            _eo.MeeInv(_e2, _e1);
            _eo.Moe(_o2, _e2);
        } else if (!_dagger && adjoint) {   // N^dagger = 1 - M_oo^-dagger M_eo^dagger M_ee^-dagger M_oe^dagger
            _eo.MoeDagger(_e1, in);
            _eo.MeeInv(_e2, _e1, true);
            _eo.MeoDagger(_o1, _e2);
            _eo.MooInv(_o2, _o1, true);
        } else if (_dagger && !adjoint) {   // N' = 1 - M_eo^dagger M_ee^-dagger M_oe^dagger M_oo^-dagger
            _eo.MooInv(_o1, in, true);
            _eo.MoeDagger(_e1, _o1);
            _eo.MeeInv(_e2, _e1, true);
            _eo.MeoDagger(_o2, _e2);
        } else {                            // N'^dagger = 1 - M_oo^-1 M_oe M_ee^-1 M_eo
            _eo.Meo(_e1, in);
            _eo.MeeInv(_e2, _e1);
            _eo.Moe(_o1, _e2);
            _eo.MooInv(_o2, _o1);
        }
        out = in;
        out.template axpyThisB<64>(static_cast<typename MDWFSpinorTraits<Spinor>::floatT>(-1.0), _o2);
    }

    void apply(Spinor &out, const Spinor &in, bool update = false) {
        schurTwo(_n, in, false);
        schurTwo(out, _n, true);
        if (update) {
            mdwfUpdateHalo(out);
        }
    }
};

/*
 * Even/odd preconditioned solves with the Mobius clover operator.
 *
 * solve(x, b), M x = b:
 *   bhat_e = b_e - M_eo M_oo^-1 b_o,  Mhat^dagger Mhat x_e = Mhat^dagger bhat_e (CG),
 *   x_o = M_oo^-1 (b_o - M_oe x_e).
 * solveDagger(y, b), M^dagger y = b (M^dagger has Schur complement Mhat^dagger):
 *   bhat_e = b_e - M_oe^dagger M_oo^-dagger b_o,  Mhat^dagger Mhat w = bhat_e (CG), y_e = Mhat w,
 *   y_o = M_oo^-dagger (b_o - M_eo^dagger y_e).
 * solveNormal(x, b), M^dagger M x = b: solveDagger then solve.
 *
 * setSchurTwo(true): Grid's / GPT's second Schur form on the odd sites instead (GPT eo2_ne = normal_equation(
 * schur_complement_two(parity odd)), GPT's EOFA solver), N = 1 - M_oe M_ee^-1 M_eo M_oo^-1:
 *   r_o = b_o - M_oe M_ee^-1 b_e,  N^dagger N y = N^dagger r_o (CG, zero guess),  x_o = M_oo^-1 y,
 *   x_e = M_ee^-1 (b_e - M_eo x_o);
 * solveDagger the same with the blocks of M^dagger (MDWFMobiusSchurTwoNormalOperator::setDagger). Same solution to
 * the tolerance, and the same CG iterations as GPT's EOFA solves.
 *
 * The CG precision is the relative residual of the even-site normal equation;
 * trueResidual() recomputes |M x - b| / |b| with the unpreconditioned operator
 * the caller supplies, for validation.
 */
template<class floatT, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t Ls>
class MDWFMobiusEvenOddSolver {
public:
    using EvenOdd = MDWFMobiusCloverEvenOdd<floatT, HaloDepthGauge, HaloDepthSpin, Ls>;
    using NormalOp = MDWFMobiusSchurNormalOperator<EvenOdd>;
    using Adapter = MDWFCoupledSolverAdapter<floatT, HaloDepthGauge, HaloDepthSpin, Ls, NormalOp>;
    using CG = MDWFCoupledCG<floatT, Adapter>;
    using NormalTwoOp = MDWFMobiusSchurTwoNormalOperator<EvenOdd>;
    using AdapterTwo = MDWFCoupledSolverAdapter<floatT, HaloDepthGauge, HaloDepthSpin, Ls, NormalTwoOp>;
    using CGTwo = MDWFCoupledCG<floatT, AdapterTwo>;
    using SpinorAll = typename EvenOdd::SpinorAll;
    using SpinorE = typename EvenOdd::SpinorE;
    using SpinorO = typename EvenOdd::SpinorO;

private:
    EvenOdd _eo;
    NormalOp _normal;
    Adapter _adapter;
    SpinorE _be, _xe, _te, _ue;
    SpinorO _bo, _xo, _to, _uo;
    int _lastIterations;
    double _lastResidue;
    // setSchurTwo(true) only (created on demand: no extra fields otherwise).
    std::unique_ptr<NormalTwoOp> _normalTwo;
    std::unique_ptr<AdapterTwo> _adapterTwo;

    template<class Cg, class Ad, class S>
    void runCG(Ad &adapter, S &x, const S &b, int maxIter, double precision, const char *what) {
        Cg cg;
        const MDWFCoupledCGResult<floatT> result = cg.invert(adapter, x, b, maxIter, precision, true);
        _lastIterations += result.iterations;
        _lastResidue = result.residue;
        if (!result.converged) {
            throw std::runtime_error(stdLogger.fatal("MDWF even/odd ", what, " solve did not converge: iterations = ",
                                                     result.iterations, ", residue = ", result.residue));
        }
    }

public:
    MDWFMobiusEvenOddSolver(Gaugefield<floatT, true, HaloDepthGauge, R18> &gauge, floatT M5, floatT mf, floatT b5,
                            floatT csw, const std::string &name)
        : _eo(gauge, M5, mf, b5, csw, name + "_eo"),
          _normal(_eo, gauge.getComm(), name + "_normal"),
          _adapter(_normal),
          _be(gauge.getComm(), name + "_be"), _xe(gauge.getComm(), name + "_xe"),
          _te(gauge.getComm(), name + "_te"), _ue(gauge.getComm(), name + "_ue"),
          _bo(gauge.getComm(), name + "_bo"), _xo(gauge.getComm(), name + "_xo"),
          _to(gauge.getComm(), name + "_to"), _uo(gauge.getComm(), name + "_uo"),
          _lastIterations(0), _lastResidue(0.0) {}

    EvenOdd &evenOdd() {
        return _eo;
    }

    Adapter &adapter() {
        return _adapter;
    }

    // Solve with Grid's / GPT's second Schur form on the odd sites (see above) instead of Mhat on the even sites.
    void setSchurTwo(bool on) {
        if (on && !_normalTwo) {
            _normalTwo = std::make_unique<NormalTwoOp>(_eo, _be.getComm());
            _adapterTwo = std::make_unique<AdapterTwo>(*_normalTwo);
        }
        if (!on) {
            _adapterTwo.reset();
            _normalTwo.reset();
        }
    }

    bool schurTwo() const {
        return static_cast<bool>(_normalTwo);
    }

    // M x = b.
    void solve(SpinorAll &x, const SpinorAll &b, int maxIter, double precision) {
        _lastIterations = 0;
        _eo.refresh();
        solveNoRefresh(x, b, maxIter, precision);
    }

    // M^dagger y = b.
    void solveDagger(SpinorAll &y, const SpinorAll &b, int maxIter, double precision) {
        _lastIterations = 0;
        _eo.refresh();
        solveDaggerNoRefresh(y, b, maxIter, precision);
    }

    // M^dagger M x = b, as M^-1 (M^dagger)^-1 b.
    void solveNormal(SpinorAll &x, SpinorAll &work, const SpinorAll &b, int maxIter, double precision) {
        _lastIterations = 0;
        _eo.refresh();
        solveDaggerNoRefresh(work, b, maxIter, precision);
        solveNoRefresh(x, work, maxIter, precision);
    }

    int lastIterations() const {
        return _lastIterations;
    }

    double lastResidue() const {
        return _lastResidue;
    }

private:
    // M x = b (dagger = false) or M^dagger x = b with the second Schur form on the odd sites.
    void solveSchurTwo(SpinorAll &x, const SpinorAll &b, int maxIter, double precision, bool dagger) {
        EvenOdd::split(_be, _bo, b);
        // r_o = b_o - M_oe M_ee^-1 b_e  (M^dagger: b_o - M_eo^dagger M_ee^-dagger b_e)
        _eo.MeeInv(_te, _be, dagger);
        if (dagger) {
            _eo.MeoDagger(_to, _te);
        } else {
            _eo.Moe(_to, _te);
        }
        _uo = _bo;
        _uo.template axpyThisB<64>(static_cast<floatT>(-1.0), _to);
        // N^dagger N y = N^dagger r_o
        _normalTwo->setDagger(dagger);
        _normalTwo->schurTwo(_to, _uo, true);
        runCG<CGTwo>(*_adapterTwo, _xo, _to, maxIter, precision, dagger ? "M^dagger (Schur two)" : "M (Schur two)");
        // x_o = M_oo^-1 y,  x_e = M_ee^-1 (b_e - M_eo x_o)  (M^dagger: M_oo^-dagger, M_ee^-dagger, M_oe^dagger)
        _eo.MooInv(_uo, _xo, dagger);
        if (dagger) {
            _eo.MoeDagger(_te, _uo);
        } else {
            _eo.Meo(_te, _uo);
        }
        _ue = _be;
        _ue.template axpyThisB<64>(static_cast<floatT>(-1.0), _te);
        _eo.MeeInv(_xe, _ue, dagger);
        EvenOdd::merge(x, _xe, _uo);
    }

    void solveNoRefresh(SpinorAll &x, const SpinorAll &b, int maxIter, double precision) {
        if (_normalTwo) {
            solveSchurTwo(x, b, maxIter, precision, false);
            return;
        }
        EvenOdd::split(_be, _bo, b);
        // bhat_e = b_e - M_eo M_oo^-1 b_o
        _eo.MooInv(_to, _bo);
        _eo.Meo(_te, _to);
        _ue = _be;
        _ue.template axpyThisB<64>(static_cast<floatT>(-1.0), _te);
        // Mhat^dagger Mhat x_e = Mhat^dagger bhat_e
        _eo.schur(_te, _ue, true);
        runCG<CG>(_adapter, _xe, _te, maxIter, precision, "M");
        // x_o = M_oo^-1 (b_o - M_oe x_e)
        _eo.Moe(_to, _xe);
        _uo = _bo;
        _uo.template axpyThisB<64>(static_cast<floatT>(-1.0), _to);
        _eo.MooInv(_xo, _uo);
        EvenOdd::merge(x, _xe, _xo);
    }

    void solveDaggerNoRefresh(SpinorAll &y, const SpinorAll &b, int maxIter, double precision) {
        if (_normalTwo) {
            solveSchurTwo(y, b, maxIter, precision, true);
            return;
        }
        EvenOdd::split(_be, _bo, b);
        // bhat_e = b_e - M_oe^dagger M_oo^-dagger b_o
        _eo.MooInv(_to, _bo, true);
        _eo.MoeDagger(_te, _to);
        _ue = _be;
        _ue.template axpyThisB<64>(static_cast<floatT>(-1.0), _te);
        // Mhat^dagger Mhat w = bhat_e, y_e = Mhat w
        runCG<CG>(_adapter, _te, _ue, maxIter, precision, "M^dagger");
        _eo.schur(_xe, _te, false);
        // y_o = M_oo^-dagger (b_o - M_eo^dagger y_e)
        _eo.MeoDagger(_to, _xe);
        _uo = _bo;
        _uo.template axpyThisB<64>(static_cast<floatT>(-1.0), _to);
        _eo.MooInv(_xo, _uo, true);
        EvenOdd::merge(y, _xe, _xo);
    }
};
