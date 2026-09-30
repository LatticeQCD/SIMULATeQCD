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
 * Correctness scaffold: single rank, no fused kernels, double precision.
 */

#pragma once

#include "MDWFAdjointOperator.h"
#include "MDWFCoupledCG.h"
#include "MDWFCoupledSolverAdapter.h"
#include "MDWFMobiusMapping.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
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
public:
    using Gauge = Gaugefield<floatT, true, HaloDepthGauge, R18>;
    using SpinorAll = MDWFSpinor<floatT, true, All, HaloDepthSpin, Ls>;
    using SpinorE = MDWFSpinor<floatT, true, Even, HaloDepthSpin, Ls>;
    using SpinorO = MDWFSpinor<floatT, true, Odd, HaloDepthSpin, Ls>;
    using Field = Spinorfield<floatT, true, All, HaloDepthGauge, 18, 1>;

private:
    Gauge &_gauge;
    MDWFMobiusOperatorParameters<floatT> _params;
    floatT _mf;
    floatT _csw;
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

    template<Layout L>
    void diagInverse(MDWFSpinor<floatT, true, L, HaloDepthSpin, Ls> &out,
                     const MDWFSpinor<floatT, true, L, HaloDepthSpin, Ls> &in, bool dagger) {
        CalcGSite<L, HaloDepthSpin> calcGSite;
        const size_t elems = GIndexer<L, HaloDepthSpin>::getLatData().sizeh;
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
            out.template iterateOverBulk<BLOCKSIZE>(
                DiracWilsonEvenOdd2<floatT, LOut, LIn, HaloDepthGauge, HaloDepthSpin, Ls, false>(
                    _gauge, tIn, 0.0, 0.0));
        } else {
            applyMDWFGamma5<floatT, true, LIn, HaloDepthSpin, Ls>(tIn, in, true);
            tOut.template iterateOverBulk<BLOCKSIZE>(
                DiracWilsonEvenOdd2<floatT, LOut, LIn, HaloDepthGauge, HaloDepthSpin, Ls, false>(
                    _gauge, tIn, 0.0, 0.0));
            applyMDWFGamma5<floatT, true, LOut, HaloDepthSpin, Ls>(out, tOut);
            tOut = out;
            applyMDWFFifthDimAdjointCoupling<floatT, true, LOut, HaloDepthSpin, Ls>(out, tOut, _params.dinCoeff);
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
          _csw(csw),
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
          _e1(gauge.getComm(), name + "_ework1"),
          _e2(gauge.getComm(), name + "_ework2"),
          _e3(gauge.getComm(), name + "_ework3"),
          _o1(gauge.getComm(), name + "_owork1"),
          _o2(gauge.getComm(), name + "_owork2"),
          _o3(gauge.getComm(), name + "_owork3"),
          _fresh(false) {}

    // Recomputes the clover blocks A and the block-inverse matrices from the current gauge field.
    void refresh() {
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
    }

    const MDWFMobiusOperatorParameters<floatT> &params() const {
        return _params;
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
            h[b]->updateAll();
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
            g[b]->updateAll();
        }
    }

    // All <-> even/odd.
    static void split(SpinorE &even, SpinorO &odd, const SpinorAll &all) {
        even.template iterateOverBulk<BLOCKSIZE>(MDWFEvenOddExtract<floatT, Even, HaloDepthSpin, Ls>(all));
        odd.template iterateOverBulk<BLOCKSIZE>(MDWFEvenOddExtract<floatT, Odd, HaloDepthSpin, Ls>(all));
    }

    static void merge(SpinorAll &all, const SpinorE &even, const SpinorO &odd) {
        all.template iterateOverBulk<BLOCKSIZE>(MDWFEvenOddMerge<floatT, HaloDepthSpin, Ls>(even, odd));
        all.updateAll();
    }
};

// Mhat^dagger Mhat on the even sites, in the shape MDWFCoupledSolverAdapter / MDWFCoupledCG expect.
template<class EvenOdd>
class MDWFMobiusSchurNormalOperator {
public:
    using Spinor = typename EvenOdd::SpinorE;

private:
    EvenOdd &_eo;
    Spinor _tmp;

public:
    MDWFMobiusSchurNormalOperator(EvenOdd &eo, CommunicationBase &comm, const std::string &name)
        : _eo(eo), _tmp(comm, name + "_tmp") {}

    void apply(Spinor &out, const Spinor &in, bool update = false) {
        _eo.schur(_tmp, in, false);
        _eo.schur(out, _tmp, true);
        if (update) {
            out.updateAll();
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

    void runCG(SpinorE &x, const SpinorE &b, int maxIter, double precision, const char *what) {
        CG cg;
        const MDWFCoupledCGResult<floatT> result = cg.invert(_adapter, x, b, maxIter, precision, true);
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
    void solveNoRefresh(SpinorAll &x, const SpinorAll &b, int maxIter, double precision) {
        EvenOdd::split(_be, _bo, b);
        // bhat_e = b_e - M_eo M_oo^-1 b_o
        _eo.MooInv(_to, _bo);
        _eo.Meo(_te, _to);
        _ue = _be;
        _ue.template axpyThisB<64>(static_cast<floatT>(-1.0), _te);
        // Mhat^dagger Mhat x_e = Mhat^dagger bhat_e
        _eo.schur(_te, _ue, true);
        runCG(_xe, _te, maxIter, precision, "M");
        // x_o = M_oo^-1 (b_o - M_oe x_e)
        _eo.Moe(_to, _xe);
        _uo = _bo;
        _uo.template axpyThisB<64>(static_cast<floatT>(-1.0), _to);
        _eo.MooInv(_xo, _uo);
        EvenOdd::merge(x, _xe, _xo);
    }

    void solveDaggerNoRefresh(SpinorAll &y, const SpinorAll &b, int maxIter, double precision) {
        EvenOdd::split(_be, _bo, b);
        // bhat_e = b_e - M_oe^dagger M_oo^-dagger b_o
        _eo.MooInv(_to, _bo, true);
        _eo.MoeDagger(_te, _to);
        _ue = _be;
        _ue.template axpyThisB<64>(static_cast<floatT>(-1.0), _te);
        // Mhat^dagger Mhat w = bhat_e, y_e = Mhat w
        runCG(_te, _ue, maxIter, precision, "M^dagger");
        _eo.schur(_xe, _te, false);
        // y_o = M_oo^-dagger (b_o - M_eo^dagger y_e)
        _eo.MeoDagger(_to, _xe);
        _uo = _bo;
        _uo.template axpyThisB<64>(static_cast<floatT>(-1.0), _to);
        _eo.MooInv(_xo, _uo, true);
        EvenOdd::merge(y, _xe, _xo);
    }
};
