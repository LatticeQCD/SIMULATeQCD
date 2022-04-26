# Dslash

**This page is work in progress!**

This module implements the Dslash operator for staggered and highly improved staggered quark (HISQ) fermions. It is a sparse matrix-vector product that is implemented as a 4 dimensional stencil kernel with nearest and third-nearest neighbor terms. For HISQ fermions, the matrix-vector product takes the form
$
\begin{align}
D[U]\psi_{x}&=\sum_{\mu=0}^{4}\left[c_{1}\left(V_{x,\mu}\psi_{x+\hat{\mu}}-V^{\dagger}_{x-\hat{\mu},\mu}\psi_{x-\hat{\mu}}\right)+c_{3}\left(W_{x,\mu}\psi_{x+3\hat{\mu}}-W^{\dagger}_{x-3\hat{\mu},\mu}\psi_{x-3\hat{\mu}}\right)\right],
\end{align}$
where $V_{x,\mu}$ and $W_{x,\mu}$ are the HISQ smeared fields described in [Hisq Smearing](https://latticeqcd.github.io/SIMULATeQCD/05_modules/08_gaugeSmearing.html).
For staggered fermions, only the nearest neighbor term is present and the basic gauge field $U_{x,\mu}$ is used in place of $V_{x,\mu}$.

In the code, Dslash operators are derived from an abstract base class that defines the interface for Dslash operators.
```C++
template<typename SpinorLHS_t, typename SpinorRHS_t>
class DSlash : public LinearOperator<SpinorRHS_t> {
public:
    virtual void Dslash(SpinorLHS_t &lhs, SpinorRHS_t &rhs, bool update = true);
    virtual void applyMdaggM(SpinorRHS_t &, SpinorRHS_t &, bool update = true) = 0;
};
```
The method `void Dslash(SpinorLHS_t &lhs, SpinorRHS_t &rhs, bool update = true)` applies the stencil operator described above to an input vector `rhs` and write the result into the vector `lhs`.

$\begin{align}
\chi=D\psi
\end{align}$


The method `void applyMdaggM(SpinorRHS_t &lhs, SpinorRHS_t &rhs, bool update = true)` computes

$\begin{align}
\chi=M^{\dagger}M\psi \;\;\;\mathrm{where} \;\;M[U]=D[U]+m_{f},
\end{align}$

and $m_{f}$ is the quark mass. In both methods, `bool update` toggles wether or not a halo update should be performed on the output spinor after the kernel is applied.

The derived class for HISQ fermions is:
```C++
template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks = 1>
class HisqDSlash : public DSlash<Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks>,
        Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> >
```
`floatT` specifies which floating point type to use and `onDevice` toggles whether or not gauge and spinor fields are residing on the host or device. The template parameter `Layout LatLayoutRHS` specifies the lattice layout of the input vector and can take the values `Even` `Odd` and `All`. `HaloDepthGauge` and `HaloDepthSpin` specify the depth of the halo buffers needed for multi-gpu calculations. Usually, `HaloDepthGauge=2` and `HaloDepthSpin=4` should be chosen.
The last template parameter `size_t NStacks` specifies how many rhs vectors are used simultaneously. Loading a gauge link once and multiplying it to multiple vectors within the same kernel call significantly increases performance as the DSlash kernels performance is mostly bandwidth bound.
A HisqDSlash object is constructed with:
```C++
HisqDSlash(Gauge_t<R18> &gaugefield_smeared, Gauge_t<U3R14> &gaugefield_Naik, const double mass, floatT naik_epsilon = 0.0,
               std::string spinorName = "SHARED_HisqDSlashSpinor")
```
Here `gaugefield_smeared` refers to $V_{x,\mu}$ and `gaugefield_Naik` to $W_{x,\mu}$. Note the different compression types for the gauge fields: the field entering the third-nearest neighbor hopping term uses `CompressionType comp=U3R14` in order to save memory bandwidth. `const double mass` specifies the quark mass that enters `void MdaggM` and `floatT naik_epsilon` specifies the coefficient $\epsilon$ that can be included in the Naik term. The last parameter that can be given to the constructor is a string identifiying the memory allocated for a temporary spinor that is used internally. By default multiple instances of HisqDslash will share the same memory for this temporary spinor. See the documentation on `MemoryManagement` for further informations.