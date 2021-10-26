# Inverter (Conjugate Gradient)


**This page is work in progress!** 

The Conjugate Gradient (CG) inverter solves the equation $Ax=b$, where $A$ is a symmetric and positive definite matrix, $b$ an input vector and $x$ the solution vector. Different versions of the CG inverter are implemented in our code in the two classes ConjugateGradient and AdvancedMultiShiftCG. The matrix $A$ in our case is, up to now, always $M^{\dagger}M$, with $M$ being the fermion matrix.

## The Basic Algorithm 

Given an initial guess $x_0$ and a target residual $\epsilon$, compute $r_0=b-Ax$ and set $p_0=r_0$.

> **while** $|r_{i+1}|^2 < \epsilon$ \
> $\qquad \alpha_{i} = \frac{|r_i|^2}{p^{\intercal}Ap}$\
> $\qquad x_{i+1} = x_{i} + \alpha_{i}p_{i}$\
> $\qquad r_{i+1} = r_{i} - \alpha_{i}r_{i}$\
> $\qquad \beta_{i} = \frac{|r_{i+1}|^2}{|r_i|^2}$\
> $\qquad p_{i+1} = r_{i+1} + \beta_{i}p_{i}$\
> $\qquad i = i+1$\
> **end while**\
> **return** $x_{i+1}$

## Residual drift and precision

In exact arithmetic, the iteratively computed residual $r_i$ is always equivalent to the true residual $r=b-Ax_{i}$. Due to rounding errors in floating point calculations, this is not true in practical use and the iterated residual $r_i$ will drift away from the true residual $r=b-Ax_{i}$. The magnitude of this drift depends on the floating point precision that you are using and the number of iterations the CG needs until convergence. This problem can be resolved by replacing the iterated residual with the exact residual occasionally.

## Multi-RHS and Multi-shift

Two important improvements of the basic CG algorithm that we employ in our code are the use of multiple right-hand sides and multiple shifts. These two optimizations cannot be used at the same time.
The multi rhs version of the CG solves $Ax_{k}=b_{k}$ for multiple "right-hand sides" $b_k$ simultaneously. This can improve performance significantly since the most expensive part in a CG iteration is the matrix-vector product $Ap$ that enters $\alpha$. The performance of this matrix-vector product, the Dslash kernel, is limited by the GPU's memory bandwidth. By applying the same matrix $A$ to multiple vectors at once, $A$ only needs to be loaded once, thereby saving memory bandwidth and increasing performance. 
The multi-shift CG solves $\left(A+\sigma_{k}\mathbb{1}\right)x_{k}=b$ with multiple shifts $\sigma_{k}$. Without going into too much technical details, the multi-shift CG essentially solves the system $\left(A+\sigma_{0}\mathbb{1}\right)x_0=b$ with the smallest shift $\sigma_0$ with the basic CG algorithm and computes the solutions for the larger shifts through some smart linear algebra on the way, thereby solving multiple problems at the cost of only one.

## The ConjugateGradient Class

This class implements the CG algorithm shown above and supports using multiple right hand sides. 
It has two template parameters, `floatT` and `NStacks` that define the floating point type to be used and the number of right hand sides, respectively.
```C++
ConjugateGradient<floatT, NStacks> cg;
```

The member functions of ConjugateGradient are different flavors of the basic algorithm listed above. Specifically, we have

```C++
template <typename Spinor_t>
void invert(LinearOperator<Spinor_t>& dslash, Spinor_t& spinorOut, Spinor_t& spinorIn, int max_iter, double precision);
```
which implements a multi-rhs CG with an absolute stopping criterion, i.e. the algorithm stops when $|r|^2<\mathrm{precision}$. The first argument `LinearOperator<Spinor_t>& dslash` can be any class that has the function `void applyMdaggM(Spinor_t&, Spinor_t&, bool update)` defined. In our case, this will usually be some dslash class. The second and third arguments are output and input Spinorfields. `max_iter` is the maximum number of iterations after which the CG exits and `precision` is the target residual used in the stopping criterion.


```C++
template <typename Spinor_t>
void invert_new(LinearOperator<Spinor_t>& dslash, Spinor_t& spinorOut, const Spinor_t& spinorIn, const int max_iter, const double precision);
```
 invert_new does the same as invert but uses a relative stopping criterion, i.e. the algorithm stops when $\frac{|r|^2}{|r_{0}|^2}<\mathrm{precision}$, where $r_0$ is the starting residual. 

```C++
template <typename Spinor_t>
void invert_res_replace(LinearOperator<Spinor_t>& dslash, Spinor_t& spinorOut, const Spinor_t& spinorIn, const int max_iter, const double precision);
```
invert_res_replace employs a residual replacement strategy where the iterated residual is replaced by the exact residual occasionally. 

## The AdvancedMultiShiftCG Class

This class implements the multi-shift CG algorithm. The template parameters are the same as for the ConjugateGradient class. 

The member function 
```C++
template <typename SpinorIn_t, typename SpinorOut_t>
void invert(LinearOperator<SpinorIn_t>& dslash, SpinorOut_t& spinorOut, const SpinorIn_t& spinorIn, SimpleArray<floatT, NStacks> sigma, const int max_iter, const double precision);
```
performs the multi-shift inversion. In contrast to ConjugateGradient::invert, this function has two template parameters, one for the input spinor type and one for the output spinor type. The input spinor is a Spinorfield with NStacks=1 and the output spinor is a Spinorfield where NStacks matches the number of shifts. The shifts are passed to the function via `SimpleArray<floatT, NStacks> sigma`.

## Mixed precision

The time to solution of the CG can be significantly decreased by using mixed precision approaches. Since it is an iterative method, we can use half precision floating point arithmetic for the bulk part of the computation and inject full precision residuals occasionally to correct for rounding errors accumulated along the way.  Two such methods are member functions of `ConjugateGradient`, namely 
```C++
template <typename Spinor_t, typename Spinor_t_half>
void invert_mixed(LinearOperator<Spinor_t>& dslash, LinearOperator<Spinor_t_half>& dslash_half, Spinor_t& spinorOut, const Spinor_t& spinorIn, const int max_iter, const double precision);
```
and
```C++
template <typename Spinor_t, typename Spinor_t_half>
void invert_mrel(LinearOperator<Spinor_t>& dslash, LinearOperator<Spinor_t_half>& dslash_half, Spinor_t& spinorOut, const Spinor_t& spinorIn, const int max_iter, const double precision);
```

`void invert_mixed` recomputes a full precision true residual if the norm of the current residual decreased by a factor 10 compared to the residual from the last restart. It then resets the CG by performing a gradient descent step ($p_{i+1}=r_{\mathrm{true}}=r_{i+1}$).


`void invert_mrel` uses the same update trigger but reprojects the gradient vector $p_{i}$ such that is orthogonal to the new, true residual. Doing this, one can retain partial information about the Krylov subspace which should lead to faster convergence compared to `void invert_mixed`.

In contrast to the other invert methods, one has to pass an additional, half-precision dslash operator to these methods. An example how to use the mixed precision inverters is given below

```C++
 //declare target precision gauge fields and half precision gauge fields
 Gaugefield<floatT, onDevice, HaloDepth, R18> gauge_smeared(commBase);
 Gaugefield<floatT, onDevice, HaloDepth, U3R14> gauge_naik(commBase); 
 Gaugefield<__half, onDevice, HaloDepth, R18> gauge_smeared_half(commBase);
 Gaugefield<__half, onDevice, HaloDepth, U3R14> gauge_naik_half(commBase);

//declare spinors
Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks> spinorIn(commBase);
Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks> spinorOut(commBase);
    

//declare CG, regular dslash and half precision dslash
ConjugateGradient<floatT, NStacks> cg;
    
HisqDSlash<floatT, onDevice, LatLayout, HaloDepth, HaloDepthSpin, NStacks> dslash(gauge_smeared, gauge_naik, param.m_ud());
HisqDSlash<__half, onDevice, LatLayout, HaloDepth, HaloDepthSpin, NStacks> dslash_half(gauge_smeared_half, gauge_naik_half, param.m_ud());    



//do stuff with the regular gaugefield and spinors, like smearing and some physics


//copy results into half prec gauge fields. THIS STEP IS IMPORTANT! 
 gauge_smeared_half.template convert_precision<floatT>(gauge_smeared);
 gauge_naik_half.template convert_precision<floatT>(gauge_naik);

//invert using floatT-half CG solver
cg.invert_mrel(dslash, dslash_half, spinorOut, spinorIn, param.cgMax(), param.residue());

```
