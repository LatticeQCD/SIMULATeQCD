# HISQ action 

This code uses the highly improved staggered quark (HISQ) action, first
introduced [in this paper](https://doi.org/10.1103/PhysRevD.75.054502).
To carry out the MD of the [RHMC](../03_applications/rhmc.md), one needs the
fermion and gauge forces. The HISQ force uses AsqTad-like smears to reduce taste
violations, along with terms to remove further $\mathcal{O}(a^2)$ errors
introduced by this smearing. Overall, the HISQ action is expected to have
no tree-level $\mathcal{O}(a^2)$ error.

The fermion force in SIMULATeQCD follows the [MILC](https://github.com/milc-qcd/milc_qcd) code. 
A very readable explanation how improved staggered forces are constructed generically
can be found [here](https://doi.org/10.22323/1.042.0047). Helpful comments on the roles
of the various terms in the smearing can be found 
[here](https://github.com/lattice/quda/wiki/HISQ-Stencil).

## Gauge force

We use the HISQ/tree action, which is a tree-level improved
Lüscher-Weisz action in the gauge sector. The relative
weights of the plaquette and rectangle terms are

$
    c_\text{plaq} = 5/4,
$

$
    c_\text{rect} = -1/6.
$

## Fermion force

### AsqTad-like smears

As stated above, the HISQ force uses smearing to suppress taste exchange.
We start by establishing some background and terminology before introducing
the HISQ smear.

Typically the gauge connection between two neighboring sites $x$ and $y$
on the lattice is just a single link $U(x,y)$, which is in some sense the
most local connection imaginable. One can also relax this locality, so that
the gauge connection contains information from a larger region around
$x$ and $y$; for example the connection could depend on a general sum,
including many paths connecting $x$ and $y$. Let's call
this sum $\Sigma(x,y)$. Then the gauge connection could be $V(x,y)$,
where $V$ is chosen by extremizing $\mathrm{tr} V\Sigma^\dagger$. These gauge
connections are called [fat links](https://link.aps.org/doi/10.1103/PhysRevD.55.R1133).
Fat links modify particle spectra, since they amount to a change of the lattice
propagator. Including up to 7-link constructs is termed _Fat7_.

Fat7 smearing removes $\mathcal{O}(a^2)$ taste-breaking effects, but introduces
$\mathcal{O}(a^2)$ errors not related to taste breaking. These are removed by the
addition of a 5-link construct called the [LePage term](https://doi.org/10.1103/PhysRevD.59.074502).
Altogether this defines [AsqTad smears](https://doi.org/10.1103/PhysRevD.60.054503),
which can be written as

$
  V_\mu=~c_1U_\mu
           +\sum_\nu\left[c_3U_{\mu\nu}^{(3)}
           +\sum_\rho\left(c_5U_{\mu\nu\rho}^{(5)}
           +\sum_\sigma c_7U^{(7)}_{\mu\nu\rho\sigma}\right)
            +c_LU^{(L)}_{\mu\nu}\right],
$

where the link constructs are given by

$ U_{\mu\nu}^{(3)}(x)
       =U_\nu(x)U_\mu(x+a\hat{\nu})U_\nu^\dagger(x+a\hat{\mu})$

$  U_{\mu\nu\rho}^{(5)}(x)
       =U_\nu(x)U_{\mu\rho}^{(3)}(x+a\hat{\nu})U_\nu^\dagger(x+a\hat{\mu})$

$  U_{\mu\nu\rho\sigma}^{(7)}(x)
       =U_\nu(x)U_{\mu\rho\sigma}^{(5)}(x+a\hat{\nu})
         U_\nu^\dagger(x+a\hat{\mu})$

$  U_{\mu\nu}^{(L)}(x)
       =U_\nu(x)U_{\mu\nu}^{(3)}(x+a\hat{\nu})U_\nu^\dagger(x+a\hat{\mu}).$

We note the $n$-link constructs always move in perpendicular directions. The LePage term
differs from the 5-link as it is instead restricted to a plane. 
Fat7 smears have the same form, but with $c_L=0$. In the parlance of many LQCD practitioners,
the original fields $U$ are called _thin links_ and the smeared $V$ _fat links_.

### HISQ smearing

Taste breaking can be thought of through _taste exchange_, where
one quark changes its taste by exchanging a virtual gluon with momentum
$p=\pi/a$; a quark with low enough momentum can thereby be pushed into
another corner of the Brillouin zone. This is an effect of our discretization,
so taste breaking vanishes in the continuum limit. A strategy at finite spacing
to reduce this discretization effect is to
modify gluon spectra to suppress these taste-exchange processes. This is the
idea behind HISQ smearing.

The HISQ smear can be expressed as

$ U^{\rm HISQ}=\mathcal{F}^{\rm corr}\mathcal{U}\mathcal{F}^{\rm Fat7}U, $

i.e. we Fat7-smear the thin link, followed by a reunitarization and then a "corrected"
smearing, which is an AsqTad smear equipped with an additional 3-link term,
the _Naik term_, which improves the dispersion relation to $\mathcal{O}(p^4)$.
The Naik term marches three steps in one direction, and its construct is thus
sometimes referred to as _long links_.

SIMULATeQCD follows MILC in that it chooses its coefficients according to
[the MILC scaling studies paper](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.82.074501).
This means

$  c_1^{\rm Fat7} = 1/8$

$  c_3^{\rm Fat7} = 1/16$

$  c_5^{\rm Fat7} = 1/64$

$  c_7^{\rm Fat7} = 1/384$

$  c_1^{\rm corr} = 1+\epsilon/8$

$  c_3^{\rm corr} = 1/16$

$  c_5^{\rm corr} = 1/64$

$  c_7^{\rm corr} = 1/384$

$  c_L^{\rm corr} = -1/8$

$  c_\text{Naik}^{\rm corr} = -(1+\epsilon)/24,$

where $c_\text{Naik}$ is the coefficient for the Naik term and $\epsilon$,
the so-called _Naik_ $\epsilon$, is needed as a correction when one wants to have charm,
bottom, and/or top quarks. (Note that SIMULATeQCD has some of the machinery
needed for dynamical charm simulations, but at the moment, it is not
fully implemented.)

We adopt the notation of MILC scaling studies paper, where
$V \equiv\mathcal{F}^{\rm Fat7}U$,
$W \equiv\mathcal{U} V$, and
$X \equiv\mathcal{F}^{\rm corr} W$.

### The U(3) projection

The $U(3)$ projection in SIMULATeQCD follows the analytic projection in MILC.
It is carried out exactly as described in Appendix C of the
MILC scaling studies paper, i.e. it is computed as

$
W=VQ^{-1/2}
$

for some analytically determined $Q$. For the force, we will need also the derivative
of the projection. Again following the MILC paper, this is done analytically.
If $Q$ is sufficiently singular, the computation of $W$ will be quite inaccurate.
To circumvent this, there is a _force filter_ that checks whether any of the eigenvalues
of $Q$ is less than a threshold $\delta$. In SIMULATeQCD, this is hard-coded
to $\delta=5\times 10^{-5}$. When $Q$ is singular in this sense, its eigenvalues are 
shifted by $\delta$ to put a hard limit on the severity of the singularity.
If the force filter is applied too much, this can lower the acceptance rate.


### Using HISQ smearing in SIMULATeQCD


In SIMULATeQCD, the Fat7 smear and 1-, 3-, 5-, and 7-link part of the corrected 
smear are referred to as "level-1" and "level-2" smears, respectively. To do a level-1 smear,
```C++
HisqSmearing<PREC, USE_GPU,HaloDepth> V(gauge_in, gauge_out, redBase);
V.hisqSmearing(getLevel1params())
```
Next we project the level-1 smeared links back to U(3):
```C++
HisqSmearing<PREC, USE_GPU,HaloDepth> W(gauge_in, gauge_out, redBase);
W.u3Project()
```
Finally we smear again the $W$ links with the corrected smear:
```C++
HisqSmearing<PREC, USE_GPU,HaloDepth> X(gauge_in, gauge_out, redBase);
X.hisqSmearing(getLevel2params())
```

For HISQ dslash,
the Naik links are constructed from the unitarized links, i.e. using the $W$ links.
To use this we have to call
```C++
HisqSmearing<PREC, USE_GPU,HaloDepth> N(gauge_in, gauge_out, redBase);
N.naikterm()
```
The smearing classes are implemented in `src/modules/hisq/hisqSmearing.cpp`.
How to construct the smeared links is shown in
```shell
src/testing/main_hisqSmearing*.cpp.
```

## Fermion force

The fermion part of the action can be expressed as

$
S_f=\bra{\Phi}\left(D^{\dagger} D\right)^{-N_f/4}\ket{\Phi},
$

where $\ket{\Phi}$ is the pseudofermion field. As discussed in the
[RHMC](../03_applications/rhmc.md) module, the fourth root is approximated
via a rational approximation

$
\left(D^\dagger D\right)^{-N_f/4}\approx\alpha_0+\sum_l\frac{\alpha_l}{D^\dagger D+\beta_l}.
$

The $\alpha_l$ are often referred to as _residues_ and the $\beta_l$ as _shifts_.
That $\left(D^\dagger D\right)^{-N_f/4}$ can be expressed in this way allows the possibility
to solve each term in the sum independently, which is a motivation why we use
a multi-shift CG, discussed in the [inverter](../05_modules/inverter.md) article.

To carry out MD, we need the derivative of $S_f$ w.r.t. MC time,
which translates via the chain rule to a derivative w.r.t. the gauge field
$U$. Successive applications of the chain rule
show us we need

$
\frac{\partial S_f}{\partial U}=\frac{\partial S_f}{\partial X}\frac{\partial X}{\partial W}\frac{\partial W}{\partial V}\frac{\partial V}{\partial U}.
$

Ignoring Naik links, the derivative can be written

$
\frac{\partial S_f}{\partial\left[U_{x,\mu}\right]_{ab}}=
\sum_{y,\nu}(-1)^y\eta_{y,\nu}\left(
\frac{\partial \left[X_{y,\nu}\right]_{mn}}{\partial \left[U_{x,\mu}\right]_{ab}}
\left[f_{y,\nu}\right]_{mn}+
\frac{\partial \left[X_{y,\nu}^{\dagger}\right]_{mn}}{\partial \left[U_{x,\mu}\right]_{ab}}
\left[f^\dagger_{y,\nu}\right]_{mn}
\right)
$

Here, we explicitly write out color indices $a$, $b$, $m$, and $n$ along with the
staggered phases $\eta$. The outer products

$
\left[f_{y,\nu}\right]_{mn}\equiv
\begin{cases}
\sum_l\alpha_l\left[R^l_{y+\nu}\right]_n\left[L^{l*}_y\right]_m & {\rm even}~y\\
\sum_l\alpha_l\left[L^l_{y+\nu}\right]_n\left[R^{l*}_y\right]_m & {\rm odd}~y\\
\end{cases}
$

are expressed in terms of the fields
$\ket{L^l}\equiv\left(D^\dagger D+\beta_l\right)^{-1}\ket{\Phi}$ and $\ket{R^l}\equiv D_0\ket{L^l}$,
where $D_0$ is the massless Dirac operator. Our implementation of the Dirac operator is
discussed [here](dslash.md).

### HISQ force in SIMULATeQCD

The fermion force used at each MD time step is implemented in `src/modules/hisq/hisqForce.cpp`,
in particular the `updateForce()` method. A high-level schematic overview of this force is
that it does the following:
1. `make_f0` populates `Force` and `_TmpForce` with the outer product
for the 1-link and Naik terms, respectively. (The pseudofermion fields
in the Naik outer product must be separated by 3 steps rather than 1.)
`_TmpForce` will accumulate the force contributions.
2. Compute $V$ with `SmearLvl1` from $U$.
3. Compute $W$ with `ProjectU3` from $V$.
4. Multiply staggered phases into $W$ and apply imaginary $\mu$ if nonzero.
5. Accumulate Naik, 3-link, 5-link, LePage, and 7-link derivatives. These are computed as a derivative w.r.t. $W$, and hence form the $\partial X/\partial W$ term discussed earlier. They are mostly computed with `contribution_*link*` kernels, except for the Naik and 3-link. There some extra set up is carried out in addition to the derivative calculation by `_createNaikF1` and `F1_create_3Link`.
6. Multiply staggered phases into $V$ and compute the derivative of the $U(3)$ projection w.r.t. $V$, i.e. $\partial W/\partial V$. Besides doing this derivative, `_createF2` multiplies $\partial X/\partial W$ with $\partial W/\partial V$, storing the result in `Force`.
7. Staggered phases are multiplied into $U$ and stored in $_GaugeU3P$ to reduce memory burden.
8. `_TmpForce` is overwritten with $(\partial X/\partial W)(\partial W/\partial V)(\partial V_3/\partial U)$, where $V_3$ indicates the 3-link part of $V$. This is done by `F3_create_3Link`.
9. We then add to `_TmpForce` contributions from the 5- and 7-link derivatives, completing the chain rule.
10. At the end, `_finalizeF3` multiplies `_TmpForce` by $U$ on the left, which maintains gauge invariance, then takes the traceless, antihermitian part of the result. This is the fermion force.

