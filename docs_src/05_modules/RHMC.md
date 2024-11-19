# RHMC in gory detail

I want to begin by noting that SIMULATeQCD's RHMC is hard-coded to have only
light and strange quarks. The HISQ parameters are also hard-coded, for instance
the different weights for each link constrcut. It is flexible to have multiple
degenerate quarks. It is also hard-coded to have a rational approximation
of degree [14,14] with one additional linear constant.

In `main_rhmc.cpp` we have a relevant starting point,
```C++
rhmc<floatT, true, HaloDepth> HMC(param, rat, gauge, d_rand.state);
```
Here, `param` is an `RhmcParameters` object, which contains most of the parameters
needed to do the RHMC. `rat` is a `RationalCoeff` object, which holds all the
rational coefficients read in from the rational approximation file. `gauge` is
the to-be-updated `Gaugefield` object, and `d_rand.state` is the random
number state of the `grnd_state` object `d_rand`.


## Initialization

When the `HMC` object is instantiated, the `private` members get initialized using
what was passed to `rhmc`:
```C++
_rhmc_param = param;
_rat = rat;
_gaugeField = gauge;
_rand_state = d_rand.state;
```
The different rational coefficients are explained in `rhmcParameters.h`.

The `private` member `_smearing` is initialized as
```C++
HisqSmearing<floatT,onDevice,HaloDepth,R18,R18,R18,U3R14> _smearing;
_smearing(_gaugeField, _smeared_W, _smeared_X)
```
which within the context of the `HisqSmearing` class initializes
```
_gauge_base = _gaugeField 
_gauge_lvl2 = _smeared_W 
_gauge_naik = _smeared_X
```
The `W` and `X` seem to reference the notation of
[this](https://link.aps.org/doi/10.1103/PhysRevD.82.074501) MILC paper.
`W` should be after a fat7 smear with reunitarization, while `X` should
be `W` after an asqtad smear, at least according to the paper.

As part of the initialization, I want to discuss the `phi_f_container` objects. There
are two private members set aside as
```C++
    Spinorfield_container<floatT, onDevice, Even, HaloDepthSpin> phi_sf_container;
    Spinorfield_container<floatT, onDevice, Even, HaloDepthSpin> phi_lf_container;
```
These `Spinorfield_container` objects are basically wrappers for
```C++
    std::vector<Spinorfield<floatT, onDevice, LatticeLayout, HaloDepth, NStacks>>
```
The reason they are on even sites only is that $M^\dagger M$ somehow doubles the number
of fermion species, and restricting this quantity to either even or odd sites only
somehow undoes the doubling. Similarly, there is a private member
```C++
    HisqDSlash<floatT,onDevice,Even,HaloDepth,HaloDepthSpin,1> dslash;
    dslash(_smeared_W, _smeared_X, 0.0) 
```
which inside of the `HisqDSlash` class initializes
```C++
    _gauge_smeared = _smeared_W
    _gauge_Naik = _smeared_X
    _mass = 0.0 
```
i.e. it initializes a massless DSlash. (Fermion masses are taken care of when solving
the DSlash using the AdvancedMultiShiftCG, where each shift is related to the masses.)

Finally the RHMC will need an integrator. The integrator is a leap frog by default, which
is decided by an RHMC parameter rather than at compile time. It is initialized as
```C++
integrator<floatT,onDevice,All ,HaloDepth,HaloDepthSpin> integrator;
integrator(_rhmc_param, _gaugeField, _p, _smeared_X, _smeared_W, dslash, _rat, _smearing)
```
which inside of the `integrator` class initializes
```C++
    _rhmc_param = _rhmc_param
    _gaugeField = _gaugeField
    _p = _p
    _X = _smeared_X
    _W = _smeared_W
    _dslash = dslash // was initialized to massles in RHMC
    _smearing = _smearing
    _rat = _rat
    _dslashM(_W, _X, 0.0) // ip_dot_f2_hisq is the hisq force object, used to calculate stuff
    ipdot(gaugeField.getComm())         // "force" gaugeField object, p-dot
    ip_dot_f2_hisq(_gaugeField, ipdot, cgM, _dslash, _dslashM, _rhmc_param, _rat, _smearing)
```

The next step is a call to `HMC.init_ratapprox()`. Using the data from the rational
approximation file `_rat`, this fills out some `std::vector` members of `HMC`, like
```C++
    std::vector<floatT> rat_sf;
    std::vector<floatT> rat_lf;
    std::vector<floatT> rat_inv_sf;
    std::vector<floatT> rat_inv_lf;
    std::vector<floatT> rat_bar_sf;
    std::vector<floatT> rat_bar_lf;
```
The entries that are filled out are first the constant, then numerator, then the
denominator[0] + $m_q^2$, then denominator[i]-denominator[0]. The `_bar_` members
have the same structure, without the constant.


## The update

The update starts off with

1. A call to `HMC::check_unitarity()` makes sure that `_gaugeField` satisfies ${\rm tr}~U^\dagger U=1$.
2. The `_gaugeField` is saved in the private member `_savedField` of `HMC`, which lets us check reversibility. 
3. `_smearing.SmearAll()` populates `_smearedW` with the `X` and populates `_smearedX` with the naik links. (I guess the names make sense later?)
4. `HMC::generate_momenta()` fills the private `Gaugefield` member `_p` with Gaussian random matrices. Under the hood, it is calling `SU3::gauss` on every link.
5. `HMC::make_phi(phi_sf_container.phi_container[i], rat_inv_sf)` populates the private member `phi_sf_container` using the data in `rat_inv_sf` that
    was initialized earlier. `phi_sf_container` is indexed by the pseudofermion number. It seems to refer to the $|\Phi\rangle$ notation
    of [Gottlieb et al](https://link.aps.org/doi/10.1103/PhysRevD.35.2531). Ultimately, `phi_sf_container` stores for each pseudofermion species the complex field
    $|\Phi\rangle=(M^\dagger M)^{-1} R$, where $R$ is a Gaussian random vector.

### The integration

After getting everything set up, we are ready to do the integration with `integrator.integrate()`.
