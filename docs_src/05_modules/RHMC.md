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
i.e. it initializes a massless [DSlash](dslash.md). (Fermion masses are taken care of when solving
the DSlash using the AdvancedMultiShiftCG, where each shift is related to the masses.)

Finally the RHMC will need an [integrator](integrator.md). The integrator is a leap frog by default, which
is decided by an RHMC parameter rather than at compile time. It is initialized as
```C++
integrator<floatT,onDevice,All ,HaloDepth,HaloDepthSpin> integrator;
integrator(_rhmc_param, _gaugeField, _p, _smeared_X, _smeared_W, dslash, _rat, _smearing)
```
which inside of the `integrator` class instantiates/initializes
```C++
    AdvancedMultiShiftCG<floatT, 12> cgM;
    _rhmc_param = _rhmc_param
    _gaugeField = _gaugeField
    _p = _p
    _X = _smeared_X
    _W = _smeared_W
    _dslash = dslash // was initialized to massless in RHMC
    _smearing = _smearing
    _rat = _rat
    _dslashM(_W, _X, 0.0) 
    ipdot(gaugeField.getComm())    
    ip_dot_f2_hisq(_gaugeField, ipdot, cgM, _dslash, _dslashM, _rhmc_param, _rat, _smearing) 
```
The AdvancedMultiShiftCG is a port of an old Bielefeld inverter. There are multiple possible 
[inverters](inverter.md) implemented in SIMULATeQCD, but I guess this one was best for the places it gets used.
The last one, `ip_dot_f2_hisq` is a `hisqForce` object that initializes as
```C++
    HisqForce<floatT, onDevice, HaloDepth, 4> ip_dot_f2_hisq;
    _GaugeBase = _gaugeField
    F1_create_3Link(_GaugeU3P,ipdot)
    F1_lepagelink(_GaugeU3P,ipdot)
    F3_create_3Link(_GaugeU3P,ipdot)
    _cg=cgM 
    _dslash=_dslash
    _dslash_multi=_dslashM
    _rhmc_param=_rhmc_param
    _rat=_rat
    _smearing=_smearing
    _TmpForce(GaugeBase.getComm())
    _createF2(_GaugeLvl1,_TmpForce)
    _finalizeF3(_GaugeU3P,_TmpForce)
    _createNaikF1(_GaugeU3P,_TmpForce)
```
There is very unfortunately a lot going on in this initialization and in the `updateForce` later, and the code 
is difficult to read. It may be difficult to reorganize in a better way, since this is trying to leverage
SIMULATeQCD's [functor](../04_codeBase/functorSyntax.md) syntax. 
This list initializes
```C++
    contribution_3link<floatT, onDevice, HaloDepth, comp, false> F1_create_3Link;
    contribution_lepagelink<floatT, onDevice, HaloDepth, comp> F1_lepagelink;
    contribution_3link<floatT, onDevice, HaloDepth, comp, true> F3_create_3Link;
    constructU3ProjForce<floatT, onDevice, HaloDepth,comp> _createF2;
    finalizeForce<floatT, onDevice, HaloDepth,comp> _finalizeF3;
    constructNaikDerivForce<floatT, onDevice, HaloDepth,comp> _createNaikF1;
```
Each of these is a functor. The `bool` template parameter in `contribution_3link` specifies whether we are
doing asqtad or fat7 [smearing](../05_modules/gaugeSmearing.md). The functors will be passed to iterators in
the `updateForce` part, which will hand the output of the functors to some other field. Each of the arguments
of these functors in the initialization list of `ip_dot_f2_hisq` is treated as input.

It is also important to point out that when the `HisqForce` object is instantiated, it creates the attributes
```C++
    Spinorfield<floatT, onDevice, Even, HaloDepthSpin, rdeg> _spinor_x;
    Spinorfield<floatT, onDevice, Odd, HaloDepthSpin, rdeg> _spinor_y;
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
    $|\Phi\rangle=(M^\dagger M)^{-1} R$, where $R$ is a Gaussian random vector. It seems to be massive, since the container uses e.g. `rat_inv_lf`, which was
    populated according to the masses in `HMC::init_ratapprox()`.

After getting everything set up, we are ready to do the integration with `integrator.integrate()`. The meat
of the integration is in `ip_dot_f2_hisq.updateForce(_phi.phi_container.at(i),ipdot,light)`. The bool `light`
tells the integrator whether we are doing a light or strange quark.

### The fermion force

This is what `updateForce` is going to do: We want to update pseudofermion field `i`. Inside of `updateForce`, we again change names
of our familiar objects:
```C++
SpinorIn=_phi.phi_container.at(i)
Force=ipdot
```
Then a call to `make_f0(SpinorIn,Force,_TmpForce,isLight)` populates `Force` with $|X\rangle\langle Y|$ where the fields
$X$ and $Y$ are separated by one site. It also populates `_TmpForce` with $|X\rangle\langle Y|$ where the fields
are separated by three sites. It also populates `_spinor_x` and `_spinor_y` with these spinor fields $|X\rangle$
and $|Y\rangle$.

Let's discuss `make_f0` in a bit more detail. Inside it uses `_cg`, which is an alias for the `AdvancedMultishiftCG`
inverter `cgM` that was instantiated in the integrator. Using this inverter, along with `_dslash`, which if you follow him
back far enough, was intialized as a massless Dirac operator in the `HMC` object. Following the notation
of [Wong and Woloshyn](http://arxiv.org/abs/0710.0737), the `shifts` inside `make_f0` are constructed from the masses
and the $\beta_l$ that go into the rational approximation for $(M^\dagger M)^{n_f/4}$. Since `SpinorIn` is $|\Phi\rangle$,
we see that $|X\rangle=(M^\dagger M+\beta_l)^{-1}|\Phi\rangle$. Then $|Y\rangle$ is constructed using `_dslash_multi`,
which you see is a massless, multi-RHS DSlash when you follow the initializer lists of `ip_dot_f2_hisq` and the `integrator`. 
Hence $|Y\rangle=D|X\rangle$. 


