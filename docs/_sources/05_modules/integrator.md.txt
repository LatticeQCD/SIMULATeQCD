# Integrator

This class takes care of the symplectic integration of the molecular dynamics trajectory. An object of this type is instantiated in the constructor of the rhmc class. And the MD evolution is done by the `integrate` function. There is the possibility of several different integration schemes. At the moment there are only two schemes: Purge gauge Leapfrog and a [Sexton-Weingarten](https://www.sciencedirect.com/science/article/pii/055032139290263B?via%3Dihub) integrator with different time steps for 
gauge, heavy fermion and light fermion parts of the force. The Sexton-Weingarten integrator uses the [Hasenbusch](https://www.sciencedirect.com/science/article/pii/S0370269301011029?via%3Dihub) trick.


The constructor has to be called with the usual template arguments and one has to pass several arguments:
```C++
integrator(RhmcParameters rhmc_param, Gaugefield<floatT,onDevice,All,HaloDepth> &gaugeField, 
        Gaugefield<floatT,onDevice,All,HaloDepth> &p, Gaugefield<floatT,onDevice,All,HaloDepth> &X, 
        CG<floatT, onDevice, HaloDepth> &cg, DSlash<floatT, onDevice, HaloDepth> &dslash, 
        std::vector<floatT> &rat1f, std::vector<floatT> &rat2f)
```
Aruments are: A `RhmcParameters` instance, the gauge field, a conjugate momentum field, two smeared gauge fields, a CG instance, an instance of a dslash and two vectors containing the coefficients for the rational approximations needed in the fermion force.

## Integrator.integrate

```C++
void integrator<floatT, onDevice, LatticeLayout, HaloDepth, HaloDepthSpin>::integrate(
    Spinorfield_container<floatT, onDevice, Even, HaloDepthSpin> &_phi_lf,
    Spinorfield_container<floatT, onDevice, Even, HaloDepthSpin> &_phi_sf)
```

`integrator.integrate` requires two `Spinorfield_container`s, a class that deals with `std::vector<Spinorfield>`.

## Force filters

 Sometimes during the force calculation, you have on some links gauge fields
 with near zero eigenvalues, and you get gigantic force term, because this turns out 
 to be proportional to the inverse eigenvalue. To prevent local force spikes, this code
 demands an eigenvalue cutoff less than $\delta=5\times10^{-5}$. If the force filter is
 applied too much, this can lower the acceptance rate.

