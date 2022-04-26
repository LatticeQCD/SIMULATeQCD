# Rational Hybrid Monte Carlo

[The rational hybrid Monte Carlo (RHMC)](https://doi.org/10.1016/S0920-5632(99)85217-7) 
algorithm is a way of updating gauge fields when simulating dynamical fermions. 
It uses a [three-step-size integrator](https://doi.org/10.1016/0550-3213(92)90263-B)
that profits from the [Hasenbusch trick](https://doi.org/10.1016/S0370-2693(01)01102-9).
By default integration uses a leapfrog, but it is possible to use an Omelyan
integrator on the largest scale, if you choose.
The inverter is a [multi-shift conjugate gradient](../05_modules/inverter.md).

To use the RHMC class, the user will only have to call the constructor and two functions 
```C++
rhmc(RhmcParameters rhmc_param, Gaugefield<floatT,onDevice,All,HaloDepth> &gaugeField, uint4* rand_state)
void init_ratapprox(RationalCoeff rat)
int update(bool metro=true, bool reverse=false)
```
The constructor has to be called with the usual template arguments and passed 
an instance of `RhmcParameters`, the gauge field, and an `uint4` array with 
the RNG state. The function `init_ratapprox` will set the coefficients for 
all the rational approximations and has to be called before update!
The function update will generate one molecular dynamics (MD) trajectory. 
If no arguments are passed to `update()` it will also perform a Metropolis 
step after the trajectory. The Metropolis step can 
also be omitted by passing the argument `false` to update. This is handy in 
the beginning of thermalization. The second argument is `false` by default; 
passing `true` to update will make the updater run the trajectory forward 
and backwards for testing if the integration is reversible. 

## Update

The update routine saves a copy of the gauge field, applies the smearing to 
the gauge field, builds the pseudo-fermion fields, generates the conjugate 
momenta and calculates the Hamiltonian. 
Then it starts the MD evolution by calling `integrate()` from the integrator 
class (the integrator object is instantiated by the RHMC constructor). After 
the MD trajectory the new Hamiltonian is calculated and - depending on the 
arguments - the Metropolis step is done.

## Multiple pseudo-fermions

When you want to use multiple pseudo-fermion fields, set `no_pf` in the RHMC 
input file to the respective number. Be aware that this changes the way you 
have to construct your ratapprox: In the remez `in.file`, if you want to 
generate Nf flavors using Npf pseudo-fermion fields, you have to use Nf/Npf 
as an input (which is then used Npf times). Note that Nf/Npf must be < 4.
`no_pf` is 1 per default.

## Imaginary chemical potential

The RHMC has the option to generate HISQ lattices with $\mu_B=i\mu_I$. This can be accomplished
by setting the RHMC parameter `mu0` to your desired value. (The default value is 0.)
This can be accomplished straightforwardly in lattice calculations by multiplying time-facing
staggered phases by an appropriate function of $\mu_I$; see for instance
[this work](https://doi.org/10.1016/0370-2693(83)91290-X).

In our code we implement the imaginary phase corresponding to the chemical potential
`chmp` in `staggeredPhases.h` as:
```C++
img_chmp.cREAL = cos(chmp);
img_chmp.cIMAG = sin(chmp);
```
