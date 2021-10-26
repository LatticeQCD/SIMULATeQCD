# Rational Hybrid Monte Carlo


The rational hybrid Monte Carlo class does the update of the gauge fields. The normal user will only have to call the constructor and two functions 
```C++
rhmc(RhmcParameters rhmc_param, Gaugefield<floatT,onDevice,All,HaloDepth> &gaugeField, uint4* rand_state)
void init_ratapprox(RationalCoeff rat)
int update(bool metro=true, bool reverse=false)
```
The constructor has to be called with the usual template arguments and passed an instance of `RhmcParameters`, a the gauge field, and an `uint4` array with the rng state.
The function `init_ratapprox` will set the coefficients for all the rational approximations and has to be called before update!
The function update will generate one molecular dynamics trajectory, if no arguments are passed to `update()` it will also perform a Metropolis step after the trajectory. The Metropolis step can also be omitted by passing the argument `false` to update. This is handy in the beginning of thermalization. The second argument is `false` by default, passing `true` to update will make the updater to run the trajectory forward and backwards for testing if the integration is reversible. 

## Update

The update routine saves a copy of the gauge field, applies the smearing to the gauge field (not implemented yes), builds the pseudo-fermion fields, generates the conjugate momenta and calculates the Hamiltonian. 
Then it starts the molecular dynamics evolution by calling `integrate()` from the integrator class (the integrator object is instantiated by the rhmc constructor). After the MD trajectory the new Hamiltonian is calculated and - depending on the arguments - the Metropolis step is done.


## TODO

- For now, there is no energy density array that can be reduced. But the different parts of the Hamiltonian are reduced separately. This is not ideal, can be changed when there is a floatT field with a `constructAtBulk()` member function.
- Hisq smearing is incooporated yet.
- The function generating the gaussian matrices does not use the operator syntax yet. This is for compatibility with the old code.
