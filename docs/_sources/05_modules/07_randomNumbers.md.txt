# Random Number Generator

The random number generator is using using the hybrid tausworthe generator from [](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch37.html). The state of the random number generator consists of four unsigned integers (We use the GPU type `uint4`) and every site of the lattice has its own state. A random number, e.g. in a Kernel is generated with the function
```C++
floatT get_rand(uint4* state)
```
The argument is a pointer on the `uint4` element on the given site, not on the whole uint4 array that runs over the lattice! 
You now have two choices how to do this: 1) pass your kernel a reference of the rng state object and use the getElement function
```C++
uint4* getElement(gSite site)
```
or 2) pass your kernel a reference of the `uint4` array state (it is a public member of the rng state class) and call
```C++
floatT get_rand(&state[site.isite])
```

## Generating the RNG State

Before drawing random numbers we have to generate the state of the random number generator from a seed. This seed is an unsigned integer that should be specified in the parameter file. The function to set up the
the rng state is
```C++
void make_rng_state(unsigned int seed)
```
It is rather slow, though that does not matter as we have to call it only at the beginning of a completely new run/measurement. It is so slow as it is backwards compatible to the old BielefeldGPU code. 
The state is generated on the Host so remember to copy it to the device! You can use the `=` operator for that.

