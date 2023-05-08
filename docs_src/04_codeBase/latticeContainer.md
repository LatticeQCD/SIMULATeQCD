# Lattice Container

This class oversees `LatticeContainer` objects, which are essentially intermediate containers used to store intermediate
results that will be reduced later. For instance if one calculates the action, one first finds each local
contribution, sums these contributions over a sublattice, then sums this result over all sublattices.
This whole process is carried out with `reduce` call.

The `LatticeContainer` can hold elements of arbitrary type, and it is spread over the processes in a similar
way as the [Gaugefield](gaugefield.md#gaugefield) or Spinorfield. The memory of
the `LatticeContainer` is by default shared with the memory of the halo buffer, because in general the
intermediate results have to be re-calculated after a halo update. As of this writing, the available reduction methods are
```C++
/// Reduce only timeslices. The result is a vector of length N_tau
void reduceTimeSlices(std::vector<elemType> &values)
/// Methods to reduce intermediate results of a stacked spinor. Currently no support for halo exchange.
void reduceStackedLocal(std::vector<elemType> &values, size_t NStacks, size_t stackSize)
void reduceStacked(std::vector<elemType> &values, size_t NStacks, size_t stackSize)
/// Reduce full lattice. Supports halo exchange.
void reduce(elemType &value, size_t size, bool rootToAll = false)
void reduceMax(elemType &value, size_t size, bool rootToAll = false)
```
At some point you may need to write into or extract from your `LatticeContainer`. This can be accomplished
with `LatticeContainerAccessor` objects. Once you have created your accessor object corresponding to
your `LatticeContainer`, the methods `getElement` and `setElement` allow you to interact with it.

A basic implementation of a 100-element `GSU3<PREC>` `LatticeContainer` on the GPU could be
```C++
LatticeContainer<true,GSU3<PREC>> latty(commBase);
latty.adjustSize(100);
```
Then you can set element 3 of `latty` to `test`, or store the contents of element 3 in `test`, using respectively
```C++
LatticeContainerAccessor _latty(latty.getAccessor());
_latty.setElement(3,test);
_latty.getElement(3,test);
```

In general if you need to create some general array-like class, it is recommended that you inherit
from the `LatticeContainer`. Besides giving you the convenience of not having to rewrite reduction code,
it also already has multi-processor functionality built in.
The [Correlator Class](../05_modules/correlator.md#correlator-class), for example,
is implemented by inheriting from the `LatticeContainer`.
