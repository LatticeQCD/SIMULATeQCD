# Gaugefield

The `Gaugefield` class provides a template who has data member `_lattice` to store the
gauge links for all
the sites on a 4D lattice. Each link is an SU$($3$)$ matrix of data type `GSU3`. One could use
the indexer `GIndexer<All, HaloDepth>` to reach any link, including those in the halo.
There are also several function members

* `one();`
* `random(uint4* rand_state);`
* `gauss(uint4* rand_state);`

initiating the gauge links to some special numbers like unity, random numbers or Gaussian
algebra elements. Function `swap_memory(Gaugefield<floatT, onDevice, HaloDepth,comp> &gauge)`
could swap two gauge fields `_lattice` and `gauge`.  Since the gauge links `_lattice` are
`protected`, one can only visit them via `getAccessor()`. `Gaugefield` also provides
overloading for `=`, so that one could easily copy one gauge field to another. And the gauge
links should always be unitary, so a `su3latunitarize` function is given.

Besides all the functions mentioned above, there are a bunch of convenient iterators with which
one could perform operations on each of the link considered:
```C++
    * void iterateOverFullAllMu(Functor op); //perform "op" on all links including those in the halo in all 4 (spacetime) directions.

    * deviceStream<onDevice> iterateOverBulkAllMu(Functor op, bool useStream = false); //perform "op" on all links excluding those in the halo in all 4 (spacetime) directions.

    * void iterateOverFullLoopMu(Functor op); //perform "op" on all links including those in the halo for the first "Nloops" (default 4) directions.

    * void iterateOverBulkLoopMu(Functor op); //perform "op" on all links excluding those in the halo for the first "Nloops" (default 4) directions.

    * void iterateOverFullAtMu(Functor op);  //perform "op" on all links including those in the halo in a specific direction "Mu".

    * void iterateOverBulkAtMu(Functor op); //perform "op" on all links excluding those in the halo in a specific direction "Mu".

    * void iterateWithConst(Object ob); //set all links including those in the halo in all 4 (spacetime) directions to a constant value "ob".
```

## Reading and writing Gaugefields

Usually
before or after a lattice calculation one needs to read in or write out the gauge fields from/to
a file, and this can be done by the `readconf_*` and `writeconf_*` function members of
this template.

Reading/writing NERSC configurations to/from `Gaugefield` objects `gauge` can be
accomplished with
```C++
gauge.readconf_nersc("path/to/gaugefile");
gauge.writeconf_nersc("path/to/gaugefile", rows = 2, diskprec = 1, endianness = ENDIAN_BIG);
```
One can set either `rows` to 2 or 3, depending on whether one wants compressed
configurations. If one stores a compressed configuration, the NERSC reader calculates
the third row of each SU(3) link automatically. The precision `diskprec` can be 1 for
`float` or 2 for `double`.

Reading/writing ILDG configurations to/from `Gaugefield` objects `gauge` can be
accomplished with
```C++
gauge.readconf_ildg("path/to/gaugefile");
gauge.writeconf_ildg("path/to/gaugefile", diskprec = 1);
```
In the case of ILDG configurations, the endianness is fixed to BIG and they are not
compressed by definition, so these are not options that can be passed.

Reading a MILC configuration to a `Gaugefield` object `gauge` is done with
```C++
gauge.readconf_milc("path/to/gaugefile");
```
There is no MILC writer.

To learn more about the various configuration formats, look
[here](configurationIO.md).
