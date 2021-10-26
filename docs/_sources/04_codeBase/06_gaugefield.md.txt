# Gaugefield



Gaugefield provides a template who has data member `_lattice` to store the gauge links for all the sites on a 4D lattice. Each link is a SU$($3$)$ matrix of data type `GSU3` . One could use the indexer `GIndexer<All, HaloDepth>` to reach any link, including those in the halo. Usually before or after a lattice calculation one needs to read in or write out the gauge fields from/to a file, and this can be done by the `readconf_nersc` and `writeconf_nersc` function members of this template. @writeconf_nersc@ takes the optional parameter @diskprec@, which can be set to either 1 or 2 for single or double precision, respectively. There are also several function members 

* `one();`
* `random(uint4* rand_state);`
* `gauss(uint4* rand_state);`

initiating the gauge links to some special numbers like unity, random numbers or gaussian algebra elements. Function `swap_memory(Gaugefield<floatT, onDevice, HaloDepth,comp> &gauge)` could swap two gauge fields `_lattice` and `gauge`.  Since the gauge links `_lattice` are `protected`, one can only visit them via `getAccessor()`. Gaugefield template also provides overloading for `=`, so that one could easily copy one gauge field to another. And the gauge links should always be unitary, so a `su3latunitarize` function is given. 

Besides all the functions mentioned above, there are a bunch of  convenient iterators with which one could perform operations on each of the link considered:
```C++
    * void iterateOverFullAllMu(Operator op); //perform "op" on all links including those in the halo in all 4 (spacetime) directions.

    * deviceStream<onDevice> iterateOverBulkAllMu(Operator op, bool useStream = false); //perform "op" on all links excluding those in the halo in all 4 (spacetime) directions.

    * void iterateOverFullLoopMu(Operator op); //perform "op" on all links including those in the halo for the first "Nloops" (default 4) directions.

    * void iterateOverBulkLoopMu(Operator op); //perform "op" on all links excluding those in the halo for the first "Nloops" (default 4) directions.


    * void iterateOverFullAtMu(Operator op);  //perform "op" on all links including those in the halo in a specific direction "Mu".

    * void iterateOverBulkAtMu(Operator op); //perform "op" on all links excluding those in the halo in a specific direction "Mu".

    * void iterateWithConst(Object ob); //set all links including those in the halo in all 4 (spacetime) directions to a constant value "ob".
```
If the above explanation does not clear up your confusion, write an email to htshu@physik.uni-bielefeld.de
