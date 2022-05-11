# Indexer

**This page is work in progress!**

When working with 4d lattices, we need a way to map the lattices sites (given by spacetime coordinates x,y,z,t) to the computer memory, which is 1-dimensional. This gets more complicated when using multi-GPU, since we have to care about sub-lattices and halos. Additionally, we often want to split the lattice into the even or odd part. A site is even (odd) if the **<u>sum</u> of its coordinates** is even (odd).

## Terminology

The 1d memory index (which is just an integer) is often called `isite` throughout the code. 
The four spacetime coordinates of a lattice site are stored in a struct named `sitexyzt`.

When using multi-GPU we use the following terminology:

* original lattice (without any splitting or halos): **global** lattice 
* sub-lattice with halos: **full** sub-lattice
* sub-lattice without halos: **bulk** sub-lattice

## Memory layout and basic indexing

The simplest possibility (which we actually don't use, this is just an example) to convert the coordinates `x,y,z,t` to the linear computer memory would be like this:

```{hidden-code-block} C++
size_t siteLocal(const sitexyzt coord) {
    return (coord.x + coord.y*getLatData().vol1 + coord.z*getLatData().vol2 + coord.t*getLatData().vol3);
}
```
In SIMULATeQCD we sometimes want to exploit  symmetries of the Dirac matrix, which requires an **even-odd memory layout** . In the linear memory we first continuously store the data for all of the even sites and then for all of the odd sites. **This is how it is done for all of the base classes like `Gaugefield`, `Spinorfield`, `LatticeContainer`, etc...**
The conversion looks like this:

```{hidden-code-block} C++
size_t siteLocal(const sitexyzt coord) {
    return (((coord.x + coord.y*getLatData().vol1 
    + coord.z*getLatData().vol2 
    + coord.t*getLatData().vol3) >> 0x1) // integer division by 2
    +getLatData().sizeh * ((coord.x + coord.y + coord.z + coord.t) & 0x1)); // 0 if x+y+z+t is even, 1 if odd
}
```

`sizeh` here is the number of of lattice sites divided by 2.

For objects that don't store data on all lattice points, but only the odd part, one needs a mapping like this:

```{hidden-code-block} C++
size_t siteLocal_eo(const sitexyzt coord) {
    return ((coord.x + coord.y*getLatData().vol1
             + coord.z*getLatData().vol2
             + coord.t*getLatData().vol3) >> 0x1);
}
```

This can of course also be used for objects that only store the even part, as adjacent odd and even sites are mapped to same index.

Sometimes one wants to obtain the coordinates from the corresponding 1d memory index. That can be done like this:

```{hidden-code-block} C++
sitexyzt de_site(const size_t site) {
    int x, y, z, t;
    int par, normInd, tmp;

    //! figure out the parity:
    divmod(site, getLatData().sizeh, par, normInd);
    //! par now contains site/sizeh (integer division), so it should be 0 (even) or 1 (odd).
    //! normInd contains the remainder.
    //! Adjacent odd and even sites will have the same remainder.

    //! Now think of an interlaced list of all even and all odd sites, such that the entries alternate
    //! between even and odd sites. Since adjacent sites have the same remainder, the remainder functions as
    //! the index of the *pairs* of adjacent sites.
    //! The next step is now to double this remainder so that we can work with it as an index for the single sites
    //! and not the pairs.
    normInd = normInd << 0x1; //! multiply remainder by two

    //! Now get the slower running coordinates y,z,t:
    //! To get these, we simply integer-divide the index by the product of all faster running lattice extents,
    //! and then use the remainder as the index for the next-faster coordinate and so on.
    divmod(normInd, getLatData().vol3, t, tmp); //! t now contains normInd/vol3, tmp the remainder
    divmod(tmp,     getLatData().vol2, z, tmp); //! z now contains tmp/vol2, tmp the remainder
    divmod(tmp,     getLatData().vol1, y, x);   //! x now contains tmp/vol1, x the remainder

    //! One problem remains: since we doubled the remainder and since the lattice extents have to be even,
    //! x is now also always even, which is of course not correct.
    //! We may need to correct it to be odd, depending on the supposed parity we found in the beginning,
    //! and depending on whether y+z+t is even or odd:
    if (!isOdd(x)){ //TODO isn't x always even? ...
        if (   ( par && !isOdd(y + z + t))    //!  odd parity but y+z+t is even, so x should be odd
               or (!par &&  isOdd(y + z + t))) { //! even parity but y+z+t is  odd, so x should be odd
            ++x;
        }
    }
    //! Note that we always stay inside of a pair of adjacent sites when incrementing x here.

    return sitexyzt(x, y, z, t);
}
```

Obtaining only either even or odd sites from the 1d memory index is done in a very similar way, except we dictate the parity:

```{hidden-code-block} C++
//! "site" should be smaller than sizeh!
sitexyzt de_site_eo(const size_t site, int par) {
    int x, y, z, t;
    int tmp;
    size_t sited = site<<0x1; // multiply by two

    divmod(sited, getLatData().vol3, t, tmp);
    divmod(tmp,   getLatData().vol2, z, tmp);
    divmod(tmp,   getLatData().vol1, y, x);

    if (par && !isOdd(x) && !isOdd(y + z + t))
        ++x;
    if (!par && !isOdd(x) && isOdd(y + z + t))
        ++x;

    return sitexyzt(x, y, z, t);
}
```

## Custom data types to store indices

In SIMULATeQCD, there are four structs that can store the spacetime coordinates and the corresponding memory index of a given lattice site:

`gSite`: 
This struct stores the spacetime coordinates and memory index for one lattice site. 
More specifically, it stores  
* the 1d memory index of the bulk sub-lattice (`isite`)
* the 1d memory index of the full sub-lattice (`isiteFull`)
* the spacetime coordinates of the lattice site on the bulk sublattice (`coords`)
* the spacetime coordinates fo the lattice site on the full sublattice (`coordsFull`)

You can create gSite objects using the static class `GIndexer` (see below). You need to remember with which template parameters you create `gSite` objects, as they don't store any information about that.
If you read somewhere something like "This function takes an odd `gSite` as input", then that means that the `gSite` should have been created using GIndexer<Odd,myHaloDepth>.


````{admonition} Temp
:class: toggle

**The static class `GIndexer`**

**getSite**

With getSite, you can convert an `isite` to a `sitexzyt` and the other way round. You will not directly obtain on or the other, but a `gSite` object that holds both the thing you input as well as the corresponding counterpart. 



Let's say `vol4` is the number of lattice sites.

**Basic functionality (no halos, local lattice)**

**Loop over all sites, or just even/odd part**

```C++
LatLayout myLayout = All;
for (size_t isite=0; isite<vol4; isite++){

gSite mysite = GIndexer<myLayout,HaloDepth>::getSite(isite);

//<do something with gSite here...>
}

```

A site is even (odd) if the **<u>sum</u> of its coordinates** is even (odd).
The **first half of the sites (from `isite=0` to `isite=vol4/2-1`) are even**, and the **second half (from `isite=vol4/2` to `isite=vol4-1`) are odd** , so you can adjust the start and end values of `isite` accordingly if you only want to loop over the even/odd part. 

Examples for even sites:
```
0 0 0 0
2 0 0 0 
4 0 0 0
...
19 17 19 15 
...
20 20 20 20
```

Example for odd sites:
```
1 0 0 0
3 0 0 0 
5 0 0 0
...
19 17 19 20
...
19 20 20 20
```


If you need to get the odd (even) sites, although you are looping from `isite=0` to `isite=vol4/2-1` (from `isite=vol4/2` to `isite=vol4-1`), you can simply change the template parameters of the GIndexer to:

```C++
GIndexer<Odd,HaloDepth>
or
GIndexer<Even,HaloDepth>
```

With `LatLayout Even` (`Odd`) you will always get an even (odd) site from the GIndexer, even if you put in an `isite` which refers to an odd (even) site.

````
