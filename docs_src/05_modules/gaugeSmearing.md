# HISQ smearing


For HISQ(Highly Improved Staggered Quarks) like action we need two stage of Asqtad like smearing and a unitarization in between them. i.e. 

Lets say the original gaugefields are U.

i) We smeared U links with various 3link, 5link and 7 link paths, In references it is called V

```C++
HisqSmearing<PREC, USE_GPU,HaloDepth> V(gauge_in, gauge_out, redBase);
```


To do this use 
```C++
V.hisqSmearing(getLevel1params())
```


II) We project back to U(3) the level 1 smeared links i.e the V links, In references it is called W

```C++
HisqSmearing<PREC, USE_GPU,HaloDepth> W(gauge_in, gauge_out, redBase);
```

To do this use  
```C++
W.u3Project()
```
 


III) We smeared  again the W links with 3link, 5link and 7 links paths, In references it is called X

```C++
HisqSmearing<PREC, USE_GPU,HaloDepth> X(gauge_in, gauge_out, redBase);
```

To do this use  
```C++
X.hisqSmearing(getLevel2params())
```
 

To do the tadpole improvement we use naik term in the dirac operator, for HISQ dslash the naik links are construct from the unitarize links i.e using the W links.  In reference this is called N
To use this we have to use,

```C++
HisqSmearing<PREC, USE_GPU,HaloDepth> N(gauge_in, gauge_out, redBase);   
 N.naikterm()
```

The smearing classes is implemented in 
> hisqSmearing.cpp.

How to constructs the smeared links are defined in 
```shell
main_HisqSmearing.cpp. 
```

Ref. [hep-lat](https://doi.org/10.1103/PhysRevD.82.074501)
