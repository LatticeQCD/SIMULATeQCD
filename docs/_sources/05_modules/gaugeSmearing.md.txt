# HISQ smearing

Much of the RHMC in SIMULATeQCD follows the MILC code. For a thorough discussion of this
implementation, look [here](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.82.074501).
Here we just list a few important details.

HISQ fermions utilize two levels of asqtad-like smearing, with a unitarization between them.
The first-level link treatment is

$  c_1 = 1/8$

$  c_3 = 1/16$

$  c_5 = 1/64$

$  c_7 = 1/384,$    

where $c_1$ is the coefficient for the 1-link, and $c_3$, $c_5$, and $c_7$
are for the 3-link staple, 5-link staple, and 7-link staples, respectively. 
The first-level smeared link is then projected
back to U$(3)$ before the application of the second-level smearing.
The second level uses

$  c_1 = 1$ 

$  c_3 = 1/16$

$  c_5 = 1/64$

$  c_7 = 1/384$

$  c_\text{Lepage} = -1/8$

$  c_\text{Naik} = -1/24+\epsilon/8,$

where $c_\text{Naik}$ and $c_\text{Lepage}$ are the coefficients for the 
Naik and Lepage terms.


Let's see how this is carried out in the code. Let the original gaugefields be $U$.
For the first level, we smear $U$ links with various 3-link, 5-link and 7-link paths. MILC calls it $V$
```C++
HisqSmearing<PREC, USE_GPU,HaloDepth> V(gauge_in, gauge_out, redBase);
```
To do this use 
```C++
V.hisqSmearing(getLevel1params())
```
Next we project the level 1 smeared links back to U(3). MILC calls it $W$
```C++
HisqSmearing<PREC, USE_GPU,HaloDepth> W(gauge_in, gauge_out, redBase);
```
To do this use  
```C++
W.u3Project()
```
Finally we smear again the $W$ links with 3-link, 5-link and 7-link paths. MILC calls it $X$
```C++
HisqSmearing<PREC, USE_GPU,HaloDepth> X(gauge_in, gauge_out, redBase);
```
To do this use  
```C++
X.hisqSmearing(getLevel2params())
```
 

To do the tadpole improvement we use the Naik term in the Dirac operator. For HISQ dslash, 
the Naik links are constructed from the unitarized links, i.e. using the $W$ links.  
MILC calls it $N$.
To use this we have to call
```C++
HisqSmearing<PREC, USE_GPU,HaloDepth> N(gauge_in, gauge_out, redBase);   
 N.naikterm()
```
The smearing classes are implemented in 
> hisqSmearing.cpp.
How to construct the smeared links are defined in 
```shell
main_HisqSmearing.cpp. 
```