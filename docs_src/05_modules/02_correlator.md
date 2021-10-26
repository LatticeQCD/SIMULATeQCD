# Correlator Class

Here we explain the ideas behind and methods contained in the `Correlator` header file. The classes described here can be found in `src/math/correlators.h`. If you have further questions after reading this wiki article, I suggest you have a look at this class yourself, which is thoroughly commented, and will explain in more detail some things that are not fully explained here.

The main class is the `CorrelatorTools` class, `CorrelatorTools<PREC,true,HaloDepth> corrTools;` which contains all the methods and attributes needed to correlate objects. What objects are to be correlated? These objects are called `CorrField` objects, which are essentially arrays of arbitrary type that inherit from the `LatticeContainer` and include some initialization methods, like the ability to set all elements to `zero()` or `one()`. Where are the results stored? Results are stored in objects called `Correlator` objects, which are basically `CorrField` objects but indexed slightly differently.

You can correlate whatever kinds of mathematical objects `A` and `B` you like; for instance you could correlate Polyakov loops or hadron interpolators. In order to give the user maximum flexibility, I tried to make the calculation of `A` and `B` separate from the correlator class. Therefore the general idea of a correlation calculation is to calculate `A` and `B` ahead of time yourself, then copy `A` and `B` into `CorrField` objects. What is meant by "correlate"? Well, that is up to you. This framework will allow one to calculate any function $\langle f(A_x, B_y)\rangle$, where $f$ is a function that you can design yourself, and $x$ and $y$ run over one of a few possible `domains`. The method `
corrTools.correlateAt< fieldType, corrType, f >("domain", CPUfield1, CPUfield2, CPUnorm, CPUcorr);` will carry out this correlator calculation. Here `fieldType` and `corrType` are the types of the `CorrField` and `Correlator`, respectively. Note that your function $f$ is implemented as a template parameter. Inside the `correlateAt` method, a kernel is called depending on the `domain` you chose. In this method, GPU copies of these CPU arrays will be made and passed to the kernel. The result will be stored in `CPUcorr`.

The `CPUnorm` vector is a `Correlator`-type object that keeps track of how many times a correlator at distance $r$ was calculated. This is something that you should pre-calculate ahead of time using `corrTools.createNorm`; the result will be saved in a normalization file that you can load later in your production runs. This is not particularly elegant, and `createNorm` runs on the CPU only so it's rather slow. One can think of a more elegant way to implement this or a way to speed it up. Nevertheless this procedure works, and after reading in with `corrTools.readNorm`, the correlator calculation seems to be rather fast. (See the benchmarking below for details.)

## Example correlator calculation

Here is an example taken from `src/testing/main_correlatorTest.cu`. In this example we will correlate the $\mu=2$ component of a gauge field with itself using the 4d all-to-all domain. Here $f(A,B)={\rm tr}~(AB)$. 
```C++
gaugeAccessor<PREC> _gauge(gauge.getAccessor());

CorrField<false,GSU3<PREC>> CPUfield1(commBase, corrTools.vol4);
CorrField<false,GSU3<PREC>> CPUfield2(commBase, corrTools.vol4);
Correlator<false,PREC> CPUnorm(commBase, corrTools.UAr2max);
Correlator<false,GCOMPLEX(PREC)> CPUcorrComplex(commBase, corrTools.UAr2max);

for(int m=0; m<corrTools.vol4; m++) {
    _CPUfield3.setValue(m, _gauge.getLink(GInd::getSiteMu(m,2)));
    _CPUfield4.setValue(m, _gauge.getLink(GInd::getSiteMu(m,2)));
}

corrTools.correlateAt<GSU3<PREC>,GCOMPLEX(PREC),AxB<PREC>>("spacetime", CPUfield1, CPUfield2, CPUnorm, CPUcorrComplex, XYswapSymmetry = false);
```
Here, we have set sizes with `vol4`, i.e. $N_s^3\times N_\tau$, and `UAr2max`, which is the largest 4d squared separation possible, given periodic BCs. `UA` stands for "unrestricted all"; this is opposed to "restricted" (`R`) domains that do not include all possible spacetime pairs, and "spatial" (`S`) domains, which only compute spatial pairs. The `XYswapSymmetry` flag is `true` if the correlator is insensitive to interchange of $x$ and $y$, which can happen for example for some Polyakov loop correlators. Setting this flag to `true` may decrease computation time by as much as 70% (see the benchmarking below for details). If you aren't sure what to do, just leave it as `false`.

## Singlet, octet, and average Polyakov loop correlators

This section contains code meant only for constructing certain Polyakov loop correlators. It was implemented by porting some older code into the parallelGPUcode, and therefore is not utilizing the `Correlator` class fully. In principle these could be implemented in the future with the `Correlator` class. Instead, the following methods, part of the more specific `PolyakovLoopCorrelator` class, can be found in `src/modules/gaugeFixing`. One can instantiate this class with, e.g. `PolyakovLoopCorrelator<PREC,true,HaloDepth> PLC(gauge)` Various correlators can be constructed from untraced Polyakov loops. Three examples include the singlet, octet, and average. Some useful information about these correlators can be found [here](https://doi.org/10.1103/PhysRevD.24.450), [here](https://doi.org/10.1103/PhysRevD.33.3738), and [here](https://doi.org/10.1103/PhysRevD.34.3904). The `PolyakovLoopCorrelator` class includes a function to compute these correlators, which can be obtained with the following snippet of code:
```C++
const int distmax=corrTools.distmax;
std::vector<PREC> vec_plca(distmax);
std::vector<PREC> vec_plc1(distmax);
std::vector<PREC> vec_plc8(distmax);
std::vector<int>  vec_factor(distmax);
vec_factor=corrTools.getFactorArray();
corrTools.PLCtoArrays(vec_plca, vec_plc1, vec_plc8);
```
The first line grabs `distmax`$=N_s^2/4+1$ so that the resultant vectors `vec_*` have the correct size. The vectors are indexed by squared distance $r^2$. Some squared separations are not possible on a lattice in 3D; for example $r^2=7$ is not allowed. Therefore several of the entries will be zero. The resultant vectors are normalized inside `PLCtoArrays` by `vec_factor`, which counts the number of contributions to each correlator at a particular $r^2$. Hence, `vec_factor[r2]==0` for every disallowed distance. *NOTE:* `getFactorArray` must be called before `PLCtoArrays`, as `getFactorArray` also initializes the factor array. This initialization routine gives back `vec_factor` because it can be used, for example, if you want to write to file the correlators at allowed distances only. For instance one accomplishes this for the average channel with
```C++
for (int r2=0 ; r2<distmax ; r2++) {
    if (vec_factor[r2]>0) {
        if (commBase.MyRank()==0) {
            plcresultfile << std::setw(7) << dx << "  " << std::setw(13) << std::scientific << vec_plca[r2] << std::endl;
        }
    }
}
``` 
**NOTE:** The singlet and octet Polyakov loop correlations depend on the gauge, and in particular, they should be measured in the Coulomb gauge. This means you must fix the gauge before measuring these quantities! Do learn how to do this, please have a look at the [[Gauge Fixing]] wiki page. Don't forget to re-unitarize before taking your measurements! The singlet, octet, and average correlators are tested in `src/testing/main_gfixplcTest.cu`.

## Some benchmarks: 

This table shows results for the 4d all-to-all tested on `p001`. This is the time it takes to multiply two `GSU3` indentity matrices with themselves, which is a good estimate for how long it takes to correlate two `CorrFields` of type `GSU3` on the lattices of the given size. For reference, 1 minute is 60 000 ms.

**Symmetric Results:**
| $N_s$ | $N_\tau$ | UA id3 x id3 [ms] | US id3 x id3 [ms] |
|:----- | -------- | ----------------- | ----------------: |
|8      |4	       |12                 | 5                 |
|24     |6         |3240               | 78                |
|24     |8         |5474               | 78                |
|20     |20        |9708               | 48                |
|32     |6         |15306              | 470               |
|32     |8         |25473              | 469               |
|40     |8         |96564              | 1901              |
|48     |6         |172147             | 5170              |
|48     |8         |285101             | 5174              |
|42     |12        |270474             | 2420              |
|48     |12        |594554             | 5183              |
|56|	8          |732532             | 13296             |

**Asymmetric Results:**
| $N_s$ | $N_\tau$ | UA id3 x id3 [ms] | US id3 x id3 [ms] |
|:----- | -------- | ----------------- | ----------------: |
|8      |4         |15                 |5                  |
|24     |6         |4027               |97                 |
|24     |8         |6746               |96                 |
|20     |20        |13410              |47                 |
|32     |6         |22265              |633                |
|32     |8         |37821              |632                |
|40     |8         |155048             |2465               |
|48     |6         |283987             |8014               |
|48     |8         |476607             |8016               |
|42     |12        |448504             |3896               |
|48     |12        |1007574            |8012               |
|56     |8         |1239982            |18401              |
