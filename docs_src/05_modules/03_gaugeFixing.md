# Gauge Fixing

The gauge fixing class allows one to gauge fix using [over-relaxation (GFOR)](https://doi.org/10.1016/0370-2693(90)90031-Z). A more recent and detailed discussion especially with respect to multi GPUs can be found [here](https://doi.org/10.1016/j.cpc.2013.03.021).

To use this class include the `src/gfixing/gfix.h` header file and include `src/gfixing/gfix.cpp` as a source file for your program in `CMakeLists.txt`.  The gauge fixing class is initialized with, for example,
```C++
GaugeFixing<PREC,true,HaloDepth> GFixing(gauge); 
```
and one GFOR gauge fixing step for the entire lattice can be performed with
```C++
GFixing.gaugefixOR();
```
Gauge fixing on the lattice is generally implemented as follows: There is a functional $F$ of the gauge field that is extremized if and only if the gauge fixing condition is satisfied. So, the idea is to evolve the lattice toward the minimum of $F$ using GFOR sweep. The value of $F$ can be obtained by
```C++
GFixing.getAction();
```
The closeness of the configuration to the desired gauge is measured with a gauge fixing quality $\theta$. One iterates the GFOR sweep until $\theta$ falls below some desired threshold. To obtain the value of $\theta$, use
```C++
GFixing.getTheta();
```
You can decide a good threshold for your purposes looking out for a window where the results for your observable of interest become relatively stable under GFOR sweeps. An undesirable artifact of this GFOR implementation is that the links lose their unitarity as time goes on. Therefore one should reunitarize the lattice every so often; for instance one can reunitarize every 20 sweeps. To reunitarize the lattice, call
```C++
gauge.su3latunitarize();
```
Don't forget to do one final unitarization before making any measurements! The gauge fixing is tested in `src/testing/main_gfixplcTest.cpp`.

## Minimal Working Example

Here's a short segment of code that will fix to the Coulomb gauge, assuming you've already initialized everything else like the Gaugefield:

```C++
Gaugefield<PREC,true,HaloDepth>   gauge(commBase);
GaugeFixing<PREC,true,HaloDepth>    GFixing(gauge); 
int ngfstep=0;
PREC gftheta=1e10;
const PREC gtol=1e-6;          /// When theta falls below this number, stop...
const int ngfstepMAX=9000;     /// ...or stop after a fixed number of steps; this way the program doesn't get stuck.
const int nunit=20;            /// Re-unitarize every 20 steps.
gauge.updateAll();
while ( (ngfstep<ngfstepMAX) && (gftheta>gtol) ) {
    /// Compute starting GF functional and update the lattice.
    GFixing.gaugefixOR();
    /// Due to the nature of the update, we have to re-unitarize every so often.
    if ( (ngfstep%nunit) == 0 ) {
        gauge.su3latunitarize();
    }
    /// Re-calculate theta to determine whether we are sufficiently fixed.
    gftheta=GFixing.getTheta();
    ngfstep+=1;
}
gauge.su3latunitarize(); /// One final re-unitarization.
```
