# Gauge Fixing

The gauge fixing class allows one to gauge fix using [over-relaxation (GFOR)](https://doi.org/10.1016/0370-2693(90)90031-Z).
A more recent and detailed discussion especially with respect to multi GPUs can be found [here](https://doi.org/10.1016/j.cpc.2013.03.021).
The over-relaxation proceeds by updating SU(2) subgroups of SU(3) matrices. We use over-relaxation parameter $\omega=1.3$.
These gauges are defined by
$\partial_\mu A_\mu=0$,
where $\mu\in\{1,2,3\}$ for the Coulomb gauge and $\mu\in\{1,2,3,4\}$ for the Landau. The gauge can be specified in the code
by setting the `D_FIX` and `I_FIX` parameters in `gfix.h`.

To use this class include the `src/gfixing/gfix.h` header file and include `src/gfixing/gfix.cpp` as a source file for your
program in `CMakeLists.txt`.  The gauge fixing class is initialized with, for example,
```C++
GaugeFixing<PREC,true,HaloDepth> GFixing(gauge);
```
and one GFOR gauge fixing step for the entire lattice can be performed with
```C++
GFixing.gaugefixOR();
```
Gauge fixing on the lattice is generally implemented as follows: There is a functional $F$ of the gauge field that is
extremized if and only if the gauge fixing condition is satisfied. So, the idea is to evolve the lattice toward the minimum of $F$
using GFOR sweep. The value of $F$ can be obtained by
```C++
GFixing.getAction();
```
The closeness of the configuration to the desired gauge is measured with a gauge fixing quality $\theta$. One iterates
the GFOR sweep until $\theta$ falls below some desired threshold. To obtain the value of $\theta$, use
```C++
GFixing.getTheta();
```
You can decide a good threshold for your purposes looking out for a window where the results for your observable of interest
become relatively stable under GFOR sweeps. An undesirable artifact of this GFOR implementation is that the links lose their
unitarity as time goes on. Therefore one should re-unitarize the lattice every so often; for instance one can re-unitarize every
20 sweeps. To re-unitarize the lattice, call
```C++
gauge.su3latunitarize();
```
Don't forget to do one final unitarization before making any measurements! The gauge fixing is tested in
`src/testing/main_gfixplcTest.cpp`.

## Minimal working example

Here's a short snippet of code that will fix the gauge, assuming you've already initialized everything else like the Gaugefield.
This snippet contains the main features needed for gauge fixing.

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

## Gauge fixing application

Under `src/applications/main_gaugeFixing.cpp` you find our main program for gauge fixing. There are already implemented some
observables that depend on the gauge, in particular Polyakov loop and Wilson line correlators. (You can learn a bit more
about the implementation of Polyakov loop correlators [here](../05_modules/correlator.md).) The available options are
given under `parameter/applications/gaugeFixing.param`, which has options like
```shell
gtolerance       = 1e-6  # stop gauge fixing below this theta
maxgfsteps       = 3000  # or after this many steps
numunit          = 20    # re-unitarize every 20 steps
polyakovLoopCorr = 1     # measure Polyakov loop correlators
WilsonLineCorr   = 0     # don't measure Wilson lines
SaveConfig       = 0     # don't saved the gauge-fixed configuration
```
If you would like to implement your own observable in this application, please follow the examples of the Wilson line
and Polyakov loop correlators.

## Polyakov loop correlators

The Polyakov loop is gauge-invarant, but several correlators related to the Polyakov loop are not, and are of
interest to renormalize the Polyakov loop or extract the Debye mass.
The gauge invariant color-averaged Polyakov loop correlator can be decomposed
into color singlet $F_1$ and color octet $F_8$
contributions. In particular

$
  \exp\left[-\frac{F_{q\bar{q}}(r,T)}{T}\right]
  =\frac{1}{9}\exp\left[-\frac{F_1(r,T)}{T}\right]
   +\frac{8}{9}\exp\left[-\frac{F_8(r,T)}{T}\right],
$

where

$
 \exp\left[-F_1(r,T)/T\right]
   =\frac{1}{3}\left<\text{tr}\; L_{\vec{x}}L^\dagger_{\vec{y}}\right>
  \exp\left[-F_8(r,T)/T\right]
    =\frac{9}{8}\left< P_{\vec{x}}P^\dagger_{\vec{y}}\right>
     -\frac{1}{24}\left< \text{tr}\; L_{\vec{x}}L^\dagger_{\vec{y}}\right>,
$

which clearly depends on the gauge. More information about the implementation of these
observables can be found [here](../05_modules/correlator.md).
