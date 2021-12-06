# Multi-level algorithm


The Multi-level algorithm has been implemented according to Martin Lüscher and Peter Weisz's idea of sub-lattice updates, see  [hep-lat/0108014v1](https://doi.org/10.1088/1126-6708/2001/09/010) (and also Harvey Meyer's paper [hep-lat/0209145](https://doi.org/10.1088/1126-6708/2003/01/048)). There are two sub-lattice updates: HB and OR,  same as the standard ones [Gauge Updates (HB and OR)](../05_modules/04_gaugeUpdates.md#gauge-updates-hb-and-or) but performed within a sub lattice. For details see luscherweisz.h and luscherweisz.cpp.

After the sub-lattice updates, the observables will be measured. Currently the calculations of energy-momentum tensor, polyakovloop and color-electric correlators have been implemented. For details, see SubLatMeas.h and SubLatMeas.cpp.

To compile, one can `make` the executable `sublatticeUpdates`. You can then find it under `applications/sublatticeUpdates`. The example parameter file is found under `parameter/sublatticeUpdates.param` and looks like this:
```shell
#parameter file for sublatticeUpdates
Lattice = 20 20 20 20
Nodes = 1 1 1 1
beta = 6.498
Gaugefile =../test_conf/l20t20b06498a_nersc.302500
conf_nr = 302500
format = nersc
endianness = auto
sublattice_lt = 6
num_meas = 100
num_update = 10
min_dist = 0
PzMax = 0
VacuumSubtractionHelper = 1.72
out_dir = ./
EnergyMomentumTensorCorr = 0
ColorElectricCorr = 0

```

Calling `./sublatticeUpdates sublatticeUpdates.param` will output the multi-level improved energy-momentum tensor correlators in both shear and bulk channel and also the traceless part of the EMT and the trace anomaly (which can be used to remove the disconnected part of the correlators), polyakovloop and color-electric correlators, all  saved in plain text.  The corresponding observables calculated without multi-level improvement will also be given.

`sublattice_lt` in the above parameter file means the temporal extension of a sub lattice. The spatial extensions of a sub lattice are the same as the full lattice.  `num_update` means after how many sweeps(1 sweep = 1*HB + 4*OR) the observables will be measured. The total number of measuring is set by `num_meas`. 

`min_dist` means the minimum distance between the right boundary of the left sub-lattice and the left boundary of the right sub-lattice.  For instance when `min_dist` is set to 0, the smallest time distance of the color-electric correlator available will be 3. This is because the `square` within a sub lattice can not hit the boundary. For the similar reason the smallest time distance for the EMT correlator is 4. 

`PzMax` means the maximum integer value of the finite momentum in z-direction. Used only in the calculation of energy-momentum correlators. For instance if we set it to 2, then we will have correlators at pz=0,1,2. Here we only consider z-direction because it is easy to implement for shear channel correlators.

`VacuumSubtractionHelper` is only bulk channel involved. it is a magic value used to subtract the first a few useless digits appearing in the trace anomaly at zero momentum, as they do not contribute the bulk correlators and introduce numerical problem. VacuumSubtractionHelper can be obtained by a test run of this program on any of the configurations of the same lattice size and beta with 0 sublattice updates. The magic number is given by the trace anomaly in the output file. 


Setting `EnergyMomentumTensorCorr=0` will calculate the energy-momentum tensor correlators. At the same time the trace anomaly will be calculated, too.
Setting `ColorElectricCorr=0` will calculate the color-electric correlators. At the same time the polyakov loop will be calculated, too.
