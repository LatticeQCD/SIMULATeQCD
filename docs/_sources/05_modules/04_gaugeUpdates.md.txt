# Gauge Updates (HB and OR)

The pure gauge update class contains both heatbath (HB) and overrelaxation (OR) updating for pure gauge fields. The heatbath is implemented using the [Kennedy-Pendleton](https://doi.org/10.1016/0370-2693(85)91632-6) algorithm, which is extended from SU2 to SU3 via the method of [Cabbibo and Marinari](https://doi.org/10.1016/0370-2693(82)90696-7).

To use this class include the `src/gauge/PureGaugeUpdates.h` header file and include `src/gauge/PureGaugeUpdates.cu` as a source file for your program in `CMakeLists.txt`.  The pure gauge update class is initialized with, for example,
```C++
GaugeUpdate<PREC,true,HaloDepth>    gUpdate(gauge); 
```
and one OR sweep of the entire lattice can be performed with
```C++
gUpdate.updateOR();
```
The HB update requires you to initialize a random number generator on the host, then pass the state to the device. This can be done by

```C++
int seed = 0;
grnd_state<false> host_state;
grnd_state<true>  dev_state;
host_state.make_rng_state(seed);
dev_state = host_state;
```
More information about the random number generator can be found in [Random Number Generator](07_randomNumbers.md#random-number-generator). The state is then passed as an argument to the HB function as
```C++
gUpdate.updateHB(dev_state.state,beta);
```


## Some benchmarks

The following use `HaloDepth=1`. Each sweep consists of 1 HB with 4 OR updates. Times are measured in [ms]. Error bars are in the last digits in parentheses. Each timing uses 50 sweeps. Each number given is an average time from between 3 and 4 test runs. Timing was done with the SIMULATeQCD code's built-in timer. Only hyperplanes and planes are communicated. Originally the tests were carried out on NVIDIA Pascal GPU, but more tests were carried out later on NVIDIA VolNVIDIA Volta GPU. Both results are included because maybe it's interesting to see the improvement from the old hardware to the new hardware. Attached are plots of improvement $I$ versus number of GPUs for both machines, where I define improvement as $I=\frac{\text{number of GPUs}}{\text{time}/\text{1 GPU time}}$

## Pascal CPU 16 GB

1 processor: $68^4$:
| no split     |
| :----------: |
| 106 855(3)   |

2 processor: $136\times68^3$:
| x split | y split | z split | t split |
| :------ | ------- | ------- | ------: |
| 171 726(8)    |  154 139(8)   |  152 932(3) | 152 064(5) |

4 processor: $136^2\times68^2$:
| xy split | xz split | xt split | yz split | yt split | zt split |
| :------- | -------- | -------- | -------- | -------- | -------: |
| 179 900(4) | 179 590(370) | 178 833(7) | 163 206(2) | 163 480(220) | 162 950(500) |

## Volta GPU 32GB

1 processor: $68^4$:
| no split     |
| :----------: |
| 71 603(40)   |

2 processor: $136\times68^3$:
| x split | y split | z split | t split |
| :------ | ------- | ------- | ------: |
| 135 389(12)   | 110 609(13)   |  109 598(16) | 109 018(29) |

4 processor: $136^2\times68^2$:
| xy split | xz split | xt split | yz split | yt split | zt split |
| :------- | -------- | -------- | -------- | -------- | -------: |
| 143 664(16) | 143 432(16) | 143 138(11) | 120 595(34) | 120 581(12) | 120 180(15) |

4 processor: $272\times68^3$:
| x split | y split | z split | t split | 
| :------ | ------- | ------- | ------: |
| 134 423(17) | 99 072(30) | 98 397(22) | 97 810(29) |
