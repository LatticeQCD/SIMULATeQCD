SIMULATeQCD
===========

The goal of SIMULATeQCD is to establish a multi-GPU interface that makes it relatively straightforward
for developers to implement lattice QCD formulae while still providing the competitive performance.

It is written in C++, allowing developers to write code which is close to hardware level 
(opening many possibilities for optimization), and it is an object-oriented language, 
facilitating organized code structure and encapsulation (making the code human-readable).

In this documentation, we try to cover many of the core features of SIMULATeQCD, some of its modules,
how to use SIMULATeQCD, how to contribute to SIMULATeQCD, and so on. We are not always able to keep
the documentation completely up-to-date, but we do the best we can: 
Remember the code itself is the ground truth.

More information about SIMULATeQCD can be found in [this PhD thesis](https://doi.org/10.4119/unibi/2956493),
[these proceedings](https://doi.org/10.22323/1.396.0196), and 
[this publication](https://doi.org/10.1016/j.cpc.2024.109164).

```{toctree}
---
maxdepth: 1
---
01_gettingStarted/gettingStarted.md
02_contributions/contributions.md
03_applications/applications.md
04_codeBase/codeBase.md
05_modules/modules.md
```


