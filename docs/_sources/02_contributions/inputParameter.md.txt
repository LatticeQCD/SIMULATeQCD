# How to make an input parameter file

## Using LatticeParameters

After creating your own executable, maybe you want to use an input file with some parameters.
In order to to that, create an input file in `SIMULATeQCD/parameter` with the name:
```
<custom_name>.param
```
Some examples can be find directly in that directory. If you are interested in a very basic input file, you can use
the following set: this one uses, for example, a lattice $20^4$ and a test configuration.

```shell
# This is pre-defined in LatticeParameters in src/base/LatticeParameters.h

#add the lattice dimensions
Lattice = 20 20 20 20

#add the number of GPU's which shall be used in each direction
Nodes = 1 1 1 1

#controlling the Layout of the GPU topology | If you do not know the topology of a given machine, just leave it a 0 0 0 0
Topology = 0 0 0 0

#add the beta value
beta = 6.498

#add the path to the lattice binary
Gaugefile = ../test_conf/l20t20b06498a_nersc.302500

#Configuration Number (optional). This Number will appear in the name of the Output-files
conf_nr = 302500

#format of the binary
format = nersc

#This is for the storage process.
endianness = auto

```

Once you have created your own input file, you have to modify your source code in the following way.
First of all define an object of a parameter class. If you need only basic input parameter you can use, for example:
```C++
LatticeParameters <YourParameterObject>;
```

Then, when you need to read the input file use the following command:
```C++
<YourParameterObject>.readfile(commBase, "../parameter/<YourInputFile>.param",argc,argv);
```

Where `argc` and `argv` are the input parameter of your main function.
If you write the .param file in the right directory, indeed, when you compile your executable, this is copied
in `<YourBuildDirectory>/parameter`. Then, go in your build directory, create a new run directory and then launch your
executable, without input parameter. Notice that your executable is in `<YourBuildDirectory>/<MeaningfulDirectory>`.
Then, the path `../parameter/<YourInputFile>.param` exists and it is always correct.

## RhmcParameters

There is also a class with all the necessary parameters for the RHMC updates. Generate an object with
```C++
RhmcParameters <YourRhmcParameterObject>;
```
This class inherits from the `LatticeParameters` class, so basically everything works like for the `LatticeParameters`
class. A typical .param file should look like this:
```shell
#
# rhmc.param
#
# Parameter file for RHMC runs with HISQ.
#
#      Lattice: Nx Ny Nz Nt
#        Nodes: Number of nodes per direction
#      mass_ud: Light quark mass
#       mass_s: Strange quark mass
#        no_pf: Number of pseudo-fermion fields
#
#    step_size: step size of trajectory
#        no_md: number of steps of trajectory
#   no_step_sf: number of steps of strange quark integration
#        no_sw: number of steps of gauge integration
#      residue: residue for inversions
#        cgMax: max cg steps for multi mass solver
#   always_acc: always accept configuration in Metropolis
#     rat_file: rational approximation input file
#
#    rand_flag: new random numbers(0)/read in random numbers(1)
#    rand_file: file name for random numbers and infos
#         seed: myseed
#    load_conf: flag_load (0=identity, 1=random, 2=getconf)
#   gauge_file: prefix for the gauge configuration's file name
#      conf_nr: configuration number
#   no_updates: number of updates
#  write_every: write out configuration every
#
Lattice = 32 32 32 8
Nodes   = 1 1 1 1
mass_ud = 0.0009875
mass_s  = 0.0790
beta    = 6.285
no_pf   = 1

step_size  = 0.07142857
no_md      = 1
no_step_sf = 1
no_sw      = 1
cgMax      = 20000
always_acc = 0
rat_file   = ../parameter/sample_eo.rat

rand_flag   = 0
rand_file   = rand
seed        = 1337
load_conf   = 2
gauge_file  = ../../test_conf/l328f21b6285m0009875m0790a_019.
conf_nr     = 995
no_updates  = 1
write_every = 11
```


## Rational Approximation Coefficients

There is a class holding the coefficients for the rational approximation, again it is a child class of `LatticeParameters`. To read in rational coefficients, just use
```C++
RationalCoeff <YourRatCoeffObject>;
<YourRatCoeffObject>.readfile(commBase, <YourRhmcParameterObject>.rat_file(), argc, argv);
```
There are two possibilities how to structure this parameter file: 1) Use the "old" syntax:
```C++
r_inv_1f_const = 9.17375410974739
r_inv_1f_num[0] = -1.29980743171857e-05
r_inv_1f_num[1] = -5.96418051684967e-05
r_inv_1f_num[2] = -1.84191327855302e-04
r_inv_1f_num[3] = -5.11036492922649e-04
...
```
or 2) use the syntax for arrays like in the `LatticeParameter` class:
```C++
r_inv_1f_const = 9.17375410974739
r_inv_1f_num = -1.29980743171857e-05 -5.96418051684967e-05 -1.84191327855302e-04 -5.11036492922649e-04 -1.36684253994444e-03 -3.61063551029741e-03 -9.53178316015436e-03 -2.54008994145594e-02 -6.93286757511966e-02 -1.99112173338843e-01 -6.35953935299528e-01 -2.55301677256660 -17.69834374752580 -797.12863440290698
```
However you like, the file has to contain the following keys:
```C++
r_inv_1f_const, r_inv_1f_num, r_inv_1f_den,
r_inv_2f_const, r_inv_2f_num, r_inv_2f_den,
r_1f_const, r_1f_num, r_1f_den,
r_2f_const, r_2f_num, r_2f_den,
r_bar_1f_const, r_bar_1f_num, r_bar_1f_den,
r_bar_2f_const, r_bar_2f_num, r_bar_2f_den.
```

CAVE: The rhmc class assumes that r_1f, r_2f, r_inv_1f and r_inv_2f are of the same order! Same goes for r_bar_1f and r_bar_2f.


## Using your own Parameter Class

If you need more parameters that they are not in the `LatticeParameters` class, consider to create your own parameter class.
Some example of how to construct an input parameter class can be found in `main_gradientFlow.cpp`, in particular see
the `gradientFlowParam` class.

In your executable file define something like that, for each parameter that you need (see the example above). Notice that the
capital words are the one that you have to change:

```C++

template<class floatT>
struct YOUROWNCLASS : LatticeParameters {

  Parameter<TYPE, DIMENSION> NAME_OF_THE_PARAMETER;

  //Constructor
  YOUROWNCLASS() {

    //Each Parameter object has a Parameter.name variable. This set that name to "x"
    add(NAME_OF_THE_PARAMETER, "NAME_OF_THE_PARAMETER");

  }

};
```

Your class should inherit from `LatticeParameters`, where there are the parameters explained in
the previous section. In the code above `TYPE` is the variable type of your
parameter (`float, std::string`...). If you want to write an array of parameters,
than put the `DIMENSION` in the second term of the template input variables.
Of course you have to modify accordingly your input file, adding a new line in the
example written in the first section. If you want to add an array of parameters, you
should separate the elements with a space, as `"Lattice = 20 20 20 20"` in the first example.
In the constructor of `YOUROWNCLASS` for each new parameter you should call `add` or `addDefault`.
This function set the name of the parameter inside the class. If you want to put a default value of
the new parameter, just use `addDefault`, with a third argument that is the default value.

Than, in your main function just declare your new parameter object, in the usual way:

```C++
YOUROWNCLASS<PREC> OBJECT_NAME;
```


Of course you should define the precision at the beginning of your `.cpp` file with, e.g.:

```C++
#define PREC float
```





