# Gradient Flow

Currently there are two different implementations of the gradient flow,
the Wilson flow and the Zeuthen flow (Symanzik improved flow. See
[arxiv:1508.05552](https://arxiv.org/abs/1508.05552)).
The flow can be integrated using a standard Runge Kutta 3 or an adaptive
step size Runge Kutta 3.

To compile the `gradientFlow` executable run:
```
make gradientFlow -j<NumberOfCores>
```
This can take up to 60+ minutes (depending on the Hardware). If you only
want to use the zeuthen force and the adaptive step size Runge-Kutta, you
can also compile `gradientFlow_zeuthen`, which will only take 20 minutes
to compile. (The most compile time is consumed by `Topology.cpp`.)

To run the program, one needs a parameter file. It should take these
parameters:

%```{hidden-code-block} Text
```{admonition} Parameters:
:class: toggle

```Text
Lattice = 20 20 20 20
Nodes = 1 1 1 1
beta = 6.498
Gaugefile = ../test_conf/l20t20b06498a_nersc.302500  # Path to input configuration.
format = nersc                                       # Format of input configuration.
endianness = auto                                    # Endianness of input configuration.
conf_nr = 302500                                     # Configuration number (optional).
force = zeuthen                                      # specify if you want to have the Wilson flow ("wilson") or Zeuthen flow ("zeuthen").
start_step_size = 0.01                               # The (start) step size of the Runge Kutta integration.
RK_method = adaptive_stepsize                        # Set to fixed_stepsize, adaptive_stepsize or adaptive_stepsize_allgpu (see below).
accuracy = 0.01                                      # Specify the accuracy of the adaptive step size method.

measurements_dir = ./                                # Measurement output directory
measurement_intervall = 0 1                          # Flow time Interval which should be iterated.
necessary_flow_times=0.25 0.5                        # Set the flow-times which shouldn't be skipped by the fixed or adaptive step size

ignore_fixed_startstepsize = 0                       # ignore the fixed step size and infer steps izes from necessary_flow_times
save_configuration = 0                               # Save the flowed configuration at each step? (0=no, 1=yes)

binsize = 8                                          # used in the calculation of energy-momentum tensor correlators.

# Set to 1 if you want to measure any of these observables (or 0 if not):
plaquette = 1
clover = 0
cloverTimeSlices = 0
topCharge = 0
topCharge_imp = 0
topChargeTimeSlices = 0
topChargeTimeSlices_imp = 0
energyMomentumTensor = 0
ColorElectricCorrTimeSlices = 0
ColorMagneticCorrTimeSlices = 0

PolyakovLoopCorrelator = 0
GaugeFixTol = 1e-6
GaugeFixNMax = 9000
GaugeFixNUnitarize = 20

```


The parameter `RK_method` specifies the Runge-Kutta integration method.
The options are `fixed_stepsize`, `adaptive_stepsize` (needs only 2 full
Gaugefields on the GPU but is slower) or `adaptive_stepsize_allgpu`
(needs 4 full Gaugefields on the GPU but is faster).

Then program should be executed as follows:
```
srun -n <NoOfGPUs> ./gradientFlow /path/to/parameterFile <optionalParam>
```
where `<optionalParam>`{=html} can be one of the above mentioned
parameters (e.g. `start_step_size=0.02`), which then will be replaced in
the parameter file.

The results of `plaquette`, `clover`, `topCharge` and `topCharge_imp` will be
writer in one (ASCII) output file. `cloverTimeSlices`,
`topChargeTimeSlices`, `topChargeTimeSlices_imp` and
`ColorElectricCorrTimeSlices` will be written in seperate (ASCII) output
files.

## Which flow times do I want?

 ```{admonition} Notation
 :class: toggle

This notation may differ from the notation you find in the literature
(e.g., Lüscher's papers). Remember temperature $T=\frac{1}{aN_\tau}$.

|  lattice spacing                                         |  $a$                                                        |
|  :-------------------------------------------------------|  ---------------------------------------------------------: |
|  "physical" dimensionful flow time                       |  $\tau_\mathrm{F}=t_\mathrm{F} a^2$                         |
|  dimensionless (lattice) flow time                       |  $\tau_\mathrm{F}/a^2\equiv t_\mathrm{F}$                   |
|  dimensionless flow time in terms of fixed temperature   |  $\tau_\mathrm{F} T^2= t_\mathrm{F} / N_\tau^2$             |
|  dimensionless flow radius in terms of fixed temperature |  $\sqrt{8\tau_\mathrm{F}}T = \sqrt{8t_\mathrm{F}}/N_\tau$   |
|  physical separation of operators on the lattice         |  $\tau = at$                                                |
|  dimensionless (lattice) sepration                       |  $\tau/a \equiv t$                                          |

 ```

In this section we briefly describe how to estimate which step size(s),
`necessary_flow_times` and upper limit to use for a given set of
lattices. In the parameter file you always specify the dimensionless
flow time(s) $t_\mathrm{F}$.

-   The leading order solution to the flowed gauge field reads
    $A^\mathrm{LO}_\mu(x,\tau_\mathrm{F}) = \int \mathrm{d}y
    \left(\sqrt{2\pi} \sqrt{8\tau_\mathrm{F}}/2\right)^{-4}
    \exp{\frac{-(x-y)^2}{\sqrt{8\tau_\mathrm{F}}^2/2}}
    A_\mu(y)$, which means that the gauge fields are smeared over a
    spherical extent with radius $\simeq \sqrt{8\tau_\mathrm{F}}$.

-   For a correlation function $G(\tau)$, one can compare the flow
    radius with the separation $\tau$ of the correlation function in
    order to obtain an upper limit for the flow time range. In order for
    the operators to be well separated at a distance $\tau$ the flow
    time should obey $\sqrt{8\tau_\mathrm{F}} \leq \tau/2$. Most
    of the time you will probably have an even stricter upper limit
    because the contamination that is caused by overlapping operators
    (especially with improved discretizations that are non-local) will
    start much earlier.

-   For the lower limit of the flow time you often want
    $\sqrt{8\tau_\mathrm{F}} \geq a $ so that
    $a^2/\tau_\mathrm{F}$-type corrections vanish and the operators
    are fully renormalized by the flow.

In most cases one wants to compare the observables on different lattices
at the same "physical" dimensionful flow time (or radius). Below you can
find two examples on how to achieve this.


 ```{admonition} Example 1: Fixed temperature; different lattice spacings:
 :class: toggle

In order to keep $\tau_\mathrm{F}=a^2 t_\mathrm{F}$ fixed we need
a larger lattice flow time $t_\mathrm{F}$ for smaller lattice
spacing $a$. Since the temperature is fixed in this example, we can
define the flow radii in terms of it and convert them, for each lattice,
to the dimensionless flow times that are then used in the integration.

Let's say we've let the adaptive stepsize algorithm run with a very
small initial stepsize and high accuracy, and saw that our correlation
function is already heavily contaminated for
$\sqrt{8\tau_\mathrm{F}} T = \sqrt{8t_\mathrm{F}}/N_\tau >
t/N_\tau = \tau T / 5$, which we want to use as the upper limit. Here
we've made the inequality dimensionless by multiplying both sides with
the fixed temperature $T$.

-   On a symmetric lattice the maximum value for the separation is
    $\tau T = a t \frac{1}{aN_\tau} = 0.5$, which means that the
    upper flow radius limit is $\sqrt{8\tau_\mathrm{F}} T = 0.5/5 =
    0.1$. Solving for the flow time gives us $\tau_\mathrm{F} = a^2
    t_\mathrm{F} = (\frac{0.1}{T})^2 / 8 = a^2 (0.1 N_\tau)^2 /
    8$. Divide both sides by $a^2$ and insert the corresponding
    $N_\tau$ for each lattice and you obtain the dimensionless flow
    time $t_\mathrm{F}$ that you can put in the parameter file for
    this $N_\tau$.

-   You may also want to compare the observables on different lattices
    at some (or many) fixed intermediate physical flow radii, let's say
    $\sqrt{8\tau_\mathrm{F}} T \in {0.01,0.02, \dots 0.09}$. You
    can compute the dimensionless flow times in the same way as above
    and then provide them via the `necessary_flow_times` parameter.

```


 ```{admonition} Example 2: Fixed lattice spacing; different temperatures:
 :class: toggle

In order to keep $\tau_\mathrm{F}=a^2 t_\mathrm{F}$ fixed we don't
need to change the lattice flow time $t_\mathrm{F}$ for each lattice
since the lattice spacing $a$ is the same for all of them.

-   First we decide again what flow radii
    $\sqrt{8\tau_\mathrm{F}}T$ we want. We then convert those to
    dimensionless flow times $t_\mathrm{F}$ as in the first example,
    but we need to decide for which lattice do to this. The natural
    choice is the the lattice with the lowest temperature (highest
    $N_\tau$), since it will have the largest physical temporal
    extent $aN_\tau$ and thus largest dimensionless flow time
    $t_\mathrm{F}$. We then adjust the upper flow time limit for the
    higher temperature lattices, since those won't allow for as much
    flow, as they have a smaller physical temporal extent $a N_\tau$.

-   This means that for a fixed number of intermediate flow times that
    we want to explicitly measure on the lowest temperature lattice
    (using the `necessary_flow_times` parameter) we maybe won't be able
    to realize all of them on the higher temperature lattices, since
    some of them could be larger than the upper flow time limit (which
    is $N_\tau$ dependent) because of the decreased $N_\tau$. One
    should adjust this upper limit accordingly in order to not waste
    computation time.
```

## Dictating **all** flowtimes manually for improved speed

In order to save GPU memory we can run the `adaptive_stepsize` algorithm
once on one configuration with high accuracy and a small initial
start-stepsize, then save the flow times it visits and use the
`fixed_stepsize` algorithm (which only needs half of the GPU memory)
with those flow times as the `necessary_flow_times` for all other
configurations. By setting the parameter `ignore_fixed_stepsize=1` there
won't be any additional flow time steps in between the
`necessary_flow_times` and we effectively obtain an adaptive stepsize
algorithm while using the fixed stepsize one!

-   For a given `accuracy` (see parameter file) of $10^{-5}$, which
    is rather high, the adaptive stepsize algorithm will, after some
    time (for $t_\mathrm{F} \gtrsim 5$), always make steps with a
    fixed stepsize of $t_\mathrm{F} \approx 0.15$, regardless of the
    lattice spacing.\
    If you dictate all flow times, you should make sure that two
    adjacent dimensionless flow times $t_\mathrm{F}$ are not
    separated by more than 0.15.

-   Additionally, one should make sure that the step sizes are small in
    the beginning and only increase gradually. If you suddenly increase
    the step size by a large amount or start with a too large one, the
    integration will become unstable quickly and fail!

## Observables

In the following, we list some details about some of the observables that can be calculated
using the gradient flow application.


### Topological charge

We use the field theory motivated definition,

$
  Q_L=a^4\sum_x q_L(x),
$

where the sum is over all lattice sites and

$
  q_L(x) = -\frac{1}{2^9\pi^2}\sum\limits_{\mu\nu\rho\sigma=\pm 1}^{\pm 4}
         \tilde{\epsilon}_{\mu\nu\rho\sigma}
         \;\text{tr}\;U^\Box_{\mu\nu}(x)U^\Box_{\rho\sigma}(x).
$

This definition is plagued by UV fluctations, which makes it not generally non-integer before
any gradient flow. Moreover it is a global quantity, so a reasonably chosen amount of
gradient flow, which affects short-distance physics most strongly, is not expected to damage it.

### Energy-Momentum Tensor

The energy-mementum tensor is calculated using blocking method, that is,
splitting the spatial surface into pieces; each piece of size
`binsize * binsize * binsize`. Then the energy-momentum tensor (both the
traceful part and traceless part) is average over each piece and used
for the calculation of EMT correlators in shear and bulk channel.

The improved topological charge is computed using additional rectangles
in the field strength tensor:

```
Plaq-Clover = (1/8)*[Q_{mu,nu}(x) - Q_{mu,nu}^dagger(x)]
Rect-Clover = (1/16)*[R_{mu,nu}(x) - R_{mu,nu}^dagger(x)]
F_{mu,nu}(x) = 5/3 * Plaq-Clover - 1/3 * Rect-Clover,
```

### Polyakov loop correlators

These correlators related to Polyakov loops generally require
[gauge fixing](gaugeFixing.md). You can also learn more in the
[correlator](../05_modules/correlator.md) article.