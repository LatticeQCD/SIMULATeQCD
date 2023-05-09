# Generating rational approximation input files

The program in this folder will generate you a rational approximation file for use with the RHMC of SIMULATeQCD.
You can call it with

```shell
rat_approx input.file > output.file
```

The input file should be structured as

```C
npff        // Number of pseudo-fermion flavors

y1
y2
mprec       // Pre-conditioner mass (reduces the condition number in CG)
mq
order1
order2
lambda_low
lambda_high
precision
```

One block will generate three rational approximations according to

f(x) = x^(`y1`/8)  (x+ `mprec`^2 -`mq`^2 )^(`y2`/8)
g(x) = x^(-`y1`/8) (x+ `mprec`^2 -`mq`^2 )^(-`y2`/8)
h(x) = x^(-`y1`/4) (x+ `mprec`^2 -`mq`^2 )^(-`y2`/4)

with m^2 = `mprec`^2 - `mq`^2. For example, consider 2+1 flavors of fermions with
standard Hasenbusch preconditioning for the light flavors. The input file will be

2

3
0
0
ms
14
12
ms^2
5.0
50

2
-2
ms
ml
14
12
ml^2
5.0
160

Two blocks generates six rational approximations. The light approximations are

f(x) = x^(1/4)  (x + ms^2 - ml^2 )^(-1/4)
g(x) = x^(-1/4) (x + ms^2 - ml^2 )^(1/4)
h(x) = x^(-1/2) (x + ms^2 + ml^2 )^(-1/2)

while the strange are

f(x) = x^(3/8)
g(x) = x^(-3/8)
h(x) = x^(-3/4)