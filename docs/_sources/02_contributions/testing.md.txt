# Testing the code

This is perhaps the most important aspect of code creation for SIMULATeQCD:
**It is crucially important that you write a test for each new feature that you implement.**
As indicated in the [Code Structure](codeStructure.md#how-to-organize-new-files) document,
new testing `main` programs should go in `src/testing`. **It is also crucially important that
you run ALL tests after you have made major changes.** Please always do this, even if
you are convinced that your changes could not have possibly broken anything.

To make running the tests a bit easier for you, we have included some scripts to do this
automatically. In the `scripts` directory one finds the Bash scripts

1. runTests_0.bash
2. runTests_1.bash
3. runTests_2.bash

which should be run in that order. To compile the tests,
navigate to your build folder and
```shell
make -j tests
```
It may take some time to compile all the tests; in particular for CUDA versions lower than
11.5, this seems to take more than an hour.
Next, you should try to reserve 4 GPUs on whatever machine you are using. (If you are
running locally and only have access to 1 GPU, you can use `runTestsSingleGPU.bash`.
This is better than not doing any test at all, but please not we will not accept your
pull request until we verify all multi-GPU tests pass.)
We have collected test executables within each `runTests` script to run in
less than 30 minutes on Pascal GPUs so that you can test interactively at most
computing centers. When you are ready to run the tests,
```shell
cd testing
../scripts/runTests_0.bash
```
(On one GPU, instead call `../scripts/runTestsSingleGPU.bash`.)
The output for each test is redirected to an `OUT` file. If there are errors for a test,
it will be redirected to a `runERR` file. All empty `runERR` files are deleted. Once you
have run script `0`, run scripts `1` and `2`.

At the end of your test run, if there are no `runERR` files, then you are done.

## Adding your test to the `runTests` scripts

After your test is working, please add it to the last `runTests` script. This way your
test will always be run automatically by future developers. To accomplish this:

1. Make sure that you added your test to the `tests` executables in `CMakeLists.txt`,
i.e. you will need the line `add_to_compound_SIMULATeQCD_target(tests myTest)`. Please
also add an entry to `runTestsSingleGPU.bash`.
2. If your test takes a fixed GPU layout, simply add the entry `testRoutinesNoParam[_myTest]="N"`,
where `N` is the number of GPUs required, to `TEST_run.bash`.
3. Otherwise if you would like to run your test with various GPU layouts, make sure it
has its own `.param` file in the `parameter/tests` with `Nodes` as an adjustable parameter and
4. add the entry `testRoutines[_BulkIndexerTest]="GPUkey"` to `runTests`.

For the last step, the `GPUkey` tells the `runTests` scrips which GPU layouts they should test.
The `GPUkey` consists for a number of GPUs concatenated with one of two layout symbols:

* `s`: Split only in spatial directions. Useful for observables like the Polyakov loop,
where one prefers not to split the lattice in the Euclidean time direction.
* `k`: No long directions. Useful whenever 4 does not divide a lattice extension.

## Using the `testing.h` header file

It is very important that your code yields correct scientific results. One of the most
straightforward ways to carry out such a test is to make a comparison with known results.
For your convenience, there are several functions in `testing.h` for exactly this
purpose.

If you are doing any calculation with affects the `Gaugefield`, for example an
over-relaxation update, please check this using a link-by-link comparison against
some known reference. One can compare `test` and `reference` `Gaugefield` objects
using
```c++
bool compare_fields(test, reference, tol=1e-6)
```
which check that every element of every link of `test` and `reference` agree
within a relative error of `tol`. It returns `true` if every single element of
both `Gaugefield` objects agree.

To compare scalars `test` and `reference` one can use
```c++
void compare_relative(test, reference, rel, prec, text)
```
which makes sure they agree within relative error `rel` as well as absolute
error `prec`. It will display custom `text` along with pass and fail messages.
