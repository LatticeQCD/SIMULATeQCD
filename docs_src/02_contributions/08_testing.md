# Testing the code

This is perhaps the most important aspect of code creation for our ParallelGPUCode: *It is crucially important that you write a test for each new feature that you implement.* As indicated in the [Code Structure](03_organizeFiles.md#how-to-organize-new-files) wiki, new testing `main` programs should go in `src/testing`. *It is also crucially important that you run ALL tests after you have made major changes.* Please always do this, even if you are convinced that your changes could not have possibly broken anything, since

1. you might be wrong about that, and
2. someone else might have been lazy and not run the tests after they made changes.

To make running the tests a bit easier for you, we have included some scripts to do this automatically. In the `scripts` directory one finds `TEST_run.bash` as well as a job script `jobscript_TEST_run`. To run the tests navigate to your build folder and
```shell
make -j tests
cd testing
sbatch jobscript_TEST_run
```

Note that the tests will take some time to compile, usually more than an hour. The jobscript requests the `devel_gpu` queue with 4 GPUs. It calls `TEST_run.bash`, which runs the tests with different numbers of processors and GPU layouts with up to 4 GPUs. The output for each test is redirected to an `OUT` file. If there are errors for a test, it will be redirected to a `runERR` file. All empty `runERR` files are deleted.

At the end of your test run, if there are no `runERR` files, that is a very good sign. A more careful check requires you to search through the `OUT` files for `fail`, `FAIL`, or `Fail`, since I am not sure whether every test script was written to return an error flag when a test fails.

## Adding your test to the `TEST_run` script

After your test is working, please add it to the `TEST_run.bash` script. This way your test will always be run automatically by future developers. To accomplish this:

1. Make sure that you added your test to the `tests` executables in `CMakeLists.txt`, i.e. you will need the line `add_to_compound_ParallelGPU_target(tests myTest)`.
2. If your test takes a fixed GPU layout, simply add the entry `testRoutinesNoParam[_myTest]="N"`, where `N` is the number of GPUs required, to `TEST_run.bash`.
3. Otherwise if you would like to run your test with various GPU layouts, make sure it has its own `.param` file in the `parameter/tests` with `Nodes` as an adjustable parameter and
4. add the entry `testRoutines[_BulkIndexerTest]="GPUkey"` to `TEST_run.bash`.

For the last step, the `GPUkey` tells `TEST_run.bash` which GPU layouts it should test. The `GPUkey` consists for a number of GPUs concatenated with one of two layout symbols:

* `s`: Split only in spatial directions. Useful for observables like the Polyakov loop, where one prefers not to split the lattice in the Euclidean time direction.
* `k`: No long directions. Useful whenever 4 does not divide a lattice extension.
