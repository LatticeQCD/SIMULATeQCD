# Generate quenched gauge configurations

To learn more about the specifics of the gauge updates, take a look at [Gauge Updates (HB and OR)].
To generate quenched gauge configurations, `make` the executable `GenerateQuenched`. You can then find it under `applications/GenerateQuenched`. The example parameter file is found under `parameter/GenerateQuenched.param` and looks like this:
```shell
#parameter file for GenerateQuenched
Lattice = 64 64 64 16
Nodes = 1 1 1 1
beta = 6.87361
format = nersc
endianness = auto
stream = a
output_dir = .
nconfs = 1000
nsweeps_ORperHB = 4
nsweeps_HBwithOR = 500

start = one
nsweeps_thermal_HB_only = 500
nsweeps_thermal_HBwithOR = 4000

#conf_nr = 500
#prev_conf = conf_s064t16_b0687361_a_U000500
#prev_rand = rand_s064t16_b0687361_a_U000500
```


Calling `./GenerateQuenched GenerateQuenched.param` will output gauge configurations (in nersc format with double precision and 2/3 compression) and their corresponding random number state in the folder `output_dir`. The output looks like this: e.g. 
```shell
conf_s064t16_b0687361_a_U000500
conf_s064t16_b0687361_a_U001000
rand_s064t16_b0687361_a_U000500
rand_s064t16_b0687361_a_U001000
...
```
The configuration number is inferred from `nsweeps_HBwithOR`, i.e. the first configuration is labeled with the suffix "`_U<nsweeps_HBwithOR>`" and the following configuration numbers increase in steps of `nsweeps_HBwithOR`. You should always specify the stream name by setting the parameter `stream`.

With `nconfs` you can specify how many configurations should be generated before the program stops.
With `nsweeps_ORperHB` you can specify how many OverRelaxation updates should be done for each HeatBath update.
With `nsweeps_HBwithOR` you can specify how many HB updates (with `nsweeps_ORperHB` OR updates per HB) should be done between each saved configuration.

**Thermalization parameters** (required when starting a new stream):
With `start` you can specify with which kind of configuration the thermalization should start. The options are `one` (all links = unity matrix), `fixed_random` (all links = same random SU3 matrix) and `all_random` (a random configuration of SU3 matrices). 
With  `nsweeps_thermal_HB_only` you can specify how many pure HB updates should be done after the cold start.
With `nsweeps_thermal_HBwithOR` you can specify how many HBOR updates should be done after the pure HB updates.
The seed for the random number generator is the time since Unix epoch in milliseconds and is output to stdout.

**Parameters for resuming a previous run** : 
With `prev_conf` you can specify the path of the last configuration you generated.
With `conf_nr` you can specify the number of this last configuration, so that the next configuration's number will be ( previous_number + `nsweeps_ORperHB` ). Don't forget to set `stream` to the correct value.
With `prev_rand` you can specify the path of the according random number state of that configuration. If you don't specify this then a new random number state will be generated (The seed for this is also time since Unix epoch in milliseconds and is output to stdout.)




