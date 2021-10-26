# How to run


## On a cluster using `slurm`

If you are on a cluster that uses slurm, e.g. the Bielefeld GPU cluster, then, inside of your sbatch script do not use mpiexec or mpirun, but instead do
```shell
srun -n <NoGPUs> ./<program> 
```

## Special rules for Job-Scripts on the Bielefeld cluster

See the [Bielefeld GPU Cluster wiki](https://rmp.physik.uni-bielefeld.de/projects/cluster/wiki)

## On your local machine (desktop, laptop, ...)

Currently the SIMULATeQCD has only multi-GPU support. Therefore, any program has to be launched using mpirun or mpiexec. 
For example:
```shell
mpiexec -np <NoGPUs> ./<program> 
```
where `<NoGPUs>` is the number of GPU's you want to use.



