# Topology


You want to distribute your MPI ranks according to your *nvlink* Topology on the respective node.
Usually the ideal mapping should be done automatically, however if this is not the case you can try
to improve it by using the topology parameter.
If you do not want to handle this manually, just set `Topology = 0 0 0 0` or do not specify it at all in your
parameter file.

## How it works

With the topology parameter we can map the cartesian coordinate of an MPI rank to a device, we set
```C++
myInfo.deviceRank = myInfo.coord[0] * Topo[0] + myInfo.coord[1] * Topo[1] + myInfo.coord[2] * Topo[2] + myInfo.coord[3] * Topo[3];
```
E.g. when you want to split your lattice in 8 parts, by e.g. setting `Nodes = 1 1 4 2` you can use `Topology = 0 0 1 4`
to get a "good" mapping of MPI ranks to deviceRanks.

## Some cluster-specific recommendations
If you want to spilt by setting `Nodes = 2 2 2 1` the optimal topology is `Topology = 1 2 4 0`.
With this topology setting you distribute the ranks over
the corners of a cube, just like the `nvlink` Topology on the Bielefeld cluster.
If you want to split the lattice in 6 parts by setting `Nodes = 1 1 2 3` you can use the `Topology = 0 0 4 1`
to distribute 3 Ranks on 3 GPUs on the top face and three ranks on the 3 GPUs on the bottom face of cube spanned by
the `nvlink` topology on the Bielefeld cluster.

On Summit, where you only have 6 GPUs per node and 3 GPUs per socket, you would choose
`Nodes = 1 1 2 3` with `Topology = 0 0 3 1`.
