# CommunicationBase

Communication between multiple CPUs and multiple nodes is handled with MPI, which also allows 
for communication between multiple GPUs.
For NVIDIA hardware, communication via CUDA
GPUDirect P2P for intra-node and CUDA-aware MPI for inter-node channels is supported.
We boost performance by allowing the code to
carry out certain computations while communicating, such as copying halo buffers
into the bulk, whenever possible. A schematic of a typical communication scheme is
shown below.


![alt](../images/communicationChannel.png)