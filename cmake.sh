CXX=hipcc \
cmake ../parallelgpucode/ \
-DARCHITECTURE="90" \
-DUSE_CUDA_AWARE_MPI=OFF \
-DUSE_CUDA_P2P=OFF \
