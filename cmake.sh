CXX=hipcc \
cmake ../ \
-DARCHITECTURE="gfx90a" \
-DUSE_GPU_AWARE_MPI=ON \
-DUSE_GPU_P2P=OFF \
-DUSE_CUDA=OFF \
-DUSE_HIP_AMD=ON \
-DMPI_CXX_COMPILER=${MPICH_DIR}/bin/mpicxx \
-DMPI_C_COMPILER=${MPICH_DIR}/bin/mpicc
#-DMPI_CXX_LIBRARIES="{MPICH_DIR}/lib;$MPICH_DIR/gtl/lib/libmpi_gtl_hsa.so" \

