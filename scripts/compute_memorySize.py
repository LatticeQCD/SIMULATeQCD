# 
# compute_memorySize.py                                                               
# 
# Short python script to figure out how much memory your parallelGPUCode executable will require.
# 
# 
import numpy as np
import argparse
import math

np.set_printoptions(precision=4)

argparser = argparse.ArgumentParser()
argparser.add_argument('--glob_lat', nargs=4, type=int)
argparser.add_argument('--split', nargs=4, type=int)
argparser.add_argument('--halodepth', type=int)
argparser.add_argument('--BytesPerFloat', type=int, default=8)
argparser.add_argument('--compression', type=float, default=1)
argparser.add_argument('--forceHalos', default=False, action="store_true") 
argparser.add_argument('--n_gaugefields', type=int, default=1)

args = argparser.parse_args()

print("Global Lattice: ", args.glob_lat)
print("Split dimensions: ", args.split)
print("HaloDepth: ", args.halodepth)
print("Bytes per float: ", args.BytesPerFloat)
print("Compression on gaugefield: ", args.compression)
print("Number of gaugefields on the gpu", args.n_gaugefields)

glob_lat = np.asarray(args.glob_lat)
splitting = np.asarray(args.split)
halodepth = args.halodepth
BytesPerFloat = args.BytesPerFloat
compression = args.compression

bulk_lat = glob_lat/splitting
full_lat = np.copy(bulk_lat)

for i in range(0, 4):
    if not args.forceHalos:
        if splitting[i] > 1:
            full_lat[i] += 2*halodepth
    else:
        full_lat[i] += 2*halodepth

print("Sub-Lattice bulk lattice: ", bulk_lat)
print("Sub-Lattice full lattice: ", full_lat)

print("---------------------------------------------")

print("\nINFO: The \"full lattice\" has the dimensions of the global lattice \n"
      "divided by the split and + 2*halodepth in each \n"
      "direction where split is larger 1")
print("\nINFO: The continuous halo buffers are shared between Gaugefields \nand spinors and other classes like "
      "LatticeContainer by default!\n")

print("---------------------------------------------")


n_gpus = splitting[0]*splitting[1]*splitting[2]*splitting[3]

n_sites_full = full_lat[0] * full_lat[1] * full_lat[2] * full_lat[3]
n_sites_bulk = bulk_lat[0] * bulk_lat[1] * bulk_lat[2] * bulk_lat[3]

n_sites_P2P = n_sites_full + 2 * (n_sites_full - n_sites_bulk)
n_sites_noP2P = n_sites_full + (n_sites_full - n_sites_bulk)

gaugefield_factor = 18 * 4 * BytesPerFloat * compression
spinorfield_factor = 6 * BytesPerFloat
unit_factor = 1000**3

Gaugefield_HaloBuffer_Mem_GB = (n_sites_full - n_sites_bulk) * gaugefield_factor / unit_factor
Spinor_HaloBuffer_Mem_GB = (n_sites_full - n_sites_bulk) * spinorfield_factor / unit_factor

Gaugefield_Mem_GB_noP2P = n_sites_noP2P * gaugefield_factor / unit_factor
Spinor_Mem_GB_noP2P = n_sites_noP2P * spinorfield_factor / unit_factor

Gaugefield_Mem_GB_P2P = n_sites_P2P * gaugefield_factor / unit_factor
Spinor_Mem_GB_P2P = n_sites_P2P * spinorfield_factor / unit_factor

print("")
print("No P2P/GPU-aware-MPI: The GPU needs memory for the full sub-lattice (bulk+halo) \n"
      "and ONE extra *continuous* buffer for halos \n"
      "that it needs for communication with the host.")
print("")
print("P2P/GPU-aware-MPI: The GPU needs memory for the full sub-lattice (bulk+halo) \n"
      "and TWO extra *continuous* buffers for halos \n"
      "that it needs for communication with the other GPUs.\n")
print("---------------------Gaugefield----------------------------------------------------------------------")
print(">>> Bulk of sub-lattice:                                     ", n_sites_bulk*gaugefield_factor/unit_factor, "GB")
print(">>> Halos:                                                   ", (n_sites_full-n_sites_bulk)*gaugefield_factor/unit_factor, "GB")
print(">>> Extra halo buffer:                                       ", Gaugefield_HaloBuffer_Mem_GB, "GB (no P2P) |", 2*Gaugefield_HaloBuffer_Mem_GB, "GB (P2P)")
print(">>> Total size (bulk+halos+buffers) first Gaugefield per GPU:", Gaugefield_Mem_GB_noP2P, "GB (no P2P) |", Gaugefield_Mem_GB_P2P, "GB (P2P)")
print(">>> Total size for each additional Gaugefield per GPU:       ", n_sites_full*gaugefield_factor/unit_factor, "GB")
print(">>> Total size first Gaugefield:                             ", Gaugefield_Mem_GB_noP2P*n_gpus, "GB (no P2P) |", Gaugefield_Mem_GB_P2P*n_gpus, "GB (P2P)")
print(">>> Total size for each additional Gaugefield                ", n_sites_full*gaugefield_factor/unit_factor*n_gpus, "GB")
print(">>> Total size for", args.n_gaugefields, "Gaugefields                             ",
      Gaugefield_Mem_GB_noP2P*n_gpus + (args.n_gaugefields-1)*n_sites_full*gaugefield_factor/unit_factor*n_gpus, "GB (no P2P) |",
      Gaugefield_Mem_GB_P2P*n_gpus + (args.n_gaugefields-1)*n_sites_full*gaugefield_factor/unit_factor*n_gpus, "GB (P2P)")
print("    >>> those need atleast", math.ceil(( Gaugefield_Mem_GB_noP2P*n_gpus + (args.n_gaugefields-1)*n_sites_full*gaugefield_factor/unit_factor*n_gpus ) / (32510*1024*1024/1000**3)), "(no P2P)",
      math.ceil(( Gaugefield_Mem_GB_P2P*n_gpus + (args.n_gaugefields-1)*n_sites_full*gaugefield_factor/unit_factor*n_gpus ) / (32510*1024*1024/1000**3)), "(P2P) GPUs (NVIDIA V100)")
print("")
print("---------------------Spinorfield---------------------------------------------------------------------")
print(">>> Bulk of sub-spinor:                                      ", n_sites_bulk*spinorfield_factor/unit_factor, "GB")
print(">>> Halos:                                                   ", (n_sites_full-n_sites_bulk)*spinorfield_factor/unit_factor, "GB")
print(">>> Extra halo buffer:                                       ", Spinor_HaloBuffer_Mem_GB, "GB (no P2P) |", 2*Spinor_HaloBuffer_Mem_GB, "GB (P2P)")
print(">>> Total size (bulk+halos+buffers) first Spinor per GPU:    ", Spinor_Mem_GB_noP2P, "GB (no P2P) |", Spinor_Mem_GB_P2P, "GB (P2P)")
print(">>> Total size for each additional Spinorfield per GPU:      ", n_sites_full*spinorfield_factor/unit_factor, "GB")
print(">>> Total size first Spinorfield:                            ", Spinor_Mem_GB_noP2P*n_gpus, "GB (no P2P) |", Spinor_Mem_GB_P2P*n_gpus, "GB (P2P)")
print(">>> Total size for each additional Spinorfield:              ", n_sites_full*spinorfield_factor/unit_factor*n_gpus, "GB")
print("")
print("\nINFO: NVIDIA V100 memory size: ", 32510*1024*1024/1000**3, "GB\n")
