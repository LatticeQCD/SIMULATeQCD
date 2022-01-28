# 
# compute_memorySize.py                                                               
# 
# Short python script to figure out how much memory your parallelGPUCode executable will require.
# 
# 

def STR(number):
    return '{:6.2f}'.format(number)


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
argparser.add_argument('--GPU_memory_in_GB', type=float, default=34.08920576) # NVIDIA V100 32GiB

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

print("\nINFO: The \"full lattice\" has the dimensions of the global lattice divided by the split and + 2*halodepth in each direction where split is larger 1")
print("INFO: The continuous halo buffers are shared between Gaugefields and spinors and other classes like LatticeContainer by default!\n")

print("---------------------------------------------")


multiplicity = splitting[0]*splitting[1]*splitting[2]*splitting[3] # usually equal to number of mpi ranks / gpus

n_sites_full = full_lat[0] * full_lat[1] * full_lat[2] * full_lat[3]
n_sites_bulk = bulk_lat[0] * bulk_lat[1] * bulk_lat[2] * bulk_lat[3]

n_sites_P2P = n_sites_full + 2 * (n_sites_full - n_sites_bulk)
n_sites_noP2P = n_sites_full + (n_sites_full - n_sites_bulk)

gaugefield_factor = 18 * 4 * BytesPerFloat * compression
spinorfield_factor = 6 * BytesPerFloat
unit_factor = 1000**3

gauge_halobuffer = (n_sites_full - n_sites_bulk) * gaugefield_factor / unit_factor
spinor_halobuffer = (n_sites_full - n_sites_bulk) * spinorfield_factor / unit_factor

gauge_noP2P_perGPU = n_sites_noP2P * gaugefield_factor / unit_factor
spinor_noP2P_perGPU = n_sites_noP2P * spinorfield_factor / unit_factor

gauge_P2P_perGPU = n_sites_P2P * gaugefield_factor / unit_factor
spinor_P2P_perGPU = n_sites_P2P * spinorfield_factor / unit_factor

gauge_bulk_sublat = n_sites_bulk*gaugefield_factor/unit_factor
gauge_halos = (n_sites_full-n_sites_bulk)*gaugefield_factor/unit_factor

gauge_add_total_perGPU = n_sites_full*gaugefield_factor/unit_factor

gauge_noP2P_first_total_perGPU = gauge_noP2P_perGPU*multiplicity
gauge_P2P_add_total_perGPU = gauge_P2P_perGPU*multiplicity

print("")
print("No P2P/CUDA-aware-MPI: GPU needs memory for full sub-lattice (bulk+halo) + ONE extra *continuous* halobuffer (for host comm.)")
print("   P2P/CUDA-aware-MPI: GPU needs memory for full sub-lattice (bulk+halo) + TWO extra *continuous* halobuffer (for host comm.)")
print("")
print("---------------------Gaugefield----------------------------------------------------------------------\n")
print("---------Memory PER GPU in GB (1000^3)--------------------------------")
print(">>> Bulk of sub-lattice:                             ", STR(gauge_bulk_sublat))
print(">>> Halos                                            ", STR(gauge_halos))
print(">>> Extra halo buffer                                ", STR(gauge_halobuffer), "(no P2P) |", STR(2*gauge_halobuffer), "(P2P)")
print(">>> size (bulk+halos+buffers) first gaugefield:      ", STR(gauge_noP2P_perGPU), "(no P2P) |", STR(gauge_P2P_perGPU), "(P2P)")
print(">>> size for each add. gaugefield (bulk+halos):      ", STR(gauge_add_total_perGPU))
print("")
print("---------Memory TOTAL in GB (1000^3)----------------------------------")
print(">>> size first Gaugefield:                           ", STR(gauge_noP2P_first_total_perGPU), "(no P2P) |", STR(gauge_P2P_add_total_perGPU), "(P2P)")
print(">>> size for each additional Gaugefield              ", STR(gauge_noP2P_first_total_perGPU))
print(">>> Total size for", args.n_gaugefields, "Gaugefields                     ",
      STR(gauge_noP2P_first_total_perGPU + (args.n_gaugefields-1)*gauge_noP2P_first_total_perGPU), "(no P2P) |",
      STR(gauge_P2P_add_total_perGPU + (args.n_gaugefields-1)*gauge_noP2P_first_total_perGPU), "(P2P)")
print("")
print("Is the memory per GPU sufficient for", args.n_gaugefields, "gaugefields given the split", args.split, " using", multiplicity,"gpus?")
result_noP2P = "" if gauge_noP2P_perGPU+(args.n_gaugefields-1)*gauge_add_total_perGPU < args.GPU_memory_in_GB else "NOT "
result_P2P = "" if gauge_P2P_perGPU+(args.n_gaugefields-1)*gauge_add_total_perGPU < args.GPU_memory_in_GB else "NOT"
print("no P2P: total size per GPU =", STR(gauge_noP2P_perGPU+(args.n_gaugefields-1)*gauge_add_total_perGPU), ">>> DOES", result_noP2P, "fit on one GPU (", STR(args.GPU_memory_in_GB), ")")
print("   P2P: total size per GPU =", STR(gauge_P2P_perGPU+(args.n_gaugefields-1)*gauge_add_total_perGPU), ">>> DOES", result_P2P, "fit on one GPU (", STR(args.GPU_memory_in_GB), ")")

print("")
print("---------------------Spinorfield---------------------------------------------------------------------")
print(">>> Bulk of sub-spinor:                                      ", n_sites_bulk*spinorfield_factor/unit_factor, "GB")
print(">>> Halos:                                                   ", (n_sites_full-n_sites_bulk)*spinorfield_factor/unit_factor, "GB")
print(">>> Extra halo buffer:                                       ", spinor_halobuffer, "GB (no P2P) |", 2*spinor_halobuffer, "GB (P2P)")
print(">>> Total size (bulk+halos+buffers) first Spinor per GPU:    ", spinor_noP2P_perGPU, "GB (no P2P) |", spinor_P2P_perGPU, "GB (P2P)")
print(">>> Total size for each additional Spinorfield per GPU:      ", n_sites_full*spinorfield_factor/unit_factor, "GB")
print(">>> Total size first Spinorfield:                            ", spinor_noP2P_perGPU*multiplicity, "GB (no P2P) |", spinor_P2P_perGPU*multiplicity, "GB (P2P)")
print(">>> Total size for each additional Spinorfield:              ", n_sites_full*spinorfield_factor/unit_factor*multiplicity, "GB")
print("")
print("\nINFO: NVIDIA V100 memory size: ", 32510*1024*1024/1000**3, "GB or ", 16130*1024*1024/1000**3, "GB\n")
