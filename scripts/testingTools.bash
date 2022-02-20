#!/bin/bash

# 
# testingTools.bash 
# 
# D. Clarke
# 
# A collection of useful functions for running BASH scripts.  
# 

cred="\e[91m"
cyan="\e[36m"
endc="\e[0m"


# ----------------------------------------------------------------------------------------------------- DEFINE FUNCTIONS


# use srun inside a slurm script, and mpiexec everywhere else
run_command="mpiexec --oversubscribe -np";


function runTestRoutine {
    routine="$1"
    nGPUs="$2"
    nodeLayout="$3"
    paramFile="${routine:1}".param
    outFile='OUT'"${routine}"
    errFile='runERR'"${routine}"
    echo '  '${routine}' using '${nGPUs}' GPUs and layout '${nodeLayout}
    $run_command ${nGPUs} ./${routine} "../parameter/tests/${paramFile}" "${nodeLayout}" >> ${outFile} 2>> ${errFile}
    echo " " >> ${outFile}
    # This is just to remove these files if they are empty.
    if [ ! -s ${errFile} ]; then rm ${errFile}; fi
    if [ ! -s ${outFile} ]; then rm ${outFile}; fi
}


function runTestRoutineNoParam {
    routine="$1"
    nGPUs="$2"
    outFile='OUT'"${routine}"
    errFile='runERR'"${routine}"
    echo '  '${routine}' using '${nGPUs}' GPUs.'
    $run_command ${nGPUs} ./${routine} >> ${outFile} 2>> ${errFile}
    echo " " >> ${outFile}
    # This is just to remove these files if they are empty.
    if [ ! -s ${errFile} ]; then rm ${errFile}; fi
    if [ ! -s ${outFile} ]; then rm ${outFile}; fi
}


# The GPUkey tells you the number of processors and sometimes extra info about the node layout:
#    s: Split only in spatial directions. Useful for observables like the Polyakov loop, where one prefers not to
#       split the lattice in the Euclidean time direction.
#    k: No long directions. Useful whenever 4 does not divide a lattice extension.
function runByLayout {
    routine="$1"
    GPUkey="$2"
    if [ $GPUkey == "4" ]; then
        declare -a layouts=("Nodes = 2 2 1 1" "Nodes = 2 1 2 1" "Nodes = 2 1 1 2" \
                            "Nodes = 1 2 2 1" "Nodes = 1 2 1 2" "Nodes = 1 1 2 2" \
                            "Nodes = 4 1 1 1" "Nodes = 1 4 1 1" "Nodes = 1 1 4 1" "Nodes = 1 1 1 4")
        nGPUs=4
    elif [ $GPUkey == "4s" ]; then
        declare -a layouts=("Nodes = 2 2 1 1" "Nodes = 2 1 2 1" "Nodes = 1 2 2 1" \
                            "Nodes = 4 1 1 1" "Nodes = 1 4 1 1" "Nodes = 1 1 4 1")
        nGPUs=4
    elif [ $GPUkey == "4k" ]; then
        declare -a layouts=("Nodes = 2 2 1 1" "Nodes = 2 1 2 1" "Nodes = 2 1 1 2" \
                            "Nodes = 1 2 2 1" "Nodes = 1 2 1 2" "Nodes = 1 1 2 2")
        nGPUs=4
    elif [ $GPUkey == "4ks" ]; then
        declare -a layouts=("Nodes = 2 2 1 1" "Nodes = 2 1 2 1" "Nodes = 1 2 2 1")
        nGPUs=4
    elif [ $GPUkey == "2" ]; then
        declare -a layouts=("Nodes = 2 1 1 1" "Nodes = 1 2 1 1" "Nodes = 1 1 2 1" "Nodes = 1 1 1 2")
        nGPUs=2
    elif [ $GPUkey == "2s" ]; then
        declare -a layouts=("Nodes = 2 1 1 1" "Nodes = 1 2 1 1" "Nodes = 1 1 2 1")
        nGPUs=2
    elif [ $GPUkey == "1" ]; then
        declare -a layouts=("Nodes = 1 1 1 1")
        nGPUs=1
    else
        echo -e "  ${cred}ERROR--Invalid GPUkey encountered by runByLayout!${endc}"
    fi
    for layout in "${layouts[@]}"; do
        runTestRoutine "$routine" "$nGPUs" "$layout"
    done
}