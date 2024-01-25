#!/bin/python3
# 
# checkParameterDefinitions.py
# 
# D. Clarke 
# 
# Check that each parameter is a parameter file is defined. Run this in
# a folder that has .param files in it. 
# 

import glob

for fileName in glob.iglob('*.param'):

    file = open(fileName,'r')

    params = []
    for line in file:
        if line.startswith('#'):
            continue
        if not '=' in line:
            continue
        col=line.split('=')
        if not '#' in line: # I assume that the # indicates there's an explanation
            params.append(col[0].strip())
    file.close()

    defs = []
    file = open(fileName,'r')
    for line in file:
        if not line.startswith('#'):
            continue
        if not ':' in line:
            continue
        col=line.split(':')
        defs.append(col[0][1:].strip())
    file.close()

    diff = set(set(params) - set(defs))
    if len(diff)>0:
        print(fileName,'-- UNDEFINED PARAMS:',diff)
    diff = set(set(defs) - set(params))
    if len(diff)>0:
        print(fileName,'-- DEFINITION BUT NO PARAM:',diff)

