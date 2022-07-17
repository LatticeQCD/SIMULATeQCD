#!/bin/bash

# 
# addMakeAutocomplete.bash                                                               
# 
# D. Clarke 
# 
# Short script to add a line to your bashrc that will allow "make" commands
# to autocomplete. 
#

bashrcFile=${HOME}/.bashrc

read -p "This will add a line to the end of your ~/.bashrc. Is that okay? (Y/y to proceed.) "
if ! [[ $REPLY =~ [Yy]$ ]]; then
  exit
fi
 
echo "" >> ${bashrcFile}
echo "# make tab completion" >> ${bashrcFile}
echo "complete -W \"\\\`grep -oE '^[a-zA-Z0-9_-]+:([^=]|$)' Makefile | sed 's/[^a-zA-Z0-9_-]*$//'\\\`\" make" >> ${bashrcFile}

source ${bashrcFile}

echo "Tab complete feature added."
