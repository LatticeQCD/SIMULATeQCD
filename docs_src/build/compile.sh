#!/bin/bash


./clean.sh

make html


mv temp_build/html/* ../../docs/ -f
touch ../../docs/.nojekyll
