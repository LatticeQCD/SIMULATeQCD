#!/bin/bash
mkdir in.files out.files

for Nfl in 2
do
for Nfs in 1
do
for ml in 0.06
do
for ms in 0.08
do
for Npf in 1 4
do (
echo "$Npf $Nfl $Nfs $ml $ms"
./rataprox.sh $ml $ms $Nfl $Nfs $Npf )&
done
done
done
done
done
wait
