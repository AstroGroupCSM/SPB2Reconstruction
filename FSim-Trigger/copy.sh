#!/bin/sh

for j in $(seq 17.8 0.1 19.7); do
  cd $j
  for filename in $(ls *.npy); do
    cp $filename /home/gfil/PythonData/v2/signal/$j$filename
  done
  cd ../
done
