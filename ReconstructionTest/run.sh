#!/bin/sh

dir=pwd
for j in $(seq 18.8 0.1 19.7); do  sed "s/XXXXXX/$j/g" bootstrap_PEStart.xml > bootstraps/bootstrap$j.xml && ./JemEusoOffline -b bootstraps/bootstrap$j.xml -l logs/$j.log >results/$j.txt 2>&1 & done;
