#!/bin/sh

dir=pwd
for j in $(seq 0.001 0.001 0.010); do  sed "s/XXXXXX/$j/g" bootstrap.xml > bootstraps/bootstrap$j.xml && ./JemEusoOffline -l logs/$j.log -b bootstraps/bootstrap$j.xml >results/$j.txt 2>&1 & done;
