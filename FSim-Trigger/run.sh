#!/bin/sh

for j in $(seq 18.5 0.1 19.0); do  sed "s/XXXXXX/$j/g" bootstrap.xml > bootstraps/bootstrap$j.xml && ./JemEusoOffline -b bootstraps/bootstrap$j.xml >>.junk.txt 2>&1 & done;
