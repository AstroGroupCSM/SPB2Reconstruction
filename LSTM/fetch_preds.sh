#!/bin/bash

MODEL_TIME_ID=$1

cd saved
echo "** Downloading tarball **"
scp "czh5372@dgx1.ist.psu.edu:/dgxhome/czh5372/projects/george_signal/saved/$MODEL_TIME_ID.tar.gz" ./

echo "** Decompressing **"
tar -xzf "$MODEL_TIME_ID.tar.gz"