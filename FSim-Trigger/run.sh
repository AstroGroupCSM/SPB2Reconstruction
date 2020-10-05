#!/bin/sh

dir=pwd
for j in $(seq 17.8 0.1 19.7); do  sed "s/XXXXXX/$j/g" bootstrap.xml > bootstrap$j.xml && ./JemEusoOffline -b ../bootstrap$j.xml 2>&1|grep TRIGGERS >$j.txt ; done;
