#!/bin/sh

for ((i=0;i<200;i+=25)); do sed "s/XXXXXX/$i/g" bootstrap.xml > bootstrap$i.xml &&./JemEusoOffline -b bootstrap$i.xml 2>&1|grep QWERTY >$i.txt & done;
