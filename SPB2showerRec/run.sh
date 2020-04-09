#!/bin/sh

for ((i=50;i<60;i+=25)); do sed "s/XXXXXX/$i/g" bootstrap.xml > bootstrap$i.xml &&./JemEusoOffline -b bootstrap$i.xml 2>&1|grep TriggerSPB2CSM >$i.txt ; done;
