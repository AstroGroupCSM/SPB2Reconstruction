#!/bin/sh

for ((i=4;i<8;i++)); do sed "s/XXXXXX/$i/g" bootstrap.xml > bootstrap5_$i.xml &&./JemEusoOffline -b bootstrap5_$i.xml 2>&1|grep TRIGGERS_test >5_$i.txt & done;


for ((i=4;i<8;i++)); do sed "s/XXXXXX/$i/g" bootstrap.xml > bootstrap3_$i.xml &&./JemEusoOffline -b bootstrap3_$i.xml 2>&1|grep TRIGGERS_test >3_$i.txt & done;
