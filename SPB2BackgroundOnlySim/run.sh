#!/bin/sh

for ((i=4;i<8;i++)); do sed "s/XXXXXX/$i/g" bootstrap5.xml > bootstrap5_$i.xml &&./JemEusoOffline -b bootstrap5_$i.xml 2>&1|grep TRIGGERS >5_$i.txt & done;


for ((i=4;i<8;i++)); do sed "s/XXXXXX/$i/g" bootstrap3.xml > bootstrap3_$i.xml &&./JemEusoOffline -b bootstrap3_$i.xml 2>&1|grep TRIGGERS >3_$i.txt & done;
