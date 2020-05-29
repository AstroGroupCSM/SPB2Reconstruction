#!/bin/sh

for ((i=0;i<8;i++)); do ./JemEusoOffline  2>&1 |grep QWERTY> $i.txt & done;
