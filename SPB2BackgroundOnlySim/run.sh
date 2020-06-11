#!/bin/sh

for ((i=0;i<8;i++)); do ./JemEusoOffline  2>&1 |grep TRIGGERS> $i.txt & done;
