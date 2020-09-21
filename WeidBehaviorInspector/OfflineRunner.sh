#!/bin/bash
export APERC=/home/gfil/.japerc
source /home/gfil/offline/apeinstall/External/geant4/4.10.02.p03/bin/geant4.sh
eval `/home/gfil/offline/ape/jape sh externals`
export JEMEUSOOFFLINEROOT="/home/gfil/offline/install/"
eval `$JEMEUSOOFFLINEROOT$1/bin/jemeuso-offline-config --env-sh`
export ROOT_OUT=$SCRATCH/ShowerLibrary/
export TEMP_SHOWERDIR=$SCRATCH/TempShowers/
export BOOTSTRAPS=$SCRATCH/bootstraps/
export OUTDIR=$SCRATCH/outputs/
export fOUTDIR=$SCRATCH/fSimOutputs/
export LOGDIR=$SCRATCH/logs/
export TERMOUT=$SCRATCH/terminalOutputs/
export RUNLOC=$SCRATCH/SPB2showerSim/

rm -r $TEMP_SHOWERDIR ; mkdir $TEMP_SHOWERDIR
rm -r $BOOTSTRAPS; mkdir $BOOTSTRAPS
rm -r $OUTDIR ; mkdir $OUTDIR
rm -r $fOUTDIR ; mkdir $fOUTDIR
rm -r $LOGDIR ; mkdir $LOGDIR
rm -r $TERMOUT ; mkdir $TERMOUT


for j in $(seq 17.8 0.1 19.7)
do
  for ((i=0;i<100;i+=1))
  do
    iter=`expr $i % 10`
    cp $ROOT_OUT$j-$iter-*.root $TEMP_SHOWERDIR$j-$i.root &&
    sed "s+ZZZZZZ+$TEMP_SHOWERDIR/$j-$i+g ;s+XXXXXX+$OUTDIR/$j-$i+g ;s+YYYYYY+$fOUTDIR/$j-$i+g" $RUNLOC/bootstrap.xml > $BOOTSTRAPS/bootstrap$j-$i.xml &&
    cd $RUNLOC &&
    $RUNLOC/JemEusoOffline -b  $BOOTSTRAPS/bootstrap$j-$i.xml -l $LOGDIR/$j-$i.log > $TERMOUT/$j-$i.txt 2>&1  &
  done;
done

wait
