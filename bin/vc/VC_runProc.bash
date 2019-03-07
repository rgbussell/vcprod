#!/bin/bash
sm="$0:"

#Deal with input
procDir=$1
M0=$2
pp=$3
id_dir=$4
uid=$5

if [ ! -d $procDir ]; then
  $sm ERROR 
  echo procdir $procDir not found
  echo EXITING
  exit
fi

VC_runProcCmd=/apps/vc/bin/VC_runProc.py
logFn=$procDir"/VC_runProc.log"

PATH=$PATH:$VC_PATH
source $VC_ACTIVATE | tee -a $logFn

echo $sm called. SHELL=$SHELL. Running in hostname `hostname`


echo $sm called at `date` | tee -a $logFn
echo $sm has these input arguments: $1 $2 $3 $4 $5 | tee -a $logFn

echo procDir $procDir | tee -a $logFn
echo M0 $M0 | tee -a $logFn
echo pp is $pp | tee -a $logFn
echo id_dir is $id_dir | tee -a $logFn
echo uid is $uid | tee -a $logFn

#source `which activate`
echo after virtenv active python is
which python
python --version

echo run the processing python code
$VC_runProcCmd $procDir $M0 $pp $id_dir $uid
