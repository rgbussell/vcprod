#!/bin/bash
sm="$0:"

#Deal with input
procDir=$1
M0=$2
pp=$3
id_dir=$4
uid=$5

logFn=$procDir"/VC_runProc.log"

PATH=$PATH:/home/rbussell/anaconda3/bin/
source /home/rbussell/anaconda3/bin/activate | tee -a $logFn


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
~/bin/vc/VC_runProc.py $procDir $M0 $pp $id_dir $uid
