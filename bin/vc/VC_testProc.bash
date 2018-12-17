#!/bin/bash
logFn="VC_testProc.log"
sm="$0:"
echo $sm called. SHELL=$SHELL. Running in hostname `hostname`

PATH=$PATH:/home/rbussell/anaconda3/bin/
source /home/rbussell/anaconda3/bin/activate

echo $sm called at `date` >> $logFn
echo $sm has these input arguments: $1 $2 $3 $4 $5 $6 >> $logFn
procDir=$1
M0=$2
pp=$3
uid=$4

echo procDir $procDir >> $logFn
echo M0 $M0 >> $logFn
echo pp is $pp >> $logFn
echo uid is $uid >> $logFn

#source `which activate`
echo after virtenv active python is
which python
python --version

echo run the processing python code
~/bin/vc/VC_testProc.py $procDir $M0 $pp $uid
