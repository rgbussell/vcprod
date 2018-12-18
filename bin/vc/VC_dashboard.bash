#!/bin/bash

# create a dashboard from compliance fitB results
# Robert Bussell, rbussell@ucsd.edu

sm="$0:"
logFn="VC_dashboard.log"
echo $sm called. SHELL=$SHELL. Running in hostname `hostname`

PATH=$PATH:/home/rbussell/anaconda3/bin/
source /home/rbussell/anaconda3/bin/activate

#----Temporary code for testing----
dataDir=~/data/testdashboard/m0
#----------------------------------


echo $sm called at `date`| tee -a >> $logFn
echo dataDir is $dataDir | tee -a>> $logFn
echo "which python" returns `which python` | tee -a>> $logFn
echo `python --version` | tee -a>> $logFn

~/bin/vc/VC_dashboard.py $dataDir
