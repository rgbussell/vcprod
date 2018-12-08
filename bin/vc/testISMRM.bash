#!/bin/bash
sm=$0 :
echo $sm called. SHELL=$SHELL. Running in hostname `hostname`

PATH=$PATH:/home/rbussell/anaconda3/bin/
source `which activate`

echo after virtenv active python is
which python
python --version

which python

echo Testing fitting a data set with vc module
python testISMRM.py 
