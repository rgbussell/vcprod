#!/bin/sh

sm=$0 :
echo $sm called. Running in hostname `hostname`

echo Testing fitting a data set with vc module
python testFit.py 
