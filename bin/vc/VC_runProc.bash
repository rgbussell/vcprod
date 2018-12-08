#!/bin/bash
sm="$0:"
echo $sm called. SHELL=$SHELL. Running in hostname `hostname`

PATH=$PATH:/home/rbussell/anaconda3/bin/
source /home/rbussell/anaconda3/bin/activate

echo $1 $2 $3 $4 $5 $6
dataDir=$1
id_dir=$2
subDir=$3
M0=$4
pp=$5
version=$6

#source `which activate`
echo after virtenv active python is
which python
python --version


echo run the processing python code
~/bin/vc/VC_runProc.py $dataDir $id_dir $subDir $M0 $pp $version
