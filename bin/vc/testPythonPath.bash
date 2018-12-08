#!/bin/bash

PATH=$PATH:/home/rbussell/anaconda3/bin/
source `which activate`

echo after virtenv active python is
which python
python --version

python ./testPythonPath.py
