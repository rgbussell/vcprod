#!/home/rbussell/anaconda3/bin/python

#test the code by reproducing a fit


print('testISMRM.py called')
#Set up the imports
#---------------
import os,sys
vcModulePath='/home/rbussell/lib/'
sys.path.append(vcModulePath)

import importlib

import vc

from vc.load import *
from vc.parallel import *
from vc.fileutils import *
from vc.fit import *

importlib.reload(vc)

#-----
#handle input
#-----

dataDir=sys.argv[1]
id_dir=sys.argv[2]
subDir=sys.argv[3]
M0=sys.argv[3]
pp=sys.argv[4]
version=sys.argv[5]
#------
#path stuff
#-------
#dataDir='/home/rbussell/notebooks/vcproc/'

#----------------------------
#These are needed for running this specific code
#-------------------

import numpy as np

#------------------------------
#main
#-----------------------------

FIT_MASK=0

try:
    os.chdir(dataDir)
except OSError:
    print('Cannot change working directory to '+dataDir+' EXITING!');
    exit(0)

nSlices=14
nTIs=7
tiArr=np.arange(250,950,100)
tiVec=np.reshape(np.tile(tiArr,(39,1)),(273,))
idxTagStart=0
idxCtrStart=1

nX=64;
nY=64;
nReps=78;


#id_dir='002_180308'
#subDir='data/180308_setup3tw_cl'
#M0=15;
#alpha=1;
fsWide=(40,10)

#---------------------
#Create a mask for the data
#--------------------
#id_dir='119_180612'
#subDir='data/PupAlz_119'
#M0=39;
#pp=50;
alpha=1;
brainMaskFn=subDir+'/'+id_dir+'/reg/vti_mask.nii.gz'

if FIT_MASK==1:
    fitMask=makeFitMaskFile(subDir,id_dir,brainMaskFn,tiVec,nBins=8,nTIs=5,minMean=0.2,nSlices=14,nReps=78)
if FIT_MASK==0:
    fitMask=np.load(subDir+'/'+'fitMask.npy')
    fitMask=np.reshape(fitMask,(nX,nY,nSlices))
#sliceDataMontage(fitMask[:,:,:,np.newaxis]);

print('shape fitMask: '+str(np.shape(fitMask)))
# set parameters that apply to all subjects


#----------------------
#Run the fitting code on the ISMRM 2019 data
#---------------------
dryRun=0

#version='vcinstalltest'
DO_ISMRM2019_NTIS_COMPARISON=1
iSlice=0
saveFn=''
if DO_ISMRM2019_NTIS_COMPARISON:
    numProcessors=multiprocessing.cpu_count()
    nWorkers=14*2
    #nWorkers=14
    if nWorkers>numProcessors:
        nWorkers=numProcessors-1

    print('Available cores: '+str(numProcessors))
    print('Cores needed:' + str(nWorkers))

    pool = multiprocessing.Pool( nWorkers )

    #initialize the task list for each core
    tasks=[]

    pool = multiprocessing.Pool( nWorkers )

    #initialize the task list for each core
    tasks=[]
    
    #set up the 7 point fits
    nTIsToFit=7
    saveDir=makeSaveDir(subDir,id_dir,version,nTIsToFit,mMethod=0)
    makeTaskAllSlices(tasks,id_dir,subDir,saveDir,fitMask,tiVec,nX=64,nY=64,nSlices=14,nBins=8,nTIs=7,nReps=39,nTIsToFit=nTIsToFit,M0=M0,alpha=alpha,mMethod=0,dryRun=dryRun,verbosity=1)

    nTIsToFit=5
    saveDir=makeSaveDir(subDir,id_dir,version,nTIsToFit,mMethod=0)
    makeTaskAllSlices(tasks,id_dir,subDir,saveDir,fitMask,tiVec,nX=64,nY=64,nSlices=14,nBins=8,nTIs=7,nReps=39,nTIsToFit=nTIsToFit,M0=M0,alpha=alpha,mMethod=0,dryRun=dryRun,verbosity=1)

    print('tiVec is ' + str(tiVec));
    
    tStart = time.time()
    # Run tasks
    if 1:
        for t in tasks:
            #print('<<<<<<<<<will run this task>>>>>>>>>>')
            #print(t)
            results=pool.apply_async( fitWithinMaskPar, t)
    #print(t)
    #results=pool.apply_async( fitWithinMaskPar2p0_test, t)
    
    #print(pool.map())
    pool.close()
    pool.join()

    
    tEnd = time.time()
    tElapsed=tEnd-tStart
    print('********Parallel fitting jobs required '+str(tElapsed)+'seconds. *********')
