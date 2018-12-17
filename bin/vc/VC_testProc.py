#!/home/rbussell/anaconda3/bin/python

#test the code by reproducing a fit
#example: /home/rbussell/bin/vc/VC_runProc.bash ~/data/vcprocRemote//119_180612_vccloud 39.2686 40 remoteproc

print('VC_testProc.py called')
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


dataDir='/home/rbussell/data/vcprocRemote//119_180612_vccloud'
M0=39;pp=50;uid='remoteproc'
dataDir=sys.argv[1]
M0=float(sys.argv[2])
pp=float(sys.argv[3])
uid=sys.argv[4]
print('dataDir is ' + dataDir);

alpha=1
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


#---------------------
#Check if files exist
#--------------------

from pathlib import Path
def checkForRequiredFile(fn,verbosity=1):

    if verbosity>0:
        print('checking for required file ', str(fn) )
    fpath = Path(fn)
    try:
        absPath = fpath.resolve()
        print('Required file found. absolute path is '+str(absPath))
    except FileNotFoundError:
        print('ERROR!')
        print('Required file not found: ',str(fn))
        print('EXITING!')
        exit(0)


#a---------------------
#check for some input files
#---------------------
tiTmp=np.arange(250,950,100)
for ti in tiTmp:
    physFn='PhysMon/PhysPars_TI'+str(ti)+'_pOxPhaseAtSlice.dat'
    checkForRequiredFile(physFn)

#---------------------
#Create a mask for the data
#--------------------
brainMaskFn='brain_mask.nii.gz'

checkForRequiredFile(brainMaskFn)

fitMaskFn=dataDir+'/fitMask.npy'

if FIT_MASK==1:
    fitMask=makeFitMaskFile(dataDir,'./',brainMaskFn,tiVec,nBins=8,nTIs=5,minMean=0.2,nSlices=14,nReps=78)
if FIT_MASK==0:
    fitMask=np.load(dataDir+'/fitMask.npy')
    fitMask=np.reshape(fitMask,(nX,nY,nSlices))
    print('fitMask sum is ', str(np.sum(fitMask)))

checkForRequiredFile(fitMaskFn)

print('shape fitMask: '+str(np.shape(fitMask)))
# set parameters that apply to all subjects

#----------------------
#Run the fitting code on the ISMRM 2019 data
#---------------------
dryRun=0

DO_ONE_SLICE_ONLY=0
DO_ISMRM2019_NTIS_COMPARISON=1

if DO_ONE_SLICE_ONLY==1:
    tasks=[]
    nTIsToFit=7
    saveDir=makeSaveDir(dataDir,'.',uid,nTIsToFit,mMethod=0)
    makeTaskAllSlices(tasks,'.',dataDir,saveDir,fitMask,tiVec,nX=64,nY=64,nSlices=14,nBins=8,nTIs=7,nReps=39,nTIsToFit=nTIsToFit,M0=M0,alpha=alpha,mMethod=0,dryRun=dryRun,verbosity=1)

    t=tasks[3]
    print(t)
#print(os.getcwd())
    fitWithinMaskPar(5,t[1],t[2],t[3],t[4],t[5],t[6],t[7],t[8],t[9],t[10],t[11],t[12],t[13],t[14],t[15],t[16],t[17])


iSlice=0
if DO_ISMRM2019_NTIS_COMPARISON==1:
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
    saveDir=makeSaveDir(dataDir,'.',uid,nTIsToFit,mMethod=0)
    makeTaskAllSlices(tasks,'.',dataDir,saveDir,fitMask,tiVec,nX=64,nY=64,nSlices=14,nBins=8,nTIs=7,nReps=39,nTIsToFit=nTIsToFit,M0=M0,alpha=alpha,mMethod=0,dryRun=dryRun,verbosity=1)

    nTIsToFit=5
    saveDir=makeSaveDir(dataDir,'.',uid,nTIsToFit,mMethod=0)
    makeTaskAllSlices(tasks,'.',dataDir,saveDir,fitMask,tiVec,nX=64,nY=64,nSlices=14,nBins=8,nTIs=7,nReps=39,nTIsToFit=nTIsToFit,M0=M0,alpha=alpha,mMethod=0,dryRun=dryRun,verbosity=1)

    print('tiVec is ' + str(tiVec));
    
    tStart = time.time()
    if 1:
        for t in tasks:
            #print('<<<<<<<<<will run this task>>>>>>>>>>')
            #print(t)
            results=pool.apply_async( fitWithinMaskPar, t)
    
    pool.close()
    pool.join()

    tEnd = time.time()
    tElapsed=tEnd-tStart
    print('********Parallel fitting jobs required '+str(tElapsed)+'seconds. *********')

#print(t)
#print(os.getcwd())
#fitWithinMaskPar(5,t[1],t[2],t[3],t[4],t[5],t[6],t[7],t[8],t[9],t[10],t[11],t[12],t[13],t[14],t[15],t[16],t[17])
