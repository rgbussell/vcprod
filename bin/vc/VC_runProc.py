#!/home/rbussell/anaconda3/bin/python

MAKE_PERC_THR_MASK=1
MAKE_MASK=0	#fit within a mean thresholded mask
USE_PRECALC_BRAIN_MASK=0	#fit within the brain mask determined previously
DO_FIT=1
DO_QA=1
DO_PARALLEL=1
DO_ONE_TEST_FIT=0

print('VC_runProc.py called')

#-----
#library imports
#-----
import os,sys
import numpy as np

#vcModulePath=os.environ['VC_PATH']+'/lib'
vcModulePath='/apps/vc/lib'
sys.path.append(vcModulePath)
import vc
from vc.load import *
from vc.parallel import *
from vc.fileutils import *
from vc.fit import *
from vc.view import *

#-----
#handle input
#-----

dataDir=sys.argv[1]
M0=float(sys.argv[2])
pp=float(sys.argv[3])
id_dir=sys.argv[4]
uid=sys.argv[5]

#dataDir='/home/rbussell/data/vcprocRemote/119_180612_vccloud'
#subDir='119_180612_vccloud'

#----------------
#set up some descriptors for standard data
#---------------
nSlices=14
nTIs=7
tiArr=np.arange(250,950,100)
tiVec=np.reshape(np.tile(tiArr,(39,1)),(273,))
idxTagStart=0
idxCtrStart=1
nX=64;
nY=64;
nReps=78;

#-----------------
#cvs for the fit
#-----------------
dryRun=0
nFitTypes=2


try:
    os.chdir(dataDir)
except OSError:
    print('Cannot change working directory to '+dataDir+' EXITING!');
    exit(0)

#---------------------
#Create a mask for the data
#--------------------
alpha=1;
#brainMaskFn='brain_mask.nii.gz'
brainMaskFn=dataDir+'/'+id_dir+'/reg/mc_TIall_thr.nii.gz'

try:
    os.chdir(dataDir)
except OSError:
    print('Cannot change working directory to '+dataDir+' EXITING!');
    exit(0)

if MAKE_MASK==1:
    print('calculated threshold mask and applying to brain mask')
    #fitMask=makeFitMaskFile(dataDir,'./',brainMaskFn,nBins=8,nTIs=5,minMean=0.2,nSlices=14,nReps=78)
    fitMask=makeFitMaskFile(dataDir,'./',brainMaskFn,nBins=8,nTIs=5,minMean=0.2,nSlices=14,nReps=78)
if USE_PRECALC_BRAIN_MASK==1:
    print('using precaculated brain mask for fitting mask')
    fitMask=makeFitMaskFile(dataDir,'./',brainMaskFn,nBins=8,nTIs=5,minMean=-1,nSlices=14,nReps=78,mode=1)
    #fitMask=makeFitMaskFile(dataDir,'./',brainMaskFn,nBins=8,nTIs=5,minMean=-1,nSlices=14,nReps=78)
    #img=nib.load(brainMaskFn)
    #fitMask=np.squeeze(img.get_data());
    #fitMask=np.reshape(fitMask,(nX,nY,nSlices))

if MAKE_PERC_THR_MASK==1:
    fitMask=makeFitMaskFile(dataDir,id_dir,brainMaskFn,nBins=8,nTIs=5,minMean=-1,nSlices=14,nReps=78,mode=1,percentile=95)
 
print('fitMask sum is ', str(np.sum(fitMask)))
print('shape fitMask: '+str(np.shape(fitMask)))

#----------------------
#Run the fitting code on the ISMRM 2019 data
#---------------------

iSlice=0
saveFn=''
if 1:
    numProcessors=multiprocessing.cpu_count()
    nWorkers=1
    if DO_PARALLEL:
        nWorkers=nSlices*nFitTypes
    #nWorkers=14
        if nWorkers>numProcessors:
            nWorkers=numProcessors-1
        pool = multiprocessing.Pool( nWorkers )

    print('VC_runProc.py: Available cores: '+str(numProcessors))
    print('VC_runProc.py: Cores needed:' + str(nWorkers))

    #initialize the task list for each core
    tasks=[]
    
    #set up the 7 point fit input params
    nTIsToFit=7
    saveDir=makeSaveDir(dataDir,'.',uid,nTIsToFit,mMethod=0)
    makeTaskAllSlices(tasks,id_dir,dataDir,saveDir,fitMask,tiVec,nX=64,nY=64,nSlices=14,nBins=8,nTIs=7,nReps=39,nTIsToFit=nTIsToFit,M0=M0,alpha=alpha,mMethod=0,dryRun=dryRun,verbosity=1)

    #set up the 5 point fit input params
    nTIsToFit=5
    saveDir=makeSaveDir(dataDir,'.',uid,nTIsToFit,mMethod=0)
    makeTaskAllSlices(tasks,id_dir,dataDir,saveDir,fitMask,tiVec,nX=64,nY=64,nSlices=14,nBins=8,nTIs=7,nReps=39,nTIsToFit=nTIsToFit,M0=M0,alpha=alpha,mMethod=0,dryRun=dryRun,verbosity=1)

    print('VC_runProc.py: tiVec is ' + str(tiVec));
    
    tStart = time.time()
    if DO_PARALLEL:
        for t in tasks:
            printf('')
            #print('<<<<<<<<<will run this task>>>>>>>>>>')
            #print(t)
            results=pool.apply_async( fitWithinMaskPar, t)
    if DO_PARALLEL and DO_FIT:
        pool.close()
        pool.join()

    
    tEnd = time.time()
    tElapsed=tEnd-tStart
    print('********Parallel fitting jobs required '+str(tElapsed)+'seconds. *********')

#print(t)
#print(os.getcwd())
if DO_ONE_TEST_FIT:
    t=tasks[0]
    fitWithinMaskPar(1,t[1],t[2],t[3],t[4],t[5],t[6],t[7],t[8],t[9],t[10],t[11],t[12],t[13],t[14],t[15],t[16],t[17])


if DO_QA:
    print('dataDir is ' + dataDir);
    qaDir=dataDir+'/qa'
    if not os.path.exists(qaDir):
        os.mkdir(qaDir)
    procSubDir='vremoteproc/5TIs/m0/'

    makeQAOutput(id_dir,dataDir,procSubDir,pp,qaDir,M0)
