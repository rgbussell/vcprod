#test the code by reproducing a fit

#Set up the imports
import os
#---------------
import sys
vcModulePath='/opt/ds/lib/'
sys.path.append(vcModulePath)

import vc
from vc.load import *
from vc.parallel import *
from vc.fileutils import *
from vc.fit import *

#------
#path stuff
#-------
dataDir='/home/ds/notebooks/'

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
idxTagStart=0
idxCtrStart=1

nX=64;
nY=64;
nReps=78;


id_dir='002_180308'
subDir='data/180308_setup3tw_cl'
M0=15;
alpha=1;
fsWide=(40,10)


#Read in a good data set
#Choose a good voxel
x=31;y=37;z=3; #original choice
#x=32;y=36;z=3;
#pick this patch for fitting
x1=23;x2=43;y1=30;y2=50;

print('loading the picore mat and the phicsmat')
picoreMat=np.zeros((nX,nY,nSlices,nReps,nTIs))
picoreMat=VC_loadPicoreData(subDir, id_dir,verbosity=0)
phiCSMat=VC_loadPhiCS(subDir,id_dir,verbosity=0)

#Get the phiCS for this slice
print('phiCSMat shape: ', np.shape(phiCSMat))
phiCS=phiCSMat[(z-1)::nSlices,:] #select a slice (incl. t and c, and TI)
phiCSTag=phiCS[idxTagStart::2,:] #now only take tags for this slice

#vectorize the inputs
phiCSVec=np.reshape(phiCSTag,(273,))
tiVec=np.reshape(np.tile(tiArr,(39,1)),(273,))

#----------------------
print("creating the tag and control vectors")
#deltaMat=picoreMat[:,:,:,idxCtrStart::2,:]-picoreMat[:,:,:,idxTagStart::2,:]
mTag=picoreMat[:,:,:,idxTagStart::2,:]
mCtr=picoreMat[:,:,:,idxCtrStart::2,:]
mCtrAve=np.mean(mCtr,3)
mTagMCtrAve=np.tile(np.reshape(mCtrAve,(nX,nY,nSlices,1,nTIs)),(1,1,1,int(nReps/2),1))-mTag
print('Input picore matrix shapes: \nmCtr: '+ str(np.shape(mCtr))+'\nmTag: '+ str(np.shape(mTag))+'\nmTagMCtrAve: '+ str((np.shape(mTagMCtrAve))))
verbosity=1
mTagMCtrAveVec=np.reshape(mTagMCtrAve,(nX,nY,nSlices,int(nTIs*nReps/2)))[x,y,z,:]
mTagVec=np.reshape(mTag,(nX,nY,nSlices,int(nTIs*nReps/2)))
mCtrVec=np.reshape(mCtr,(nX,nY,nSlices,int(nTIs*nReps/2)))
#deltaMSeqVec=mCtrVec[x,y,z,:]-mTagVec[x,y,z,:]
dataVec=mTagMCtrAveVec


#---------------------
#Create a mask for the data
#--------------------
id_dir='119_180612'
subDir='data/PupAlz_119'
M0=39;
pp=50;
alpha=1;
brainMaskFn=subDir+'/'+id_dir+'/reg/vti_mask.nii.gz'

if FIT_MASK==1:
    fitMask=makeFitMaskFile(subDir,id_dir,brainMaskFn,tiVec,nBins=8,nTIs=5,minMean=0.2,nSlices=14,nReps=78)
if FIT_MASK==0:
    fitMask=np.load(subDir+'/'+'fitMask.npy')
    fitMask=np.reshape(fitMask,(nX,nY,nSlices))
sliceDataMontage(fitMask[:,:,:,np.newaxis]);

#----------------------
#Run the fitting code on the ISMRM 2019 data
#---------------------
version='containertest'
DO_ISMRM2019_NTIS_COMPARISON=1
iSlice=0
saveFn=''
if DO_ISMRM2019_NTIS_COMPARISON:
    numProcessors=multiprocessing.cpu_count()
    #nWorkers=14*3
    nWorkers=14
    if nWorkers>numProcessors:
        nWorkers=numProcessors-1

    print('Available cores: '+str(numProcessors))
    print('Cores needed:' + str(nWorkers))

    pool = multiprocessing.Pool( nWorkers )

    #initialize the task list for each core
    tasks=[]

    # set parameters that apply to all subjects
    verbosity=0
    alpha=1

    id_dir='119_180612'
    subDir='data/PupAlz_119'
    M0=39;
    pp=50;
    alpha=1;

    dryRun=0

    pool = multiprocessing.Pool( nWorkers )

    #initialize the task list for each core
    tasks=[]
    
    #set up the 7 point fits
    nTIsToFit=7
    saveDir=makeSaveDir(subDir,id_dir,version,nTIsToFit,mMethod=0)
    makeTaskAllSlices2p0test(tasks,id_dir,subDir,nTIsToFit,tiVec,fitMask,saveDir,M0=M0,alpha=alpha,verbosity=0,dryRun=dryRun,nBins=8,mMethod=0)

    #set up the 5 point fits
    nTIsToFit=5
    saveDir=makeSaveDir(subDir,id_dir,version,nTIsToFit,mMethod=0)
    makeTaskAllSlices2p0test(tasks,id_dir,subDir,nTIsToFit,tiVec,fitMask,saveDir,M0=M0,alpha=alpha,verbosity=0,dryRun=dryRun,nBins=8,mMethod=0)

    #set up the 3 point fits
    nTIsToFit=3
    saveDir=makeSaveDir(subDir,id_dir,version,nTIsToFit,mMethod=0)
    makeTaskAllSlices2p0test(tasks,id_dir,subDir,nTIsToFit,tiVec,fitMask,saveDir,M0=M0,alpha=alpha,verbosity=0,dryRun=dryRun,nBins=8,mMethod=0)

    print('tiVec is ' + str(tiVec));
    
    tStart = time.time()
    # Run tasks
    if 1:
        for t in tasks:
            print('<<<<<<<<<will run this task>>>>>>>>>>')
            print(t)
            results=pool.apply_async( fitWithinMaskPar2p0_test, t)
    #print(t)
    results=pool.apply_async( fitWithinMaskPar2p0_test, t)
    pool.close()
    pool.join()

    tEnd = time.time()
    tElapsed=tEnd-tStart
    print('********Parallel fitting jobs required '+str(tElapsed)+'seconds. *********')

    print(tasks)


    nTIsToFit=5
    for iSlice in np.arange(0,14,1):
        saveDir=makeSaveDir(subDir,id_dir,version,nTIsToFit,mMethod=0)
        fitWithinMaskPar2p0_test(iSlice,id_dir,subDir,fitMask,saveDir,nTIsToFit,tiVec,M0=M0,alpha=alpha,verbosity=0,dryRun=dryRun,nBins=8,mMethod=0)
    
    #fitWithinMaskPar2p0_test(0,id_dir,subDir,nTIsToFit,fitMask,saveDir,M0=M0,alpha=alpha,verbosity=0,dryRun=dryRun,nBins=8,mMethod=0)
    #fitWithinMaskPar2p0_test(0,id_dir,subDir,nTIsToFit,fitMask,saveDir,M0=M0,alpha=alpha,verbosity=0,dryRun=dryRun,nBins=8,mMethod=0)

