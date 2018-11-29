#test the code by reproducing a fit

#Set up the imports
import os

#---------------
import sys
vcModulePath='/opt/ds/lib/'
sys.path.append(vcModulePath)

import vc
from vc.load import *

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

os.chdir(dataDir)

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

