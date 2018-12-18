#!/home/rbussell/anaconda3/bin/python

# create a dashboard from compliance fitB results
# Robert Bussell, rbussell@ucsd.edu

print('VC_dashboard.py called')
#Set up the imports
#---------------
import os,sys
vcModulePath='/home/rbussell/lib/'
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
subDir=sys.argv[1]
subDir='/home/rbussell/data/vcprocRemote/119_180612_vccloud/'
#---------go to data dir-------
print('subDir is ' + subDir);

import numpy as np

try:
    os.chdir(subDir)
except OSError:
    print('Cannot change working directory to '+subDir+' EXITING!');
    exit(0)
#-------------------

id_dir='./'

nSlices=14
nTIs=7
tiArr=np.arange(250,950,100)
idxTagStart=0
idxCtrStart=1

nX=64;
nY=64;
nReps=78;

M0=35;
alpha=1;
fsWide=(40,10)

## Load in a good data set
#Choose a good voxel
x=31;y=37;z=3; #original choice
#x=29;y=37;z=3;

#x=32;y=36;z=3;
#pick this patch for fitting
x1=23;x2=43;y1=30;y2=50;


picoreMat=np.zeros((nX,nY,nSlices,nReps,nTIs))
picoreMat=VC_loadPicoreData(subDir, id_dir,verbosity=0)
phiCSMat=VC_loadPhiCS(subDir,id_dir,verbosity=0)

deltaMat=picoreMat[:,:,:,idxCtrStart::2,:]-picoreMat[:,:,:,idxTagStart::2,:]
mTag=picoreMat[:,:,:,idxTagStart::2,:]
mCtr=picoreMat[:,:,:,idxCtrStart::2,:]
mCtrAve=np.mean(mCtr,3)
mTagMCtrAve=np.tile(np.reshape(mCtrAve,(nX,nY,nSlices,1,nTIs)),(1,1,1,int(nReps/2),1))-mTag
print('Input picore matrix shapes: \nmCtr: '+ str(np.shape(mCtr))+'\nmTag: '+ str(np.shape(mTag))+'\nmTagMCtrAve: '+ str((np.shape(mTagMCtrAve))))


mTagMCtrAveVec=np.reshape(mTagMCtrAve,(nX,nY,nSlices,int(nTIs*nReps/2)))[x,y,z,:]
mTagVec=np.reshape(mTag,(nX,nY,nSlices,int(nTIs*nReps/2)))
mCtrVec=np.reshape(mCtr,(nX,nY,nSlices,int(nTIs*nReps/2)))
deltaMSeqVec=mCtrVec[x,y,z,:]-mTagVec[x,y,z,:]

cbfMap=np.mean(np.mean(deltaMat,3),3)


plotOnePlane(cbfMap[:,:,z]/np.max(cbfMap[:,:,z]),cmap='viridis');plt.title('$<\Delta M_{seq}>$, mean over TI and reps',fontsize=15)
plotOnePlane(cbfMap[x1:x2,y1:y2,z]/np.max(cbfMap[x1:x2,y1:y2,z]),cmap='viridis');plt.title('$<\Delta M>$, mean over TI and reps',fontsize=15);
plt.title('zoomed in patch at [x1:x2,y1:y2,z]',fontsize=15)
picorePatch=picoreMat[x1:x2,y1:y2,z,:,:];
print('selecting a patch of picore data of size '+str(np.shape(picorePatch)))
