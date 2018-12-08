import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import matplotlib.pyplot as plt
#%matplotlib inline
from scipy import signal
import math
from scipy import optimize
from scipy.optimize import least_squares
import time

import sys
def printf(format, *args):
    sys.stdout.write(format % args)

from .binning import *

#----------
#Data loading and plotting helper functions
#---------

def VC_loadPicoreData(subDir,id_dir,verbosity=1):
    #load all the motion corrected picore data
    #remove the first two reps
    print('VC_loadPicoreData called')
    if 0:
        import os
        print(os.getcwd())
    nDummy=2
    dataDir=subDir+'/'+id_dir+"/reg";
    tiArr=np.arange(250,950,100)
    aslMat=np.zeros([64,64,14,78,7])
    cnt=0;
    for ti in tiArr:
        fn=dataDir+'/mc_TI'+str(ti)+'_raw.nii.gz'
        img=nib.load(fn)
        tmp=np.squeeze(img.get_data());
        aslMat[:,:,:,:,cnt]=tmp[:,:,:,nDummy:78+nDummy]
        if verbosity>0:
            print(fn + ' loaded, shape: ' +str(np.shape(picoreMat)))
        labelStr='TI='+str(ti);
        cnt=cnt+1
    #plt.title('same voxel, registered picore data, diff TIs')
    #plt.legend(loc='best',fancybox=True,framealpha=0.5)
    return aslMat

def VC_loadPhiCS(subDir,id_dir,verbosity=1):
    #Get the phics vector for this data
    #output the phics vectors for every slice acquired at each TI
    # dims: SLICE,REP,TI
    #discard firs two reps
    nDummy=2;nSlices=14;
    tiArr=np.arange(250,950,100)
    subnum=id_dir[0:3];
    physMonDir='PhysMon'
    poxOrderPrefix='PhysPars_'+str(subnum)+'_TI'
    poxOrderPostfix='_pOxPhaseAtSlice.dat'
    phiCSMat=np.zeros((nSlices*78,7))
    cnt=0;
    for ti in tiArr:
        poxOrderFn=subDir+'/'+physMonDir+ '/'+poxOrderPrefix+str(ti)+poxOrderPostfix;
        tmp=np.loadtxt(poxOrderFn);
        phiCSMat[:,cnt]=tmp[nDummy*nSlices:] # I am excluding two dummy reps here
        if verbosity>0:
            print('loaded: '+poxOrderFn+' size: '+str(np.shape(tmp)))
            print(str(np.shape(phiCSMat)))
        cnt=cnt+1;
    return phiCSMat


#Functions for loading data

def hello():
	print('hello from' + __name__ +' in vc load module')

#----
#load for fitting
#-------
def loadPhiCSVecOneSlice(subDir,id_dir,iSlice,idxTagStart=0,idxCtrStart=1,nSlices=14,verbosity=1):
    phiCSMat=VC_loadPhiCS(subDir,id_dir,verbosity=0)
    #Get the phiCS for this slice

    phiCS=phiCSMat[(iSlice-1)::nSlices,:] #select a slice (incl. t and c, and TI)
    phiCSTag=phiCS[idxTagStart::2,:] #now only take tags for this slice
    phiCSCtr=phiCS[idxCtrStart::2,:] #now only take tags for this slice

    #vectorize the inputs
    phiCSVecTagOneSlice=np.reshape(phiCSTag,(273,))
    phiCSVecCtrOneSlice=np.reshape(phiCSCtr,(273,))

    if verbosity>0:
        print('------ loadPhiCSVecOneSlice called -----')
        print('loading data from subDir/id_dir:', subDir+'/'+id_dir)
        print('slice (starting with 1)', iSlice)

        print('phiCSMat shape: ', np.shape(phiCSMat))
        print('assuming tags start at index',idxTagStart)
        print('assuming ctrls start at index',idxCtrStart)
        #plt.plot(phiCSVecTagOneSlice);plt.plot(phiCSVecCtrOneSlice)
        #plt.title('$\phi_c^s$')
    
    return phiCSVecTagOneSlice,phiCSVecCtrOneSlice


def loadDataToFit(picoreMat,x1f,x2f,y1f,y2f,zf,phiCSVecCurSlice,tiVec,nBins=8,nTIs=5,verbosity=5):
    # input:
    #   picoreMat [nX nY nZ ......
    #   zf is the slice (index starts with 1)

    if verbosity>=5:
        print('loadDataToFit has picoreMat shape: ',str(np.shape(picoreMat)) )   
        print('loadDataToFit has x1f,x2f,y1f,y2f: ',str(x1f),str(x2f),str(y1f),str(y2f) )   
        print('loadDataToFit has zf: ',str(zf) ) 
        print('loadDataToFit has phiCSVecCurSlice shape: ',str(np.shape(phiCSVecCurSlice)) )   
        print('loadDataToFit has tiVec shape: ',str(np.shape(tiVec)) )
        print('loadDataToFit has nBins: ',str(nBins) )
        print('loadDataToFit has nTIs: ',str(nTIs) )
 
    nX=x2f-x1f+1
    nY=y2f-y1f+1

    mTagMCtrAvePatch=getMTagMCtrAvePatch(picoreMat,x1f,x2f,y1f,y2f,zf,zf)
    #if verbosity>0:
    #    plt.imshow(np.reshape(mTagMCtrAvePatch,(45,45,7*39))[:,:,1]);plt.title('from getMTagMCtrAvePatch');plt.ion();plt.show()
    mTagMCtrAvePatchBin=binAPatch(mTagMCtrAvePatch,tiVec,phiCSVecCurSlice,nBins=8)

    nVox=np.shape(mTagMCtrAvePatchBin)[0]
    nBins=np.shape(mTagMCtrAvePatchBin)[1]
    nTIsOld=np.shape(mTagMCtrAvePatchBin)[2]

    mTagMCtrAvePatchBin=np.reshape(mTagMCtrAvePatchBin,(nVox,nBins,nTIsOld))
    mTagMCtrAvePatchBin5pt=mTagMCtrAvePatchBin[:,:,0:nTIs:1]

    dataMat=np.reshape(mTagMCtrAvePatchBin5pt,(nX,nY,nBins,nTIs))
    
    #phiCSMat=VC_loadPhiCS(subDir,id_dir,verbosity=0)
    print('loadDataToFit has dataMat shape', np.shape(dataMat));
    print('returning from loadDataToFit')
    return dataMat


def getMCtrAveMTagVol(subDir,id_dir,nX=64,nY=64,nSlices=14,nReps=78,nTIs=7):
    picoreMat=np.zeros((nX,nY,nSlices,nReps,nTIs))
    picoreMat=VC_loadPicoreData(subDir, id_dir,verbosity=0)
    print('picoreMat shape: ',np.shape(picoreMat))
    mTag=picoreMat[:,:,:,idxTagStart::2,:]
    mCtr=picoreMat[:,:,:,idxCtrStart::2,:]
    mCtrAve=np.mean(mCtr,3)
    mTagMCtrAve=np.tile(np.reshape(mCtrAve,(nX,nY,nSlices,1,nTIs)),(1,1,1,int(nReps/2),1))-mTag
    #(phiCSVecOneSlice,junk)=loadPhiCSVecOneSlice(subDir,id_dir,iSlice,verbosity=1)
    #dataMat=loadDataToFit(picoreMat,0,64,0,64,3,phiCSVecOneSlice,nBins=8,nTIs=5,verbosity=0)
    return mTagMCtrAve
