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

regDir='reg'
#----------
#Data loading and plotting helper functions
#---------

def VC_loadPicoreData(subDir,id_dir,verbosity=1):
    #load all the motion corrected picore data
    #remove the first two reps
    print('VC_loadPicoreData called')
    print('. subDir is ',subDir)
    print('. id_dir is ',id_dir)
    if 0:
        import os
        print(os.getcwd())
    nDummy=2
    dataDir=subDir+'/'+id_dir+'/'+regDir+'/';
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
    poxOrderPrefix='PhysPars_TI'
    #poxOrderPrefix='PhysPars_'+str(subnum)+'_TI'
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
def loadPhiCSVecOneSlice(subDir,id_dir,iSlice,idxTagStart=0,idxCtrStart=1,nSlices=14,nTags=273,verbosity=1):
    phiCSMat=VC_loadPhiCS(subDir,id_dir,verbosity=0)
    #Get the phiCS for this slice
    nCtrls=nTags

    phiCS=phiCSMat[(iSlice-1)::nSlices,:] #select a slice (incl. t and c, and TI)
    phiCSTag=phiCS[idxTagStart::2,:] #now only take tags for this slice
    phiCSCtr=phiCS[idxCtrStart::2,:] #now only take tags for this slice

    #vectorize the inputs
    phiCSVecTagOneSlice=np.reshape(phiCSTag,(nTags,))
    phiCSVecCtrOneSlice=np.reshape(phiCSCtr,(nCtrls,))

    if verbosity>0:
        print('loadPhiCSVecOneSlice called -----')
        print('.. loading data from subDir/id_dir:', subDir+'/'+id_dir)
        print('.. slice (starting with 1)', iSlice)
        print('.. lpcsosphiCSMat shape: ', np.shape(phiCSMat))
        print('.. nTags/nCtrls  ',str(nTags),'/',str(nCtrls),'starting at index',str(idxTagStart)+'/'+str(idxCtrStart))
    
    return phiCSVecTagOneSlice,phiCSVecCtrOneSlice


def loadDataToFit(picoreMat,x1f,x2f,y1f,y2f,zf,phiCSVecCurSlice,tiVec,nBins=8,nTIs=5,verbosity=5):
    # input:
    #   picoreMat [nX nY nZ ......
    #   zf is the slice (index starts with 1)

    if verbosity>=5:
        print('loadDataToFit has picoreMat shape: ',str(np.shape(picoreMat)),' sum ',str(np.sum(picoreMat)) )   
        print(' ... has x1f,x2f,y1f,y2f: ',str(x1f),str(x2f),str(y1f),str(y2f) )   
        print(' ... has zf: ',str(zf) ) 
        print(' ... has phiCSVecCurSlice shape: ',str(np.shape(phiCSVecCurSlice)) )   
        print(' ... has tiVec shape: ',str(np.shape(tiVec)), ' ... nBins, nTIs: ',str(nBins),' ',str(nTIs) )
 
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
    print('loadDataToFit has dataMat shape', np.shape(dataMat), ' sum ', str(np.sum(dataMat)));
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


def readMultisliceFitVol(prefix,imgShape,M0,pp,sliceArr=0,nSlices=14,alpha=1,fitType=0,nPhases=8,saveNii=0,tdUB=450,tdLB=75,verb=0):
    #function
    #  read the fit output npy files for each slice, assemble into a multislice matrix
    #input 
    #  sliceArr: [nSlices,] array of slice numbers available
    #  prefix: string, prefix of fit npy name
    #  imgShape: [4,]=(nX,nY,nSlices,nPars)
    #output:
    #  fitVolPar0 [nX nY nSlices nBins] for 0th par
    #  fitVolPar1 [nX nY nSlices nBins] for 1st par
    #  fitVolParN [nX nY nSlices nBins] for nth par    
    (nX,nY,nSlices,nBins)=imageShape
    abvVol=np.zeros((nX,nY,nSlices,nBins))
    tdVol=np.zeros((nX,nY,nSlices,nBins))
    dispVol=np.zeros((nX,nY,nSlices,nBins))
    compVol=np.zeros((nX,nY,nSlices))
    abvPhaseVol=np.zeros((nX,nY,nSlices,6)) #[:,:,:,max/min/iMax/iMin/iMax-iMin/modulo(iMax-iMin,nPhases)]
    
    
    if sliceArr==0:
        sliceArr=np.arange(1,nSlices+1,1).astype(int)
    
    if fitType==0:
        nPars=3
        ABV_IDX=0;TD_IDX=1;DISP_IDX=2
    else:
        print('ERROR: fitType='+str(fitType)+' not implemented. readMultisliceFitFn EXITING.')
        return 0

    fitVec=np.zeros(nX*nY*nPars)
    for iSlice in sliceArr:
        fitFn=prefix+str(iSlice)+'.npy'
        if verb>0:
            print('reading pars from: ', fitFn)

        fitVec=np.load(fitFn)

        abvVol[:,:,iSlice-1,:]=np.reshape(fitVec[:,:,ABV_IDX],(nX,nY,nBins))
        tdVol[:,:,iSlice-1,:]=np.reshape(fitVec[:,:,TD_IDX],(nX,nY,nBins))
        dispVol[:,:,iSlice-1,:]=np.reshape(fitVec[:,:,DISP_IDX],(nX,nY,nBins))  
        abvMat=calcAbvMatB(np.reshape(fitVec,(nX,nY,nBins,nPars)),alpha=alpha,M0=M0)
        compMat,abvMaxMat,abvMinMat,abvMaxIdxMat,abvMinIdxMat=calcComp(abvMat,pp=pp)
        compVol[:,:,iSlice-1]=compMat
        abvPhaseVol[:,:,iSlice-1,0]=abvMaxMat
        abvPhaseVol[:,:,iSlice-1,1]=abvMinMat
        abvPhaseVol[:,:,iSlice-1,2]=abvMaxIdxMat
        abvPhaseVol[:,:,iSlice-1,3]=abvMinIdxMat
        abvPhaseVol[:,:,iSlice-1,4]=abvMaxIdxMat-abvMinIdxMat
        abvPhaseVol[:,:,iSlice-1,5]=np.mod(abvMaxIdxMat-abvMinIdxMat,nPhases)

        #make a masked compliance map using tdUB
        tdMask=np.zeros(np.shape(compVol))
        tmp=np.zeros(np.shape(tdVol))
        tmp[tdVol>tdUB]=1
        tmp=np.sum(tmp,3)
        tdMask[tmp==0]=1
        compTdMaskedVol=compVol*tdMask

    if saveNii==1:
        nib.save(nib.Nifti1Image(abvVol, np.eye(4)), os.path.join('abv.nii.gz'))
        nib.save(nib.Nifti1Image(tdVol, np.eye(4)), os.path.join('td.nii.gz'))
        nib.save(nib.Nifti1Image(dispVol, np.eye(4)), os.path.join('disp.nii.gz'))
        nib.save(nib.Nifti1Image(compVol, np.eye(4)), os.path.join('comp.nii.gz'))
        nib.save(nib.Nifti1Image(abvPhaseVol, np.eye(4)), os.path.join('abvPhase.nii.gz'))
        nib.save(nib.Nifti1Image(compTdMaskedVol, np.eye(4)), os.path.join('compTdMasked.nii.gz'))

    return abvVol,tdVol,dispVol,compVol,abvPhaseVol

def loadFitVol(stem):
    nX=64;nY=64;nSlice=14;nPhase=8;nParam=3
    fitVol=np.zeros((64*64,14,8,3))
    print('loadFitVol: loading these fits files fitVol')
    for iSlice in np.arange(0,14,1):
        fitFn=stem+str(iSlice)+'.npy'
        printf('||| %s\\n',fitFn)
        fitVol[:,iSlice,:,:]=np.load(fitFn)
    return np.reshape(fitVol,(nX,nY,nSlice,nPhase,nParam))


