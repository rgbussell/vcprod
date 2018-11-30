def hello():
	print('hello from' + __name__ +' in vc fit module')


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
    
import multiprocessing

from .load import *
from .view import sliceDataMontage,plot2DFit3Par
from .model import getBolusModel

#------
#Fitting models
#------

#fitfuncB=lambda p,tiVec:p[0]*2*M0*alpha*getBolusModel(transitDelay=p[1],sigma=p[2])[tiVec]
#errfuncB=lambda p,tiVec,dataVec:fitfuncB(p,tiVec)-dataVec

#------
#Fitting functions
#

def checkBounds(p,pUB,pLB):
    if np.sum(np.abs(np.where(pUB<pLB)))>0:
        return 0
    else:
        return 1
    
def checkData(dataVec,thrNegFrac=0.2):
    retVal=1
    #print('check Data has dataVec shape ', np.shape(dataVec))
    #if np.mean(dataVec)<0:
    #    retVal=0
    if np.sum( dataVec<=0 )>thrNegFrac*np.shape( dataVec )[0]:
        retVal=0
        
    return retVal

#---------
#set initial guess
#---------
def setInitialGuessB(dataVec,tiVec,M0=1,alpha=1,verbosity=1):
    nFitPars=3
    p0=np.zeros((nFitPars,))
    pLB=np.zeros((nFitPars,));
    pUB=np.zeros((nFitPars,))
    pScale=np.zeros((nFitPars,))
    if verbosity>1:
        print('setInitialGuessB has M0=',str(M0))
    if verbosity>2:
        print('setInitialGuess B has dataVec=',str(dataVec))

    fitfuncB=lambda p,tiVec:p[0]*2*M0*alpha*getBolusModel(transitDelay=p[1],sigma=p[2])[tiVec]
    errfuncB=lambda p,tiVec,dataVec:fitfuncB(p,tiVec)-dataVec

    
    tdLB=50;tdUB=800;td0=0.2*(tdLB+tdUB);
    sigmaLB=10;sigmaUB=200;sigma0=0.5*(sigmaLB+sigmaUB);
    sigMean=np.abs(np.mean(dataVec)/M0/alpha/2)
    sigMax=np.max(dataVec)/M0/alpha/2
    p0[0]=sigMax;                          pLB[0]=0.01*sigMax;        pUB[0]=2*sigMax;   pScale[0]=p0[0]
    p0[1]=td0;                             pLB[1]=tdLB;               pUB[1]=tdUB;       pScale[1]=p0[1]
    p0[2]=sigma0;                          pLB[2]=sigmaLB;            pUB[2]=sigmaUB;    pScale[2]=p0[2]

    fsWide=[40,10]

    if verbosity>0:
        plt.figure(figsize=fsWide)
        plt.plot(dataVec,label='data')
        plt.plot(fitfuncB(p0,tiVec),label='B initial guess');
        plt.title('model B\nabv='+str(p0[0])+'td='+str(p0[1])+' $\sigma=$'+str(p0[2]),fontsize=25)
        plt.legend(fontsize=25)
        
    return p0,pUB,pLB,pScale

#--------
#perform the fit
#--------

def fitB(dataMat,tiVec,saveDir,nTIsToFit,M0=1,alpha=1,saveFn='fitB',verbosity=1,dryRun=0):
    #input data: 
    #   dataMat   [nX nY nBins nTIs]
    #   tiVec     [nTIs]
    #output:
    #   fit pars for the B fit to the data [nVox nFitPars]
    #   mseMat [nVox nBins ] mean squared error of the fit
    #Break the data into individual TIs and fit separately
    #only fit 5 point data    thrNegFrac=0.4 # reject data if this fraction is negative
    nTIs=7
    #thrNegFrac=0.5
    nFitPars=3
    saveFnPartial=saveFn+'_partial'
    saveFnComplete=saveFn
   
    fitfuncB=lambda p,tiVec:p[0]*2*M0*alpha*getBolusModel(transitDelay=p[1],sigma=p[2])[tiVec]
    errfuncB=lambda p,tiVec,dataVec:fitfuncB(p,tiVec)-dataVec
 
    if 1:
        print('fitB called: saving output to', saveFnPartial)
        print('fitB has nTIsToFit=', str(nTIsToFit))
        print('fitB has saveDir='+ saveDir)
        print('fitB has tiVec='+str(tiVec))
        print('fitB has M0='+ str(M0))
        print('fitB has alpha='+ str(alpha))
        print('fitB has saveFn='+ saveFn)
        print('fitB has verbosity='+ str(verbosity))
        print('fitB has dryRun='+ str(dryRun))
    
    (nX,nY,nBins,nTIs)=np.shape(dataMat)
    nVox=nX*nY
    dataMat=np.reshape(dataMat,(nX*nY,nBins*nTIs))

    mseMat=np.zeros((nVox,nBins))
    
    fitVec=np.zeros((nVox,nBins,nFitPars))

    tStart=time.time()

    for iVox in np.arange(0,nVox,1):
        doFit=1
        if dryRun:
            doFit=0
            
        dataOneVoxel=np.reshape(dataMat[iVox,:],(nBins,nTIs))
        dataOneVoxel=dataOneVoxel[:,0:nTIsToFit]
        

        if np.sum(np.reshape(dataOneVoxel,(nBins*nTIsToFit,)))==0:
            doFit=0

        #if doFit and checkData(dataOneVoxel,thrNegFrac)==0:
        #    if verbosity>1:
        #        printf('BAD DATA, mean=%f, stdev=%f',np.mean(dataOneVoxel),np.std(dataOneVoxel))
        #    doFit=0;
        #if doFit and np.sum(dataOneVoxel)<0:
        #    doFit=0
        if doFit:
            p0,pUB,pLB,pScale=setInitialGuessB(dataOneVoxel,tiVec,M0=M0,alpha=alpha,verbosity=0)
            if verbosity>3:
                print('p0:', p0)
        if doFit and checkBounds(p0,pUB,pLB)==0:
            if verbosity>2:
                printf('BAD BOUNDS. SKIP.\n')
            doFit=0
        if doFit==1:
            if verbosity>1:
                printf('fitting voxel %d of %d\n',iVox,nVox)
            for iBin in np.arange(0,nBins,1):   #Loop through each bin
                dataOneBin=dataOneVoxel[iBin,0:nTIsToFit]
                tiVecToFit=tiVec[0:nTIsToFit]
                #temp=optimize.least_squares(errfuncB,p0,x_scale=pScale,args=(tiVec,np.reshape(dataMat[iVox,:],(nBins,nTIs))[iBin,:]),bounds=(pLB,pUB),verbose=0)
                #print("shape dataOneBin"+str(np.shape(dataOneBin)))
                #print('shape tiVec'+str(np.shape(tiVec)))
                tiVecToFit=tiVecToFit[~np.isnan(dataOneBin)]
                dataToFit=dataOneBin[~np.isnan(dataOneBin)]
                temp=optimize.least_squares(errfuncB,p0,x_scale=pScale,args=(tiVecToFit,dataToFit),bounds=(pLB,pUB),verbose=0)
                fitVec[iVox,iBin,:]=temp.x
                
                if verbosity>2:
                    print ('p0:', str(p0))
                    print('temp.x', str(temp.x))
        
                if verbosity>2:
                    #print('mse td0/tdF disp0/dispF abv0/abvF',p0[1],fitVec[iVox,iBin,1],p0[2],fitVec[iVox,iBin,2],p0[0],fitVec[iVox,iBin,0])                            
                    printf('mse%d=%0.2f  ',iBin,mseMat[iVox,iBin])                            
        eTime=time.time()-tStart;
        if np.mod(iVox,nX*10)==0:
            if verbosity>=0:
                print('Vox='+str(iVox)+' saving to '+saveFnPartial)
            np.save(saveFnPartial,fitVec)
            printf('\n')
        printf('.')
    eTime=time.time()-tStart;print('total time:',eTime)
    np.save(saveFnComplete,fitVec)
    return fitVec,mseMat

#----------
#mask prior to fitting
#----------

def getFitMask(dataVol,tiVec,M0=1,alpha=1,minMean=-1,verbosity=1):
    #input data: 
    #   dataMat   [nX nY nSlices nBins nTIs]
    #   tiVec     [nTIs]
    #   thrMin=-1  scalar  min threshold on mean across all TI and bins, -1 means do not apply threshold
    #output:
    #   fitMask [nX nY nBins]
    #     value is 1 if data passes
    #
    STAT_GOOD=1

    (nX,nY,nSlices,nBins,nTIs)=np.shape(dataVol)
    dataVol=np.reshape(dataVol,(nX*nY*nSlices,nBins,nTIs))
    nVox=nX*nY*nSlices

    fitMask=np.zeros((nVox))

    for iVox in np.arange(0,nVox,1):
        dataOneVoxel=np.reshape(dataVol[iVox,:,:],(nBins*nTIs))
        statFlg=STAT_GOOD
            #dataOneBin=dataVol[iVox,iBin,:]
        if minMean != -1:
            if np.mean(dataOneVoxel,0) < minMean:
                statFlg=0
        if statFlg and np.mean(dataOneVoxel)<=0:
            statFlg=0
        if statFlg and np.std(dataOneVoxel)/np.mean(dataOneVoxel) <0.2:
            statFlg=0
        if statFlg:
            binAve=np.mean(np.reshape(dataOneVoxel,(nBins,nTIs)),0)
            if np.std(binAve)/np.mean(binAve)<0.15:
                statFlg=0
            if statFlg and np.sum(np.where(binAve<=0))>np.ceil(0.4*nTIs):
                statFlg=0
                
        fitMask[iVox]=statFlg
    fitMask=np.reshape( fitMask,(nX,nY,nSlices))
                
    return fitMask


def makeFitMaskFile(subDir,id_dir,brainMaskFn,tiVec,nBins=8,nTIs=5,nSlices=14,nX=64,nY=64,nReps=39,minMean=-1,saveNii=1):
    # use the brain mask and also screen each voxel to make a fit mask
    # ave the fit mask to a file and also return it.
    # Assum the images is 64x64x14 with 8 bins and 5 TIs
    
    dataVol5pt=np.zeros((nX,nY,nSlices,nBins,nTIs))
    fitMask=np.zeros((nX,nY,nSlices))
    #print('tiVec: ',tiVec)

    tiArr=np.arange(250,950,100)
    tiVec=np.reshape(np.tile(tiArr,(np.int(np.floor(nReps/2)),1)),(273,))
    tiVec5pt=tiVec[tiVec<tiArr[nTIs]]

    #load in the binned data -- the full volume
    for iSlice in np.arange(0,nSlices,1):
        printf('.')
        picoreMat=np.zeros((nX,nY,nSlices,nReps,nTIs))
        picoreMat=VC_loadPicoreData(subDir, id_dir,verbosity=0)
        (phiCSVecOneSlice,junk)=loadPhiCSVecOneSlice(subDir,id_dir,iSlice+1,nSlices=nSlices,verbosity=0)
        dataVol5pt[:,:,iSlice,:,:]=loadDataToFit(picoreMat,1,nX,1,nY,iSlice+1,phiCSVecOneSlice,tiVec,nBins=8,nTIs=5)

    img=nib.load(brainMaskFn)
    brainMask=np.squeeze(img.get_data());
    np.shape(fitMask)
    fitMask=getFitMask(dataVol5pt,tiVec,M0=1,alpha=1,minMean=minMean,verbosity=1)
    mask=fitMask*brainMask
    print('fittable voxels: ',(mask==1).sum(),'out of ', np.prod(np.shape(mask)),' total')
    sliceDataMontage(mask[:,:,:,np.newaxis]);
    sliceDataMontage(brainMask[:,:,:,np.newaxis]);

    saveFn=subDir+'/fitMask'
    np.save(saveFn,mask)
    print('makeFitMaskFile: mask saved to ',saveFn)
    
    if saveNii:
        nib.save(nib.Nifti1Image(fitMask,np.eye(4)),os.path.join(subDir,'fitMask.nii.gz'))
    
    return fitMask

def fitWithinMask(id_dir,subDir,mask,saveDir,M0=1,alpha=1,verbosity=0,dryRun=0):
    # volSz [5,] = (nX,nY,nSlices,nReps,nTIs)
    volSz=(64,64,14,78,7)

    #(nX,nY,nSlices,nReps,nTIs)=volSz
    tiVecOne=np.arange(250,750,100)    
    picoreMat=np.zeros((nX,nY,nSlices,nReps,nTIs))
    picoreMat=VC_loadPicoreData(subDir, id_dir,verbosity=0)
    
    for iSlice in np.arange(0,3,1):
        print('.......fitWithinMask setting up fit data for slice ',iSlice,'..............')
        saveFn=saveDir+'fitB_slice'+str(iSlice)
        mseSaveFn=saveDir+'mse_slice'+str(iSlice)   
        (phiCSVecOneSlice,junk)=loadPhiCSVecOneSlice(subDir,id_dir,iSlice+1,verbosity=1)
        plt.figure(figsize=[40,5]);plt.plot(phiCSVecOneSlice,label='slice'+str(iSlice));
        plt.title('$\phi_c^s$',fontsize=30);plt.legend(fontsize=20);plt.show()
        dataMat=loadDataToFit(picoreMat,1,nX,1,nY,iSlice+1,phiCSVecOneSlice,tiVec,nBins=8,nTIs=5)
        sliceDataMontage(dataMat)
        dataMatMasked=dataMat*np.tile(fitMask[:,:,iSlice,np.newaxis,np.newaxis],(1,1,8,5))
        sliceDataMontage(dataMatMasked)
        fitVec,mseMat=fitB(dataMatMasked,tiVec,M0=M0,alpha=alpha,saveFn=saveFn,verbosity=verbosity,dryRun=dryRun)
        plot2DFit3Par(fitVec,fitMatType=0)
        np.save(saveFn,fitVec)
        np.save(mseSaveFn,mseMat)


def fitWithinMaskPar2p0_test(iSlice,id_dir,subDir,fitMask,saveDir,nTIsToFit,tiVec,nX=64,nY=64,nSlices=14,nTIs=7,nReps=39,M0=1,alpha=1,verbosity=0,dryRun=0,nBins=8,mMethod=0):
    #fitWithinMaskPar2p0_test(0, '119_180612', 'data/PupAlz_119',fitMask,'data/PupAlz_119/119_180612/vISMRM2019.0/7TIs/m0/', 7, 39, 1, 0, 0, 8, 0)
    # volSz [5,] = (nX,nY,nSlices,nReps,nTIs)
    #volSz=(64,64,14,78,7)
    print(str(fitMask))
    
    saveFn=saveDir+'/fitB_slice'+str(iSlice)
    mseSaveFn=saveDir+'/mse_slice'+str(iSlice) 
    
    if 1:
        print('fitWithinMaskPar2p0_test has: iSlice '+str(iSlice))
        print('fitWithinMaskPar2p0_test has: id_dir '+str(id_dir))
        print('fitWithinMaskPar2p0_test has: subDir '+str(subDir))
        print('fitWithinMaskPar2p0_test has: nTIsToFit '+str(nTIsToFit))
        print('fitWithinMaskPar2p0_test has: mask size '+str(np.shape(fitMask)))
        print('fitWithinMaskPar2p0_test has: M0 '+str(M0))
        print('fitWithinMaskPar2p0_test has: alpha '+str(alpha))
        print('fitWithinMaskPar2p0_test has: verbosity '+str(verbosity))
        print('fitWithinMaskPar2p0_test has: dryRun '+str(dryRun))
        print('fitWithinMaskPar2p0_test has: nBins '+str(nBins))
        print('fitWithinMaskPar2p0_test has: mMethod '+str(mMethod))
        print('fitWithinMaskPar2p0_test has: saveFn '+str(saveFn))
        print('fitWithinMaskPar2p0_test has: tiVec '+str(tiVec))
        
    #(nX,nY,nSlices,nReps,nTIs)=volSz
    #tiVecOne=np.arange(250,750,100)    
    picoreMat=np.zeros((nX,nY,nSlices,nReps,nTIs))
    picoreMat=VC_loadPicoreData(subDir, id_dir,verbosity=0)
    
    #for iSlice in np.arange(0,3,1):
    print('.......fitWithinMask setting up fit data for slice ',iSlice,'..............')
  
    (phiCSVecOneSlice,junk)=loadPhiCSVecOneSlice(subDir,id_dir,iSlice+1,verbosity=1)
    plt.figure(figsize=[40,5]);plt.plot(phiCSVecOneSlice,label='slice'+str(iSlice));
    plt.title('$\phi_c^s$',fontsize=30);plt.legend(fontsize=20);plt.show()
    dataMat=loadDataToFit(picoreMat,1,nX,1,nY,iSlice+1,phiCSVecOneSlice,tiVec,nBins=8,nTIs=nTIs)
    sliceDataMontage(dataMat)
    print('fitWithinMaskPar2p0_test: shape fitMask '+str(np.shape(fitMask)))
    dataMatMasked=dataMat*np.tile(fitMask[:,:,iSlice,np.newaxis,np.newaxis],(1,1,nBins,nTIs))
    sliceDataMontage(dataMatMasked)
    print('saveDir is '+str(saveDir))
    print('dataMatMasked is '+str(np.shape(dataMatMasked)))
    if 1:
        print('---fitWithinMaskPar2p0_test---')
        print(' dataMatMasked is '+str(np.shape(dataMatMasked)))
        print(' saveDir is: '+saveDir)
        print(' tiVec is: '+str(tiVec))
        print(' saveFn: '+saveFn)
        print(' nTIs: '+str(nTIs))
        print('------------------------------')
    if 1:
        fitVec,mseMat=fitB(dataMatMasked,tiVec,saveDir,nTIsToFit,M0=M0,alpha=alpha,saveFn=saveFn,verbosity=verbosity,dryRun=dryRun)
        plot2DFit3Par(fitVec,fitMatType=0)

