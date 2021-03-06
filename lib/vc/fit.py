FIGURES_ON=0
FIGURES_ONSCREEN=0
FIGURES_SAVED=0

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
#Fitting functions
#

def checkBounds(p,pUB,pLB,verb=1):
    if verb>0:
        print('pUB',pUB)
        print('pLB',pLB)
        print('checkBounds: called')
    if np.sum( np.abs( np.where(pUB<=pLB) ) )>0:
        return 0
        print('checkBounds: bad bounds')
    else:
        return 1
    if verb>0:
        print('checkBounds exiting')

def checkData(dataVec,thrNegFrac=0.2):
    retVal=1
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
    if verbosity>1:
        print('setInitialGuess B has dataVec=\n',str(dataVec))

    fitfuncB=lambda p,tiVec:p[0]*2*M0*alpha*getBolusModel(transitDelay=p[1],sigma=p[2])[tiVec]
    errfuncB=lambda p,tiVec,dataVec:fitfuncB(p,tiVec)-dataVec

    
    tdLB=50;tdUB=800;td0=0.2*(tdLB+tdUB);
    sigmaLB=10;sigmaUB=200;sigma0=0.5*(sigmaLB+sigmaUB);
    sigMean=np.abs(np.mean(dataVec[~np.isnan(dataVec)])/M0/alpha/2)
    sigMax=np.max(dataVec[~np.isnan(dataVec)])/M0/alpha/2
    p0[0]=sigMax;                          pLB[0]=0.01*sigMax;        pUB[0]=2*sigMax;   pScale[0]=p0[0]
    p0[1]=td0;                             pLB[1]=tdLB;               pUB[1]=tdUB;       pScale[1]=p0[1]
    p0[2]=sigma0;                          pLB[2]=sigmaLB;            pUB[2]=sigmaUB;    pScale[2]=p0[2]

    return p0,pUB,pLB,pScale

#--------
#perform the fit
#--------

def fitB(dataMat,tiVec,saveDir,nTIsToFit,M0=1,alpha=1,saveFn='fitB',verbosity=5,dryRun=0):
    #input data: 
    #   dataMat   [nX nY nBins nTIs]
    #   tiVec     [nTIs]
    #output:
    #   fit pars for the B fit to the data [nVox nFitPars]
    #   mseMat [nVox nBins ] mean squared error of the fit
    #Break the data into individual TIs and fit separately
    #only fit 5 point data    thrNegFrac=0.4 # reject data if this fraction is negative
    DEBUG_NPOINTS_TO_FIT=0
    #verbosity=5
    nTIs=7
    #thrNegFrac=0.5
    nFitPars=3
    saveFnPartial=saveFn+'_partial';mseFnPartial=saveFnPartial+'_mse'
    saveFnComplete=saveFn;mseFnComplete=saveFnComplete+'_mse'
    fitFnComplete=saveFn+"_fit";fitFnPartial=saveFn+'_fit_partial'
    saveFnDataMat=saveFn+'_dataMat';
    np.save(saveFnDataMat,dataMat)

    fitfuncB=lambda p,tiVec:p[0]*2*M0*alpha*getBolusModel(transitDelay=p[1],sigma=p[2])[tiVec]
    errfuncB=lambda p,tiVec,dataVec:fitfuncB(p,tiVec)-dataVec
 
    if verbosity>1:
        print('fitB called: dataMat has shape '+str(np.shape(dataMat))+' sum '+str(np.sum(dataMat)))
        print('... saving partial output to',saveFnPartial)
        print('... saving complete output to',saveFnComplete)
        print('... saving partial mse output to',mseFnPartial)
        print('... saving complete mse output to',mseFnComplete)
        print('... has saveDir='+ saveDir)
        print('... has tiVec shape='+str(np.shape(tiVec)))
        print('... has M0,alpha,nTIsToFit,dryRun', str(M0),str(alpha),str(nTIsToFit),str(dryRun))
        print('... has saveFn='+ saveFn)
        print('... has dryRun='+ str(dryRun))
    
    (nX,nY,nBins,nTIs)=np.shape(dataMat)

    nVox=nX*nY
    if DEBUG_NPOINTS_TO_FIT>0:
        nVox=min(nVox,DEBUG_NPOINTS_TO_FIT)

    #inits
    dataMat=np.reshape(dataMat,(nX*nY,nBins*nTIs))
    mseMat=np.zeros((nVox,nBins))
    fitVec=np.zeros((nVox,nBins,nFitPars))
    #this is the fit vector
    fit=np.zeros((nVox,nBins,nTIsToFit))

    tStart=time.time()

    #loop over voxels, fitting each one
    for iVox in np.arange(0,nVox,1):
        doFit=1
        if dryRun:
            doFit=0
            
        dataOneVoxel=np.reshape(dataMat[iVox,:],(nBins,nTIs))
        dataOneVoxel=dataOneVoxel[:,0:nTIsToFit]
        
        if np.sum(np.reshape(dataOneVoxel,(nBins*nTIsToFit,)))==0:
            doFit=0
            if verbosity>4:
                print('fitB: sum dataOneVoxel=0')
        if doFit:
            p0,pUB,pLB,pScale=setInitialGuessB(dataOneVoxel,tiVec,M0=M0,alpha=alpha,verbosity=5)

            if verbosity>3:
                print('p0:', p0)
        if doFit and checkBounds(p0,pUB,pLB)==0:
            doFit=0
            if verbosity>2:
                printf('BAD BOUNDS. SKIP.\n')
        if doFit==1:
            if verbosity>2:
                printf('fitting voxel %d of %d\n',iVox,nVox)
            for iBin in np.arange(0,nBins,1):   #Loop through each bin
                dataOneBin=dataOneVoxel[iBin,0:nTIsToFit]
                tiVecToFit=tiVec[0:nTIsToFit]
                tiVecToFit=tiVecToFit[~np.isnan(dataOneBin)]
                dataToFit=dataOneBin[~np.isnan(dataOneBin)]
                try:
                    temp=optimize.least_squares(errfuncB,p0,x_scale=pScale,args=(tiVecToFit,dataToFit),bounds=(pLB,pUB),verbose=0)
                    fitVec[iVox,iBin,:]=temp.x
                    mseMat[iVox,iBin]=temp.cost
                    #fit[iVox,iBin,:]=fitfuncB(temp.x,tiVec[0:nTIsToFit])
                    fit[iVox,iBin,:]=np.zeros(nTIsToFit)
                except:
                    if verbosity>2:
                    	print('optimize.least_squares threw an exception') 
                if verbosity>2:
                    print ('fitB: p0:', str(p0))
                    print('fitB: temp.x', str(temp.x))
                    print('fitB: temp.cost', str(temp.cost))
                    print('fitB: shape fitVec ', str(np.shape(fitVec)), 'sum fitVec ', str(np.sum(fitVec)))
                    print('fitB: shape mseMat ', str(np.shape(mseMat)), 'sum mseMat ', str(np.sum(mseMat)))
        
                if verbosity>2:
                    printf('fitB: mse%d=%0.2f  ',iBin,mseMat[iVox,iBin])                            
        eTime=time.time()-tStart;
        if np.mod(iVox,nX*10)==0:
            np.save(saveFnPartial,fitVec)
            np.save(mseFnPartial,mseMat)
            np.save(fitFnPartial,fit)
            if verbosity>=0:
                print('fitB: Vox='+str(iVox)+' saving to '+saveFnPartial)
            printf('\n')
        printf('.')
    eTime=time.time()-tStart;print('fitB: total time:',eTime)
    np.save(saveFnComplete,fitVec)
    np.save(mseFnComplete,mseMat)
    np.save(fitFnComplete,fit)
    if verbosity>0:
        print('fitB: fitVec has size ', str(np.shape(fitVec)), ' and sum ',str(np.sum(fitVec)) )
        print('fitB: saving fitVec to file ', saveFnComplete)
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
    print('getFitMask called')
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


def makeFitMaskFile(subDir,id_dir,brainMaskFn,nBins=8,nTIs=5,nSlices=14,nX=64,nY=64,nReps=39,minMean=-1,pctThr=-1,saveNii=1,mode=0,percentile=0):
    """makeFitMaskFile: internal function for combining a threshold mask with brain mask in compliance pipeline.
       If you set minMean=-1 (the default) no mean threshold will be applied and you get your input mask back."""
    # use the brain mask and also screen each voxel to make a fit mask
    # ave the fit mask to a file and also return it.
    
    dataVol5pt=np.zeros((nX,nY,nSlices,nBins,nTIs))
    fitMask=np.zeros((nX,nY,nSlices))

    tiArr=np.arange(250,950,100)
    tiVec=np.reshape(np.tile(tiArr,(np.int(np.floor(nReps/2)),1)),(273,))
    tiVec5pt=tiVec[tiVec<tiArr[nTIs]]

    #load in the precalculated brain mask
    img=nib.load(brainMaskFn)
    brainMask=np.squeeze(img.get_data());

    #do mean thresholding if that is requested
    if minMean == -1 and percentile == 0:     #no masking case
        print('makeFitMaskFile: no thresholding will be applied to input brain mask')
        fitMask=np.ones(np.shape(brainMask))
    if percentile == 0 and minMean > -1:                              #do some masking 
        print('makeFitMaskFile: thresholding on mean signal at ', str(minMean));
        picoreMat=np.zeros((nX,nY,nSlices,nReps,nTIs))
        picoreMat=VC_loadPicoreData(subDir, id_dir,verbosity=0)
        for iSlice in np.arange(0,nSlices,1):
            printf('.')
            (phiCSVecOneSlice,junk)=loadPhiCSVecOneSlice(subDir,id_dir,iSlice+1,nSlices=nSlices,verbosity=0)
            dataVol5pt[:,:,iSlice,:,:]=loadDataToFit(picoreMat,1,nX,1,nY,iSlice+1,phiCSVecOneSlice,tiVec,nBins=8,nTIs=5)
        fitMask=getFitMask(dataVol5pt,tiVec,M0=1,alpha=1,minMean=minMean,verbosity=1)

    if percentile > 0 and minMean == -1:
        #do percentile masking and brain mask
        picoreMat=np.zeros((nX,nY,nSlices,nReps,nTIs))
        picoreMat=VC_loadPicoreData(subDir, id_dir,verbosity=0)
        picoreMat=np.tile(brainMask[:,:,:,np.newaxis,np.newaxis],(1,1,1,78,7))*picoreMat
        idxTagStart=0;idxCtrStart=1
        deltaMat=picoreMat[:,:,:,idxCtrStart::2,:]-picoreMat[:,:,:,idxTagStart::2,:]
    
        #average over reps and TIs
        tmpMat=np.mean(np.mean(deltaMat,3),3)
        tmpMat=np.reshape(tmpMat,(np.int(nX*nY),np.int(nSlices)))
        fitMask=np.zeros(np.shape(tmpMat))

        thrVec=np.percentile( tmpMat,percentile,0)

        for iThr in np.arange(0,nSlices,1):
            fitMask[tmpMat[:,iThr]>thrVec[iThr],iThr]=1

        fitMask=np.reshape(fitMask,(nX,nY,nSlices))
 
    #apply the mask
    mask=fitMask*brainMask

    print('makeFitMaskFile: fittable voxels: ',(mask==1).sum(),'out of ', np.prod(np.shape(mask)),' total')
    if FIGURES_ONSCREEN:
        sliceDataMontage(mask[:,:,:,np.newaxis]);
        sliceDataMontage(brainMask[:,:,:,np.newaxis]);

    #save the mask
    saveFn=subDir+'/fitMask'
    np.save(saveFn,mask)
    print('makeFitMaskFile: mask saved to ',saveFn)
    
    if saveNii:
        nib.save(nib.Nifti1Image(fitMask,np.eye(4)),os.path.join(subDir,'fitMask.nii.gz'))
    
    return fitMask

def fitWithinMaskPar(iSlice,id_dir,subDir,fitMask,saveDir,nX,nY,nSlices,nBins,nReps,nTIs,nTIsToFit,tiVec,M0,alpha,mMethod,dryRun,verbosity):
    """"Load data, take a mask as input and call the fit function in a way that is suitable for MP execution."""

    verbosity=2
    if verbosity>0:
        print('fitWithinMaskPar called')
    saveFn=saveDir+'/fitB_slice'+str(iSlice)
    mseSaveFn=saveDir+'/mse_slice'+str(iSlice) 
    
    if verbosity>=1:
        print(' ... has: iSlice '+str(iSlice))
        print(' ... has: id_dir '+str(id_dir))
        print(' ... has: subDir '+str(subDir))
        print(' ... has: nTIs '+str(nTIs))
        print(' ... has: nTIsToFit '+str(nTIsToFit))
        print(' ... has: fitMask size '+str(np.shape(fitMask)))
        print(' ... has: M0 '+str(M0))
        print(' ... has: alpha '+str(alpha))
        print(' ... has: verbosity '+str(verbosity))
        print(' ... has: dryRun '+str(dryRun))
        print(' ... has: nBins '+str(nBins))
        print(' ... has: mMethod '+str(mMethod))
        print(' ... has: saveFn '+str(saveFn))
        print(' ... has: tiVec shape '+str(np.shape(tiVec)))
        
    picoreMat=np.zeros((nX,nY,nSlices,nReps,nTIs))
    picoreMat=VC_loadPicoreData(subDir, id_dir,verbosity=0)
    print(' ... has picoreMat size'+str(np.shape(picoreMat))+' sum '+str(np.sum(picoreMat)))
    print(' ... setting up fit data for slice ',iSlice,'..............')
  
    (phiCSVecOneSlice,junk)=loadPhiCSVecOneSlice(subDir,id_dir,iSlice+1,verbosity=1)
    if 0:
        plt.figure(figsize=[40,5]);plt.plot(phiCSVecOneSlice,label='slice'+str(iSlice));
        plt.title('$\phi_c^s$',fontsize=30);plt.legend(fontsize=20);plt.ion();plt.show()
    dataMat=loadDataToFit(picoreMat,1,nX,1,nY,iSlice+1,phiCSVecOneSlice,tiVec,nBins=8,nTIs=nTIs)
    if 0:
        sliceDataMontage(dataMat)
    if verbosity>1:
        print(' ...: shape fitMask,sum fitMask '+str(np.shape(fitMask))+' '+str(np.sum(fitMask)))
        print(' ...: shape dataMat,sum '+str(np.shape(dataMat)),',',str(np.sum(dataMat)))
        print(' ...: before masking')
    #fitMask=np.ones(np.shape(fitMask))
    tmpData=np.reshape(dataMat[:,:,:,0:nTIsToFit:1],(nX,nY,nBins,nTIsToFit))
    tmpMask=np.reshape( np.tile( fitMask[:,:,np.newaxis,np.newaxis], (1,1,nBins,nTIsToFit) ), (nX,nY,nBins,nTIsToFit) )
    dataMatMasked=np.zeros(np.shape(dataMat))
    #This seems pretty inefficient, but I am getting an error from python fromnumeric.py if I use np.multiply
    #dataMatMasked=np.multiply(tmpData,tmpMask)
    for iX in np.arange(0,nX,1):
        for iY in np.arange(0,nY,1):
            for iBin in np.arange(0,nBins,1):
                for iTI in np.arange(0,nTIsToFit,1):
                    dataMatMasked[iX,iY,iBin,iTI]=dataMat[iX,iY,iBin,iTI]*fitMask[iX,iY]
    if verbosity>1:
        print(' ...: after masking')
        print(' ...: shape dataMat,sum '+str(np.shape(dataMat)),',',str(np.sum(dataMat)))

    #dataMatMasked=np.dot(dataMat[:,:,:,0:nTIsToFit:1],np.tile(fitMask[:,:,np.newaxis,np.newaxis],(1,1,nBins,nTIsToFit)))
    #datamatmasked=datamat[:,:,:,0:ntistofit:1]*np.tile(fitmask[:,:,islice,np.newaxis,np.newaxis],(1,1,nbins,ntistofit))
    #dataMatMasked=np.ones(np.shape(dataMatMasked))
    if FIGURES_ONSCREEN:
        sliceDataMontage(dataMatMasked)
    if verbosity>1:
        print(' ... saveDir is '+str(saveDir))
        print(' ... dataMatMasked is shape '+str(np.shape(dataMatMasked))+' sum '+str(np.sum(dataMatMasked)))
        print(' ... dataMatMasked is '+str(np.shape(dataMatMasked)))
        print(' ... saveDir is: '+saveDir)
        print(' tiVec is shape: '+str(np.shape(tiVec))+' has max min '+str(np.max(tiVec))+' '+str(np.min(tiVec)))
        print(' ... saveFn: '+saveFn)
        print(' ... nTIs: '+str(nTIs))
    if 1:
        fitVec,mseMat=fitB(dataMatMasked,tiVec,saveDir,nTIsToFit,M0=M0,alpha=alpha,saveFn=saveFn,verbosity=verbosity,dryRun=dryRun)
        if 0:
            plot2DFit3Par(fitVec,fitMatType=0)

