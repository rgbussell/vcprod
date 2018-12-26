import multiprocessing
import numpy as np

#def makeTaskAllSlices(tasks,maskMat,id_dir,subDir,saveDir,M0=1,alpha=1,verbosity=1,dryRun=0,nTIs=5,nBins=8,mMethod=0):
#    iSlice=0;nSlices=14
#    while iSlice<nSlices:
#        fitMask=np.zeros(np.shape(maskMat))
#        fitMask[:,:,iSlice]=maskMat[:,:,iSlice]
#        tasks.append( (iSlice, id_dir, subDir,fitMask,saveDir,M0,alpha,verbosity,dryRun,nTIs,nBins,mMethod ) )
#        iSlice+=1
#    return tasks

def makeTaskAllSlices(tasks,id_dir,subDir,saveDir,maskMat,tiVec,nX=64,nY=64,nSlices=14,nBins=8,nTIs=7,nReps=39,nTIsToFit=5,M0=1,alpha=1,mMethod=0,dryRun=0,verbosity=2):
    """Create a tuple of single slice fitting tasks for a parallel run. 
       Handles default so that no vars are assigned in the call signature so multiprocessor does not choke."""
    iSlice=0;nSlices=14
    if verbosity>=2:
        print('---makeTaskAllSlices---')
        print(' maskMat is,sum '+str(np.shape(maskMat)),',',str(np.sum(maskMat)))
        print(' saveDir is: '+saveDir)
        print(' subDir is: '+subDir)
        print(' id_dir is: '+id_dir)
        print(' nTIsToFit: '+str(nTIsToFit))
        print('------------------------------')
    while iSlice<nSlices:
        fitMask=np.zeros((nX,nY))
        #fitMask[:,:,iSlice]=np.reshape(maskMat[:,:,iSlice],(64,64))
        fitMask=maskMat[:,:,iSlice]
        print(' fitMask is,sum '+str(np.shape(fitMask)),',',str(np.sum(fitMask)))
        tasks.append( (iSlice, id_dir, subDir,fitMask,saveDir,nX,nY,nSlices,nBins,nReps,nTIs,nTIsToFit,tiVec,M0,alpha,mMethod,dryRun,verbosity ) )
        iSlice+=1
    return tasks
