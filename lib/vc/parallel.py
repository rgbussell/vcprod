import multiprocessing

def makeTaskAllSlices(tasks,maskMat,id_dir,subDir,saveDir,M0=1,alpha=1,verbosity=1,dryRun=0,nTIs=5,nBins=8,mMethod=0):
    iSlice=0;nSlices=14
    while iSlice<nSlices:
        fitMask=np.zeros(np.shape(maskMat))
        fitMask[:,:,iSlice]=maskMat[:,:,iSlice]
        tasks.append( (iSlice, id_dir, subDir,fitMask,saveDir,M0,alpha,verbosity,dryRun,nTIs,nBins,mMethod ) )
        iSlice+=1
    return tasks

def makeTaskAllSlices2p0test(tasks,id_dir,subDir,nTIsToFit,maskMat,saveDir,M0=1,alpha=1,verbosity=1,dryRun=0,nBins=8,mMethod=0):
    iSlice=0;nSlices=14
    if verbosity>=2:
        print('---makeTaskAllSlices2p0test---')
        print(' maskMat is '+str(np.shape(maskMat)))
        print(' saveDir is: '+saveDir)
        print(' subDir is: '+subDir)
        print(' id_dir is: '+id_dir)
        print(' nTIsToFit: '+str(nTIsToFit))
        print('------------------------------')
    while iSlice<nSlices:
        fitMask=np.zeros(np.shape(maskMat))
        fitMask[:,:,iSlice]=maskMat[:,:,iSlice]
        tasks.append( (iSlice, id_dir, subDir,fitMask,saveDir,nTIsToFit,M0,alpha,verbosity,dryRun,nBins,mMethod ) )
        #fitWithinMaskPar2p0_test(0, '119_180612', 'data/PupAlz_119',fitMask,'data/PupAlz_119/119_180612/vISMRM2019.0/7TIs/m0/', 7, 39, 1, 0, 0, 8, 0)
        iSlice+=1
    return tasks
