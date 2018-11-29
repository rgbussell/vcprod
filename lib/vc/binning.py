import numpy as np

#Functions for reading patches, plotting them and binning

def getMTagMCtrAvePatch(picoreMat,x1,x2,y1,y2,z1,z2,idxTagStart=0,idxCtrStart=1):
    return getPatch(picoreMat,x1,x2,y1,y2,z1,z2,patchType=0,idxTagStart=0,idxCtrStart=1)

def getMTagPatch(picoreMat,x1,x2,y1,y2,z1,z2,idxTagStart=0,idxCtrStart=1):
    return getPatch(picoreMat,x1,x2,y1,y2,z1,z2,patchType=1,idxTagStart=0,idxCtrStart=1)

def getMCtrPatch(picoreMat,x1,x2,y1,y2,z1,z2,idxTagStart=0,idxCtrStart=1):
    return getPatch(picoreMat,x1,x2,y1,y2,z1,z2,patchType=2,idxTagStart=0,idxCtrStart=1)

def getPatch(picoreMat,x1,x2,y1,y2,z1,z2,patchType,idxTagStart=0,idxCtrStart=1):
    #input picore raw data for all TIS and reps
    #output <M_ctr>-M_tag [nx,ny,nz,nRep,nTI]
    nTIs=np.shape(picoreMat)[4]
    nReps=np.shape(picoreMat)[3]
    nVoxels=int((x2-x1+1)*(y2-y1+1)*(z2-z1+1))

    patch=picoreMat[x1-1:x2:1,y1-1:y2:,z1-1:z2:,0:78:,0:nTIs:]
    patch=np.reshape(patch,(nVoxels,nReps,nTIs))
    mTag=patch[:,idxTagStart::2,:]
    mCtrAve=np.mean(patch[:,idxCtrStart::2,:],1)
    mTagMCtrAvePatch=np.tile(np.reshape(mCtrAve,(nVoxels,1,nTIs)),(1,int(nReps/2),1))-mTag
    mTagMCtrAvePatch=np.reshape(mTagMCtrAvePatch,(nVoxels,int(nReps*nTIs/2)))

    if patchType==0: # <M_{ctr}>-M_{tag}
        return mTagMCtrAvePatch
    if patchType==1: # M_tag only
        return np.reshape(picoreMat[x1-1:x2:1,y1-1:y2:,z1-1:z2:,idxTagStart::2,0:nTIs:],(nVoxels,int(nReps*nTIs/2)))
    if patchType==2: #M_{ctr} only
        return np.reshape(picoreMat[x1-1:x2:1,y1-1:y2:,z1-1:z2:,idxCtrStart::2,0:nTIs:],(nVoxels,int(nReps*nTIs/2)))

    
def binData(dataVec,tiVec,phiCSVec,nBins=8,nTIs=7):
#Returns a 2D binned array of dims [nBins nTIs]
#i.e. phiCSVec is reduced to some number of bins, nBins
    dataBinMat=np.zeros((nBins,nTIs))
    iBin=0;
    binStarts = np.arange(0,1,1/nBins)
    for binStart in binStarts:
        binEnd=binStart+1/nBins
        iTI=0;
        TI=np.arange(250,950,100)
        #print(binStart,binEnd)
        for curTI in TI:
            dataBinMat[iBin,iTI]=np.mean(dataVec[ (tiVec==curTI) & (phiCSVec>=binStart) & (phiCSVec<binEnd) ])
            iTI=iTI+1
        iBin=iBin+1
    
    return dataBinMat

def binAPatch(patch,tiVec,phiCSVec,nBins=8,nTIs=7):
    #input [nVox, nTI*nReps]
    #output [nVox, nBins, nTIs]
    nVox=np.shape(patch)[0]
    binPatch=np.zeros((nVox,nBins,nTIs))

    for iVox in np.arange(0,nVox,1):
        binPatch[iVox,:,:]=binData(patch[iVox,:],tiVec,phiCSVec,nBins=8)

    return binPatch

def getBinnedPhiCSVec(nBins,nTIs):
    return np.reshape(np.tile(np.reshape(np.arange(0,nBins,1)/(nBins-1),(nBins,1)),(1,nTIs)),(nBins*nTIs,))

def getBinnedTiVec(nBins,nTIs):
    tiArr=np.arange(250,nTIs*100+250,100)
    return np.reshape(np.tile(np.reshape(tiArr,(1,nTIs)),(nBins,1)),(nBins*nTIs,)).astype(int)
