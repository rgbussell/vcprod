import numpy as np
import matplotlib.pyplot as plt

#Tools for finding cardiac phase
def getExtremePhiMap(abvMat,tiVec,phiCSVec,method=0,verbosity=1):
    #create a map from the extreme values of the phase across bins
    # m method 0 for min, 1 for max
    # data: [nX nY nBins]
    print('getExtremePhiMap: WIP')
    (nX,nY,nBins)=np.shape(abvMat)
    phiMap=np.zeros((nX,nY))
    if method==0:
        phiMap=np.argmin(abvMat,2)
        ttlStr='$argmax( f(\phi_c^s) )$'
    if method==1:
        phiMap=np.argmax(abvMat,2)
        ttlStr='$argmin( f(\phi_c^s) )$'
    if method==2:
        phiMap=getExtremePhiMap(abvMat,tiVec,phiCSVec,method=1,verbosity=0)-getExtremePhiMap(abvMat,tiVec,phiCSVec,method=0,verbosity=0)
        ttlStr='$argmax( f(\phi_c^s) ) - argmin( f(\phi_c^s) )$'

    if verbosity>0:
        print('np.shape(phiMap)'+str(np.shape(phiMap)))
        plt.imshow(phiMap,cmap='viridis');plt.title(ttlStr,fontsize=20);plt.colorbar();plt.show()
        
    return phiMap

def getDeltaPhi(data,tiVec,phiCSVec,mask):
    #estimate max phase (bin) based on various methods
    # m method
    print('getDeltaPhi')
    return getMaxPhi(data,tiVec,phiCSVec,mask)-getMinPhi(data,tiVec,phiCSVec,mask)

def plotPhaseCurves(fitVec):
    binVec=np.arange(0,8,1)
    (nX,nY,nBins,nPars)=np.shape(fitVec)
    fitVec=np.reshape(fitVec,(nX*nY,nBins,nPars))
    totAbv=np.zeros(np.shape(fitVec[0,:,0]))
    totTd=totAbv;totDisp=totAbv  

    cntVox=0
    for iVox in np.arange(0,nX*nY,1):
        normAbv=np.sum(fitVec[iVox,:,0])
        totAbv=totAbv+fitVec[iVox,:,0]
        normTd=np.sum(fitVec[iVox,:,1])
        totTd=totTd+fitVec[iVox,:,1]
        normDisp=np.sum(fitVec[iVox,:,2])
        totDisp=totDisp+fitVec[iVox,:,2]    
        cntVox+=(not np.all(fitVec[iVox,:,:])==0)
    print('cntVox='+str(cntVox))
    totAbv=totAbv/cntVox
    totTd=totTd/cntVox
    totDisp=totDisp/cntVox
    
        
    plt.figure()
    plt.plot(binVec,totAbv,label='abv norm ave');
    plt.title('average of abv bin estimates');plt.xlabel('bin num (arb)')
    plt.show()
    
    plt.figure()
    plt.plot(binVec,totTd,label='TD norm ave');
    plt.title('average of TD bin estimates');plt.xlabel('bin num (arb)')
    plt.show()
    
    plt.figure()
    plt.plot(binVec,totDisp,label='disp norm ave');
    plt.title('average of disp bin estimates');plt.xlabel('bin num (arb)')
    plt.show()

# abv calculation functions here
def calcAbvMatB(fitMat,alpha=1,M0=1):
    #input: 
    #  fitMat is [nX nY nBins 3]
    #  where the 3 parmeters are ABV-TD-SIGMA, in that order
    # output:
    #  abvMat [nX nY nBins]
    print('input is ', np.shape(fitMat))
    
    nX=np.shape(fitMat)[0];nY=np.shape(fitMat)[1]
    nBins=np.shape(fitMat)[2]
    #nTIs=np.shape(tiVec)[0]
    nVox=nX*nY
    fitMatVec=np.reshape(fitMat,(nVox,nBins,3))
    abvVec=np.zeros((nVox,nBins))
    abvVec=fitMatVec[:,:,0]/alpha/M0/2
    abvMat=np.reshape(abvVec,(nX,nY,nBins))
    return abvMat


def getTdMatB(fitMat):
    #input: 
    #  fitMat is [nX nY nBins 3]
    #  where the 3 parmeters are ABV-TD-SIGMA, in that order
    # output:
    #  tdMat [nX nY nBins]    
    nX=np.shape(fitMat)[0];nY=np.shape(fitMat)[1]
    nBins=np.shape(fitMat)[2]
    nVox=nX*nY
    fitMatVec=np.reshape(fitMat,(nVox,nBins,3))
    tdVec=np.zeros((nVox,nBins))
    tdVec=fitMatVec[:,:,1]
    tdMat=np.reshape(tdVec,(nX,nY,nBins))
    return tdMat

def calcComp(abvMat,tiVec,phiCSVec,pp=1,tdMat='',method=0):
    #input:
    #  abvMat [nX nY nBins]
    nX=np.shape(abvMat)[0]
    nY=np.shape(abvMat)[1]
    nBins=np.shape(abvMat)[2]
    compMat=np.zeros((nX,nY))
    abvMaxMat=np.zeros((nX,nY))
    abvMinMat=np.zeros((nX,nY))
    abvMaxIdxMat=np.zeros((nX,nY))
    abvMaxIdxMat=np.zeros((nX,nY))

    if method==0:   #ID phases based on abv max and min
        print('calcComp: Estimating $\phi_c^s$ based on abvmax and abvmin')
        abvMaxMat=100*np.max(abvMat,2)
        abvMinMat=100*np.min(abvMat,2)
        abvMaxIdxMat=np.argmax(abvMat,2)
        abvMinIdxMat=np.argmin(abvMat,2)

    if method==1:   #ID phases based on the td max and min
        print('calcComp: Estimating $\phi_c^s$ based on tdmax and tdmin')
        if tdMat=='':
            print('ERROR: calcComp needs a tdMat for this method='+str(method))
            return 0
        abvMaxIdxMat=np.argmax(tdMat,2)
        abvMinIdxMat=np.argmin(tdMat,2)
        abvMaxMat=np.zeros((nX,nY))
        abvMinMat=np.zeros((nX,nY))
        for i in np.arange(0,nX,1):
            for j in np.arange(0,nY,1):
                abvMaxMat[i,j]=100*abvMat[i,j,abvMaxIdxMat[i,j]]
                abvMinMat[i,j]=100*abvMat[i,j,abvMinIdxMat[i,j]]
              
    if method==2 or method==3:
        #data=np.tile(abvMat[:,np.newaxis],(1,1,3))
        data=abvMat
        print('shape data= '+str(np.shape(data)))
        mask=np.ones(np.shape(abvMat))
        mask[np.sum(abvMat)==0]=0
    
    if method==2:
        #ID phases based on using the mean phase of the entire slice
        maxPhiMap=getExtremePhiMap(data,tiVec,phiCSVec,method=0,verbosity=1)
        minPhiMap=getExtremePhiMap(data,tiVec,phiCSVec,method=1,verbosity=1)
        maxPhiSc=np.sum(maxPhiMap)/np.sum(maxPhiMap>0)
        minPhiSc=np.sum(minPhiMap)/np.sum(minPhiMap>0)
        
        abvMaxMat=100*abvMat[:,:,maxPhiSc]
        abvMinMat=100*abvMat[:,:,minPhiSc]
        abvMinIdxMat=np.ones((nX,nY))*minPhiSc
        abvMaxIdxMat=np.ones((nX,nY))*maxPhiSc

    if method==3:
        #ID phases based on the max for the voxel
        #and difference between max and min phase
        # estimated across the entire slice
        deltaPhi=getExtremePhiMap(data,tiVec,phiCSVec,method=2,verbosity=1)
        print('shape abvMat ', str(np.shape(abvMat)))
        #print('shape idxMaxBin ', str(np.shape(idxMaxBin)))
        print('shape deltaPhi ', str(np.shape(deltaPhi)))
        abvMaxIdxMat=np.zeros((nX,nY))
        abvMinIdxMat=np.zeros((nX,nY))
        
        for i in np.arange(0,nX,1):
            for j in np.arange(0,nY,1):
                idxMaxBin=np.argmax(abvMat[i,j,:])
                idxMinBin=np.mod(idxMaxBin-deltaPhi[i,j],8)
                abvMaxIdxMat[i,j]=idxMaxBin
                abvMinIdxMat[i,j]=idxMinBin
                abvMaxMat[i,j]=100*abvMat[i,j,idxMaxBin]
                abvMinMat[i,j]=100*abvMat[i,j,idxMinBin] 

    for i in np.arange(0,nX,1):
        for j in np.arange(0,nY,1):
            if abvMinMat[i,j]==0:
                compMat[i,j]=-1;
            else:
                compMat[i,j]=100*(abvMaxMat[i,j]-abvMinMat[i,j])/abvMinMat[i,j]/pp

    return compMat,abvMaxMat,abvMinMat,abvMaxIdxMat,abvMinIdxMat
