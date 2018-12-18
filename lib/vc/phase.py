FIGURES_ONSCREEN=0
PLOT_BLOCKING=0

import numpy as np

#Tools for finding cardiac phase
def getExtremePhiMap(data,tiVec,phiCSVec,method=0,verbosity=1):
    #create a map from the extreme values of the phase across bins
    # m method 0 for min, 1 for max
    # data: [nX nY nBins]
    print('getExtremePhiMap: WIP')
    (nX,nY,nBins)=np.shape(data)
    phiMap=np.zeros((nX,nY))
    if method==0:
        phiMap=np.argmin(data,2)
        ttlStr='$argmax( f(\phi_c^s) )$'
    if method==1:
        phiMap=np.argmax(data,2)
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
