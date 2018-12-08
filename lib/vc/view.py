#Data plotting functions
import matplotlib.pyplot as plt
import numpy as np

def hello():
	print('hello from' + __name__ +' in vc view module')

def plotOnePlane(plane,minVal=0,maxVal=0,cmap=''):
    plt.figure(dpi=300)
    plt.axes().set_aspect('equal','datalim')
    plt.set_cmap(plt.gray())
    dims=np.shape(plane)
    x=np.arange(0,dims[0],1)
    y=np.arange(0,dims[1],1)
    if minVal!=maxVal:
        if cmap=='':
            plt.pcolormesh(x,y,plane,vmin=minVal,vmax=maxVal)
        else:
            plt.pcolormesh(x,y,plane,vmin=minVal,vmax=maxVal,cmap=cmap)
    else:
        plt.pcolormesh(x,y,plane,cmap=cmap)

    plt.colorbar()

#-------------
## abv calculation helper functions
#-------------
import pylab
def plotCompPanel(compMat,abvMaxMat,abvMinMat,abvMaxIdxMat,abvMinIdxMat,cbfMap,nPhases=100,abvMinVal=0.0,abvMaxVal=2,compMaxVal=1,compMinVal=0):
    cmapComp='viridis'
    cmapDisc = pylab.cm.get_cmap('PiYG', nPhases)
    cbfMinVal=0.2;cbfMaxVal=1.0
    print('plotCompPanel got cbfMat size', np.shape(cbfMap))
    h=plt.figure(figsize=(20,20))
    h.subplots_adjust(hspace=0.2,wspace=0.3)
    plt.subplot(342)
    plt.title('$max(abv(\phi_c^s))$',fontsize=20)
    plt.imshow(np.log(abvMaxMat),vmin=-11,vmax=-2,cmap='viridis',interpolation='none');
    cbar = plt.colorbar();cbar.set_label('(log(mL/mL))')

    plt.subplot(343)
    plt.title('$min(abv(\phi_c^s))$',fontsize=20)
    plt.imshow(np.log(abvMinMat),vmin=-11,vmax=-2,cmap='viridis',interpolation='none');
    cbar = plt.colorbar();cbar.set_label('(log(mL/mL))')
    
    plt.subplot(344)
    plt.title('$max(abv(\phi_c^s))-min(abv(\phi_c^s))$',fontsize=20)
    diffAbvMaxVal=abvMaxVal-abvMinVal
    plt.imshow(np.log(abvMaxMat-abvMinMat),cmap='viridis',interpolation='none');plt.colorbar()

    plt.subplot(345)
    plt.title('compliance\n (%/mmHg)',fontsize=20)
    #compMat[compMat>4]=0
    plt.imshow(compMat,cmap=cmapComp,vmin=0,vmax=compMaxVal,interpolation='none');plt.colorbar()
    #plt.imshow(compMat,cmap='viridis',vmin=compMinVal,vmax=compMaxVal,interpolation='none');plt.colorbar()
    
    #plt.subplot(346)
    #plt.title('$\Delta abv/(max(abv)+min(abv))$',fontsize=20)
    #compMat2[compMat2>3]=0
    #plt.imshow(compMat2,cmap='viridis',vmin=0,vmax=1,interpolation='none');plt.colorbar()

    plt.subplot(347)
    plt.title('$argmax(abv(\phi_c^s))$',fontsize=20)
    plt.imshow(abvMaxIdxMat,cmap=cmapDisc,vmin=0,vmax=nPhases,interpolation='none');plt.colorbar()

    plt.subplot(348)
    plt.title('$argmin(abv(\phi_c^s))$',fontsize=20)
    plt.imshow(abvMinIdxMat,cmap=cmapDisc,vmin=0,vmax=nPhases,interpolation='none');plt.colorbar()
    
    plt.subplot(349)
    plt.title('$mod(|\phi_{c,sys}^s-\phi_{c,dia}^s|)$',fontsize=20)
    deltaPhaseMax=np.floor(nPhases/2)
    plt.imshow(np.mod(abs(abvMaxIdxMat-abvMinIdxMat),deltaPhaseMax),cmap=cmapDisc,vmin=0,vmax=deltaPhaseMax,interpolation='none');plt.colorbar()

    plt.subplot(341)
    plt.title('$<S_{ctr}-S_{tag}>$',fontsize=20)
    plt.imshow(cbfMap,cmap='viridis',vmin=cbfMinVal,vmax=cbfMaxVal,interpolation='none');plt.colorbar()


#--------
#Masking
#--------

def makeMask(fitMat,uThr=350,lThr=100,iThr=11,pltFlg=0):
    tdMask=np.zeros((np.shape(fitMat)[0],np.shape(fitMat)[1]))
    tdMask[np.where( fitMat[:,:,iThr]<uThr ) ] =1
    tdMask[np.where (fitMat[:,:,iThr]<lThr) ]=0
    if pltFlg:
        plt.figure()
        plt.imshow(tdMask,interpolation='none');plt.title('threshold mask range: '+str(lThr)+'-'+str(uThr))
    return tdMask

def maskAllBins(fitVec,iThr,uThr,lThr=0,pltFlg=0):
    # Create a threshold mask for each bin of a parameter image
    #input:
    #  fitVec [nVox nBins nPars]
    #  iThr index to threshold on, corresponding to nPar dimension
    #  uThr highest accepted value
    #  lThr lowest accepted value
    #output:
    #  tdMask [nVox nBins]
    #  all pars and bins are masked where the parameter excedes the range
    (nVox,nBins,nPars)=np.shape(fitVec)
    tdMask=np.zeros((nVox,nBins))
    tdMask[np.where( fitVec[:,:,iThr]<uThr ) ] =1
    tdMask[np.where (fitVec[:,:,iThr]<lThr) ]=0
    tdMask=np.prod(np.reshape(tdMask,(nVox,nBins)),1)
    fitVecMasked=fitVec*np.tile(tdMask[:,np.newaxis,np.newaxis],(1,nBins,nPars))

    return fitVecMasked

#-------
#calcAbv
#------

def calcAbvMatBF50(fitMat,alpha=1,M0=1,nPhases=100):
    #example: abvMat=calcAbvMatBF50(fitMat,alpha=1,M0=15,nPhases=8)
    nX=np.shape(fitMat)[0];nY=np.shape(fitMat)[1]
    #nTIs=np.shape(tiVec)[0]
    nVox=nX*nY
    fitMatVec=np.reshape(fitMat,(nVox,np.shape(fitMat)[2]))
    abvVec=np.zeros((nVox,nPhases))
    for iVox in np.arange(0,nVox,1):
        if (fitMatVec[iVox,11]>0 and fitMatVec[iVox,12]>0):
            abvVec[iVox,:]=fitfuncF50(fitMatVec[iVox,:],np.arange(0,1,1/nPhases))/alpha/M0/2
    abvMat=np.reshape(abvVec,(nX,nY,nPhases))
    return abvMat

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

def calcComp(abvMat,pp=1,tdMat='',method=0):
    #input:
    #  abvMat [nX nY nBins]
    nX=np.shape(abvMat)[0]
    nY=np.shape(abvMat)[1]
    compMat=np.zeros((nX,nY))
    abvMaxMat=np.zeros((nX,nY))
    abvMinMat=np.zeros((nX,nY))
    abvMaxIdxMat=np.zeros((nX,nY))
    abvMaxIdxMat=np.zeros((nX,nY))

    if method==0:   #ID phases based on abv max and min
        abvMaxMat=100*np.max(abvMat,2)
        abvMinMat=100*np.min(abvMat,2)
        abvMaxIdxMat=np.argmax(abvMat,2)
        abvMinIdxMat=np.argmin(abvMat,2)

    if method==1:   #ID phases based on the td max and min
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

    #divTemp=abvMinMat
    #divTemp2=abvMaxMat+abvMinMat
    
    #print(np.shape(divTemp))
    #Handle div by zero by looking at every value before the divide
    # set the div by zero compliance values to -1.0
    for i in np.arange(0,nX,1):
        for j in np.arange(0,nY,1):
            if abvMinMat[i,j]==0:
                compMat[i,j]=-1;
            else:
                compMat[i,j]=100*(abvMaxMat[i,j]-abvMinMat[i,j])/abvMinMat[i,j]/pp

    return compMat,abvMaxMat,abvMinMat,abvMaxIdxMat,abvMinIdxMat


#------
#plot fit mat
#-----
cbfMap=''
def plot2DFitMat(fitMat,cbfMap=cbfMap,x1f=10,x2f=54,y1f=10,y2f=54,zf=3):
    TD_PARNUM=11
    SIGMA_PARNUM=12
    P0_PARNUM=0;P1_PARNUM=1;P2_PARNUM=3;P3_PARNUM=4;P4_PARNUM=5;P5_PARNUM=6

    print(np.shape(cbfMap))
    if FIGURES_ONSCREEN:
        h=plt.figure(figsize=[10,10])
        h.subplots_adjust(hspace=0.2,wspace=0.2)
        plt.subplot(332)
        plt.imshow(np.squeeze(fitMat[:,:,TD_PARNUM]),interpolation='none',cmap='viridis');plt.xticks([]);plt.yticks([])
        plt.title('td')
        plt.colorbar()
        plt.subplot(333)
        plt.imshow(np.squeeze(fitMat[:,:,SIGMA_PARNUM]),interpolation='none',cmap='viridis')
        plt.title('$\sigma$')
        plt.colorbar()
        plt.subplot(334)
        plt.imshow(np.squeeze(fitMat[:,:,P0_PARNUM]),interpolation='none',cmap='viridis')
        plt.title('P0')
        plt.colorbar()
        plt.subplot(331); plt.title('<cbf>')
        plt.imshow(np.squeeze(cbfMap[x1f:x2f,y1f:y2f,zf-1:zf]),interpolation='none',cmap='viridis')
        plt.colorbar()
        plt.subplot(335)
        plt.imshow(np.squeeze(fitMat[:,:,P1_PARNUM]),interpolation='none',cmap='viridis')
        plt.title('P1')
        plt.colorbar()
        plt.subplot(336)
        plt.imshow(np.squeeze(fitMat[:,:,P2_PARNUM]),interpolation='none',cmap='viridis')
        plt.title('P2')
        plt.colorbar()
        plt.subplot(337)
        plt.imshow(np.squeeze(fitMat[:,:,P3_PARNUM]),interpolation='none',cmap='viridis')
        plt.title('P3')
        plt.colorbar()
        plt.subplot(338)
        plt.imshow(np.squeeze(fitMat[:,:,P4_PARNUM]),interpolation='none',cmap='viridis')
        plt.title('P4')
        plt.colorbar()
        plt.subplot(339)
        plt.imshow(np.squeeze(fitMat[:,:,P5_PARNUM]),interpolation='none',cmap='viridis')
        plt.title('P5')
        plt.colorbar()

        plt.ion();plt.show()

#--------
#view fit data
#-------
def plot2DFitMat(fitMat):
    TD_PARNUM=11
    SIGMA_PARNUM=12
    P0_PARNUM=0;P1_PARNUM=1;P2_PARNUM=3;P3_PARNUM=4;P4_PARNUM=5;P5_PARNUM=6

    h=plt.figure(figsize=[10,10])
    h.subplots_adjust(hspace=0.2,wspace=0.2)
    plt.subplot(332)
    plt.imshow(np.squeeze(fitMat[:,:,TD_PARNUM]),interpolation='none',cmap='viridis')
    plt.title('td')
    plt.colorbar()
    plt.subplot(333)
    plt.imshow(np.squeeze(fitMat[:,:,SIGMA_PARNUM]),interpolation='none',cmap='viridis')
    plt.title('$\sigma$')
    plt.colorbar()
    plt.subplot(334)
    plt.imshow(np.squeeze(fitMat[:,:,P0_PARNUM]),interpolation='none',cmap='viridis')
    plt.title('P0')
    plt.colorbar()
    #plt.subplot(331); plt.title('<cbf>')
    #plt.imshow(np.squeeze(cbfMap[x1:x2,y1:y2,(z-1):z]),interpolation='none',cmap='viridis')
    #plt.colorbar()
    plt.subplot(335)
    plt.imshow(np.squeeze(fitMat[:,:,P1_PARNUM]),interpolation='none',cmap='viridis')
    plt.title('P1')
    plt.colorbar()
    plt.subplot(336)
    plt.imshow(np.squeeze(fitMat[:,:,P2_PARNUM]),interpolation='none',cmap='viridis')
    plt.title('P2')
    plt.colorbar()
    plt.subplot(337)
    plt.imshow(np.squeeze(fitMat[:,:,P3_PARNUM]),interpolation='none',cmap='viridis')
    plt.title('P3')
    plt.colorbar()
    plt.subplot(338)
    plt.imshow(np.squeeze(fitMat[:,:,P4_PARNUM]),interpolation='none',cmap='viridis')
    plt.title('P4')
    plt.colorbar()
    plt.subplot(339)
    plt.imshow(np.squeeze(fitMat[:,:,P5_PARNUM]),interpolation='none',cmap='viridis')
    plt.title('P5')
    plt.colorbar()
    

    
def plot2DFit3Par(fitMat,fitMatType=0):
    #input:
    #  fitMat [nX nY nPars]
    #output:
    #  plots of the ABV, TD and SIGMA
    
    if fitMatType==0:
        ABV_PARNUM=0;
        TD_PARNUM=1
        SIGMA_PARNUM=2
    else:
        print('ERROR: plot2DFit3Par fit mat type unsupported. EXITING.')
        return 0

        
    h=plt.figure(figsize=[10,10])
    h.subplots_adjust(hspace=0.2,wspace=0.2)
    plt.subplot(332)
    plt.imshow(np.squeeze(fitMat[:,:,TD_PARNUM]),interpolation='none',cmap='viridis')
    plt.title('td')
    plt.colorbar()
    plt.subplot(333)
    plt.imshow(np.squeeze(fitMat[:,:,SIGMA_PARNUM]),interpolation='none',cmap='viridis')
    plt.title('$\sigma$')
    plt.colorbar()
    plt.subplot(331)
    plt.imshow(np.squeeze(fitMat[:,:,ABV_PARNUM]),interpolation='none',cmap='viridis')
    plt.title('ABV')
    plt.colorbar()
    return 1

def sliceDataMontage(dataMat,fs=(40,10)):
    (nX,nY,nBins,nTIs)=np.shape(dataMat)
    dataMat=np.reshape(np.transpose(dataMat,(0,2,1,3)),(nX,nY*nBins,nTIs))
    dataMat=np.reshape( np.transpose(dataMat,(2,0,1) ),(nX*nTIs,nY*nBins))
    if FIGURES_ONSCREEN:
        plt.figure(figsize=fs)
        plt.imshow(dataMat)
        plt.xlabel('bins',fontsize=15);plt.ylabel('TIs',fontsize=15)
        plt.xticks([]);plt.yticks([]);plt.title('single slice data')
        plt.colorbar();
        plt.ion();plt.show()
