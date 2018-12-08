#Functions for modelling the ASL data
import matplotlib.pyplot as plt
import numpy as np
import math as math

FIGURES_ONSCREEN=0

def picoreComplianceModelDeltaM_t(to,td,tau,abv,sigma,T1b=1660,alpha=0.95,M0=1):
    # td: transit delay (msec)
    # abv: blood volume (mL)
    # T1b: T1 of blood (msec)
    # M0: tissue magnetization scaling
    # tau: temporal bolus
    # to: observation time
    
    #Note: This is a step-wise logic, so hard to calc the timecourse
    # with an array broadcast. So, I'm doing this calculation pointwise
    
    if to<td:
        dM=0
    elif to<tau+td and to>=td:
        #dM=2*M0*alpha*f*(to-td)*np.exp(-to/T1b)/1000
        dM=2*M0*alpha*abv*disp(to,sigma)*np.exp(-to/T1b)/1000

    elif to>=tau+td:
        dM=0
    
    return dM

def picoreComplianceModelTimecourse(tstart,tstop,tstep,td,tau,abv,sigma,T1b=1660,alpha=0.95,M0=1,plotFlag=0):
    t=np.arange(tstart,tstop,tstep)
    dM=np.zeros(np.shape(t)[0])
    
    for i in np.arange(np.shape(t)[0]):
        dM[i]=picoreModelDeltaM_t(t[i],td,tau,f,T1b,alpha,M0)
    if plotFlag>0:
        if plotFlag==2:
            plt.plot(t,dM)
        else:
            plt.plot(t,dM,'o')
        plt.title('dM versus observation time -- model')
        plt.grid
    #print("observation times are", t)
    
    return dM

def gaussian(t,td,sigma,testFlag=0):
    gauss=( 1.0/sigma/math.sqrt(2.0*math.pi) ) * np.exp( -( (td-t)/sigma )**2.0/2.0 )
    
    return gauss

def notGaussian(t,td,sigma,testFlag=0):
    gauss=( 1.0/sigma/math.sqrt(2.0*math.pi) ) * np.exp( -( (td-t) )**2.0/sigma )
    
    return gauss

def dispKernel(bolusDuration,sigma,transitDelay,testFlag=0,nPts=4000):
    t=np.arange(nPts)*1.0
    gauss=gaussian(t,transitDelay,sigma)
    w=np.zeros(nPts)
    w[t<bolusDuration]=1
    #plt.plot(w)
    kern=np.conv(w,gauss,'full')
    
def getBolusModel(nPts=4000,bolusDuration=700,transitDelay=400,sigma=100,T1b=1660,plotFlag=0,norm=0):
    #plot a gaussian with 
    # nPts: num points (msec)
    # sigma: dispersion width (msec)
    # bolusDuration: bolus duration (msec)
    # transitDelay: transit delay
    
    t=np.arange(nPts)
    
    #get dispersion kernel
    dispKernel=gaussian(t,transitDelay,sigma=sigma)
    notDispKernel=notGaussian(t,transitDelay,sigma=sigma)
    
    #make plug flow model
    plug=np.zeros(nPts)
    plug[t<bolusDuration]=1
    
    #convolve to get the dispersed bolus
    bolus=np.convolve(plug,dispKernel,'full')
    notBolus=np.convolve(plug,notDispKernel,'full')

    #add relaxation
    t1Atten=np.exp(-np.arange(np.shape(bolus)[0])/T1b)
    bolus=bolus*t1Atten
    notBolus=notBolus*t1Atten
    
    #scale the bolus
    if norm==1:
        bolus=bolus/(np.max(bolus))
    
    if plotFlag==1 and FIGURES_ONSCREEN:
        #plt.plot(dispKernel/np.max(dispKernel),label="kernel/max kernel")
        #plt.plot(plug/np.max(plug),'o',label="plug/max plug")
        #plt.plot(bolus/(np.max(bolus)),'g.',label="bolus model/max bolus")
        plt.plot(dispKernel,label="kernel")
        plt.plot(plug,'o',label="plug")
        plt.plot(bolus,'g.',label="bolus model")
        #plt.plot(notDispKernel/np.max(notDispKernel),'r',label="NOT kernel")
        #plt.plot(notBolus/(np.max(notBolus)),'r*',label="NOT bolus model")
     
        plt.title("transit delay="+str(transitDelay)+" msec, dispersion="+str(sigma)+" msec, bolusDuration=" +str(bolusDuration)+" msec")
        plt.grid();plt.legend();plt.ion();plt.show()
        print("integral under gaussian" + str(np.sum(dispKernel)))
        
    return bolus

