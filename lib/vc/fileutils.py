import os

def makeSaveDir(subDir,id_dir,version,nTIs,mMethod=0):
    #added mMethod in v2.3.3
    saveDir=subDir+'/'+id_dir+'/v'+version+'/'+str(nTIs)+'TIs/'+'m'+str(mMethod)+'/'
    os.makedirs(saveDir,exist_ok=True)
    return saveDir
