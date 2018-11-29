def makeSaveDir(subDir,id_dir,version,nTIs,mMethod=0):
    #added mMethod in v2.3.3
    return subDir+'/'+id_dir+'/v'+version+'/'+str(nTIs)+'TIs/'+'m'+str(mMethod)+'/'
