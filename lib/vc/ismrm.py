import multiprocessing

DO_ISMRM2019_NTIS_COMPARISON=1
iSlice=0
saveFn=''
if DO_ISMRM2019_NTIS_COMPARISON:
    numProcessors=multiprocessing.cpu_count()
    nWorkers=14*3
    if nWorkers>numProcessors:
        nWorkers=numProcessors-1

    print('Available cores: '+str(numProcessors))
    print('Cores needed:' + str(nWorkers))

    pool = multiprocessing.Pool( nWorkers )

    #initialize the task list for each core
    tasks=[]

    # set parameters that apply to all subjects
    verbosity=0;dryRun=0
    alpha=1

    id_dir='119_180612'
    subDir='data/PupAlz_119'
    M0=39;
    pp=50;
    alpha=1;

    dryRun=0

    #nTIsToFit=7
    #saveDir=makeSaveDir(subDir,id_dir,notebookVersion,nTIsToFit,mMethod=0)
    #makeTaskAllSlices2p0test(tasks,id_dir,subDir,nTIsToFit,fitMask,saveDir,M0=M0,alpha=alpha,verbosity=0,dryRun=dryRun,nBins=8,mMethod=0)

    tStart = time.time()
    # Run tasks
    if 0:
        for t in tasks:
            print('<<<<<<<<<will run this task>>>>>>>>>>')
            #print(t)
            results=pool.apply_async( fitWithinMaskPar2p0_test, t)
    #results=pool.apply_async( fitWithinMaskPar2p0_test, t)
    #pool.close()
    #pool.join()

    tEnd = time.time()
    tElapsed=tEnd-tStart
    print('********Parallel fitting jobs required '+str(tElapsed)+'seconds. *********')
    
    pool = multiprocessing.Pool( nWorkers )

    #initialize the task list for each core
    tasks=[]
    
    #set up the 7 point fits
    nTIsToFit=7
    saveDir=makeSaveDir(subDir,id_dir,notebookVersion,nTIsToFit,mMethod=0)
    makeTaskAllSlices2p0test(tasks,id_dir,subDir,nTIsToFit,fitMask,saveDir,M0=M0,alpha=alpha,verbosity=0,dryRun=dryRun,nBins=8,mMethod=0)

    #set up the 5 point fits
    nTIsToFit=5
    saveDir=makeSaveDir(subDir,id_dir,notebookVersion,nTIsToFit,mMethod=0)
    makeTaskAllSlices2p0test(tasks,id_dir,subDir,nTIsToFit,fitMask,saveDir,M0=M0,alpha=alpha,verbosity=0,dryRun=dryRun,nBins=8,mMethod=0)

    #set up the 3 point fits
    nTIsToFit=3
    saveDir=makeSaveDir(subDir,id_dir,notebookVersion,nTIsToFit,mMethod=0)
    makeTaskAllSlices2p0test(tasks,id_dir,subDir,nTIsToFit,fitMask,saveDir,M0=M0,alpha=alpha,verbosity=0,dryRun=dryRun,nBins=8,mMethod=0)

    
    tStart = time.time()
    # Run tasks
    if 1:
        for t in tasks:
            print('<<<<<<<<<will run this task>>>>>>>>>>')
            #print(t)
            results=pool.apply_async( fitWithinMaskPar2p0_test, t)
    #print(t)
    #results=pool.apply_async( fitWithinMaskPar2p0_test, t)
    pool.close()
    pool.join()

    tEnd = time.time()
    tElapsed=tEnd-tStart
    print('********Parallel fitting jobs required '+str(tElapsed)+'seconds. *********')

    #fitWithinMaskPar2p0_test(1,id_dir,subDir,fitMask,saveDir,nTIsToFit,M0=M0,alpha=alpha,verbosity=0,dryRun=dryRun,nBins=8,mMethod=0)
    
    #fitWithinMaskPar2p0_test(0,id_dir,subDir,nTIsToFit,fitMask,saveDir,M0=M0,alpha=alpha,verbosity=0,dryRun=dryRun,nBins=8,mMethod=0)
    #fitWithinMaskPar2p0_test(0,id_dir,subDir,nTIsToFit,fitMask,saveDir,M0=M0,alpha=alpha,verbosity=0,dryRun=dryRun,nBins=8,mMethod=0)
