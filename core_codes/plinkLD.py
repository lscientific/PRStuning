'''
Calculate correlation coefficient based on PLINK binary file
Created on 2016-6-7

Add hdf5 and shrinkage estimator feature on 2020-8-3

@author: Wei Jiang (w.jiang@yale.edu)
'''

import sys,time,re,os
import numpy as np
import logger
import multiprocessing
from sklearn.covariance import ledoit_wolf
import pandas as pd
import plink2SnpReader
# from ldsc import ldscore

def ldscore(R, N=None):
    R2 = np.square(R)
    if N is not None:
        R2 = R2-(1-R2)/(N-2)
    #	diag(R2) = 1
    ldsc = np.ravel(np.sum(R2, axis=1))
    return(ldsc)

def cov2corr(A):
    diagA = np.diag(A)
    sqrtDiagA=np.sqrt(diagA)
    return A/np.outer(sqrtDiagA, sqrtDiagA)

def calBlockCorr(blockGenotype, method):
    # Standardize
    indNum, blockSNPnum = blockGenotype.shape
    af = np.nanmean(blockGenotype, axis=0)/2.
    expectation = np.outer(np.ones(indNum), 2.*af)
    scale = np.nanstd(blockGenotype, axis=0)#np.sqrt(2*af*(1-af))
    scaleMat = np.outer(np.ones(indNum), scale)
    blockGenotype_norm = (blockGenotype-expectation)/scaleMat
    mask = (~np.isnan(blockGenotype_norm)).astype(int)
    blockGenotype_norm[mask==0] = 0
        
    if method == 'Pearson':
        # blockLD = np.around(np.ma.corrcoef(np.ma.masked_invalid(blockGenotype)), decimals=3)
        #blockLD = corr.corrmatrix(blockGenotype)

        #blockLD, rho_shrinkage = corr.covEst_shrinkage_b( X = np.transpose(blockGenotype) )
        #blockLD = corr.cov2corr(blockLD)

        #blockGenotypeFrame = DataFrame(transpose(blockGenotype))
        #blockLD = blockGenotypeFrame.corr()
        #blockLD = blockLD.fillna(0)
        # nanIdx = np.isnan(blockLD)
        # if np.sum(nanIdx)>0:
        if np.sum(mask)<(indNum*blockSNPnum):
            effN = np.dot(mask.T, mask)
            blockLD = np.around(np.dot(blockGenotype_norm.T, blockGenotype_norm)/effN, decimals=3)
        else:
            blockLD = np.around(np.dot(blockGenotype_norm.T, blockGenotype_norm)/indNum, decimals=3)
    elif method == 'LW':
        blockLD = np.around(cov2corr(ledoit_wolf(blockGenotype_norm, assume_centered=True)[0]), decimals=3)
        
    ldscVal = ldscore(blockLD, indNum)
    return blockLD, af, ldscVal

def main_with_args(args):
    startTime=time.time()
    print('===============plinkLD================')
    helpText = '''
=================HELP=================
--bfile: Binary data file (Default: test)
--bed: Binary data file (Genotypes; Default: test.bed)
--bim: Binary data file (SNP info; Default: test.bim)
--fam: Binary data file (Individual info; Default: test.fam)
--block: Block file (By default, all SNPs are in one block)
--snplist: SNP list file (By default, all SNP pairs are calculated)
--output: output filename (Default: LD.h5)
--method: Correlation estimation method, including: Pearson, LW (Default:Pearson)
--thread: Thread number for calculation (Default: Total CPU number)
--compress: compression level for output (Default: 9) 
--log: log file (Default: plinkLD.log)
--help: Help
======================================'''
        
    arg={'--bfile':None,'--bed':'test.bed','--bim':'test.bim','--fam':'test.fam',\
        '--block':None,'--snplist':None, '--output':'LD.h5','--log': 'plinkLD.log',\
        '--thread':multiprocessing.cpu_count(), '--method':'Pearson', '--compress': 9}
    
    if len(sys.argv)%2==1:
        for i in range(int(len(sys.argv)/2)):
            arg[sys.argv[2*i+1]]=sys.argv[2*i+2]
    else:
        if sys.argv[1]!='--help': print('The number of arguments is wrong!')
        print(helpText)
        exit()

    sys.stdout = logger.logger(arg['--log'])

    cpuNum = multiprocessing.cpu_count()
    try:
        threadNum = round(float(arg['--thread']))
    except ValueError:
        print('Warning: --thread must be a numeric value')
        threadNum = cpuNum

    if threadNum<=0: 
        threadNum=cpuNum
    print(cpuNum, 'CPUs detected, using', threadNum, 'thread...' )
  
    print(arg['--method'],'method for LD calculation')
    
    try:
        complevel = round(float(arg['--compress']))
    except ValueError:
        print('Warning: --compress must be a numeric value')
        complevel = 0

    if arg['--bfile']!=None:
        arg['--bed']=arg['--bfile']+'.bed'
        arg['--bim']=arg['--bfile']+'.bim'
        arg['--fam']=arg['--bfile']+'.fam'

    print('Start loading SNP information...')
    try:
        #with open(arg['--bim'],'r') as f:
        #snpInfo = [i.strip() for i in f.readlines() if len(i.strip())!=0]
        snpInfo = pd.read_table(arg['--bim'], sep='\s+', names=['CHR','SNP','GD','BP','A1','A2'], dtype={'SNP':str,'CHR':str,'A1':str,'A2':str})
    except:
        print("Could not read SNP Info file:", arg['--bim'])
        exit()
  
    snpNum = len(snpInfo)
    print(snpNum,'SNPs.')
    #Formatting CHR
    pattern = re.compile(r'^(?i:chr?)?(\w+)')
    vmatch = np.vectorize(lambda x: pattern.match(x).group(1))
    snpInfo['CHR'] = np.char.lower(vmatch(snpInfo['CHR'].astype('str')))
    #Change A1 and A2 to lower case for easily matching
    snpInfo['SNP'] = snpInfo['SNP'].str.lower()
    snpInfo['A1'] = snpInfo['A1'].str.lower()
    snpInfo['A2'] = snpInfo['A2'].str.lower()
        
    ch = snpInfo['CHR'].tolist()#[0]*snpNum  #Chromosome
    snpID = snpInfo['SNP'].tolist()#['']*snpNum #SNP ID
    #gdist = snpInfo['GD'].tolist()#[0.]*snpNum #Genetic distance
    bp = snpInfo['BP'].tolist()#[0]*snpNum  #Base position
    #A1 = snpInfo['A1'].tolist()#['']*snpNum #Allele 1
    #A2 = snpInfo['A2'].tolist()#['']*snpNum #Allele 2
    #for i in range(snpNum):
        #print(str(i))
        #match = re.match(r'^(\d+)\s+([\w\:\;-]+)\s+-?[\d\.]+\s+(\d+)', snpInfo[i])
        #ch[i] = int(match.group(1))
        #snpID[i] = match.group(2)
        #bp[i] = int(match.group(3))
        #######
        #tmpSNPinfo = snpInfo[i].split()
        #ch[i] = int(tmpSNPinfo[0])
        #snpID[i] = tmpSNPinfo[1]
        #gdist[i] = float(tmpSNPinfo[2])
        #bp[i] = int(tmpSNPinfo[3])
        #A1[i] = tmpSNPinfo[4]
        #A2[i] = tmpSNPinfo[5]

    chSet = list(set(ch))

    print('Start loading individual information...')
    try:
        with open(arg['--fam'],'r') as f:
            indInfo = [i.strip() for i in f.readlines() if len(i.strip())!=0]  
    except IOError:
        print("Could not read Individual Info file:", arg['--fam'])
        exit()
  
    indNum = len(indInfo)
    print(indNum,'individuals.')

    print('Start loading block file...')
    if arg['--block']!=None:
        try:
            with open(arg['--block'],'r') as f:
                block = [i.strip() for i in f.readlines() if len(i.strip())!=0]           
        except IOError:
            print("Could not read block file:", arg['--block'])
            exit()
    
        blockNum = len(block)-1  #heading
        print(blockNum, 'Blocks.')
        blockCH = [0]*blockNum
        blockStart = [0]*blockNum
        blockStop = [0]*blockNum
        for i in range(0,blockNum):
            match = re.match(r'^\s*chr(\w+)\s+(\d+)\s+(\d+)', block[i+1])
            blockCH[i] = match.group(1)
            blockStart[i] = int(match.group(2))
            blockStop[i] = int(match.group(3))
            if blockStart[i]>blockStop[i]:
                print('Block',i,': Start position is larger than stop position!', blockStart[i],blockStop[i])
                exit()
            
    else:
        blockNum = len(chSet)
        blockCH = ['']*blockNum
        blockStart = [0]*blockNum
        blockStop = [0]*blockNum
        print(blockNum, 'Blocks (Each chromosome is an unique block).')
    
        for i in range(0,blockNum):
            blockCH[i] = chSet[i]
            tmpBP = [bp[j] for j in range(0, snpNum) if ch[j]==blockCH[i]]
            blockStart[i] = min(tmpBP)
            blockStop[i] = max(tmpBP)

    if arg['--snplist']!=None:
        try:
            with open(arg['--snplist'],'r') as f:
                snplist = [i.strip() for i in f.readlines() if len(i.strip())!=0]
        except IOError:
            print("Could not read snplist file:", arg['--snplist'])
            exit()
        isHead = False
        snpDict = {}
        for i in range(len(snplist)):
            #match = re.match(r'^\s*(\w+)\s*', snplist[i])
            #snpDict[match.group(1)] = i
            tmpID = snplist[i].split()[0]
            if i==0:
                if tmpID in ('SNP','snp','SNPID','snpID','ID','id','rsID','rsid'):
                    isHead=True
                    continue
            if isHead: snpDict[tmpID] = i-1
            else: snpDict[tmpID] = i
  
    filename = arg['--output']
    dirname = os.path.dirname(filename)
    if not (dirname=='' or os.path.exists(dirname)): os.makedirs(dirname)
    try:
        store = pd.HDFStore(filename, 'w', complevel=complevel)
    except:
        print('Unable to create file', filename)
        exit()

    print('Reading BED file...')
    genotype = plink2SnpReader.getSnpReader(bed=arg['--bed'], bim=arg['--bim'], fam=arg['--fam'], thread= threadNum)

    print('Start LD calculation...')

    totalSNPnum = 0

    pool = multiprocessing.Pool(processes = threadNum) 
    effectiveI = []
    tmpResults = []
    snpInfoList = []
    for i in range(blockNum):
        #SNP index in the block
        if i==0 or not blockCH[i] in blockCH[0:i]:
            if arg['--snplist']==None:
                idx = [j for j in range(0, snpNum) if ch[j]==blockCH[i] \
                        and blockStart[i]<= bp[j] and bp[j]<=blockStop[i]]
            else: 
                idx = [j for j in range(0, snpNum) if ch[j]==blockCH[i] \
                        and blockStart[i]<= bp[j] and bp[j]<=blockStop[i] and snpDict.get(snpID[j])!=None]
        else:
            if arg['--snplist']==None:
                idx = [j for j in range(0, snpNum) if ch[j]==blockCH[i] \
                        and blockStart[i]< bp[j] and bp[j]<=blockStop[i]]
            else: 
                idx = [j for j in range(0, snpNum) if ch[j]==blockCH[i] \
                        and blockStart[i]< bp[j] and bp[j]<=blockStop[i] and snpDict.get(snpID[j])!=None]
        
        if len(idx)==0: continue
        if len(idx)==1 : 
            print('Block '+ str(i)+ ' : Only '+str(len(idx))+ ' SNP in the block [Ignored]') #\r
            continue
        print('Block '+ str(i)+ ' : '+str(len(idx))+ ' SNPs [Calculating LD ...]') #\r
        totalSNPnum += len(idx)
        blockGenotype = genotype[:,idx].read(dtype='float32').val
        blockSNPinfo = snpInfo.iloc[idx]
        #blockSNPinfo = pd.DataFrame({'ch': [ch[j] for j in idx], 
            #'id':[snpID[j] for j in idx], 
            #'gd':[gdist[j] for j in idx], 
            #'bp':[bp[j] for j in idx],
            #'A1':[A1[j] for j in idx],
            #'A2':[A2[j] for j in idx]})

        #Mantain total process number to processes，Automatically add a process when a process is finished in the pool
        effectiveI.append(i)
        snpInfoList.append(blockSNPinfo)
        tmpResults.append(pool.apply_async(calBlockCorr, args=(blockGenotype, arg['--method'])))       
  
    for k in range(len(effectiveI)):
        i = effectiveI[k]
        blockSNPinfo = snpInfoList[k]
        blockLD, af, ldscVal = tmpResults[k].get()
        #print('Block '+ str(i)+ ' : '+str(blockLD.shape[0])+ ' SNPs [Finished]')
        print('Block '+ str(i)+ ' : '+str(blockLD.shape[0])+ ' SNPs [Finished]')
        blockSNPinfo.insert(6,'F',af)
        blockSNPinfo.insert(7,'LDSC', ldscVal)
        blockLD = pd.DataFrame(data=blockLD)
        ###Output Large file#####
        store.put('SNPINFO'+str(i),value=blockSNPinfo, complevel=complevel, format='fixed')
        store.put('LD'+str(i), value=blockLD, complevel=complevel, format='fixed')

    pool.close()
    #Call close function before join, otherwise error will raise. No process will be added to pool after close.
    #Join function: waiting the complete of subprocesses
    pool.join()

    store.put('BID',value=pd.Series(effectiveI), complevel=complevel)
    print('Total effective SNPs:',totalSNPnum)
    store.put('TOTAL',value=pd.Series(totalSNPnum), complevel=complevel)
    store.put('INDNUM', value=pd.Series(indNum), complevel=complevel)
    store.close()
    totalTime=time.time()-startTime
    print('======================================')
    print('Finish Calculation, Total time=',float(totalTime)/60, 'min.')

if __name__ == '__main__':
    main_with_args(sys.argv)
