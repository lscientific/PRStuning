'''
Get PRS for individuals based on their genotypes
'''
import numpy as np
import pandas as pd
import PRSalign
from functools import reduce
import warnings
import psutil
import os
import strUtils
import re
import subprocess

def adjustWeight(weight, genoRefObj, weightRefObj, byID=True):
    startBk = 0
    endBk = len(weightRefObj['BID'])
    start = 0
    end = 0
    newWeightList = []
    for i in range(len(genoRefObj['BID'])):
        bid = genoRefObj['BID'][i]
        while startBk < endBk:
            if weightRefObj['BID'][startBk]!=bid:
                startBk += 1
                start += len(weightRefObj['SNPINFO'][startBk])
            else:
                end = start + len(weightRefObj['SNPINFO'][startBk])
                blockWeight = weight[start:end]
                tmpIdx = np.isin(PRSalign.getIndex(weightRefObj['SNPINFO'][startBk], byID), PRSalign.getIndex(genoRefObj['SNPINFO'][i], byID))  
                newWeightList.append(np.linalg.lstsq(genoRefObj['LD'][i], weightRefObj['LD'][startBk][tmpIdx, :].dot(blockWeight), rcond=None)[0])
                startBk += 1
                start = end
                break
    
    newWeight = reduce(lambda a, b: np.concatenate((a,b), axis=-1), newWeightList)
    return(newWeight)

def adjustWeightStore(weight, genoRefStore, weightRefStore, byID=True):
    startBk = 0
    endBk = len(weightRefStore.get('BID'))
    start = 0
    end = 0
    newWeightList = []
    bidListGeno = genoRefStore.get('BID').to_numpy()
    bidListWeight = weightRefStore.get('BID').to_numpy()
    for i in range(len(bidListGeno)):
        bidGeno = bidListGeno[i]
        bidWeight = bidListWeight[startBk]
        while startBk < endBk:
            if bidWeight!=bidGeno:
                startBk += 1
                start += len(weightRefStore.get('SNPINFO'+str(bidWeight)))
            else:
                snpInfoWeight = weightRefStore.get('SNPINFO'+str(bidWeight))
                end = start + len(snpInfoWeight)
                blockWeight = weight[start:end]
                tmpIdx = np.isin(PRSalign.getIndex(snpInfoWeight, byID), PRSalign.getIndex(genoRefStore.get('SNPINFO'+str(bidGeno)), byID))  
                newWeightList.append(np.linalg.lstsq(genoRefStore.get('LD'+str(bidGeno)).to_numpy(), weightRefStore.get('LD'+str(bidWeight)).to_numpy()[tmpIdx, :].dot(blockWeight), rcond=None)[0])
                startBk += 1
                start = end
                break
    
    newWeight = reduce(lambda a, b: np.concatenate((a,b), axis=-1), newWeightList)
    return(newWeight)

def getAlleleFreq(genoObj, mem=0):
    indNum = len(genoObj['INDINFO'])
    snpNum = len(genoObj['SNPINFO'])
    if mem <=0:
        availMem = psutil.virtual_memory().available
    else:
        availMem = mem*(1024**3)
    maxIndNum = availMem/(snpNum*4)
    if indNum<maxIndNum:
        batchSize = indNum
    else:
        baseNum = 10**np.floor(np.log10(maxIndNum)).astype(int)
        batchSize = np.floor(maxIndNum/baseNum).astype(int)*baseNum
    print('Calculating allele frequencies from genotype data ...')
    start = 0
    f = np.zeros(snpNum)
    actualIndNum = np.zeros(snpNum)
    while start<indNum:
        end = np.min((start+2*batchSize, indNum))
        print('Processing individuals',start,'-',end-1, end='\r', flush=True)
        tmpIndNum = end-start
        genotype = genoObj['GENOTYPE'][start:end,:].read(dtype='float32').val
        f += np.nansum(genotype, axis=0)
        actualIndNum += (tmpIndNum-np.sum(np.isnan(genotype), axis=0))
        start = end
    print('')
    f = f/(2.*actualIndNum)
    return(f)

def scoring(genoObj, weight, f=None, splitByChr=True, normalize=True, mem=0):
    scoreDict = {}
    indNum = len(genoObj['INDINFO'])
    snpNum = len(genoObj['SNPINFO'])
    if mem <=0:
        availMem = psutil.virtual_memory().available
    else:
        availMem = mem*(1024**3)
    
    # print('Available memory:', availMem/(1024**3),'G')
    if normalize: 
        maxIndNum = availMem/(snpNum*4*2)
    else:
        maxIndNum = availMem/(snpNum*4*1.5)
    if indNum<maxIndNum:
        batchSize = indNum
    else:
        baseNum = 10**np.floor(np.log10(maxIndNum)).astype(int)
        batchSize = np.floor(maxIndNum/baseNum).astype(int)*baseNum
    
    if (f is None):
        f = getAlleleFreq(genoObj, mem=mem)

    f = f.astype(np.float16)
    scoreDict['all'] = np.array([])
    if splitByChr:
        chSet = list(set(genoObj['SNPINFO'].loc['CHR']))
        boolIdx = {}
        for ch in chSet:
            boolIdx[ch] = (genoObj['SNPINFO'].loc['CHR']==ch)
            scoreDict[ch] = np.array([])
    
    start = 0
    while start<indNum:
        end = np.min((start+batchSize,indNum))
        tmpIndNum = end-start
        print('Calculating PRS for individuals',start,'-',end-1,end='\r',flush=True)
        genotype = genoObj['GENOTYPE'][start:end,:].read(dtype='float32').val
        genotype[:,genoObj['FLIPINFO']] = 2.-genotype[:,genoObj['FLIPINFO']]
        f[genoObj['FLIPINFO']] = 1.-f[genoObj['FLIPINFO']]
        expectation = np.outer(np.ones(tmpIndNum, dtype=np.float16), 2.*f)
        genotype[np.isnan(genotype)] = expectation[np.isnan(genotype)]
        if normalize:
            scaleMat = np.outer(np.ones(tmpIndNum, dtype=np.float16), np.sqrt(2*f*(1-f)))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                genotype = (genotype-expectation)/scaleMat
        scoreDict['all'] = np.append(scoreDict['all'], genotype.dot(weight))
    
        if splitByChr:
            for ch in chSet:
                scoreDict[ch] = np.append(scoreDict[ch], genotype[:,boolIdx[ch]].dot(weight[boolIdx[ch]]))
        start = end

    print('')
    score = pd.DataFrame.from_dict(scoreDict)
    return score

def selectInd(genoObj, idx):
    return({'SNPINFO': genoObj['SNPINFO'].copy(), 'INDINFO': genoObj['INDINFO'].iloc[idx,:].reset_index(drop=True).copy(), 'GENOTYPE': genoObj['GENOTYPE'][idx,:], 'FLIPINFO': genoObj['FLIPINFO'].copy() })

def getIndIndex(df):
    index = (df['FID'].astype(str)+':'+df['IID'].astype(str)).to_numpy()
    return index

def scoringByPlink(genoObj, weight, out='./result/', splitByChr=True, thread=-1):
    #The genotype will not be normalized
    scoreDict = {}

    scoreDict['all'] = np.array([])
    if splitByChr:
        chSet = list(set(genoObj['SNPINFO'].loc['CHR']))
        boolIdx = {}
        for ch in chSet:
            boolIdx[ch] = (genoObj['SNPINFO'].loc['CHR']==ch)
            scoreDict[ch] = np.array([])

    pattern = re.compile(r'Bed\([\'\"]([\w\\\/\.]+)[\'\"]')
    bfilename = os.path.splitext(pattern.match(genoObj['GENOTYPE'].__str__())[1])[0]
    weightObj = genoObj['SNPINFO'][['RAW_SNP', 'RAW_A1']].copy()
    weightObj.rename(columns={"RAW_SNP": "SNP", "RAW_A1": "A1"})
    weightObj.loc[:, 'BETAJ'] = weight
    weightObj.loc[genoObj['FLIPINFO'],'BETAJ'] = -weightObj.loc[genoObj['FLIPINFO'],'BETAJ']
    weightFilename = out+'plink_weight_all.txt' 
    try:
        weightObj.to_csv(weightFilename, sep='\t', index=False, header=False)
    except:
        print('Unable to write weights into file:', weightFilename)
        return
    
    prsFilename = out+'plink_prs_all'
    # os.system('plink --bfile '+bfilename+' --score '+ weightFilename+' --out '+prsFilename)
    from shutil import which
    version = 1
    if which('plink2') is not None:
        plinkExe = 'plink2' 
        version = 2
    elif which('plink') is not None:
        plinkExe = 'plink'
        version = 1
    else:
        print('Error: Unable to identify plink')
        return
    if version == 2 and thread!=-1:
        os.system(plinkExe+' --bfile '+bfilename+' --score '+ weightFilename+' --out '+prsFilename+' --threads '+str(thread))
        # runPlink = subprocess.run([plinkExe, "--bfile", bfilename,"--score",weightFilename,"--out", prsFilename,"--threads", str(thread)], stdout=subprocess.DEVNULL)
    else:
        os.system(plinkExe+' --bfile '+bfilename+' --score '+ weightFilename+' --out '+prsFilename)
        # runPlink = subprocess.run([plinkExe, "--bfile", bfilename,"--score",weightFilename,"--out", prsFilename], stdout=subprocess.DEVNULL)
    if version == 1:
        try:
            prsDF = pd.read_table(prsFilename+'.profile', sep='\t|\s+', engine='python')
        except:
            print("Unable to read prs file:", prsFilename+'.profile')
            return
    else:
        try:
            prsDF = pd.read_table(prsFilename+'.sscore', sep='\t|\s+', engine='python')
        except:
            print("Unable to read prs file:", prsFilename+'.sscore')
            return
        prsDF = prsDF.rename(columns={'#FID':'FID','SCORE1_AVG':'SCORE'})

    idx, pos=strUtils.listInDict(getIndIndex(genoObj['INDINFO']), pd.Series(range(len(prsDF)), index=getIndIndex(prsDF)))
    scoreDict['all'] = np.append(scoreDict['all'], prsDF['SCORE'].iloc[pos]*len(weightObj))

    if splitByChr:
        for ch in chSet:
            weightObj = genoObj['SNPINFO'][boolIdx[ch],['RAW_SNP', 'RAW_A1']].copy()
            weightObj.rename(columns={"RAW_SNP": "SNP", "RAW_A1": "A1"})
            weightObj.loc[:,'BETAJ'] = weight[boolIdx[ch]]
            weightObj.loc[genoObj['FLIPINFO'][boolIdx[ch]],'BETAJ'] = -weightObj.loc[genoObj['FLIPINFO'][boolIdx[ch]],'BETAJ']
            weightFilename = out+'plink_weight_'+ch+'.txt' 
        try:
            weightObj.to_csv(weightFilename, sep='\t', index=False, header=False)
        except:
            print('Unable to write weights into file:', weightFilename)
            return
        
        prsFilename = out+ 'plink_prs_'+ch

        if version == 2 and thread!=-1:
            os.system(plinkExe+' --bfile '+bfilename+' --score '+ weightFilename+' --out '+prsFilename+' --threads '+str(thread))
            # runPlink = subprocess.run([plinkExe, "--bfile", bfilename,"--score",weightFilename,"--out", prsFilename,"--threads", str(thread)], stdout=subprocess.DEVNULL)
        else:
            os.system(plinkExe+' --bfile '+bfilename+' --score '+ weightFilename+' --out '+prsFilename)
            # runPlink = subprocess.run([plinkExe, "--bfile", bfilename,"--score",weightFilename,"--out", prsFilename], stdout=subprocess.DEVNULL)
        if version == 1:
            try:
                prsDF = pd.read_table(prsFilename+'.profile', sep='\t|\s+', engine='python')
            except:
                print("Unable to read prs file:", prsFilename+'.profile')
                return
        else:
            try:
                prsDF = pd.read_table(prsFilename+'.sscore', sep='\t|\s+', engine='python')
            except:
                print("Unable to read prs file:", prsFilename+'.sscore')
                return
            prsDF = prsDF.rename(columns={'#FID':'FID','SCORE1_AVG':'SCORE'})

        scoreDict[ch] = np.append(scoreDict[ch], prsDF['SCORE'].iloc[pos]*len(weightObj))

    score = pd.DataFrame.from_dict(scoreDict)
    return score
