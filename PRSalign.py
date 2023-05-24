'''
Align summary stats, annotation, genotype, and reference panel into the same coordinate system.

@Author: Wei Jiang(w.jiang@yale.edu)
'''

import strUtils
import numpy as np
import pandas as pd
import copy
import multiprocessing
import warnings

compAllele = {'a':'t','g':'c','t':'a','c':'g'}

def matchAllelePair(A1, A2, A1s, A2s):
    isAmbi=isInv=isComp=False
    isValid = True
    if A2!='' and A2s!='':
        if A2==compAllele[A1] and A2s==compAllele[A1s]:
            isAmbi = True
            if A1==A1s:
                isInv = False
            elif A1==compAllele[A1s]:
                isInv = True
            else:
                isValid = False
        else:
            isAmbi = False
            if A1==A1s and A2==A2s:
                isInv = isComp = False
                isValid = True
            elif A1==A2s and A2==A1s:
                isComp = False
                isInv = isValid = True
            elif A1==compAllele[A1s] and A2==compAllele[A2s]:
                isInv = False
                isComp = isValid = True
            elif A1==compAllele[A2s] and A2==compAllele[A1s]:
                isInv = isComp = isValid = True
            else:
                isValid = False
    elif A2!='' and A2s=='':
        isAmbi=False
        if A1s==A1:
            isInv = isComp = False
            isValid = True
        elif A1s==A2:
            isComp = False
            isInv = isValid = True
        elif A1s==compAllele[A1]:
            isInv = False
            isComp = isValid = True
        elif A1s==compAllele[A2]:
            isInv = isComp = isValid = True
        else:
            isValid = False
    elif A2=='' and A2s!='':
        isAmbi=False
        if A1==A1s:
            isInv = isComp = False
            isValid = True
        elif A1==A2s:
            isComp = False
            isInv = isValid = True
        elif A1==compAllele[A1s]:
            isInv = False
            isComp = isValid = True
        elif A1==compAllele[A2s]:
            isInv = isComp = isValid = True
        else:
            isValid = False
    else:
        isAmbi=False
        isComp=False
        isValid=True
        isInv=(A1!=A1s) 
    return({'ambi':isAmbi,'inv':isInv,'comp':isComp,'valid':isValid})           

def matchAlleles(A1, A2, A1s, A2s, posStrand=None):
    A1 = np.asarray(A1)
    if A2 is not None:
        A2 = np.asarray(A2)
    else:
        A2 = np.array(['']*A1.shape[0])
    A1s = np.asarray(A1s)
    if A2s is not None:
        A2s = np.asarray(A2s)
    else:
        A2s = np.array(['']*A1s.shape[0])
    if A1.shape[0]!=A2.shape[0] or A2.shape[0]!=A1s.shape[0] or A1s.shape[0]!=A2s.shape[0]:
        print('The lengths of A1, A2, A1s and A2s should be matched!')
        return
    m = A1.shape[0]
    isAmbi = np.array([False]*m)
    isInv = np.array([False]*m)
    isComp = np.array([False]*m)
    isValid = np.array([True]*m)
    for i in range(m):
        alignStatus = matchAllelePair(A1[i],A2[i],A1s[i],A2s[i])
        isAmbi[i] = alignStatus['ambi']
        isInv[i] = alignStatus['inv']
        isComp[i] = alignStatus['comp']
        isValid[i] = alignStatus['valid']
    if posStrand is None:
        isValid = (~isAmbi) & isValid
    else:
        if not isinstance(posStrand,bool):
            ambiNum = np.sum(isAmbi)
            if ambiNum != m:
                compRate = np.sum((~isAmbi) & isComp)/(m-ambiNum)
                if compRate <= 0.5:
                    posStrand = True
                else:
                    posStrand = False
            else:
                posStrand = True
        if not posStrand:
            isInv[isAmbi] = ~isInv[isAmbi]
            isComp[isAmbi] = True
    return({'inv': isInv,'comp': isComp, 'valid':isValid})

def isPdNan(obj):
    try:
        aSize = obj.size
    except AttributeError:
        if isinstance(obj, (int, float, np.float, np.int)):
            return np.isnan(obj)
        else:
            return False
    return aSize==1 and np.isnan(obj[0])

def selectSS(ssDF, idx):
    return(ssDF.iloc[idx,:].reset_index(drop=True).copy())

def selectGeno(genoObj, idx):
    if isPdNan(genoObj['GENOTYPE']):
        gt = np.nan
    else:
        gt = genoObj['GENOTYPE'][:,idx]
    return({'SNPINFO':genoObj['SNPINFO'].iloc[idx,:].reset_index(drop=True).copy(),'INDINFO':genoObj['INDINFO'].copy(),'GENOTYPE':gt,'FLIPINFO':genoObj['FLIPINFO'][idx].copy()})

def selectAnno(annoObj, idx):
    return({'SNPINFO':annoObj['SNPINFO'].iloc[idx,:].reset_index(drop=True).copy(),'ANNODATA':annoObj['ANNODATA'].iloc[idx,:].reset_index(drop=True).copy()})

def extendAnno(annoObj, snpInfo, idxA, idxS):
    #idxA: Position in annoObj
    #idxS: Position in snpInfo
    newAnnoObj = {'SNPINFO':snpInfo.loc[:,snpInfo.columns.isin(['SNP','CHR','BP'])], 'ANNODATA': pd.DataFrame(np.nan, index=np.arange(len(snpInfo)), columns=annoObj['ANNODATA'].columns)}
    newAnnoObj['ANNODATA'].iloc[idxS,:]=annoObj['ANNODATA'].iloc[idxA,:].values
    return(newAnnoObj)

def selectRefSNP(refObj, bidx, idx):
    if len(idx)>0:
        refObj['SNPINFO'][bidx] = refObj['SNPINFO'][bidx].iloc[idx,].reset_index(drop=True).copy()
        refObj['LD'][bidx] = refObj['LD'][bidx][np.ix_(idx,idx)].copy()
    else:
        del refObj['BID'][bidx]
        del refObj['SNPINFO'][bidx]
        del refObj['LD'][bidx]
    return(refObj)

def selectRefBlock(refObj, bidx):
    # newRefObj = {}
    refObj['BID'] = [ refObj['BID'][k] for k in bidx ]
    refObj['SNPINFO'] = [ refObj['SNPINFO'][k] for k in bidx ]
    refObj['LD'] = [ refObj['LD'][k] for k in bidx ]
    # newRefObj['TOTAL'] = refObj['TOTAL']
    # newRefObj['INDNUM'] = refObj['INDNUM']
    return(refObj)

def getIndex(df, byID):
    if byID:
        index = df['SNP'].tolist()
    else:
        index = (df['CHR']+':'+df['BP'].astype(str)).tolist()
    return index

def selectDict(aDict, idx, A1list=None, A2list=None):
    aDict = aDict.iloc[idx,].copy()
    aDict[0:len(aDict)] = range(len(aDict))
    if (A1list is not None):
        A1list = A1list[idx]
        if (A2list is not None):
            A2list = A2list[idx]
        return aDict, A1list, A2list
    else:
        return aDict

def getCol(df, colName):
    if colName in df.columns:
        return df[colName].values
    else:
        return None

def unvAligner(ssList=[], genoList=[], refList=[], annoList=[], byID=True, posStrand=None, annoExt=False, alignRefs=True, copydata=False, thread=1):
    isInit = False
    ignoreAlleles = False
    
    if copydata:
        ssList = copy.deepcopy(ssList)
        genoList = copy.deepcopy(genoList)
        refList = copy.deepcopy(refList)
        annoList = copy.deepcopy(annoList)
    
    #align all genotypes
    genoNum = len(genoList)
    if isInit or (genoNum==0):
        start = 0
    else:
        snpInfo = genoList[0]['SNPINFO']
        snpDict = pd.Series(range(len(snpInfo)),index=getIndex(snpInfo,byID))
        A1list = snpInfo['A1'].values
        A2list = snpInfo['A2'].values
        isInit = True
        start = 1

    for i in range(start,genoNum):
        #Find overlapping SNPs
        idx, pos = strUtils.listInDict(getIndex(genoList[i]['SNPINFO'], byID), snpDict.to_dict())
        genoList[i] = selectGeno(genoList[i], idx) #Adjust current genotype obj
        snpDict, A1list, A2list = selectDict(snpDict, pos, A1list, A2list)
        # for j in range(ssNum):
            # ssList[j] = selectSS(ssList[j], pos)
        for j in range(i):
            genoList[j] = selectGeno(genoList[j], pos) #Adjust Previous genotypes
        #Match Alleles
        alleleMatchStatus = matchAlleles(A1list, A2list, genoList[i]['SNPINFO']['A1'], genoList[i]['SNPINFO']['A2'],posStrand)
        invPos = np.arange(len(snpDict))[alleleMatchStatus['inv']]

        if not isPdNan(genoList[i]['GENOTYPE']):
            genoList[i]['FLIPINFO'][invPos] = ~genoList[i]['FLIPINFO'][invPos]
        # genoList[i]['SNPINFO'].loc[:,'A1'] = A1list
        # genoList[i]['SNPINFO'].loc[:,'A2'] = A2list
        newPos = np.arange(len(snpDict))[alleleMatchStatus['valid']]
        snpDict, A1list, A2list = selectDict(snpDict, newPos, A1list, A2list)
        # for j in range(ssNum):
            # ssList[j] = selectSS(ssList[j], newPos)
        for j in range(i+1):
            genoList[j] = selectGeno(genoList[j], newPos)

    #align all summary stats
    ssNum = len(ssList)
    ssInit = -1
    if isInit or (ssNum==0):
        start = 0
    else:
        ssInit = 0
        for i in range(ssNum):
            if 'A2' in ssList[i].columns:
                ssInit = i
                break
        snpDict = pd.Series(range(len(ssList[ssInit])),index=getIndex(ssList[ssInit],byID))
        A1list = ssList[ssInit]['A1'].values
        A2list = getCol(ssList[ssInit], 'A2')
        isInit = True
        start = 0

    for i in range(start,ssNum):
        if i == ssInit: continue
        #Find overlapping SNPs
        idx, pos = strUtils.listInDict(getIndex(ssList[i], byID), snpDict.to_dict())
        ssList[i] = selectSS(ssList[i], idx) #Adjust summary stats
        snpDict, A1list, A2list = selectDict(snpDict, pos, A1list, A2list)
        for j in range(genoNum):
            genoList[j] = selectGeno(genoList[j], pos) #Adjust Previous genotypes
        for j in range(i):
            ssList[j] = selectSS(ssList[j], pos) #Adjust Previous summary stats
        #Match Alleles
        alleleMatchStatus = matchAlleles(A1list, A2list, ssList[i]['A1'], getCol(ssList[i],'A2'),posStrand)
        invBool = pd.array(alleleMatchStatus['inv'],'boolean')
        ssList[i].loc[invBool,'BETA'] = (-ssList[i].loc[invBool, 'BETA'])
        if 'F' in ssList[i].columns:
            ssList[i].loc[invBool, 'F'] = (1.-ssList[i].loc[invBool,'F'])
        ssList[i].loc[:,'A1'] = A1list
        if A2list is not None:
            ssList[i].loc[:,'A2'] = A2list
        newPos = np.arange(len(snpDict))[alleleMatchStatus['valid']]
        snpDict, A1list, A2list = selectDict(snpDict, newPos, A1list, A2list)
        for j in range(genoNum):
            genoList[j] = selectGeno(genoList[j], newPos)
        for j in range(i+1):
            ssList[j] = selectSS(ssList[j], newPos)

    #merge all annotations into one
    annoNum = len(annoList)

    if annoNum > 1:
        # annoObj = {key: value.copy() for key, value in annoList[0].items()} #or use copy.deepcopy()
        annoObj = annoList[0]
        annoObj['SNPINFO'].index=getIndex(annoObj['SNPINFO'], byID)
        annoObj['ANNODATA'].index=getIndex(annoObj['SNPINFO'], byID)
        annoObj['ANNODATA'].columns = np.char.add('F0.', annoObj['ANNODATA'].columns.values)

        for i in range(1,annoNum):
            # annoObj2 = {key: value.copy() for key, value in annoList[i].items()} #or use copy.deepcopy()
            annoObj2 = annoList[i]
            annoObj2['SNPINFO'].index = getIndex(annoList[i]['SNPINFO'], byID)
            annoObj2['ANNODATA'].index = getIndex(annoList[i]['SNPINFO'], byID)
            annoObj2['ANNODATA'].columns = np.char.add('F'+str(i)+'.', annoList[i]['ANNODATA'].columns.values)
            if annoExt:
                #Extending SNPs
                annoObj['SNPINFO'] = pd.concat([annoObj['SNPINFO'], annoObj2['SNPINFO']])
                annoObj['SNPINFO'] = annoObj['SNPINFO'][~annoObj['SNPINFO'].duplicated(keep='first')]
                annoObj['ANNODATA'] = pd.merge(annoObj['ANNODATA'], annoObj2['ANNODATA'], left_index=True, right_index=True, how='outer', sort=False).loc[annoObj['SNPINFO'].index]
            else:
                #Find overlapping SNPs
                idx = annoObj['SNPINFO'].index.isin(annoObj2['SNPINFO'].index)
                annoObj['SNPINFO'] = annoObj['SNPINFO'][idx]
                annoObj['ANNODATA'] = pd.merge(annoObj['ANNODATA'], annoObj2['ANNODATA'], left_index=True, right_index=True, how='inner', sort=False).loc[annoObj['SNPINFO'].index]
        
            # annoObj['ANNODATA'].fillna(0)
    
        annoObj['SNPINFO'] = annoObj['SNPINFO'].reset_index(drop=True)
        annoObj['ANNODATA'] = annoObj['ANNODATA'].reset_index(drop=True)
    elif annoNum==1:
        # annoObj = {key: value.copy() for key, value in annoList[0].items()} #or use copy.deepcopy()
        annoObj = annoList[0]
    else:
        annoObj = None

    #align annotations with other files (summary stats /genotypes)
    if annoNum >=1:
        if (not isInit):
            snpInfo = annoObj['SNPINFO']
            snpDict = pd.Series(range(len(snpInfo)),index=getIndex(snpInfo,byID))
            isInit = True
            ignoreAlleles = True

        else:
            idx, pos = strUtils.listInDict(getIndex(annoObj['SNPINFO'], byID), snpDict.to_dict())
            if annoExt:
                if ssNum>=1:
                    snpInfo = ssList[0]
                else:
                    snpInfo = genoList[0]['SNPINFO']
                annoObj = extendAnno(annoObj, snpInfo, idx, pos)
            else:
                #Find overlapping SNPs
                annoObj = selectAnno(annoObj, idx) #Adjust annotation obj
                snpDict, A1list, A2list = selectDict(snpDict, pos, A1list, A2list)
                for j in range(ssNum):
                    ssList[j] = selectSS(ssList[j], pos)
                for j in range(genoNum):
                    genoList[j] = selectGeno(genoList[j], pos) #Adjust Previous genotypes

    #align all references
    refNum = len(refList)
    if alignRefs:
        if refNum >1: 
            bidDict = pd.Series(range(len(refList[0]['BID'])),index=refList[0]['BID'])
            snpDictList = []
            refA1lists = []
            refA2lists = []
            reportgap = int(np.ceil(len(refList[0]['BID'])/10))
            for k in range(len(refList[0]['BID'])):
                snpInfo = refList[0]['SNPINFO'][k]
                snpDictList.append(pd.Series(range(len(snpInfo)),index=getIndex(snpInfo,byID)))
                refA1lists.append(snpInfo['A1'].values)
                refA2lists.append(snpInfo['A2'].values)
                if (k%reportgap == 0) or (k==len(refList[0]['BID'])-1): 
                    print('Building index, complete',int(np.ceil(k/reportgap))*10,'%', end='\r', flush=True)

        for i in range(1,refNum):
            #Find overlapping blocks
            bidx, bpos = strUtils.listInDict(refList[i]['BID'], bidDict.to_dict())
            refList[i] = selectRefBlock( refList[i], bidx )
            bidDict = selectDict(bidDict, bpos)
            # snpDictList = [ snpDictList[k] for k in bpos]
            # refA1lists = [ refA1lists[k] for k in bpos ]
            # refA2lists = [ refA2lists[k] for k in bpos ]
            for j in range(i):
                refList[j] = selectRefBlock( refList[j], bpos )
            
            reportgap = int(np.ceil(len(refList[i]['BID'])/10))
            currBk = 0
            for k in range(len(refList[i]['BID'])):
                #Find overlapping SNPs
                idx, pos = strUtils.listInDict(getIndex(refList[i]['SNPINFO'][currBk], byID), snpDictList[currBk].to_dict())
                refList[i] = selectRefSNP(refList[i], currBk, idx) #Adjust current reference obj
                snpDictList[k], refA1lists[k], refA2lists[k] = selectDict(snpDictList[k], pos, refA1lists[k], refA2lists[k])
                for j in range(i):
                    refList[j] = selectRefSNP(refList[j], currBk, pos) #Adjust Previous reference
                if len(idx)>0:
                    #Match Alleles
                    alleleMatchStatus = matchAlleles(refA1lists[k], refA2lists[k], refList[i]['SNPINFO'][currBk]['A1'], refList[i]['SNPINFO'][currBk]['A2'], posStrand)
                    invPos = np.arange(len(snpDictList[k]))[alleleMatchStatus['inv']]
                    nonInvPos = np.arange(len(snpDictList[k]))[~alleleMatchStatus['inv']]
                    refList[i]['LD'][currBk][nonInvPos,invPos] = (-refList[i]['LD'][currBk][nonInvPos,invPos])
                    refList[i]['LD'][currBk][invPos,nonInvPos] = (-refList[i]['LD'][currBk][invPos,nonInvPos])
                    refList[i]['SNPINFO'][currBk]['A1'] = refA1lists[k]
                    refList[i]['SNPINFO'][currBk]['A2'] = refA2lists[k]
                    refList[i]['SNPINFO'][currBk].iloc[invPos, 6] = 1.-refList[i]['SNPINFO'][currBk].iloc[invPos, 6] #Inverse Frequency in the 6th column
                    newPos = np.arange(len(snpDictList[k]))[alleleMatchStatus['valid']]
                    snpDictList[k], refA1lists[k], refA2lists[k] = selectDict(snpDictList[k], newPos, refA1lists[k], refA2lists[k])
                    for j in range(i+1):
                        refList[j] = selectRefSNP(refList[j], currBk, newPos)
                    currBk += 1
                if (k%reportgap == 0) or (k==len(refList[i]['BID'])-1): 
                    print('Aligning ref',i,'to ref 0, complete',int(np.ceil(k/reportgap))*10,'%', end='\r', flush=True)
            print('') 

    #align all files with references
    if isInit and (refNum>=1):
        if not (ignoreAlleles and annoExt):    
            finalPos = []
            reportgap = int(np.ceil(len(refList[0]['BID'])/10))
            currBk = 0
            try:
                thread = round(float(thread))
            except ValueError:
                print('PRSalign:unvAligner: thread must be a numeric value')
                thread = 0
            if thread!=1:
                cpuNum = multiprocessing.cpu_count()
                if thread<=0: #Default setting; using total cpu number
                    thread = cpuNum

            if thread>1:
                pool = multiprocessing.Pool(processes = thread)
                tmpResults = []
                finalInvPos = []
                finalA1 = []
                finalA2 = []
                for k in range(len(refList[0]['BID'])):
                    bkSNPinfo = refList[0]['SNPINFO'][k]
                    tmpResults.append(pool.apply_async(alignDict2RefBlock, args=(bkSNPinfo, snpDict, A1list, A2list, byID, ignoreAlleles, posStrand)))

                for k in range(len(refList[0]['BID'])):
                    tmpResult = tmpResults[k].get()
                    idx = tmpResult[0]
                    pos = tmpResult[1]
                    invPos = tmpResult[2]
                    newA1list = tmpResult[3]
                    newA2list = tmpResult[4]
                    for j in range(refNum):
                        refList[j] = selectRefSNP(refList[j], currBk, idx) #Adjust reference obj
                    if len(idx)>0:
                        finalPos.extend(pos)
                        finalInvPos.extend(invPos)
                        finalA1.extend(newA1list)
                        finalA2.extend(newA2list)
                        currBk += 1
                    if (k%reportgap == 0) or (k==len(refList[0]['BID'])-1): 
                        print('Aligning file(s) to reference panel, complete',int(np.ceil(k/reportgap))*10,'%', end='\r', flush=True)

                pool.close()
                #Call close function before join, otherwise error will raise. No process will be added to pool after close.
                #Join function: waiting the complete of subprocesses
                pool.join()
                invBool = pd.array([False]*len(snpDict), 'boolean')
                invBool[finalInvPos] = True
                for j in range(ssNum):
                    ssList[j].loc[invBool,'BETA'] = (-ssList[j].loc[invBool,'BETA'])
                    if 'F' in ssList[j].columns:
                        ssList[j].loc[invBool,'F'] = (1.-ssList[j].loc[invBool,'F'])
                    if 'A2' not in ssList[j].columns:
                        ssList[j].insert(ssList[j].columns.get_loc('A1')+1, 'A2', '')
                    ssList[j].iloc[finalPos, ssList[j].columns.get_loc('A1')] = finalA1
                    ssList[j].iloc[finalPos, ssList[j].columns.get_loc('A2')] = finalA2
                    
                for j in range(genoNum):
                    if not isPdNan(genoList[j]['GENOTYPE']):
                        genoList[i]['FLIPINFO'][finalInvPos] = ~genoList[i]['FLIPINFO'][finalInvPos]
                        # genoList[j]['SNPINFO'].iloc[finalPos, genoList[j]['SNPINFO'].columns.get_loc('A1')] = finalA1
                        # genoList[j]['SNPINFO'].iloc[finalPos, genoList[j]['SNPINFO'].columns.get_loc('A2')] = finalA2
            else:
                for k in range(len(refList[0]['BID'])):
                    idx, pos = strUtils.listInDict(getIndex(refList[0]['SNPINFO'][currBk], byID), snpDict.to_dict())
                    #Find overlapping SNPs
                    for j in range(refNum):
                        refList[j] = selectRefSNP(refList[j], currBk, idx) #Adjust reference obj
                    if len(idx)>0:
                        if not ignoreAlleles:
                            newA1list = A1list[pos]
                            if A2list is not None:
                                newA2list = A2list[pos]
                            else:
                                newA2list = None
                            #Match Alleles
                            alleleMatchStatus = matchAlleles(newA1list, newA2list, refList[0]['SNPINFO'][currBk]['A1'], refList[0]['SNPINFO'][currBk]['A2'], posStrand)
                            invPos = pos[alleleMatchStatus['inv']]
                            invBool = pd.array([False]*len(snpDict), 'boolean')
                            invBool[invPos] = True
                            for j in range(ssNum):
                                ssList[j].loc[invBool,'BETA'] = (-ssList[j].loc[invBool,'BETA'])
                                if 'F' in ssList[j].columns:
                                    ssList[j].loc[invBool,'F'] = (1.-ssList[j].loc[invBool,'F'])
                                if 'A2' not in ssList[j].columns:
                                    ssList[j].insert(ssList[j].columns.get_loc('A1')+1, 'A2', '')
                                ssList[j].iloc[pos, ssList[j].columns.get_loc('A1')] = refList[0]['SNPINFO'][currBk]['A1'].values
                                ssList[j].iloc[pos, ssList[j].columns.get_loc('A2')] = refList[0]['SNPINFO'][currBk]['A2'].values
                    
                            for j in range(genoNum):
                                if not isPdNan(genoList[j]['GENOTYPE']):
                                    genoList[i]['FLIPINFO'][invPos] = ~genoList[i]['FLIPINFO'][invPos]
                                # genoList[j]['SNPINFO'].iloc[pos, genoList[j]['SNPINFO'].columns.get_loc('A1')] = refList[0]['SNPINFO'][currBk]['A1'].values
                                # genoList[j]['SNPINFO'].iloc[pos, genoList[j]['SNPINFO'].columns.get_loc('A2')] = refList[0]['SNPINFO'][currBk]['A2'].values

                            validIdx = np.arange(len(idx))[alleleMatchStatus['valid']]
                            for j in range(refNum):
                                refList[j] = selectRefSNP(refList[j], currBk, validIdx) #Adjust reference obj
                            pos = pos[alleleMatchStatus['valid']]
                        finalPos.extend(pos)
                        currBk +=1
                    if (k%reportgap == 0) or (k==len(refList[0]['BID'])-1): 
                        print('Aligning file(s) to reference panel, complete',int(np.ceil(k/reportgap))*10,'%', end='\r', flush=True)
            
            print('') 
            for j in range(ssNum):
                ssList[j] = selectSS(ssList[j], finalPos)
            for j in range(genoNum):
                genoList[j] = selectGeno(genoList[j], finalPos)
            if annoNum>=1:    
                annoObj = selectAnno(annoObj, finalPos)
        else:
            snpInfo = pd.concat(refList[0]['SNPINFO'])
            snpInfo = snpInfo.reset_index(drop=True)
            refDict = pd.Series(range(len(snpInfo)),index=getIndex(snpInfo,byID))
            idx, pos = strUtils.listInDict(getIndex(annoObj['SNPINFO'], byID), refDict.to_dict())
            annoObj = extendAnno(annoObj, snpInfo, idx, pos)

    return({'SS': ssList, 'GENO': genoList, 'ANNO': annoObj, 'REF': refList})

def selectRefStoreSNP(refStore, outRefStore, bidx, idx, complevel):
    k = refStore.get('BID')[bidx]
    if len(idx)>0:
        outRefStore.put('SNPINFO'+str(k), refStore.get('SNPINFO'+str(k)).iloc[idx,].reset_index(drop=True).copy(), complevel=complevel)
        LDdf = refStore.get('LD'+str(k)).iloc[idx,idx].reset_index(drop=True).copy()
        LDdf.columns = np.arange(len(idx))
        outRefStore.put('LD'+str(k), LDdf, complevel=complevel)
    else:
        bidList = outRefStore.get('BID').tolist()
        bidList.remove(k)
        outRefStore.put('BID', pd.Series(bidList), complevel=complevel)
        try:
            outRefStore.remove('SNPINFO'+str(k))
            outRefStore.remove('LD'+str(k))
        except:
            pass
    return(outRefStore)

def selectRefStoreBlock(refStore, outRefStore, bidx, complevel=9):
    # newRefObj = {}
    bidList = refStore.get('BID')
    outRefStore.put('BID', pd.Series([ bidList[k] for k in bidx ]), complevel=complevel)
    delList = np.setdiff1d(np.arange(len(bidList)), bidx)
    for i in range(len(delList)):
        k = bidList[delList[i]]
        try:
            outRefStore.remove('SNPINFO'+str(k))
            outRefStore.remove('LD'+str(k))
        except:
            pass
    # newRefObj['TOTAL'] = refObj['TOTAL']
    # newRefObj['INDNUM'] = refObj['INDNUM']
    return(outRefStore)

def alignDict2RefBlock(bkSNPinfo, snpDict, A1list, A2list, byID=True, ignoreAlleles=False, posStrand=True):
    idx, pos = strUtils.listInDict(getIndex(bkSNPinfo, byID), snpDict.to_dict())
    invPos = []
    newA1list = []
    newA2list = []
    if len(idx)>0:
        if not ignoreAlleles:
            newA1list = A1list[pos]
            if A2list is not None:
                newA2list = A2list[pos]
            #Match Alleles
            alleleMatchStatus = matchAlleles(newA1list, newA2list, bkSNPinfo['A1'].iloc[idx], bkSNPinfo['A2'].iloc[idx], posStrand)
            invPos = pos[alleleMatchStatus['inv']]
            idx = idx[alleleMatchStatus['valid']]
            pos = pos[alleleMatchStatus['valid']]
        newA1list = bkSNPinfo['A1'].iloc[idx].values 
        newA2list = bkSNPinfo['A2'].iloc[idx].values 
    return idx, pos, invPos, newA1list, newA2list

def alignDict2RefStoreBlock(refFile, bk, snpDict, A1list, A2list, byID=True, ignoreAlleles=False, posStrand=True):
    with pd.HDFStore(refFile, 'r') as refStore:
        bkSNPinfo = refStore.get('SNPINFO'+str(bk))
    result = alignDict2RefBlock(bkSNPinfo, snpDict, A1list, A2list, byID, ignoreAlleles, posStrand)
    return result
               
def unvStoreAligner(ssList=[], genoList=[], refStoreList=[], annoList=[], outRefStoreList=[], byID=True, posStrand=None, annoExt=False, alignRefs=True, copydata=False, complevel=9, thread=1):
    isInit = False
    ignoreAlleles = False
    
    try:
        assert len(outRefStoreList) == len(refStoreList)
    except AssertionError:
        print('PRSaligner.unvStoreAligner: refStoreList and outRefStoreList should have the same length!')
    for i in range(len(refStoreList)):
        outRefStoreList[i].put('TOTAL', refStoreList[i].get('TOTAL'), complevel=complevel)
        outRefStoreList[i].put('INDNUM', refStoreList[i].get('INDNUM'), complevel=complevel)
        outRefStoreList[i].put('BID', refStoreList[i].get('BID'), complevel=complevel)

    if copydata:
        ssList = copy.deepcopy(ssList)
        genoList = copy.deepcopy(genoList)
        annoList = copy.deepcopy(annoList)
    
    #align all genotypes
    genoNum = len(genoList)
    if isInit:
        start = 0
    else:
        if genoNum >=1:
            snpInfo = genoList[0]['SNPINFO']
            snpDict = pd.Series(range(len(snpInfo)),index=getIndex(snpInfo,byID))
            A1list = snpInfo['A1'].values
            A2list = snpInfo['A2'].values
            isInit = True
            start = 1
        else: 
            start = 0

    for i in range(start,genoNum):
        #Find overlapping SNPs
        idx, pos = strUtils.listInDict(getIndex(genoList[i]['SNPINFO'], byID), snpDict.to_dict())
        genoList[i] = selectGeno(genoList[i], idx) #Adjust current genotype obj
        snpDict, A1list, A2list = selectDict(snpDict, pos, A1list, A2list)
        # for j in range(ssNum):
            # ssList[j] = selectSS(ssList[j], pos)
        for j in range(i):
            genoList[j] = selectGeno(genoList[j], pos) #Adjust Previous genotypes
        #Match Alleles
        alleleMatchStatus = matchAlleles(A1list, A2list, genoList[i]['SNPINFO']['A1'], genoList[i]['SNPINFO']['A2'],posStrand)
        invPos = np.arange(len(snpDict))[alleleMatchStatus['inv']]

        if not isPdNan(genoList[i]['GENOTYPE']):
            genoList[i]['FLIPINFO'][invPos] = ~genoList[i]['FLIPINFO'][invPos]
        # genoList[i]['SNPINFO'].loc[:,'A1'] = A1list
        # genoList[i]['SNPINFO'].loc[:,'A2'] = A2list
        newPos = np.arange(len(snpDict))[alleleMatchStatus['valid']]
        snpDict, A1list, A2list = selectDict(snpDict, newPos, A1list, A2list)
        # for j in range(ssNum):
            # ssList[j] = selectSS(ssList[j], newPos)
        for j in range(i+1):
            genoList[j] = selectGeno(genoList[j], newPos)

    #align all summary stats
    ssNum = len(ssList)
    ssInit = -1
    if isInit or (ssNum==0):
        start = 0
    else:
        ssInit = 0
        for i in range(ssNum):
            if 'A2' in ssList[i].columns:
                ssInit = i
                break
        snpDict = pd.Series(range(len(ssList[ssInit])),index=getIndex(ssList[ssInit],byID))
        A1list = ssList[ssInit]['A1'].values
        A2list = getCol(ssList[ssInit], 'A2')
        isInit = True
        start = 1

    for i in range(start,ssNum):
        if i==ssInit: continue
        #Find overlapping SNPs
        idx, pos = strUtils.listInDict(getIndex(ssList[i], byID), snpDict.to_dict())
        ssList[i] = selectSS(ssList[i], idx) #Adjust summary stats
        snpDict, A1list, A2list = selectDict(snpDict, pos, A1list, A2list)
        for j in range(genoNum):
            genoList[j] = selectGeno(genoList[j], pos) #Adjust Previous genotypes
        for j in range(i):
            ssList[j] = selectSS(ssList[j], pos) #Adjust Previous summary stats
        #Match Alleles
        alleleMatchStatus = matchAlleles(A1list, A2list, ssList[i]['A1'], getCol(ssList[i], 'A2'),posStrand)
        invBool = pd.array(alleleMatchStatus['inv'],'boolean')
        ssList[i].loc[invBool,'BETA'] = (-ssList[i].loc[invBool, 'BETA'])
        if 'F' in ssList[i].columns:
            ssList[i].loc[invBool, 'F'] = (1.-ssList[i].loc[invBool,'F'])
        ssList[i].loc[:,'A1'] = A1list
        if A2list is not None:
            ssList[i].loc[:,'A2'] = A2list
        newPos = np.arange(len(snpDict))[alleleMatchStatus['valid']]
        snpDict, A1list, A2list = selectDict(snpDict, newPos, A1list, A2list)
        for j in range(genoNum):
            genoList[j] = selectGeno(genoList[j], newPos)
        for j in range(i+1):
            ssList[j] = selectSS(ssList[j], newPos)
    
    #merge all annotations into one
    annoNum = len(annoList)

    if annoNum > 1:
        # annoObj = {key: value.copy() for key, value in annoList[0].items()} #or use copy.deepcopy()
        annoObj = annoList[0]
        annoObj['SNPINFO'].index=getIndex(annoObj['SNPINFO'], byID)
        annoObj['ANNODATA'].index=getIndex(annoObj['SNPINFO'], byID)
        annoObj['ANNODATA'].columns = np.char.add('F0.', annoObj['ANNODATA'].columns.values)

        for i in range(1,annoNum):
            # annoObj2 = {key: value.copy() for key, value in annoList[i].items()} #or use copy.deepcopy()
            annoObj2 = annoList[i]
            annoObj2['SNPINFO'].index = getIndex(annoList[i]['SNPINFO'], byID)
            annoObj2['ANNODATA'].index = getIndex(annoList[i]['SNPINFO'], byID)
            annoObj2['ANNODATA'].columns = np.char.add('F'+str(i)+'.', annoList[i]['ANNODATA'].columns.values)
            if annoExt:
                #Extending SNPs
                annoObj['SNPINFO'] = pd.concat([annoObj['SNPINFO'], annoObj2['SNPINFO']])
                annoObj['SNPINFO'] = annoObj['SNPINFO'][~annoObj['SNPINFO'].duplicated(keep='first')]
                annoObj['ANNODATA'] = pd.merge(annoObj['ANNODATA'], annoObj2['ANNODATA'], left_index=True, right_index=True, how='outer', sort=False).loc[annoObj['SNPINFO'].index]
            else:
                #Find overlapping SNPs
                idx = annoObj['SNPINFO'].index.isin(annoObj2['SNPINFO'].index)
                annoObj['SNPINFO'] = annoObj['SNPINFO'][idx]
                annoObj['ANNODATA'] = pd.merge(annoObj['ANNODATA'], annoObj2['ANNODATA'], left_index=True, right_index=True, how='inner', sort=False).loc[annoObj['SNPINFO'].index]
        
            # annoObj['ANNODATA'].fillna(0)
    
        annoObj['SNPINFO'] = annoObj['SNPINFO'].reset_index(drop=True)
        annoObj['ANNODATA'] = annoObj['ANNODATA'].reset_index(drop=True)
    elif annoNum==1:
        # annoObj = {key: value.copy() for key, value in annoList[0].items()} #or use copy.deepcopy()
        annoObj = annoList[0]
    else:
        annoObj = None

    #align annotations with other files (summary stats /genotypes)
    if annoNum >=1:
        if (not isInit):
            snpInfo = annoObj['SNPINFO']
            snpDict = pd.Series(range(len(snpInfo)),index=getIndex(snpInfo,byID))
            isInit = True
            ignoreAlleles = True

        else:
            idx, pos = strUtils.listInDict(getIndex(annoObj['SNPINFO'], byID), snpDict.to_dict())
            if annoExt:
                if ssNum>=1:
                    snpInfo = ssList[0]
                else:
                    snpInfo = genoList[0]['SNPINFO']
                annoObj = extendAnno(annoObj, snpInfo, idx, pos)
            else:
                #Find overlapping SNPs
                annoObj = selectAnno(annoObj, idx) #Adjust annotation obj
                snpDict, A1list, A2list = selectDict(snpDict, pos, A1list, A2list)
                for j in range(ssNum):
                    ssList[j] = selectSS(ssList[j], pos)
                for j in range(genoNum):
                    genoList[j] = selectGeno(genoList[j], pos) #Adjust Previous genotypes

    #align all references
    refNum = len(refStoreList)
    if alignRefs:
        if refNum >1: 
            bidDict = pd.Series(range(len(refStoreList[0].get('BID'))),index=refStoreList[0].get('BID'))
            snpDictList = []
            refA1lists = []
            refA2lists = []
            bidList = refStoreList[0].get('BID')
            reportgap = int(np.ceil(len(bidList)/10))
            for k in range(len(bidList)):
                snpInfo = refStoreList[0].get('SNPINFO'+str(bidList[k]))
                snpDictList.append(pd.Series(range(len(snpInfo)),index=getIndex(snpInfo,byID)))
                refA1lists.append(snpInfo['A1'].values)
                refA2lists.append(snpInfo['A2'].values)
                if (k%reportgap == 0) or (k==len(bidList)-1): 
                    print('Building index, complete',int(np.ceil(k/reportgap))*10,'%', end='\r', flush=True)

        for i in range(1,refNum):
            #Find overlapping blocks
            bidx, bpos = strUtils.listInDict(refStoreList[i].get('BID'), bidDict.to_dict())
            outRefStoreList[i] = selectRefStoreBlock( refStoreList[i], outRefStoreList[i], bidx , complevel=complevel)
            bidDict = selectDict(bidDict, bpos)
            # snpDictList = [ snpDictList[k] for k in bpos]
            # refA1lists = [ refA1lists[k] for k in bpos ]
            # refA2lists = [ refA2lists[k] for k in bpos ]
            for j in range(i):
                outRefStoreList[j] = selectRefStoreBlock( refStoreList[j], outRefStoreList[j], bpos, complevel=complevel )
            
            bidList = outRefStoreList[i].get('BID').to_numpy()
            reportgap = int(np.ceil(len(bidList)/10))
            currBk = 0
            for k in range(len(bidList)):
                #Find overlapping SNPs
                idx, pos = strUtils.listInDict(getIndex(refStoreList[i].get('SNPINFO'+str(bidList[k])), byID), snpDictList[k].to_dict())
                outRefStoreList[i] = selectRefStoreSNP(refStoreList[i], outRefStoreList[i], k, idx, complevel=complevel) #Adjust current reference obj
                snpDictList[k], refA1lists[k], refA2lists[k] = selectDict(snpDictList[k], pos, refA1lists[k], refA2lists[k])
                for j in range(i):
                    outRefStoreList[j] = selectRefStoreSNP(outRefStoreList[j], outRefStoreList[j], currBk, pos, complevel=complevel) #Adjust Previous reference
                if len(idx)>0:
                    #Match Alleles
                    alleleMatchStatus = matchAlleles(refA1lists[k], refA2lists[k], outRefStoreList[i].get('SNPINFO'+str(bidList[k]))['A1'], outRefStoreList[i].get('SNPINFO'+str(bidList[k]))['A2'], posStrand)
                    invPos = np.arange(len(snpDictList[k]))[alleleMatchStatus['inv']]
                    nonInvPos = np.arange(len(snpDictList[k]))[~alleleMatchStatus['inv']]
                    tmpLD = outRefStoreList[i].get('LD'+str(bidList[k]))
                    tmpLD.iloc[nonInvPos,invPos] = (-tmpLD.iloc[nonInvPos,invPos])
                    tmpLD.iloc[invPos,nonInvPos] = (-tmpLD.iloc[invPos,nonInvPos])
                    outRefStoreList[i].put('LD'+str(bidList[k]), tmpLD, complevel=complevel)
                    tmpSNPINFO = outRefStoreList[i].get('SNPINFO'+str(bidList[k]))
                    tmpSNPINFO['A1'] = refA1lists[k]
                    tmpSNPINFO['A2'] = refA2lists[k]
                    tmpSNPINFO.iloc[invPos, 6] = 1.-tmpSNPINFO.iloc[invPos, 6] #Inverse Frequency in the 6th column
                    outRefStoreList[i].put('SNPINFO'+str(bidList[k]), tmpSNPINFO, complevel=complevel)
                    newPos = np.arange(len(snpDictList[k]))[alleleMatchStatus['valid']]
                    snpDictList[k], refA1lists[k], refA2lists[k] = selectDict(snpDictList[k], newPos, refA1lists[k], refA2lists[k])
                    for j in range(i+1):
                        outRefStoreList[j] = selectRefStoreSNP(outRefStoreList[j], outRefStoreList[j], currBk, newPos, complevel=complevel)
                    currBk += 1 
                if (k%reportgap == 0) or (k==len(bidList)-1): 
                    print('Aligning ref',i,'to ref 0, complete',int(np.ceil(k/reportgap))*10,'%', end='\r', flush=True)
            print('') 

    #align all files with references
    if isInit and (refNum>=1):
        if alignRefs and (refNum>1):
            bidList = outRefStoreList[0].get('BID').to_numpy()
        else:
            bidList = refStoreList[0].get('BID').to_numpy()
        if not (ignoreAlleles and annoExt):    
            finalPos = []
            reportgap = int(np.ceil(len(bidList)/10))
            currBk = 0
            try:
                thread = round(float(thread))
            except ValueError:
                print('PRSalign:unvStoreAligner: thread must be a numeric value')
                thread = 0
            if thread!=1:
                cpuNum = multiprocessing.cpu_count()
                if thread<=0: #Default setting; using total cpu number
                    thread = cpuNum

            if thread>1:
                pool = multiprocessing.Pool(processes = thread)
                tmpResults = []
                finalInvPos = []
                finalA1 = []
                finalA2 = []
                for k in range(len(bidList)):
                    if alignRefs and (refNum>1):
                        bkSNPinfo = outRefStoreList[0].get('SNPINFO'+str(bidList[k]))
                    else:
                        bkSNPinfo = refStoreList[0].get('SNPINFO'+str(bidList[k]))

                    tmpResults.append(pool.apply_async(alignDict2RefBlock, args=(bkSNPinfo, snpDict, A1list, A2list, byID, ignoreAlleles, posStrand)))
                
                for k in range(len(bidList)): 
                    tmpResult = tmpResults[k].get()
                    idx = tmpResult[0]
                    pos = tmpResult[1]
                    invPos = tmpResult[2]
                    newA1list = tmpResult[3]
                    newA2list = tmpResult[4]
                    if alignRefs and (refNum>1):
                        for j in range(refNum):
                            outRefStoreList[j] = selectRefStoreSNP(outRefStoreList[j], outRefStoreList[j], currBk, idx, complevel=complevel) #Adjust reference obj
                    else:
                        for j in range(refNum):
                            outRefStoreList[j] = selectRefStoreSNP(refStoreList[j], outRefStoreList[j], currBk, idx, complevel=complevel) #Adjust reference obj
                    if len(idx)>0:
                        finalPos.extend(pos)
                        finalInvPos.extend(invPos)
                        finalA1.extend(newA1list)
                        finalA2.extend(newA2list)
                        currBk += 1
                    if (k%reportgap == 0) or (k==len(bidList)-1): 
                        print('Aligning file(s) to reference panel, complete', int(np.ceil(k/reportgap))*10,'%', end='\r', flush=True)

                pool.close()
                #Call close function before join, otherwise error will raise. No process will be added to pool after close.
                #Join function: waiting the complete of subprocesses
                pool.join()
                invBool = pd.array([False]*len(snpDict), 'boolean')
                invBool[finalInvPos] = True
                for j in range(ssNum):
                    ssList[j].loc[invBool,'BETA'] = (-ssList[j].loc[invBool,'BETA'])
                    if 'F' in ssList[j].columns:
                        ssList[j].loc[invBool,'F'] = (1.-ssList[j].loc[invBool,'F'])
                    if 'A2' not in ssList[j].columns:
                        ssList[j].insert(ssList[j].columns.get_loc('A1')+1, 'A2', '')
                    ssList[j].iloc[finalPos, ssList[j].columns.get_loc('A1')] = finalA1
                    ssList[j].iloc[finalPos, ssList[j].columns.get_loc('A2')] = finalA2
                    
                for j in range(genoNum):
                    if not isPdNan(genoList[j]['GENOTYPE']):
                        genoList[i]['FLIPINFO'][finalInvPos] = ~genoList[i]['FLIPINFO'][finalInvPos]
                        # genoList[j]['SNPINFO'].iloc[finalPos, genoList[j]['SNPINFO'].columns.get_loc('A1')] = finalA1
                        # genoList[j]['SNPINFO'].iloc[finalPos, genoList[j]['SNPINFO'].columns.get_loc('A2')] = finalA2

            else:        
                for k in range(len(bidList)):
                    if alignRefs and (refNum>1):
                        idx, pos = strUtils.listInDict(getIndex(outRefStoreList[0].get('SNPINFO'+str(bidList[k])), byID), snpDict.to_dict())
                        #Find overlapping SNPs
                        for j in range(refNum):
                            outRefStoreList[j] = selectRefStoreSNP(outRefStoreList[j], outRefStoreList[j], currBk, idx, complevel=complevel) #Adjust reference obj
                    else:
                        idx, pos = strUtils.listInDict(getIndex(refStoreList[0].get('SNPINFO'+str(bidList[k])), byID), snpDict.to_dict())
                        #Find overlapping SNPs
                        for j in range(refNum):
                            outRefStoreList[j] = selectRefStoreSNP(refStoreList[j], outRefStoreList[j], k, idx, complevel=complevel) #Adjust reference obj
                    if len(idx)>0:
                        if not ignoreAlleles:
                            newA1list = A1list[pos]
                            if A2list is not None:
                                newA2list = A2list[pos]
                            #Match Alleles
                            alleleMatchStatus = matchAlleles(newA1list, newA2list, outRefStoreList[0].get('SNPINFO'+str(bidList[k]))['A1'], outRefStoreList[0].get('SNPINFO'+str(bidList[k]))['A2'], posStrand)
                            invPos = pos[alleleMatchStatus['inv']]
                            invBool = pd.array([False]*len(snpDict), 'boolean')
                            invBool[invPos] = True
                            for j in range(ssNum):
                                ssList[j].loc[invBool,'BETA'] = (-ssList[j].loc[invBool,'BETA'])
                                if 'F' in ssList[j].columns:
                                    ssList[j].loc[invBool,'F'] = (1.-ssList[j].loc[invBool,'F'])
                            
                                if 'A2' not in ssList[j].columns:
                                    ssList[j].insert(ssList[j].columns.get_loc('A1')+1, 'A2', '')
                                ssList[j].iloc[pos, ssList[j].columns.get_loc('A1')] = outRefStoreList[0].get('SNPINFO'+str(bidList[k]))['A1'].values
                                ssList[j].iloc[pos, ssList[j].columns.get_loc('A2')] = outRefStoreList[0].get('SNPINFO'+str(bidList[k]))['A2'].values
                    
                            for j in range(genoNum):
                                if not isPdNan(genoList[j]['GENOTYPE']):
                                    genoList[i]['FLIPINFO'][invPos] = ~genoList[i]['FLIPINFO'][invPos]
                                # genoList[j]['SNPINFO'].iloc[pos, genoList[j]['SNPINFO'].columns.get_loc('A1')] = outRefStoreList[0].get('SNPINFO'+str(bidList[k]))['A1'].values
                                # genoList[j]['SNPINFO'].iloc[pos, genoList[j]['SNPINFO'].columns.get_loc('A2')] = outRefStoreList[0].get('SNPINFO'+str(bidList[k]))['A2'].values

                            validIdx = np.arange(len(idx))[alleleMatchStatus['valid']]
                            for j in range(refNum):
                                outRefStoreList[j] = selectRefStoreSNP(outRefStoreList[j], outRefStoreList[j], currBk, validIdx, complevel=complevel) #Adjust reference obj
                            pos = pos[alleleMatchStatus['valid']]
                        finalPos.extend(pos)
                        currBk += 1
                    if (k%reportgap == 0) or (k==len(bidList)-1): 
                        print('Aligning file(s) to reference panel, complete', int(np.ceil(k/reportgap))*10,'%', end='\r', flush=True)
            
            print('') 
            for j in range(ssNum):
                ssList[j] = selectSS(ssList[j], finalPos)
            for j in range(genoNum):
                genoList[j] = selectGeno(genoList[j], finalPos)
            if annoNum>=1:    
                annoObj = selectAnno(annoObj, finalPos)
        else:
            snpInfoList = []
            for k in range(len(bidList)):
                if alignRefs and (refNum>1):
                    snpInfoList.append(outRefStoreList[0].get('SNPINFO'+str(bidList[k])))        
                else:
                    snpInfoList.append(refStoreList[0].get('SNPINFO'+str(bidList[k])))        

            snpInfo = pd.concat(snpInfoList)
            snpInfo = snpInfo.reset_index(drop=True)
            refDict = pd.Series(range(len(snpInfo)),index=getIndex(snpInfo,byID))
            idx, pos = strUtils.listInDict(getIndex(annoObj['SNPINFO'], byID), refDict.to_dict())
            annoObj = extendAnno(annoObj, snpInfo, idx, pos)

    return({'SS': ssList, 'GENO': genoList, 'ANNO': annoObj, 'REF': outRefStoreList})

def saveSS(ssObj, out):
    try:
        ssObj.to_csv(out, sep='\t', index=False)
    except:
        print('Can not write summary stats into file:', out)
        return False
    return True

def saveGeno(genoObj, out, complevel=9):
    try:
        store = pd.HDFStore(out, 'w')
    except:
        print('Unable to create genotype file:',out)
        return False
    genotypePd = pd.Series(genoObj['GENOTYPE'])
    store.put('SNPINFO', value=genoObj['SNPINFO'], complevel=complevel, format='fixed')
    store.put('INDINFO', value=genoObj['INDINFO'], complevel=complevel, format='fixed')
    warnings.filterwarnings("ignore")
    store.put('GENOTYPE', value=genotypePd, complevel=complevel, format='fixed')
    warnings.resetwarnings()
    store.put('FLIPINFO', value=pd.Series(genoObj['FLIPINFO']), complevel=complevel, format='fixed')
    store.close()
    return True

def saveAnno(annoObj, out, complevel=9):
    try:
        store = pd.HDFStore(out, 'w', complevel=complevel)
    except:
        print('Unable to create annotation file',out)
        return False
    store.put('SNPINFO',value=annoObj['SNPINFO'], complevel=complevel, format='fixed')
    store.put('ANNODATA', value=annoObj['ANNODATA'], complevel=complevel, format='fixed')
    store.close()
    return True

def saveRef(refObj, out, complevel=9):
    try:
        store = pd.HDFStore(out, 'w', complevel=complevel)
    except:
        print('Unable to create LD file:', out)
        return False
    reportgap = int(np.ceil(len(refObj['BID'])/10))
    for i in range(len(refObj['BID'])):
        bid = refObj['BID'][i]
        store.put('SNPINFO'+str(bid),value=refObj['SNPINFO'][i], complevel=complevel, format='fixed')
        store.put('LD'+str(bid), value=pd.DataFrame(data=refObj['LD'][i]), complevel=complevel, format='fixed')
        if (i%reportgap == 0) or (i==len(refObj['BID'])-1): 
            print('Saving reference panel, complete', int(np.ceil(i/reportgap))*10,'%', end='\r', flush=True)
    print('')
    store.put('TOTAL', value=pd.Series(refObj['TOTAL']), complevel=complevel)
    store.put('INDNUM', value=pd.Series(refObj['INDNUM']), complevel=complevel)
    store.put('BID', value=pd.Series(refObj['BID']), complevel=complevel)
    store.close()
    return True

def alignSmry2Ref(ssObj, refObj, annoObj=None, outSS=None, outRef=None, outAnno=None, byID=True, posStrand=None, annoExt=False, complevel=9, thread=1):
    if annoObj is None:
        result = unvAligner(ssList=[ssObj], refList=[refObj], byID=byID, posStrand=posStrand, alignRefs=False, thread=thread)
    else:
        result = unvAligner(ssList=[ssObj], annoList=[annoObj], refList=[refObj], byID=byID, posStrand=posStrand, annoExt=annoExt, alignRefs=False, thread=thread)
    if outSS is not None: 
        print('Saving aligned summary stats to', outSS)
        saveSS(result['SS'][0], outSS)
    if outRef is not None:
        print('Saving aligned reference panel to', outRef)
        saveRef(result['REF'][0], outRef, complevel=complevel)
    if (annoObj is not None) and (outAnno is not None):
        print('Saving aligned annotations to', outAnno)
        saveAnno(result['ANNO'], outAnno, complevel=complevel)
    return({'SS': result['SS'][0], 'REF': result['REF'][0], 'ANNO': result['ANNO']})

def alignSmry2Geno(smryObj, genoObj, outSmry=None, outGeno=None, byID=True, posStrand=None, complevel=9):
    result = unvAligner(ssList=[smryObj], genoList=[genoObj], byID=byID, posStrand=posStrand, alignRefs=False)
    if outSmry is not None:
        print('Saving aligned summary statistics to', outSmry)
        saveSS(result['SS'][0], outSmry)
    if outGeno is not None:
        print('Saving aligned genotypes to', outGeno)
        saveGeno(result['GENO'][0], outGeno, complevel=complevel)
    return({'SS': result['SS'][0],'GENO': result['GENO'][0]})

def alignGeno2Ref(genoObj, refObj, outGeno=None, outRef=None, byID=True, posStrand=None, complevel=9, thread=1):
    result = unvAligner(genoList=[genoObj], refList=[refObj], byID=byID, posStrand=posStrand, alignRefs=False, thread=thread)
    if outGeno is not None:
        print('Saving aligned genotypes to', outGeno)
        saveGeno(result['GENO'][0], outGeno, complevel=complevel)
    if outRef is not None:
        print('Saving aligned reference panel to', outRef)
        saveRef(result['REF'][0], outRef, complevel=complevel)
    return({'GENO': result['GENO'][0], 'REF': result['REF'][0]})

def alignSmryGeno2Ref(ssObj, genoObj, refObj, annoObj=None, outSS=None, outGeno=None, outRef=None, outAnno=None, byID=True, posStrand=None, annoExt=False, complevel=9, thread=1):
    if annoObj is None:
        result = unvAligner(ssList=[ssObj], genoList=[genoObj], refList=[refObj], byID=byID, posStrand=posStrand, alignRefs=False, thread=thread)
    else:
        result = unvAligner(ssList=[ssObj], genoList=[genoObj], annoList=[annoObj], refList=[refObj], byID=byID, posStrand=posStrand, annoExt=annoExt, alignRefs=False, thread=thread)
    if outSS is not None: 
        print('Saving aligned summary stats to', outSS)
        saveSS(result['SS'][0], outSS)
    if outGeno is not None:
        print('Saving aligned genotypes to', outGeno)
        saveGeno(result['GENO'][0], outGeno, complevel=complevel)
    if outRef is not None:
        print('Saving aligned reference panel to', outRef)
        saveRef(result['REF'][0], outRef, complevel=complevel)
    if (annoObj is not None) and (outAnno is not None):
        print('Saving aligned annotations to', outAnno)
        saveAnno(result['ANNO'], outAnno, complevel=complevel)
    return({'SS': result['SS'][0], 'GENO': result['GENO'][0],'REF': result['REF'][0], 'ANNO': result['ANNO']})
 
def alignSmry2RefStore(ssObj, refStore, outRefStore, annoObj=None, outSS=None, outAnno=None, byID=True, posStrand=None, annoExt=False, complevel=9, thread=1):
    if annoObj is None:
        result = unvStoreAligner(ssList=[ssObj], refStoreList=[refStore], outRefStoreList=[outRefStore], byID=byID, posStrand=posStrand, alignRefs=False, complevel=complevel, thread=thread)
    else:
        result = unvStoreAligner(ssList=[ssObj], annoList=[annoObj], refStoreList=[refStore], outRefStoreList=[outRefStore], byID=byID, posStrand=posStrand, annoExt=annoExt, alignRefs=False, complevel=complevel, thread=thread)
    if outSS is not None: 
        print('Saving aligned summary stats to', outSS)
        saveSS(result['SS'][0], outSS)
    if (annoObj is not None) and (outAnno is not None):
        print('Saving aligned annotations to', outAnno)
        saveAnno(result['ANNO'], outAnno, complevel=complevel)
    return({'SS': result['SS'][0], 'REF': result['REF'][0], 'ANNO': result['ANNO']})
        
def alignGeno2RefStore(genoObj, refStore, outRefStore, outGeno=None, byID=True, posStrand=None, complevel=9):
    result = unvStoreAligner(genoList=[genoObj], refStoreList=[refStore], outRefStoreList=[outRefStore], byID=byID, posStrand=posStrand, alignRefs=False, complevel=complevel)
    if outGeno is not None:
        print('Saving aligned genotypes to', outGeno)
        saveGeno(result['GENO'][0], outGeno, complevel=complevel)
    return({'GENO': result['GENO'][0], 'REF': result['REF'][0]})

def alignSmryGeno2RefStore(ssObj, genoObj, refStore, outRefStore, annoObj=None, outSS=None, outGeno=None, outAnno=None, byID=True, posStrand=None, annoExt=False, complevel=9, thread=1):
    if annoObj is None:
        result = unvStoreAligner(ssList=[ssObj], genoList=[genoObj], refStoreList=[refStore], outRefStoreList=[outRefStore], byID=byID, posStrand=posStrand, alignRefs=False, complevel=complevel, thread=thread)
    else:
        result = unvStoreAligner(ssList=[ssObj], genoList=[genoObj], annoList=[annoObj], refStoreList=[refStore], outRefStoreList=[outRefStore], byID=byID, posStrand=posStrand, annoExt=annoExt, alignRefs=False, complevel=complevel, thread=thread)
    if outSS is not None: 
        print('Saving aligned summary stats to', outSS)
        saveSS(result['SS'][0], outSS)
    if outGeno is not None:
        print('Saving aligned genotypes to', outGeno)
        saveGeno(result['GENO'][0], outGeno, complevel=complevel)
    if (annoObj is not None) and (outAnno is not None):
        print('Saving aligned annotations to', outAnno)
        saveAnno(result['ANNO'], outAnno, complevel=complevel)
    return({'SS': result['SS'][0], 'GENO': result['GENO'][0],'REF': result['REF'][0], 'ANNO': result['ANNO']})
      
def updateLDSC(refObj):
    from ldsc import ldscore, ldscoreAnno
    newRefObj = copy.deepcopy(refObj)
    for i in range(len(refObj['BID'])):
        tmpLdscVal = ldscore(refObj['LD'][i], refObj['INDNUM'])
        newRefObj['SNPINFO'][i]['LDSC']= tmpLdscVal
    return(newRefObj)

def getAnnoLDSC(refObj, annoObj):
    from ldsc import ldscoreAnno
    start = 0
    ldscMatList = []
    for i in range(len(refObj['BID'])):
        end = start+refObj['LD'][i].shape[0]
        A = annoObj['ANNODATA'].iloc[start:end,:].to_numpy()
        tmpLdscMat = ldscoreAnno(refObj['LD'][i], A, refObj['INDNUM'])
        ldscMatList.append(tmpLdscMat)
        start = end
    ldscMat = np.concatenate(ldscMatList) 
    return(ldscMat)

def unvAligner2Pointer(ssList=[], genoList=[], refList=[], annoList=[], refFileList=[], byID=True, posStrand=None, annoExt=False, copydata=False):
    isInit = False
    ignoreAlleles = False
    
    if copydata:
        ssList = copy.deepcopy(ssList)
        genoList = copy.deepcopy(genoList)
        refList = copy.deepcopy(refList)
        annoList = copy.deepcopy(annoList)
    
    #align all genotypes
    genoNum = len(genoList)
    if isInit or (genoNum==0):
        start = 0
    else:
        snpInfo = genoList[0]['SNPINFO']
        snpDict = pd.Series(range(len(snpInfo)),index=getIndex(snpInfo,byID))
        A1list = snpInfo['A1'].values
        A2list = snpInfo['A2'].values
        isInit = True
        start = 1

    for i in range(start,genoNum):
        #Find overlapping SNPs
        idx, pos = strUtils.listInDict(getIndex(genoList[i]['SNPINFO'], byID), snpDict.to_dict())
        genoList[i] = selectGeno(genoList[i], idx) #Adjust current genotype obj
        snpDict, A1list, A2list = selectDict(snpDict, pos, A1list, A2list)
        # for j in range(ssNum):
            # ssList[j] = selectSS(ssList[j], pos)
        for j in range(i):
            genoList[j] = selectGeno(genoList[j], pos) #Adjust Previous genotypes
        #Match Alleles
        alleleMatchStatus = matchAlleles(A1list, A2list, genoList[i]['SNPINFO']['A1'], genoList[i]['SNPINFO']['A2'],posStrand)
        invPos = np.arange(len(snpDict))[alleleMatchStatus['inv']]

        if not isPdNan(genoList[i]['GENOTYPE']):
            genoList[i]['FLIPINFO'][invPos] = ~genoList[i]['FLIPINFO'][invPos]
        # genoList[i]['SNPINFO'].loc[:,'A1'] = A1list
        # genoList[i]['SNPINFO'].loc[:,'A2'] = A2list
        newPos = np.arange(len(snpDict))[alleleMatchStatus['valid']]
        snpDict, A1list, A2list = selectDict(snpDict, newPos, A1list, A2list)
        # for j in range(ssNum):
            # ssList[j] = selectSS(ssList[j], newPos)
        for j in range(i+1):
            genoList[j] = selectGeno(genoList[j], newPos)

    #align all summary stats
    ssNum = len(ssList)
    ssInit = -1
    if isInit or (ssNum==0):
        start = 0
    else:
        ssInit = 0
        for i in range(ssNum):
            if 'A2' in ssList[i].columns:
                ssInit = i
                break
        snpDict = pd.Series(range(len(ssList[ssInit])),index=getIndex(ssList[ssInit],byID))
        A1list = ssList[ssInit]['A1'].values
        A2list = getCol(ssList[ssInit], 'A2')
        isInit = True
        start = 0

    for i in range(start,ssNum):
        if i == ssInit: continue
        #Find overlapping SNPs
        idx, pos = strUtils.listInDict(getIndex(ssList[i], byID), snpDict.to_dict())
        ssList[i] = selectSS(ssList[i], idx) #Adjust summary stats
        snpDict, A1list, A2list = selectDict(snpDict, pos, A1list, A2list)
        for j in range(genoNum):
            genoList[j] = selectGeno(genoList[j], pos) #Adjust Previous genotypes
        for j in range(i):
            ssList[j] = selectSS(ssList[j], pos) #Adjust Previous summary stats
        #Match Alleles
        alleleMatchStatus = matchAlleles(A1list, A2list, ssList[i]['A1'], getCol(ssList[i],'A2'),posStrand)
        invBool = pd.array(alleleMatchStatus['inv'],'boolean')
        ssList[i].loc[invBool,'BETA'] = (-ssList[i].loc[invBool, 'BETA'])
        if 'F' in ssList[i].columns:
            ssList[i].loc[invBool, 'F'] = (1.-ssList[i].loc[invBool,'F'])
        ssList[i].loc[:,'A1'] = A1list
        if A2list is not None:
            ssList[i].loc[:,'A2'] = A2list
        newPos = np.arange(len(snpDict))[alleleMatchStatus['valid']]
        snpDict, A1list, A2list = selectDict(snpDict, newPos, A1list, A2list)
        for j in range(genoNum):
            genoList[j] = selectGeno(genoList[j], newPos)
        for j in range(i+1):
            ssList[j] = selectSS(ssList[j], newPos)

    #merge all annotations into one
    annoNum = len(annoList)

    if annoNum > 1:
        # annoObj = {key: value.copy() for key, value in annoList[0].items()} #or use copy.deepcopy()
        annoObj = annoList[0]
        annoObj['SNPINFO'].index=getIndex(annoObj['SNPINFO'], byID)
        annoObj['ANNODATA'].index=getIndex(annoObj['SNPINFO'], byID)
        annoObj['ANNODATA'].columns = np.char.add('F0.', annoObj['ANNODATA'].columns.values)

        for i in range(1,annoNum):
            # annoObj2 = {key: value.copy() for key, value in annoList[i].items()} #or use copy.deepcopy()
            annoObj2 = annoList[i]
            annoObj2['SNPINFO'].index = getIndex(annoList[i]['SNPINFO'], byID)
            annoObj2['ANNODATA'].index = getIndex(annoList[i]['SNPINFO'], byID)
            annoObj2['ANNODATA'].columns = np.char.add('F'+str(i)+'.', annoList[i]['ANNODATA'].columns.values)
            if annoExt:
                #Extending SNPs
                annoObj['SNPINFO'] = pd.concat([annoObj['SNPINFO'], annoObj2['SNPINFO']])
                annoObj['SNPINFO'] = annoObj['SNPINFO'][~annoObj['SNPINFO'].duplicated(keep='first')]
                annoObj['ANNODATA'] = pd.merge(annoObj['ANNODATA'], annoObj2['ANNODATA'], left_index=True, right_index=True, how='outer', sort=False).loc[annoObj['SNPINFO'].index]
            else:
                #Find overlapping SNPs
                idx = annoObj['SNPINFO'].index.isin(annoObj2['SNPINFO'].index)
                annoObj['SNPINFO'] = annoObj['SNPINFO'][idx]
                annoObj['ANNODATA'] = pd.merge(annoObj['ANNODATA'], annoObj2['ANNODATA'], left_index=True, right_index=True, how='inner', sort=False).loc[annoObj['SNPINFO'].index]
        
            # annoObj['ANNODATA'].fillna(0)
    
        annoObj['SNPINFO'] = annoObj['SNPINFO'].reset_index(drop=True)
        annoObj['ANNODATA'] = annoObj['ANNODATA'].reset_index(drop=True)
    elif annoNum==1:
        # annoObj = {key: value.copy() for key, value in annoList[0].items()} #or use copy.deepcopy()
        annoObj = annoList[0]
    else:
        annoObj = None

    #align annotations with other files (summary stats /genotypes)
    if annoNum >=1:
        if (not isInit):
            snpInfo = annoObj['SNPINFO']
            snpDict = pd.Series(range(len(snpInfo)),index=getIndex(snpInfo,byID))
            isInit = True
            ignoreAlleles = True

        else:
            idx, pos = strUtils.listInDict(getIndex(annoObj['SNPINFO'], byID), snpDict.to_dict())
            if annoExt:
                if ssNum>=1:
                    snpInfo = ssList[0]
                else:
                    snpInfo = genoList[0]['SNPINFO']
                annoObj = extendAnno(annoObj, snpInfo, idx, pos)
            else:
                #Find overlapping SNPs
                annoObj = selectAnno(annoObj, idx) #Adjust annotation obj
                snpDict, A1list, A2list = selectDict(snpDict, pos, A1list, A2list)
                for j in range(ssNum):
                    ssList[j] = selectSS(ssList[j], pos)
                for j in range(genoNum):
                    genoList[j] = selectGeno(genoList[j], pos) #Adjust Previous genotypes

    #align all references
    refNum = len(refList)
    
    #align all files with references
    if isInit and (refNum>=1):
        if not (ignoreAlleles and annoExt):    
            finalPos = []
            reportgap = int(np.ceil(len(refList[0]['BID'])/10))
            currBk = 0
            
            idxlist = []
            for k in range(len(refList[0]['BID'])):
                idx, pos = strUtils.listInDict(getIndex(refList[0]['SNPINFO'][currBk], byID), snpDict.to_dict())
                if len(idx)>0:
                    if not ignoreAlleles:
                        newA1list = A1list[pos]
                        if A2list is not None:
                            newA2list = A2list[pos]
                        else:
                            newA2list = None
                        #Match Alleles
                        alleleMatchStatus = matchAlleles(newA1list, newA2list, refList[0]['SNPINFO'][currBk]['A1'].iloc[idx], refList[0]['SNPINFO'][currBk]['A2'].iloc[idx], posStrand)
                        invPos = pos[alleleMatchStatus['inv']]
                        invBool = pd.array([False]*len(snpDict), 'boolean')
                        invBool[invPos] = True
                        for j in range(ssNum):
                            ssList[j].loc[invBool,'BETA'] = (-ssList[j].loc[invBool,'BETA'])
                            if 'F' in ssList[j].columns:
                                ssList[j].loc[invBool,'F'] = (1.-ssList[j].loc[invBool,'F'])
                            if 'A2' not in ssList[j].columns:
                                ssList[j].insert(ssList[j].columns.get_loc('A1')+1, 'A2', '')
                            ssList[j].iloc[pos, ssList[j].columns.get_loc('A1')] = refList[0]['SNPINFO'][currBk]['A1'].iloc[idx].values
                            ssList[j].iloc[pos, ssList[j].columns.get_loc('A2')] = refList[0]['SNPINFO'][currBk]['A2'].iloc[idx].values
                    
                        for j in range(genoNum):
                            if not isPdNan(genoList[j]['GENOTYPE']):
                                genoList[i]['FLIPINFO'][invPos] = ~genoList[i]['FLIPINFO'][invPos]
                            # genoList[j]['SNPINFO'].iloc[pos, genoList[j]['SNPINFO'].columns.get_loc('A1')] = refList[0]['SNPINFO'][currBk]['A1'].values
                            # genoList[j]['SNPINFO'].iloc[pos, genoList[j]['SNPINFO'].columns.get_loc('A2')] = refList[0]['SNPINFO'][currBk]['A2'].values

                        idx = idx[alleleMatchStatus['valid']]
                        pos = pos[alleleMatchStatus['valid']]
                    finalPos.extend(pos)
                idxlist.append(idx)
                currBk +=1
                if (k%reportgap == 0) or (k==len(refList[0]['BID'])-1): 
                    print('Aligning file(s) to reference panel, complete',int(np.ceil(k/reportgap))*10,'%', end='\r', flush=True)
            
            print('') 
            for j in range(ssNum):
                ssList[j] = selectSS(ssList[j], finalPos)
            for j in range(genoNum):
                genoList[j] = selectGeno(genoList[j], finalPos)
            if annoNum>=1:    
                annoObj = selectAnno(annoObj, finalPos)
            refPointerList = []
            for j in range(refNum):
                refPointerList.append({'FILE':refFileList[j], 'IDXLIST':idxlist})
        else:
            snpInfo = pd.concat(refList[0]['SNPINFO'])
            snpInfo = snpInfo.reset_index(drop=True)
            refDict = pd.Series(range(len(snpInfo)),index=getIndex(snpInfo,byID))
            idx, pos = strUtils.listInDict(getIndex(annoObj['SNPINFO'], byID), refDict.to_dict())
            annoObj = extendAnno(annoObj, snpInfo, idx, pos)
            refPointerList = []
            for j in range(refNum):
                refPointerList.append({'FILE':refFileList[j]})

    return({'SS': ssList, 'GENO': genoList, 'ANNO': annoObj, 'REF': refPointerList})

def unvStoreAligner2Pointer(ssList=[], genoList=[], refStoreList=[], annoList=[], outRefStoreList=[], refFileList=[], byID=True, posStrand=None, annoExt=False, copydata=False, complevel=9, thread=1):
    isInit = False
    ignoreAlleles = False
    
    try:
        assert len(outRefStoreList) == len(refStoreList)
    except AssertionError:
        print('PRSaligner.unvStoreAligner: refStoreList and outRefStoreList should have the same length!')

    if copydata:
        ssList = copy.deepcopy(ssList)
        genoList = copy.deepcopy(genoList)
        annoList = copy.deepcopy(annoList)
    
    #align all genotypes
    genoNum = len(genoList)
    if isInit:
        start = 0
    else:
        if genoNum >=1:
            snpInfo = genoList[0]['SNPINFO']
            snpDict = pd.Series(range(len(snpInfo)),index=getIndex(snpInfo,byID))
            A1list = snpInfo['A1'].values
            A2list = snpInfo['A2'].values
            isInit = True
            start = 1
        else: 
            start = 0

    for i in range(start,genoNum):
        #Find overlapping SNPs
        idx, pos = strUtils.listInDict(getIndex(genoList[i]['SNPINFO'], byID), snpDict.to_dict())
        genoList[i] = selectGeno(genoList[i], idx) #Adjust current genotype obj
        snpDict, A1list, A2list = selectDict(snpDict, pos, A1list, A2list)
        # for j in range(ssNum):
            # ssList[j] = selectSS(ssList[j], pos)
        for j in range(i):
            genoList[j] = selectGeno(genoList[j], pos) #Adjust Previous genotypes
        #Match Alleles
        alleleMatchStatus = matchAlleles(A1list, A2list, genoList[i]['SNPINFO']['A1'], genoList[i]['SNPINFO']['A2'],posStrand)
        invPos = np.arange(len(snpDict))[alleleMatchStatus['inv']]

        if not isPdNan(genoList[i]['GENOTYPE']):
            genoList[i]['FLIPINFO'][invPos] = ~genoList[i]['FLIPINFO'][invPos]
        # genoList[i]['SNPINFO'].loc[:,'A1'] = A1list
        # genoList[i]['SNPINFO'].loc[:,'A2'] = A2list
        newPos = np.arange(len(snpDict))[alleleMatchStatus['valid']]
        snpDict, A1list, A2list = selectDict(snpDict, newPos, A1list, A2list)
        # for j in range(ssNum):
            # ssList[j] = selectSS(ssList[j], newPos)
        for j in range(i+1):
            genoList[j] = selectGeno(genoList[j], newPos)

    #align all summary stats
    ssNum = len(ssList)
    ssInit = -1
    if isInit or (ssNum==0):
        start = 0
    else:
        ssInit = 0
        for i in range(ssNum):
            if 'A2' in ssList[i].columns:
                ssInit = i
                break
        snpDict = pd.Series(range(len(ssList[ssInit])),index=getIndex(ssList[ssInit],byID))
        A1list = ssList[ssInit]['A1'].values
        A2list = getCol(ssList[ssInit], 'A2')
        isInit = True
        start = 1

    for i in range(start,ssNum):
        if i==ssInit: continue
        #Find overlapping SNPs
        idx, pos = strUtils.listInDict(getIndex(ssList[i], byID), snpDict.to_dict())
        ssList[i] = selectSS(ssList[i], idx) #Adjust summary stats
        snpDict, A1list, A2list = selectDict(snpDict, pos, A1list, A2list)
        for j in range(genoNum):
            genoList[j] = selectGeno(genoList[j], pos) #Adjust Previous genotypes
        for j in range(i):
            ssList[j] = selectSS(ssList[j], pos) #Adjust Previous summary stats
        #Match Alleles
        alleleMatchStatus = matchAlleles(A1list, A2list, ssList[i]['A1'], getCol(ssList[i], 'A2'),posStrand)
        invBool = pd.array(alleleMatchStatus['inv'],'boolean')
        ssList[i].loc[invBool,'BETA'] = (-ssList[i].loc[invBool, 'BETA'])
        if 'F' in ssList[i].columns:
            ssList[i].loc[invBool, 'F'] = (1.-ssList[i].loc[invBool,'F'])
        ssList[i].loc[:,'A1'] = A1list
        if A2list is not None:
            ssList[i].loc[:,'A2'] = A2list
        newPos = np.arange(len(snpDict))[alleleMatchStatus['valid']]
        snpDict, A1list, A2list = selectDict(snpDict, newPos, A1list, A2list)
        for j in range(genoNum):
            genoList[j] = selectGeno(genoList[j], newPos)
        for j in range(i+1):
            ssList[j] = selectSS(ssList[j], newPos)
    
    #merge all annotations into one
    annoNum = len(annoList)

    if annoNum > 1:
        # annoObj = {key: value.copy() for key, value in annoList[0].items()} #or use copy.deepcopy()
        annoObj = annoList[0]
        annoObj['SNPINFO'].index=getIndex(annoObj['SNPINFO'], byID)
        annoObj['ANNODATA'].index=getIndex(annoObj['SNPINFO'], byID)
        annoObj['ANNODATA'].columns = np.char.add('F0.', annoObj['ANNODATA'].columns.values)

        for i in range(1,annoNum):
            # annoObj2 = {key: value.copy() for key, value in annoList[i].items()} #or use copy.deepcopy()
            annoObj2 = annoList[i]
            annoObj2['SNPINFO'].index = getIndex(annoList[i]['SNPINFO'], byID)
            annoObj2['ANNODATA'].index = getIndex(annoList[i]['SNPINFO'], byID)
            annoObj2['ANNODATA'].columns = np.char.add('F'+str(i)+'.', annoList[i]['ANNODATA'].columns.values)
            if annoExt:
                #Extending SNPs
                annoObj['SNPINFO'] = pd.concat([annoObj['SNPINFO'], annoObj2['SNPINFO']])
                annoObj['SNPINFO'] = annoObj['SNPINFO'][~annoObj['SNPINFO'].duplicated(keep='first')]
                annoObj['ANNODATA'] = pd.merge(annoObj['ANNODATA'], annoObj2['ANNODATA'], left_index=True, right_index=True, how='outer', sort=False).loc[annoObj['SNPINFO'].index]
            else:
                #Find overlapping SNPs
                idx = annoObj['SNPINFO'].index.isin(annoObj2['SNPINFO'].index)
                annoObj['SNPINFO'] = annoObj['SNPINFO'][idx]
                annoObj['ANNODATA'] = pd.merge(annoObj['ANNODATA'], annoObj2['ANNODATA'], left_index=True, right_index=True, how='inner', sort=False).loc[annoObj['SNPINFO'].index]
        
            # annoObj['ANNODATA'].fillna(0)
    
        annoObj['SNPINFO'] = annoObj['SNPINFO'].reset_index(drop=True)
        annoObj['ANNODATA'] = annoObj['ANNODATA'].reset_index(drop=True)
    elif annoNum==1:
        # annoObj = {key: value.copy() for key, value in annoList[0].items()} #or use copy.deepcopy()
        annoObj = annoList[0]
    else:
        annoObj = None

    #align annotations with other files (summary stats /genotypes)
    if annoNum >=1:
        if (not isInit):
            snpInfo = annoObj['SNPINFO']
            snpDict = pd.Series(range(len(snpInfo)),index=getIndex(snpInfo,byID))
            isInit = True
            ignoreAlleles = True

        else:
            idx, pos = strUtils.listInDict(getIndex(annoObj['SNPINFO'], byID), snpDict.to_dict())
            if annoExt:
                if ssNum>=1:
                    snpInfo = ssList[0]
                else:
                    snpInfo = genoList[0]['SNPINFO']
                annoObj = extendAnno(annoObj, snpInfo, idx, pos)
            else:
                #Find overlapping SNPs
                annoObj = selectAnno(annoObj, idx) #Adjust annotation obj
                snpDict, A1list, A2list = selectDict(snpDict, pos, A1list, A2list)
                for j in range(ssNum):
                    ssList[j] = selectSS(ssList[j], pos)
                for j in range(genoNum):
                    genoList[j] = selectGeno(genoList[j], pos) #Adjust Previous genotypes

    #align all references
    refNum = len(refStoreList)
    
    #align all files with references
    if isInit and (refNum>=1):
        bidList = refStoreList[0].get('BID').to_numpy()
        if not (ignoreAlleles and annoExt):    
            finalPos = []
            reportgap = int(np.ceil(len(bidList)/10))
            currBk = 0
            idxlist = []
            try:
                thread = round(float(thread))
            except ValueError:
                print('PRSalign:unvStoreAligner2Pointer: thread must be a numeric value')
                thread = 0
            if thread!=1:
                cpuNum = multiprocessing.cpu_count()
                if thread<=0: #Default setting; using total cpu number
                    thread = cpuNum

            if thread>=1:
                pool = multiprocessing.Pool(processes = thread)
                tmpResults = []
                finalInvPos = []
                finalA1 = []
                finalA2 = []
                for k in range(len(bidList)):
                    tmpResults.append(pool.apply_async(alignDict2RefStoreBlock, args=(refFileList[0], bidList[k], snpDict, A1list, A2list, byID, ignoreAlleles, posStrand)))

                for k in range(len(bidList)): 
                    tmpResult = tmpResults[k].get()
                    idx = tmpResult[0]
                    pos = tmpResult[1]
                    invPos = tmpResult[2]
                    newA1list = tmpResult[3]
                    newA2list = tmpResult[4]
                    if len(idx)>0:
                        finalPos.extend(pos)
                        finalInvPos.extend(invPos)
                        finalA1.extend(newA1list)
                        finalA2.extend(newA2list)
                    idxlist.append(idx)
                    currBk += 1
                    if (k%reportgap == 0) or (k==len(bidList)-1): 
                        print('Aligning file(s) to reference panel, complete', int(np.ceil(k/reportgap))*10,'%', end='\r', flush=True)

                pool.close()
                #Call close function before join, otherwise error will raise. No process will be added to pool after close.
                #Join function: waiting the complete of subprocesses
                pool.join()
                invBool = pd.array([False]*len(snpDict), 'boolean')
                invBool[finalInvPos] = True
                for j in range(ssNum):
                    ssList[j].loc[invBool,'BETA'] = (-ssList[j].loc[invBool,'BETA'])
                    if 'F' in ssList[j].columns:
                        ssList[j].loc[invBool,'F'] = (1.-ssList[j].loc[invBool,'F'])
                    if 'A2' not in ssList[j].columns:
                        ssList[j].insert(ssList[j].columns.get_loc('A1')+1, 'A2', '')
                    ssList[j].iloc[finalPos, ssList[j].columns.get_loc('A1')] = finalA1
                    ssList[j].iloc[finalPos, ssList[j].columns.get_loc('A2')] = finalA2
                    
                for j in range(genoNum):
                    if not isPdNan(genoList[j]['GENOTYPE']):
                        genoList[i]['FLIPINFO'][finalInvPos] = ~genoList[i]['FLIPINFO'][finalInvPos]
                        # genoList[j]['SNPINFO'].iloc[finalPos, genoList[j]['SNPINFO'].columns.get_loc('A1')] = finalA1
                        # genoList[j]['SNPINFO'].iloc[finalPos, genoList[j]['SNPINFO'].columns.get_loc('A2')] = finalA2

            else:        
                for k in range(len(bidList)):
                    idx, pos = strUtils.listInDict(getIndex(refStoreList[0].get('SNPINFO'+str(bidList[k])), byID), snpDict.to_dict())
                    if len(idx)>0:
                        if not ignoreAlleles:
                            newA1list = A1list[pos]
                            if A2list is not None:
                                newA2list = A2list[pos]
                            #Match Alleles
                            alleleMatchStatus = matchAlleles(newA1list, newA2list, refStoreList[0].get('SNPINFO'+str(bidList[k]))['A1'].iloc[idx], refStoreList[0].get('SNPINFO'+str(bidList[k]))['A2'].iloc[idx], posStrand)
                            invPos = pos[alleleMatchStatus['inv']]
                            invBool = pd.array([False]*len(snpDict), 'boolean')
                            invBool[invPos] = True
                            for j in range(ssNum):
                                ssList[j].loc[invBool,'BETA'] = (-ssList[j].loc[invBool,'BETA'])
                                if 'F' in ssList[j].columns:
                                    ssList[j].loc[invBool,'F'] = (1.-ssList[j].loc[invBool,'F'])
                            
                                if 'A2' not in ssList[j].columns:
                                    ssList[j].insert(ssList[j].columns.get_loc('A1')+1, 'A2', '')
                                ssList[j].iloc[pos, ssList[j].columns.get_loc('A1')] = refStoreList[0].get('SNPINFO'+str(bidList[k]))['A1'].iloc[idx].values
                                ssList[j].iloc[pos, ssList[j].columns.get_loc('A2')] = refStoreList[0].get('SNPINFO'+str(bidList[k]))['A2'].iloc[idx].values
                    
                            for j in range(genoNum):
                                if not isPdNan(genoList[j]['GENOTYPE']):
                                    genoList[i]['FLIPINFO'][invPos] = ~genoList[i]['FLIPINFO'][invPos]
                                # genoList[j]['SNPINFO'].iloc[pos, genoList[j]['SNPINFO'].columns.get_loc('A1')] = refStoreList[0].get('SNPINFO'+str(bidList[k]))['A1'].iloc[idx].values
                                # genoList[j]['SNPINFO'].iloc[pos, genoList[j]['SNPINFO'].columns.get_loc('A2')] = refStoreList[0].get('SNPINFO'+str(bidList[k]))['A2'].iloc[idx].values

                            idx = idx[alleleMatchStatus['valid']]
                            pos = pos[alleleMatchStatus['valid']]
                        finalPos.extend(pos)
                    idxlist.append(idx)
                    currBk += 1
                    if (k%reportgap == 0) or (k==len(bidList)-1): 
                        print('Aligning file(s) to reference panel, complete', int(np.ceil(k/reportgap))*10,'%', end='\r', flush=True)
            
            print('') 
            for j in range(ssNum):
                ssList[j] = selectSS(ssList[j], finalPos)
            for j in range(genoNum):
                genoList[j] = selectGeno(genoList[j], finalPos)
            if annoNum>=1:    
                annoObj = selectAnno(annoObj, finalPos)
            for j in range(refNum):
                outRefStoreList[j].put('FILE', pd.Series(refFileList[j]), complevel=complevel)
                warnings.filterwarnings("ignore")
                outRefStoreList[j].put('IDXLIST', pd.Series(idxlist), complevel=complevel)
                warnings.resetwarnings()
        else:
            snpInfoList = []
            for k in range(len(bidList)):
                snpInfoList.append(refStoreList[0].get('SNPINFO'+str(bidList[k])))        

            snpInfo = pd.concat(snpInfoList)
            snpInfo = snpInfo.reset_index(drop=True)
            refDict = pd.Series(range(len(snpInfo)),index=getIndex(snpInfo,byID))
            idx, pos = strUtils.listInDict(getIndex(annoObj['SNPINFO'], byID), refDict.to_dict())
            annoObj = extendAnno(annoObj, snpInfo, idx, pos)
            for j in range(refNum):
                outRefStoreList[j].put('FILE', pd.Series(refFileList[j]), complevel=complevel)

    return({'SS': ssList, 'GENO': genoList, 'ANNO': annoObj, 'REF': outRefStoreList})

def saveRefPointer(refObj, out, complevel=9):
    try:
        store = pd.HDFStore(out, 'w', complevel=complevel)
    except:
        print('Unable to create LD file:', out)
        return False
    filename = refObj['FILE']
    idxlist = refObj['IDXLIST']
    store.put('FILE', pd.Series(filename), complevel=complevel)
    warnings.filterwarnings("ignore")
    store.put('IDXLIST', pd.Series(idxlist), complevel=complevel)
    warnings.resetwarnings()
    store.close()
    return True

def alignSmry2RefPointer(ssObj, refObj, refFile, annoObj=None, outSS=None, outRef=None, outAnno=None, byID=True, posStrand=None, annoExt=False, complevel=9, thread=1):
    if annoObj is None:
        result = unvAligner2Pointer(ssList=[ssObj], refList=[refObj], refFileList=[refFile], byID=byID, posStrand=posStrand)
    else:
        result = unvAligner2Pointer(ssList=[ssObj], annoList=[annoObj], refList=[refObj],refFileList=[refFile], byID=byID, posStrand=posStrand, annoExt=annoExt)
    if outSS is not None: 
        print('Saving aligned summary stats to', outSS)
        saveSS(result['SS'][0], outSS)
    if outRef is not None:
        print('Saving aligned reference panel to', outRef)
        saveRefPointer(result['REF'][0], outRef, complevel=complevel)
    if (annoObj is not None) and (outAnno is not None):
        print('Saving aligned annotations to', outAnno)
        saveAnno(result['ANNO'], outAnno, complevel=complevel)
    return({'SS': result['SS'][0], 'REF': result['REF'][0], 'ANNO': result['ANNO']})

def alignGeno2RefPointer(genoObj, refObj, refFile, outGeno=None, outRef=None, byID=True, posStrand=None, complevel=9, thread=1):
    result = unvAligner2Pointer(genoList=[genoObj], refList=[refObj], refFileList=[refFile], byID=byID, posStrand=posStrand)
    if outGeno is not None:
        print('Saving aligned genotypes to', outGeno)
        saveGeno(result['GENO'][0], outGeno, complevel=complevel)
    if outRef is not None:
        print('Saving aligned reference panel to', outRef)
        saveRefPointer(result['REF'][0], outRef, complevel=complevel)
    return({'GENO': result['GENO'][0], 'REF': result['REF'][0]})

def alignSmryGeno2RefPointer(ssObj, genoObj, refObj, refFile, annoObj=None, outSS=None, outGeno=None, outRef=None, outAnno=None, byID=True, posStrand=None, annoExt=False, complevel=9, thread=1):
    if annoObj is None:
        result = unvAligner2Pointer(ssList=[ssObj], genoList=[genoObj], refList=[refObj], refFileList=[refFile], byID=byID, posStrand=posStrand)
    else:
        result = unvAligner2Pointer(ssList=[ssObj], genoList=[genoObj], annoList=[annoObj], refList=[refObj], refFileList=[refFile], byID=byID, posStrand=posStrand, annoExt=annoExt)
    if outSS is not None: 
        print('Saving aligned summary stats to', outSS)
        saveSS(result['SS'][0], outSS)
    if outGeno is not None:
        print('Saving aligned genotypes to', outGeno)
        saveGeno(result['GENO'][0], outGeno, complevel=complevel)
    if outRef is not None:
        print('Saving aligned reference panel to', outRef)
        saveRefPointer(result['REF'][0], outRef, complevel=complevel)
    if (annoObj is not None) and (outAnno is not None):
        print('Saving aligned annotations to', outAnno)
        saveAnno(result['ANNO'], outAnno, complevel=complevel)
    return({'SS': result['SS'][0], 'GENO': result['GENO'][0],'REF': result['REF'][0], 'ANNO': result['ANNO']})
 
def alignSmry2RefStorePointer(ssObj, refStore, outRefStore, refFile, annoObj=None, outSS=None, outAnno=None, byID=True, posStrand=None, annoExt=False, complevel=9, thread=1):
    if annoObj is None:
        result = unvStoreAligner2Pointer(ssList=[ssObj], refStoreList=[refStore], outRefStoreList=[outRefStore], refFileList=[refFile], byID=byID, posStrand=posStrand, complevel=complevel, thread=thread)
    else:
        result = unvStoreAligner2Pointer(ssList=[ssObj], annoList=[annoObj], refStoreList=[refStore], outRefStoreList=[outRefStore], refFileList=[refFile], byID=byID, posStrand=posStrand, annoExt=annoExt, complevel=complevel, thread=thread)
    if outSS is not None: 
        print('Saving aligned summary stats to', outSS)
        saveSS(result['SS'][0], outSS)
    if (annoObj is not None) and (outAnno is not None):
        print('Saving aligned annotations to', outAnno)
        saveAnno(result['ANNO'], outAnno, complevel=complevel)
    return({'SS': result['SS'][0], 'REF': result['REF'][0], 'ANNO': result['ANNO']})
        
def alignGeno2RefStore(genoObj, refStore, outRefStore, refFile, outGeno=None, byID=True, posStrand=None, complevel=9, thread=1):
    result = unvStoreAligner2Pointer(genoList=[genoObj], refStoreList=[refStore], outRefStoreList=[outRefStore], refFileList=[refFile], byID=byID, posStrand=posStrand, complevel=complevel, thread=thread)
    if outGeno is not None:
        print('Saving aligned genotypes to', outGeno)
        saveGeno(result['GENO'][0], outGeno, complevel=complevel)
    return({'GENO': result['GENO'][0], 'REF': result['REF'][0]})

def alignSmryGeno2RefStorePointer(ssObj, genoObj, refStore, outRefStore, refFile, annoObj=None, outSS=None, outGeno=None, outAnno=None, byID=True, posStrand=None, annoExt=False, complevel=9, thread=1):
    if annoObj is None:
        result = unvStoreAligner2Pointer(ssList=[ssObj], genoList=[genoObj], refStoreList=[refStore], outRefStoreList=[outRefStore], refFileList=[refFile], byID=byID, posStrand=posStrand, complevel=complevel, thread=thread)
    else:
        result = unvStoreAligner2Pointer(ssList=[ssObj], genoList=[genoObj], annoList=[annoObj], refStoreList=[refStore], outRefStoreList=[outRefStore], refFileList=[refFile], byID=byID, posStrand=posStrand, annoExt=annoExt, complevel=complevel, thread=thread)
    if outSS is not None: 
        print('Saving aligned summary stats to', outSS)
        saveSS(result['SS'][0], outSS)
    if outGeno is not None:
        print('Saving aligned genotypes to', outGeno)
        saveGeno(result['GENO'][0], outGeno, complevel=complevel)
    if (annoObj is not None) and (outAnno is not None):
        print('Saving aligned annotations to', outAnno)
        saveAnno(result['ANNO'], outAnno, complevel=complevel)
    return({'SS': result['SS'][0], 'GENO': result['GENO'][0],'REF': result['REF'][0], 'ANNO': result['ANNO']})

def adjustLDbyTarget(R, jointSS, start, end, targetDict, targetSS):
    #R_tilde = S(S1^-1 R S1^-1+S2^-1 R S2^-1)S
    idx, pos = strUtils.listInDict(jointSS['SNP'].iloc[start:end].to_list(), targetDict.to_dict())
    if len(idx)>0:
        s = jointSS['SE'].iloc[start:end].to_numpy()
        s_2 = targetSS['SE'].iloc[pos].to_numpy()
        invS_2ext = np.zeros(len(s))
        invS_2ext[idx] = 1./s_2
        invS_2ext[np.isnan(invS_2ext)|np.isinf(invS_2ext)] = 0.
        invS_1ext2 = 1/(s**2)-invS_2ext**2
        invS_1ext2[(invS_1ext2<=0)|np.isnan(invS_1ext2)|np.isinf(invS_1ext2)] = 0
        invS_1ext = np.sqrt(invS_1ext2)
        innerR = R*invS_1ext[:, None]*invS_1ext[None,:]+R*invS_2ext[:, None]*invS_2ext[None,:]
        R_tilde = innerR*s[:,None]*s[None,:]
    else:
        R_tilde = R 
    return R_tilde

def adjustRefByTarget(jointSS, targetSS, refObj, thread=-1):
    newRefObj = refObj.copy()
    try:
        thread = round(float(thread))
    except ValueError:
        print('PRSalign.adjustRefByTarget: thread must be a numeric value')
        thread = 0
    if thread!=1:
        cpuNum = multiprocessing.cpu_count()
        if thread<=0: #Default setting; using total cpu number
            thread = cpuNum

    targetDict = pd.Series(range(len(targetSS)),index=targetSS['SNP'])
    reportgap = int(np.ceil(len(refObj['BID'])/10))
    if thread == 1:
        start = 0
        for k in range(len(refObj['BID'])):
            end = start+refObj['LD'][k].shape[0]
            newRefObj['LD'][k] = adjustLDbyTarget(refObj['LD'][k], jointSS, start, end, targetDict, targetSS)
            start = end
            if (k%reportgap == 0) or (k==len(refObj['BID'])-1): 
                print('Complete',int(np.ceil(k/reportgap))*10,'%', end='\r', flush=True)
    else:
        pool = multiprocessing.Pool(processes = thread)
        tmpResults = []
        start = 0
        for k in range(len(refObj['BID'])):
            end = start+refObj['LD'][k].shape[0]
            tmpResults.append(pool.apply_async(adjustLDbyTarget, args=(refObj['LD'][k], jointSS, start, end, targetDict, targetSS)))
            start = end
        for k in range(len(refObj['BID'])):
            newRefObj['LD'][k] = tmpResults[k].get()
            if (k%reportgap == 0) or (k==len(refObj['BID'])-1): 
                print('Complete',int(np.ceil(k/reportgap))*10,'%', end='\r', flush=True)
        pool.close()
        #Call close function before join, otherwise error will raise. No process will be added to pool after close.
        #Join function: waiting the complete of subprocesses
        pool.join()
    print('')
    return newRefObj

