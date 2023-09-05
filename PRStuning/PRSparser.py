'''
Parse summary stats/Genotype File/LD Reference panel
Created on 2020-10

@author: w.jiang@yale.edu
'''
import re
import strUtils
import numpy as np
from scipy.stats import norm
import plink2SnpReader
import multiprocessing
import warnings
import csv
from PRSalign import unvAligner
import pandas as pd

def smryParser(smryFile, SNP='SNP',CHR='CHR', BP='BP',A1='A1', A2='A2', BETA='BETA',OR='OR', SE='SE', CI='CI', ORCI='OR_CI', L95='L95', U95='U95', ORL95='OR_95L', ORU95='OR_95U', P='P', Z='Z', N='N', N0='N0', N1='N1', skip=0, comment='#', out=None,  INFO='INFO', FREQ='F', FA='FA', FU='FU', CHRBP='CHR_BP', A1A2='A1A2', CHRBPA1A2='CHR:BP_A1_A2',P_HET='P_HET', infoFilter= 0.9, mafFilter= 0.05, hetFilter=0, z2Filter=np.inf, NFilter=None, chrList=None, eff=False, noFilter=False, outDel=None):
    try:
        with open(smryFile,'r') as fp:
            i = 0
            for line in fp:
                line = line.strip()
                if i == skip:
                    if (line == '') or (line[0]==comment): continue #ignore commented and empty lines
                    header = line.split()
                    break
                i+=1
    except:
        print('Unable to read summary statistics from File:', smryFile)
        return
    usefulHeader = []
    Nval = None
    if (SNP not in header) and ((CHR not in header) or (BP not in header)) and (CHRBP not in header):
        print('Both SNP ID and (Chromosome & Base position) can not be located')
        return
    if (A1 not in header) and (A1A2 not in header):
        print('A1 can not be located')
        return
    if SNP in header: usefulHeader.append(SNP)
    if CHR in header: usefulHeader.append(CHR)
    if BP in header: usefulHeader.append(BP)
    if CHRBP in header: usefulHeader.append(CHRBP)
    if CHRBPA1A2 in header: usefulHeader.append(CHRBPA1A2)
    if (A1 in header):
        usefulHeader.append(A1)
        if (A2 in header):
            usefulHeader.append(A2)
    else:
        usefulHeader.append(A1A2)
    if FREQ in header: usefulHeader.append(FREQ)
    tmpFA = list(filter(re.compile(FA+"[_%d]*").match, header))
    if len(tmpFA)>0: 
        FA = tmpFA[0]
        usefulHeader.append(FA)
    tmpFU = list(filter(re.compile(FU+"[_%d]*").match, header))
    if len(tmpFU)>0: 
        FU = tmpFU[0]
        usefulHeader.append(FU)
    if INFO in header: usefulHeader.append(INFO)
    newHeader = [elem for elem in usefulHeader]
    # if Z in header: usefulHeader.append(Z)
    # if P in header: usefulHeader.append(P)
    if P_HET in header: usefulHeader.append(P_HET)
    if N in header: usefulHeader.append(N)
    if N0 in header: usefulHeader.append(N0)
    if N1 in header: usefulHeader.append(N1)

    def parseCI(x):
        splitList = x.split('-')
        return float(splitList[0]), float(splitList[1])

    vParseCI = np.vectorize(parseCI)

    if (BETA in header) or (OR in header):
        getBeta = 0
        getSE = 0
        if BETA in header: usefulHeader.append(BETA)
        else: 
            usefulHeader.append(OR)
            getBeta = 1

        if SE in header:
            usefulHeader.append(SE)
        elif Z in header:
            getSE = 1
            usefulHeader.append(Z)
        elif P in header:
            getSE = 2
            usefulHeader.append(P)
        elif CI in header:
            usefulHeader.append(CI)
            getSE = 3
        elif ORCI in header:
            usefulHeader.append(ORCI)
            getSE = 4
        elif (L95 in header) and (U95 in header):
            usefulHeader.append(L95)
            usefulHeader.append(U95)
            getSE = 5
        elif (ORL95 in header) and (ORU95 in header):
            usefulHeader.append(ORL95)
            usefulHeader.append(ORU95)
            getSE = 6
        elif N is not None:
            if strUtils.isnumeric(N) or (N in header): getSE = 7
            else:
                print('N is not correctly specified and not located from the header')
                return
        elif (N0 is not None) and (N1 is not None):
            if (strUtils.isnumeric(N0) or (N0 in header)) and (strUtils.isnumeric(N1) or (N1 in header)): getSE = 8 
            else:
                print('Either N0 or N1 can not be identified')
                return
        else:
            print('Located BETA, SE/Z/P/N/CI can not be located')
            return
        
    else:
        getBeta = 2
        getSE = 0
        if SE in header:
            usefulHeader.append(SE)
        elif CI in header:
            usefulHeader.append(CI)
            getSE = 3
        elif ORCI in header:
            usefulHeader.append(ORCI)
            getSE = 4
        elif (L95 in header) and (U95 in header):
            usefulHeader.append(L95)
            usefulHeader.append(U95)
            getSE = 5
        elif (ORL95 in header) and (ORU95 in header):
            usefulHeader.append(ORL95)
            usefulHeader.append(ORU95)
            getSE = 6
        elif N is not None or ((N0 is not None) and (N1 is not None)):
            if strUtils.isnumeric(N) or (N in header) or ((strUtils.isnumeric(N0) or (N0 in header)) and (strUtils.isnumeric(N1) or (N1 in header))):
                getSE = 7
            else:
                print('N is not correctly specified and not located from the header')
                return
        else:
            print('SE/N/CI can not be located')
            return
        
        if Z in header:
            usefulHeader.append(Z)
        # No direction information 
        # elif P in header: 
            # usefulHeader.append(P)
            # getBeta = 1
        else:
            print('Located SE, BETA/OR/Z can not be located')
            return
        
    try:
        ssDF = pd.read_table(smryFile, sep='\t|\s+', dtype={SNP:str,CHR:str,A1:str, A2:str}, usecols=usefulHeader, skiprows=skip, comment=comment, engine='python')
    except:
        print("Unable to correctly read summary stats:", smryFile)
        return

    if ((N is None) or not(strUtils.isnumeric(N) or (N in header))) and (N0 is not None) and (N1 is not None):
        if (strUtils.isnumeric(N0) or (N0 in header)) and (strUtils.isnumeric(N1) or (N1 in header)): 
            if strUtils.isnumeric(N0):
                N0val = float(N0)
            else:
                N0val = ssDF[N0].copy()
                ssDF = ssDF.drop(columns=[N0])
            if strUtils.isnumeric(N1):
                N1val = float(N1)
            else:
                N1val = ssDF[N1].copy()
                ssDF = ssDF.drop(columns=[N1])
            if eff:
                Nval = 4*N0val*N1val/(N0val+N1val)
            else:
                Nval = (N0val+N1val)
            N = 'N'
            ssDF[N] = Nval 
            header.append(N)
    
    if getBeta==1:
        ssDF[BETA] = np.log(ssDF[OR])
        ssDF = ssDF.drop(columns=[OR])
    
    if getSE == 1:
        ssDF[SE] = ssDF[BETA]/ssDF[Z]
        ssDF = ssDF.drop(columns=[Z])
    elif getSE == 2:
        ssDF[SE] = np.abs(ssDF[BETA]/norm.ppf(ssDF[P]/2))
        ssDF = ssDF.drop(columns=[P])
    elif getSE == 3:
        parseResult = vParseCI(ssDF[CI])
        ssDF[SE] = (parseResult[1]-parseResult[0])/(2.*1.96)
        ssDF = ssDF.drop(columns=[CI])
    elif getSE == 4:
        parseResult = vParseCI(ssDF[ORCI])
        ssDF[SE] = (np.log(parseResult[1])-np.log(parseResult[0]))/(2.*1.96)
        ssDF = ssDF.drop(columns=[ORCI])
    elif getSE == 5:
        ssDF[SE] = (ssDF[U95]-ssDF[L95])/(2.*1.96)
        ssDF = ssDF.drop(columns=[L95, U95])
    elif getSE == 6:
        ssDF[SE] = (np.log(ssDF[ORU95])-np.log(ssDF[ORL95]))/(2.*1.96)
        ssDF = ssDF.drop(columns=[ORL95, ORU95])
    elif getSE == 7:
        if strUtils.isnumeric(N):
            ssDF[SE] = 1./np.sqrt(float(N))
        else:
            ssDF[SE] = 1./np.sqrt(ssDF[N])
        
    if getBeta == 2:
        ssDF[BETA] = ssDF[SE]*ssDF[Z]
    # The direction information is lost
    # elif getBeta == 3:
        # ssDF[BETA] = np.abs(ssDF[SE]*norm.ppf(ssDF[P]/2))
        # ssDF = ssDF.drop(columns=[P])

    #Formatting CHR
    pattern = re.compile(r'^(?i:chr?)?(\w+)')
    vmatch = np.vectorize(lambda x: pattern.match(x).group(1))
    if CHR in header:
        ssDF[CHR] = np.char.lower(vmatch(ssDF[CHR]))
    
    if not((CHR in header) and (BP in header)) and (CHRBP in header):
        pattern = re.compile(r'^(?i:chr?)?(\w+):(\d+)')
        def parseCHRBP(x):
            matchResult = pattern.match(x)
            return matchResult.group(1), matchResult.group(2)
        vParseCHRBP = np.vectorize(parseCHRBP)
        parseResult = vParseCHRBP(ssDF[CHRBP])
        if not(CHR in header):
            ssDF[CHR] = np.char.lower(parseResult[0])
            header.insert(header.index(CHRBP), CHR)
            newHeader.insert(newHeader.index(CHRBP), CHR)
        if not(BP in header):
            ssDF[BP] = parseResult[1].astype(int)
            header.insert(header.index(CHRBP), BP)
            newHeader.insert(newHeader.index(CHRBP), BP)
    
    if (CHRBP in header):
        ssDF = ssDF.drop(columns=[CHRBP])
        newHeader.remove(CHRBP)

    if (A1A2 in header):
        def parseA1A2(x):
            return x[0], x[1]
        vParseA1A2 = np.vectorize(parseA1A2)
        parseResult = vParseA1A2(ssDF[A1A2])
        ssDF[A1] = parseResult[0]
        ssDF[A2] = parseResult[1]
        newHeader.insert(newHeader.index(A1A2), A1)
        newHeader.insert(newHeader.index(A1A2), A2)
        header.insert(header.index(A1A2), A1)
        header.insert(header.index(A1A2), A2)
        ssDF = ssDF.drop(columns=[A1A2])
        newHeader.remove(A1A2)
    
    if not((CHR in header) and (BP in header) and (A1 in header) and (A2 in header)) and (CHRBPA1A2 in header):
        pattern = re.compile(r'^(?i:chr?)?(\w+):(\d+)_(\w)_(\w)')
        def parseCHRBPA1A2(x):
            matchResult = pattern.match(x)
            return matchResult.group(1), matchResult.group(2), matchResult.group(3), matchResult.group(4)
        vParseCHRBPA1A2 = np.vectorize(parseCHRBPA1A2)
        parseResult = vParseCHRBPA1A2(ssDF[CHRBPA1A2])
        if not(CHR in header):
            ssDF[CHR] = np.char.lower(parseResult[0])
            header.insert(header.index(CHRBPA1A2), CHR)
            newHeader.insert(newHeader.index(CHRBPA1A2), CHR)
        if not(BP in header):
            ssDF[BP] = parseResult[1].astype(int)
            header.insert(header.index(CHRBPA1A2), BP)
            newHeader.insert(newHeader.index(CHRBPA1A2), BP)
        if not(A1 in header):
            ssDF[A1] = np.char.lower(parseResult[2])
            header.insert(header.index(CHRBPA1A2), A1)
            newHeader.insert(newHeader.index(CHRBPA1A2), A1)
        if not(A2 in header):
            ssDF[A2] = np.char.lower(parseResult[3])
            header.insert(header.index(CHRBPA1A2), A2)
            newHeader.insert(newHeader.index(CHRBPA1A2), A2)
    
    if (CHRBPA1A2 in header):
        ssDF = ssDF.drop(columns=[CHRBPA1A2])
        newHeader.remove(CHRBPA1A2)
        
    if SNP in header:
        ssDF[SNP] = ssDF[SNP].str.lower()
    #Change A1 and A2 to lower case for easily matching
    ssDF[A1] = ssDF[A1].str.lower()
    if A2 in ssDF.columns:
        ssDF[A2] = ssDF[A2].str.lower()
    
    ssDFdel = pd.DataFrame()

    alleleList = ['a','t','g','c']
    if A2 in ssDF.columns:
        invalidIdx = ~(ssDF[A1].isin(alleleList) & ssDF[A2].isin(alleleList))
    else:
        invalidIdx = ~(ssDF[A1].isin(alleleList))
    ssDFdel = pd.concat([ssDFdel,ssDF[invalidIdx]])
    ssDF = ssDF[~invalidIdx]

    print(len(ssDF),'SNPs in the file.')
    invalidNum = 0
    if (CHR in header) and (chrList is not None) and (not noFilter): 
        pattern = re.compile(r'^(\d+)-(\d+)')
        extChrList = []
        for i in range(len(chrList)):
            matchObj = pattern.match(chrList[i])
            if matchObj:
                for j in range(int(matchObj.group(1)), int(matchObj.group(2))+1):
                    extChrList.append(str(j))
            else:
                extChrList.append(chrList[i])
        invalidIdx = ~ssDF[CHR].isin(extChrList)
        invalidNum = np.sum(invalidIdx)
        if invalidNum>0:
            print(invalidNum,'SNP(s) are not in the chromosome list',chrList)
            ssDFdel = pd.concat([ssDFdel,ssDF[invalidIdx]])
            ssDF = ssDF[~invalidIdx]

    if FREQ in header:
        ssDF[FREQ] = 1.-ssDF[FREQ] #Change Freq from A1 to A2 
        if not noFilter:
            invalidIdx = (ssDF[FREQ]<mafFilter) | (ssDF[FREQ]>(1.-mafFilter)) | (np.isinf(ssDF[FREQ])) #| (np.isnan(ssDF[FREQ]))
            invalidNum = np.sum(invalidIdx)
            if invalidNum > 0:
                print(invalidNum,'SNP(s) with minor allele frequencies <', mafFilter)
                ssDFdel = pd.concat([ssDFdel,ssDF[invalidIdx]])
                ssDF = ssDF[~invalidIdx]
    
    if FA in header:
        ssDF[FA] = 1.-ssDF[FA] #Change Freq from A1 to A2 
        if not noFilter:
            invalidIdx = (ssDF[FA]<mafFilter) | (ssDF[FA]>(1.-mafFilter))  | (np.isinf(ssDF[FA])) # | (np.isnan(ssDF[FA]))
            invalidNum = np.sum(invalidIdx)
            if invalidNum > 0:
                print(invalidNum,'SNP(s) with minor allele frequencies (Cases) <', mafFilter)
                ssDFdel = pd.concat([ssDFdel,ssDF[invalidIdx]])
                ssDF = ssDF[~invalidIdx]
    
    if FU in header:
        ssDF[FU] = 1.-ssDF[FU] #Change Freq from A1 to A2 
        if not noFilter:
            invalidIdx = (ssDF[FU]<mafFilter) | (ssDF[FU]>(1.-mafFilter)) | (np.isinf(ssDF[FU])) #| (np.isnan(ssDF[FU]))
            invalidNum = np.sum(invalidIdx)
            if invalidNum > 0:
                print(invalidNum,'SNP(s) with minor allele frequencies (Controls) <', mafFilter)
                ssDFdel = pd.concat([ssDFdel,ssDF[invalidIdx]])
                ssDF = ssDF[~invalidIdx]
    
    if INFO in header and (not noFilter):
        invalidIdx = (ssDF[INFO]<infoFilter) | (np.isinf(ssDF[INFO])) #|(np.isnan(ssDF[INFO]))
        invalidNum = np.sum(invalidIdx)
        if invalidNum > 0:
            print(invalidNum,'SNP(s) with imputation quality INFO <',infoFilter)
            ssDFdel = pd.concat([ssDFdel, ssDF[invalidIdx]])
            ssDF = ssDF[~invalidIdx]

    if P_HET in header and (not noFilter):
        if not noFilter:
            invalidIdx = (ssDF[P_HET]<hetFilter) | (np.isinf(ssDF[P_HET])) #| (np.isnan(ssDF[P_HET]))
            invalidNum = np.sum(invalidIdx)
            if invalidNum > 0:
                print(invalidNum,'SNP(s) with Heterogeneity test p-value <',hetFilter)
                ssDFdel = pd.concat([ssDFdel,ssDF[invalidIdx]])
                ssDF = ssDF[~invalidIdx]
        ssDF = ssDF.drop(columns=[P_HET])
    
    #Is there any SNPs with SE<=0?
    if not noFilter:
        invalidIdx = (ssDF[SE]<=0) | (np.isnan(ssDF[SE])) | (np.isinf(ssDF[SE])) | (ssDF[BETA]==0) | (np.isnan(ssDF[BETA])) | (np.isinf(ssDF[BETA]))
        invalidNum = np.sum(invalidIdx)
        if invalidNum > 0:
            print(invalidNum,'SNP(s) with confused direction(s) of effect(s)')
            ssDFdel = pd.concat([ssDFdel,ssDF[invalidIdx]])
            ssDF = ssDF[~invalidIdx]

    #Is there any SNPs with extreme large effect size?
    if not noFilter:
        z2 = (ssDF[BETA]/ssDF[SE])**2
        invalidIdx = (z2>z2Filter) | (np.isnan(z2)) | (np.isinf(z2))
        invalidNum = np.sum(invalidIdx)
        if invalidNum > 0:
            print(invalidNum,'SNP(s) with z^2>',z2Filter)
            ssDFdel = pd.concat([ssDFdel,ssDF[invalidIdx]])
            ssDF = ssDF[~invalidIdx]

    if SNP in header:
        if not noFilter:
            invalidIdx =  pd.Index(ssDF[SNP]).duplicated(keep='first')
            invalidNum = np.sum(invalidIdx)
            if invalidNum > 0:
                print(invalidNum,'SNP(s) with duplicated SNP ID')
                ssDFdel = pd.concat([ssDFdel,ssDF[invalidIdx]])
                ssDF = ssDF[~invalidIdx]

    if (CHR in header) and (BP in header):
        if not noFilter:
            invalidIdx =  pd.Index(ssDF[CHR]+':'+ssDF[BP].astype(str)).duplicated(keep='first')
            invalidNum = np.sum(invalidIdx)
            if invalidNum > 0:
                print(invalidNum,'SNP(s) having identical positions with others')
                ssDFdel = pd.concat([ssDFdel,ssDF[invalidIdx]])
                ssDF = ssDF[~invalidIdx]
    '''
    if Z in header:
        if not noFilter:
            invalidIdx = (ssDF[Z] == 0) | (np.isnan(ssDF[Z])) | (np.isinf(ssDF[Z])) 
            invalidNum = np.sum(invalidIdx)
            if invalidNum > 0:
                print('Additional',invalidNum,'SNP(s) with Z==0')
                ssDFdel = pd.concat([ssDFdel,ssDF[invalidIdx]])
                ssDF = ssDF[~invalidIdx]
        
            invalidIdx = (ssDF[Z]**2 >=z2Filter)  
            invalidNum = np.sum(invalidIdx)
            if invalidNum > 0:
                print('Additional',invalidNum,'SNP(s) with z^2>=',z2Filter)
                ssDFdel = pd.concat([ssDFdel,ssDF[invalidIdx]])
                ssDF = ssDF[~invalidIdx]
        ssDF = ssDF.drop(columns=[Z])

    if P in header:
        if not noFilter:
            invalidIdx = (ssDF[P] == 1) | (np.isnan(ssDF[P])) | (np.isinf(ssDF[P])) 
            invalidNum = np.sum(invalidIdx)
            if invalidNum > 0:
                print('Additional',invalidNum,'SNP(s) with P==1')
                ssDFdel = pd.concat([ssDFdel,ssDF[invalidIdx]])
                ssDF = ssDF[~invalidIdx]
        ssDF = ssDF.drop(columns=[P])
    '''
    if N in header:
        if not noFilter:
            print('Maximum sample size:', ssDF[N].max())
            N90 = np.quantile(ssDF[N], 0.9)
            print('90% quantile of sample size:', N90)
            if NFilter is not None:
                invalidIdx = (ssDF[N]<(NFilter*N90))  | (np.isinf(ssDF[N]))#| (np.isnan(ssDF[N]))
            else:
                invalidIdx = (np.isinf(ssDF[N])) #| (np.isnan(ssDF[N]))

            invalidNum = np.sum(invalidIdx)
            if invalidNum > 0:
                print(invalidNum,'SNP(s) with sample size less than','{:.2f}'.format(NFilter), 'of 90% sample size quantile')
                ssDFdel = pd.concat([ssDFdel,ssDF[invalidIdx]])
                ssDF = ssDF[~invalidIdx]
        newHeader += [N]
        # ssDF = ssDF.drop(columns=[N])
    
    if SNP not in newHeader:
        #Use CHR:BP as SNP ID
        ssDF[SNP] = (ssDF[CHR]+':'+ssDF['BP'].astype(str))
        newHeader.insert(0, SNP)
    
    newHeader+= [BETA,SE]
    ssDF = ssDF[newHeader]
    ssDF = ssDF.rename(columns={SNP: 'SNP',CHR:'CHR',BP:'BP',A1:'A1',A2:'A2',BETA:'BETA',SE:'SE', INFO:'INFO',FREQ:'F', FA:'FA', FU:'FU', N:'N'})
    
    if out is not None:
        try:
            ssDF.to_csv(out, sep='\t', index=False)
        except:
            print('Can not write parsed summary stats into file:', out)
    print(len(ssDF),'SNPs passed filters and preserved')
    if outDel is not None:
        try:
            ssDFdel.to_csv(outDel, sep='\t', index=False)
        except:
            print('Can not write deleted SNPs into file:', outDel)
        
    return(ssDF)

def recoverGWAS(jointSmryObj, targetSmryObj, z2Filter=np.inf, NFilter=None, maxN=None, outDel=None):
    #Recover Stage II data from joint/meta analysis
    #fixed effect model is used
    alignResult = unvAligner(ssList=[jointSmryObj, targetSmryObj]) 
    tmpJointSmryObj = alignResult['SS'][0]
    tmpTargetSmryObj = alignResult['SS'][1]
    nonnanidx = (tmpTargetSmryObj['SE']>tmpJointSmryObj['SE'])
    tmpSe = np.array([np.nan]*len(tmpJointSmryObj))
    tmpSe[nonnanidx]=1/np.sqrt(1/(tmpJointSmryObj['SE'][nonnanidx]**2)-1/(tmpTargetSmryObj['SE'][nonnanidx]**2))
    tmpBeta = np.array([np.nan]*len(tmpJointSmryObj))
    tmpBeta[nonnanidx]=(tmpJointSmryObj['BETA'][nonnanidx]/(tmpJointSmryObj['SE'][nonnanidx]**2)-tmpTargetSmryObj['BETA'][nonnanidx]/(tmpTargetSmryObj['SE'][nonnanidx]**2))*(tmpSe[nonnanidx]**2)
    tmpN = tmpJointSmryObj['N']-tmpTargetSmryObj['N']
    tmpJointSmryObj['BETA']=tmpBeta
    tmpJointSmryObj['SE']=tmpSe
    tmpJointSmryObj['N']=tmpN
    invalidIdx = (tmpJointSmryObj['SE']<=0) | (np.isnan(tmpJointSmryObj['SE'])) | (np.isinf(tmpJointSmryObj['SE'])) | (tmpJointSmryObj['BETA']==0) | (np.isnan(tmpJointSmryObj['BETA'])) | (np.isinf(tmpJointSmryObj['BETA'])) | (tmpJointSmryObj['N']<=0)
    tmpJointSmryObj = tmpJointSmryObj[~invalidIdx]
    # targetSmryObj = tmpTargetSmryObj[~invalidIdx]
    # ssDFdelTarget = pd.DataFrame()
    # ssDFdelTarget = pd.concat([ssDFdelTarget,targetSmryObj.iloc[invalidIdx,:]])
    idx1, pos1 = strUtils.listInDict(jointSmryObj['SNP'].tolist(), pd.Series(range(len(targetSmryObj)),index=targetSmryObj['SNP']).to_dict())
    idx2, pos2 = strUtils.listInDict(jointSmryObj['SNP'].tolist(), pd.Series(range(len(tmpJointSmryObj)),index=tmpJointSmryObj['SNP']).to_dict())
    gwasSmryObj = jointSmryObj.copy()
    # if removeAll:
        # idx2 = []
        # pos2 = []
    gwasSmryObj.iloc[idx2,:]=tmpJointSmryObj.iloc[pos2,:]
    idx_diff = np.array([i for i in idx1 if i not in idx2])
    ssDFdelGWAS = pd.DataFrame()
    if len(idx_diff)>0:
        invalidNum = len(idx_diff)
        print(invalidNum,'SNP(s) in target array unable to recover GWAS summary statistics')
        ssDFdelGWAS = pd.concat([ssDFdelGWAS,jointSmryObj.iloc[idx_diff,:]])
        included = np.setdiff1d(np.arange(len(jointSmryObj)), idx_diff, assume_unique=True)
        gwasSmryObj = gwasSmryObj.iloc[included,:]
        # jointSmryObj = jointSmryObj.iloc[included,:]
   
    # print(len(jointSmryObj),'SNPs preserved in the joint analysis')
    Z = gwasSmryObj['BETA']/gwasSmryObj['SE']
    invalidIdx = (Z**2 >=z2Filter)  
    invalidNum = np.sum(invalidIdx)
    if invalidNum > 0:
        print('Additional',invalidNum,'SNP(s) with z^2>=',z2Filter)
        ssDFdelGWAS = pd.concat([ssDFdelGWAS,gwasSmryObj[invalidIdx]])
        gwasSmryObj = gwasSmryObj[~invalidIdx]
    
    if maxN is not None:
        invalidIdx = (gwasSmryObj['N']>(maxN))
        invalidNum = np.sum(invalidIdx)
        if invalidNum > 0:
            print(invalidNum,'SNP(s) with sample size larger than N_max=',maxN)
            ssDFdelGWAS = pd.concat([ssDFdelGWAS,gwasSmryObj[invalidIdx]])
            gwasSmryObj = gwasSmryObj[~invalidIdx]

    print('Maximum sample size:', gwasSmryObj['N'].max())
    N90 = np.quantile(gwasSmryObj['N'], 0.9)
    print('90% quantile of sample size:', N90)
    if NFilter is not None:
        invalidIdx = (gwasSmryObj['N']<(NFilter*N90)) | (np.isinf(gwasSmryObj['N'])) #| (np.isnan(gwasSmryObj['N']))
    else:
        invalidIdx =  (np.isinf(gwasSmryObj['N'])) #| (np.isnan(gwasSmryObj['N']))

    invalidNum = np.sum(invalidIdx)
    if invalidNum > 0:
        print(invalidNum,'SNP(s) with sample size less than','{:.2f}'.format(NFilter),'of 90% sample size quantile')
        ssDFdelGWAS = pd.concat([ssDFdelGWAS,gwasSmryObj[invalidIdx]])
        gwasSmryObj = gwasSmryObj[~invalidIdx]
    
    print(len(gwasSmryObj),'SNPs preserved in stage I GWAS')

    if outDel is not None:
        try:
            ssDFdelGWAS.to_csv(outDel, sep='\t', index=False)
        except:
            print('Can\'t write deleted SNPs into file:', outDel)
    return gwasSmryObj

def getInfoFromStore(LDfile, bid, useFilter, snpDict, idx=None):
    with pd.HDFStore(LDfile, 'r') as store:
        try:
            snpInfo = store.get('SNPINFO'+str(bid))
        except:
            print('Unable to access SNP information of Block',bid)
            return 0
        try:
            LD = store.get('LD'+str(bid))
        except:
            print('Unable to access LD information of Block',bid)
            return 0
    
        #Formatting CHR
        # pattern = re.compile(r'^(?i:chr?)?(\w+)')
        # vmatch = np.vectorize(lambda x: pattern.match(x).group(1))
        # snpInfo['CHR'] = np.char.lower(vmatch(snpInfo['CHR']))
        #Change A1 and A2 to lower case for easily matching
        # snpInfo['SNP'] = snpInfo['SNP'].str.lower()
        # snpInfo['A1'] = snpInfo['A1'].str.lower()
        # snpInfo['A2'] = snpInfo['A2'].str.lower()
        LD = LD.to_numpy()
        if idx is not None:
            if len(idx)==0: return 0 
            snpInfo = snpInfo.iloc[idx,:]
            LD = LD[np.ix_(idx, idx)]
        # else:
            # idx = np.arange(len(snpInfo))
        snpNum0 = len(snpInfo)
        if useFilter:
            idx2, pos2 = strUtils.listInDict(snpInfo['SNP'].tolist(), snpDict)
            if len(idx2)==0: return snpNum0
            snpInfo = snpInfo.iloc[idx2] 
            LD = LD[np.ix_(idx2, idx2)]
            # idx = idx[idx2]
    # return snpInfo, LD, idx, snpNum0
    return snpInfo, LD, snpNum0

def refParser(LDfile, snpListFile=None, out=None, thread=-1, complevel=0):
    if isinstance(LDfile, str):
        try:
            store = pd.HDFStore(LDfile, 'r')
        except:
            print('Unable to read LD file:',LDfile)
            return

        isPointer = True
        idxlist = None
        try:
            newLDfile = store.get('FILE')[0]
        except:
            isPointer = False
        if isPointer:
            try:
                idxlist = store.get('IDXLIST').to_list()
            except:
                idxlist = None
            store.close()
            try:
                newStore = pd.HDFStore(newLDfile, 'r')
            except:
                print('Unable to read original LD file:',newLDfile)
                return
        else:
            newStore = store
            newLDfile = LDfile
    else:
        try:
            newLDfile = LDfile['FILE'][0]
        except:
            isPointer = False
        if isPointer:
            try:
                idxlist = LDfile['IDXLIST']
            except:
                idxlist = None
            try:
                newStore = pd.HDFStore(newLDfile, 'r')
            except:
                print('Unable to read original LD file:',newLDfile)
                return
        else:
            print('PRSparser.refParser: Unable to parse variable LDfile')
            return
    
    total = newStore.get('TOTAL')
    indNum = newStore.get('INDNUM')
    # try:
    blockID = newStore.get('BID').to_numpy()
    # except KeyError:
        # keyList = newStore.keys() 
        # pattern = re.compile(r'^/LD(\d+)')
        # blockID = []
        # for key in keyList:
            # match = pattern.match(key)
            # if match is not None:
                # blockID.append(int(match.group(1)))
        # blockID.sort()
        # newStore.put('BID', pd.Series(blockID))
    
    snpDict = {}
    if snpListFile is not None:
        try:
            with open(snpListFile,'r') as f:
                tmpSnplist = [i.strip() for i in f.readlines() if len(i.strip())!=0]
        except IOError:
            print("Can not read snplist file:", snpListFile)
            print('Load all SNPs in the LD panel')
        isHead = False
        for i in range(len(tmpSnplist)):
            tmpID = tmpSnplist[i].split()[0]
            if i==0:
                if tmpID in ('SNP','snp','SNPID','snpID','ID','id','rsID','rsid'): 
                    isHead=True
                    continue
            if isHead: snpDict[tmpID] = i-1
            else: snpDict[tmpID] = i
    
    useFilter = (len(snpDict)>0)
    if useFilter:
        print(len(snpDict),'SNPs in the snplist file')
    if out is not None:
        if useFilter:
            try:
                newStore_out = pd.HDFStore(out, 'w', complevel=complevel)
            except:
                print('Unable to create file', out)
                out=None
        else:
            print('No need to update LD file')
    newBlockID = []
    snpInfoList = []
    Rlist = []
    print(len(blockID),'blocks in the panel')
    totalSNP0 = 0
    totalSNP1 = 0
    try:
        thread = round(float(thread))
    except ValueError:
        print('PRSparser.refParser: thread must be a numeric value')
        thread = 0
    if thread!=1:
        cpuNum = multiprocessing.cpu_count()
        if thread<=0: #Default setting; using total cpu number
            thread = cpuNum
        print(cpuNum, 'CPUs detected, using', thread, 'thread(s) for parsing reference panel...')

    if thread>1:
        pool = multiprocessing.Pool(processes = thread)
        tmpResults = []
        for i in range(len(blockID)):
            bid = blockID[i]
            if idxlist is None:
                tmpResults.append(pool.apply_async(getInfoFromStore, args=(newLDfile, bid, useFilter, snpDict)))
            else:
                tmpResults.append(pool.apply_async(getInfoFromStore, args=(newLDfile, bid, useFilter, snpDict, idxlist[i])))
        
    reportgap = int(np.ceil(len(blockID)/10))
    for i in range(len(blockID)): 
        bid = blockID[i]
        if thread == 1:
            try:
                snpInfo = newStore.get('SNPINFO'+str(bid))
            except:
                print('\nUnable to access SNP information of Block',bid)
                continue
            try:
                LD = newStore.get('LD'+str(bid))
            except:
                print('\nUnable to access LD information of Block',bid)
                continue
    
            #Formatting CHR
            # pattern = re.compile(r'^(?i:chr?)?(\w+)')
            # vmatch = np.vectorize(lambda x: pattern.match(x).group(1))
            # snpInfo['CHR'] = np.char.lower(vmatch(snpInfo['CHR']))
            #Change A1 and A2 to lower case for easily matching
            # snpInfo['SNP'] = snpInfo['SNP'].str.lower()
            # snpInfo['A1'] = snpInfo['A1'].str.lower()
            # snpInfo['A2'] = snpInfo['A2'].str.lower()
            LD = LD.to_numpy()
            if idxlist is not None:
                idx = idxlist[i]
                if len(idx)==0: continue 
                snpInfo = snpInfo.iloc[idx,:]
                LD = LD[np.ix_(idx, idx)]
            totalSNP0 += len(snpInfo)
            if useFilter:
                idx, pos = strUtils.listInDict(snpInfo['SNP'].tolist(), snpDict)
                if len(idx)==0: continue
                snpInfo = snpInfo.iloc[idx] 
                LD = LD[np.ix_(idx, idx)]
        else:
            tmpResult = tmpResults[i].get()
            try:
                snpInfo = tmpResult[0]
                LD = tmpResult[1]
                totalSNP0 += tmpResult[2]
            except:
                totalSNP0 += tmpResult
                continue
        
        snpInfoList.append(snpInfo)
        Rlist.append(LD)
        newBlockID.append(bid)
        totalSNP1 += len(snpInfo)
        if (out is not None) and useFilter:
            newStore_out.put('SNPINFO'+str(bid),value=snpInfo, complevel=complevel, format='fixed')
            newStore_out.put('LD'+str(bid), value=pd.DataFrame(data=LD), complevel=complevel, format='fixed')
        if (i%reportgap == 0) or (i==len(blockID)-1): 
            print('Complete',int(np.ceil(i/reportgap))*10,'%', end='\r', flush=True)
    if thread >1:
        pool.close()
        #Call close function before join, otherwise error will raise. No process will be added to pool after close.
        #Join function: waiting the complete of subprocesses
        pool.join()
    newStore.close()
    print('')
    if (out is not None) and useFilter:
        newStore_out.put('BID',value=pd.Series(newBlockID), complevel=complevel)
        newStore_out.put('TOTAL',value=total, complevel=complevel)
        newStore_out.put('INDNUM', value=indNum, complevel=complevel)
        newStore_out.close()
    print(totalSNP0,'SNPs parsed in the panel')
    if useFilter:
        print('After filtering,',totalSNP1,'SNPs remaining...')
    return({'BID':newBlockID,'SNPINFO':snpInfoList,'LD':Rlist,'TOTAL':int(total.iloc[0]), 'INDNUM': int(indNum.iloc[0])}, newLDfile)

def genoParser(bfile=None, bed=None, bim=None, fam=None, out=None, complevel=9, thread=-1):
    if (bfile is None) and (bim is None) and (fam is None):
        return
    if bfile is not None:
        if bed is None:
            bed = bfile + '.bed'
        if bim is None:
            bim = bfile + '.bim'
        if fam is None:
            fam = bfile + '.fam'
    if out is not None:
        try:
            store = pd.HDFStore(out, 'w', complevel=complevel)
        except:
            print('Unable to create hdf5 file',out)
            out = None
   
    # print('Start loading SNP information...')
    if bim is None:
        snpInfo = pd.Series(np.nan)
    else:
        try:
            # with open(bim,'r') as f:
                # snpInfo = [i.strip() for i in f.readlines() if len(i.strip())!=0]
            snpInfo = pd.read_table(bim, sep='\t|\s+',names=['CHR','RAW_SNP','GD','BP','RAW_A1','RAW_A2'], dtype={'RAW_SNP':str,'CHR':str,'RAW_A1':str,'RAW_A2':str}, engine='python')
        except:
            print("Can not read SNP Info file:", bim)
            return
  
        snpNum = len(snpInfo)
        print(snpNum,'SNPs.')
        #Formatting CHR
        pattern = re.compile(r'^(?i:chr?)?(\w+)')
        vmatch = np.vectorize(lambda x: pattern.match(x).group(1))
        snpInfo['CHR'] = np.char.lower(vmatch(snpInfo['CHR']))
        #Change A1 and A2 to lower case for easily matching
        snpInfo['SNP'] = snpInfo['RAW_SNP'].str.lower()
        snpInfo['A1'] = snpInfo['RAW_A1'].str.lower()
        snpInfo['A2'] = snpInfo['RAW_A2'].str.lower()

    # print('Start loading individual information...')
    if fam is None:
        indInfo = pd.Series(np.nan)
    else:
        try:
            # with open(fam,'r') as f:
                # indInfo = [i.strip() for i in f.readlines() if len(i.strip())!=0]  
            indInfo = pd.read_table(fam, sep='\t|\s+',names=['FID','IID','PID','MID','SEX','PHE'], engine='python')
        except:
            print("Can not read Individual Info file:", fam)
            exit()
  
        indNum = len(indInfo)
        print(indNum,'individuals.')
 
    if (bed is None) or (bim is None) or (fam is None):
        genotype = np.nan
        flipInfo = np.nan
    else:
        genotype = plink2SnpReader.getSnpReader(bed=bed, bim=bim, fam=fam, thread=thread)
        flipInfo = np.array([False]*snpNum)
    
    if out is not None:
        genotypePd = pd.Series(genotype)
        store.put('SNPINFO', value=snpInfo, complevel=complevel, format='fixed')
        store.put('INDINFO', value=indInfo, complevel=complevel, format='fixed')
        warnings.filterwarnings("ignore")
        store.put('GENOTYPE', value=genotypePd, complevel=complevel, format='fixed')
        warnings.resetwarnings()
        store.put('FLIPINFO', value=pd.Series(flipInfo), complevel=complevel, format='fixed')
        store.close()

    return({'SNPINFO': snpInfo, 'INDINFO': indInfo, 'GENOTYPE': genotype, 'FLIPINFO': flipInfo}) 

def removeTmp(bfile=None, bed=None, bim=None, fam=None):
    plink2SnpReader.removeTmp(bfile=bfile, bed=bed, bim=bim, fam=fam)

def annoParser(annoFile, SNP='SNP', CHR='CHR',BP='BP', dataCol=None, skip=0, comment='#', out=None, complevel=9):
    try:
        with open(annoFile,'r') as fp:
            i = 0
            for line in fp:
                line = line.strip()
                if i == skip:
                    if (line == '') or (line[0]==comment): continue #ignore commented and empty lines
                    header = line.split()
                    break
                i+=1
    except:
        print('Unable to read functional annotation data from File:', annoFile)
        return
    usefulHeader = []
    usefulHeaderIdx = []
    if (SNP not in header) and ((CHR not in header) or (BP not in header)):
        print('Both SNP ID and (Chromosome & Base position) can not be located')
        return
    if SNP in header:
        usefulHeader.append(SNP)
        usefulHeaderIdx.append(header.index(SNP))
    if CHR in header: 
        usefulHeader.append(CHR)
        usefulHeaderIdx.append(header.index(CHR))
    if BP in header: 
        usefulHeader.append(BP)
        usefulHeaderIdx.append(header.index(BP))
    
    if dataCol is None:
        #Annotation data starting from column (max(index(SNP, CHR, BP))+1) till the last
        dataColIdx = list(range(max(usefulHeaderIdx)+1,len(header)))
    elif isinstance(dataCol,int):
        dataColIdx = list(range(dataCol-1, len(header)))
    elif isinstance(dataCol, list):
        dataColIdx = []
        for i in dataCol:
            if isinstance(i, int) and i<len(header):
                dataColIdx.append(i)
            elif isinstance(i, str) and (i in header):
                dataColIdx.append(header.index(i))
    else:
        print('Invalid value of dataCol, it should be either an integer/List/None(Default)')
        return

    try:
        annoDF = pd.read_table(annoFile, sep='\t|\s+', skiprows=skip, comment=comment, dtype={SNP:str,CHR:str}, engine='python')
    except:
        print("Can not correctly read annotation file:", annoFile)
        return
    
    snpInfo = annoDF.iloc[:,usefulHeaderIdx].copy()
    #Formatting CHR
    pattern = re.compile(r'^(?i:chr?)?(\w+)')
    vmatch = np.vectorize(lambda x: pattern.match(x).group(1))
    if CHR in header:
        snpInfo[CHR] = np.char.lower(vmatch(snpInfo[CHR]))
    snpInfo[SNP] = snpInfo[SNP].str.lower()
    snpInfo = snpInfo.rename(columns={SNP:'SNP',CHR:'CHR',BP:'BP'})

    annoData = annoDF.iloc[:,dataColIdx].copy()
    if out is not None:
        try:
            store = pd.HDFStore(out, 'w', complevel=complevel)
            store.put('SNPINFO',value=snpInfo, complevel=complevel, format='fixed')
            store.put('ANNODATA', value=annoData, complevel=complevel, format='fixed')
            store.close()
        except:
            print('Unable to create hdf5 file',out)
    print(len(snpInfo),'SNPs in the annotation data')
    return({'SNPINFO': snpInfo, 'ANNODATA': annoData})

def adjustRefStore(LDfile, store, outStore, snpListFile=None, thread=-1, complevel=9):
    isPointer = True
    idxlist = None
    try:
        newLDfile = store.get('FILE')[0]
    except:
        isPointer = False
    if isPointer:
        try:
            idxlist = store.get('IDXLIST').to_list()
        except:
            idxlist = None
        store.close()
        try:
            newStore = pd.HDFStore(newLDfile, 'r')
        except:
            print('Unable to read original LD file:',newLDfile)
    else:
        newLDfile = LDfile
        newStore = store
    
    # try:
    blockID = newStore.get('BID').to_numpy()
    # except KeyError:
        # keyList = newStore.keys() 
        # pattern = re.compile(r'^/LD(\d+)')
        # blockID = []
        # for key in keyList:
            # match = pattern.match(key)
            # if match is not None:
                # blockID.append(int(match.group(1)))
        # blockID.sort()
        # newStore.put('BID', pd.Series(blockID))
    
    snpDict = {}
    if snpListFile is not None:
        try:
            with open(snpListFile,'r') as f:
                tmpSnplist = [i.strip() for i in f.readlines() if len(i.strip())!=0]
        except IOError:
            print("Can not read snplist file:", snpListFile)
            print('Load all SNPs in the LD panel')
        isHead = False
        for i in range(len(tmpSnplist)):
            tmpID = tmpSnplist[i].split()[0]
            if i==0:
                if tmpID in ('SNP','snp','SNPID','snpID','ID','id','rsID','rsid'): 
                    isHead=True
                    continue
            if isHead: snpDict[tmpID] = i-1
            else: snpDict[tmpID] = i
    
    useFilter = (len(snpDict)>0)
    if len(snpDict)>0:
        print(len(snpDict),'SNPs in the snplist file')
    
    print(len(blockID),'blocks in the panel')
        
    if useFilter or (idxlist is not None):
        # newIdxlist = []
        # if not savePointer:
        outStore.put('TOTAL',newStore.get('TOTAL'), complevel=complevel)
        outStore.put('INDNUM',newStore.get('INDNUM'), complevel=complevel)
        newBlockID = []
        totalSNP0 = 0
        totalSNP1 = 0
        try:
            thread = round(float(thread))
        except ValueError:
            print('PRSparser.adjustRefStore: thread must be a numeric value')
            thread = 0
        if thread!=1:
            cpuNum = multiprocessing.cpu_count()
            if thread<=0: #Default setting; using total cpu number
                thread = cpuNum
            print(cpuNum, 'CPUs detected, using', thread, 'thread(s) for parsing reference panel...')

        if thread>1:
            pool = multiprocessing.Pool(processes = thread)
            tmpResults = []
            for i in range(len(blockID)):
                bid = blockID[i]
                if idxlist is None:
                    tmpResults.append(pool.apply_async(getInfoFromStore, args=(newLDfile, bid, useFilter, snpDict)))
                else:
                    tmpResults.append(pool.apply_async(getInfoFromStore, args=(newLDfile, bid, useFilter, snpDict, idxlist[i])))
        
        reportgap = int(np.ceil(len(blockID)/10))
        for i in range(len(blockID)): 
            bid = blockID[i]
            if thread == 1:
                try:
                    snpInfo = newStore.get('SNPINFO'+str(bid))
                except:
                    print('\nUnable to access SNP information of Block',bid)
                    continue
                try:
                    LD = newStore.get('LD'+str(bid))
                except:
                    print('\nUnable to access LD information of Block',bid)
                    continue
    
                #Formatting CHR
                # pattern = re.compile(r'^(?i:chr?)?(\w+)')
                # vmatch = np.vectorize(lambda x: pattern.match(x).group(1))
                # snpInfo['CHR'] = np.char.lower(vmatch(snpInfo['CHR']))
                #Change A1 and A2 to lower case for easily matching
                # snpInfo['SNP'] = snpInfo['SNP'].str.lower()
                # snpInfo['A1'] = snpInfo['A1'].str.lower()
                # snpInfo['A2'] = snpInfo['A2'].str.lower()
                LD = LD.to_numpy()
                if idxlist is not None:
                    idx = idxlist[i]
                    if len(idx)==0: 
                        # newIdxlist.append([])
                        continue 
                    snpInfo = snpInfo.iloc[idx,:]
                    LD = LD[np.ix_(idx, idx)]
                # else:
                    # idx = np.arange(len(snpInfo))
                totalSNP0 += len(snpInfo)
                if useFilter:
                    idx2, pos2 = strUtils.listInDict(snpInfo['SNP'].tolist(), snpDict)
                    if len(idx2)==0:
                        # newStore.remove('SNPINFO'+str(bid))
                        # newStore.remove('LD'+str(bid))
                        # newIdxlist.append([])
                        continue
                    else:
                        # newIdxlist.append(idx[idx2])
                        snpInfo = snpInfo.iloc[idx2] 
                        LD = LD[np.ix_(idx2, idx2)]
            else:
                tmpResult = tmpResults[i].get()
                try:
                    snpInfo = tmpResult[0]
                    LD = tmpResult[1]
                    # newIdxlist.append(tmpResult[2])
                    totalSNP0 += tmpResult[2]
                except:
                    # newIdxlist.append([])
                    totalSNP0 += tmpResult
                    # try:
                        # newStore.remove('SNPINFO'+str(bid))
                        # newStore.remove('LD'+str(bid))
                    # except:
                        # pass
                    continue
            # if not savePointer:
            outStore.put('SNPINFO'+str(bid),value=snpInfo, complevel=complevel, format='fixed')
            outStore.put('LD'+str(bid), value=pd.DataFrame(data=LD), complevel=complevel, format='fixed')
            newBlockID.append(bid)
            totalSNP1 += len(snpInfo)
            if (i%reportgap == 0) or (i==len(blockID)-1): 
                print('Complete',int(np.ceil(i/reportgap))*10,'%', end='\r', flush=True)
        
        if thread >1:
            pool.close()
            #Call close function before join, otherwise error will raise. No process will be added to pool after close.
            #Join function: waiting the complete of subprocesses
            pool.join()
   
        print('')
        # if not savePointer:
        outStore.put('BID',value=pd.Series(newBlockID), complevel=complevel)
        # else:
            # outStore.put('FILE', value=pd.Serie(newLDfile), complevel=complevel)
            # outStore.put('IDXLIST', value=pd.Serie(newIdxlist), complevel=complevel)
        print(totalSNP0,'SNPs parsed in the panel')
        if useFilter:
            print('After filtering,',totalSNP1,'SNPs remaining...')
    else:
        outStore.close()
        outStore = pd.HDFStore(newLDfile, 'r')
    return outStore, newLDfile

def refStoreParser(LDfile, store, thread=-1):
    warnings.filterwarnings("ignore")
    isPointer = True
    idxlist = None
    try:
        newLDfile = store.get('FILE')[0]
    except:
        isPointer = False
    if isPointer:
        try:
            idxlist = store.get('IDXLIST').to_list()
        except:
            idxlist = None
        try:
            newStore = pd.HDFStore(newLDfile, 'r')
        except:
            print('Unable to read original LD file:',newLDfile)
    else:
        newLDfile = LDfile
        newStore = store
    
    total = newStore.get('TOTAL')
    indNum = newStore.get('INDNUM')
    # try:
    blockID = newStore.get('BID').to_numpy()
    # except KeyError:
        # keyList = newStore.keys() 
        # pattern = re.compile(r'^/LD(\d+)')
        # blockID = []
        # for key in keyList:
            # match = pattern.match(key)
            # if match is not None:
                # blockID.append(int(match.group(1)))
        # blockID.sort()
        # newStore.put('BID', pd.Series(blockID))
    
    snpInfoList = []
    Rlist = []
    print(len(blockID),'blocks in the panel')
    totalSNP0 = 0
    try:
        thread = round(float(thread))
    except ValueError:
        print('PRSparser.refStoreParser: thread must be a numeric value')
        thread = 0
    if thread!=1:
        cpuNum = multiprocessing.cpu_count()
        if thread<=0: #Default setting; using total cpu number
            thread = cpuNum
        print(cpuNum, 'CPUs detected, using', thread, 'thread(s) for parsing reference panel...')

    if thread>1:
        pool = multiprocessing.Pool(processes = thread)
        tmpResults = []
        for i in range(len(blockID)):
            bid = blockID[i]
            if idxlist is None:
                tmpResults.append(pool.apply_async(getInfoFromStore, args=(newLDfile, bid, False, {})))
            else:
                tmpResults.append(pool.apply_async(getInfoFromStore, args=(newLDfile, bid, False, {}, idxlist[i])))
        
    reportgap = int(np.ceil(len(blockID)/10))
    for i in range(len(blockID)): 
        bid = blockID[i]
        if thread == 1:
            try:
                snpInfo = newStore.get('SNPINFO'+str(bid))
            except:
                print('\nUnable to access SNP information of Block',bid)
                continue
            try:
                LD = newStore.get('LD'+str(bid))
            except:
                print('\nUnable to access LD information of Block',bid)
                continue
    
            #Formatting CHR
            # pattern = re.compile(r'^(?i:chr?)?(\w+)')
            # vmatch = np.vectorize(lambda x: pattern.match(x).group(1))
            # snpInfo['CHR'] = np.char.lower(vmatch(snpInfo['CHR']))
            #Change A1 and A2 to lower case for easily matching
            # snpInfo['SNP'] = snpInfo['SNP'].str.lower()
            # snpInfo['A1'] = snpInfo['A1'].str.lower()
            # snpInfo['A2'] = snpInfo['A2'].str.lower()
            LD = LD.to_numpy()
            if idxlist is not None:
                idx = idxlist[i]
                if len(idx)==0: continue 
                snpInfo = snpInfo.iloc[idx,:]
            totalSNP0 += len(snpInfo)
        else:
            tmpResult = tmpResults[i].get()
            try:
                snpInfo = tmpResult[0]
                LD = tmpResult[1]
                totalSNP0 += tmpResult[2]
            except:
                continue
        snpInfoList.append(snpInfo)
        Rlist.append(LD)
        if (i%reportgap == 0) or (i==len(blockID)-1): 
            print('Complete',int(np.ceil(i/reportgap))*10,'%', end='\r', flush=True)
        
    if thread >1:
        pool.close()
        #Call close function before join, otherwise error will raise. No process will be added to pool after close.
        #Join function: waiting the complete of subprocesses
        pool.join()
    if isPointer:
        newStore.close()
    
    print('')
    print(totalSNP0,'SNPs parsed in the panel')
    warnings.resetwarnings()
    return({'BID':blockID,'SNPINFO':snpInfoList,'LD':Rlist,'TOTAL':int(total.iloc[0]), 'INDNUM': int(indNum.iloc[0])}, newLDfile)

def weightParser(weightFile, SNP='SNP',CHR='CHR', BP='BP',A1='A1', A2='A2', BETAJ='BETAJ', skip=0, comment=None):
    header = []
    firstLine = secondLine = ''
    try:
        with open(weightFile,'r') as fp:
            i = 0
            for line in fp:
                line = line.strip()
                if i == skip:
                    if (line == '') or (line[0]==comment): continue #ignore commented and empty lines
                    header = line.split()
                    firstLine = line
                if i == skip+1:
                    if (line == '') or (line[0]==comment): continue #ignore commented and empty lines
                    secondLine = line
                    break
                i+=1
    except:
        print('Unable to read weights from File:', weightFile)
        return
    sniffer = csv.Sniffer()
    has_header = sniffer.has_header(firstLine+'\n'+secondLine+'\n')
    if has_header:
        usefulHeader = []
        if (SNP not in header) and ((CHR not in header) or (BP not in header)):
            print('Both SNP ID and (Chromosome & Base position) can not be located')
            return
        if (A1 not in header):
            print('A1 can not be located')
            return
        if BETAJ not in header: 
            print('Effect size can not be located')
            return
        if SNP in header: usefulHeader.append(SNP)
        if CHR in header: usefulHeader.append(CHR)
        if BP in header: usefulHeader.append(BP)
        usefulHeader.append(A1)
        if (A2 in header):
            usefulHeader.append(A2)

        usefulHeader.append(BETAJ)
        try:
            ssDF = pd.read_table(weightFile, sep='\t|\s+', dtype={SNP:str, CHR:str, A1: str, A2: str}, usecols=usefulHeader, skiprows=skip, comment= comment, engine='python')
        except:
            print("Could not correctly read weight file:", weightFile)
            return
    else:
        'PLINK score format (No header)'
        'SNP ID    Reference Allele    Score (numeric)'
        header = ['SNP','A1','BETAJ']
        usefulHeader = header
        try:
            ssDF = pd.read_table(weightFile, sep='\t|\s+', header=None, names=header, usecols=[0,1,2], skiprows=skip, comment=comment, engine='python')
        except:
            print("Could not correctly read weight file:", weightFile)
            return

    #Formatting CHR
    pattern = re.compile(r'^(?i:chr?)?(\w+)')
    vmatch = np.vectorize(lambda x: pattern.match(x).group(1))
    if CHR in header:
        ssDF[CHR] = np.char.lower(vmatch(ssDF[CHR]))
    
    if SNP in header:
        ssDF[SNP] = ssDF[SNP].str.lower()
    ssDF[A1] = ssDF[A1].str.lower()
    if A2 in header:
        ssDF[A2] = ssDF[A2].str.lower()

    print(len(ssDF),'SNPs in the file.')
    
    invalidNum1 = invalidNum2 = 0
    if SNP in header:
        invalidIdx =  pd.Index(ssDF[SNP]).duplicated(keep='first')
        invalidNum1 = np.sum(invalidIdx)
        if invalidNum1 > 0:
            print(invalidNum1,'SNP(s) with duplicated SNP ID')
            ssDF = ssDF[~invalidIdx]

    if (CHR in header) and (BP in header):
        invalidIdx =  pd.Index(ssDF[CHR]+':'+ssDF[BP].astype(str)).duplicated(keep='first')
        invalidNum2 = np.sum(invalidIdx)
        if invalidNum2 > 0:
            print(invalidNum2,'SNP(s) having identical positions with others')
            ssDF = ssDF[~invalidIdx]

    ssDF = ssDF[usefulHeader]
    ssDF = ssDF.rename(columns={SNP: 'SNP',CHR:'CHR',BP:'BP',A1:'A1',A2:'A2',BETAJ:'BETA'})
    if invalidNum1>0 or invalidNum2>0:
        print(len(ssDF),'SNPs preserved')
        
    return(ssDF)

