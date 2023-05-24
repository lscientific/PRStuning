'''
Evaluate PRS
'''
import strUtils
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn import linear_model, metrics
import compare_auc_delong_xu
import matplotlib.pyplot as plt
import warnings
from PRScoring import selectInd, getIndIndex

def indParser(indFile, dataCol=None, FID='FID', IID='IID', skip=0, defaultCol='SCORE', tillEnd=True,comment=None):
    #if dataCol is a integer and tillEnd=True, all columns beyond dataCol are selected.
    try:
        with open(indFile,'r') as fp:
            i = 0
            for line in fp:
                line = line.strip()
                if i == skip:
                    if (line == '') or (line[0]==comment): continue #ignore commented and empty lines
                    header = line.split()
                    break
                i+=1
    except:
        print('Unable to read file:', indFile)
        return
        
    if (FID in header) and (IID in header): 
        #header exists
        if dataCol is None:
            #data content starting from column (max(index(FID, IID))+1) till the last
            if defaultCol in header:
                dataColIdx=[header.index(defaultCol)]
            else:
                dataColIdx = list(range(max([header.index(FID), header.index(IID)])+1,len(header)))
        elif isinstance(dataCol,int):
            if tillEnd:
                dataColIdx = list(range(dataCol-1, len(header)))
            else:
                dataColIdx = [dataCol-1]
        elif isinstance(dataCol, list):
            dataColIdx = []
            for i in dataCol:
                if isinstance(i, int) and i<len(header):
                    dataColIdx.append(i)
                elif isinstance(i, str) and (i in header):
                    dataColIdx.append(header.index(i))
        elif isinstance(dataCol, str) and (dataCol in header):
            dataColIdx=[header.index(dataCol)]
        else:
            print('Invalid value of dataCol, it should be either an integer/List/None(Default)')
            return

        try:
            dataDF = pd.read_table(indFile, sep='\t|\s+', skiprows=skip, engine='python', comment=comment)
        except:
            print("Unable correctly to read individual-level data file:", indFile)
            return
        dataDF = dataDF.rename(columns={FID:'FID',IID:'IID'})
    else:
        #No header
        if dataCol is None:
            #data content starting from column 2 till the last
            dataColIdx = list(range(2,len(header)))
        elif isinstance(dataCol,int):
            if tillEnd:
                dataColIdx = list(range(dataCol-1, len(header)))
            else:
                dataColIdx = [dataCol-1]
        elif isinstance(dataCol, list):
            dataColIdx = []
            for i in dataCol:
                if isinstance(i, int) and i<len(header):
                    dataColIdx.append(i)
        else:
            print('Invalid value of dataCol, it should be either an integer/List/None(Default)')
            return

        try:
            dataDF = pd.read_table(indFile, sep='\t|\s+', header=None, skiprows=skip, engine='python', comment=comment)
        except:
            print("Unable correctly to read individual-level data file:", indFile)
            return
        dataDF.columns = dataDF.columns.astype(str)
        dataDF.columns.values[0] = 'FID'
        dataDF.columns.values[1] = 'IID'

    dataColIdx = [dataDF.columns.get_loc('FID'), dataDF.columns.get_loc('IID')]+[ii for a,ii in enumerate(dataColIdx) if ii not in dataColIdx[:a]]#list(set(dataColIdx))
    dataDF = dataDF.iloc[:, dataColIdx].copy()
    return dataDF

def phenoParser(genoObj=None, phenoFile=None, PHE='PHE', mpheno=1, missing=-9, FID='FID', IID='IID', skip=0):
    if (phenoFile is None) and (genoObj is not None):
        phenoDF = genoObj['INDINFO'][['FID','IID','PHE']]
        phenoDF = phenoDF.loc[phenoDF['PHE']!=missing,:].reset_index(drop=True).copy()
    elif (phenoFile is not None):
        phenoDF = indParser(phenoFile, dataCol=[PHE, 1+mpheno], FID=FID, IID=IID, skip=skip, tillEnd=False)
        phenoDF = phenoDF.iloc[np.arange(len(phenoDF))[phenoDF.iloc[:,2]!=missing],0:3]
        if genoObj is not None:
            idx, pos=strUtils.listInDict(getIndIndex(genoObj['INDINFO']), pd.Series(range(len(phenoDF)), index=getIndIndex(phenoDF)))
            phenoDF = phenoDF.iloc[pos, :].reset_index(drop=True).copy()
    else:
        phenoDF = None
    
    return phenoDF

def covParser(covFile, dataCol=None, FID='FID', IID='IID', skip=0):
    covDF = indParser(covFile, dataCol=dataCol, FID=FID, IID=IID, skip=skip, tillEnd=True)
    covDF = pd.concat([covDF.iloc[:,0:2],pd.get_dummies(data=covDF.iloc[:,2:], drop_first=True)], axis=1)
    return covDF

def overlapIndDf(indDfList):
    newIndDfList = [indDfList[0].copy()]
    for i in range(1,len(indDfList)):
        idx, pos=strUtils.listInDict(getIndIndex(newIndDfList[i-1]), pd.Series(range(len(indDfList[i])), index=getIndIndex(indDfList[i])).to_dict())
        newIndDfList.append(indDfList[i].iloc[pos,:].reset_index(drop=True).copy())
        for j in range(i):
            newIndDfList[j] = newIndDfList[j].iloc[idx,:].reset_index(drop=True)
    return newIndDfList

def isBinary(phe):
    uniquePhe = np.unique(phe)
    if len(uniquePhe)==2:
        return True
    else:
        return False

def convert01(phe):
    uniquePhe, counts = np.unique(phe, return_counts=True)
    uPheNum = len(uniquePhe)
    # if uPheNum == 1:
        # phe = np.array([0]*len(phe))
    if uPheNum == 2:
        zeroIdx = (phe==uniquePhe[0])
        if counts[0]>=counts[1]:
            phe[zeroIdx] = 0
            phe[~zeroIdx] = 1
        else:
            phe[~zeroIdx] = 0
            phe[zeroIdx] = 1
    return phe

def fit(phe, prs=None, covData=None):
    X = np.ones((len(phe),1))
    varnames = np.array(['int'])
    if prs is not None:
        X = np.hstack([X, np.array([prs]).T])
        varnames = np.append(varnames, ['prs'])
    if covData is not None:
        X = np.hstack([X, covData])
        varnames = np.append(varnames, covData.columns.values)
    if isBinary(phe):
        #Logistic regression
        lr = linear_model.LogisticRegression(fit_intercept=False, penalty='none')
        Xmean = np.mean(X,axis=0)
        Xsd = np.std(X, axis=0)
        Xsd[Xsd==0] = 1.
        XmeanMat = np.outer(np.ones(X.shape[0]), Xmean)
        XsdMat = np.outer(np.ones(X.shape[0]), Xsd)
        if X.shape[1]>1:
            X[:,1:] = (X[:,1:]-XmeanMat[:,1:])/XsdMat[:,1:]
        lr.fit(X, phe)
        coeff = lr.coef_
        if len(coeff)>1:
            coeff[0] -= np.sum(coeff[1:]*Xmean[1:]/Xsd[1:])
            coeff[1:] = coeff[1:]/Xsd[1:]
        pheProb = lr.predict_proba(X)
        rss = metrics.log_loss(phe, pheProb)
    else:
        #Linear regression
        (coeff, rss, _, _) = np.linalg.lstsq(X, phe, rcond=None) 
        rss = rss[0]
    
    phePred = np.ravel(np.dot(X, coeff.T))
    return({'yhat':phePred, 'rss':rss,'coeff':coeff, 'vars':varnames})

def auc(phe, sc, figname=None, fmt='o-r', **kwargs):
    aucVal1 = metrics.roc_auc_score(phe, sc) 
    alpha = 0.95
    aucVal2, auc_cov = compare_auc_delong_xu.delong_roc_variance(phe, sc)

    se = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

    ci = stats.norm.ppf(lower_upper_q,loc=aucVal2,scale=se)
    ci[ci < 0] = 0
    ci[ci > 1] = 1

    fpr, tpr, thresholds = metrics.roc_curve(phe, sc)
    fig, ax = plt.subplots()
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Receiver operating characteristic (ROC) curve")
    ax.plot(fpr, tpr, fmt, **kwargs)
    x_vals = np.array(ax.get_xlim())
    ax.plot(x_vals, x_vals, '--')
    if figname is not None:
        fig.savefig(figname)
    return({'auc':aucVal2, 'se':se, 'ci': ci, 'auc1':aucVal1,'fig': fig, 'ax': ax})

def se_rsq(r2, n):
    if n<=60:
        print('Warning: The standard error is estimated based on the Large sample theory (n>60)')
        print('Be cautious when n=',n)
    #See https://stats.stackexchange.com/questions/175026/formula-for-95-confidence-interval-for-r2 for details
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        se = np.sqrt((4*r2*(1-r2)**2*(n-2)**2)/((n**2-1)*(n+3)))
    
    # se for r
    # https://stats.stackexchange.com/questions/196689/expected-value-and-variance-of-sample-correlation
    # https://stats.stackexchange.com/questions/73621/standard-error-from-correlation-coefficient
    # https://stats.stackexchange.com/questions/226380/derivation-of-the-standard-error-for-pearsons-correlation-coefficient/375616#375616
    # se = (1-r2)/n
    return(se) 

def rsq(phe, prs, covData=None):
    r2dict = {}
    n = len(phe)
    #Correlation 
    r = np.corrcoef(prs, phe)[0,1]
    # r = stats.pearsonr(prs,phe)
    r2dict['raw'] = [r**2]
    r2dict['raw'].append(se_rsq(r2dict['raw'][0], n))
    n = len(prs)
    
    pred00 = fit(phe)
    pred10 = fit(phe, prs)
    r2dict['McF'] = [1.-pred10['rss']/pred00['rss']]
    r2dict['McF'].append(se_rsq(r2dict['McF'][0], n))
    if covData is not None:
        pred01 = fit(phe, None, covData)
        pred11 = fit(phe, prs, covData)
        r2dict['McF-adj'] = [1.-pred11['rss']/pred01['rss']]
        r2dict['McF-adj'].append(se_rsq(r2dict['McF-adj'][0], n))
        r2dict['McF-all'] = [1.-pred11['rss']/pred00['rss']]
        r2dict['McF-all'].append(se_rsq(r2dict['McF-all'][0], n))

    r2df = pd.DataFrame.from_dict(r2dict, orient='index', columns=['r2', 'se'])
    # lo, hi = r2val-1.96*se, r2val+1.96*se
    # if lo<0: lo = 0
    # if hi>1: hi = 1
    return(r2df)
    
def strataPlot(phe, sc, strata=9, bins=None, tail=True, figname=None, fmt='o-r', **kwargs):
    if bins is None:
        if tail:
            bins = 100*np.concatenate(([0], np.linspace(0.05, 0.95, strata+1), [1]))
        else:
            bins = 100*np.linspace(0, 1, strata+1)
    
    binNum = len(bins)-1
    label = np.digitize(sc, np.percentile(sc, bins), right=True)
    fig, ax = plt.subplots()
    if isBinary(phe):
        midBinIdx = np.digitize(50, bins, right=True)-1
        n0 = np.array([np.sum(phe[label==(i+1)]==0) for i in range(binNum) ])
        n1 = np.array([np.sum(phe[label==(i+1)]==1) for i in range(binNum) ])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            odds = n1/n0
            y = odds/odds[midBinIdx] #Odds Ratio
            se_logOR = np.sqrt(1/n1+1/n0+1/n1[midBinIdx]+1/n0[midBinIdx])
            se_logOR[midBinIdx] = 0.
            y_ul = np.exp(np.log(y)+1.96*se_logOR)
            y_ll = np.exp(np.log(y)-1.96*se_logOR)
        ax.set_ylabel("Odds Ratio for Score on Phenotype")
    else:
        midBinIdx = np.digitize(50, bins, right=True)-1
        y = np.array([np.mean(phe[label==(i+1)]) for i in range(binNum) ])
        y_ul = np.array([np.quantile(phe[label==(i+1)], 0.975) for i in range(binNum) ])
        y_ll = np.array([np.quantile(phe[label==(i+1)], 0.025) for i in range(binNum) ])
        baseY = y[midBinIdx]
        # baseY_ul = y_ul[0]
        # baseY_ll = y_ll[0]
        y = y-baseY
        y_ul = y_ul-baseY
        y_ll = y_ll-baseY
        # y_ul[0] = 0
        # y_ll[0] = 0
        ax.set_ylabel("Average Phenotype")
    
    x = np.array([(bins[i]+bins[i+1])/2 for i in range(binNum) ])
    ax.set_xlabel("Percentile for Risk Score (%)")
    ax.set_title("Strata Plot of Risk Score")
    ax.set_xticks(x)
    binsInt = np.round(bins).astype(int)
    xlabel = ['[0,'+str(binsInt[1])+']'] +['('+str(binsInt[i+1])+','+str(binsInt[i+2])+']' for i in range(binNum-1)]
    ax.set_xticklabels(xlabel, rotation=45)
    fig.subplots_adjust(bottom=0.2)
    ax.errorbar(x, y, np.vstack((y-y_ll, y_ul-y)), fmt=fmt, capsize=5, **kwargs)

    if figname is not None:
        fig.savefig(figname)
    return fig, ax
