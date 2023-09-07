#!/usr/bin/env python3

import os, time, functools
import numpy as np
import pandas as pd
import PRSparser
import PRSalign
import ldsc
from GWEButils import blockSize, blocking
import PRSeval
from PRScoring import scoringByPlink
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from plinkLD import ldscore

def concat(x):
    '''
    Convert x from 2d list into 1d by concatenating
    '''
    return functools.reduce(lambda a, b: np.concatenate((a,b), axis=-1), x)

def extractRef(refObj, col='LDSC'):
    list2d = []
    for i in range(len(refObj['BID'])):
        list2d.append(refObj['SNPINFO'][i][col])
    aList = concat(list2d)
    return(aList)

def GWEB_prstuning(weight, beta_EB, n0, n1, refObj, ssObj):
    '''
    :param weight: weights of the PRS model to be evaluated
    :param beta_EB: matrix of sampled empirical Bayes beta from the gibbs sampler
    :param Rlist: list of LD matrices
    :param n0: training data sample size for control group
    :param n1: training data sample size for case group
    :param alignResult: aligned object saved to the current path from GWEB.py
    :return: PRStuning AUC
    '''

    Ne = 4 / (1/n0 + 1/n1)  # effective sample size
    n = beta_EB.shape[0] # number of empirical Bayes samples
    Rlist = refObj['LD']
    betaSE = ssObj['SE']
    flist = []
    for bk in range(len(refObj['SNPINFO'])):
        flist.append(refObj['SNPINFO'][bk].loc[:, 'F'])
    bkSize = blockSize(Rlist)
    bkNum = len(bkSize)
    weightBk = blocking(weight, bkSize)
    betaSE_Bk = blocking(betaSE, bkSize)
    betaBk = []
    start = 0
    for bk in range(bkNum):
        end = start + bkSize[bk]
        betaBk.append(beta_EB[:, start:end])
        start = end

    wf_bk = np.zeros(n*bkNum).reshape((n, bkNum))  # sample_size * bkNum
    s2 = 0
    for bk in range(bkNum):
        SE_bk = np.sqrt(2 * flist[bk] * (1 - flist[bk]))
        R_beta = np.matmul(Rlist[bk], np.transpose(np.matmul(betaBk[bk], np.diag(1/betaSE_Bk[bk]))))
        f_bk = np.matmul(np.diag(SE_bk), R_beta)
        f_bk = f_bk / np.sqrt(Ne)
        wf_bk[:, bk] = np.matmul(np.array(weightBk[bk]), f_bk)
        wSE_bk = np.array(weightBk[bk] * SE_bk)
        s2 += np.matmul(np.matmul(wSE_bk, Rlist[bk]), np.transpose(wSE_bk))

    delta_samples = 2 * (np.array(np.sum(wf_bk, axis=1) / np.sqrt(2*s2))).flatten()
    delta_samples = [np.abs(delta) for delta in delta_samples]
    AUC_samples = [norm.cdf(i) for i in delta_samples]
    AUC_prstuning = np.mean(AUC_samples)
    return AUC_prstuning

def main(p_dict):
    if p_dict['iprefix'] is not None:
        if os.path.exists(p_dict['iprefix'] + 'ssf.txt'):
            p_dict['ssf'] = p_dict['iprefix'] + 'ssf.txt'
        if os.path.exists(p_dict['iprefix'] + 'weight.txt'):
            p_dict['weight'] = p_dict['iprefix'] + 'weight.txt'
        if os.path.exists(p_dict['iprefix'] + 'ref.h5'):
            p_dict['ref'] = p_dict['iprefix'] + 'ref.h5'
        if os.path.exists(p_dict['iprefix'] + 'geno.h5'):
            p_dict['h5geno'] = p_dict['iprefix'] + 'geno.h5'

    print('Parsing GWAS summary statistics from', p_dict['ssf'])
    smryObj = PRSparser.smryParser(p_dict['ssf'])
    if smryObj is None:
        print('Can not read GWAS summary statistics:', p_dict['ssf'])
        return

    print('Parsing PRS weight file from', p_dict['weight'])
    weightObj = pd.read_csv(p_dict['weight'], delimiter="\t")
    if weightObj is None:
        print('Can not PRS weight file :', p_dict['ssf'])
        return

    print('Reading reference panel from', p_dict['ref'])
    refObj, newLDfile = PRSparser.refParser(p_dict['ref'], snpListFile=None, thread=p_dict['thread'])

    if p_dict['h5geno'] is not None:
        print('Parsing testing genotype data from', p_dict['h5geno'])
        genoStore = pd.HDFStore(p_dict['h5geno'], 'r')
        genoObj = {'SNPINFO': genoStore.get('SNPINFO'), 'INDINFO': genoStore.get('INDINFO'),
                   'GENOTYPE': genoStore.get('GENOTYPE')[0], 'FLIPINFO': genoStore.get('FLIPINFO').to_numpy()}
        genoStore.close()
        print(len(genoObj['SNPINFO']), 'SNPs')

        if PRSalign.isPdNan(genoObj['INDINFO']) or PRSalign.isPdNan(genoObj['GENOTYPE']):
            print('No individual in the genotype dataset')
        else:
            print(len(genoObj['INDINFO']), 'Individuals')
        del genoStore
        # gc.collect()
    else:
        print('Parsing testing genotype data from', p_dict['geno'])
        genoObj = PRSparser.genoParser(bfile=p_dict['geno'])

    if p_dict['aligned']:
        if genoObj is None:
            alignResult = {'SS': smryObj, 'REF': refObj}
        else:
            alignResult = {'SS': smryObj, 'GENO': genoObj, 'REF': refObj}

    else:
        print('Start aligning all files...')
        # overlap weight and ssf
        SNPs_intersect = pd.DataFrame({'SNP': list(set(weightObj.loc[:, "SNP"]).intersection(smryObj.loc[:, "SNP"]))})
        weightObj = weightObj.merge(SNPs_intersect, how='inner', on='SNP')
        smryObj = smryObj.merge(SNPs_intersect, how='inner', on='SNP')

        alignRefFile = os.path.join(p_dict['dir'], 'align_ref.h5')
        if genoObj is None:
            # Without testing genotype data
            alignResult = PRSalign.alignSmry2RefPointer(smryObj, refObj, newLDfile, annoObj=None,
                                                        outSS=os.path.join(p_dict['dir'], 'align_ssf.txt'),
                                                        outRef=alignRefFile,
                                                        outAnno=os.path.join(p_dict['dir'], 'align_anno.txt'),
                                                        byID=True, complevel=9)
            refObj = alignResult['REF']
        else:
            alignResult = PRSalign.alignSmryGeno2RefPointer(smryObj, genoObj, refObj, newLDfile, annoObj=None,
                                                            outSS=os.path.join(p_dict['dir'], 'align_ssf.txt'),
                                                            outGeno=os.path.join(p_dict['dir'], 'align_geno.h5'),
                                                            outRef=alignRefFile,
                                                            outAnno=os.path.join(p_dict['dir'], 'align_anno.h5'),
                                                            byID=True, complevel=9)
            refObj = alignResult['REF']
        # align weightObj with ssf
        smryObj = alignResult['SS']
        smryObj['order'] = range(smryObj.shape[0])
        weightObj = weightObj.merge(smryObj.loc[:, ["SNP", 'order']], how='inner', on='SNP')
        weightObj = weightObj.set_index("order").sort_index()
        weightObj.to_csv(os.path.join(p_dict['dir'], 'align_weight.txt'), sep='\t', index=False)

        print('After alignment,', len(alignResult['SS']), 'SNPs remaining')

    if not p_dict['align-only']:
        p_dict['n'] = p_dict['n0'] + p_dict['n1']
        if 'N' in alignResult['SS'].columns:
            rawN = p_dict['n']
            N90 = np.quantile(alignResult['SS']['N'], 0.9)
            p_dict['n'] = int(N90)
            print('Use 90% quantile of SNP-level sample size:', p_dict['n'])

        z = alignResult['SS']['BETA'] / alignResult['SS']['SE']
        inf_gc = np.median(z ** 2) / 0.455
        print('Genomic Control:', "{:.4g}".format(inf_gc))
        alignResult['REF'] = PRSalign.updateLDSC(alignResult['REF'])
        ldscVal = extractRef(alignResult['REF'], col='LDSC')
        if p_dict['n'] != 0:
            ldscResult = ldsc.calH2_2stage(z, ldscVal, p_dict['n'])
            print('Estimated h2 from LDSC:', "{:.4g}".format(ldscResult['h2']))
            inf_ldsc = 1. + p_dict['n'] * ldscResult['a']
            totalNormVar = ldscResult['h2'] * p_dict['n'] / len(z)
            print('Intercept from LDSC:', "{:.4g}".format(inf_ldsc))
        else:
            # print('Sample size of summary statistics is not provided and LDSC is not run for adjusting confounding inflation!')
            newLDSCresult = np.linalg.lstsq(np.vstack((np.ones(len(z)), ldscVal)).T, z ** 2, rcond=None)[0]
            inf_ldsc = newLDSCresult[0]
            totalNormVar = newLDSCresult[1]
            print('Estimated confounding inflation factor:', "{:.4g}".format(inf_ldsc))

        alignResult['SS']['BETA'] = alignResult['SS']['BETA'] / np.sqrt(inf_ldsc)
        adj = inf_ldsc

        print('Start estimation...')

        import GWEButils_cy as GWEButils

        estResultList = []

        # Add 0.1 to LD variance part for stability if not homo
        if (not p_dict['homo']):
            penality = 0.1
            # The penality will increase when most SNPs genotyped on smaller sample size
            if 'N' in alignResult['SS'].columns:
                if rawN == 0:
                    rawN = alignResult['SS']['N'].max()
                if N90 / rawN <= (2 / 3):
                    penality = 0.2
            for bk in range(len(alignResult['REF']['LD'])):
                alignResult['REF']['LD'][bk] = (1 - penality) * alignResult['REF']['LD'][bk] + penality * np.identity(
                    alignResult['REF']['LD'][bk].shape[0])

        # Start iteration
        K = 1
        print('======Without annotation data, K=', K, '======', sep='')
        for iterRep in range(5):
            startTime1 = time.time()
            if iterRep == 4:
                momentEst = True
            else:
                momentEst = False
            try:
                estResult = GWEButils.ebEst(alignResult['SS']['BETA'], alignResult['SS']['SE'],
                                            alignResult['REF']['LD'], ldscVal, N=p_dict['n'], K=K,
                                            thread=p_dict['thread'], momentEst=momentEst, adj=adj)
            except Exception as e:
                print('Error:' + str(e))
                print('Rerunning the estimation')
                continue
            if estResult is not None:
                print('MCMC completed! MCMC time elapsed:', time.time() - startTime1, 's')
                break
            else:
                # Increasing penality
                for bk in range(len(alignResult['REF']['LD'])):
                    alignResult['REF']['LD'][bk] += 0.1 * np.identity(alignResult['REF']['LD'][bk].shape[0])

        result = None
        pars_prstuning = None
        AUC_prstuning = None
        if estResult is not None:
            estResultList.append(estResult)
            paramDF = pd.DataFrame({'pi': estResult['piEst'], 'sigma2': estResult['sigma2Est']})
            paramFile = os.path.join(p_dict['dir'], 'param.txt')
            paramDF.to_csv(paramFile, sep='\t')
            print('Estimated parameter values are saved to', paramFile)
            np.savetxt(os.path.join(p_dict['dir'], 'beta_sample.txt'), estResult['beta'], fmt="%f", delimiter="\t")

            if 'GENO' in alignResult:
                betaObj = alignResult['GENO']['SNPINFO'][['RAW_SNP', 'RAW_A1']].copy()
                betaObj.rename(columns={"RAW_SNP": "SNP", "RAW_A1": "A1"})
                betaObj.loc[:, 'BETAJ'] = estResult['betaEst']
                betaObj.loc[alignResult['GENO']['FLIPINFO'], 'BETAJ'] = -betaObj.loc[
                    alignResult['GENO']['FLIPINFO'], 'BETAJ']
            else:
                betaObj = alignResult['SS'].loc[:, ['SNP', 'A1']].copy()
                betaObj.loc[:, 'BETAJ'] = estResult['betaEst']
            try:
                betaObj.to_csv(os.path.join(p_dict['dir'], 'beta_est.txt'), sep='\t', index=False, header=False)
            except:
                print('Can\'t write weights into file:', os.path.join(p_dict['dir'], 'beta_est.txt'))

            # PRStuning if estimation is successful
            print('Start calculating PRStuning AUC...')
            AUC_prstuning = []
            for col in range(5, weightObj.shape[1]):
                newWeight = weightObj.iloc[:, col]
                auc = GWEB_prstuning(newWeight, estResult['beta'], p_dict['n0'], p_dict['n1'], refObj, alignResult['SS'])
                AUC_prstuning.append(auc)
                print("PRStuning AUC for parameter", weightObj.columns[col], "is", auc)
            result = pd.DataFrame(pd.Series(AUC_prstuning, index=weightObj.columns[range(5, weightObj.shape[1])]), columns=['PRStuning'])
            pars_prstuning = weightObj.columns[AUC_prstuning.index(max(AUC_prstuning)) + 5]
        else:
            print("Five times tried! Can't find a converged chain. Not returning PRStuning AUC")

        # calculate testing AUC if geno is provided
        pars_test = None
        AUC_test = None
        if genoObj is not None:
            phenoDF = PRSeval.phenoParser(alignResult['GENO'], phenoFile=p_dict['pheno'])
            isBinPhe = PRSeval.isBinary(phenoDF.loc[:, 'PHE'])
            if not isBinPhe:
                print("Warning: phenotype needs to be binary! Not returning testing AUC")
            print('Start calculating PRS score using plink...')
            AUC_test = []
            for col in range(5, weightObj.shape[1]):
                newWeight = weightObj.iloc[:, col]
                if not os.path.exists(os.path.join(p_dict['dir'] + 'prs_results/')):
                    os.makedirs(os.path.join(p_dict['dir'] + 'prs_results/'))
                score = scoringByPlink(alignResult['GENO'], newWeight, splitByChr=False,
                                       out=os.path.join(p_dict['dir'] + 'prs_results/' + weightObj.columns[col] + "_"),
                                       thread=p_dict['thread'])
                auc = roc_auc_score(PRSeval.convert01(phenoDF.loc[:, 'PHE'].to_numpy()), score.iloc[:, 0])
                auc = auc if auc >= 0.5 else 1 - auc
                AUC_test.append(auc)
                print("Testing AUC for parameter", weightObj.columns[col], "is", auc)
            pars_test = weightObj.columns[AUC_test.index(max(AUC_test)) + 5]
            if result is not None:
                result['Testing'] = AUC_test
        else:
            print("Testing genotype data not available. Not calculating testing AUC")

        if result is not None:
            print("AUC results saved to", os.path.join(p_dict['dir'], "auc_results.txt"))
            result.to_csv(os.path.join(p_dict['dir'], "auc_results.txt"), header=True, index=True, sep="\t")
        else:
            print('No AUC result.')

        if pars_prstuning is not None:
            print("The best-performing parameter based on PRStuning:", pars_prstuning)
        if pars_test is not None:
            print("The best-performing parameter based on testing data:", pars_test)
        if (AUC_prstuning is not None) and (AUC_test is not None):
            cor = np.corrcoef(AUC_prstuning, AUC_test)[0, 1]
            rd = abs(max(AUC_prstuning) - max(AUC_test)) / max(AUC_test)
            print("The correlation between AUCs from PRStuning and testing data is", "{:.4g}".format(cor))
            print("The relative difference between AUC from PRStuing and testing data is", "{:.4g}".format(rd))
