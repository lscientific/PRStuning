import os
import pandas as pd
import numpy as np
import PRSparser
import PRSalign
import PRSeval
from PRScoring import scoringByPlink
import math
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

def snpEM(z, maxIter=1000, tol=1e-4, beta0=0, info=True):
    def logF(x):
        return 0 if x == 0 else math.log(x)
    logF = np.vectorize(logF)
    m = len(z)
    # initialization
    pi0_0 = 0.99
    pi1_0 = 1 - pi0_0
    sigma2_0 = np.random.gamma(1, 1, 1)
    pi0_t = pi0_0
    pi1_t = pi1_0
    sigma2_t = sigma2_0
    h = list()
    h0 = 0
    def nanVal(x):
        return 0 if math.isnan(x) else x
    nanVal = np.vectorize(nanVal)
    Qval_save = []
    for iter in range(maxIter):
        # E step
        tmpH0 = pi0_t * norm.pdf(z)
        tmpH = pi1_t * norm.pdf(z, scale=math.sqrt(1 + sigma2_t))
        norH = tmpH0 + tmpH
        h0 = nanVal(tmpH0 / norH)
        h = nanVal(tmpH / norH)
        # M step
        pi0_t1 = (np.sum(h0) + beta0) / (m + beta0)
        pi1_t1 = 1 - pi0_t1
        if np.sum(h) == 0:
            sigma2_t1 = 0
        else:
            sigma2_t1 = np.max(np.sum(h * z ** 2) / np.sum(h) - 1, 0)
        def g_s2(x, y):
            return 0 if np.power(y, 2) == 0 else np.sqrt(np.power(x - y, 2) / np.power(y, 2))
        if (abs(nanVal((pi0_t1 - pi0_t) / pi0_t)) < tol) & \
                (np.sqrt(nanVal(np.sum((pi1_t1 - pi1_t) ** 2) / np.sum(np.power(pi1_t, 2)))) < tol) & \
                (g_s2(sigma2_t1, sigma2_t) < tol):
            break
        else:
            pi0_t = pi0_t1
            pi1_t = pi1_t1
            sigma2_t = sigma2_t1
            Qval_save.append(np.sum(h0 * logF(tmpH0)) + np.sum(np.array([x * y for x, y in zip(h, logF(tmpH))])))
    pi0 = pi0_t
    pi1 = pi1_t
    sigma2 = sigma2_t
    Qval = np.sum(h0 * logF(tmpH0)) + np.sum(np.array([x * y for x, y in zip(h, logF(tmpH))]))
    if info:
        print('pi0:', pi0_t)
        print('pi1:', pi1_t)
        print('sigma2:', sigma2_t)
        print('Iteration:', iter)
        print('Log-likelihood:', Qval)
    return pi0, pi1, sigma2, h0, h, iter, Qval, Qval_save


def main(p_dict):
    if p_dict['iprefix'] is not None:
        if os.path.exists(p_dict['iprefix'] + 'ssf.txt'):
            p_dict['ssf'] = p_dict['iprefix'] + 'ssf.txt'
        if os.path.exists(p_dict['iprefix'] + 'weight.txt'):
            p_dict['weight'] = p_dict['iprefix'] + 'weight.txt'
        if os.path.exists(p_dict['iprefix'] + 'geno.h5'):
            p_dict['h5geno'] = p_dict['iprefix'] + 'geno.h5'

    print('Parsing GWAS summary statistics from', p_dict['ssf'], "...")
    smryObj = PRSparser.smryParser(p_dict['ssf'])
    if smryObj is None:
        print('Can not GWAS summary statistics:', p_dict['ssf'])
        return

    print('Parsing PRS weight file from', p_dict['weight'], "...")
    weightObj = pd.read_csv(p_dict['weight'], delimiter="\t")
    if weightObj is None:
        print('Can not PRS weight file :', p_dict['weight'])
        return

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
            alignResult = {'SS': smryObj, 'WEIGHT': weightObj}
        else:
            alignResult = {'SS': smryObj, 'GENO': genoObj, 'WEIGHT': weightObj}

    else:
        print('Start aligning all files...')
        # overlap weight and ssf
        SNPs_intersect = pd.DataFrame({'SNP': list(set(weightObj.loc[:, "SNP"]).intersection(smryObj.loc[:, "SNP"]))})
        weightObj = weightObj.merge(SNPs_intersect, how='inner', on='SNP')
        smryObj = smryObj.merge(SNPs_intersect, how='inner', on='SNP')
        if genoObj is not None:
            # Without testing genotype data
            alignResult = PRSalign.alignSmry2Geno(smryObj, genoObj, outSmry = os.path.join(p_dict['out'], 'align_ssf.txt'),
                                                  outGeno=os.path.join(p_dict['out'], 'align_geno.h5'), byID=True, complevel=9)
        smryObj = alignResult['SS']
        smryObj['order'] = range(smryObj.shape[0])
        weightObj = weightObj.merge(smryObj.loc[:, ["SNP", 'order']], how='inner', on='SNP')
        weightObj = weightObj.set_index("order").sort_index()
        weightObj.to_csv(os.path.join(p_dict['dir'], 'align_weight.txt'), sep='\t', index=False)

    if not p_dict['align-only']:
        result = None
        # PRStuning AUC
        Ne = 4 / (1 / p_dict['n0'] + 1 / p_dict['n1'])
        AUC_prstuning = []
        for col in range(5, alignResult['WEIGHT'].shape[1]):
            z = alignResult['SS']['BETA'] / alignResult['SS']['SE']
            newWeight = alignResult['WEIGHT'].iloc[:, col]
            # only use SNPs with non-zero weight: pruning
            idx = (np.where(newWeight == 0)[0]).tolist()
            if len(idx) > 0:
                z = z.drop(idx).reset_index(drop=True)
                newWeight = newWeight.drop(idx).reset_index(drop=True)
            pi0, _, sigma2, _, _, _, Qval, _ = snpEM(z, maxIter=1000, tol=1e-4, beta0=len(z) / 100, info=False)
            paramDF = pd.DataFrame({'pi0': pi0, 'sigma2': sigma2, 'logLik': Qval})
            paramFile = os.path.join(p_dict['dir'], 'param.txt')
            paramDF.to_csv(paramFile, sep='\t')
            print('Estimated parameter values are saved to', paramFile)

            pi0_tilte = pi0 * norm.pdf(z) / (pi0 * norm.pdf(z) + (1 - pi0) * norm.pdf(z, scale=np.sqrt(sigma2 + 1)))
            lmd = 1 / (1 + 1 / sigma2)
            auc_post_rs = np.zeros(100)
            for j in range(100):
                z_post_rs = np.zeros(len(z))
                for m in range(len(z)):
                    c = np.random.binomial(1, pi0_tilte[m], 1)
                    if c == 0:
                        z_post_rs[m] = np.random.normal(lmd * z[m], np.sqrt(lmd), 1)
                delta_post_rs = 2 * np.sum(
                    newWeight * z_post_rs * alignResult['SS']['SE'].drop(idx) / np.sqrt(Ne)) / np.sqrt(np.sum(newWeight ** 2 * 2 * alignResult['SS']['SE'].drop(idx) ** 2))
                auc_post_rs[j] = norm.cdf(np.abs(delta_post_rs))
            auc = np.mean(auc_post_rs)
            AUC_prstuning.append(auc)
            print("PRStuning AUC for parameter", alignResult['WEIGHT'].columns[col], "is", auc)

        result = pd.DataFrame(pd.Series(AUC_prstuning, index=alignResult['WEIGHT'].columns[range(5, alignResult['WEIGHT'].shape[1])]), columns=['PRStuning'])

        # testing AUC
        if genoObj is not None:
            phenoDF = PRSeval.phenoParser(alignResult['GENO'], phenoFile=p_dict['pheno'])
            isBinPhe = PRSeval.isBinary(phenoDF.loc[:, 'PHE'])
            if not isBinPhe:
                print("Warning: phenotype needs to be binary!")
            print('Start calculating PRS score using plink...')
            AUC_test = []
            for col in range(5, alignResult['WEIGHT'].shape[1]):
                newWeight = alignResult['WEIGHT'].iloc[:, col]
                if not os.path.exists(os.path.join(p_dict['dir'] + 'prs_results/')):
                    os.makedirs(os.path.join(p_dict['dir'] + 'prs_results/'))
                score = scoringByPlink(alignResult['GENO'], newWeight, splitByChr=False,
                                       out=os.path.join(p_dict['dir'] + 'prs_results/' + weightObj.columns[col] + "_"),
                                       thread=p_dict['thread'])
                auc = roc_auc_score(PRSeval.convert01(phenoDF.loc[:, 'PHE'].to_numpy()), score.iloc[:, 0])
                auc = auc if auc >= 0.5 else 1 - auc
                AUC_test.append(auc)
                print("Testing AUC for parameter", alignResult['WEIGHT'].columns[col], "is", auc)
            if result is not None:
                result['Testing'] = AUC_test

        else:
            print("Testing genotype data not available. Not calculating testing AUC")

        if result is not None:
            print("AUC results saved to", os.path.join(p_dict['dir'], "auc_results.txt"))
            result.to_csv(os.path.join(p_dict['dir'], "auc_results.txt"), header=True, index=True, sep="\t")
        else:
            print('No AUC result.')


