import numpy as np
from scipy import stats
from GWEButils import blockSize, blocking

def prstuning_calculate(weight, beta_EB, n0, n1, alignResult):
    '''
    :param weight: weights of the PRS model to be evaluated
    :param beta_EB: matrix of sampled empirical Bayes beta from the gibbs sampler
    :param Rlist: list of LD matrices
    :param n0: training data sample size for control group
    :param n1: training data sample size for case group
    :param alignResult: aligned object saved to the current path from GWEB.py
    :return: PRStuning AUC
    '''
    Ne = 4 * n0 * n1 / (n0 + n1)  # effective sample size
    n = beta_EB.shape[0] # number of empirical Bayes samples
    Rlist = alignResult['REF']['LD']
    betaSE = alignResult['SS']['SE']
    flist = []
    for bk in range(len(alignResult['REF']['SNPINFO'])):
        flist.append(alignResult['REF']['SNPINFO'][bk].loc[:, 'F'])

    bkSize = blockSize(Rlist)
    bkNum = len(bkSize)
    weightBk = blocking(weight, bkSize)
    betaSE_Bk = blocking(betaSE, bkSize)
    betaBk = []
    start = 0
    for bk in range(bkNum):
        end = start + bkSize[bk]
        betaBk.append(beta[:, start:end])
        start = end

    wf_bk = np.zeros(n*bkNum).reshape((n, bkNum))  # sample_size * bkNum
    s2 = 0
    for bk in range(bkNum):
        SE_bk = np.sqrt(2 * flist[bk] * (1 - flist[bk]))
        R_beta = np.matmul(Rlist[bk], np.transpose(np.matmul(betaBk[bk], np.diag(1/betaSE_Bk[bk]))))
        f_bk = np.matmul(np.diag(SE_bk), R_beta)
        f_bk = f_bk / np.sqrt(Ne)
        wf_bk[:, bk] = np.matmul(np.matrix(weightBk[bk]), f_bk)
        wSE_bk = np.matrix(weightBk[bk] * SE_bk)
        s2 += np.matmul(np.matmul(wSE_bk, Rlist[bk]), np.transpose(wSE_bk))
    delta_samples = 2 * (np.array(np.sum(wf_bk, axis=1) / np.sqrt(2*s2))).flatten()
    delta_samples = [np.abs(delta) for delta in delta_samples]
    AUC_samples = [stats.norm.cdf(i) for i in delta_samples]
    AUC_prstuning = np.mean(AUC_samples)
    print("The PRStuning AUC is" + AUC_prstuning)

    return AUC_prstuning
