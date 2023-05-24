
import numpy as np
import random
from ...GWEButils_cy import ebEst
from ...plinkLD import ldscore

from scipy import sparse
import GWEButils
import GWEButils_cy
import time
import matplotlib.pyplot as plt
from functools import reduce
path = './LD_data/'

random.seed(123)
for n_size in range(2, 6):
    for n_data in range(1, 21):
        string = str(n_size) + '_' + str(n_data)
        m = 10000
        n0 = 1000 * n_size
        n1 = 1000 * n_size
        N = int(4 / (1 / n0 + 1 / n1))  # effective sample size
        pi0 = 0.9
        pi = [pi0, 1 - pi0]
        sigma2 = [0., 5e-4]  # true_values
        rho = 0.2
        blockNum = 10

        tmpSize = int(np.floor(m / blockNum))
        bkSize = [tmpSize] * (blockNum - 1) + [m - tmpSize * (blockNum - 1)]

        s = np.asarray([1. / np.sqrt(N)] * m)
        Rlist = []
        betaHatBk = []
        start = 0
        for b in range(blockNum):
            end = start + bkSize[b]
            Rlist.append(np.abs(np.subtract.outer(np.arange(bkSize[b]), np.arange(bkSize[b]))))
            Rlist[b] = rho ** Rlist[b]
            Rlist[b][Rlist[b] < 0.01] = 0
            # Rlist[b] = np.eye(bkSize[b])
            Rlist[b] = np.matrix(Rlist[b])
            S_b = np.diag(s[start:end])
            # Sparse matrix
            # Rlist[b] = sparse.csr_matrix(Rlist[b])
            # S_b = sparse.diags(s[start:end])
            # betaHatBk.append(np.ravel(np.random.multivariate_normal(np.ravel(S_b*Rlist[b]*
            # np.matrix(sparse.linalg.spsolve(S_b.tocsr(), beta[start:end])).transpose()), (S_b*Rlist[b]*S_b).todense(), 1)))
            start = end

        X = np.loadtxt(path + string + '/geno_' + string + '.txt', delimiter="\t")
        X0 = X[:(1000 * n_size)]
        X1 = X[(1000 * n_size):]
        f0_hat = np.mean(X0, axis=0) / 2
        f1_hat = np.mean(X1, axis=0) / 2
        se_hat = np.sqrt(2 * f0_hat * (1 - f0_hat))
        with open(path + string + '/' + 'SSF' + string + '.txt') as file:
            ssf = np.loadtxt(file, dtype=np.ndarray, skiprows=1)

        betaHat = ssf[:, 8].astype(str).astype(float)

        if __name__ == '__main__':
            print(n_size)
            print(n_data)
            print('True Values: Pi:', pi, '; Sigma2:', sigma2)
            startTime = time.time()
            ldscore_est = np.reshape(list(map(ldscore, Rlist)), newshape=m)
            result = ebEst(betaHat, s, Rlist, ldscore=ldscore_est, N=2000*n_size, K=1, empNull=False, sigma02=0., maxIter=100, nsample=500, thread=-1, burnin=500)
            endTime = time.time()
            print('Time cost', endTime - startTime, 's')
            # plt.plot(result['pi'][:, 0], '.')
            # plt.show()
            # plt.savefig('/Users/lingchen/Desktop/GWAS/Simulation_Dep/data/' + string + '/' + string + '_pi0.png')
            # plt.plot(result['sigma2'][:, 1], '.')
            # plt.show()
            # plt.savefig('/Users/lingchen/Desktop/GWAS/Simulation_Dep/data/' + string + '/' + string + '_sigma2.png')
            # pdb.set_trace()
            address = path + string
            np.savetxt(address + '/Beta' + string + '.txt', result['beta'], fmt="%f", delimiter="\t")
            np.savetxt(address + '/piEst' + string + '.txt', result['piEst'], fmt="%f", delimiter="\t")
            np.savetxt(address + '/sigma2Est' + string + '.txt', result['sigma2Est'], fmt="%f", delimiter="\t")
            np.savetxt(address + '/gammaEst_' + string + '.txt', result['gammaEst'], fmt="%f", delimiter="\t")
            np.savetxt(address + '/betaEst_' + string + '.txt', result['betaEst'], fmt="%f", delimiter="\t")


