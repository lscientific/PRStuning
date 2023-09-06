import numpy as np
from functools import reduce
import multiprocessing
try:
    from GWEButils_cFunc_gsl import *
except ModuleNotFoundError:
    from GWEButils_cFunc import *
from scipy.stats import norm
import warnings, time

def isnumeric(array):
    """
    Determine whether the argument has a numeric datatype, when converted to a NumPy array.
    Booleans, unsigned integers, signed integers, floats and complex
    numbers are the kinds of numeric datatype.
                    
    Parameters
    ----------
    array : array-like The array to check.
                                        
    Returns
    -------
    `bool` True if the array has a numeric datatype, False if not.
    """
    # Boolean, unsigned integer, signed integer, float, complex.
    _NUMERIC_KINDS = set('buifc')
    if isinstance(array, (list,int ,float)):
        array = np.asarray(array)
    try:
        return(array.dtype.kind in _NUMERIC_KINDS)
    except:
        return(False)

def isSqMatrix(X):
    '''
    Check whether X is a square matrix
    '''
    if isinstance(X, list):
        X = np.asarray(X)
    try:
        assert X.ndim==2
    except AssertionError:
        print('GWEButils.isSqMatrix: The dimension of X should be 2')
        return False
    except:
        return False
    try:
        assert X.shape[0]==X.shape[1]
    except AssertionError:
        print('GWEButils.isSqMatrix: The matrix X should be a square matrix')
        return False
    except:
        return False
    return True

def checkInput(betaHat, s, Rlist, A=None):
    '''
    Check whether the input of estimate function is valid
    '''
    try:
        assert isnumeric(betaHat) and isnumeric(s)
    except AssertionError:
        print('GWEButils.checkInput: betaHat and s should be a numeric list!')
        return False
    betaHat = np.asarray(betaHat)
    s = np.asarray(s)
    if s.size==1:
        s = np.repeat(s, betaHat.size)
    try: 
        assert betaHat.size==s.size
    except AssertionError:
        print('GWEButils.checkInput: The length of betaHat and s should be equal!')
        return False
    m = betaHat.size
    try:
        assert isinstance(Rlist, list) and all(isnumeric(item) and isSqMatrix(item) for item in Rlist )
    except AssertionError:
        print('GWEButils.checkInput: Rlist should be a list of square matrix!')
        return False
    try:
        assert np.sum(blockSize(Rlist))==m
    except AssertionError:
        print('GWEButils.checkInput: The total length of Rlist should be equal to length of betaHat and s!')
        return False

    if A is not None:
        try:
            assert isnumeric(A)
        except AssertionError:
            print('GWEButils.checkInput: A should be a numeric matrix!')
            return False
        A = np.asarray(A)
        try: 
            assert A.shape[1]==m
        except AssertionError:
            print('GWEButils.checkInput: The column dimension of A and s should be matched!')
            return False
    return True

def blockSize(Rlist):
    '''
    Obtain the size of each block via Rlist
    '''
    return [x.shape[0] for x in Rlist]

def blocking(x, bkSize):
    '''
    partition x into different blocks according to bkSize (a list of each block's size)
    '''
    bkNum = len(bkSize)
    start = np.append(0,np.cumsum(bkSize))
    x = np.asarray(x)
    if x.ndim ==1:
        return [x[start[i]:start[i+1]] for i in range(bkNum) ]
    else:
        return [x[...,start[i]:start[i+1]] for i in range(bkNum)]

def concat(x):
    '''
    Convert x from 2d list into 1d by concatenating
    '''
    return reduce(lambda a, b: np.concatenate((a,b), axis=-1), x)

def avg(X, start=0, end=None):
    return(np.mean(X[start:end], axis=0))

def changeOrder(X, permute=None):
    if permute is None:
        return(X)
    else:
        return(X[permute])
    
def calPropVar(m, pi, sigma2, s, N=0, adj=1):
    if len(pi)>1:
        causalProp = np.sum(pi[1:])
        if N==0:
            totalVar = m*np.sum(pi[1:]*sigma2[1:])*adj
        else:
            totalVar = np.sum(np.sum(pi[1:]*sigma2[1:])/((s**2)*N))*adj
    else:
        causalProp = 1.
        if N==0:
            totalVar = m*sigma2[0]*adj
        else:
            totalVar = np.sum(sigma2[0]/((s**2)*N))*adj

    return({'prop':causalProp, 'var': totalVar})

def getPSD(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.real(eigvec)
    xdiag = np.diag(np.maximum(np.real(eigval), 0))
    return Q.dot(xdiag).dot(Q.T)

def gibbsEst(betaHat, s, Rlist, ldscore, N=0, K=1, empNull=False, sigma02=0., burnin=100, maxIter=200, thread=-1, sparseMat=False, adj=1):
    '''
    Gibbs Sampling estimation method 
    empNull =T:  Using empirical null distribution, which means sigma02 will be estimated from data
    '''
    seed = int(time.time())
    np.random.seed(seed)
    set_seed(seed+123456789)
    if not checkInput(betaHat, s, Rlist):
        return
    else:
        try:
            thread = round(float(thread))
        except ValueError:
            print('GWEButils.gibbsEst:thread must be a numeric value')
            thread = 0
        if thread!=1:
            cpuNum = multiprocessing.cpu_count()
            if thread<=0: #Default setting; using total cpu number
                thread = cpuNum
            # print(cpuNum, 'CPUs detected, using', thread, 'threads...')

        m = len(betaHat)
        bkSize = blockSize(Rlist)
        bkNum = len(bkSize)
        betaHat = np.asarray(betaHat); s = np.asarray(s)
        betaHatBk = blocking(betaHat,bkSize)
        sBk = blocking(s, bkSize)

        if sparseMat:
            RSmatBk = list(map(lambda a, b: sparse.csr_matrix(a /b[:, None] /b[None, :]), Rlist, sBk))
        else:
            RSmatBk = list(map(lambda a, b: a /b[:, None] /b[None, :], Rlist, sBk))
        s2 = s**2
        s2Bk = blocking(s2, bkSize)
        invS2betaHat = betaHat/s2
        invS2betaHatBk = blocking(invS2betaHat, bkSize)

        ###Initialization
        if empNull:
            sigma02 = 1e-6
        #Moment estimator to initial the overall variance of a single SNP
        sumZ2 = np.sum(betaHat**2/s2)
        if sumZ2<=m:
            print('mean(Z^2)<=1! Please check the summary statistics...')
            raise
        sumAdjustLdscore = np.sum(ldscore/s2)
        totalVar1Init = (sumZ2-m)/sumAdjustLdscore
        if K<1:
            K=0
            empNull=True
            tmpGamma = np.ones((1,m))
            tmpBeta = np.copy(betaHat)
            tmpSigma2 = np.array([totalVar1Init])
            tmpPi = np.array([1.])
        else:
            K = int(K)
            tmpPi0 = 0.99
            tmpPi = np.append(tmpPi0, np.repeat((1-tmpPi0)/K,K))
            tmpGamma = np.asfortranarray(np.random.multinomial(1, tmpPi, m).T, dtype=np.float64)
            a = 10**np.arange(0, K)
            tmpSigma12_0 = max((totalVar1Init-tmpPi0*sigma02)/(1-tmpPi0), sigma02)
            tmpSigma2 = np.append(sigma02, K*tmpSigma12_0*a/np.sum(a))
            tmpBeta = np.zeros(m)
            for k in range(K+1):
                tmpBeta[np.ravel(tmpGamma[k]==1)] = np.random.normal(0, scale=np.sqrt(tmpSigma2[k]),size=int(np.sum(tmpGamma[k])))
        
        tmpPhi = 1.
        tmpBetaBk = blocking(tmpBeta/tmpPhi, bkSize)
        tmpGammaBk = blocking(tmpGamma, bkSize)
        
        nsample = maxIter
        beta = np.zeros([nsample,m])
        gamma = np.zeros([nsample, K+1, m])
        pi = np.zeros([nsample, K+1])
        sigma2 = np.zeros([nsample, K+1])
        
        causalProp = np.zeros(maxIter)
        totalVar = np.zeros(maxIter)
        if thread!=1:
            pool = multiprocessing.Pool(processes = thread)
        
        chunkSize = int(np.ceil(maxIter/10))
        for iter0 in range(maxIter):
            phi_mu_tilde = 0.
            phi_lambda2_inv = 0.
            tmpSigma2_phi = tmpSigma2/(tmpPhi**2)
            if thread==1:
                #Serial programming
                for b in range(bkNum):
                    tmpBetaGammaBkResult = sampleBetaGamma(invS2betaHatBk[b], s2Bk[b], RSmatBk[b], tmpPhi, tmpBetaBk[b], tmpSigma2_phi, tmpGammaBk[b], tmpPi, True, False)
                    tmpGammaBk[b] = tmpBetaGammaBkResult['gamma']
                    tmpBetaBk[b] = tmpBetaGammaBkResult['beta']
                    phi_mu_tilde += tmpBetaGammaBkResult['phi_mu_tilde']
                    phi_lambda2_inv += tmpBetaGammaBkResult['phi_lambda2_inv']
            else:
                #Parallel Programming
                tmpResults = []
                
                for b in range(bkNum):
                    #Mantain total process number to processes，Automatically add a process when a process is finished in the pool
                    tmpResults.append(pool.apply_async(sampleBetaGamma, args=(invS2betaHatBk[b],s2Bk[b],RSmatBk[b], tmpPhi, tmpBetaBk[b], tmpSigma2_phi, tmpGammaBk[b], tmpPi, True, False)))

                for b in range(bkNum):
                    tmpBetaGammaBkResult = tmpResults[b].get()
                    tmpGammaBk[b] = np.asfortranarray(tmpBetaGammaBkResult['gamma'])
                    tmpBetaBk[b] = tmpBetaGammaBkResult['beta']
                    phi_mu_tilde += tmpBetaGammaBkResult['phi_mu_tilde']
                    phi_lambda2_inv += tmpBetaGammaBkResult['phi_lambda2_inv']

            tmpPhi = 1#samplePhi(phi_mu_tilde, phi_lambda2_inv, 0)
            tmpBeta = concat(tmpBetaBk)
            tmpGamma = concat(tmpGammaBk)
            tmpBeta0 = tmpPhi*tmpBeta
            tmpPi = samplePi2(tmpGamma)
            tmpSigma2 = sampleSigma2_2(tmpBeta0, tmpGamma, tmpSigma2, empNull)

            ###Change order to avoid identifiability problem
            permute = np.argsort(tmpSigma2)#np.argsort(tmpPi)[::-1]
            if ~np.array_equal(permute, np.arange(K+1)):
                tmpGamma = changeOrder(tmpGamma, permute)
                tmpGammaBk = blocking(tmpGamma, bkSize)
                tmpPi = changeOrder(tmpPi, permute)
                tmpSigma2 = changeOrder(tmpSigma2, permute)

            beta[iter0, :] = tmpBeta0*np.sqrt(adj)
            gamma[iter0, :, :] = tmpGamma
            pi[iter0,:] = tmpPi
            sigma2[iter0, :] = tmpSigma2*adj

            propVar = calPropVar(m, tmpPi, tmpSigma2, s, N, adj=adj)
            causalProp[iter0] = propVar['prop']
            totalVar[iter0] = propVar['var']

            # print(iter0, tmpPi, tmpSigma2)
            if iter0 % chunkSize==0 or (iter0==maxIter-1):
                chunk = int(np.ceil(iter0/chunkSize))*10
                if iter0 == 0:
                    # print(chunk, '% Pi:',tmpPi, 'Sigma2:',tmpSigma2)
                    print(chunk, '% CausalProp:',"{:.4g}".format(causalProp[iter0]),'TotalVar:',"{:.4g}".format(totalVar[iter0]))
                else:
                    # print(chunk, '% Pi:',avg(pi, iter0-chunkSize+1, iter0+1), 'Sigma2:', avg(sigma2, iter0-chunkSize+1, iter0+1))
                    print(chunk, '% CausalProp:',"{:.4g}".format(avg(causalProp, iter0-chunkSize+1, iter0+1)),'TotalVar:', "{:.4g}".format(avg(totalVar, iter0-chunkSize+1, iter0+1)))
        if thread!=1:             
            pool.close()
            # Call close function before join, otherwise error will raise. No process will be added to pool after close.
            #Join function: waiting the complete of subprocesses
            pool.join()
        
        betaEst = avg(beta, burnin)
        gammaEst = avg(gamma, burnin)
        piEst = avg(pi, burnin)
        sigma2Est = avg(sigma2, burnin)
        causalPropEst = avg(causalProp, burnin)
        totalVarEst = avg(totalVar, burnin)
        np.set_printoptions(precision=6)
        print('Final Estimates: CausalProp:',"{:.4g}".format(causalPropEst),'TotalVar:', "{:.4g}".format(totalVarEst))
        print('Final Estimates:')
        print('Pi:',piEst)
        print('Sigma2:',sigma2Est)
        return({'beta': beta, 'gamma': gamma, 'pi': pi, 'sigma2': sigma2,'causalProp': causalProp,'totalVar': totalVar, 'betaEst': betaEst, 'gammaEst': gammaEst, 'piEst': piEst, 'sigma2Est': sigma2Est, 'causalPropEst': causalPropEst, 'totalVarEst': totalVarEst})

def calError(x, y, type='rel'):
    if type=='rel':
        try:
            return np.linalg.norm(y-x)/np.linalg.norm(x)
        except:
            return np.linalg.norm(y-x)/np.asarray(x).size
    elif type == 'abs':
        return np.linalg.norm(y-x)
    elif type=='avg':
        return np.linalg.norm(y-x)/np.asarray(x).size
    else:
        return np.linalg.norm(y-x)/np.asarray(x).size

def ebEst(betaHat, s, Rlist, ldscore, N=0, K=1, empNull=False, sigma02=0., maxIter=35, nsample=100, thread=-1, propThld=5e-2, varThld=5e-2, burnin=100, sparseMat=False, momentEst=False, adj=1): 
    '''
    Sampling-based EB estimates; SAME algorithm
    See Doucet, A., Godsill, S.J. & Robert, C.P. Marginal maximum a posteriori estimation using Markov chain Monte Carlo. Statistics and Computing 12, 77–84 (2002). https://doi.org/10.1023/A:1013172322619
    https://link.springer.com/article/10.1023/A:1013172322619
    '''
    seed = int(time.time())
    np.random.seed(seed)
    set_seed(seed+123456789)
    if not checkInput(betaHat, s, Rlist):
        return
    else:
        try:
            thread = round(float(thread))
        except ValueError:
            print('GWEButils.ebEst:thread must be a numeric value')
            thread = 0
        if thread!=1:
            cpuNum = multiprocessing.cpu_count()
            if thread<=0: #Default setting; using total cpu number
                thread = cpuNum
            # print(cpuNum, 'CPUs detected, using', thread, 'threads...')

        m = len(betaHat)
        bkSize = blockSize(Rlist)
        bkNum = len(bkSize)
        betaHat = np.asarray(betaHat); s = np.asarray(s)
        betaHatBk = blocking(betaHat,bkSize)
        sBk = blocking(s, bkSize)
        
        if sparseMat:
            RSmatBk = list(map(lambda a, b: sparse.csr_matrix(a /b[:, None] /b[None, :]), Rlist, sBk))
        else:
            RSmatBk = list(map(lambda a, b: a /b[:, None] /b[None, :], Rlist, sBk))

        s2 = s**2
        s2Bk = blocking(s2, bkSize)
        invS2betaHat = betaHat/s2
        invS2betaHatBk = blocking(invS2betaHat, bkSize)

        ###Initialization
        if empNull:
            sigma02 = 1e-6

        if momentEst and (not empNull):
            sigma02 = 0

        #Moment estimator to initial the overall variance of a single SNP
        sumZ2 = np.sum(betaHat**2/s2)
        if sumZ2<=m:
            print('mean(Z^2)<=1! Please check the summary statistics...')
            raise
        sumAdjustLdscore = np.sum(ldscore/s2)
        totalVar1Init = (sumZ2-m)/sumAdjustLdscore
        # newLDSCresult = np.linalg.lstsq(np.vstack((np.ones(m),ldscore/s2)).T, betaHat**2/s2, rcond=None)[0]
        # totalVar1Init = newLDSCresult[1]
        # inf_ldsc = newLDSCresult[0]
        # print('Estimated inflation factor:', inf_ldsc)
        # betaHat = betaHat/np.sqrt(inf_ldsc)
        if K<1:
            K=0
            empNull=True
            tmpGamma = np.ones((1,m))
            tmpBeta = np.copy(betaHat)
            # tmpSigma2 = np.array([max(np.mean(betaHat**2-s**2), 1./m)])
            tmpSigma2 = np.array([totalVar1Init])
            tmpPi = np.array([1.])
        else:
            K = int(K)
            tmpPi0 = 0.99
            tmpPi = np.append(tmpPi0, np.repeat((1-tmpPi0)/K,K))
            tmpGamma = np.asfortranarray(np.random.multinomial(1, tmpPi, m).T, dtype=np.float64)
            a = 10**np.arange(0, K)
            tmpSigma12_0 = max((totalVar1Init-tmpPi0*sigma02)/(1-tmpPi0), sigma02)
            tmpSigma2 = np.append(sigma02, K*tmpSigma12_0*a/np.sum(a))
            tmpBeta = np.zeros(m)
            for k in range(K+1):
                tmpBeta[np.ravel(tmpGamma[k]==1.)] = np.random.normal(0, scale=np.sqrt(tmpSigma2[k]),size=int(np.sum(tmpGamma[k])))
        
        tmpPhi = 1. 
        tmpBetaBk = blocking(tmpBeta, bkSize)
        tmpGammaBk = blocking(tmpGamma, bkSize)
        
        pi = np.zeros([maxIter, K+1])
        sigma2 = np.zeros([maxIter, K+1])
        causalProp = np.zeros(maxIter)
        totalVar = np.zeros(maxIter)
        
        if thread!=1:
            pool = multiprocessing.Pool(processes = thread)
        
        outputGap0 = int(np.ceil(burnin/10))
        for iter0 in range(burnin):
            #maxSigma = np.sqrt(np.max(tmpSigma2))
            #tmpBeta = tmpBeta*tmpPhi/maxSigma
            #tmpBetaBk = [elem*tmpPhi/maxSigma for elem in tmpBetaBk ]
            #tmpPhi = maxSigma
            phi_mu_tilde = 0.
            phi_lambda2_inv = 0.
            tmpSigma2_phi = tmpSigma2/(tmpPhi**2)
            if thread == 1: 
                # Serial programming
                for b in range(bkNum):
                    tmpBetaGammaBkResult = sampleBetaGamma(invS2betaHatBk[b], s2Bk[b], RSmatBk[b], tmpPhi, tmpBetaBk[b], tmpSigma2_phi, tmpGammaBk[b], tmpPi, True, False)
                    tmpGammaBk[b] = np.asfortranarray(tmpBetaGammaBkResult['gamma'])
                    tmpBetaBk[b] = tmpBetaGammaBkResult['beta']
                    phi_mu_tilde += tmpBetaGammaBkResult['phi_mu_tilde']
                    phi_lambda2_inv += tmpBetaGammaBkResult['phi_lambda2_inv']
            else:
                # Parallel Programming
                tmpResults = []
                
                for b in range(bkNum):
                    # Mantain total process number to processes，Automatically add a process when a process is finished in the pool
                    tmpResults.append(pool.apply_async(sampleBetaGamma, args=(invS2betaHatBk[b],s2Bk[b],RSmatBk[b], tmpPhi, tmpBetaBk[b], tmpSigma2_phi, tmpGammaBk[b], tmpPi, True, False)))

                for b in range(bkNum):
                    tmpBetaGammaBkResult = tmpResults[b].get()
                    tmpGammaBk[b] = np.asfortranarray(tmpBetaGammaBkResult['gamma'])
                    tmpBetaBk[b] = tmpBetaGammaBkResult['beta']
                    phi_mu_tilde += tmpBetaGammaBkResult['phi_mu_tilde']
                    phi_lambda2_inv += tmpBetaGammaBkResult['phi_lambda2_inv']

            tmpPhi = 1#samplePhi(phi_mu_tilde, phi_lambda2_inv, 0)#100/totalVar1Init)
            tmpBeta = concat(tmpBetaBk)
            tmpGamma = concat(tmpGammaBk)
            tmpBeta0 = tmpPhi*tmpBeta
            tmpPi = samplePi2(tmpGamma)
            tmpSigma2 = sampleSigma2_2(tmpBeta0, tmpGamma, tmpSigma2, empNull)
            if momentEst:
                ratio = totalVar1Init/np.sum(tmpPi*tmpSigma2)
                tmpSigma2 = tmpSigma2*ratio
                tmpBeta = tmpBeta*np.sqrt(ratio)
                tmpBetaBk = [elem*np.sqrt(ratio) for elem in tmpBetaBk]
                tmpBeta0 = tmpPhi*tmpBeta
            ### Change order to avoid identifiability problem
            permute = np.argsort(tmpSigma2)#np.argsort(tmpPi)[::-1]
            if ~np.array_equal(permute, np.arange(K+1)):
                tmpGamma = changeOrder(tmpGamma, permute)
                tmpGammaBk = blocking(tmpGamma, bkSize)
                tmpPi = changeOrder(tmpPi, permute)
                tmpSigma2 = changeOrder(tmpSigma2, permute)
            # print(calPropVar(m, tmpPi, tmpSigma2, s, N, adj=adj))
            if iter0 % outputGap0 == 0 or (iter0==burnin-1):
                print('Burn-in, Complete',int(np.ceil(iter0/outputGap0))*10,'% ...', end='\r', flush=True)
        mcSampleNum = np.zeros(maxIter, dtype=int)
        iter0 = 0
        outputGap = 5#int(np.floor(maxIter/10))
        print('\nInferring hyperparameters ( Maximum Iter=',maxIter,')...')
        isNonconv=False
        for iter0 in range(maxIter):
            #Linear Increasing; Simulated annealing
            mcSampleNum[iter0] = iter0+1
            tmpBetaIter = np.zeros([mcSampleNum[iter0], m])
            tmpGammaIter = np.zeros([mcSampleNum[iter0], K+1, m])
            # maxSigma = np.sqrt(np.max(tmpSigma2))
            tmpSigma2_phi = tmpSigma2/(tmpPhi**2)
            for mcIdx in range(mcSampleNum[iter0]):
                # tmpBeta = tmpBeta*tmpPhi/maxSigma
                # tmpBetaBk = [elem*tmpPhi/maxSigma for elem in tmpBetaBk ]
                # tmpPhi = maxSigma
                phi_mu_tilde = 0.
                phi_lambda2_inv = 0.
                if thread == 1: 
                    #Serial programming
                    for b in range(bkNum):
                        tmpBetaGammaBkResult = sampleBetaGamma(invS2betaHatBk[b], s2Bk[b], RSmatBk[b], tmpPhi, tmpBetaBk[b], tmpSigma2_phi, tmpGammaBk[b], tmpPi, True, False)
                        tmpGammaBk[b] = np.asfortranarray(tmpBetaGammaBkResult['gamma'])
                        tmpBetaBk[b] = tmpBetaGammaBkResult['beta']
                        phi_mu_tilde += tmpBetaGammaBkResult['phi_mu_tilde']
                        phi_lambda2_inv += tmpBetaGammaBkResult['phi_lambda2_inv']
                else:
                    #Parallel Programming
                    tmpResults = []
                
                    for b in range(bkNum):
                        #Mantain total process number to processes，Automatically add a process when a process is finished in the pool
                        tmpResults.append(pool.apply_async(sampleBetaGamma, args=(invS2betaHatBk[b],s2Bk[b],RSmatBk[b], tmpPhi, tmpBetaBk[b], tmpSigma2_phi, tmpGammaBk[b], tmpPi,True, False)))
                    for b in range(bkNum):
                        tmpBetaGammaBkResult = tmpResults[b].get()
                        tmpGammaBk[b] = tmpBetaGammaBkResult['gamma']
                        tmpBetaBk[b] = tmpBetaGammaBkResult['beta']
                        phi_mu_tilde += tmpBetaGammaBkResult['phi_mu_tilde']
                        phi_lambda2_inv += tmpBetaGammaBkResult['phi_lambda2_inv']
                
                tmpPhi = 1#samplePhi(phi_mu_tilde, phi_lambda2_inv, 0)#100/totalVar1Init)
                tmpBeta = concat(tmpBetaBk)
                tmpGamma = concat(tmpGammaBk)
                tmpBeta0 = tmpPhi*tmpBeta
                tmpBetaIter[mcIdx,:] = tmpBeta0
                tmpGammaIter[mcIdx,:,:] = tmpGamma
            
            tmpPi = samplePi3(tmpGammaIter)
            tmpSigma2 = sampleSigma2_3(tmpBetaIter, tmpGammaIter, tmpSigma2, empNull)
            if momentEst:
                ratio = totalVar1Init/np.sum(tmpPi*tmpSigma2)
                tmpSigma2 = tmpSigma2*ratio
                tmpBeta = tmpBeta*np.sqrt(ratio)
                tmpBetaBk = [elem*np.sqrt(ratio) for elem in tmpBetaBk]
                tmpBeta0 = tmpPhi*tmpBeta
                tmpBetaIter[mcIdx,:] = tmpBeta0

            ###Change order to avoid identifiability problem
            permute = np.argsort(tmpSigma2)#np.argsort(tmpPi)[::-1]
            if ~np.array_equal(permute, np.arange(K+1)):
                tmpGamma = changeOrder(tmpGamma, permute)
                tmpGammaBk = blocking(tmpGamma, bkSize)
                tmpPi = changeOrder(tmpPi, permute)
                tmpSigma2 = changeOrder(tmpSigma2, permute)

            pi[iter0,:] = tmpPi
            sigma2[iter0, :] = tmpSigma2*adj
            propVar = calPropVar(m, tmpPi, tmpSigma2, s, N, adj=adj)
            causalProp[iter0] = propVar['prop']
            totalVar[iter0] = propVar['var']
            if iter0>= 1 and totalVar[iter0]/totalVar[iter0-1]>5:
                if not isNonconv:
                    isNonconv = True
                else:
                    print('Nonconvergence detected! Rerun the estimation')
                    return
            else:
                isNonconv = False
            # print(iter0, tmpPi, tmpSigma2)
            if iter0 % outputGap==0:
                # print('Iter', iter0, 'Pi:',tmpPi, 'Sigma2:',tmpSigma2)
                # print(np.sum(tmpPi*tmpSigma2))
                print('Iter', iter0, 'CausalProp:',"{:.4g}".format(causalProp[iter0]), 'TotalVar:',"{:.4g}".format(totalVar[iter0]))
            # if iter0>=5 and calError(pi[iter0-1,:], pi[iter0,:], type='avg')<=thld and calError(sigma2[iter0-1,:], sigma2[iter0,:], type='avg')<=thld and calError(pi[iter0-2,:], pi[iter0-1,:], type='avg')<=thld and calError(sigma2[iter0-2,:], sigma2[iter0-1,:], type='avg')<=thld:
            if iter0>=5 and calError(causalProp[iter0-1], causalProp[iter0], type='rel')<=propThld and calError(totalVar[iter0-1], totalVar[iter0], type='rel')<=varThld and calError(causalProp[iter0-2], causalProp[iter0-1], type='rel')<=propThld and calError(totalVar[iter0-2], totalVar[iter0-1], type='rel')<=varThld:
                print('Hyperparameters converged in iter', iter0)
                pi = pi[0:(iter0+1),:]
                sigma2 = sigma2[0:(iter0+1), :]
                causalProp = causalProp[0:(iter0+1)]
                totalVar = totalVar[0:(iter0+1)]
                mcSampleNum = mcSampleNum[0:(iter0+1)]
                break
        
        piEst = pi[iter0,:]
        sigma2Est = sigma2[iter0,:]
        causalPropEst = causalProp[iter0]
        totalVarEst = totalVar[iter0]
        np.set_printoptions(precision=6)
        print('Final Estimates: CausalProp:',"{:.4g}".format(causalPropEst),'TotalVar:', "{:.4g}".format(totalVarEst))
        print('Final Estimates:')
        print('Pi:',piEst)
        print('Sigma2:',sigma2Est)

        print('Start generating MC samples based on estimated values...')
        beta = np.zeros([nsample,m])
        gamma = np.zeros([nsample, K+1, m])
        outputGap2 = int(np.ceil(nsample/10))
        logLikelihoodList = np.zeros(nsample)
        tmpSigma2_phi = tmpSigma2/(tmpPhi**2)
        for mcIdx in range(nsample):
            if thread==1:
                #Serial programming
                for b in range(bkNum):
                    tmpBetaGammaBkResult = sampleBetaGamma(invS2betaHatBk[b], s2Bk[b], RSmatBk[b], tmpPhi, tmpBetaBk[b], tmpSigma2_phi, tmpGammaBk[b], piEst, False, True)
                    tmpGammaBk[b] = tmpBetaGammaBkResult['gamma']
                    tmpBetaBk[b] = tmpBetaGammaBkResult['beta']
                    logLikelihoodList[mcIdx]+=tmpBetaGammaBkResult['logL']
            else:
                #Parallel Programming
                tmpResults = []
                
                for b in range(bkNum):
                    #Mantain total process number to processes，Automatically add a process when a process is finished in the pool
                    tmpResults.append(pool.apply_async(sampleBetaGamma, args=(invS2betaHatBk[b],s2Bk[b],RSmatBk[b], tmpPhi, tmpBetaBk[b], tmpSigma2_phi, tmpGammaBk[b], piEst, False, True)))

                for b in range(bkNum):
                    tmpBetaGammaBkResult = tmpResults[b].get()
                    tmpGammaBk[b] = tmpBetaGammaBkResult['gamma']
                    tmpBetaBk[b] = tmpBetaGammaBkResult['beta']
                    logLikelihoodList[mcIdx]+=tmpBetaGammaBkResult['logL']

            tmpBeta = concat(tmpBetaBk)
            tmpGamma = concat(tmpGammaBk)
            tmpBeta0 = tmpPhi*tmpBeta
            beta[mcIdx,:] = tmpBeta0*np.sqrt(adj)
            gamma[mcIdx,:,:] = tmpGamma
            if mcIdx % outputGap2 == 0 or (mcIdx == nsample-1 ):
                print('Complete',int(np.ceil(mcIdx/outputGap2))*10,'% ...', end='\r', flush=True)
        #Harmonic mean estimator for marginal likelihood
        logLikelihood = -logSumExp(-logLikelihoodList)+np.log(nsample)
        if N>0:
            if empNull:
                BIC = (2*K+1+m)*np.log(N)-2*logLikelihood
            else:
                BIC = (2*K+m*causalPropEst)*np.log(N)-2*logLikelihood

            print('\nBIC:', "{:.4g}".format(BIC))
        else:
            if empNull:
                BIC = (2*K+1+m)*10-2*logLikelihood
            else:
                BIC = (2*K+m*causalPropEst)*10-2*logLikelihood
            print('\nWarning: N is not provided, BIC is approximated with effective sample size e^10=22026')
            print('BIC:', "{:.4g}".format(BIC))

        if thread!=1:             
            pool.close()
            #Call close function before join, otherwise error will raise. No process will be added to pool after close.
            #Join function: waiting the complete of subprocesses
            pool.join()
        betaEst = avg(beta)
        gammaEst = avg(gamma)

        return({'beta': beta, 'gamma': gamma, 'pi': pi, 'sigma2': sigma2, 'causalProp': causalProp, 'totalVar': totalVar,'betaEst': betaEst, 'gammaEst': gammaEst, 'piEst': piEst, 'sigma2Est': sigma2Est, 'causalPropEst': causalPropEst, 'totalVarEst': totalVarEst, 'BIC': BIC})

def getAB(evalFunc, a, b):
    if (b-a)<=10: 
        aVal = evalFunc(a)
        bVal = evalFunc(b)
        if (aVal>10) and (bVal<=-10):
            return a, b
        elif (aVal<=10) and (bVal<=-10):
            return a-5, b
        elif (aVal>10) and (bVal>-10):
            return a, b+5
        else:
            return a-5, b+5
    c = (a+b)/2.
    if evalFunc(c)>=0:
        return getAB(evalFunc, c, b)
    else:
        return getAB(evalFunc, a, c)


