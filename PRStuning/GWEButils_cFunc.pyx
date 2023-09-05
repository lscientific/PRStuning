#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, log, sqrt, M_PI, abs
from scipy.stats import norm, truncnorm
from libc.time cimport time,time_t

ctypedef np.float64_t DTYPE_t
cdef time_t t = time(NULL)
np.random.seed(int(t))

def set_seed(seed):
    np.random.seed(seed)

# cdef int intsum(int[:] x):
    # cdef Py_ssize_t xsize = x.shape[0]
    # cdef int result = 0
    # cdef Py_ssize_t i
    # for i in range(xsize):
        # result += x[i]
    # return result

# cdef DTYPE_t floatsum(DTYPE_t[:] x):
    # cdef Py_ssize_t xsize = x.shape[0]
    # cdef DTYPE_t result = 0.
    # cdef Py_ssize_t i
    # for i in range(xsize):
        # result += x[i]
    # return result

cdef DTYPE_t mymax(DTYPE_t[:] x):
    cdef Py_ssize_t xsize = x.shape[0]
    cdef DTYPE_t result = x[0]
    cdef Py_ssize_t i
    for i in range(1, xsize):
        if x[i]>result:
            result = x[i]
    return result

cpdef DTYPE_t logSumExp(DTYPE_t[:] x):
    cdef Py_ssize_t xsize = x.shape[0], i
    cdef DTYPE_t maxVal, sumOfExp = 0., result=0.
    maxVal = mymax(x)
    for i in range(xsize):
        sumOfExp += exp(x[i]-maxVal)
    result = maxVal+log(sumOfExp)
    return result

cdef np.ndarray[DTYPE_t,ndim=1] logH2prob(DTYPE_t[:] logH):
    cdef Py_ssize_t K1 = logH.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] prob = np.zeros(K1)
    cdef Py_ssize_t i
    '''
    cdef np.ndarray[DTYPE_t, ndim=2] expDiffMat = np.zeros((K1, K1))
    cdef Py_ssize_t j
    for i in range(K1):
        for j in range(K1):
            expDiffMat[i, j] = exp(logH[j]-logH[i])
    
    cdef DTYPE_t[:,:] expDiff_view =expDiffMat
    '''
    cdef DTYPE_t sumH = logSumExp(logH)
    for i in range(K1):
        # prob[i] = 1./floatsum(expDiff_view[i,:])
        prob[i] = exp(logH[i]-sumH)
    return prob

cdef DTYPE_t mydot(DTYPE_t[:] a, DTYPE_t[:] b):
    cdef Py_ssize_t n
    cdef DTYPE_t s=0.
    cdef Py_ssize_t i
    if a.shape[0] != b.shape[0]:
        raise ValueError('shape not matched')
    n = a.shape[0]
    c = np.zeros(n)
    for i in range(n):
        s += a[i] * b[i]
    return s

cdef np.ndarray[DTYPE_t,ndim=1] myinv(DTYPE_t[:] a):
    cdef Py_ssize_t n = a.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] s = np.zeros(n)
    cdef Py_ssize_t i
    for i in range(n):
        s[i] = 1./a[i]
    return s

cdef myassign(DTYPE_t[:] a, DTYPE_t[:] b):
    cdef Py_ssize_t n
    cdef Py_ssize_t i
    if a.shape[0] != b.shape[0]:
        raise ValueError('shape not matched')
    n = a.shape[0]
    for i in range(n):
        a[i] = b[i]

cdef DTYPE_t my_norm_logpdf(DTYPE_t x, DTYPE_t loc, DTYPE_t scale):
    cdef DTYPE_t root2, root2pi, prefactor, summand
    root2 = sqrt(2)
    root2pi = sqrt(2*M_PI)
    prefactor = - log(scale * root2pi)
    summand = -((x - loc)/(root2 * scale))**2                         
    return  prefactor + summand

cpdef dict sampleBetaGamma(DTYPE_t[:] invS2betaHat, DTYPE_t[:] s2, DTYPE_t[:,:] RSmat, DTYPE_t phi, DTYPE_t[:] beta, DTYPE_t[:] sigma2, DTYPE_t[:,:] gamma, DTYPE_t[:] pi, bint updatePhi=True, bint logLik=False):
    cdef Py_ssize_t i, k
    cdef Py_ssize_t K1, m
    cdef np.ndarray[DTYPE_t, ndim=1] logH_i, prob_i
    cdef DTYPE_t[:] logH_i_view
    cdef DTYPE_t mu_tilde_i, lamda2_i, mu_i
    cdef DTYPE_t[:] tmpBeta_view = beta.copy()
    cdef DTYPE_t[::1,:] tmpGamma_view = gamma.copy_fortran()
    cdef DTYPE_t logL = 0.
    cdef DTYPE_t[:] tmpWork_view = beta.copy()
    cdef DTYPE_t phi_mu_tilde=0., phi_lambda2_inv=0.

    m = invS2betaHat.shape[0]
    K1 = pi.shape[0]
    logH_i = np.zeros(K1)
    logH_i_view = logH_i
    if sigma2[0]>0:
        for i in range(m):
            ###Sample Beta
            mu_tilde_i = invS2betaHat[i]*phi-mydot(RSmat[i,:],tmpBeta_view)*phi*phi+RSmat[i, i]*tmpBeta_view[i]*phi*phi
            lambda2_i = 1./(phi*phi/s2[i]+mydot(tmpGamma_view[:,i],myinv(sigma2)))
            mu_i = lambda2_i*mu_tilde_i
            tmpBeta_view[i] = np.random.normal(mu_i, sqrt(lambda2_i), 1)
            ###Sample Gamma
            for k in range(K1):
                logH_i_view[k] = log(pi[k])+my_norm_logpdf(tmpBeta_view[i], 0, sqrt(sigma2[k])) 
            #Suppress overflow warning
            # old_settings = np.seterr(over='ignore')
            prob_i = logH2prob(logH_i_view)
            # np.seterr(**old_settings)
            myassign(tmpGamma_view[:,i], np.asarray(np.random.multinomial(1, prob_i), dtype=np.float64))
    ###Null=Delta Function; Collapsed Gibbs sampling
    else:
        #Suppress overflow warning
        # old_settings = np.seterr(over='ignore')
        for i in range(m):
            ###Sample Beta
            mu_tilde_i = invS2betaHat[i]*phi-mydot(RSmat[i,:], tmpBeta_view)*phi*phi+RSmat[i, i]*tmpBeta_view[i]*phi*phi
            lambda2_i = 1./(phi*phi/s2[i]+mydot(tmpGamma_view[1:,i], myinv(sigma2[1:])))
            mu_i = lambda2_i*mu_tilde_i
            tmpBeta_view[i] = np.random.normal(mu_i, np.sqrt(lambda2_i), 1)
            ###Sample Gamma
            for k in range(K1):
                if k==0:
                    logH_i_view[k] = log(pi[0])
                else:
                    logH_i_view[k] = log(pi[k]*sqrt(s2[i]/(s2[i]+phi*phi*sigma2[k])))+(mu_tilde_i*mu_tilde_i)/(2.*(phi*phi/s2[i]+1./sigma2[k]))
            prob_i = logH2prob(logH_i_view)
            myassign(tmpGamma_view[:,i], np.asarray(np.random.multinomial(1, prob_i), dtype=np.float64))
            tmpBeta_view[i] = tmpBeta_view[i]*(1-tmpGamma_view[0,i])
        # np.seterr(**old_settings)
    if updatePhi or logLik:
        for i in range(m):
            tmpWork_view[i] = mydot(RSmat[i,:], tmpBeta_view)
            phi_mu_tilde += tmpBeta_view[i]*invS2betaHat[i]
            phi_lambda2_inv += tmpBeta_view[i]*tmpWork_view[i] 
    if logLik:
        logL = (phi*phi_mu_tilde-phi*phi*phi_lambda2_inv/2.)
        # if sigma2[0]>0:
            # for i in range(m):
                # tmpWork_view[i] = mydot(sigma2, tmpGamma_view[:,i])
                # logL += (-log(phi*phi*tmpWork_view[i])-tmpBeta_view[i]*tmpBeta_view[i]/tmpWork_view[i])/2.
        # else:
            # for i in range(m):
                # if tmpGamma_view[0,i] == 0:
                    # tmpWork_view[i] = mydot(sigma2[1:], tmpGamma_view[1:,i])
                    # logL += (-log(phi*phi*tmpWork_view[i])-tmpBeta_view[i]*tmpBeta_view[i]/tmpWork_view[i])/2.
        # for i in range(m):
            # logL += log(mydot(pi, tmpGamma_view[:,i]))

    return({'beta': np.asarray(tmpBeta_view), 'gamma': np.asfortranarray(tmpGamma_view), 'phi_mu_tilde':phi_mu_tilde, 'phi_lambda2_inv':phi_lambda2_inv,'logL': logL})

cpdef np.ndarray[DTYPE_t, ndim=1] samplePi2(DTYPE_t[:,:] gamma):
    #Gibbs sampling, Bayesian
    cdef Py_ssize_t m, k, j, K1
    K1 = gamma.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] pi, a = np.zeros(K1)
    cdef DTYPE_t[:] a_view = a
    m = gamma.shape[1]
    for k in range(K1):
        a_view[k] += 1.
        for j in range(m):
            a_view[k] += gamma[k,j]
        # if a_view[k] == 0.:
            # a_view[k] = 1.
    # a_view[0] += (1e-4)*m
    pi=np.random.dirichlet(a)
    return pi

cpdef np.ndarray[DTYPE_t,ndim=1] samplePi3(DTYPE_t[:,:,:] gamma):
    #SAME algorithm, Empirical Bayes
    cdef Py_ssize_t n, m, i, j, k, K1
    K1 = gamma.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] pi, a = np.zeros(K1)
    cdef DTYPE_t[:] a_view = a
    m = gamma.shape[2]
    n = gamma.shape[0]
    for k in range(K1):
        a_view[k] += n
        for i in range(n):
            for j in range(m):
                a_view[k] += gamma[i,k,j]
        # if a_view[k] == 0.:
            # a_view[k] = 1.
    # a_view[0] += (1e-4)*m*n
    pi = np.random.dirichlet(a)
    return pi

cpdef np.ndarray[DTYPE_t,ndim=1] sampleSigma2_2(DTYPE_t[:] beta, DTYPE_t[:,:] gamma, DTYPE_t[:] sigma2, bint empNull=False):
    cdef Py_ssize_t m, k, j, K1
    K1 = gamma.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] newSigma2 = np.asarray(sigma2.copy()), halfSumGamma = np.zeros(K1), halfSumGammaBeta2 = np.zeros(K1)
    cdef DTYPE_t[:] halfSumGamma_view = halfSumGamma, halfSumGammaBeta2_view = halfSumGammaBeta2
    m = gamma.shape[1]
    for k in range(K1):
        for j in range(m):
            halfSumGamma_view[k] += gamma[k,j]
            halfSumGammaBeta2_view[k] += gamma[k,j]*(beta[j]*beta[j])
        halfSumGamma_view[k] /= 2.
        halfSumGammaBeta2_view[k] /= 2.
    if empNull:
        zeroComp = (halfSumGammaBeta2<=1e-16)
        newSigma2[~zeroComp] = 1./np.random.gamma(shape=1.+halfSumGamma[~zeroComp], scale=1./halfSumGammaBeta2[~zeroComp])
        # newSigma2[zeroComp] = 1./np.random.uniform(0.,1e16)
    else:
        zeroComp = (halfSumGammaBeta2[1:]<=1e-16)
        idx = np.arange(1,K1)[~zeroComp]
        newSigma2[idx] = 1./np.random.gamma(shape=1.+halfSumGamma[idx], scale=1./halfSumGammaBeta2[idx])
        # newSigma2[np.arange(1,K1)[zeroComp]] = 1./np.random.uniform(0., 1e16)
    return(newSigma2)

cpdef np.ndarray[DTYPE_t,ndim=1] sampleSigma2_3(DTYPE_t[:,:] beta, DTYPE_t[:,:,:] gamma, DTYPE_t[:] sigma2, bint empNull=False):
    cdef Py_ssize_t m, n, i, k, j, K1
    K1 = gamma.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] newSigma2=np.array(sigma2.copy()), halfSumGamma = np.zeros(K1), halfSumGammaBeta2 = np.zeros(K1)
    cdef DTYPE_t[:] halfSumGamma_view = halfSumGamma, halfSumGammaBeta2_view = halfSumGammaBeta2
    m = gamma.shape[2]
    n = gamma.shape[0]
    for k in range(K1):
        for i in range(n):
            for j in range(m):
                halfSumGamma_view[k] += gamma[i,k,j]
                halfSumGammaBeta2_view[k] += gamma[i,k,j]*(beta[i,j]**2)
        halfSumGamma_view[k] /= 2.
        halfSumGammaBeta2_view[k] /= 2.
    
    if empNull:
        zeroComp = (halfSumGammaBeta2<=1e-16)
        newSigma2[~zeroComp] = 1./np.random.gamma(shape=n+halfSumGamma[~zeroComp], scale=1./halfSumGammaBeta2[~zeroComp])
        # newSigma2[zeroComp] = 1./np.random.uniform(0.,1e16)
    else:
        zeroComp = (halfSumGammaBeta2[1:]<=1e-16)
        idx = np.arange(1,K1)[~zeroComp]
        newSigma2[idx] = 1./np.random.gamma(shape=n+halfSumGamma[idx], scale=1./halfSumGammaBeta2[idx])
        # newSigma2[np.arange(1,K1)[zeroComp]] = 1./np.random.uniform(0., 1e16)
    return sigma2

cpdef DTYPE_t samplePhi(DTYPE_t phi_mu_tilde, DTYPE_t phi_lambda2_inv, DTYPE_t phi_invVar_prior=0):
    cdef DTYPE_t lambda2_inv_tilde = phi_lambda2_inv+phi_invVar_prior
    if abs(phi_mu_tilde/lambda2_inv_tilde)>=1e3:
        return phi_mu_tilde/lambda2_inv_tilde
    else:
        return phi_mu_tilde/lambda2_inv_tilde+np.random.normal(0., 1./sqrt(lambda2_inv_tilde), 1)

###Including annotations
cpdef dict sampleBetaGammaAnno(DTYPE_t[:] invS2betaHat, DTYPE_t[:] s2, DTYPE_t[:,:] RSmat, DTYPE_t phi, DTYPE_t[:,:] A, DTYPE_t[:] beta, DTYPE_t[:] gamma, DTYPE_t sigma02, DTYPE_t[:] alpha, DTYPE_t[:] eta, bint updatePhi, bint logLik):
    cdef Py_ssize_t i, m = invS2betaHat.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] logH_i = np.zeros(2), prob_i
    cdef DTYPE_t[:] logH_i_view = logH_i
    cdef DTYPE_t mu_tilde_i, lamda2_i, mu_i, sigma12_i
    cdef DTYPE_t[:] tmpBeta_view = beta.copy(), tmpGamma_view = gamma.copy(), pi1
    cdef DTYPE_t logL = 0.
    cdef DTYPE_t[:] tmpWork_view = beta.copy()
    cdef DTYPE_t phi_mu_tilde=0., phi_lambda2_inv=0.

    pi1 = norm.cdf(myMatVecMul(A.T, alpha))
    if sigma02>0:
        for i in range(m):
            sigma12_i = exp(mydot(A[:,i], eta))
            # pi1_i =norm.cdf(mydot(A[:,i], alpha))
            ###Sample Beta
            mu_tilde_i = invS2betaHat[i]*phi-mydot(RSmat[i,:], tmpBeta_view)*phi*phi+RSmat[i, i]*tmpBeta_view[i]*phi*phi
            lambda2_i = 1./(phi*phi/s2[i]+(1.-tmpGamma_view[i])/sigma02+tmpGamma_view[i]/sigma12_i)
            mu_i = lambda2_i*mu_tilde_i
            tmpBeta_view[i] = np.random.normal(mu_i, np.sqrt(lambda2_i), 1)
            ###Sample Gamma
            # old_settings = np.seterr(divide='ignore',over='ignore')
            logH_i_view[0] = log(1-pi1[i])+my_norm_logpdf(tmpBeta_view[i], 0, sqrt(sigma02))
            logH_i_view[1] = log(pi1[i])+my_norm_logpdf(tmpBeta_view[i], 0, sqrt(sigma12_i))
            prob_i = logH2prob(logH_i_view)
            # np.seterr(**old_settings)
            tmpGamma_view[i] = np.float64(np.random.multinomial(1, prob_i)[1])
            # tmpGamma_view[i] = np.float64(np.random.binomial(1, prob_i[1]))
            # if logLik:
                # tmpWork_view[i] = (1-tmpGamma_view[i])*sigma02+tmpGamma_view[i]*sigma12_i
                # logL += (-log(phi*phi*tmpWork_view[i])-tmpBeta_view[i]*tmpBeta_view[i]/tmpWork_view[i])/2.
                # logL += log((1-pi1[i])*(1-tmpGamma_view[i])+pi1[i]*tmpGamma_view[i])
    ###Null=Delta Function; Collapsed Gibbs sampling
    else:
        # old_settings = np.seterr(divide='ignore',over='ignore')
        for i in range(m):
            sigma12_i = exp(mydot(A[:,i], eta))
            ###Sample Beta
            mu_tilde_i = invS2betaHat[i]*phi-mydot(RSmat[i,:], tmpBeta_view)*phi*phi+RSmat[i, i]*tmpBeta_view[i]*phi*phi
            lambda2_i = 1./(phi*phi/s2[i]+tmpGamma_view[i]/sigma12_i)
            mu_i = lambda2_i*mu_tilde_i
            tmpBeta_view[i] = np.random.normal(mu_i, sqrt(lambda2_i), 1)
            ###Sample Gamma        
            logH_i_view[0] = log(1-pi1[i])
            logH_i_view[1] = log(pi1[i]*sqrt(s2[i]/(s2[i]+phi*phi*sigma12_i)))+(mu_tilde_i*mu_tilde_i)/(2.*(phi*phi/s2[i]+1./sigma12_i))
            prob_i = logH2prob(logH_i_view)
            tmpGamma_view[i] = np.float64(np.random.multinomial(1, prob_i)[1])
            ####################################
            #########Warning: Although it is a binary data following bernoulli distribution, neither bernoulli/binomial random samples from numpy/gsl will make the sampler converge
            #########Pi1 will keep decreasing! 
            #########Solution: Using multinomial distribution instead!
            # tmpGamma_view[i] = np.float64(np.random.binomial(1, prob_i[1]))
            tmpBeta_view[i] = tmpBeta_view[i]*tmpGamma_view[i]
            # if logLik:
                # if tmpGamma_view[i] == 1:
                    # tmpWork_view[i] = sigma12_i 
                    # logL += (-log(phi*phi*tmpWork_view[i])-tmpBeta_view[i]*tmpBeta_view[i]/tmpWork_view[i])/2.
                # logL += log((1-pi1[i])*(1-tmpGamma_view[i])+pi1[i]*tmpGamma_view[i])
        # np.seterr(**old_settings)

    if updatePhi or logLik:
        for i in range(m):
            tmpWork_view[i] = mydot(RSmat[i,:], tmpBeta_view)
            phi_mu_tilde += tmpBeta_view[i]*invS2betaHat[i]
            phi_lambda2_inv += tmpBeta_view[i]*tmpWork_view[i] 
    if logLik:
        logL += (phi*phi_mu_tilde-phi*phi*phi_lambda2_inv/2.)
    return({'beta': np.asarray(tmpBeta_view), 'gamma': np.asfortranarray(tmpGamma_view), 'phi_mu_tilde':phi_mu_tilde, 'phi_lambda2_inv':phi_lambda2_inv,'logL': logL})

cdef np.ndarray[DTYPE_t, ndim=1] myMatVecMul(DTYPE_t[:,:] u, DTYPE_t[:] v):
    cdef Py_ssize_t i, j, m, n
    cdef np.ndarray[DTYPE_t, ndim=1] res 
    cdef DTYPE_t[:] res_view

    m = u.shape[0]
    n = u.shape[1]
    if v.shape[0] != n:
        raise ValueError('shape not matched')
    res = np.zeros(m)
    res_view = res

    for i in range(m):
        for j in range(n):
            res_view[i] += u[i,j] * v[j]

    return(res)

cpdef np.ndarray[DTYPE_t, ndim=1] sampleAlpha1(DTYPE_t[:] gamma, DTYPE_t[:,:] A, DTYPE_t[:,:] invAA, DTYPE_t[:] alpha, bint fixProp=False):
    cdef Py_ssize_t nA = A.shape[0], m = A.shape[1], i
    cdef np.ndarray[DTYPE_t, ndim=1] tmpPi, tmpAlpha = np.zeros(nA), Z = np.zeros(m), AZ = np.zeros(nA)
    cdef np.ndarray[DTYPE_t, ndim=2] gammaExt = np.zeros((2,m))
    cdef DTYPE_t[:] tmpAlpha_view = tmpAlpha, Z_view = Z, mu, mu1, mu0, Z0, Z1, negMu1, negMu0
    cdef DTYPE_t[:,:] gammaExt_view = gammaExt
    cdef int sumIdx = 0, j=0, k=0

    if fixProp:
        for i in range(m):
            gammaExt_view[0,i] = 1.-gamma[i]
            gammaExt_view[1,i] = gamma[i]
        tmpPi = samplePi2(gammaExt_view)
        tmpAlpha_view[0] = norm.ppf(tmpPi[1])
        return(tmpAlpha)

    mu = myMatVecMul(A.T, alpha)
    for i in range(m):
        if gamma[i]==1:
            sumIdx += 1
    mu1 = np.zeros(sumIdx)
    mu0 = np.zeros(m-sumIdx)
    negMu1 = np.zeros(sumIdx)
    negMu0 = np.zeros(m-sumIdx)
    for i in range(m):
        if gamma[i]==1:
            mu1[j]=mu[i]
            negMu1[j] = (-mu[i])
            j+=1
        else:
            mu0[j]=mu[i]
            negMu0[j] = (-mu[i])
            k+=1
    Z0 = truncnorm.rvs(a=-np.inf, b=negMu0, loc=mu0, scale=1, size=m-sumIdx)
    Z1 = truncnorm.rvs(a=negMu1, b=np.inf, loc=mu1, scale=1, size=sumIdx)
    j=0
    k=0
    for i in range(m):
        if gamma[i] == 1:
            Z_view[i] = Z1[k]
            k+=1
        else:
            Z_view[i] = Z0[j]
            j+=1
    # for i in range(m):
        # mu = mydot(A[:,i], alpha)
        # if gamma[i] == 1:
            # Z_view[i] = truncnorm.rvs(a=-mu, b=np.inf, loc=mu, scale=1)
        # else:
            # Z_view[i] = truncnorm.rvs(a=-np.inf, b=-mu, loc=mu, scale=1)
    
    AZ = myMatVecMul(A, Z_view)
    AZ = myMatVecMul(invAA, AZ)
    tmpAlpha = np.random.multivariate_normal(AZ, invAA)
    return tmpAlpha

cpdef np.ndarray[DTYPE_t, ndim=1] sampleAlpha2(DTYPE_t[:,:] gamma, DTYPE_t[:,:] A, DTYPE_t[:,:] invAA, DTYPE_t[:] alpha, bint fixProp=False):
    cdef Py_ssize_t nA = A.shape[0], m = A.shape[1], iterNum = gamma.shape[0], totalNum = gamma.size, i,j
    cdef np.ndarray[DTYPE_t, ndim=1] tmpPi, tmpAlpha = np.zeros(nA), meanZ = np.zeros(m), AZ = np.zeros(nA)
    cdef DTYPE_t[:] tmpAlpha_view = tmpAlpha, meanZ_view = meanZ, AZ_view=AZ, mu, mu1, mu0, negMu1, negMu0, Z1, Z0
    cdef np.ndarray[DTYPE_t, ndim=3] gammaExt = np.zeros((iterNum, 2, m))
    cdef DTYPE_t[:,:,:] gammaExt_view = gammaExt
    cdef int sumIdx = 0, k=0, l=0
    
    if fixProp:
        for i in range(iterNum):
            for j in range(m):
                gammaExt_view[i,0,j] = 1.- gamma[i,j]
                gammaExt_view[i,1,j] = gamma[i,j]
        tmpPi = samplePi3(gammaExt_view)
        tmpAlpha_view[0] = norm.ppf(tmpPi[1])
        return(tmpAlpha)
 
    mu = myMatVecMul(A.T, alpha)
    for i in range(m):
        for j in range(iterNum):
            if gamma[j,i]==1:
                sumIdx += 1
    mu1 = np.zeros(sumIdx)
    mu0 = np.zeros(totalNum-sumIdx)
    negMu1 = np.zeros(sumIdx)
    negMu0 = np.zeros(totalNum-sumIdx)
    for i in range(m):
        for j in range(iterNum):
            if gamma[j, i]==1:
                mu1[k]=mu[i]
                negMu1[k] = (-mu[i])
                k+=1
            else:
                mu0[l]=mu[i]
                negMu0[l] = (-mu[i])
                l+=1
    Z0 = truncnorm.rvs(a=-np.inf, b=negMu0, loc=mu0, scale=1, size=totalNum-sumIdx)
    Z1 = truncnorm.rvs(a=negMu1, b=np.inf, loc=mu1, scale=1, size=sumIdx)
    k=0
    l=0
    for i in range(m):
        for j in range(iterNum):
            if gamma[j,i] == 1:
                meanZ_view[i] += Z1[l]
                l+=1
            else:
                meanZ_view[i] += Z0[k]
                k+=1
        meanZ_view[i] /= iterNum
    # for i in range(m):
        # mu = mydot(A[:,i], alpha)
        # for j in range(iterNum):
            # if gamma[j,i] == 1:
                # meanZ_view[i] += truncnorm.rvs(a=-mu, b=np.inf, loc=mu, scale=1)
            # else:
                # meanZ_view[i] += truncnorm.rvs(a=-np.inf, b=-mu, loc=mu, scale=1)
        # meanZ_view[i] /= iterNum 
     
    AZ = myMatVecMul(A, meanZ_view)
    AZ = myMatVecMul(invAA, AZ)
    tmpAlpha = np.random.multivariate_normal(AZ, invAA)
    return tmpAlpha

cpdef DTYPE_t sumLogNormPdf(DTYPE_t[:] x, DTYPE_t[:] mean, DTYPE_t[:] logVar):
    cdef Py_ssize_t n = x.shape[0], i
    cdef DTYPE_t result=0., sumLogVar=0.
    cdef DTYPE_t[:] tmpX_view = x.copy()
    for i in range(n):
        tmpX_view[i] = 2.*log(abs(x[i]-mean[i]))-log(2.)-logVar[i]
        sumLogVar += logVar[i]
    result = -exp(logSumExp(tmpX_view))-sumLogVar/2.-log(2.*M_PI)*n/2.
    return result

cpdef DTYPE_t sampleSigma02_1(DTYPE_t[:] beta, DTYPE_t[:] gamma, DTYPE_t sigma02):
    cdef Py_ssize_t m = beta.shape[0], j
    cdef DTYPE_t halfSumGamma = 0., halfSumGammaBeta2=0.
    for j in range(m):
        halfSumGamma += (1.-gamma[j])
        halfSumGammaBeta2 += (1.-gamma[j])*(beta[j]*beta[j])
    halfSumGamma /= 2.
    halfSumGammaBeta2 /= 2.
    if halfSumGammaBeta2>1e-16:
        sigma02 = 1./np.random.gamma(shape=1.+halfSumGamma, scale=1./halfSumGammaBeta2)
    # else:
        # sigma02 = 1./np.random.uniform(0.,1e16)
    return(sigma02)

cpdef DTYPE_t sampleSigma02_2(DTYPE_t[:,:] beta, DTYPE_t[:,:] gamma, DTYPE_t sigma02):
    cdef Py_ssize_t m = beta.shape[1], n = beta.shape[0], i, j
    cdef DTYPE_t halfSumGamma = 0., halfSumGammaBeta2=0.
    for i in range(n):    
        for j in range(m):
            halfSumGamma += (1.-gamma[i,j])
            halfSumGammaBeta2 += (1.-gamma[i,j])*(beta[i,j]*beta[i,j])
    halfSumGamma /= 2.
    halfSumGammaBeta2 /= 2.
    if halfSumGammaBeta2>1e-16:
        sigma02 = 1./np.random.gamma(shape=n+halfSumGamma, scale=1./halfSumGammaBeta2)
    # else:
        # sigma02 = 1./np.random.uniform(0.,1e16)
    return(sigma02)

cpdef DTYPE_t sampleSigma12_1(DTYPE_t[:] beta, DTYPE_t[:] gamma, DTYPE_t sigma12):
    cdef Py_ssize_t m = beta.shape[0], j
    cdef DTYPE_t halfSumGamma = 0., halfSumGammaBeta2=0.
    for j in range(m):
        halfSumGamma += (gamma[j])
        halfSumGammaBeta2 += (gamma[j])*(beta[j]*beta[j])
    halfSumGamma /= 2.
    halfSumGammaBeta2 /= 2.
    if halfSumGammaBeta2>1e-16:
        sigma12 = 1./np.random.gamma(shape=1.+halfSumGamma, scale=1./halfSumGammaBeta2)
    # else:
        # sigma12 = 1./np.random.uniform(0.,1e16)
    return(sigma12)

cpdef DTYPE_t sampleSigma12_2(DTYPE_t[:,:] beta, DTYPE_t[:,:] gamma, DTYPE_t sigma12):
    cdef Py_ssize_t m = beta.shape[1], n = beta.shape[0], i, j
    cdef DTYPE_t halfSumGamma = 0., halfSumGammaBeta2=0.
    for i in range(n):    
        for j in range(m):
            halfSumGamma += (gamma[i,j])
            halfSumGammaBeta2 += (gamma[i,j])*(beta[i,j]*beta[i,j])
    halfSumGamma /= 2.
    halfSumGammaBeta2 /= 2.
    if halfSumGammaBeta2>1e-16:
        sigma12 = 1./np.random.gamma(shape=1.+halfSumGamma, scale=1./halfSumGammaBeta2)
    # else:
        # sigma12 = 1./np.random.uniform(0.,1e16)
    return(sigma12)

