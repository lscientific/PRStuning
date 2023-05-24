#LD score regression
#Wei Jiang (w.jiang@yale.edu)

import numpy as np
import rpy2
import rpy2.robjects as robjects

def calH2_2stage(z, ldsc, N, a=None, s1thld=30):
    z_rvector = robjects.FloatVector(z)
    ldsc_rvector = robjects.FloatVector(ldsc)
    tmp = robjects.r.source('ldsc.R')
    calH2_rfunc = robjects.globalenv['calH2.2stage']
    if a is None:
        a = robjects.rinterface.NULL
    result = calH2_rfunc(z_rvector, ldsc_rvector, N, a, s1thld)
    # result = np.array(result, dtype=np.float64)
    h2est = result.rx2('h2')[0]
    a = result.rx2('a')[0]

    #Clear memory
    robjects.r('rm(list=ls(all=TRUE))')
    robjects.r('gc()')
    # robjects.r('print(memory.size())')
    # del result
    # gc.collect()
    return({'h2':h2est, 'a':a})

def ldscore(R, N=None):
    R2 = np.square(R)
    if N is not None:
        R2 = R2-(1-R2)/(N-2)
    #	diag(R2) = 1
    ldsc = np.ravel(np.sum(R2, axis=1))
    return(ldsc)

def ldscoreAnno(R, A, N=None):
    R2 = np.square(R)
    if N is not None:
        R2 = R2-(1-R2)/(N-2)
    #	diag(R2) = 1
    ldscMat = R2.dot(A)
    return(ldscMat)

