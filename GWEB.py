#!/usr/bin/env python3

from __init__ import *
import argparse
import sys, os, time, functools, gc, psutil
import textwrap
import logger
import PRSparser
import PRSalign
import PRScoring
import PRSeval
import numpy as np
import pandas as pd
import ldsc
import statsmodels.api as sm
import traceback
import pickle
# from sklearn.linear_model import LassoCV

window_length=74
title = '\033[96mGWEB v%s\033[0m'%version
title_len= len(title)-9
num_dashes = window_length-title_len-2
left_dashes = '='*int(num_dashes/2)
right_dashes = '='*int(num_dashes/2+num_dashes%2)
title_string = '\n'+left_dashes  + ' '+title+' '+right_dashes +'\n'

description_string = """
\033[1mGWEB\033[0m: An Empirical-Bayes-based polygenic risk prediction approach
using GWAS summary statistics and functional annotations

Typical workflow:
    0a*. Use\033[1m plinkLD.py\033[0m to calculate LD matrix for PLINK binary format 
    encoded genotype data of a reference panel. 
    See plinkLD.py --help for further usage description and options.

    0b*. Use\033[1m formatSS.py\033[0m to convert GWAS summary statistics from 
    different cohorts into the standard input format of GWEB.
    See formatSS.py --help for further usage description and options.

    \033[96m1. Use GWEB.py to obtain SNP weights for polygenic scoring. 
    See GWEB.py --help for further usage description and options.\033[0m
    
    2*. Use\033[1m scoring.py\033[0m to calculate polygenic scores for an external 
    individual-level genotype data using SNP weights from the previous step. 
    See scoring --help for further usage description and options.

    (*) indicates the step is optional.

(c) %s
%s

Thank you for using\033[1m GWEB\033[0m!
"""%(author,contact)
description_string += '='*window_length

parser = argparse.ArgumentParser(prog='GWEB',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=textwrap.dedent(description_string))
                                                             

#General arguments
parser.add_argument('--ssf', type=str, required=not ('--iprefix' in sys.argv),
                    help='GWAS Summary statistic File. '
                        'Should be a text file with columns SNP/CHR/BP/BETA/SE')

parser.add_argument('--ref', type=str, required=not ('--iprefix' in sys.argv),
                    help='Reference LD File. '
                         'Should be a (full path) hdf5 file storing the LD matrix and corresponding SNP information. (plinkLD.py can be used to convert PLINK binary files into the LD hdf5 file.)')

parser.add_argument('--bfile', type=str, default=None, required=False,
                    help='Individual-level genotype data for testing purpose. '
                        'Should be PLINK binary format files with extension .bed/.bim/.fam')

parser.add_argument('--bed', type=str, default=None, required=False,
                    help='Binary genotype data with suffix .bed (Used if bfile is not provided)')

parser.add_argument('--bim', type=str, default=None, required=False,
                    help='SNP infomation file with suffix .bim (Used if bfile is not provided)')

parser.add_argument('--fam', type=str, default=None, required=False,
                    help='Individual information file with suffix .fam (Used if bfile is not provided)')

parser.add_argument('--h5geno', type=str, default=None,required=False,
                    help='Individual-level genotype data with hdf5 format')

# parser.add_argument('--anno', type=str, required=False,
                    # help='Functional annotation file. '
                        # 'Should be a text file storing annotations for SNPs')

parser.add_argument('--anno', type=str, default=None, required=False,
                    help='Functional annotation file. '
                        'Should be a hdf5 file storing annotations for SNPs')

parser.add_argument('--snplist', type=str, default=None, required=False,
                    help='SNP list file used to filter SNPs')

parser.add_argument('--iprefix', type=str, default=None, required=False,
                    help='Common prefix for input files (summary statistics, reference panel, genotypes, annotations)')

parser.add_argument('--n', type=int, default=0, required=('--norm' in sys.argv),
                    help='Sample size of the GWAS summary statistics. (If provided, LDSC will be used to adjust the inflation caused by potential confounding effect.)')

parser.add_argument('--K', nargs='+', type=int, default=[3],
        help='Number of causal components (Default:3)')

parser.add_argument('--pheno', type=str, default=None, required=False,
                    help="External phenotype file."
                        "Should be a tabular text file. If header is not provided, the first and second columns should be FID and IID, respectively. Otherwise, there are two columns named 'FID' and 'IID'")

parser.add_argument('--mpheno', type=int, default=1, required=False,
                    help='m-th phenotype in the file to be used (default: 1)')

parser.add_argument('--pheno-name', type=str, default='PHE', required=False,
                    help='Column name for the phenotype in the phenotype file (default:PHE)')

parser.add_argument('--cov', type=str, default=None,
                    help='covariates file, format is tabulated file with columns FID, IID, PC1, PC2, etc.')

parser.add_argument('--dir', type=str, default='./output', required=False,
                    help='Output directory')

parser.add_argument('--aligned', default=False, action='store_true',
                    help='The input has already been aligned')

parser.add_argument('--align-only', default=False, action='store_true',
                    help='Align all input files only')

parser.add_argument('--weight-only', default=False, action='store_true',
                    help='Weighting only, without scoring and evaluation')

parser.add_argument('--thread', type=int, default=-1,
                    help='Number of parallel threads, by default all CPUs will be utilized.')

parser.add_argument('--target', type=str, default=None, required=False,
                    help="Summary statistics from target array")

parser.add_argument('--homo', default=False, action='store_true', help='If the summary statistics are from a single homogeneous GWAS cohort, no need to shrink LD')

parser.add_argument('--link-var', default=False, action='store_true',
                    help='Link variances of effect sizes with annotations ')

parser.add_argument('--link-both', default=False, action='store_true',
                    help='Link both variance of effect sizes and causal SNP proportions with annotations')

parser.add_argument('--rfreq', default=False, action='store_true',
                    help='Use allele frequencies inferred from reference panel to normalize the genotype')

parser.add_argument('--pos', default=False, action='store_true',
                    help='All files will be aligned by SNP position (chromosome and base pair position), instead of SNP identifier.')

parser.add_argument('--norm', default=False, action='store_true',
                    help='Use normalized genotypes to calculate PRS (Standard errors from marginal regressions will also be scaled based on given sample size)')

parser.add_argument('--split-by-chr', default=False, action='store_true',
                    help='Split analysis by chromosome')

parser.add_argument('--compress', type=int, default=9, required=False,
                    help='Compression level for output (default: 9)')

parser.add_argument('--savemem', default=False, action='store_true',
                    help='Save memory during the alignment phase but with slower speed')

parser.add_argument('--noanno', default=False, action='store_true',
                    help='Annotations are not used for estimation')

parser.add_argument('--annolist', type=str, default=None, required=False,
                    help="A text file including indices of included annotations.")

parser.add_argument('--init', type=str, default=None, required=False,
                    help='Initial param file for leveraging annotations')

parser.add_argument('--nosave', default=False, action='store_true',
                    help='Not saving the aligned files for reducing disk space usage')

parser.add_argument('--moment', default=False, action='store_true', help='Moment Estimator for estimating total variance')

parser.add_argument('--np', default=False, action='store_true', help='Using numpy instead of PLINK to calculate PRS')

parser.add_argument('--py', default=False, action='store_true',
                    help='Use python version for estimation (We recommend only to use this option when cython code fails to be compiled. The default cython version is faster)')

parser.add_argument('--mem', type=float, default=0, required=False,
                    help='Available memory in Gigabytes (Using maximum available memory by default)')

parser.add_argument('--nsample', type=int, default=100, required=False,
                    help='Number of samples of generated beta_est.')

parser.add_argument('--nburnin', type=int, default=100, required=False,
                    help='Number of samples of generated beta_est.')


#parser.add_argument('--debug', default=False, action='store_true', help='Debug mode')

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

def marginal_selection(X, y, threshold = 0.01, verbose=True, intercept=True, fixed_list=[], use_coeff=False, weight=None):
    """ Perform a marginal selection 
        based on p-value from statsmodels.api.OLS
        Arguments:
            X - pandas.DataFrame with candidate features
            y - list-like with the target
            threshold - include a feature if its p-value < threshold
            verbose - whether to print the sequence of inclusions and exclusions
        Returns: list of selected features 
    """
    print('Threshold:{:.6}'.format(threshold))
    excluded = list(set(X.columns)-set(fixed_list))
    new_pval = pd.Series(index=excluded, dtype=np.float64)
    new_coeff = pd.Series(index=excluded, dtype=np.float64)
    for new_column in excluded:
        if intercept:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X.loc[:,fixed_list+[new_column]]))).fit()
        else:
            model = sm.OLS(y, pd.DataFrame(X.loc[:,fixed_list+[new_column]])).fit()
        new_pval[new_column] = model.pvalues[new_column]
        if weight is None:
            new_coeff[new_column] = model.params[new_column]
        else:
            new_coeff[new_column] = model.params[new_column]*weight[new_column]

    included = []
    if not use_coeff:
        sorted_pval = new_pval.sort_values(ascending=True, inplace=False)
        for i, val in sorted_pval.items():
            if val<= threshold:
                if verbose:
                    print('Add  {:6} with p-value {:.6}'.format(i, val))
                included.append(i)
            else:
                break
    else:
        sorted_coeff = new_coeff.sort_values(ascending=False, inplace=False)
        for i, val in sorted_coeff.items():
            if val>= threshold:
                if verbose:
                    print('Add  {:6} with adjusted coefficient {:.6}'.format(i, val))
                included.append(i)
            else:
                break
    return included

#def stepwise_selection(X, y, initial_list=[], threshold=0.01, threshold_out = 0.05, verbose=True, intercept=True, fixed_list=[]):
def stepwise_selection(X, y, initial_list=[], threshold=0.01, verbose=True, intercept=True, fixed_list=[], use_coeff=False, weight=None):
    """ Perform a forward-backward feature selection 
        based on p-value from statsmodels.api.OLS
        Arguments:
            X - pandas.DataFrame with candidate features
            y - list-like with the target
            initial_list - list of features to start with (column names of X)
            threshold - include a feature if its p-value < threshold
            threshold_out - exclude a feature if its p-value > threshold_out
            verbose - whether to print the sequence of inclusions and exclusions
        Returns: list of selected features 
        Always set threshold < threshold_out to avoid infinite looping.
        See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    print('Threshold:{:.6}'.format(threshold))
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included)-set(fixed_list))
        new_pval = pd.Series(index=excluded, dtype=np.float64)
        new_coeff = pd.Series(index=excluded, dtype=np.float64)
        for new_column in excluded:
            if intercept:
                model = sm.OLS(y, sm.add_constant(pd.DataFrame(X.loc[:,fixed_list+included+[new_column]]))).fit()
            else:
                model = sm.OLS(y, pd.DataFrame(X.loc[:,fixed_list+included+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
            if weight is None:
                new_coeff[new_column] = model.params[new_column]
            else:
                new_coeff[new_column] = model.params[new_column]*weight[new_column]
        if not use_coeff:
            best_pval = new_pval.min()
            if best_pval < threshold:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed=True
                if verbose:
                    print('Add  {:6} with p-value {:.6}'.format(best_feature, best_pval))
        else:
            best_coeff = new_coeff.max()
            if best_coeff > threshold:
                best_feature = new_coeff.idxmax()
                included.append(best_feature)
                changed=True
                if verbose:
                    print('Add  {:6} with adjusted coefficient {:.6}'.format(best_feature, best_coeff))

        '''
        # backward step
        if intercept:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X.loc[:, fixed_list+included]))).fit()
            # use all coefs except intercept
            pvalues = model.pvalues.iloc[(len(fixed_list)+1):]
        else:
            model = sm.OLS(y, pd.DataFrame(X.loc[:, fixed_list+included])).fit()
            pvalues = model.pvalues.iloc[len(fixed_list):]

        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:6} with p-value {:.6}'.format(worst_feature, worst_pval))
        '''
        if not changed:
            break
    return included

def stagewise_selection(X, y, initial_list=[], threshold=0.05, verbose=True, intercept=True, fixed_list=[], use_coeff=False, weight=None):
    """ Perform a forward-stagewise feature selection 
        based on p-value from statsmodels.api.OLS
        Arguments:
            X - pandas.DataFrame with candidate features
            y - list-like with the target
            initial_list - list of features to start with (column names of X)
            threshold - include a feature if its p-value < threshold
            verbose - whether to print the sequence of inclusions and exclusions
        Returns: list of selected features 
        Always set threshold < threshold_out to avoid infinite looping.
        See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    # varX = X.var(axis=1, ddof=0)
    print('Threshold:{:.6}'.format(threshold))
    while True:
        changed=False
        if intercept:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X.loc[:, fixed_list+included]))).fit()
            res = model.resid
        else:
            if len(included) == 0:
                res = y
            else:
                model = sm.OLS(y, pd.DataFrame(X.loc[:, fixed_list+included])).fit()
                res = model.resid
        # forward step
        excluded = list(set(X.columns)-set(included)-set(fixed_list))
        new_pval = pd.Series(index=excluded, dtype=np.float64)
        new_coeff = pd.Series(index=excluded, dtype=np.float64)
        for new_column in excluded:
            if intercept:
                model = sm.OLS(res, sm.add_constant(pd.DataFrame(X.loc[:,new_column]))).fit()
            else:
                model = sm.OLS(res, pd.DataFrame(X.loc[:,new_column])).fit()
            new_pval[new_column] = model.pvalues[new_column]
            if weight is None:
                new_coeff[new_column] = model.params[new_column]
            else:
                new_coeff[new_column] = model.params[new_column]*weight[new_column]
        if not use_coeff:
            best_pval = new_pval.min()
            if best_pval < threshold:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed=True
                if verbose:
                    print('Add  {:6} with p-value {:.6}'.format(best_feature, best_pval))
        else:
            best_coeff = new_coeff.max()
            if best_coeff > threshold:
                best_feature = new_coeff.idxmax()
                included.append(best_feature)
                changed=True
                if verbose:
                    print('Add  {:6} with adjusted coefficient {:.6}'.format(best_feature, best_coeff))

        if not changed:
            break
    return included

def main_with_args(args):
    print(title_string)
    if len(args)<1:
        parser.print_usage()
        print(description_string)
        return
        
    startTime0 = time.time()
    parameters = parser.parse_args(args)
    p_dict= vars(parameters)
    #By default, only link causal proportions with annotations
    p_dict['fixp'] = False
    p_dict['fixv'] = True
    if p_dict['link_var']:
        p_dict['fixp'] = True
        p_dict['fixv'] = False
    if p_dict['link_both']:
        p_dict['fixp'] = False
        p_dict['fixv'] = False
    # if p_dict['debug']:
        # print ('Parsed parameters:')
        # print(p_dict)
    if p_dict['iprefix'] is not None:
        if os.path.exists(p_dict['iprefix']+'ssf.txt'): 
            p_dict['ssf'] = p_dict['iprefix']+'ssf.txt'
        if os.path.exists(p_dict['iprefix']+'ref.h5'): 
            p_dict['ref'] = p_dict['iprefix']+'ref.h5'
        if os.path.exists(p_dict['iprefix']+'geno.h5'): 
            p_dict['h5geno'] = p_dict['iprefix']+'geno.h5'
        if os.path.exists(p_dict['iprefix']+'anno.h5'): 
            p_dict['anno'] = p_dict['iprefix']+'anno.h5'
    if p_dict['noanno']:
        p_dict['anno'] = None
    sys.stdout = logger.logger(os.path.join(p_dict['dir'], 'log.txt'))
    print('Parsing GWAS summary statistics from', p_dict['ssf'])
    smryObj = PRSparser.smryParser(p_dict['ssf'])
    print('Reading Reference Panel from',p_dict['ref'])
    if p_dict['snplist'] is not None:
        print('Intersecting with SNP List file', p_dict['snplist'])
    if p_dict['aligned'] or not p_dict['savemem']:
        refObj, newLDfile = PRSparser.refParser(p_dict['ref'], snpListFile=p_dict['snplist'], thread=p_dict['thread'])
    else:
        refStore = pd.HDFStore(p_dict['ref'], 'r')
        tmpRefFile = os.path.join(p_dict['dir'], 'ref_tmp.h5')
        tmpRefStore = pd.HDFStore(tmpRefFile, 'w', complevel=p_dict['compress'])
        tmpRefStore, newLDfile = PRSparser.adjustRefStore(p_dict['ref'], refStore, tmpRefStore, snpListFile=p_dict['snplist'], thread=p_dict['thread'], complevel=p_dict['compress'])
        refStore.close()
    if not ((p_dict['h5geno'] is None) and (p_dict['bfile'] is None) and (p_dict['bim'] is None)):
        print('Parsing testing genotype data ...')
    if p_dict['h5geno'] is not None:
        genoStore = pd.HDFStore(p_dict['h5geno'], 'r')
        genoObj = {'SNPINFO': genoStore.get('SNPINFO'),'INDINFO': genoStore.get('INDINFO'), 'GENOTYPE': genoStore.get('GENOTYPE')[0], 'FLIPINFO': genoStore.get('FLIPINFO').to_numpy()}
        genoStore.close()
        print(len(genoObj['SNPINFO']),'SNPs')
        
        if PRSalign.isPdNan(genoObj['INDINFO']) or PRSalign.isPdNan(genoObj['GENOTYPE']):
            print('No individual in the genotype dataset')
        else:
            print(len(genoObj['INDINFO']),'Individuals')
        # del genoStore
        # gc.collect()
    else:
        genoObj = PRSparser.genoParser(bfile=p_dict['bfile'], bed=p_dict['bed'], bim=p_dict['bim'], fam=p_dict['fam'])
    
    if p_dict['anno'] is not None:
        withAnno=True
        print('Parsing annotation file from', p_dict['anno'])
        annoStore = pd.HDFStore(p_dict['anno'], 'r')
        annoObj = {'SNPINFO': annoStore.get('SNPINFO'), 'ANNODATA': annoStore.get('ANNODATA')}
        print(len(annoObj['SNPINFO']),'SNPs')
        print(annoObj['ANNODATA'].shape[1],'functional annotations')
        annoStore.close()
    # elif p_dict['anno'] is not None:
        # withAnno=True
        # print('Parsing annotation file from', p_dict['anno'])
        # annoObj = PRSparser.annoParser(p_dict['anno'])
    else:
        withAnno=False
        annoObj = None
    if p_dict['aligned']:
        if genoObj is None:
            withGeno = False
            alignResult = {'SS': smryObj, 'REF': refObj, 'ANNO': annoObj}
        else:
            withGeno = not(PRSalign.isPdNan(genoObj['INDINFO']) or PRSalign.isPdNan(genoObj['GENOTYPE']))
            alignResult = {'SS': smryObj, 'GENO': genoObj, 'REF': refObj, 'ANNO': annoObj}
        
    else:
        print('Start aligning all files...')
        alignRefFile = os.path.join(p_dict['dir'], 'align_ref.h5')
        if genoObj is None:
            #Without testing genotype data
            withGeno = False
            if not p_dict['savemem']:
                if (not p_dict['nosave']) or p_dict['align_only']:
                    alignResult = PRSalign.alignSmry2RefPointer(smryObj, refObj, newLDfile, annoObj=annoObj, outSS = os.path.join(p_dict['dir'], 'align_ssf.txt'), outRef=alignRefFile, outAnno=os.path.join(p_dict['dir'],'align_anno.txt'), byID=(not p_dict['pos']), complevel=p_dict['compress']) 
                else:
                    alignResult = PRSalign.alignSmry2RefPointer(smryObj, refObj, newLDfile, annoObj=annoObj, outSS = None, outRef=None, outAnno=None, byID=(not p_dict['pos']), complevel=p_dict['compress']) 
            else:
                outRefStore = pd.HDFStore(alignRefFile, 'w', complevel=p_dict['compress'])
                if (not p_dict['nosave']) or p_dict['align_only']:
                    alignResult = PRSalign.alignSmry2RefStorePointer(smryObj, tmpRefStore, outRefStore, newLDfile, annoObj=annoObj, outSS = os.path.join(p_dict['dir'], 'align_ssf.txt'), outAnno=os.path.join(p_dict['dir'],'align_anno.txt'), byID=(not p_dict['pos']), complevel=p_dict['compress'], thread=p_dict['thread'])  
                else:
                    alignResult = PRSalign.alignSmry2RefStorePointer(smryObj, tmpRefStore, outRefStore, newLDfile, annoObj=annoObj, outSS = None, outAnno=None, byID=(not p_dict['pos']), complevel=p_dict['compress'], thread=p_dict['thread'])  
                tmpRefStore.close()
                os.remove(tmpRefFile)
                print('Saving aligned reference panel to', alignRefFile)
            # del smryObj, refObj,annoObj
        else:
            withGeno = not(PRSalign.isPdNan(genoObj['INDINFO']) or PRSalign.isPdNan(genoObj['GENOTYPE']))
            if not p_dict['savemem']:
                if (not p_dict['nosave']) or p_dict['align_only']:
                    alignResult = PRSalign.alignSmryGeno2RefPointer(smryObj, genoObj, refObj, newLDfile, annoObj=annoObj, outSS = os.path.join(p_dict['dir'], 'align_ssf.txt'), outGeno=os.path.join(p_dict['dir'],'align_geno.h5'), outRef=alignRefFile, outAnno=os.path.join(p_dict['dir'],'align_anno.h5'), byID=(not p_dict['pos']), complevel=p_dict['compress'])  
                else:
                    alignResult = PRSalign.alignSmryGeno2RefPointer(smryObj, genoObj, refObj, newLDfile, annoObj=annoObj, outSS = None, outGeno=None, outRef=None, outAnno=None, byID=(not p_dict['pos']), complevel=p_dict['compress'])  

            else:
                outRefStore = pd.HDFStore(alignRefFile, 'w', complevel=p_dict['compress'])
                if (not p_dict['nosave']) or p_dict['align_only']:
                    alignResult = PRSalign.alignSmryGeno2RefStorePointer(smryObj, genoObj, tmpRefStore, outRefStore, newLDfile, annoObj=annoObj, outSS = os.path.join(p_dict['dir'], 'align_ssf.txt'), outGeno=os.path.join(p_dict['dir'],'align_geno.h5'), outAnno=os.path.join(p_dict['dir'],'align_anno.h5'), byID=(not p_dict['pos']), complevel=p_dict['compress'])
                else:
                    alignResult = PRSalign.alignSmryGeno2RefStorePointer(smryObj, genoObj, tmpRefStore, outRefStore, newLDfile, annoObj=annoObj, outSS = None, outGeno=None, outAnno=None, byID=(not p_dict['pos']), complevel=p_dict['compress'], thread=p_dict['thread'])

                tmpRefStore.close()
                os.remove(tmpRefFile)
                outRefStore.close()
                if (not p_dict['nosave']) or p_dict['align_only']:
                    print('Saving aligned reference panel to', alignRefFile)
            # del smryObj, genoObj, refObj, annoObj
        print('After alignment,', len(alignResult['SS']),'SNPs remaining')
        

    # gc.collect()
    
    if not p_dict['align_only']:
        if (not (p_dict['aligned'])):
            print('Reloading aligned reference panel...')
            outRefStore = pd.HDFStore(alignRefFile, 'r')
            alignResult['REF'], _ = PRSparser.refStoreParser(alignRefFile, outRefStore, thread=p_dict['thread'])
            outRefStore.close()
            if p_dict['nosave']:
                os.remove(alignRefFile)

        if 'N' in alignResult['SS'].columns:
            rawN = p_dict['n']
            N90 = np.quantile(alignResult['SS']['N'], 0.9)
            p_dict['n'] = int(N90)
            print('Use 90% quantile of SNP-level sample size:', p_dict['n'])
        z = alignResult['SS']['BETA']/alignResult['SS']['SE']
        inf_gc = np.median(z**2) / 0.455
        print('Genomic Control:', "{:.4g}".format(inf_gc))
        alignResult['REF'] = PRSalign.updateLDSC(alignResult['REF'])
        ldscVal = extractRef(alignResult['REF'], col='LDSC')
        if p_dict['n']!= 0:
            ldscResult = ldsc.calH2_2stage(z, ldscVal, p_dict['n'])
            print('Estimated h2 from LDSC:', "{:.4g}".format(ldscResult['h2']))
            inf_ldsc = 1.+p_dict['n']*ldscResult['a']
            totalNormVar = ldscResult['h2']*p_dict['n']/len(z)
            print('Intercept from LDSC:', "{:.4g}".format(inf_ldsc))
        else:
            # print('Sample size of summary statistics is not provided and LDSC is not run for adjusting confounding inflation!')
            newLDSCresult = np.linalg.lstsq(np.vstack((np.ones(len(z)),ldscVal)).T, z**2, rcond=None)[0]
            inf_ldsc = newLDSCresult[0]
            totalNormVar = newLDSCresult[1]
            print('Estimated confounding inflation factor:', "{:.4g}".format(inf_ldsc))
        # adj = 1.
        # if inf_ldsc<1.1:
            # print('No need to adjust inflation!')
        # else:
        alignResult['SS']['BETA'] = alignResult['SS']['BETA']/np.sqrt(inf_ldsc)
        totalNormVar = totalNormVar/inf_ldsc
        adj = inf_ldsc
            # print('The confounding inflation has been adjusted!')
        
        print('Start estimation...')
        
        rawRefObj = alignResult['REF']
        if p_dict['target'] is not None:
            print('Adjusting reference panel based on target array...')
            targetSS = PRSparser.smryParser(p_dict['target'], noFilter=True)
            alignResult['REF'] = PRSalign.adjustRefByTarget(alignResult['SS'], targetSS, alignResult['REF'], p_dict['thread'])
        if (p_dict['norm']) and (p_dict['n'] != 0):
            alignResult['SS']['BETA'] = z/np.sqrt(p_dict['n'])
            alignResult['SS']['SE'] = np.ones(len(alignResult['SS']['SE']))/np.sqrt(p_dict['n'])
        
        if p_dict['py']:
            import GWEButils
        else:
            import GWEButils_cy as GWEButils
        
        estResultList = []
        methodList = []
        minBIC = np.inf
        minBICidx = 0
        initDf = pd.DataFrame(columns = ['beta', 'gamma'])
        #Without annotation
        if p_dict['init'] is not None:
            try:
                initDf = pd.read_table(p_dict['init'], sep='\t|\s+', usecols=['beta','gamma'], engine='python')
            except:
                print("Unable to read initial params from file:", p_dict['init'])
            p_dict['K'] = []

        #with open(os.path.join(p_dict['dir'], 'SS_LD.obj'), 'wb') as file:
        #    pickle.dump(alignResult['REF']['LD'], file)

        # Add 0.1 to LD variance part for stability
        if (not p_dict['homo']):
            penality = 0.1
            #The penality will increase when most SNPs genotyped on smaller sample size
            if 'N' in alignResult['SS'].columns:
                # N90 = np.quantile(alignResult['SS']['N'], 0.9)
                if rawN==0:
                    rawN = alignResult['SS']['N'].max()
                # if N90/N == 1:
                    # penality = 0
                if N90/rawN<=(2/3):
                    penality = 0.2
            for bk in range(len(alignResult['REF']['LD'])):
                alignResult['REF']['LD'][bk] = (1-penality)*alignResult['REF']['LD'][bk]+penality*np.identity(alignResult['REF']['LD'][bk].shape[0])

        for iterK in np.arange(len(p_dict['K'])):
            K = p_dict['K'][iterK]
            methodName = 'K'+str(K)
            print('======Without annotation data, K=', K,'======', sep='')
            for iterRep in range(5):
                startTime1 = time.time()
                if iterRep == 4: momentEst = True
                else: momentEst = p_dict['moment']
                # estResult = GWEButils.gibbsEst(alignResult['SS']['BETA'], alignResult['SS']['SE'], alignResult['REF']['LD'], ldscVal, N=p_dict['n'], K=K, thread=p_dict['thread'], adj=adj)
                try:
                    nsample = p_dict['nsample']
                    nburnin = p_dict['nburnin']
                    estResult = GWEButils.ebEst(alignResult['SS']['BETA'], alignResult['SS']['SE'],
                                                alignResult['REF']['LD'], ldscVal, N=p_dict['n'], K=K,
                                                thread=p_dict['thread'], momentEst=momentEst, adj=adj, nsample=nsample,
                                                burnin=nburnin)
                except Exception as e:
                    print('Error:'+str(e))
                    print('Rerunning the estimation')
                    continue
                if estResult is not None:
                    print('Estimation completed! Time elapsed:',time.time()-startTime1,'s')
                    break
                else:
                    #Increasing penality
                    for bk in range(len(alignResult['REF']['LD'])):
                        alignResult['REF']['LD'][bk] += 0.1*np.identity(alignResult['REF']['LD'][bk].shape[0])
            if estResult is not None:
                methodList.append(methodName)
                estResultList.append(estResult)
                paramDF = pd.DataFrame({'pi': estResult['piEst'], 'sigma2': estResult['sigma2Est'] })
                paramFile = os.path.join(p_dict['dir'], methodName+'_param.txt')
                paramDF.to_csv(paramFile, sep='\t')
                print('Estimated parameter values are saved to', paramFile)
                if 'GENO' in alignResult:
                    weightObj = alignResult['GENO']['SNPINFO'][['RAW_SNP', 'RAW_A1']].copy()
                    weightObj.rename(columns={"RAW_SNP": "SNP", "RAW_A1": "A1"})
                    weightObj.loc[:, 'BETAJ'] = estResult['betaEst']
                    weightObj.loc[alignResult['GENO']['FLIPINFO'],'BETAJ'] = -weightObj.loc[alignResult['GENO']['FLIPINFO'],'BETAJ']
                else:
                    weightObj = alignResult['SS'].loc[:,['SNP','A1']].copy()
                    weightObj.loc[:,'BETAJ'] = estResult['betaEst']
                try:
                    weightObj.to_csv(os.path.join(p_dict['dir'], methodName+'_weight.txt'), sep='\t', index=False, header=False)
                except:
                    print('Can\'t write weights into file:', os.path.join(p_dict['dir'], methodName+'_weight.txt'))
                if 'GENO' in alignResult:
                    pipObj = alignResult['GENO']['SNPINFO'][['SNP', 'A1']].copy()
                else:
                    pipObj = alignResult['SS'].loc[:,['SNP','A1']].copy()
                pipObj.loc[:,'PIP'] = 1.-estResult['gammaEst'][0,:]
                try:
                    pipObj.to_csv(os.path.join(p_dict['dir'], methodName+'_pip.txt'), sep='\t', index=False, header=False)
                except:
                    print('Can\'t write pips into file:', os.path.join(p_dict['dir'], methodName+'_pip.txt'))

                betaInit = estResult['beta'][-1,:]
                gammaInit =(1.-estResult['gamma'][-1,0,:])
                tmpInitDf = pd.DataFrame(data={'beta': betaInit, 'gamma': gammaInit})
                if estResult['BIC']<minBIC:
                    minBIC = estResult['BIC']
                    minBICidx = iterK
                    initDf = tmpInitDf
                try:
                    tmpInitDf.to_csv(os.path.join(p_dict['dir'], methodName+'_init.txt'), sep='\t', index=False, header=True)
                except:
                    print('Can\'t write initial params into file:', os.path.join(p_dict['dir'], methodName+'_init.txt'))
                try:
                    np.savetxt(os.path.join(p_dict['dir'], methodName +'_beta_sample.txt'), estResult['beta'], fmt="%f",
                               delimiter="\t")
                    print('Beta samples are saved to', os.path.join(p_dict['dir'], methodName +'_beta_sample.txt'))
                except:
                    print('Can\'t write beta into file:', os.path.join(p_dict['dir'], methodName +'_beta_sample.txt'))
                try:
                    print(f"Estimated causal proportion is {estResult['causalPropEst']}. Estimated total variance is "
                          f"{estResult['totalVarEst']}. BIC is {estResult['BIC']}.")
                except:
                    print('Can\'t print estimated values.')
                try:
                    with open(os.path.join(p_dict['dir'], methodName + '_alignResult.obj'), 'wb') as file:
                        pickle.dump(alignResult, file)
                    f.close()
                    print("Saved aligned result to file.")
                except Exception as e:
                    print(e)
            else:
                print("Five times tried! Can't find a converged chain.")

        if withAnno:
            print('======With annotation data======')
            A = alignResult['ANNO']['ANNODATA'].to_numpy().transpose()

            if p_dict['annolist'] is not None:
                topAnno = []
                with open(p_dict['annolist']) as annolistFile:
                    topAnnoList = annolistFile.readlines()
                    topAnno = [int(elem.strip()) for elem in topAnnoList]
            else:
                print('Selecting informative annotations...')
                # ldscMat = np.outer(np.ones(A.shape[0], dtype=np.float16), ldscVal)
                ldscMat = PRSalign.getAnnoLDSC(rawRefObj, alignResult['ANNO'])
                #annoDF = pd.DataFrame(A.transpose()*ldscMat, columns = np.arange(A.shape[0]))
                annoDF = pd.DataFrame(ldscMat, columns = np.arange(A.shape[0]))
                annoWeight = pd.Series(np.sum(A, axis=1)/A.shape[1], index=annoDF.columns.values)
                annoDF['ldsc'] = ldscVal
                annoWeight['ldsc'] = 1
                topAnno = stagewise_selection(annoDF, (alignResult['SS']['BETA']/alignResult['SS']['SE'])**2-1, threshold=totalNormVar/annoDF.shape[1], intercept=False, fixed_list = ['ldsc'], use_coeff=True, weight=annoWeight)

                # annoDF = pd.DataFrame(alignResult['ANNO']['ANNODATA'], columns=np.arange(A.shape[0]))
                # topAnno = stagewise_selection(annoDF, (alignResult['SS']['BETA']/alignResult['SS']['SE'])**2, threshold=0.05/annoDF.shape[1],intercept=True)

                # topAnno = marginal_selection(annoDF, (alignResult['SS']['BETA']/alignResult['SS']['SE'])**2, intercept=True)

                # pip = 1.-estResultList[minBICidx]['gammaEst'][0,:]
                # topAnno = stepwise_selection(annoDF, pip)

                # corrVal = np.zeros(A.shape[0])
                # for iterA in range(A.shape[0]):
                    # corrVal[iterA] = np.corrcoef(pip, A[iterA, :])[0,1]
                # topAnno = np.argpartition(np.abs(corrVal), -1)[-1:]

                # reg = LassoCV(cv=5).fit(A.T, pip)
                # topAnno = np.arange(A.shape[0])[np.abs(reg.coef_)>1e-6]

                # topAnno = list(range(A.shape[0]))

            if len(topAnno)>0:
                print('Annotation(s)', topAnno,'are selected')
            else:
                print('No annotations are selected')
                withAnno = False
            with open(os.path.join(p_dict['dir'],'anno_index.txt'), 'w') as annolistFile:
                topAnnoList = [str(elem)+'\n' for elem in topAnno]
                annolistFile.writelines(topAnnoList)

        hasAnnoResult = False
        if withAnno:
            A = A[topAnno, :]
            methodName = 'anno'
            for iterRep in range(5):
                startTime1 = time.time()
                if iterRep == 4: momentEst = True
                else: momentEst = p_dict['moment']
                try:
                    estResult = GWEButils.ebEstAnno(alignResult['SS']['BETA'], alignResult['SS']['SE'], alignResult['REF']['LD'],
                                                    ldscVal, A, N=p_dict['n'], fixProp=p_dict['fixp'], fixVar=p_dict['fixv'],
                                                    thread=p_dict['thread'], betaInit = initDf['beta'], gammaInit = initDf['gamma'],
                                                    momentEst=momentEst, adj=adj, nsample=nsample, burnin=nburnin)
                except Exception as e:
                    print('Error:'+str(e))
                    print('Rerunning the estimation')
                    continue
                if estResult is not None:
                    print('Estimation completed! Time elapsed:',time.time()-startTime1,'s')
                    break
                else:
                    #Increasing constraint for l2 norm of beta
                    for bk in range(len(alignResult['REF']['LD'])):
                        alignResult['REF']['LD'][bk] += 0.1*np.identity(alignResult['REF']['LD'][bk].shape[0])
            if estResult is not None:
                hasAnnoResult = True
                methodList.append(methodName)
                estResultList.append(estResult)
                paramDF = pd.DataFrame({'alpha': estResult['alphaEst'], 'eta': estResult['etaEst'] }, index=np.insert(topAnno, 0, -1))
                paramFile = os.path.join(p_dict['dir'], 'anno_param.txt')
                paramDF.to_csv(paramFile, sep='\t')
                print('Estimated parameter values are saved to', paramFile)
                weightObj = alignResult['GENO']['SNPINFO'][['RAW_SNP', 'RAW_A1']].copy()
                weightObj.rename(columns={"RAW_SNP": "SNP", "RAW_A1": "A1"})
                weightObj.loc[:, 'BETAJ'] = estResult['betaEst']
                weightObj.loc[alignResult['GENO']['FLIPINFO'],'BETAJ'] = -weightObj.loc[alignResult['GENO']['FLIPINFO'],'BETAJ']
                try:
                    weightObj.to_csv(os.path.join(p_dict['dir'], 'anno_weight.txt'), sep='\t', index=False, header=False)
                except:
                    print('Can\'t write weights into file:', os.path.join(p_dict['dir'], 'anno_weight.txt'))
                pipObj = alignResult['GENO']['SNPINFO'][['SNP', 'A1']].copy()
                pipObj.loc[:,'PIP'] = estResult['gammaEst']
                try:
                    pipObj.to_csv(os.path.join(p_dict['dir'], 'anno_pip.txt'), sep='\t', index=False, header=False)
                except:
                    print('Can\'t write pips into file:', os.path.join(p_dict['dir'], 'anno_pip.txt'))
                annoBIC = estResult['BIC']

                betaInit = estResult['beta'][-1,:]
                gammaInit = estResult['gamma'][-1,:]
                tmpInitDf = pd.DataFrame(data={'beta': betaInit, 'gamma': gammaInit})
                try:
                    tmpInitDf.to_csv(os.path.join(p_dict['dir'], methodName+'_init.txt'), sep='\t', index=False, header=True)
                except:
                    print('Can\'t write initial params into file:', os.path.join(p_dict['dir'], methodName+'_init.txt'))
            else:
                print("Five times tried! Can't find a converged chain.")

        if withGeno and (not p_dict['weight_only']):
            print('=======================')
            if p_dict['rfreq']:
                af = extractRef(alignResult['REF'], col='F')
            else:
                af = None

            print('Getting PRS for genotype data...')
            if p_dict['mem'] is not None:
                mem = p_dict['mem']-sys.getsizeof(alignResult)/(1024**3)
            else:
                mem = None
            if (af is None) and (p_dict['np']):
                af = PRScoring.getAlleleFreq(alignResult['GENO'], mem=mem)

            scoreDFlist = []
            for iterEst in range(len(estResultList)):
                estResult = estResultList[iterEst]
                methodName = methodList[iterEst]
                if not p_dict['np']:
                    score = PRScoring.scoringByPlink(alignResult['GENO'], estResult['betaEst'], splitByChr=p_dict['split_by_chr'], out=os.path.join(p_dict['dir'],methodName+'_'), thread=p_dict['thread'])
                else:
                    score = PRScoring.scoring(alignResult['GENO'], estResult['betaEst'], f=af, splitByChr=p_dict['split_by_chr'], normalize=p_dict['norm'], mem=mem)
                scoreDF = score.copy()
                scoreDF.insert(loc=0, column='FID', value=alignResult['GENO']['INDINFO']['FID'])
                scoreDF.insert(loc=1, column='IID', value=alignResult['GENO']['INDINFO']['IID'])
                scoreDFlist.append(scoreDF)
                try:
                    scoreDF.to_csv(os.path.join(p_dict['dir'], methodName+'_PRS.txt'), sep='\t', index=False)
                except:
                    print('Unable to write risk scores into file:', os.path.join(p_dict['dir'],methodName+'_PRS.txt'))

            print('Extracting phenotype...')
            phenoDF = PRSeval.phenoParser(alignResult['GENO'], phenoFile=p_dict['pheno'], PHE=p_dict['pheno_name'], mpheno=p_dict['mpheno'])
            isBinPhe = False
            if len(phenoDF)>0:
                isBinPhe = PRSeval.isBinary(phenoDF.iloc[:,2])
                if isBinPhe:
                    phenoDF.iloc[:,2] = PRSeval.convert01(phenoDF.iloc[:,2].to_numpy())
                    print(sum(phenoDF.iloc[:,2]==0),'controls,',sum(phenoDF.iloc[:,2]==1),'cases')
                else:
                    print(len(phenoDF),'individuals')

                if p_dict['cov'] is not None:
                    print('Extracting covariates...')
                    covDF = PRSeval.covParser(p_dict['cov'])

                r2Df_minBIC = 0
                r2Df_anno = 0
                aucDf_minBIC = 0
                aucDf_anno = 0
                for iterEst in range(len(estResultList)):
                    methodName = methodList[iterEst]
                    scoreDF = scoreDFlist[iterEst]
                    if p_dict['cov'] is not None:
                        newScoreDF, newPhenoDF, newCovDF = PRSeval.overlapIndDf([scoreDF, phenoDF, covDF])
                    else:
                        newScoreDF, newPhenoDF = PRSeval.overlapIndDf([scoreDF, phenoDF])
                    score = newScoreDF.iloc[:,2:]
                    phe = newPhenoDF.iloc[:,2]
                    if p_dict['cov'] is not None:
                        covData = newCovDF.iloc[:,2:]
                    else:
                        covData = None

                    if len(phe)>0:
                        if isBinPhe:
                            aucDict = {}
                        r2Dict = {}
                        print('======',iterEst+1,'. ',methodName,'======', sep='')
                        if covData is not None:
                            pred01 = PRSeval.fit(phe, covData=covData)
                            PRSeval.strataPlot(phe, pred01['yhat'], figname=os.path.join(p_dict['dir'],methodName+'_strata_cov.pdf'))
                            if isBinPhe:
                                aucObj01 = PRSeval.auc(phe, pred01['yhat'], figname=os.path.join(p_dict['dir'], methodName+'_roc_cov.pdf'))
                                print('AUC based on covariates only: %.3f'%aucObj01['auc']+'(%.3f'%aucObj01['se']+')')

                        for label, prs in score.items():
                            r2obj = PRSeval.rsq(phe, prs, covData=covData)
                            r2Dict[label] = r2obj['r2'].map('{:,.4f}'.format)+'('+r2obj['se'].map('{:,.4f}'.format)+')'
                            pred10 = PRSeval.fit(phe, prs=prs)

                            PRSeval.strataPlot(phe, pred10['yhat'], figname=os.path.join(p_dict['dir'], methodName+'_strata_'+label+'_prs.pdf'))

                            if covData is not None:
                                pred11 = PRSeval.fit(phe, prs=prs, covData=covData)
                                PRSeval.strataPlot(phe, pred11['yhat'], figname=os.path.join(p_dict['dir'], methodName+'_strata_'+label+'_prs_cov.pdf'))
                            if isBinPhe:
                                aucObj10 = PRSeval.auc(phe, pred10['yhat'], figname=os.path.join(p_dict['dir'], methodName+'_roc_'+label+'_prs.pdf'))
                                aucDict[label] = ['%.3f'%(aucObj10['auc'])+'('+'%.3f'%(aucObj10['se'])+')']
                                if covData is not None:
                                    aucObj11 = PRSeval.auc(phe, pred11['yhat'], figname=os.path.join(p_dict['dir'], methodName+'_roc_'+label+'_prs_cov.pdf'))
                                    aucDict[label].append('%.3f'%(aucObj11['auc'])+'('+'%.3f'%(aucObj11['se'])+')')
                                    aucDict[label].append('%.3f'%(aucObj11['auc']-aucObj01['auc'])+'('+'%.3f'%(np.sqrt(aucObj11['se']**2+aucObj01['se']**2))+')')
                                    aucDict[label].append('%.3f'%((aucObj11['auc']-aucObj01['auc'])/aucObj01['auc']))

                        if covData is not None:
                            r2Df = pd.DataFrame.from_dict(r2Dict, orient='index', columns=['raw','McF','McF-adj','McF-all'])
                        else:
                            r2Df = pd.DataFrame.from_dict(r2Dict, orient='index', columns=['raw','McF'])

                        pd.set_option("display.max_rows", None, "display.max_columns", None)
                        print('Predictive r2:')
                        print(r2Df)
                        if iterEst == minBICidx:
                            r2Df_minBIC = r2Df
                        if hasAnnoResult and (iterEst == len(estResultList)-1):
                            r2Df_anno = r2Df
                        if isBinPhe:
                            print('AUC:')
                            if covData is not None:
                                aucDf = pd.DataFrame.from_dict(aucDict, orient='index', columns=['prs','all','inc','inc%'])
                            else:
                                aucDf = pd.DataFrame.from_dict(aucDict, orient='index', columns=['prs'])
                            print(aucDf)
                            if iterEst == minBICidx:
                                aucDf_minBIC = aucDf
                            if hasAnnoResult and (iterEst == len(estResultList)-1):
                                aucDf_anno = aucDf

                print('===============Summary===============')
                pd.set_option("display.max_rows", None, "display.max_columns", None)
                if len(p_dict['K'])>0:
                    print('Without Annotations, BIC=', "{:.4g}".format(minBIC), '(',methodList[minBICidx],')')
                    print('Predictive r2:')
                    print(r2Df_minBIC)
                    if isBinPhe:
                        print('AUC:')
                        print(aucDf_minBIC)
                if hasAnnoResult:
                    if annoBIC<minBIC:
                        print('With Annotations, BIC=', "{:.4g}".format(annoBIC))
                    else:
                        print("***Note: Including annotations doesn't improve BIC, BIC=","{:.4g}".format(annoBIC))
                    print('Predictive r2:')
                    print(r2Df_anno)
                    if isBinPhe:
                        print('AUC:')
                        print(aucDf_anno)
            else:
                print('No phenotypes extracted!')
            # PRSparser.removeTmp(bfile=p_dict['bfile'], bed=p_dict['bed'], bim=p_dict['bim'], fam=p_dict['fam'])

    print('Completed! Total time elapsed:',time.time()-startTime0,'s')
    print('Memory usage:',"{:.1g}".format(psutil.Process(os.getpid()).memory_info().rss/(1024**2)),"MB")
    print('Thank you for using GWEB!')


def main():
    main_with_args(sys.argv[1:])

if __name__ == '__main__':
    startTime = time.time()
    main_with_args(sys.argv[1:])
    endTime = time.time()
    print('Time cost', endTime - startTime, 's')



