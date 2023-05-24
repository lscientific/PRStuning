#!/usr/bin/env python

from __init__ import *
import argparse
import sys, os
import textwrap
from PRSparser import smryParser, recoverGWAS
import pandas as pd
import numpy as np
from scipy.stats import norm
import logger

window_length=74
title = '\033[96mformatSS\033[0m'
title_len= len(title)-9
num_dashes = window_length-title_len-2
left_dashes = '='*int(num_dashes/2)
right_dashes = '='*int(num_dashes/2+num_dashes%2)
title_string = '\n'+left_dashes  + ' '+title+' '+right_dashes +'\n'

description_string = """
    A general program to parse summary statistics from 
    different GWAS cohorts into standard input formats of different 
    analysis softwares
    
    See\033[1m formatSS.py --help\033[0m for further usage description and options.

(c) %s
%s

Thank you for using\033[1m formatSS\033[0m!
"""%(author,contact)
description_string += '='*window_length

parser = argparse.ArgumentParser(prog='formatSS',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=textwrap.dedent(description_string))

#arguments
# parser.add_argument('--debug', default=False, action='store_true',
                    # help="Activate debugging mode (more verbose)")

parser.add_argument('--ssf', type=str, required=True,
                    help='Summary Statistic File. '
                         'Filename for a text file with the GWAS summary statistics')
parser.add_argument('--snp', type=str, default='SNP',
                    help='Column name for SNP identifier')

parser.add_argument('--chr', type=str, default='CHR',
                    help='Column name for chromosome')

parser.add_argument('--bp', type=str, default='BP',
                    help='Column name for base pair position')

parser.add_argument('--chrbp', type=str, default='CHR_BP',
                    help="Column name for position with 'chr:bp' format")

parser.add_argument('--a1', type=str, default='A1',
                    help='Column name for reference allele')

parser.add_argument('--a2', type=str, default='A2',
                    help='Column name for alternative allele')

parser.add_argument('--a1a2', type=str, default='A1A2',
                    help="Column name for alleles with 'A1A2' format")

parser.add_argument('--chrbpa1a2', type=str, default='CHR:BP_A1_A2',
                    help="Column name for position with 'chr:bp_a1_a2' format")

parser.add_argument('--beta', type=str, default='BETA',
                    help='Column name for effect size from marginal linear/logistic regression or log(odds ratio)')

parser.add_argument('--or', type=str, default='OR',
                    help='Column name for odds ratio')

parser.add_argument('--se', type=str, default='SE',
                    help='Column name for standard error of BETA')

parser.add_argument('--ci', type=str, default='CI',
                    help="Column name for 95%% confidence interval of BETA (sep by '-')")

parser.add_argument('--or-ci', type=str, default='OR_CI',
                    help="Column name for 95%% confidence interval of OR (sep by '-')")

parser.add_argument('--l95', type=str, default='L95',
                    help="Column name for lower limit of 95%% confidence interval of BETA")
parser.add_argument('--u95', type=str, default='U95',
                    help="Column name for upper limit of 95%% confidence interval of BETA")
parser.add_argument('--or-l95', type=str, default='OR_L95',
                    help="Column name for lower limit of 95%% confidence interval of OR")
parser.add_argument('--or-u95', type=str, default='OR_U95',
                    help="Column name for upper limit of 95%% confidence interval of OR")

parser.add_argument('--p', type=str, default='P',
                    help='Column name for marginal p-value')

parser.add_argument('--z', type=str, default='Z',
                    help='Column name for marginal z-score')

parser.add_argument('--n', type=str, default='N',
                    help='Number of individuals or corresponding column name in GWAS summary statistic file.')

parser.add_argument('--n0', type=str, default='N0',
                    help='Number of controls or corresponding column name in GWAS summary statistic file.')

parser.add_argument('--n1', type=str, default='N1',
                    help='Number of cases or corresponding column name in GWAS summary statistic file.')

parser.add_argument('--eff', default=False, action='store_true',
        help='Use effective sample size as individual number (default: total sample size)')

parser.add_argument('--skip', type=int, default=0,
                    help='Number of lines to skip (int) at the start of the file.')

parser.add_argument('--info', type=str, default='INFO',
                    help='Column name for INFO score derived from genotype imputation')

parser.add_argument('--freq', type=str, default='F',
                    help='Column name for allele frequency of A1')

parser.add_argument('--freqA', type=str, default='FA',
                    help='Column name for allele frequency of A1 among cases')

parser.add_argument('--freqU', type=str, default='FU',
                    help='Column name for allele frequency of A1 among controls')

parser.add_argument('--phet', type=str, default='P_HET',
                    help='Column name for P values of Heterogeneity Tests in Meta-analysis')

parser.add_argument('--chrlist', nargs='+', type=str, default=None,
        help='Selecting SNPs within the chromosome list (e.g., 1-22, default:None)')

parser.add_argument('--infothld', type=float, default=0.9,
        help='Threshold for INFO to filter SNPs (Default:0.9)')

parser.add_argument('--maf', type=float, default=0.05,
        help='Threshold for minor allele frequency to filter SNPs (Default:0.05)')

parser.add_argument('--hetthld', type=float, default=0,
        help='Threshold for P_HET to filter SNPs (Default:0)')

parser.add_argument('--z2thld', type=float, default=np.inf,
        help='Threshold for z^2 to filter SNPs with extreme large effects (Default: No filtering)')

parser.add_argument('--Nthld', type=float, default=0.67,
        help='The ratio threshold with respect to 90\% sample size quantile for filtering SNPs with small sample size (Default: 0.67)') 
        # help='The ratio threshold with respect to 90\% sample size quantile for filtering SNPs with small sample size (Default: 0.67; However, if the summary statistics of target array are provided and the SNPs with small sample sizes have inflated test statistics due to polygenicity, SNPs are not filtered)')

parser.add_argument('--target', type=str, required=False, default=None,
        help='summary statistics from a target array')

'''
parser.add_argument('--ref', type=str, required=False, default=None,
        help='Reference panel used to derive LD scores for detecting confounding inflation')

parser.add_argument('--thread', type=int, default=-1,
                    help='Number of parallel threads for loading reference panel, by default all CPUs will be utilized.')
'''
parser.add_argument('--nmax', type=float, default=None,
        help='Maximum individual number of GWAS (Default:None)')

parser.add_argument('--ifmt', type=str, default='basic',
                    help='Input format from certain consortia (basic[default]/standard/PGC/PGC2/GIANT/GIANT2/GWEB)')

parser.add_argument('--ofmt', type=str, default='GWEB',
                    help='Output format (GWEB[default]/PRSCS/EBPRS)')

parser.add_argument('--out', type=str, default='smrystats',
        help='Output file name without suffix (Default:smrystats)')

def main_with_args(args):
    print(title_string)
    if len(args)<1:
        parser.print_usage()
        print(description_string)
        return
    
    parameters = parser.parse_args(args)
    p_dict= vars(parameters)
    
    dirname = os.path.dirname(p_dict['out'])
    if not (dirname=='' or os.path.exists(dirname)): os.makedirs(dirname)
    outLog = p_dict['out']+'.log'
    sys.stdout = logger.logger(outLog)

    # if p_dict['debug']:
        # print ('Parsed parameters:')
        # print(p_dict)
    
    if p_dict['ifmt'] == 'basic':
        #hg19chrc    snpid    a1    a2    bp    or    p
        p_dict['chr'] = 'hg19chrc'
        p_dict['snp'] = 'snpid'
        p_dict['a1'] = 'a1'
        p_dict['a2'] = 'a2'
        p_dict['bp'] = 'bp'
        p_dict['or'] = 'or'
        p_dict['p'] = 'p'
    elif p_dict['ifmt'] == 'standard':
        #chr     pos     ref     alt     reffrq  info    rs       pval    effalt
        p_dict['chr'] = 'chr'
        p_dict['snp'] = 'rs'
        p_dict['a1'] = 'ref'
        p_dict['a2'] = 'alt'
        p_dict['bp'] = 'pos'
        p_dict['beta'] = 'effalt'
        p_dict['p'] = 'pval'
        p_dict['freq'] = 'reffreq'
        p_dict['info'] = 'info'
    elif p_dict['ifmt'] == 'PGC':
        #hg19chrc        snpid   a1      a2      bp      info    or      se      p       ngt
        p_dict['chr'] = 'hg19chrc'
        p_dict['snp'] = 'snpid'
        p_dict['a1'] = 'a1'
        p_dict['a2'] = 'a2'
        p_dict['bp'] = 'bp'
        p_dict['or'] = 'or'
        p_dict['p'] = 'p'
        p_dict['se'] = 'se'
        p_dict['info'] = 'info'
    elif p_dict['ifmt'] == 'PGC2':
        # CHR    SNP    BP    A1    A2    FRQ_A_30232    FRQ_U_40578    INFO    OR    SE    P    ngt    Direction    HetISqt    HetChiSq    HetDf    HetPVa
        p_dict['chr'] = 'CHR'
        p_dict['snp'] = 'SNP'
        p_dict['a1'] = 'A1'
        p_dict['a2'] = 'A2'
        p_dict['bp'] = 'BP'
        p_dict['or'] = 'OR'
        p_dict['p'] = 'P'
        p_dict['se'] = 'SE'
        p_dict['info'] = 'INFO'
        p_dict['freqA'] = 'FRQ_A'
        p_dict['freqU'] = 'FRQ_U'
    elif p_dict['ifmt'] == 'GIANT':
        #MarkerName Allele1 Allele2 Freq.Allele1.HapMapCEU b SE p
        p_dict['snp'] = 'MarkerName'
        p_dict['a1'] = 'Allele1'
        p_dict['a2'] = 'Allele2'
        p_dict['beta'] = 'b'
        p_dict['p'] = 'p'
        p_dict['se'] = 'SE'
        p_dict['freq'] = 'Freq.Allele1.HapMapCEU'
        p_dict['n'] = 'N' 
    elif p_dict['ifmt'] == 'GIANT2':
        #SNP A1 A2 b se Freq1.Hapmap p
        p_dict['snp'] = 'SNP'
        p_dict['a1'] = 'A1'
        p_dict['a2'] = 'A2'
        p_dict['beta'] = 'b'
        p_dict['p'] = 'p'
        p_dict['se'] = 'se'
        p_dict['freq'] = 'Freq1.Hapmap'
        p_dict['n'] = 'N' 
    elif p_dict['ifmt'] == 'GWEB':
        #CHR BP SNP A1 A2 BETA SE
        p_dict['chr'] = 'CHR'
        p_dict['bp'] = 'BP'
        p_dict['snp'] = 'SNP'
        p_dict['a1'] = 'A1'
        p_dict['a2'] = 'A2'
        p_dict['beta'] = 'BETA'
        p_dict['se'] = 'SE'
    elif p_dict['ifmt'] == 'CARDIoGRAM':
        #SNP chr_pos_(b36) reference_allele other_allele ref_allele_frequency pvalue het_pvalue log_odds log_odds_se N_case N_control model
        p_dict['snp'] = 'SNP'
        p_dict['chrbp'] = 'chr_pos_(b36)'
        p_dict['a1'] = 'reference_allele'
        p_dict['a2'] = 'other_allele'
        p_dict['beta'] = 'log_odds'
        p_dict['p'] = 'pvalue'
        p_dict['se'] = 'log_odds_se'
        p_dict['freq'] = 'ref_allele_frequency'
        p_dict['phet'] = 'het_pvalue'
        p_dict['n0'] = 'N_control'
        p_dict['n1'] = 'N_case'
    elif p_dict['ifmt'] == 'IIBDGC':
        #CHR SNP BP A1 A2 OR SE FRQ_A FRQ_U INFO P
        p_dict['snp'] = 'SNP'
        p_dict['chr'] = 'CHR'
        p_dict['bp'] = 'BP'
        p_dict['a1'] = 'A1'
        p_dict['a2'] = 'A2'
        p_dict['or'] = 'OR'
        p_dict['p'] = 'P'
        p_dict['se'] = 'SE'
        p_dict['freqA'] = 'FRQ_A'
        p_dict['freqU'] = 'FRQ_U'
        p_dict['info'] = 'INFO'
        p_dict['phet'] = 'HetPva'
    elif p_dict['ifmt'] == 'GLGC':
        #SNP_hg18 SNP_hg19 rsid A1 A2 beta se N P-value Freq.A1.1000G.EUR
        p_dict['snp'] = 'rsid'
        p_dict['chrbp'] = 'SNP_hg19'
        p_dict['a1'] = 'A1'
        p_dict['a2'] = 'A2'
        p_dict['beta'] = 'beta'
        p_dict['p'] = 'P-value'
        p_dict['se'] = 'se'
        p_dict['freq'] = 'Freq.A1.1000G.EUR'
        p_dict['n'] = 'N' 
    elif p_dict['ifmt'] == 'DIAGRAM':
        #SNP CHROMOSOME POSITION RISK_ALLELE OTHER_ALLELE P_VALUE OR OR_95L OR_95U N_CASES N_CONTROLS
        p_dict['snp'] = 'SNP'
        p_dict['chr'] = 'CHROMOSOME'
        p_dict['bp'] = 'POSITION'
        p_dict['a1'] = 'RISK_ALLELE'
        p_dict['a2'] = 'OTHER_ALLELE'
        p_dict['or'] = 'OR'
        p_dict['p'] = 'P_VALUE'
        p_dict['n0'] = 'N_CONTROLS'
        p_dict['n1'] = 'N_CASES'
        # p_dict['or_l95'] = 'OR_95L'
        # p_dict['or_u95'] = 'OR_95U'
    else:
        print('The input format can not be supported!')
        print('We will try to parse the summary statistics based on BASIC format!')
        p_dict['chr'] = 'hg19chrc'
        p_dict['snp'] = 'snpid'
        p_dict['a1'] = 'a1'
        p_dict['a2'] = 'a2'
        p_dict['bp'] = 'bp'
        p_dict['or'] = 'or'
        p_dict['p'] = 'p'

    try:
        p_dict['n']=int(p_dict['n'])
    except:
        pass

    try:
        p_dict['n0']=int(p_dict['n0'])
    except:
        pass

    try:
        p_dict['n1']=int(p_dict['n1'])
    except:
        pass
    print('Start conversion...')
    
    outputList = []
    outputFile = []
    if p_dict['target'] is None:
        # if p_dict['Nthld'] is None: p_dict['Nthld'] = 0.67
        outDel = p_dict['out']+'.del'
        smryObj = smryParser(p_dict['ssf'], SNP=p_dict['snp'], CHR=p_dict['chr'], BP=p_dict['bp'], A1=p_dict['a1'], A2=p_dict['a2'], BETA=p_dict['beta'], OR=p_dict['or'], SE=p_dict['se'], CI=p_dict['ci'], ORCI=p_dict['or_ci'], L95=p_dict['l95'],U95=p_dict['u95'], ORL95=p_dict['or_l95'], ORU95=p_dict['or_u95'], P=p_dict['p'], Z=p_dict['z'], N=p_dict['n'], N0=p_dict['n0'], N1=p_dict['n1'], eff=p_dict['eff'], skip=p_dict['skip'], INFO=p_dict['info'], FREQ=p_dict['freq'], FA=p_dict['freqA'], FU=p_dict['freqU'], CHRBP = p_dict['chrbp'], A1A2 = p_dict['a1a2'], CHRBPA1A2 = p_dict['chrbpa1a2'], P_HET=p_dict['phet'], infoFilter=p_dict['infothld'], mafFilter=p_dict['maf'], chrList = p_dict['chrlist'], hetFilter=p_dict['hetthld'], z2Filter=p_dict['z2thld'], NFilter=p_dict['Nthld'], outDel=outDel)
        '''
        if (p_dict['ref'] is not None):
            from PRSparser import refParser
            import PRSalign
            import ldsc
            refObj, _ = refParser(p_dict['ref'], thread=p_dict['thread'])
            refSnpInfo = pd.concat(refObj['SNPINFO'], ignore_index=True)
            alignResult = PRSalign.unvAligner(ssList=[smryObj, refSnpInfo])
            z = alignResult['SS'][0]['BETA']/alignResult['SS'][0]['SE']
            if ('N' in smryObj.columns):
                N = int(np.quantile(alignResult['SS'][0]['N'], 0.9))
                ldscVal = alignResult['SS'][1]['LDSC']
                ldscResult = ldsc.calH2_2stage(z, ldscVal, N)
                print('Estimated h2 from LDSC:', "{:.4g}".format(ldscResult['h2']))
                inf_ldsc = 1.+N*ldscResult['a']
                print('Intercept from LDSC:', "{:.4g}".format(inf_ldsc))
            else:
                newLDSCresult = np.linalg.lstsq(np.vstack((np.ones(len(z)),ldscVal)).T, z**2, rcond=None)[0]
                inf_ldsc = newLDSCresult[0]
                print('Estimated confounding inflation factor:', "{:.4g}".format(inf_ldsc))
        '''
        outputList.append(smryObj)
        outputFile.append(p_dict['out']+'.txt')
    else:
        outDel1 = p_dict['out']+'.joint.del'
        print('====Loading summary stats from joint analysis====')
        jointSmryObj = smryParser(p_dict['ssf'], SNP=p_dict['snp'], CHR=p_dict['chr'], BP=p_dict['bp'], A1=p_dict['a1'], A2=p_dict['a2'], BETA=p_dict['beta'], OR=p_dict['or'], SE=p_dict['se'], CI=p_dict['ci'], ORCI=p_dict['or_ci'], L95=p_dict['l95'],U95=p_dict['u95'], ORL95=p_dict['or_l95'], ORU95=p_dict['or_u95'], P=p_dict['p'], Z=p_dict['z'], N=p_dict['n'], N0=p_dict['n0'], N1=p_dict['n1'], eff=p_dict['eff'], skip=p_dict['skip'], INFO=p_dict['info'], FREQ=p_dict['freq'], FA=p_dict['freqA'], FU=p_dict['freqU'], CHRBP = p_dict['chrbp'], A1A2 = p_dict['a1a2'], CHRBPA1A2 = p_dict['chrbpa1a2'] , P_HET=p_dict['phet'], infoFilter=p_dict['infothld'], mafFilter=p_dict['maf'], chrList = p_dict['chrlist'], hetFilter=p_dict['hetthld'], z2Filter=p_dict['z2thld'], NFilter=p_dict['Nthld'], outDel=outDel1)
        '''
        if (p_dict['ref'] is not None):
            from PRSparser import refParser
            import strUtils
            import ldsc
            refObj, _ = refParser(p_dict['ref'], thread=p_dict['thread'])
            refSnpInfo = pd.concat(refObj['SNPINFO'], ignore_index=True)
            snpDict = pd.Series(range(len(refSnpInfo)),index=refSnpInfo['SNP'])
            idx, pos = strUtils.listInDict(jointSmryObj['SNP'].to_list(), snpDict.to_dict())
            alignResult = [jointSmryObj.iloc[idx,:].reset_index(drop=True).copy(), refSnpInfo.iloc[pos,:].reset_index(drop=True).copy()]
            z = alignResult[0]['BETA']/alignResult[0]['SE']
            if ('N' in jointSmryObj.columns):
                N = int(np.quantile(alignResult[0]['N'], 0.9))
                ldscVal = alignResult[1]['LDSC']
                ldscResult = ldsc.calH2_2stage(z, ldscVal, N)
                print('Estimated h2 from LDSC:', "{:.4g}".format(ldscResult['h2']))
                inf_ldsc = 1.+N*ldscResult['a']
                print('Intercept from LDSC:', "{:.4g}".format(inf_ldsc))
            else:
                newLDSCresult = np.linalg.lstsq(np.vstack((np.ones(len(z)),ldscVal)).T, z**2, rcond=None)[0]
                inf_ldsc = newLDSCresult[0]
                print('Estimated confounding inflation factor:', "{:.4g}".format(inf_ldsc))
            if (p_dict['Nthld'] is None) and ('N' in jointSmryObj.columns):
                smallSampleSS = jointSmryObj[jointSmryObj['N']<(0.67*N)]
                smallSampleZ2 = (smallSampleSS['BETA']/smallSampleSS['SE'])**2
                if np.median(smallSampleZ2)/(0.455*inf_ldsc)<1.01:
                    print('No inflation detected in SNPs with sample size less than 0.67 of 90% quantile')
                    print('Resetting --Nthld to 0.67')
                    p_dict['Nthld'] = 0.67
                    print(len(smallSampleSS),'SNP(s) with sample size less than 0.67 of 90% sample size quantile')
                    outDel1_2 = p_dict['out']+'.joint2.del'
                    try:
                        smallSampleSS.to_csv(outDel1_2, sep='\t', index=False)
                    except:
                        print('Can not write deleted SNPs into file:', outDel1_2)
                    jointSmryObj = jointSmryObj[jointSmryObj['N']>=(0.67*N)]
        '''        
        print('====Loading summary stats from target array====')
        targetSmryObj = smryParser(p_dict['target'], SNP=p_dict['snp'], CHR=p_dict['chr'], BP=p_dict['bp'], A1=p_dict['a1'], A2=p_dict['a2'], BETA=p_dict['beta'], OR=p_dict['or'], SE=p_dict['se'], CI=p_dict['ci'], ORCI=p_dict['or_ci'], L95=p_dict['l95'],U95=p_dict['u95'], ORL95=p_dict['or_l95'], ORU95=p_dict['or_u95'], P=p_dict['p'], Z=p_dict['z'], N=p_dict['n'], N0=p_dict['n0'], N1=p_dict['n1'], eff=p_dict['eff'], skip=p_dict['skip'], INFO=p_dict['info'], FREQ=p_dict['freq'], FA=p_dict['freqA'], FU=p_dict['freqU'], CHRBP = p_dict['chrbp'], A1A2 = p_dict['a1a2'], CHRBPA1A2 = p_dict['chrbpa1a2'], P_HET=p_dict['phet'], infoFilter=0, mafFilter=0, chrList = p_dict['chrlist'], hetFilter=0, z2Filter=np.inf, NFilter=None,outDel=None,noFilter=True)
        outDel2 = p_dict['out']+'.gwas.del'
        print('====Recovering summary stats from stage I GWAS====')
        GWASsmryObj = recoverGWAS(jointSmryObj, targetSmryObj, z2Filter=p_dict['z2thld'], NFilter=p_dict['Nthld'], maxN=p_dict['nmax'], outDel=outDel2)
        outputList.append(jointSmryObj)
        outputList.append(GWASsmryObj)
        outputList.append(targetSmryObj)
        outputFile.append(p_dict['out']+'.txt')
        outputFile.append(p_dict['out']+'.gwas.txt')
        outputFile.append(p_dict['out']+'.target.txt')
    
    for i in range(len(outputList)):
        if p_dict['ofmt'] == 'PRSCS':
            #SNP A1   A2   BETA      P
            outputList[i]['A1'] = outputList[i]['A1'].str.upper() 
            outputList[i]['A2'] = outputList[i]['A2'].str.upper()
            outputList[i]['P'] = 2*norm.cdf(-np.abs(outputList[i]['BETA']/outputList[i]['SE']))
            outputList[i] = outputList[i].drop(columns=['CHR','BP', 'SE'], errors='ignore')
            outputList[i] = outputList[i][['SNP', 'A1', 'A2', 'BETA', 'P']]
        elif p_dict['ofmt'] == 'EBPRS':
            #A1, A2, OR, P, SNP
            outputList[i]['A1'] = outputList[i]['A1'].str.upper() 
            outputList[i]['A2'] = outputList[i]['A2'].str.upper()
            outputList[i]['P'] = 2*norm.cdf(-np.abs(outputList[i]['BETA']/outputList[i]['SE']))
            outputList[i]['OR'] = np.exp(outputList[i]['BETA'])
            outputList[i] = outputList[i].drop(columns=['CHR','BP', 'BETA' ,'SE'], errors='ignore')
            outputList[i] = outputList[i][['SNP', 'A1', 'A2', 'OR', 'P']]
        elif p_dict['ofmt'] == 'GWEB':
            pass
        elif p_dict['ofmt'] == 'basic':
            #hg19chrc    snpid    a1    a2    bp    or    p
            outputList[i]['A1'] = outputList[i]['A1'].str.upper() 
            outputList[i]['A2'] = outputList[i]['A2'].str.upper()
            outputList[i]['P'] = 2*norm.cdf(-np.abs(outputList[i]['BETA']/outputList[i]['SE']))
            outputList[i]['OR'] = np.exp(outputList[i]['BETA'])
            outputList[i] = outputList[i].drop(columns=['BETA' ,'SE'], errors='ignore')
            outputList[i] = outputList[i][['CHR', 'SNP','A1', 'A2', 'BP', 'OR', 'P']]
            outputList[i] = outputList[i].rename(columns={'SNP': 'snpid','CHR':'hg19chrc','BP':'bp','A1':'a1','A2':'a2','OR':'or','P':'p'})
        else:
            print('The output format is not supported!')
            print('GWEB format is used!')
        try:
            outputList[i].to_csv(outputFile[i], sep='\t', index=False)
        except:
            print('Can\'t write parsed summary stats into file:', outputFile[i])
    print('Finished!')
    
def main():
    main_with_args(sys.argv[1:])

if __name__ == '__main__':
    main_with_args(sys.argv[1:])
