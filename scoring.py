#!/usr/bin/env python3

from __init__ import *
import argparse
import sys, os, time, functools
import textwrap
import logger
import PRSparser
import PRSalign
import PRScoring
import PRSeval
import numpy as np
import pandas as pd

window_length=74
title = '\033[96mscoring\033[0m'
title_len= len(title)-9
num_dashes = window_length-title_len-2
left_dashes = '='*int(num_dashes/2)
right_dashes = '='*int(num_dashes/2+num_dashes%2)
title_string = '\n'+left_dashes  + ' '+title+' '+right_dashes +'\n'

description_string = """
    A general program to calculate polygenic risk scores based on a given weight
    file and evaluate the corresponding performance.
    See\033[1m scoring.py --help\033[0m for further usage description and options.

(c) %s
%s

Thank you for using\033[1m scoring\033[0m!
"""%(author,contact)
description_string += '='*window_length

parser = argparse.ArgumentParser(prog='scoring',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=textwrap.dedent(description_string))

#General arguments
# parser.add_argument('--debug', default=False, action='store_true',
                    # help="Activate debugging mode (more verbose)")

parser.add_argument('--weight', type=str, required=not('--prs' in sys.argv), default=None, 
                    help='A text file storing weight for each SNP')

parser.add_argument('--prs', type=str, required=not('--weight' in sys.argv), default=None,
                    help='A text file storing calculated PRS for each individual')

parser.add_argument('--bfile', type=str, default=None, required=not((('--bed' in sys.argv) and ('--bim' in sys.argv) and ('--fam' in sys.argv)) or ('--h5geno' in sys.argv) or ('--iprefix' in sys.argv) or ('--prs' in sys.argv)),
                    help='Individual-level genotype data for testing purpose.'
                        'Should be PLINK binary format files with extension .bed/.bim/.fam')

parser.add_argument('--bed', type=str, default=None, required=False,
                    help='Binary genotype data with suffix .bed (Required if bfile and h5geno are not provided)')

parser.add_argument('--bim', type=str, default=None, required=False,
                    help='SNP infomation file with suffix .bim (Required if bfile and h5geno are not provided)')

parser.add_argument('--fam', type=str, default=None, required=False,
                    help='Individual information file with suffix .fam (Required if bfile and h5geno are not provided)')

parser.add_argument('--h5geno', type=str, default=None, required=False,
                    help='Individual-level genotype data with hdf5 format')

parser.add_argument('--aligned', default=False, action='store_true',
                    help='The input has already been aligned')

parser.add_argument('--pheno', type=str, default=None, required=('--prs' in sys.argv),
                    help="External phenotype file."
                        "Should be a tabular text file. If header is not provided, the first and second columns should be FID and IID. Otherwise, there are two columns named 'FID' and 'IID'")

parser.add_argument('--mpheno', type=int, default=1, required=False,
                    help='m-th phenotype in the phenotype file to be used (default: 1)')

parser.add_argument('--pheno-name', type=str, default='PHE', required=False,
                    help='Column name for the phenotype in the phenotype file (default:PHE)')

parser.add_argument('--cov', type=str, default=None,
                    help='covariates file, format is tabulated file with columns FID, IID, PC1, PC2, etc.')

parser.add_argument('--out', type=str, default='./result/',
                    help='Prefix for output files (default: ./result/)')

parser.add_argument('--prs-name', type=str, default=None, required=False,
                    help='Column name for the prs in the prs file (default:Every column behind FID/IID)')

parser.add_argument('--thread', type=int, default=-1,
                    help='Number of parallel threads, by default all CPUs will be utilized.')

parser.add_argument('--rfreq', default=False, action='store_true',
                    help='Use allele frequencies inferred from reference panel to normalize the genotype')

parser.add_argument('--norm', default=False, action='store_true',
                    help='Use normalized genotypes to calculate PRS')

parser.add_argument('--split-by-chr', default=False, action='store_true',
                    help='Split analysis by chromosome')

parser.add_argument('--snp', type=str, default='SNP', required=False,
                    help='Column name of SNP ID in the header')

parser.add_argument('--chr', type=str, default='CHR', required=False,
                    help='Column name of Chromosome in the header')

parser.add_argument('--bp', type=str, default='BP', required=False,
                    help='Column name of base pair position in the header')

parser.add_argument('--a1', type=str, default='A1', required=False,
                    help='Column name of reference allele in the header')

parser.add_argument('--a2', type=str, default='A2', required=False,
                    help='Column name of alternative allele in the header')

parser.add_argument('--eff', type=str, default='BETAJ', required=False,
                    help='Column name of effect size in the header')

parser.add_argument('--pos', default=False, action='store_true',
                    help='All files will be aligned by SNP position (chromosome and base pair position), instead of SNP identifier.')


parser.add_argument('--compress', type=int, default=9, required=False,
                    help='Compression level for output (default: 9)')

parser.add_argument('--bigmem', default=False, action='store_true',
                    help='Consuming more memory in the alignment phase but with faster speed')

parser.add_argument('--nosave', default=False, action='store_true',
                    help='Not saving the aligned files for reducing disk space usage')

parser.add_argument('--wref', type=str, default=None, required=False,
                    help='Reference LD File for weights. '
                         'Should be a (full path) hdf5 file storing the LD matrix and corresponding SNP information. (plinkLD.py can be used to convert PLINK binary files into the LD hdf5 file.)')

parser.add_argument('--iprefix', type=str, default=None, required=False,
                    help='Common prefix for input files (summary statistics, reference panel, genotypes, annotations)')

parser.add_argument('--mem', type=float, default=0, required=False,
                    help='Available memory in Gigabytes (Using maximum available memory by default)')

parser.add_argument('--np', default=False, action='store_true', help='Using numpy instead of PLINK to calculate PRS')

# parser.add_argument('--debug', default=False, action='store_true',
                    # help='Debug mode')

def main_with_args(args):
    print(title_string)
    if len(args)<1:
        parser.print_usage()
        print(description_string)
        return
        
    startTime0 = time.time()
    parameters = parser.parse_args(args)
    p_dict= vars(parameters)
    # if p_dict['debug']:
        # print ('Parsed parameters:')
        # print(p_dict)
    
    sys.stdout = logger.logger(p_dict['out']+'log.txt')
    if p_dict['prs'] is not None:
        if os.path.splitext(p_dict['prs'])[1]=='.sscore':
            #Result from PLINK2
            scoreDF = PRSeval.indParser(p_dict['prs'], dataCol=p_dict['prs_name'], FID='#FID', defaultCol='SCORE1_AVG')
        else:
            scoreDF = PRSeval.indParser(p_dict['prs'], dataCol=p_dict['prs_name'])

    else:
        if p_dict['iprefix'] is not None:
            if os.path.exists(p_dict['iprefix']+'ref.h5'): 
                p_dict['wref'] = p_dict['iprefix']+'ref.h5'
            if os.path.exists(p_dict['iprefix']+'geno.h5'): 
                p_dict['h5geno'] = p_dict['iprefix']+'geno.h5'
    
        print('Parsing weight file from', p_dict['weight'])
        weightObj = PRSparser.weightParser(p_dict['weight'], SNP=p_dict['snp'], CHR=p_dict['chr'], BP=p_dict['bp'], A1=p_dict['a1'], A2=p_dict['a2'], BETAJ=p_dict['eff'])
        if weightObj is None:
            print('Can not read weight file:',p_dict['weight'])
            return
        if p_dict['wref'] is not None:
            print('Reading Reference Panel from',p_dict['wref'])
            if (p_dict['aligned'] or p_dict['bigmem']):
                wrefObj = PRSparser.refParser(p_dict['wref'], thread=p_dict['thread'])
            else:
                wrefStore = pd.HDFStore(p_dict['wref'], 'r', complevel=p_dict['compress'])
                wrefStore = PRSparser.adjustRefStore(p_dict['wref'], wrefStore, wrefStore, thread=p_dict['thread'], complevel=p_dict['compress'])
    
        print('Parsing testing genotype data ...')
        if p_dict['h5geno'] is not None:
            genoStore = pd.HDFStore(p_dict['h5geno'])
            genoObj = {'SNPINFO': genoStore.get('SNPINFO'),'INDINFO': genoStore.get('INDINFO'), 'GENOTYPE': genoStore.get('GENOTYPE')[0]}
            genoStore.close()
            genoObj['FLIPINFO'] = np.array([False]*len(genoObj['SNPINFO']))
            print(len(genoObj['SNPINFO']),'SNPs')
            print(len(genoObj['INDINFO']),'Individuals')
            # del genoStore
            # gc.collect()
        else:
            genoObj = PRSparser.genoParser(bfile=p_dict['bfile'], bed=p_dict['bed'], bim=p_dict['bim'], fam=p_dict['fam'])
    
        if p_dict['aligned']:
            newWeight = weightObj['BETA']
            alignResult = {'GENO': genoObj }
        elif p_dict['wref'] is not None:
            print('Aligning reference file and genotype data ...')
            alignRefFile = p_dict['out']+'align_ref.h5'
            if p_dict['bigmem']:
                if not p_dict['nosave']:
                    alignResult = PRSalign.alignGeno2Ref(genoObj, wrefObj, outGeno = p_dict['out']+'align_geno.h5', outRef=alignRefFile, byID=(not p_dict['pos']), complevel=p_dict['compress'])
                else:
                    alignResult = PRSalign.alignGeno2Ref(genoObj, wrefObj, outGeno = None, outRef=None, byID=(not p_dict['pos']), complevel=p_dict['compress'])
            else:
                outRefStore = pd.HDFStore(alignRefFile, 'w', complevel=p_dict['compress'])
                alignResult = PRSalign.alignGeno2RefStore(genoObj, wrefStore, outRefStore, outGeno = p_dict['out']+'align_geno.h5', byID=(not p_dict['pos']), complevel=p_dict['compress'])
                outRefStore.close()
            print('After alignment,', len(alignResult['GENO']['SNPINFO']),'SNPs remaining')
            if not p_dict['bigmem']:
                outRefStore = pd.HDFStore(alignRefFile, 'r')
                newWeight = PRScoring.adjustWeightStore(alignResult['GENO'], weightObj['BETA'], outRefStore, wrefStore, byID=(not p_dict['pos']))
                outRefStore.close()
                wrefStore.close()
                if p_dict['nosave']:
                    os.remove(alignRefFile)
            else:
                newWeight = PRScoring.adjustWeight(alignResult['GENO'], weightObj['BETA'], alignResult['REF'], wrefObj, byID=(not p_dict['pos']))
        else:
            if not p_dict['nosave']:
                alignResult = PRSalign.alignSmry2Geno(weightObj, genoObj, outSmry = p_dict['out']+'align_weight.txt', outGeno=p_dict['out']+'align_geno.h5', byID=(not p_dict['pos']), complevel=p_dict['compress'])
            else:
                alignResult = PRSalign.alignSmry2Geno(weightObj, genoObj, outSmry = None, outGeno=None, byID=(not p_dict['pos']), complevel=p_dict['compress'])
            print('After alignment,', len(alignResult['SS']),'SNPs remaining')
            newWeight = alignResult['SS']['BETA']

        if p_dict['rfreq']:
            af = extractRef(alignResult['REF'], col='F')
        else:
            af = None
        print('Getting PRS for genotype data...')
    
        if (af is None) and (p_dict['np']):
            af = PRScoring.getAlleleFreq(alignResult['GENO'], mem=p_dict['mem'])
    
        if not p_dict['np']:
            score = PRScoring.scoringByPlink(alignResult['GENO'], newWeight, splitByChr=p_dict['split_by_chr'], out=p_dict['out'], thread=p_dict['thread'])
        else:
            score = PRScoring.scoring(alignResult['GENO'], newWeight, f=af, splitByChr=p_dict['split_by_chr'], normalize=p_dict['norm'], mem=p_dict['mem']) 
        scoreDF = score.copy()
        scoreDF.insert(loc=0, column='FID', value=alignResult['GENO']['INDINFO']['FID'])
        scoreDF.insert(loc=1, column='IID', value=alignResult['GENO']['INDINFO']['IID'])
        try:
            scoreDF.to_csv(p_dict['out']+'PRS.txt', sep='\t', index=False)
        except:
            print('Can not write risk scores into file:', p_dict['out']+'PRS.txt')
    
    print('Extracting phenotype...')
    if p_dict['prs'] is not None:
        phenoDF = PRSeval.phenoParser(phenoFile=p_dict['pheno'], PHE=p_dict['pheno_name'], mpheno=p_dict['mpheno'])
    else:
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
        if covData is not None:
            pred01 = PRSeval.fit(phe, covData=covData)
            PRSeval.strataPlot(phe, pred01['yhat'], figname=p_dict['out']+'strata_cov.pdf')
            if isBinPhe:
                aucObj01 = PRSeval.auc(phe, pred01['yhat'], figname=p_dict['out']+'roc_cov.pdf') 
                print('AUC based on covariates only: %.3f'%aucObj01['auc']+'(%.3f'%aucObj01['se']+')')
        
        for label, prs in score.items(): 
            r2obj = PRSeval.rsq(phe, prs, covData=covData)
            r2Dict[label] = r2obj['r2'].map('{:,.4f}'.format)+'('+r2obj['se'].map('{:,.4f}'.format)+')'
            pred10 = PRSeval.fit(phe, prs=prs)

            PRSeval.strataPlot(phe, pred10['yhat'], figname=p_dict['out']+'strata_'+label+'_prs.pdf')
            
            if covData is not None:
                pred11 = PRSeval.fit(phe, prs=prs, covData=covData)
                PRSeval.strataPlot(phe, pred11['yhat'], figname=p_dict['out']+'strata_'+label+'_prs_cov.pdf')
            if isBinPhe:
                aucObj10 = PRSeval.auc(phe, pred10['yhat'], figname=p_dict['out']+'roc_'+label+'_prs.pdf') 
                aucDict[label] = ['%.3f'%(aucObj10['auc'])+'('+'%.3f'%(aucObj10['se'])+')']
                if covData is not None:
                    aucObj11 = PRSeval.auc(phe, pred11['yhat'], figname=p_dict['out']+'roc_'+label+'_prs_cov.pdf') 
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
        if isBinPhe:
            print('AUC:')
            if covData is not None:
                aucDf = pd.DataFrame.from_dict(aucDict, orient='index', columns=['prs','all','inc','inc%'])
            else:
                aucDf = pd.DataFrame.from_dict(aucDict, orient='index', columns=['prs'])
            print(aucDf)
    print('Completed! Time elapsed:',time.time()-startTime0,'s')
    print('Thank you for using scoring.py!')

def main():
    main_with_args(sys.argv[1:])

if __name__ == '__main__':
    main_with_args(sys.argv[1:])
