from __init__ import *
import argparse
import sys, os, time, logger
import GWEB
import GWEB_pruning

window_length = 74
title = '\033[96mGWEB v%s\033[0m' % version
title_len = len(title) - 9
num_dashes = window_length - title_len - 2
left_dashes = '=' * int(num_dashes / 2)
right_dashes = '=' * int(num_dashes / 2 + num_dashes % 2)
title_string = '\n' + left_dashes + ' ' + title + ' ' + right_dashes + '\n'

description_string = """
\033[1mGWEB\033[0m: An Empirical-Bayes-based polygenic risk prediction approach
using GWAS summary statistics

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
""" % (author, contact)
description_string += '=' * window_length


parser = argparse.ArgumentParser(prog='GWEB', formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('--ssf', type=str, required=True,
                    help='GWAS Summary statistic file. '
                         'Should be a text file with columns SNP/CHR/BP/BETA/SE')

parser.add_argument('--weight', type=str, required=True,
                    help='PRS model weight file. '
                         'Should be a tabular text file with the five columns being SNP/CHR/BP/A1/A2/ ' 
                         'and the following columns correspond to weights with different PRS model parameters')

parser.add_argument('--n0', type=int, default=0, required=True,
                    help='Control sample size of the GWAS summary statistics.')

parser.add_argument('--n1', type=int, default=0, required=True,
                    help='Case sample size of the GWAS summary statistics.')

parser.add_argument('--pruning', default=False, required=not ('--ref' in sys.argv), action='store_true',
                    help='If pruning, EM algorithm is used on SNPs with non-zero PRS weights. No reference LD file is provided')

parser.add_argument('--ref', type=str, required=not ('--pruning' in sys.argv),
                    help='Reference LD file. Provided if not pruning. '
                         'Should be a (full path) hdf5 file storing the LD matrix and corresponding SNP information. (plinkLD.py can be used to convert PLINK binary files into the LD hdf5 file.)')

parser.add_argument('--homo', default=False, action='store_true',
                    help='If the summary statistics are from a single homogeneous GWAS cohort, use homo to not to shrink LD')

parser.add_argument('--geno', type=str, default=None, required=False,
                    help='Individual-level genotype data for testing purpose (if provided) '
                        'Should be PLINK binary format files with extension .bed/.bim/.fam')

parser.add_argument('--pheno', type=str, default=None, required=('--geno' in sys.argv),
                    help="External phenotype file. Should be provided if geno. "
                        "Should be a tabular text file. If header is not provided, the first, second columns and third columns should be FID, IID, and PHE. Otherwise, there are three columns named 'FID', 'IID' and 'PHE'")

parser.add_argument('--aligned', default=False, action='store_true',
                    help='Whether the ssf, weight, (geno, ref) files are aligned. Do not use aligned for real data')

parser.add_argument('--dir', type=str, default='./output', required=False,
                    help='Output directory')

parser.add_argument('--thread', type=int, default=-1,
                    help='Number of parallel threads, by default all CPUs will be utilized.')


def main_with_args(args):
    print(title_string)
    if len(args) < 1:
        parser.print_usage()
        print(description_string)
        return

    startTime0 = time.time()
    parameters = parser.parse_args(args)
    p_dict = vars(parameters)
    p_dict['n'] = p_dict['n0'] + p_dict['n1']

    if not os.path.exists(p_dict['dir']):
        os.makedirs(p_dict['dir'])

    sys.stdout = logger.logger(os.path.join(p_dict['dir'], 'log.txt'))

    if p_dict['pruning']:
        # align using GWEB_pruning: no need to align ref file
        p_dict['h5geno'] = None
        p_dict['iprefix'] = None
        p_dict['aligned'] = False
        p_dict['align-only'] = True
        GWEB_pruning.main(p_dict)
        print('Alignment time elapsed:', time.time() - startTime0, 's')

        # PRStuning: EM
        p_dict['iprefix'] = os.path.join(p_dict['dir'], 'align_')
        p_dict['align-only'] = False
        p_dict['aligned'] = True
        GWEB_pruning.main(p_dict)

    else:
        # align using GWEB
        p_dict['iprefix'] = None
        p_dict['h5geno'] = None
        p_dict['aligned'] = False
        p_dict['align-only'] = True
        GWEB.main(p_dict)
        print('Alignment time elapsed:', time.time() - startTime0, 's')

        # PRStuning: MCMC
        p_dict['iprefix'] = os.path.join(p_dict['dir'], 'align_')
        p_dict['align-only'] = False
        p_dict['aligned'] = True
        GWEB.main(p_dict)

    print('Completed! Total time elapsed:', time.time() - startTime0, 's')
    print('Thank you for using PRStuning!')

def main():
    main_with_args(sys.argv[1:])

if __name__ == '__main__':
    main_with_args(sys.argv[1:])
