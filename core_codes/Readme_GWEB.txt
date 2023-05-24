Python Version: >=3.0
Dependency: 
	Numpy;
	Scipy;
	Cython; 
	GSL (https://www.gnu.org/software/gsl/); 
	CythonGSL (https://github.com/twiecki/CythonGSL)
	pysnptools
	scikit-learn
	Pandas
	arspy (https://arspy.readthedocs.io/en/latest/)

1. Install cython using PIP or conda (https://cython.readthedocs.io/en/latest/src/quickstart/install.html)
2. Install GSL 
For Linux/Mac, see
https://solarianprogrammer.com/2020/01/26/getting-started-gsl-gnu-scientific-library-windows-macos-linux/#gsl_usage_example)

For Windows, you can download old version directly from 
https://code.google.com/p/oscats/downloads/list

3. Install CythonGSL
See https://github.com/twiecki/CythonGSL

Special Tips for windows: See http://joonro.github.io/blog/posts/installing-gsl-and-cythongsl-in-windows/
If the DLLs can not be found, copy the gsl.dll and gslcblas.dll from GSL/bin to current project directory

4. Compilation:
python setup.py build_ext --inplace
(cython -a GWEButils_cFunc.pyx to check the speed bottleneck) 

5. Usage:
module load plink/1.90

usage: GWEB [-h] --ssf SSF --ref REF [--bfile BFILE] [--bed BED] [--bim BIM] [--fam FAM] [--h5geno H5GENO] [--anno ANNO] [--snplist SNPLIST]
            [--iprefix IPREFIX] [--n N] [--K K [K ...]] [--pheno PHENO] [--mpheno MPHENO] [--pheno-name PHENO_NAME] [--cov COV] [--dir DIR] [--aligned]
            [--align-only] [--weight-only] [--thread THREAD]

GWEB: An Empirical-Bayes-based polygenic risk prediction approach
using GWAS summary statistics and functional annotations

Typical workflow:
    0a*. Use plinkLD.py to calculate LD matrix for PLINK binary format
    encoded genotype data of a reference panel.
    See plinkLD.py --help for further usage description and options.

    0b*. Use formatSS.py to convert GWAS summary statistics from
    different cohorts into the standard input format of GWEB.
    See formatSS.py --help for further usage description and options.

    1. Use GWEB.py to obtain SNP weights for polygenic scoring.
    See GWEB.py --help for further usage description and options.

    2*. Use scoring.py to calculate polygenic scores for an external
    individual-level genotype data using SNP weights from the previous step.
    See scoring --help for further usage description and options.

    (*) indicates the step is optional.

  --ssf SSF             GWAS Summary statistic File. Should be a text file with columns SNP/CHR/BP/BETA/SE
  --ref REF             Reference LD File. Should be a (full path) hdf5 file storing the LD matrix and corresponding SNP information. (plinkLD.py can be
                        used to convert PLINK binary files into the LD hdf5 file.)
  --bfile BFILE         Individual-level genotype data for testing purpose. Should be PLINK binary format files with extension .bed/.bim/.fam
  --bed BED             Binary genotype data with suffix .bed (Used if bfile is not provided)
  --bim BIM             SNP infomation file with suffix .bim (Used if bfile is not provided)
  --fam FAM             Individual information file with suffix .fam (Used if bfile is not provided)
  --h5geno H5GENO       Individual-level genotype data with hdf5 format
  --anno ANNO           Functional annotation file. Should be a hdf5 file storing annotations for SNPs
  --snplist SNPLIST     SNP list file used to filter SNPs
  --iprefix IPREFIX     Common prefix for input files (summary statistics, reference panel, genotypes, annotations)
  --n N                 Sample size of the GWAS summary statistics. (If provided, LDSC will be used to adjust the inflation caused by potential confounding
                        effect.)
  --K K [K ...]         Number of causal components (Default:3)
  --pheno PHENO         External phenotype file.Should be a tabular text file. If header is not provided, the first and second columns should be FID and
                        IID, respectively. Otherwise, there are two columns named 'FID' and 'IID'
  --mpheno MPHENO       m-th phenotype in the file to be used (default: 1)
  --pheno-name PHENO_NAME
                        Column name for the phenotype in the phenotype file (default:PHE)
  --cov COV             covariates file, format is tabulated file with columns FID, IID, PC1, PC2, etc.
  --dir DIR             Output directory
  --aligned             The input has already been aligned
  --align-only          Align all input files only
  --weight-only         Weighting only, without scoring and evaluation
  --thread THREAD       Number of parallel threads, by default all CPUs will be utilized.

  6. Example Demonstration:
###T2D

Step1: Align all datasets
python GWEB.py --ssf ./data/T2D/T2D_DIAGRAM_69033.txt --bfile ./data/ukbb_impv3_eur_hm3 --ref ./refdata/1000G_EUR_phase3_hm3_ld_shrink.h5 --align-only --dir ./data/T2D/aligned_ukbb/ --thread ${SLURM_CPUS_PER_TASK}

Step2: Conducting analysis
python GWEB.py --iprefix ./data/T2D/aligned_ukbb/align_ --dir ./result/T2D/ --aligned --n 69033 --thread ${SLURM_CPUS_PER_TASK} --K 1 --weight-only

Step3: Calculating PRS for individuals in testing dataset.
python scoring.py --h5geno ./data/T2D/aligned_ukbb/align_geno.h5 --weight ./result/T2D/K1_weight.txt --aligned --out ./result/T2D/K1/ --pheno ./data/ukbb_eur_traits_bin_withMed.txt --pheno-name T2D --cov ./data/ukbb_eur_covar.txt


