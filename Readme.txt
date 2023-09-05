Python Version: >=3.8
Dependency:
	GSL;
	PLINK;
	numpy;
	scipy;
	pandas;
	scikit-learn;	
	rpy2;
	cython;
	tables;
	pysnptools;
	CythonGSL;

Installation:
1. Install GSL
For Linux/Mac, see
https://solarianprogrammer.com/2020/01/26/getting-started-gsl-gnu-scientific-library-windows-macos-linux/#gsl_usage_example
for detailed installation.

For Windows, you can download compiled library directly from 
https://code.google.com/p/oscats/downloads/list
Copy bin/libgsl-0.dll and bin/libgslcblas-0.dll into the working directory
Special tips for installing GSL on windows can be found on http://joonro.github.io/blog/posts/installing-gsl-and-cythongsl-in-windows/

2. Install python packages via conda or pip:
	conda install python">=3.8" scipy numpy pandas scikit-learn rpy2 cython sklearn pip
or 	pip install scipy numpy pandas scikit-learn rpy2 cython sklearn

	pip install tables
	pip install pysnptools
        git clone git@github.com:twiecki/CythonGSL.git
	cd ./CythonGSL
	python setup.py build
	python setup.py install
	cd ../
	rm -rf ./CythonGSL
	python setup.py build_ext --inplace

Typical workflow:
(*) indicates the step is optional.
0a*. Use plinkLD.py to calculate LD matrix for PLINK binary format encoded genotype data of a reference panel.
     See plinkLD.py --help for further usage description and options.

0b*. Use formatSS.py to convert GWAS summary statistics from different cohorts into the standard input format of GWEB.
     See formatSS.py --help for further usage description and options.

1. Use PRStuning.py to obtain PRStuning AUC and testing AUC (if testing genotype data provided).
   See PRStuning.py --help for further usage description and options.

4. plinkLD.py: Calculates LD matrix for 

   python plinkLD.py  --bfile ./refdata/1000G.EUR.QC --block ./refdata/fourier_ls-all.bed --thread -1
	--bfile: Binary data file (Default: test)
	--bed: Binary data file (Genotypes; Default: test.bed)
	--bim: Binary data file (SNP info; Default: test.bim)
	--fam: Binary data file (Individual info; Default: test.fam)
	--block: Block file (By default, all SNPs are in one block)
	--snplist: SNP list file (By default, all SNP pairs are calculated)
	--output: output filename (Default: LD.h5)
	--method: Correlation estimation method, including: Pearson, LW (Default:Pearson)
	--thread: Thread number for calculation (Default: Total CPU number)
	--compress: compression level for output (Default: 9)
	--log: log file (Default: plinkLD.log)
	--help: Help

5. GWEB: An empirical-Bayes-based polygenic risk prediction approach
	 using GWAS summary statistics and functional annotations
   
   Usage:
   python GWEB.py [-h] --ssf SSF --ref REF [--bfile BFILE] [--bed BED] [--bim BIM] [--fam FAM] [--snplist SNPLIST]
                  [--iprefix IPREFIX] [--n N] [--dir DIR] [--aligned]
                  [--align-only] [--thread THREAD]



  --ssf SSF             GWAS Summary statistic File. Should be a text file with columns SNP/CHR/BP/BETA/SE. Optional: N
  --ref REF             Reference LD File. Should be a (full path) hdf5 file storing the LD matrix and corresponding SNP information. (plinkLD.py can be
                        used to convert PLINK binary files into the LD hdf5 file.)
  --bfile BFILE         Individual-level genotype data for testing purpose. Should be PLINK binary format files with extension .bed/.bim/.fam
  --bed BED             Binary genotype data with suffix .bed (Used if bfile is not provided)
  --bim BIM             SNP infomation file with suffix .bim (Used if bfile is not provided)
  --fam FAM             Individual information file with suffix .fam (Used if bfile is not provided)
  --snplist SNPLIST     SNP list file used to filter SNPs
  --iprefix IPREFIX     Common prefix for input files (summary statistics, reference panel, genotypes, annotations)
  --n N                 Sample size of the GWAS summary statistics. (If provided, LDSC will be used to adjust the inflation caused by potential confounding
                        effect.)
  --dir DIR             Output directory
  --aligned             The input has already been aligned
  --align-only          Align all input files only
  --thread THREAD      Number of parallel threads, by default all CPUs will be utilized.

  6. Example Demonstration:
###T2D

Step1: Align all datasets
python GWEB.py --ssf ./data/T2D/T2D_DIAGRAM_69033.txt --bfile ./data/ukbb_impv3_eur_hm3 --ref ./refdata/1000G_EUR_phase3_hm3_ld_shrink.h5 --align-only --dir ./data/T2D/aligned_ukbb/ --thread 20

Step2: Conducting analysis
python GWEB.py --iprefix ./data/T2D/aligned_ukbb/align_ --dir ./result/T2D/ --aligned --n 69033 --thread 20
