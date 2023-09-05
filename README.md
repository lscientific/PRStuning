# PRStuning: Estimate Testing AUC for Binary Phenotype Using GWAS Summary Statistics from the Training Data

Python Version: >=3.8 \
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
	CythonGSL

## Installation and Compilation

### Install GSL 
For Linux/Mac, see
https://solarianprogrammer.com/2020/01/26/getting-started-gsl-gnu-scientific-library-windows-macos-linux/#gsl_usage_example
for detailed installation.

For Windows, you can download the compiled library directly from https://code.google.com/p/oscats/downloads/list
Copy bin/libgsl-0.dll and bin/libgslcblas-0.dll into the working directory
Special tips for installing GSL on Windows can be found at http://joonro.github.io/blog/posts/installing-gsl-and-cythongsl-in-windows/

### Install python packages via conda or pip:
First \
```conda install python">=3.8" scipy numpy pandas scikit-learn rpy2 cython sklearn pip``` \
or 	```pip install scipy numpy pandas scikit-learn rpy2 cython sklearn```
Then \
```
pip install tables
pip install pysnptools
git clone git@github.com:twiecki/CythonGSL.git
cd ./CythonGSL
python setup.py build
python setup.py install
cd ../
rm -rf ./CythonGSL
python setup.py build_ext --inplace
```

Installation is expected to be finished within a few minutes.

## Typical Workflow:
(Optional) Use ```plinkLD.py``` to calculate the LD matrix for PLINK binary format encoded genotype data of a reference panel.
  See ```python plinkLD.py --help``` for further usage description and options.\

  Use ```PRStuning.py``` to obtain estimated AUC using GWAS summary statistics from the training data
  See ```python PRStuning.py --help``` for further usage description and options.

     
## Usage
```
python plinkLD.py --bfile BFILE [--bed BED] [--bim BIM] [--fam FAM] [--block: BLOCK_FILE] [--snplist SNPLIST] [--output OUTPUT] [--method METHOD] [--thread THREAD] [--compress COMPRESS] [--log LOG]
```

--bfile BFILE     <br />        Binary data file \
--bed: BIM                Binary data file (Genotypes) \
--bim: BIM                Binary data file (SNP info) \
--fam: FAM                Binary data file (Individual info) \
--block: BLOCK_FILE       Block file (Default: all SNPs are in one block)
--snplist: SNPLIST        SNP list file (Default: all SNP pairs are calculated)
--output: OUTPUT          Output filename (Default: LD.h5)
--method: METHOD          Correlation estimation method, including Pearson, LW (Default: Pearson)
--thread: THREAD          Thread number for calculation (Default: Total CPU number)
--compress: COMPRESS      Compression level for output (Default: 9)
--log: LOG                log file (Default: plinkLD.log)



```
python GWEB.py [-h] --ssf SSF --ref REF [--bfile BFILE] [--bed BED] [--bim BIM] [--fam FAM] [--h5geno H5GENO] [--anno ANNO] [--snplist SNPLIST]
               [--iprefix IPREFIX] [--n N] [--K K [K ...]] [--pheno PHENO] [--mpheno MPHENO] [--pheno-name PHENO_NAME] [--cov COV] [--dir DIR] [--aligned]
               [--align-only] [--weight-only] [--thread THREAD]```

GWEB: An Empirical-Bayes-based polygenic risk prediction approach using GWAS summary statistics and functional annotations

  ```
  --ssf SSF                 GWAS Summary statistic File. Should be a text file with columns SNP/CHR/BP/BETA/SE
  
  --ref REF                 Reference LD File. Should be a (full path) hdf5 file storing the LD matrix and corresponding SNP information. (plinkLD.py can be used to convert PLINK binary files into the LD hdf5 file.)
			
  --bfile BFILE             Individual-level genotype data for testing purpose. Should be PLINK binary format files with extension .bed/.bim/.fam
  
  --bed BED                 Binary genotype data with suffix .bed (Used if bfile is not provided)
  
  --bim BIM                 SNP infomation file with suffix .bim (Used if bfile is not provided)
  
  --fam FAM                 Individual information file with suffix .fam (Used if bfile is not provided)
  
  --h5geno H5GENO           Individual-level genotype data with hdf5 format
  
  --anno ANNO               Functional annotation file. Should be a hdf5 file storing annotations for SNPs
  
  --snplist SNPLIST         SNP list file used to filter SNPs
  
  --iprefix IPREFIX         Common prefix for input files (summary statistics, reference panel, genotypes, annotations)
  
  --n N                     Sample size of the GWAS summary statistics. (If provided, LDSC will be used to adjust the inflation caused by potential confounding effect.)
			
  --K K [K ...]             Number of causal components (Default:3)
  
  --pheno PHENO             External phenotype file.Should be a tabular text file. If header is not provided, the first and second columns should be FID and IID, respectively. Otherwise, there are two columns named 'FID' and 'IID'
			
  --mpheno MPHENO           m-th phenotype in the file to be used (default: 1)
  
  --pheno-name PHENO_NAME   Column name for the phenotype in the phenotype file (default:PHE)
  
  --cov COV                 covariates file, format is tabulated file with columns FID, IID, PC1, PC2, etc.
  
  --dir DIR                 Output directory
  
  --aligned                 The input has already been aligned
  
  --align-only              Align all input files only
  
  --weight-only             Weighting only, without scoring and evaluation
  
  --thread THREAD           Number of parallel threads, by default all CPUs will be utilized.
  ```

## Example Demonstration:
