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
First 

```conda install python">=3.8" scipy numpy pandas scikit-learn rpy2 cython sklearn pip``` \
or 	```pip install scipy numpy pandas scikit-learn rpy2 cython sklearn``` 

Then
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

  Use ```PRStuning.py``` to obtain estimated AUC using GWAS summary statistics from the training data. 
  See ```python PRStuning.py --help``` for further usage description and options.

     
## Usage
### plinkLD.py (optional step)
```
python plinkLD.py --bfile BFILE [--bed BED] [--bim BIM] [--fam FAM] [--block: BLOCK_FILE] [--snplist SNPLIST] \
[--output OUTPUT] [--method METHOD] [--thread THREAD] [--compress COMPRESS] [--log LOG]
```

```--bfile BFILE```    &nbsp;&nbsp;&nbsp;     Binary data file \
```--bed: BIM```   &nbsp;&nbsp;&nbsp;         Binary data file (Genotypes) \
```--bim: BIM```   &nbsp;&nbsp;&nbsp;         Binary data file (SNP info) \
```--fam: FAM```   &nbsp;&nbsp;&nbsp;         Binary data file (Individual info) \
```--block: BLOCK_FILE``` &nbsp;&nbsp;&nbsp;  Block file (Default: all SNPs are in one block) \
```--snplist: SNPLIST``` &nbsp;&nbsp;&nbsp;    SNP list file (Default: all SNP pairs are calculated) \
```--output: OUTPUT``` &nbsp;&nbsp;&nbsp;      Output filename (Default: LD.h5) \
```--method: METHOD```  &nbsp;&nbsp;&nbsp;     Correlation estimation method, including Pearson, LW (Default: Pearson) \
```--thread: THREAD```  &nbsp;&nbsp;&nbsp;     Thread number for calculation (Default: Total CPU number) \
```--compress: COMPRESS```  &nbsp;&nbsp;&nbsp; Compression level for output (Default: 9) \
```--log: LOG```     &nbsp;&nbsp;&nbsp;     log file (Default: plinkLD.log) 

### PRStuning.py
```
python PRStuning.py --ssf SSF --weight WEIGHT --n0 N0 --n1 N1 [--pruning] [--ref REF] [--homo HOMO] [--geno GENO] \
 [--pheno PHENO] [--n N] [--dir DIR] [--thread THREAD]
```
```--ssf SSF```    &nbsp;&nbsp;&nbsp;      GWAS Summary statistic file. Should be a text file with columns SNP/CHR/BP/BETA/SE \
```--weight WEIGHT```   &nbsp;&nbsp;&nbsp;   PRS model weight file. Should be a tabular text file with the five columns being SNP/CHR/BP/A1/A2/ and the following columns correspond to weights with different PRS model parameters \
```--n0 N0```   &nbsp;&nbsp;&nbsp;        Control sample size of the GWAS summary statistics \
```--n1 N1```   &nbsp;&nbsp;&nbsp;         Case sample size of the GWAS summary statistics \
```--pruning``` &nbsp;&nbsp;&nbsp;  If not pruning, an Gibbs-sampling-based State-Augmentation for Marginal Estimation (SAME) is used. If pruning, EM algorithm is used on SNPs with non-zero PRS weights. Do not provide a reference LD file if pruning.  \
```--ref REF``` &nbsp;&nbsp;&nbsp;    Reference LD file. Provide if not pruning. Should be a (full path) hdf5 file  storing the LD matrix and corresponding SNP information (plinkLD.py can be used to convert PLINK binary files into the LD hdf5 file) \
```--homo HOMO``` &nbsp;&nbsp;&nbsp;      If the summary statistics are from a single homogeneous GWAS cohort, use homo to not to shrink LD (Default: False) \
```--geno GENO```  &nbsp;&nbsp;&nbsp;     Individual-level genotype data for testing purposes (if provided). Should be PLINK binary format files with extension .bed/.bim/.fam \
```--pheno PHENO```  &nbsp;&nbsp;&nbsp;     External phenotype file. Should be provided if geno. Should be a tabular text file. If the header is not provided, the first, second columns and third columns should be FID, IID, and PHE. Otherwise, there are three columns named 'FID', 'IID', and 'PHE' \
```--n N```  &nbsp;&nbsp;&nbsp; Sample size of the GWAS summary statistics. (If provided, LDSC will be used to adjust the inflation caused by potential confounding effects. Default: 0) \
```--dir DIR```     &nbsp;&nbsp;&nbsp;     Output directory (Default: ./output)\
```--thread THREAD```  &nbsp;&nbsp;&nbsp;   Number of parallel threads, by default all CPUs will be utilized (Default: all CPUs are utilized)



## Example Demonstration:



