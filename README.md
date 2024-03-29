# PRStuning: Estimate Testing AUC for Binary Phenotype Using GWAS Summary Statistics from the Training Data

- If you use this software, please cite https://www.nature.com/articles/s41467-023-44009-0

- Software dependency: Python version: >=3.8; R; PLINK; GSL (optional for speedup) \
Python package dependency: numpy; pandas; scipy; scikit-learn;	rpy2; cython; tables; pysnptools; CythonGSL; bgen-reader

## Installation and Compilation
### Download PRStuning
```wget https://github.com/lscientific/PRStuning/archive/refs/heads/master.zip -O PRStuning.zip``` \
```unzip PRStuning.zip``` \
```rm PRStuning.zip``` \
```cd ./PRStuning-master```


### Install GSL (optional for speedup)
For Linux/Mac, see
https://solarianprogrammer.com/2020/01/26/getting-started-gsl-gnu-scientific-library-windows-macos-linux/#gsl_usage_example
for detailed installation

For Windows, you can download the compiled library directly from https://code.google.com/p/oscats/downloads/list \
Copy bin/libgsl-0.dll and bin/libgslcblas-0.dll into the working directory \
Special tips for installing GSL on Windows can be found at http://joonro.github.io/blog/posts/installing-gsl-and-cythongsl-in-windows/

### Install Python packages via pip:
```
pip install scipy numpy pandas scikit-learn rpy2 cython tables pysnptools CythonGSL bgen-reader
```

### Compilation
```
python setup.py build_ext --inplace
```

Installation is expected to be finished within a few minutes

## Typical Workflow:
- (Optional) Use ```plinkLD.py``` to calculate the LD matrix for PLINK or Oxford binary format genotype data of a reference panel.
  See ```python plinkLD.py --help``` for further usage description and options

- Use ```PRStuning.py``` to obtain estimated AUC using GWAS summary statistics from the training data. 
  See ```python PRStuning.py --help``` for further usage description and options

     
## Usage
### plinkLD.py (optional)
```
python plinkLD.py --bfile BFILE [--bed BED] [--bim BIM] [--fam FAM] [--block: BLOCK_FILE] [--snplist SNPLIST] \
[--output OUTPUT] [--method METHOD] [--thread THREAD] [--compress COMPRESS] [--log LOG]
```

```--bfile BFILE```    &nbsp;&nbsp;&nbsp;     Binary data file \
```--bed: BIM```   &nbsp;&nbsp;&nbsp;         Binary data file (Genotypes) \
```--bim: BIM```   &nbsp;&nbsp;&nbsp;         Binary data file (SNP info) \
```--fam: FAM```   &nbsp;&nbsp;&nbsp;         Binary data file (Individual info) \
```--bgen: BGEN```   &nbsp;&nbsp;&nbsp;       Binary data file with Oxford format for getting dosages \
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
 [--pheno PHENO] [--aligned] [--n N] [--dir DIR] [--thread THREAD]
```
```--ssf SSF```    &nbsp;&nbsp;&nbsp;      GWAS Summary statistic file. Should be a text file with columns SNP/CHR/BP/BETA/SE \
```--weight WEIGHT```   &nbsp;&nbsp;&nbsp;   PRS model weight file. Should be a tabular text file with the five columns being SNP/CHR/BP/A1/A2/ and the following columns correspond to weights with different PRS model parameters \
```--n0 N0```   &nbsp;&nbsp;&nbsp;        Control sample size of the GWAS summary statistics \
```--n1 N1```   &nbsp;&nbsp;&nbsp;         Case sample size of the GWAS summary statistics \
```--pruning``` &nbsp;&nbsp;&nbsp;  If not pruning, a Gibbs-sampling-based State-Augmentation for Marginal Estimation (SAME) is used. If pruning, EM algorithm is used on SNPs with non-zero PRS weights. Do not provide a reference LD file if pruning.  \
```--ref REF``` &nbsp;&nbsp;&nbsp;    Reference LD file. Provide if not pruning. Should be a (full path) hdf5 file  storing the LD matrix and corresponding SNP information (plinkLD.py can be used to convert PLINK binary files into the LD hdf5 file) \
```--homo HOMO``` &nbsp;&nbsp;&nbsp;      If the summary statistics are from a single homogeneous GWAS cohort, use --homo option to not further shrink LD (Default: False) \
```--geno GENO```  &nbsp;&nbsp;&nbsp;     Individual-level genotype data for testing purposes (if provided). Should be PLINK binary format files with extension .bed/.bim/.fam \
```--pheno PHENO```  &nbsp;&nbsp;&nbsp;     External phenotype data in a tabluar text file. Should be provided if testing genotype data are provided. If the header is not provided, the first, second columns and third columns should be FID, IID, and PHE. Otherwise, there are three columns named 'FID', 'IID', and 'PHE' \
```--dir DIR```     &nbsp;&nbsp;&nbsp;     Output directory (Default: ./output)\
```--thread THREAD```  &nbsp;&nbsp;&nbsp;   Number of parallel threads, by default all CPUs will be utilized (Default: all CPUs are utilized)

## Reference Data
The reference panel based on the 1000 Genomes Project is available [here](https://figshare.com/s/8fc8167b4b5569e53f78 )
- 1000G.EUR.QC.bim&fam&bed are the 1000G plink files
- fourier_ls-all.bed is the SNP block file
  
Generate the reference LD matrix file using ```plinkLD.py```
```
python plinkLD.py --bfile 1000G.EUR.QC --block fourier_ls-all.bed --out LD.h5
```
Note that binary data file with Oxford format (bgen file) can be an alternative to the PLINK file

For European data, the LD matrix file is already available [here](https://figshare.com/s/648d346f8092c5c71685)
- ld_1kg_EUR_hm3_shrink.h5 is the LD matrix file
  


## Example Demonstration

./simdata/: Simulated data with correlated SNPs 
- GWAS training dataset ssf.txt with sample size N0=5000, N1=5000 
- PRS weight file weight_ldpred.txt is generated using [LDpred](https://github.com/bvilhjal/ldpred). The parameters are the fractions of causal markers
- PRS weight file weight_pt.txt only contains independent SNPs after pruning. The parameters are p-value thresholds
- Testing genotype data ./simdata/geno(.bim/bed/fam) have 500 controls and 500 cases 
- Reference data are generated from 2000 controls 
- All datasets share the same LD structure AR(1) with rho=0.2

### Not pruning
When ```--pruning``` is not used, the SNPs are treated as dependent. Thus use Gibbs sampling-based SAME algorithm to estimate.
- Step 1
  \
Use ```plinkLD.py``` to generate the reference LD matrix file
```
python plinkLD.py --bfile ./simdata/ref --output ./simdata/ref.h5 
```
- Step 2 
  \
Use ```--homo``` here since using simulated data. For real GWAS summary statistics that are from a single homogeneous GWAS cohort, do not use this option.
```
python PRStuning.py --ssf ./simdata/ssf.txt --weight ./simdata/weight_ldpred.txt --ref ./simdata/ref.h5 --pheno ./simdata/pheno.txt --geno ./simdata/geno --n0 5000 --n1 5000 --dir ./simdata/output/ --homo
```
- Results \
	```./simdata/output/auc_results.txt``` includes the PRStuning and testing AUC results. Each row corresponds to a PRS parameter and the two columns correspond to PRStuning AUC and testing AUC respectively  \
	Files with prefix ```./simdata/output/align_``` are aligned datasets \
	```./simdata/output/prs_results/``` contains PRS scoring results using PLINK \
	```./simdata/output/param.txt``` contains the estimated parameters using SAME algorithm \
	```./simdata/output/beta_est.txt``` contains the estimated ground truth effect sizes \
	```./simdata/output/log.txt``` is the log file for PRStuning.py \
	```./simdata/ref.log``` is the log file for plinkLD.py  

### Pruning
When ```--pruning``` is used, the SNPs are treated as independent. Thus use EM algorithm to estimate. No reference data should be provided
```
python PRStuning.py --ssf ./simdata/ssf.txt --weight ./simdata/weight_pt.txt --pheno ./simdata/pheno.txt --geno ./simdata/geno --n1 5000 --n0 5000 --dir ./simdata/output_pt/ --pruning 
```
- Results \
 	```./simdata/output/auc_results.txt``` includes the PRStuning and testing AUC results. Each row corresponds to a PRS parameter and the two columns correspond to PRStuning AUC and testing AUC respectively  \
  	Files with prefix ```./simdata/output_pt/align_``` are aligned datasets \
  	```./simdata/output_pt/prs_results/``` contains PRS scoring results using PLINK \
  	```./simdata/output_pt/param.txt``` contains the estimated parameters using EM algorithm \
  	```./simdata/output_pt/log.txt``` is the log file for PRStuning.py





