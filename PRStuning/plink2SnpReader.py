import sys, os, shutil
import numpy as np
tmp = sys.stdout
sys.stdout = None
from pysnptools.snpreader import Bed
sys.stdout = tmp

def getGenotype(bfile=None, bed=None, bim=None, fam=None, rmtmp=True, thread=None):
    if bfile is not None:
        if bed is None:
            bed = bfile + '.bed'
        if bim is None:
            bim = bfile + '.bim'
        if fam is None:
            fam = bfile + '.fam'
    bedRoot, _ = os.path.splitext(bed)
    bimRoot, _ = os.path.splitext(bim)
    famRoot, _ = os.path.splitext(fam)
    if bimRoot!=bedRoot:
        shutil.copyfile(bim, bedRoot+'.bim')
    if famRoot!=bedRoot:
        shutil.copyfile(fam, bedRoot+'.fam')
    
    snp_on_disk = Bed(bedRoot, count_A1=False, num_threads=thread)
    genotype = snp_on_disk.read(dtype='int8',_require_float32_64=False).val
    genotype = genotype.astype(np.float32)
    genotype[genotype==-127] = np.nan #3
    
    if rmtmp:
        if bimRoot!=bedRoot:
            os.remove(bedRoot+'.bim')
        if famRoot!=bedRoot:
            os.remove(bedRoot+'.fam')
    
    return genotype

def getSnpReader(bfile=None, bed=None, bim=None, fam=None, thread=-1):
    if bfile is not None:
        if bed is None:
            bed = bfile + '.bed'
        if bim is None:
            bim = bfile + '.bim'
        if fam is None:
            fam = bfile + '.fam'
    bedRoot, _ = os.path.splitext(bed)
    bimRoot, _ = os.path.splitext(bim)
    famRoot, _ = os.path.splitext(fam)
    if bimRoot!=bedRoot:
        shutil.copyfile(bim, bedRoot+'.bim')
    if famRoot!=bedRoot:
        shutil.copyfile(fam, bedRoot+'.fam')
    
    if thread <0: thread=None
    snp_on_disk = Bed(bedRoot, count_A1=False, num_threads=thread)
   
    return snp_on_disk

def removeTmp(bfile=None, bed=None, bim=None, fam=None):
    if bfile is not None:
        if bed is None:
            bed = bfile + '.bed'
        if bim is None:
            bim = bfile + '.bim'
        if fam is None:
            fam = bfile + '.fam'
    bedRoot, _ = os.path.splitext(bed)
    bimRoot, _ = os.path.splitext(bim)
    famRoot, _ = os.path.splitext(fam)
    
    if bimRoot!=bedRoot:
        os.remove(bedRoot+'.bim')
    if famRoot!=bedRoot:
        os.remove(bedRoot+'.fam')
    return
