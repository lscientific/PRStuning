# Require: plink, ldpred in the current path

library(MASS)
library(CorBin)
library(data.table)

M <- 10000 # number of SNPs
Np <- 1000 # number of SNPs in each block: 10 blocks
kappa <- 0.01 # prevalence
pi0 <- 0.9 # non-risk effect proportion
var0 <- 5e-4 # proportional to risk effect variance
rho <- 0.2 # auto-regressive coefficient

R <- rho^abs(outer(0:(Np-1), 0:(Np-1), "-"))
Rlist <- replicate(10, R, simplify=F)

system("mkdir LD_data") # make data folder in the current path
path <- './LD_data/'

set.seed(123)

for(n_size in 2:5){
  n0 <- 1000 * n_size; n1 <- 1000 * n_size
  n <- n0 + n1
  Ne_train <- 4/(1/n0 + 1/n1)
  y <- c(rep(0, n0), rep(1, n1))
  
  for(n_data in 1:20){
    str <- paste0(as.character(n_size), '_', as.character(n_data))
    system(paste0('mkdir ', path, str)) # make data folder for this replication in the current path
    
    ### training data
    Ctr <- list()
    Ref <- list()
    Case <- list()
    Ep0 <- rep(0, M)
    Ep1 <- rep(0, M)
    
    gen <- function(p0, p1){
      Ctr <- cBern(n0, p0, rho, 'DCP') + cBern(n0, p0, rho, 'DCP');
      Case <- cBern(n1, p1, rho, 'DCP') + cBern(n1, p1, rho, 'DCP')
      return(cbind(Ctr, Case))
    }
    
    # generate data for each block
    for (i in 1:10){
      num <- 0
      while(TRUE){ # p0, p1 have to satisfy conditions for gen() function to work. Thus use tryCatch
        num <- num + 1
        f <- matrix(runif(Np, 0.05, 0.5), Np, 1) # cannot use 0.05-0.95...
        s <- sqrt(2 * f * (1-f)); S <- diag(s[, 1])
        beta <- rbinom(Np, 1, 1-pi0)
        beta[which(beta == 1)] <- rnorm(length(beta[which(beta==1)]), 0, sd=sqrt(var0))
        beta <- matrix(beta, Np, 1)
        p0 <- f - kappa * S %*% R %*% beta
        p1 <- f + (1-kappa) * S %*% R %*% beta
        test <- tryCatch(gen(p0, p1), warning = function(w) {})
        if(!is.null(test)) break
      }
      Ep0[((i-1) * Np + 1):(i*Np)] <- p0
      Ep1[((i-1) * Np + 1):(i*Np)] <- p1
      Ref[[i]] <- cBern(500, p0, rho, 'DCP') + cBern(500, p0, rho, 'DCP')
      Ctr[[i]] <- test[, 1:Np]
      Case[[i]] <- test[, (Np+1):(2*Np)]
    }
    X0 <- do.call(cbind, Ctr)
    X1 <- do.call(cbind, Case)
    X <- rbind(X0, X1)
    X_ref <- do.call(cbind, Ref)
    fwrite(data.table(matrix(c(Ep0, Ep1), nc=2)), paste0(path, str, '/Ep_', str,'.txt'), row.names=F, col.names=F, sep='\t')
    fwrite(data.table(X), paste0(path, str, '/geno_', str, '.txt'), row.names=F, col.names=F, sep='\t')
    fwrite(data.table(X_ref), paste0(path, str, '/ref_', str, '.txt'), row.names=F, col.names=F, sep='\t')
    
    # generate GWAS file
    f0_hat <- apply(X0, 2, mean)/2
    f1_hat <- apply(X1, 2, mean)/2
    se_hat <- sqrt(2 * f0_hat * (1-f0_hat))
    X_standardized <- scale(X)
    y_standardized <- scale(y)
    eff <- (t(X_standardized) %*% matrix(y_standardized, nc=1) / n)[, 1]
    eff_se <- rep(1/sqrt(n), M)
    p_value = 2 * pnorm(-abs(eff/eff_se))
    
    SSF <- data.frame(chr=rep('chr1', M), pos=1:M, ref=rep('A', M), alt=rep('C', M), 
                      reffrq=1-f0_hat, info=rep(1,M),
                      rs=sapply(1:M, function(x) paste0('rs', x)),
                      pval=p_value, effalt=eff, eff_se=eff_se)
    fwrite(data.table(SSF), paste0(path, str, '/SSF', str, '.txt'), 
           row.names=F, col.names=T, sep='\t')
    
    # generate reference genotype data
    geno_ref <- lapply(as.list(data.frame(X_ref)), function(x){
      y = matrix(NA, nr=length(x), nc=2)
      for(i in 1:length(x)){
        if(x[i]==0) y[i, ] <- c('A', 'A')
        else{
          if(x[i]==1) y[i, ] <- c('A', 'C')
          else y[i,] <- c('C', 'C')
        }
      }
      return(y)
    })
    X_ref1 <- do.call("cbind", geno_ref)
    
    # generate reference genoype PED, MAP files, bfiles, and coord file for LDpred
    ref_ped <- data.frame(V1=1:500, V2=1:500, V3=rep(0,500), V4=rep(0,500), V5=1+rbinom(500, 1, 0.5), V6=rep(1,500), X_ref1)
    fwrite(data.table(ref_ped), paste0(path, str, '/REF', str, '.ped'), row.names=F, col.names=F, sep='\t')
    V2 <- sapply(1:M, function(x) paste0('rs', x))
    ref_map <- data.frame(V1=rep(1,M), V2=V2, V3=rep(0,M), V4=1:M)
    fwrite(data.table(ref_map), paste0(path, str, '/REF', str, '.map'), row.names=F, col.names=F, sep='\t')
    system(paste0('./plink --file ', path, str, '/REF', str, ' --out ', path, str, '/REF', str))  # generate bfiles
    system(paste0('ldpred coord --gf ', path, str, '/REF', str, ' --ssf ', path, str, '/SSF', str, '.txt --ssf-format=\'STANDARD\' --N=', n, ' --out=', path, str, '/coord_', str))
    
    cat("n =", n_size, "rep =", n_data, "\n")
  }
}


#### test data generation ####
N_test <- 1
n0_test <- 1000; n1_test <- 1000; n_test <- n0_test + n1_test

set.seed(123)
for(n_size in 2:5){
  n0 <- 1000 * n_size; n1 <- 1000 * n_size
  n <- n0 + n1
  y <- c(rep(0, n0), rep(1, n1))
  for(n_data in 1:10){
    str <- paste0(as.character(n_size), '_', as.character(n_data))
    system(paste0('mkdir ', path, str, '/test_data'))
    Ep <- as.matrix(fread(paste(path, str, '/Ep_', str, '.txt', sep=''), sep='\t'))
    Ep0 <- Ep[, 1]
    Ep1 <- Ep[, 2]
    
    for(j in 1:N_test){
      cat(j, " ")
      Ctr_test<-list()
      Case_test<-list()
      for(i in 1:10){
        Ctr_test[[i]] <- cBern(n0_test, Ep0[((i-1)*Np+1):(i*Np)], rho, 'DCP') + 
          cBern(n0_test, Ep0[((i-1)*Np+1):(i*Np)], rho, 'DCP')
        Case_test[[i]] <- cBern(n1_test, Ep1[((i-1)*Np+1):(i*Np)], rho, 'DCP') + 
          cBern(n1_test, Ep1[((i-1)*Np+1):(i*Np)], rho, 'DCP')
      }
      X0_test <- do.call(cbind, Ctr_test)
      X1_test <- do.call(cbind, Case_test)
      X_test <- rbind(X0_test, X1_test)
      save(X_test, file=paste0(path, str, '/test_data/test_', j, '.RData'))
    }
    cat("\t")
  }
}





