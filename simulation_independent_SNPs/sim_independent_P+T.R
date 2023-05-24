library(MASS)
library(pROC)
library(fMultivar)
library(ggplot2)

source("./snpEM.R")
thresholds <- c(5e-6, 5e-5, 5e-4, 5e-3, 5e-2, 5e-1, 1)

M <- 10000  # number of SNPs
kappa <- 0.01  # prevalence
pi0 <- 0.95  # non-risk effect proportion
var0 <- 0.001  # proportional to risk effect variance

n0_test <- 1000; n1_test <- 1000; n_test <- n0_test + n1_test # testing sample size
Ne_test <- 4 * n0_test * n1_test / (n0_test + n1_test) # testing effective sample size
y_test <- c(rep(0, n0_test), rep(1, n1_test)) # phenotypes of testing data
N_test <- 1
N2 <- 100 # number of Monte Carlo
n_rep <- 50 # replication number of datasets

set.seed(1234)
res_AUC <- array(NA, dim = c(length(thresholds), 5, n_rep, 5),
                 dimnames = list(threshold=thresholds, size=2000*1:5, data=1:n_rep,
                                 Methods=c("Testing", "test_summ", "train_geno", "Unadj. tuning", "PRStuning")))
for(n_size in 2:5){
  n0 <- n_size * 1000; n1 <- n_size * 1000; n <- n0 + n1  #training sample size
  y <- c(rep(0, n0), rep(1, n1))  # phenotype
  Ne_train <- 4 * n0 * n1 / (n0 + n1)  #training effective sample size
  
  for(n_data in 1:n_rep){
    t1 <- Sys.time()
    f_p <- runif(M, 0.05, 0.95) # population allel frequency
    se <- sqrt(2 * f_p * (1 - f_p))
    u <- rbinom(M, 1, 1 - pi0)
    u[which(u==1)] <- rnorm(length(u[which(u == 1)]), 0, sqrt(var0)) #u_m=(f_1m-f_0m)/se_m
    f1 <- f_p + (1-kappa) * u * se
    f0 <- f_p - kappa * u * se
    f1[which(f1 < 0.05)] <- 0.05; f1[which(f1 > 0.95)] <- 0.95
    f0[which(f0 < 0.05)] <- 0.05; f0[which(f0 > 0.95)] <- 0.95
    
    #### Generate training geno data ####
    X <- matrix(data=NA, nrow=n, ncol=M)  #genotype
    for(m in 1:M){
      X[1:n0, m] <- rbinom(n0, 2, f0[m])
      X[(n0+1):n, m] <- rbinom(n1, 2, f1[m])
    }
    # estimated MAF
    f0_hat <- apply(X[1:n0, ], 2, mean) / 2
    f1_hat <- apply(X[(n0+1):n, ], 2, mean) / 2
    se_hat <- sqrt(2 * f0_hat * (1 - f0_hat)) # approx
    X_standardized <- scale(X)
    y_standardized <- scale(y)
    eff <- (t(X_standardized) %*% matrix(y_standardized, nc=1) / n)[, 1]
    eff_se <- rep(1/sqrt(n), M)
    Z <- eff / eff_se
    p = 2 * pnorm(-abs(Z))
    
    #### generate testing geno data ####
    X_test <- list()
    for(num in 1:N_test){
      X_test[[num]] <- matrix(NA, n_test, M)
      for(m in 1:M){
        X_test[[num]][1:n0_test, m] <- rbinom(n0_test, 2, f0[m])
        X_test[[num]][(n0_test+1):n_test, m] <- rbinom(n1_test, 2, f1[m])
      }
    }
    
    #### different thresholds ####
    for(t in 1:length(thresholds)){
      coef <- eff
      threshold = thresholds[t]
      coef[which(p > threshold)] <- 0 # coefficients after pruning
      
      auc_test <- rep(0, N_test)
      auc_test_summ <- rep(0, N_test)
      for(num in 1:N_test){
        # AUC from testing geno data
        prs_test <- apply(X_test[[num]], 1, function(x) sum(x * coef) )
        auc_test[num] <- auc(y_test, prs_test, levels=c(0, 1), direction="<")
        
        # SummaryAUC from testing summary data
        f0_hat_test <- apply(X_test[[num]][1:n0_test,], 2, mean)/2
        f1_hat_test <- apply(X_test[[num]][(n0_test+1):n_test,], 2, mean)/2
        se_hat_test <- sqrt(2 * f0_hat_test * (1 - f0_hat_test))
        Z_test <- (f1_hat_test - f0_hat_test) / (se_hat_test / sqrt(Ne_test))
        delta_hat_test <- sqrt(1 / n0_test + 1 / n1_test) *
          sum(coef * Z_test * se_hat_test) / sqrt(2 * sum(coef^2 * se_hat_test^2))
        auc_test_summ[num] <- pnorm(delta_hat_test)
      }
      res_AUC[t, n_size, n_data, "Testing"] <- mean(auc_test)
      res_AUC[t, n_size, n_data, "test_summ"] <- mean(auc_test_summ[!is.na(auc_test_summ)])
      
      #### AUC from training geno data
      prs <- apply(X, 1, function(x) sum(x * coef) )
      res_AUC[t, n_size, n_data, "train_geno"] <- auc(y, prs, levels=c(0, 1), direction="<")
      
      #### AUC from training summary data
      delta_hat <- sqrt(1/n0 + 1/n1) * sum(coef * Z * se_hat) / sqrt(2 * sum(coef^2 * se_hat^2))
      res_AUC[t, n_size, n_data, "Unadj. tuning"] <- pnorm(delta_hat)
      
      #### adjusted AUC from training summary data ####
      em <- snpEM(Z, K=1, maxIter=1000, tol=1e-4, beta0=0, info=F)
      pi0_tilte <- em$pi0*dnorm(Z) / (em$pi0*dnorm(Z) + (1-em$pi0)*dnorm(Z, sd=sqrt(em$sigma2+1)))
      lambda <- 1 / (1 + 1 / em$sigma2)
      sig2_star <- 1 + lambda * Ne_test / Ne_train
      
      auc_post_rs <- rep(NA, N2)
      for(j in 1:N2){
        Z_post_rs <- rep(0, M)
        for(m in 1:M){
          c <- rbinom(1, 1, pi0_tilte[m])
          if(c == 0) Z_post_rs[m] <- rnorm(1, lambda * Z[m], sqrt(lambda))
        }
        delta_post_rs <- 2 * sum(coef * Z_post_rs * se_hat / sqrt(Ne_train)) /
          sqrt(sum(coef^2 * 2 * se_hat^2))
        auc_post_rs[j] <- pnorm(delta_post_rs)
      }
      res_AUC[t, n_size, n_data, "PRStuning"] <- mean(as.vector(auc_post_rs))
    }
    t2 <- Sys.time()
    cat("n =", n, "rep = ", n_data, "\n")
    print(t2-t1) # time used for each data replication
  }
}
save(res_AUC, file="./ind_sim_P+T_res.RData") # save data to current path


#### results ####
library(ggplot2)
library(latex2exp)
library(ggpubr)
load("./ind_sim_P+T_res.RData")

thr_chars <- c("5e-6","5e-5", "5e-4", "5e-3", "5e-2", "5e-1", "1")

theme <- theme(legend.title = element_text(size=6), legend.text = element_text(size=5), 
               legend.key.size = unit(0.3, 'cm'), plot.title = element_text(size=7), 
               axis.text=element_text(size=6), axis.title=element_text(size=7))
theme_update(plot.title = element_text(hjust = 0.5))
p <- list()
for(t in 1:7){
  dat_auc <- as.data.frame.table(res_AUC[t, 2:5, , c(5, 1, 4)])
  means <- aggregate(Freq ~ Methods + size, dat_auc, mean)
  pd <- position_dodge(.6)
  p[[t]] <- ggplot(data=dat_auc, aes(x=size, y=Freq, fill=Methods)) + 
    geom_boxplot(lwd=0.1, position=pd, width=.5, outlier.size=.1) + 
    #geom_boxplot(data=means, fatten=0.1, aes(x = size, y = Freq), position=pd, width=.5) +
    #scale_fill_brewer(palette = "Set2") + 
    ylab("AUC") + xlab("Sample size") + 
    ylim(c(0.55, 1)) +
    ggtitle(paste("threshold =", thr_chars[t])) + theme +
    scale_fill_manual(values=c("#797979", "#F5CB1D", "#E94F42"))
}
p <- ggarrange(p[[7]], p[[6]], p[[5]], p[[4]], p[[3]], p[[2]], p[[1]], 
               ncol=2, nrow=4, common.legend = TRUE, legend="right")
ggsave("./sim_ind_P+T_res.pdf", p, width = 4, height = 6) # save figure to current path



