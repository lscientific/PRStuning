library(MASS)
library(pROC)
library(fMultivar)
library(CorBin)
library(data.table)

M <- 10000; Np <- 1000 # 10 blocks
rho <- 0.2 # auto-regressive coefficient

n0_test <- 1000; n1_test <- 1000; n_test <- n0_test + n1_test
N_test <- 1 # number of testing datasets
y_test <- c(rep(0, n0_test), rep(1, n1_test))
R <- rho^abs(outer(0:(Np-1), 0:(Np-1),"-"))
Rlist <- replicate(10, R, simplify=F)
R_w <- as.matrix(Matrix::bdiag(Rlist))
n_rep <- 20 # number of replications

path <- './LD_data/'

fracs <- c(1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 3e-1, 3e-2, 3e-3, 3e-4, 3e-5)
char_frac <- c('_p1.0000e+00.txt', '_p1.0000e-01.txt', '_p1.0000e-02.txt', 
               '_p1.0000e-03.txt', '_p1.0000e-04.txt', '_p1.0000e-05.txt', '_p3.0000e-01.txt', 
               '_p3.0000e-02.txt', '_p3.0000e-03.txt', '_p3.0000e-04.txt', '_p3.0000e-05.txt')


res_LDpred_AUC <- array(NA, dim = c(5, n_rep, 3, length(fracs)), 
                        dimnames = list(size=2000 * 1:5, data=1:n_rep, Methods = c("Testing", "Unadj. tuning", "PRStuning"), 
                                        fracs = fracs))

for(t in 1:length(fracs)){
  for(n_size in 2:5){
    n0 <- 1000 * n_size; n1 <- 1000 * n_size
    n <- n0 + n1
    y <- c(rep(0, n0), rep(1, n1))
    Ne_train <- 4/(1/n0 + 1/n1)
    
    for(n_data in 1:20){
      str <- paste0(as.character(n_size), '_', as.character(n_data))
  
      X <- as.matrix(fread(paste0(path, str, '/geno_', str, '.txt'), sep='\t'))
      f0_hat <- apply(X[1:n0, ], 2, mean) / 2
      f1_hat <- apply(X[(n0+1):n, ], 2, mean) / 2
      f_pool_hat <- (n0 * f0_hat + n1 * f1_hat) / (n0 + n1)
      se_hat <- sqrt(2 * f0_hat * (1 - f0_hat))
      
      SSF <- data.frame(fread(file=paste0(path, str, '/SSF', str, '.txt'), sep='\t'))
      beta_SE <- SSF$eff_se
      betaHat <- SSF$effalt
      Z <- betaHat / beta_SE
      Beta <- as.matrix(fread(paste0(path, str, '/Beta', str, '.txt'), sep='\t'))
      N_sample <- nrow(Beta)
      
      # get the LDpred PRS coefficients
      system(paste0("ldpred gibbs --cf=", path, str, "/coord_", str, " --ldr=5 --ldf=", 
                    path, str,"/ldf --out=", path, str, "/weight --f ", fracs[t]))
      LDpred <- data.frame(fread(paste0(path, str, '/weight_LDpred', char_frac[t])))
      coef <- rep(0, M)
      coef[LDpred$pos]<- LDpred$ldpred_beta 
      
      ### testing data
      auc_test <- rep(0, N_test)
      for(j in 1:N_test){
        X_test <- get(load(file=paste0(path, str, '/test_data/test_', j, '.RData')))
        prs_test <- apply(X_test, 1, function(x) sum(x * coef) )
        auc_test[j] <- auc(y_test, prs_test, levels=c(0, 1), direction="<")
      }
      res_LDpred_AUC[n_size, n_data, "Testing", t] <- mean(auc_test) 
      
      ### Unadj. tuning
      sig2 <- matrix((coef * se_hat), nr=1) %*% R_w %*% matrix((coef * se_hat), nc=1) # variance of PRS
      f_d_unad <- Z * se_hat / sqrt(Ne_train)
      delta_hat <- 2 * sum(coef * f_d_unad) / sqrt(2 * sig2)
      res_LDpred_AUC[n_size, n_data, "Unadj. tuning", t] <- pnorm(delta_hat)  

      ### PRStuning
      wp <- matrix(NA, N_sample, 10)
      for(i in 1:10){
        R_Beta <- apply(Beta[, ((i-1)*Np+1):(Np*i)], 1, 
                        function(x) as.vector(R %*% diag(1/beta_SE[((i-1)*Np+1):(Np*i)]) %*% matrix(x, nr = Np)) )
        wp[, i] <- apply(R_Beta, 2, function(x) {
          sum(coef[((i-1)*Np+1):(Np*i)] * se_hat[((i-1)*Np+1):(Np*i)] * as.vector(x) / sqrt(Ne_train))
        })
      }
      delta_ad <- rep(0, N_sample)
      for(i in 1:N_sample){
        delta_ad[i] <- 2 * sum(wp[i, ]) / sqrt(2 * sig2)
      }
      res_LDpred_AUC[n_size, n_data, "PRStuning", t] <- mean(pnorm(delta_ad))
      
      cat("\n", "LDpred: n_size =", n_size, "n_data =", n_data, "frac =", fracs[t], "\n",
          "PRStuning=", round(res_LDpred_AUC[n_size, n_data, "PRStuning", t], 3),
          "testing=", round(res_LDpred_AUC[n_size, n_data, "Testing", t], 3), "\n")
    }
  }
}
save(res_LDpred_AUC, file = "./LD_sim_LDpred_res.RData") 



#### results ####
library(ggplot2)
library(ggpubr)

load("./LD_sim_LDpred_res.RData")
frac_chars <- c('1', '1e-1', '1e-2', '1e-3', '1e-4', '1e-5', '3e-1', '3e-2', '3e-3', '3e-4', '3e-5')
theme <- theme(legend.title = element_text(size=6), legend.text = element_text(size=5), 
               legend.key.size = unit(0.3, 'cm'), plot.title = element_text(size=7), 
               axis.text=element_text(size=6), axis.title=element_text(size=7))
theme_update(plot.title = element_text(hjust = 0.5))

p <- list()
for(t in 1:length(fracs)){
  dat <- as.data.frame.table(res_LDpred_AUC[2:5, , c(3, 1, 2), t])
  pd <- position_dodge(.6)
  p[[t]] <- ggplot(dat, aes(x = size, y = Freq, fill = Methods))  + 
    geom_boxplot(lwd=0.1, position=pd, width=.5, outlier.size=.1) + 
    ggtitle(paste("fraction =", frac_chars[t])) + 
    ylim(c(0.5, 1)) +
    labs(x = "Sample size", y = "AUC") + theme  +
    scale_fill_manual(values=c("#797979", "#F5CB1D", "#E94F42"))
}
p_list <- ggarrange(p[[1]], p[[7]], p[[2]], p[[8]], p[[3]], p[[9]], p[[4]], p[[10]],
                    p[[5]], p[[11]], p[[6]], 
                    ncol=2, nrow=6, common.legend = TRUE, legend="right")
p_list
ggsave("./sim_LD_LDpred_res.pdf", p_list, width = 4, height = 7)




