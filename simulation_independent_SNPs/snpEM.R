# TODO: Infer the z value's distribution using EM with finite mixture model
# Input: z value-z[vector], number of non-null components-K[scalar,4]
#        Maximum iteration number-maxIter[scalar,1000]
#        Tolerance-tol[scalar, 1e-8], Show iteration info-info[bool, T]
# Output: The proportion of null components-pi0[scale]
#         The proportion of non-null components-Pi1[vector]
#         The estimated variance of non-null components-sigma2[vector]
#         The probability of being null component-h0[vector]
#                 The probability of being each of the non-null components-h[list]
#         Iteration number-iter[scalar]
#         Expectation of complete log likelihood-Qval[scalar]
snpEM <- function(z, K=1, maxIter=1000, tol=1e-4, beta0=0, info=TRUE) {
  #For real data usage, a certain penality to pi0, such as beta0=length(z)/5 is needed.
  #For simulated data, this penalization will increase bias.
  m <- length(z)
  #initialization
  pi0_0 <- 0.95
  Pi1_0 <- rep((1-pi0_0)/K, K)
  sigma2_0 <- rgamma(K,1)
  
  pi0_t <- pi0_0
  Pi1_t <- Pi1_0
  sigma2_t <- sigma2_0
  h <- list()
  h0 <- 0
  tmpH0 <- 0
  tmpH <- list()
  nanVal <- function(x) ifelse(is.nan(x), 0, x)
  for(iter in 1:maxIter) {
    #E step
    for(i in 0:K) {
      if(i == 0) tmpH0 <- pi0_t * dnorm(z)
      else tmpH[[i]] <- Pi1_t[i] * dnorm(z, sd = sqrt(1 + sigma2_t[i]))
    }
    norH <- tmpH0 + Reduce('+', tmpH)
    h0 <- nanVal(tmpH0/norH)
    h <- lapply(tmpH, FUN = function(x) return(nanVal(x / norH)))
    
    #M step
    pi0_t1 <- (sum(h0) + beta0) / (m + beta0)
    Pi1_t1 <- sapply(h, FUN = sum) / (m + beta0)
    sigma2_t1 <- sapply(h, FUN = function(x) {
      if(sum(x) == 0) return(0)
      else return(max(sum(x * z^2) / sum(x) - 1, 0))
    } )
    
    if( (abs(nanVal((pi0_t1-pi0_t)/pi0_t)) < tol)
        && sqrt(nanVal(sum((Pi1_t1-Pi1_t)^2) / sum(Pi1_t^2))) < tol
        && Reduce('+',Map(function(x,y){return(ifelse(sum(y^2)==0, 0, sqrt(sum((x-y)^2)/sum(y^2))))},sigma2_t1,sigma2_t))<tol) break
    else{
      pi0_t <- pi0_t1
      Pi1_t <- Pi1_t1
      sigma2_t <- sigma2_t1
    }
  }
  
  #       Qval<-ifelse(tmpH0==0,0,sum(h0*log(tmpH0)))+sum(Reduce('+',Map('*',h,lapply(tmpH,function(x){return(ifelse(x==0,0,log(x)))}))))
  logF <- function(x) ifelse(x == 0, 0, log(x))
  Qval <- sum(h0 * logF(tmpH0)) + sum(Reduce('+', Map('*', h, lapply(tmpH, logF))))
  if(info){
    cat('pi0:', pi0_t,'\n')
    cat('Pi1:\n')
    print(Pi1_t)
    cat('sigma^2:\n')
    print(sigma2_t)
    cat('Iteration:', iter,'\nLog-likelihood:', Qval,'\n')
  }
  return(list(pi0=pi0_t,Pi1=Pi1_t,sigma2=sigma2_t,h0=h0,h=h,iter=iter,Qval=Qval))
}
