#'@title Model selection using extended BIC
#'@description
#' This function performs model selection using extended BIC and the ridge partial correlation coefficients
#' @rdname eBIC
#' @param rpc.obj an object of class \code{rpc}, containing the following items: \cr
#'  - X: matrix of covariates (excluding intercept) \cr
#'  - y: response vector \cr
#'  - rpc: vector of ridge partial correlation coefficients, one for each column in X. \cr
#' @returns BIC.PATH vector of eBIC of each model, starting with the intercept only model.
#' @returns model.best sorted indices (in increased order) of the model with the smallest EBIC
#' 
#' @author An Nguyen
#' @author Somak Dutta
#' @author Maintainer: Somak Dutta <somakd@iastate.edu>
#' 
#' 
#' @examples
#' n <- 50; p <- 400;
#' trueidx <- 1:3
#' truebeta <- c(4,5,6)
#' X <- matrix(rnorm(n*p), n, p) # n x p covariate matrix
#' y <- 0.5 + X[,trueidx] %*% truebeta + rnorm(n)
#' res <- rpc(X,y, lambda = 0.1, ncores = 1)
#' eBIC(res) # model.best: model selected by eBIC
#' 
#' 
#' @export
eBIC <- function(rpc.obj) {
  
  X <- rpc.obj$X
  Y <- rpc.obj$y
  r <- rpc.obj$rpc
  n <- nrow(X)
  
  if(ncol(X) != length(r)) {
    stop("The length of rpc vector must match the number of columns of X")    
  }
  
  vars.top <- order(abs(r),decreasing = T)[1:min({n-2}, length(r))]
  
  # guard against duplicated columns in Z
  r.top = abs(r)[vars.top]
  dups <- duplicated(r.top)
  if(sum(dups) > 0)
    warning("Duplicated variables found among top variables.
            Duplicates are removed.")
  vars.top <- vars.top[!dups]
  
  W <- cbind(1, X[,vars.top])
  R <- chol(crossprod(W))
  
  v <- backsolve(R,crossprod(W, Y), transpose = T)
  d0 <- ncol(X)
  RSS <- sum(Y^2)-cumsum(v^2)
  d <- 1:length(RSS)
  EBIC <- log(RSS/n)+(d*log(n*d0^2)/n)
  min_ebic = which.min(EBIC)
  if(min_ebic == 1)   {
    model.best <- integer(0)
  } else   {
    model.best <- vars.top[c(1:(min_ebic-1))]
  }
  
  list(BIC.PATH = EBIC, model.best = sort(model.best))
}


