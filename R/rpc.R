#'@title Ridge Partial Correlations
#'@description
#'The sample ridge partial correlations (RPC) can be used for a simple, efficient, and scalable variables screening method 
#'in ultra-high dimensional linear regression. 
#'These coefficients incorporate both the ridge regularized estimates of the regression coefficients
#'and the ridge regularized partial variances of the predictor variables, 
#'providing the screening method with sure screening property 
#'without strong assumptions on the marginal correlations. 
#'For more details, please see Wang et al. (2025).
#'
#'This function computes the ridge partial correlations for each variable in the covariate matrix.
#'
#'@param X The input matrix of type `matrix' or `dgCMatrix'. No
#'need to center or scale X as that would be done implicitly.
#'@param y vector containing the response variable.
#'@param lambda the regularization parameter.
#'@param XXt the matrix UU' where U = scale(X), will be computed
#'if not provided. When the matrix X is large, recomputing XXt can be costly,
#'for example if the rpc's for a new lambda value is required. It is thus
#'wise to pass XXt (see function \code{XXt.compute}).
#'@param ncores the number of cores to be used. Default is 1.
#'@param RAM The algorithm will use a maximum of this much additional RAM. 
#'Default is 6GB. Increasing this number, within the limit of the machine,
#'will allow the algorithm to run faster.
#' 
#'@return an object containing the following items: \cr
#'          - rpc: vector of rpc coefficients. \cr
#'          - X: the original matrix of covariates. \cr
#'          - XXt: the matrix UU', where U is the centered-scaled X. \cr
#'          - y: vector of centered and scaled response variable.\cr
#'          - lambda: the original lambda used.
#'
#'@details
#'Consider the linear regression model:
#'\deqn{y = \beta_0 + X\beta + \epsilon,}
#'where \eqn{X} is the \eqn{n \times p} design matrix, \eqn{\beta} is a p-dimensional vector, 
#'and \eqn{\epsilon} is the n-dimensional vector of iid residual errors with mean 0 and variance \eqn{\sigma^2}, \eqn{\epsilon} is independent of \eqn{X}.
#'
#'Assuming the design matrix \eqn{X} is centered and scaled, and letting \eqn{\tilde{Y} = Y - \bar{Y}_n},
#'the (sample) partial correlation between the \eqn{i^{th}} predictor and \eqn{Y} given the remaining predictors is 
#'\eqn{-P_{i+1, 1} / \sqrt{P_{1, 1}P_{i+1, i+1}}}, where \eqn{P} is the \eqn{(p+1) \times (p+1)} joint precision matrix of the response and the covariates.
#'In cases when \eqn{p > n}, \eqn{P} is not invertible. 
#'Hence, we consider the ridge regularized version of \eqn{P}, 
#'given by \eqn{R}, where:
#'\deqn{R := (n-1)\begin{bmatrix}\tilde{Y}^T\tilde{Y} & \tilde{Y}^TX\\ X^T\tilde{Y} & X^TX + \lambda I_p \end{bmatrix}^{-1}.}
#'
#'The ridge partial correlations between \eqn{y} and \eqn{X_{i}} is then defined in terms of elements of \eqn{R} as follows:
#'\deqn{\rho_{i, \lambda} = - \dfrac{r_{i+1,1}}{\sqrt{r_{1,1} r_{i+1,i+1}}}}
#'
#'where \eqn{r_{i,j}} is the \eqn{(i.j)}th entry in \eqn{R}.
#'
#'For variable screening, one may choose the \eqn{K} variables with largest absolute RPC values,
#'for some \eqn{K > 0} (e.g., \eqn{K = n} or \eqn{n/\log p}). Alternatively, the extended BIC
#'criteria may be used. See \code{eBIC} function in this package.
#'
#'\bold{N.B.} Further hyper-threading speed-up can be obtained by
#'loading the \code{MatrixExtra} package.
#'
#'@author Somak Dutta
#'@author An Nguyen
#'@author Maintainer: Somak Dutta <somakd@iastate.edu>
#'
#'
#'@seealso [eBIC()] for model selecting, [XXt.compute()] for computing crossproduct.
#'@examples
#' ## Toy example:
#' n <- 50; p <- 400;
#' trueidx <- 1:3 ## First three variables are important
#' truebeta <- c(4,5,6)
#' X <- matrix(rnorm(n*p), n, p) ## n x p covariate matrix
#' y <- 0.5 + X[,trueidx] %*% truebeta + rnorm(n) ## Response
#' res <- rpc(X,y, lambda = 0.1, ncores = 1)
#' order(abs(res$rpc),decreasing = TRUE)[1:10] # Top 10 variables
#' ## Run another case with the same X and y, but pass XXt to make it faster
#' res2 <- rpc(X,y, lambda = 0.001,XXt = res$XXt , ncores = 1)
#' order(abs(res2$rpc),decreasing = TRUE)[1:10] # Top 10 variables
#' 
#'@export

rpc <- function(X, y, lambda = 1, XXt = NULL, ncores = 1, RAM = 6) {
  
  if(!(class(X)[1] %in% c("matrix","dgCMatrix")))
  {
    stop("Only matrix or dgCMatrix (Matrix) are permitted")
  }
  
  issparse <- class(X)[1] == "dgCMatrix" 
  
  y <- as.numeric(y)
  y <- (y - mean(y))/sd(y)
  if(anyNA(y) || any(is.infinite(y)))
    stop("NA/Infinite values encountered in the response (y),
         possibly after centering and scaling.")
  if(nrow(X) != length(y))
    stop("Length of y and nrow(X) do not match")
  
  
  memsize <- RAM*1024^3 - 8*nrow(X)^2 - 2*8*ncol(X) 
  chunk_size <- floor(memsize / {nrow(X) * 8})
  if(chunk_size == 0) 
    stop("Need more RAM than ",RAM,"GB: increase RAM.")
  chunks <- ceiling(ncol(X)/chunk_size)
  
  
  mX <- colMeans(X);
  
  if(issparse){
    varX <- colMSD_dgc(X, mX); # remember to take sqrt later
  } else {
    varX <- colMSD_matrix(X,mX);
  }
  
  small_varX <- which(varX <= .Machine$double.eps | is.na(varX))
  if(length(small_varX) > 0)
    stop("Following columns have NaN or (near) zero sd: ",
         paste0(small_varX,sep=", "))
  
  if(is.null(XXt)) {
    XXt <- XXt.compute(X = X, meanX = mX, varX = varX, 
                       ncores = ncores,check = FALSE)
  } else  {
    stopifnot(nrow(XXt) == nrow(X))
  }
  
  sdX = sqrt(varX);
  
  if(is.null(XXt)) { # unlikely
    stop("XXt still null")
  }
  
  W <- XXt + lambda * diag(ncol(XXt))
  
  S <- backsolve(chol(W), diag(ncol(W)),transpose = T) # computes W^{-T/2}
  theta <- as.numeric(S %*% y)
  
  Rs <- numeric(ncol(X))
  
  rowSumS <- rowSums(S)

  for(ii in seq_len(chunks)) {
    start <- {ii-1}*chunk_size + 1
    end <- min(ii*chunk_size, ncol(X))
    
    if(chunks == 1)   {
      Z <- X
    } else {
      Z <- X[,start:end,drop=F]
    }
    U <- S %*% Z;
    Rs[start:end] <- compute_rpc(Z = U, colMeansZ = mX[start:end],
                                 rowSumS = rowSumS, y = y,
                                 D = sdX[start:end], theta = theta,
                                 ncores = ncores)
    
    names(Rs) <- colnames(X)
  }
  ret <- list(rpc = Rs, XXt = XXt,X=X,y=y,lambda=lambda)
  class(ret) <- "RPC"
  return(ret)
}



