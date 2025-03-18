#'@title XXt Computation
#'@description
#'
#'For an input matrix X, This function computes UU' = tcrossprod(U)
#'where U = scale(X) in a memory efficient way.
#'
#'@param X The input matrix of type NumericMatrix or dgCMatrix. No
#'need to center or scale X as that would be done implicitly.
#'@param meanX Vector of column means of X. Will be computed if not provided.
#'@param varX Vector of column variances of X. Will be computed if not provided.
#'@param ncores the number of cores to be used. Default is 1.
#'@param check check for zero variances.
#'@param chunksize When using openmp parallelization, this would be the chunk
#'size under a dynamic scheduling.
#'@return a nrow(X) x nrow(X) dense symmetric matrix.
#'@author Somak Dutta
#'@author Maintainer: Somak Dutta <somakd@iastate.edu>
#'@examples
#' \donttest{
#' library(Matrix)
#' set.seed(123)
#' x <- rsparsematrix(100,100,density = 0.3)
#' norm(XXt.compute(x) - tcrossprod(scale(x))) # (4.783951e-13 on my machine)
#' }
#' @export
XXt.compute <- function(X, meanX = NULL, varX = NULL,ncores = 4, check = TRUE, chunksize = 1000) {
  
  if(!{{classX <- class(X)[1]} %in% c("matrix","dgCMatrix")})
    stop("Only matrix and dgCMatrix are supported")
  
  if(is.null(meanX))
    meanX = colMeans(X)
  if(is.null(varX)) {
    if(classX == "dgCMatrix") {
      varX <- colMSD_dgc(X, meanX);
    } else {
      varX <- colMSD_matrix(X,meanX);
    }
  }
  
  if(check && 
     length(idx <- which(varX < .Machine$double.eps | is.na(varX))) > 0) {
    stop("Some columns have (near) zero or NaN variance: ", 
         paste0(idx,sep=","))
  }
  if(classX == "dgCMatrix") {
    XXt <- .getXXt_sparse(X, D = varX, ncores = ncores, chunksize = chunksize)
  } else {
    XXt <- .getXXt_dense(X, mX = meanX, Dvec = varX, ncores = ncores, chunksize = chunksize)
  }
  return(XXt)
}


