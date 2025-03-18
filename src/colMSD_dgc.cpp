#include <Rcpp.h>

using Rcpp::NumericVector;
using Rcpp::IntegerVector;
using Rcpp::S4;

#ifdef _OPENMP
#include<omp.h>
#endif


// Compute the variance of each column of a sparse matrix
// mat: sparse input matrix of type dgCMatrix
// param m: vector containing the means of each column in 'mat'
// return a vector containing the variance of each column of 'mat'
//[[Rcpp::export]]
NumericVector colMSD_dgc(S4 mat,NumericVector m) {
  IntegerVector dims = mat.slot("Dim");
  int ncol = dims[1];
  int nrow = dims[0];

  IntegerVector p = mat.slot("p");
  NumericVector x = mat.slot("x");

  NumericVector y(ncol);
  for(int j=0;j<ncol;++j)
  {
    long double ssq = 0.0;
    double mj = m[j];
    size_t start = p[j];
    size_t end = p[j+1];
#ifdef _OPENMP
#pragma omp simd reduction(+:ssq)
#endif
    for(size_t i=start;i<end;++i)   {
      double temp = x[i] - mj;
      ssq += temp*temp;
    }
    y[j] = (ssq + mj*mj*(nrow - (end - start + 0.0)))/(nrow-1.0);
  }
  return(y);
}
