
/*
 * Computes the columnwise variances of a matrix using SIMD whenever available
 */

#include <Rcpp.h>
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

#ifdef _OPENMP
#include<omp.h>
#endif

//[[Rcpp::export]]
NumericVector colMSD_matrix(NumericMatrix X,NumericVector meanX) {
  
  NumericVector v(X.ncol());
  
  for(int j=0;j<X.ncol();++j) {
    long double ssq = 0.0;
    double mx = meanX[j];
    double *Xj = X.begin() + j*X.nrow();
#ifdef _OPENMP
#pragma omp simd reduction(+:ssq)
#endif
    for(int i=0;i<X.rows();++i) {
      double temp = Xj[i] - mx;
      ssq += temp * temp;
    }
    
    v[j] = ssq/(X.nrow()-1);
  }
  
  return(v);
}
