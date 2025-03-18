
#include <Rcpp.h>
using namespace Rcpp;
/* Adapted from http://www.netlib.org/lapack/explore-html/dc/d05/dsyrk_8f_source.html
 * Description:
 * Compute S = (Z - 1b')diag(D^-1)(Z - 1b')' where
 * Z is nxp dense matrix stored in column major format
 * D is p-vector and
 * b is p-vector
 * Only upper triangular (incl. diagonals) of S are computed.
 * This is to be passed to a Cholesky factorization that will compute
 * the upper triangular Cholesky factorization
 * stored in column-major format
 * Author: Somak Dutta
 * Copyright: GPL3
 */

#ifdef _OPENMP
#include<omp.h>
#endif

#ifndef _OPENMP
#define omp_get_max_threads() 1
#endif



//' Compute the cross product of a sparse matrix X
 //' @param mat: input matrix of type dgCMatrix of dimension (n, p)
 //' @param b: vector of length p containing the means of each column in X
 //' @param D: vector of length p containing the variances of each column in X
 //' @param ncores: the number of computing cores to be used
 //' @return XXt as a dense matrix (default matrix format of R)
 // [[Rcpp::export(.getXXt_dense)]]
 NumericMatrix __getXXt_dense__(NumericMatrix mat,NumericVector mX,
                                NumericVector Dvec,int ncores=4,
                                int chunksize = 1000) {
   
   int n = mat.nrow(), p = mat.ncol();
   NumericMatrix Smat(n,n);
   
   if(ncores <= 0)
     ncores = omp_get_max_threads() - 1;
   else
     ncores = std::min(omp_get_max_threads(), ncores);
   
#ifdef _OPENMP
#pragma omp parallel num_threads(ncores) firstprivate(n,p)
#endif
{
  double *Z = mat.begin();
  double *b = mX.begin();
  double *D = Dvec.begin();
  double *S = Smat.begin();
  double bl;
  double *Zl;
  double *Sj;
  double temp;
#ifdef _OPENMP  
#pragma omp for schedule(dynamic,chunksize)
#endif
  for(int j=0;j<n;++j)
  {
    Sj = S + j*n;
    std::fill(Sj,Sj+j+1,0);
    for(int l=0;l<p;++l)
    {
      bl = b[l];
      temp = (Z[j + n*l] - bl)/D[l];
      Sj = S + n*j;
      Zl = Z + l*n;
      for(int i=0;i<=j;++i)
        *(Sj++) += temp*(*(Zl++) - bl);
    }
  }
}

return(Smat);
 }
 
 