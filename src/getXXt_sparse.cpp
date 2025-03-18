/***
 * This program computes the matrix XDX' and stores it into a dense matrix. Where X = (Z - 1 Zbar')
 * Inputs:
 * Z: a sparse matrix of class double in dgCMatrix format
 * D: a vector of class double, same length as the ncol(Z)
 * 
 * Returns a symmetric matrix of class matrix.
 * Author: Somak Dutta
 * Copyright: GPL3
 */


#include <Rcpp.h>
using namespace Rcpp;

#ifdef _OPENMP
#include<omp.h>
#endif





#ifndef _OPENMP
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#endif



//' Compute the cross product of a sparse matrix X
 //' @param mat: input matrix of type dgCMatrix of dimension (n, p)
 //' @param D: vector of length p containing the variance of each column in X
 //' @param ncores: the number of computing cores to be used
 //' @return XXt as a dense matrix (default matrix format of R)
 // [[Rcpp::export(.getXXt_sparse)]]
 NumericMatrix __getXXt_sparse__(S4 mat, NumericVector D,int ncores = 4,
                                 int chunksize = 1000) {
   
   if(ncores <= 0)
     ncores = omp_get_max_threads() - 1;
   else
     ncores = std::min(omp_get_max_threads(), ncores);
   
   // Get the members from the sparse matrix mat
   
   IntegerVector Z_i = mat.slot("i");
   IntegerVector Z_p = mat.slot("p");
   NumericVector Z_x = mat.slot("x");
   IntegerVector Dim = mat.slot("Dim");
   
   int n = Dim[0];
   int p = Dim[1];
   
   if(D.length() != p)
     stop("Length of D must be same as the number of columns in Z");
   
   
   NumericMatrix Stotal(n,n);
   
#ifdef _OPENMP
#pragma omp parallel num_threads(ncores) firstprivate(n)
#endif
{
  // Create a version of S = ZDZ' matrix for each thread to avoid memory racing
  double *S;
  if(omp_get_thread_num() == 0)
    S = Stotal.begin();
  else   {
    S = new double[n*n];
    std::fill(S,S+n*n,0.0);
  }
  
  
#ifdef _OPENMP  
#pragma omp for schedule(dynamic,chunksize)
#endif
  for(int k = 0; k < p; ++k)
  {
    // Note: A chunk of contiguous columns are assigned to each thread
    double Dk_inv = 1.0/D[k];
    for(int jj=Z_p[k]; jj < Z_p[k+1]; ++jj)
    {
      int j = Z_i[jj];
      double temp = Z_x[jj]*Dk_inv;
      for(int ii=Z_p[k]; ii <= jj;++ii)
      {
        int i = Z_i[ii];
        S[i+n*j] += Z_x[ii]*temp; //  = Z_x[ii]*Z_x[jj]/D[k]
      }
    }
  }
  // Next, combine the results from different threads into one.
#ifdef _OPENMP
#pragma omp critical 
#endif  
  if(omp_get_thread_num() > 0)
  {
    std::transform (Stotal.begin(), Stotal.end(), S, Stotal.begin(), std::plus<double>());
    delete[] S;
  }
}


  for(int j=0;j < n;++j)
    for(int i=j+1; i<n; ++i)
      Stotal(i,j) = Stotal(j,i);

  NumericVector b = colMeans(Stotal);
  double c = mean(b);

  for(int j=0;j<n;++j)
    for(int i=0;i<n;++i)
      Stotal[n*j + i] += c - b[i] - b[j];


  return(Stotal);
}

 