#include <Rcpp.h>
using namespace Rcpp;


#ifdef _OPENMP
#include<omp.h>
#endif

#ifndef _OPENMP
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#endif

//'Compute the ridge partial correlation
 //'@param Z: A pointer to either a sparse matrix (dgCMatrix) or dense matrix (NumericMatrix)
 //'@param colMeansZ: vector of column means of Z
 //'@param S: the transpose(inverse(S')) where S' is the Cholesky decomposition (upper triangular)
 //'of W = XXt + lambda * I, X is the scaled matrix of Z
 //'@param rowSumS: vector containing the row sums of S
 //'@param Y: vector containing the response variable
 //'@param D: vector containing the column variance of Z
 //'@param theta: vector containing the product of S and and Y
 //'@param lamda: the ridge coefficient
 //'@param ncores: the number of cores to be used
 //'@return: a vector containing the ridge partial correlation, one for each column of Z
 // [[Rcpp::export]]
 NumericVector compute_rpc(SEXP Z, 
                           NumericVector colMeansZ,
                           NumericVector rowSumS,
                           NumericVector y, 
                           NumericVector D,
                           NumericVector theta,
                           int ncores = 4) {
   
   if(ncores <= 0)
     ncores = omp_get_max_threads();
   else
     ncores = std::min(omp_get_max_threads(), ncores);
   
   
   int n,p;
   double *U;
   if(Rf_isS4(Z))  {
     S4 Z_S4 = Rcpp::traits::input_parameter< S4 >::type(Z);
     U = REAL(Z_S4.slot("x"));
     IntegerVector Dim = Z_S4.slot("Dim");
     n = Dim[0];
     p = Dim[1];
   } else
   {
     NumericMatrix Z_mat = Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type(Z);
     U = Z_mat.begin();
     n = Z_mat.nrow();
     p = Z_mat.ncol();
   }
   
   NumericVector rpc(p);
   double norm_theta2 = std::inner_product(theta.begin(), theta.end(), theta.begin(), 
                                           (long double) 0.0);
   
   
   // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
   
#ifdef _OPENMP
#pragma omp parallel num_threads(ncores) firstprivate(n,norm_theta2)
#endif
{
  double *ui = new double[n];
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1000) 
#endif
  for (int j = 0; j < p; ++j) {
    // size_t nj = n*j;
    double *Uj = U + n*j;
#ifdef _OPENMP
#pragma omp simd
#endif
    for (int i = 0; i < n; ++i)
      ui[i] = (Uj[i] - rowSumS[i] * colMeansZ[j]) / D[j];     
    
    double norm_ui2 = std::inner_product(ui, ui+n, ui, (long double) 0.0);
    double ui_theta = std::inner_product(theta.begin(), theta.end(), ui, (long double) 0.0);
    
    rpc[j] = ui_theta/sqrt(norm_theta2 + ui_theta*ui_theta - norm_ui2*norm_theta2);
  }
  
  delete[] ui;
  
}
// std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
// Rcpp::Rcout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;

return rpc;
 }
 
 
