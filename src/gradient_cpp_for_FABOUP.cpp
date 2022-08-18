#include <RcppArmadillo.h>   
#include <iostream>

// [[Rcpp::depends(RcppArmadillo)]]

//using namespace Rcpp;
using namespace arma;
using namespace std;

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::Mat<double> calc_Sigma_star_inv(double ni, arma::mat Lambda,
                                      arma::mat Sigma_u, arma::mat Theta,
                                      arma::mat Psi) {
  arma::mat diag_ni = arma::eye(ni, ni);
  arma::mat ones_ni = arma::ones(ni, ni);
  arma::mat Lambda_t = arma::trans(Lambda);
  arma::mat A;
  arma::mat BC;
  arma::mat Sigma_star;
  arma::mat Sigma_star_inv;
  A = arma::kron(diag_ni, Lambda) * Psi * arma::kron(diag_ni, Lambda_t);
  BC = arma::kron(ones_ni, Sigma_u) + arma::kron(diag_ni, Theta);
  
  Sigma_star = A + BC;
  Sigma_star = (Sigma_star + arma::trans(Sigma_star)) / 2;
  Sigma_star_inv = arma::inv(Sigma_star);
  
  return Sigma_star_inv;
}

// [[Rcpp::export]]
arma::Mat<double> calc_Sigma_star(double ni, arma::mat Lambda,
                                  arma::mat Sigma_u, arma::mat Theta,
                                  arma::mat Psi) {
  arma::mat diag_ni = arma::eye(ni, ni);
  arma::mat ones_ni = arma::ones(ni, ni);
  arma::mat Lambda_t = arma::trans(Lambda);
  arma::mat A;
  arma::mat BC;
  arma::mat Sigma_star;
  arma::mat Sigma_star_inv;
  A = arma::kron(diag_ni, Lambda) * Psi * arma::kron(diag_ni, Lambda_t);
  BC = arma::kron(ones_ni, Sigma_u) + arma::kron(diag_ni, Theta);
  
  Sigma_star = A + BC;
  Sigma_star = (Sigma_star + arma::trans(Sigma_star)) / 2;
  // Sigma_star_inv = arma::inv(Sigma_star);
  
  return Sigma_star;
}

//lambda_vec, Yi, Sigma_u, Theta, Psi
// [[Rcpp::export]]
arma::Mat<double> grad_lambda_cpp_slow(double k, double ni, arma::vec Yi, arma::mat Lambda,
                                       arma::mat Psi, arma::mat Sigma_star_inv,
                                       arma::mat nonzero) {
  
  arma::mat diag_ni = arma::eye(ni, ni);
  arma::vec grad_vec = vec(k);
  
  // mat::const_row_col_iterator it     = nonzero.begin_row_col();
  // mat::const_row_col_iterator it_end = nonzero.end_row_col();
  arma::sp_mat temp(nonzero);
  int p = Lambda.n_cols; //NEW
  arma::mat J_k(k, p, arma::fill::zeros);
  arma::mat tJ_k(p, k, arma::fill::zeros);
  arma::mat Ci;
  arma::mat Ci2;
  arma::mat inside;
  arma::mat term2;
  arma::mat term1;
  arma::mat term3;
  int j;
  j = 0;
  
  for (arma::sp_mat::const_iterator it = temp.begin(); it != temp.end(); ++it) {
    // result += *i * (i.row() + 1);
    //J_k(it.row(), it.col(), size(1, 1)) = 1;
    tJ_k(it.col(), it.row(), size(1, 1)) = 1;
    
    Ci = arma::kron(diag_ni, Lambda) * Psi * arma::kron(diag_ni, tJ_k);
    Ci2 = Ci + arma::trans(Ci);
    inside = -(Sigma_star_inv * Ci2 * Sigma_star_inv);
    
    term2 = arma::trans(Yi) * inside * Yi;
    term3 = Sigma_star_inv * Ci2;
    term1 = arma::trace(term3);
    
    grad_vec.subvec(j,j) = term1 + term2;
    j = j + 1;
    
    //J_k(it.row(), it.col(), size(1, 1)) = 0; //reset to 0
    tJ_k(it.col(), it.row(), size(1, 1)) = 0;
  }
  
  // for (; it != it_end; ++it) {
  //   if ((*it) == 1){
  //     J_k(it.row(), it.col(), size(1, 1)) = 1;
  //     
  //     Ci = arma::kron(diag_ni, Lambda) * Psi * arma::kron(diag_ni, arma::trans(J_k));
  //     Ci2 = Ci + arma::trans(Ci);
  //     inside = -(Sigma_star_inv * Ci2 * Sigma_star_inv);
  //     
  //     term2 = arma::trans(Yi) * inside * Yi;
  //     term3 = Sigma_star_inv * Ci2;
  //     term1 = arma::trace(term3);
  //     
  //     grad_vec.subvec(j,j) = term1 + term2;
  //     j = j + 1;
  //     
  //     J_k(it.row(), it.col(), size(1, 1)) = 0; //reset to 0
  //   }
  // }
  
  arma::vec ddlog = vec(k, arma::fill::value(1));
  ddlog.subvec(0,0) = Lambda(0,0, size(1,1));
  arma::vec neghalfvec = vec(k, arma::fill::value(-0.5));
  
  return ((neghalfvec % grad_vec % ddlog).eval()); //element-wise multiplication
}


// [[Rcpp::export]]
arma::Mat<double> grad_sigma_u_cpp_slow(double k, double ni, arma::vec Yi,
                                        arma::mat Sigma_u, arma::mat Sigma_star_inv){
  
  arma::mat ones_ni = arma::ones(ni, ni);
  arma::vec grad_vec = vec(k);
  arma::mat J_k(k, k, arma::fill::zeros);
  arma::mat Ci;
  arma::mat inside;
  arma::mat term2;
  arma::mat term1;
  
  for (int j = 0; j < k; j++) {
    J_k(j, j, size(1,1)) = 2*sqrt(Sigma_u(j,j, size(1,1)));
    Ci = arma::kron(ones_ni, J_k);
    term1 = arma::trace(Sigma_star_inv * Ci);
    inside = -(Sigma_star_inv * Ci * Sigma_star_inv);
    term2 = arma::trans(Yi) * inside * Yi;
    grad_vec.subvec(j,j) = term1 + term2;
    J_k(j, j, size(1,1)) = 0; //reset to 0
  }
  
  arma::vec sqrtSigma_u = sqrt(Sigma_u.diag());
  arma::vec neghalfvec = vec(k, arma::fill::value(-0.5));
  return ((neghalfvec % grad_vec % sqrtSigma_u).eval()); //element-wise multiplication
}

// [[Rcpp::export]]
arma::Mat<double> grad_sigma_e_cpp_slow(double k, double ni, arma::vec Yi,
                                        arma::mat Theta, arma::mat Sigma_star_inv){
  
  arma::mat diag_ni = arma::eye(ni, ni);
  
  arma::vec grad_vec = vec(k);
  arma::mat J_k(k, k, arma::fill::zeros);
  arma::mat Ci;
  arma::mat inside;
  arma::mat term2;
  arma::mat term1;
  
  for (int j = 0; j < k; j++) {
    J_k(j, j, size(1,1)) = 2*sqrt(Theta(j,j, size(1,1)));
    Ci = arma::kron(diag_ni, J_k);
    term1 = arma::trace(Sigma_star_inv * Ci);
    inside = -(Sigma_star_inv * Ci * Sigma_star_inv);
    term2 = arma::trans(Yi) * inside * Yi;
    grad_vec.subvec(j,j) = term1 + term2;
    J_k(j, j, size(1,1)) = 0; //reset to 0
  }
  
  arma::vec sqrtTheta = sqrt(Theta.diag());
  arma::vec neghalfvec = vec(k, arma::fill::value(-0.5));
  return ((neghalfvec % grad_vec % sqrtTheta).eval()); //element-wise multiplication
}


// [[Rcpp::export]]
arma::Mat<double> grad_lambda_cpp(double k, double ni, arma::vec Yi,
                                  arma::mat Lambda, arma::mat Lambda_Psi,
                                  arma::mat nonzero,
                                  arma::mat Sigma_term) {
  
  arma::mat diag_ni = arma::eye(ni, ni);
  arma::vec grad_vec = vec(k);
  
  // mat::const_row_col_iterator it     = nonzero.begin_row_col();
  // mat::const_row_col_iterator it_end = nonzero.end_row_col();
  arma::sp_mat temp(nonzero);
  int p = Lambda.n_cols; //NEW
  //arma::mat J_k(k, p, arma::fill::zeros);
  arma::mat tJ_k(p, k, arma::fill::zeros);
  // arma::mat Ci;
  arma::mat Ci2;
  arma::mat inside;
  arma::mat term2;
  arma::mat term1;
  arma::mat term3;
  int j;
  j = 0;
  
  // Ci = Lambda_kron * Psi;
  
  for (arma::sp_mat::const_iterator it = temp.begin(); it != temp.end(); ++it) {
    // result += *i * (i.row() + 1);
    //J_k(it.row(), it.col(), size(1, 1)) = 1;
    tJ_k(it.col(), it.row(), size(1, 1)) = 1;
    // cout << it.col() << it.row() << endl;
    
    Ci2 = Lambda_Psi * arma::kron(diag_ni, tJ_k);
    Ci2 += arma::trans(Ci2);
    
    term3 = Sigma_term * Ci2;
    term1 = arma::trace(term3);
    
    grad_vec.subvec(j,j) = term1;
    j = j + 1;
    
    //J_k(it.row(), it.col(), size(1, 1)) = 0; //reset to 0
    tJ_k(it.col(), it.row(), size(1, 1)) = 0;
  }
  
  //these three lines are very quick
  arma::vec ddlog = vec(k, arma::fill::value(1));
  ddlog.subvec(0,0) = Lambda(0,0, size(1,1));
  arma::vec neghalfvec = vec(k, arma::fill::value(-0.5));
  
  return ((neghalfvec % grad_vec % ddlog).eval()); //element-wise multiplication
}

// [[Rcpp::export]]
arma::Mat<double> grad_sigma_u_cpp(double k, double ni, arma::vec Yi,
                                   arma::mat Sigma_u, arma::mat Sigma_term){
  
  arma::mat ones_ni = arma::ones(ni, ni);
  arma::vec grad_vec = vec(k);
  arma::mat J_k(k, k, arma::fill::zeros);
  arma::mat Ci;
  arma::mat inside;
  arma::mat term2;
  arma::mat term1;
  
  for (int j = 0; j < k; j++) {
    J_k(j, j, size(1,1)) = 2*sqrt(Sigma_u(j,j, size(1,1)));
    Ci = arma::kron(ones_ni, J_k);
    //inside = (Sigma_star_inv * Ci);
    inside = Sigma_term * Ci;
    //grad_vec.subvec(j,j) = arma::trace(inside * I_SYYt);
    grad_vec.subvec(j,j) = arma::trace(inside);
    
    J_k(j, j, size(1,1)) = 0; //reset to 0
  }
  
  arma::vec sqrtSigma_u = sqrt(Sigma_u.diag());
  arma::vec neghalfvec = vec(k, arma::fill::value(-0.5));
  return ((neghalfvec % grad_vec % sqrtSigma_u).eval()); //element-wise multiplication
}

// [[Rcpp::export]]
arma::Mat<double> grad_sigma_e_cpp(double k, double ni, arma::vec Yi,
                                   arma::mat Theta, arma::mat Sigma_term){
  
  arma::mat diag_ni = arma::eye(ni, ni);
  
  arma::vec grad_vec = vec(k);
  arma::mat J_k(k, k, arma::fill::zeros);
  arma::mat Ci;
  arma::mat inside;
  arma::mat term2;
  arma::mat term1;
  
  for (int j = 0; j < k; j++) {
    J_k(j, j, size(1,1)) = 2*sqrt(Theta(j,j, size(1,1)));
    Ci = arma::kron(diag_ni, J_k);
    //inside = (Sigma_star_inv * Ci);
    inside = Sigma_term * Ci;
    grad_vec.subvec(j,j) = arma::trace(inside);
    //grad_vec.subvec(j,j) = arma::trace(inside * I_SYYt);
    J_k(j, j, size(1,1)) = 0; //reset to 0
  }
  
  arma::vec sqrtTheta = sqrt(Theta.diag());
  arma::vec neghalfvec = vec(k, arma::fill::value(-0.5));
  return ((neghalfvec % grad_vec % sqrtTheta).eval()); //element-wise multiplication
}

// [[Rcpp::export]]
arma::Mat<double> fa_grads_slow(double k, double ni, arma::vec Yi, arma::mat Lambda,
                                arma::mat Sigma_u, arma::mat Theta, arma::mat Psi,
                                arma::mat Sigma_star_inv, arma::mat nonzero){
  
  arma::mat lambda_grad;
  arma::mat sigma_u_grad;
  arma::mat sigma_e_grad;
  
  lambda_grad = grad_lambda_cpp_slow(k, ni, Yi, Lambda, Psi, Sigma_star_inv, nonzero);
  
  sigma_u_grad = grad_sigma_u_cpp_slow(k, ni, Yi, Sigma_u, Sigma_star_inv);
  
  sigma_e_grad = grad_sigma_e_cpp_slow(k, ni, Yi, Theta, Sigma_star_inv);
  
  return(arma::join_cols(lambda_grad, sigma_u_grad, sigma_e_grad));
  
}


// [[Rcpp::export]]
arma::Mat<double> fa_grads(double k, double ni, arma::vec Yi, arma::mat Lambda,
                           arma::mat Sigma_u, arma::mat Theta, arma::mat Psi,
                           arma::mat Sigma_star_inv, arma::mat nonzero,
                           arma::mat I_SYYt, arma::mat Sigma_term){
  
  arma::mat lambda_grad;
  arma::mat sigma_u_grad;
  arma::mat sigma_e_grad;
  
  arma::mat diag_ni = arma::eye(ni, ni);
  arma::mat Lambda_kron = arma::kron(diag_ni, Lambda);
  arma::mat Lambda_Psi = Lambda_kron * Psi;
  
  lambda_grad = grad_lambda_cpp(k, ni, Yi, Lambda, Lambda_Psi,
                                nonzero, Sigma_term);
  
  sigma_u_grad = grad_sigma_u_cpp(k, ni, Yi, Sigma_u, Sigma_term);
  
  sigma_e_grad = grad_sigma_e_cpp(k, ni, Yi, Theta, Sigma_term);
  
  return(arma::join_cols(lambda_grad, sigma_u_grad, sigma_e_grad));
  
}

// [[Rcpp::export]]
arma::Mat<double> sb_grad_lambda_cpp(double k, double ni, arma::vec Yi, arma::mat Lambda,
                                     arma::mat Sigma_u, arma::mat Theta,
                                     arma::mat Psi, arma::mat nonzero, arma::mat grad_id) {
  
  arma::mat diag_ni = arma::eye(ni, ni);
  arma::mat ones_ni = arma::ones(ni, ni);
  arma::mat Lambda_t = arma::trans(Lambda);
  arma::mat A;
  arma::mat BC;
  arma::mat Sigma_star;
  arma::mat Sigma_star_inv;
  A = arma::kron(diag_ni, Lambda) * Psi * arma::kron(diag_ni, Lambda_t);
  BC = arma::kron(ones_ni, Sigma_u) + arma::kron(diag_ni, Theta);
  
  Sigma_star = A + BC;
  Sigma_star = (Sigma_star + arma::trans(Sigma_star)) / 2;
  Sigma_star_inv = arma::inv(Sigma_star);
  
  double n_grad = arma::accu(grad_id);
  arma::vec grad_vec = vec(n_grad);//NEW//vec(k);
  
  mat::const_row_col_iterator it     = grad_id.begin_row_col();
  mat::const_row_col_iterator it_end = grad_id.end_row_col();
  int p = Lambda.n_cols; //NEW
  arma::mat J_k(k, p,  arma::fill::zeros);
  arma::mat Ci;
  arma::mat Ci2;
  arma::mat inside;
  arma::mat term2;
  arma::mat term1;
  arma::mat term3;
  int j;
  j = 0;
  
  for (; it != it_end; ++it) {
    if ((*it) == 1){
      J_k(it.row(), it.col(), size(1, 1)) = 1;
      
      Ci = arma::kron(diag_ni, Lambda) * Psi * arma::kron(diag_ni, arma::trans(J_k));
      Ci2 = Ci + arma::trans(Ci);
      inside = -(Sigma_star_inv * Ci2 * Sigma_star_inv);
      
      term2 = arma::trans(Yi) * inside * Yi;
      term3 = Sigma_star_inv * Ci2;
      term1 = arma::trace(term3);
      
      grad_vec.subvec(j,j) = term1 + term2;
      j = j + 1;
      
      J_k(it.row(), it.col(), size(1, 1)) = 0; //reset to 0
    }
  }
  
  arma::vec ddlog = vec(n_grad,  arma::fill::value(1)); //used to be k, not n_grad
  if (grad_id(0, 0) == 1){
    ddlog.subvec(0,0) = Lambda(0,0, size(1,1));
  }
  arma::vec neghalfvec = vec(n_grad,  arma::fill::value(-0.5)); // used to be k, not n_grad
  
  return (neghalfvec % grad_vec % ddlog); //element-wise multiplication
}

// [[Rcpp::export]]
arma::Mat<double> sb_grad_sigma_u_cpp(double k, double ni, arma::vec Yi, arma::mat Lambda,
                                      arma::mat Sigma_u, arma::mat Theta,
                                      arma::mat Psi, arma::mat nonzero, arma::mat grad_id){
  
  arma::mat diag_ni = arma::eye(ni, ni);
  arma::mat ones_ni = arma::ones(ni, ni);
  arma::mat Lambda_t = arma::trans(Lambda);
  arma::mat A;
  arma::mat BC;
  arma::mat Sigma_star;
  arma::mat Sigma_star_inv;
  A = arma::kron(diag_ni, Lambda) * Psi * arma::kron(diag_ni, Lambda_t);
  BC = arma::kron(ones_ni, Sigma_u) + arma::kron(diag_ni, Theta);
  
  Sigma_star = A + BC;
  Sigma_star = (Sigma_star + arma::trans(Sigma_star)) / 2;
  Sigma_star_inv = arma::inv(Sigma_star);
  
  arma::vec grad_vec = vec(k);
  
  arma::mat J_k(k, k,  arma::fill::zeros);
  arma::mat Ci;
  arma::mat inside;
  arma::mat term2;
  arma::mat term1;
  
  for (int j = 0; j < k; j++) {
    
    if (grad_id(j, j) == 1){
      J_k(j, j, size(1,1)) = 2*sqrt(Sigma_u(j,j, size(1,1)));
      
      Ci = arma::kron(ones_ni, J_k);
      term1 = arma::trace(Sigma_star_inv * Ci);
      inside = -(Sigma_star_inv * Ci * Sigma_star_inv);
      term2 = arma::trans(Yi) * inside * Yi;
      
      grad_vec.subvec(j,j) = term1 + term2;
      
      J_k(j, j, size(1,1)) = 0; //reset to 0
    }
    
  }
  
  arma::vec sqrtSigma_u = sqrt(Sigma_u.diag());
  arma::vec neghalfvec = vec(k,  arma::fill::value(-0.5));
  
  return (neghalfvec % grad_vec % sqrtSigma_u); //element-wise multiplication
}

// [[Rcpp::export]]
arma::Mat<double> sb_grad_sigma_e_cpp(double k, double ni, arma::vec Yi, arma::mat Lambda,
                                      arma::mat Sigma_u, arma::mat Theta,
                                      arma::mat Psi, arma::mat nonzero, arma::mat grad_id){
  
  arma::mat diag_ni = arma::eye(ni, ni);
  arma::mat ones_ni = arma::ones(ni, ni);
  arma::mat Lambda_t = arma::trans(Lambda);
  arma::mat A;
  arma::mat BC;
  arma::mat Sigma_star;
  arma::mat Sigma_star_inv;
  A = arma::kron(diag_ni, Lambda) * Psi * arma::kron(diag_ni, Lambda_t);
  BC = arma::kron(ones_ni, Sigma_u) + arma::kron(diag_ni, Theta);
  
  Sigma_star = A + BC;
  Sigma_star = (Sigma_star + arma::trans(Sigma_star)) / 2;
  Sigma_star_inv = arma::inv(Sigma_star);
  
  arma::vec grad_vec = vec(k);
  
  arma::mat J_k(k, k,  arma::fill::zeros);
  arma::mat Ci;
  arma::mat inside;
  arma::mat term2;
  arma::mat term1;
  
  for (int j = 0; j < k; j++) {
    
    if (grad_id(j, j) == 1){
      
      J_k(j, j, size(1,1)) = 2*sqrt(Theta(j,j, size(1,1)));
      
      Ci = arma::kron(diag_ni, J_k);
      term1 = arma::trace(Sigma_star_inv * Ci);
      inside = -(Sigma_star_inv * Ci * Sigma_star_inv);
      term2 = arma::trans(Yi) * inside * Yi;
      
      grad_vec.subvec(j,j) = term1 + term2;
      
      J_k(j, j, size(1,1)) = 0; //reset to 0
    }
    
  }
  
  arma::vec sqrtTheta = sqrt(Theta.diag());
  arma::vec neghalfvec = vec(k,  arma::fill::value(-0.5));
  
  return (neghalfvec % grad_vec % sqrtTheta); //element-wise multiplication
}



