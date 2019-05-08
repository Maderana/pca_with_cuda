#include <iostream>
#include "lab3_io.h"
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

MatrixXd ptr_to_mat(int M, int N, double *ptr)
{
  MatrixXd ret(M,N);
  for (size_t i = 0; i < M; ++i)
    for (size_t j = 0; j < N; ++j)
      ret(i,j) = *(ptr+i*N+j);
  return ret;
}

pair<bool, double> compare_under_tolerance(MatrixXd const& m, MatrixXd const& n, double tolerances[], size_t t_sz)
{
  assert(m.rows() == n.rows());
  assert(m.cols() == n.cols());
  double m_norm = m.norm();
  double n_norm = n.norm();

  double m_norm_abs = fabs(m_norm);
  double n_norm_abs = fabs(n_norm);
  double d = fabs(m_norm_abs-n_norm_abs)*100.0f/n_norm_abs;
  for (size_t i = 0; i < t_sz; ++i) {
    if (d <= tolerances[i]) {
      return make_pair(true, tolerances[i]);
    }
  }
  return make_pair(false, 100.0f);
}

void write_result (int M,
		int N,
		double* D,
		double* U,
		double* SIGMA,
		double* V_T,
		int SIGMAm,
		int SIGMAn,
		int K,
		double* D_HAT,
		double computation_time,
		const char* dhat_fname)
{
	assert(SIGMAm > 0 && SIGMAn > 0);
	double tolerances[] = { 0.001, 0.01, 0.1, 1.0 };

  	MatrixXd Dm = ptr_to_mat(M, N, D);
  	MatrixXd Um = ptr_to_mat(SIGMAm, SIGMAm, U);
  	MatrixXd Vm = ptr_to_mat(SIGMAn, SIGMAn, V_T);
  	MatrixXd sigma = MatrixXd::Zero(SIGMAm,SIGMAn);
  	for (size_t i = 0; i < min(SIGMAm,SIGMAn); ++i)
    		sigma(i, i) = *(SIGMA+i);

  	MatrixXd Dm_n = Um*sigma*Vm;
  	if (SIGMAm == N && SIGMAn == M) {
  		Dm_n.transposeInPlace();
  	}

  	auto p = compare_under_tolerance(Dm, Dm_n, tolerances, sizeof(tolerances)/sizeof(double));
  	bool d_equal = p.first;
  	double d_tolerance = p.second;

  	double* dhat_correct = nullptr;
  	int Mh, Nh;
  	read_matrix (dhat_fname, &Mh, &Nh, &dhat_correct);
  	assert(Mh == M);
  	bool K_equal = (Nh == K);
  	bool d_h_equal = false;
  	double d_h_tolerance = 100.0f;
  	if (K_equal) {
    		MatrixXd Dhatm = ptr_to_mat(M, K, D_HAT);
    		MatrixXd Dhatm_correct = ptr_to_mat(Mh, Nh, dhat_correct);
    		auto p = compare_under_tolerance(Dhatm_correct, Dhatm, tolerances, sizeof(tolerances)/sizeof(double));
    		d_h_equal = p.first;
    		d_h_tolerance = p.second;
  	}

#define FLAG(f) ((f) ? 'T' : 'F')
  	cout << FLAG(d_equal) << ", " << d_tolerance << ", " << FLAG(K_equal) << ", " << K << ", " << FLAG(d_h_equal) << ", " << d_h_tolerance <<  ", " << computation_time << endl;
}
