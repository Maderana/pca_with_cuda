#include "lab3_io.h"
#include "lab3_cuda.h"

#include <stdlib.h>
#include <omp.h>

/*
	Arguments:
		arg1: input filename (consist M, N and D)
		arg2: retention (percentage of information to be retained by PCA)
		arg3: correct dhat filename
*/
void printd_matrix(double *D, int m, int n){
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            printf("%.7f ",*(D+i*n+j));
        }
        printf(";\n");
    }
}

int main(int argc, char const *argv[])
{
	if (argc < 4){
		printf("\nLess Arguments\n");
		return 0;
	}

	if (argc > 4){
		printf("\nTOO many Arguments\n");
		return 0;
	}

	//---------------------------------------------------------------------
	int M;          //no of rows (samples) in input matrix D (input)
	int N;          //no of columns (features) in input matrix D (input)
	double* D;      //1D array of M x N matrix to be reduced (input)
	double* U = nullptr;      //1D array of N x N (or M x M) matrix U (to be computed by SVD)
	double* SIGMA = nullptr;  //1D array of N x M (or M x N) diagonal matrix SIGMA (to be computed by SVD)
                    //SIGMA consists only digonal elements, #elements = N.
                    //So it is vector of N elements
	double* V_T = nullptr;    //1D array of M x M (or N x N) matrix V_T (to be computed by SVD)
	int SIGMAm = 0;     //#rows in SIGMA, read note in lab3_cuda.h (to be computed by SVD)
	int SIGMAn = 0;     //#columns in SIGMA, read note in lab3_cuda.h (to be computed by SVD)
	int K = 0;          //no of coulmns (features) in reduced matrix D_HAT (to be computed by PCA)
	double *D_HAT = nullptr;  //1D array of M x K reduced matrix (to be computed by PCA)
	int retention;  //percentage of information to be retained by PCA (command line input)
	//---------------------------------------------------------------------

	retention = atoi(argv[2]);	//retention = 90 means 90% of information should be retained

	float computation_time;

	/*
		-- Pre-defined function --
		reads matrix and its dimentions from input file and creats array D
	    #elements in D is M * N
        format -
        --------------------------------------------------------------------------------------
        | D[0][0] | D[0][1] | ... | D[0][N-1] | D[1][0] | ... | D[1][N-1] | ... | D[M-1][N-1] |
        --------------------------------------------------------------------------------------
	*/
	read_matrix (argv[1], &M, &N, &D);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	// /*
	// 	*****************************************************
	// 		TODO -- You must implement this function
	// 	*****************************************************
	// */
	SVD_and_PCA(M, N, D, &U, &SIGMA, &V_T, &SIGMAm, &SIGMAn, &D_HAT, &K, retention);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&computation_time, start, stop);

	/*
		--Pre-defined functions --
		checks for correctness of results computed by SVD and PCA
		and outputs the results
	*/
	write_result(M, N, D, U, SIGMA, V_T, SIGMAm, SIGMAn, K, D_HAT, computation_time, argv[3]);

	return 0;
}
