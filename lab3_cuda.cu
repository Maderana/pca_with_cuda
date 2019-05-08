// #include <stdio.h>
// #include <stdlib.h>
#include "lab3_cuda.h"
#include <bits/stdc++.h>
// #include vector
#include <math.h>
#include <cuda.h>
#include <cmath>
 
// CUDA kernel. Each thread takes care of one element of c
// __global__ void vecAdd(double *a, double *b, double *c, int n)
// {
//     // Get our global thread ID
//     int id = blockIdx.x*blockDim.x+threadIdx.x;
//     // printf("this is in GPU\n");
 
//     // Make sure we do not go out of bounds
//     if (id < n){
//         c[id] = a[id] + b[id];
//         // printf("%lf\n",c[id]);
//     }
        
// }
#define TILE_DIM 16

#define CONVRG 1e-3
#define cs_CONVRG 1e-3

using namespace std;

// void read_matrix (const char* input_filename, int* M, int* N, double** D){
// 	FILE *fin = fopen(input_filename, "r");

// 	fscanf(fin, "%d%d", M, N);
	
// 	int num_elements = (*M) * (*N);
// 	*D = (double*) malloc(sizeof(double)*(num_elements));
	
// 	for (int i = 0; i < num_elements; i++){
// 		fscanf(fin, "%lf", (*D + i));
// 	}
// 	fclose(fin);
// }
// void printd_matrix(double *D, int m, int n){
//     for(int i=0; i<m; i++){
//         for(int j=0; j<n; j++){
//             printf("%.7f ",*(D+i*n+j));
//         }
//         printf(";\n");
//     }
// }
// void printi_matrix(int *D, int m, int n){
//     for(int i=0; i<m; i++){
//         for(int j=0; j<n; j++){
//             printf("%d ",*(D+i*n+j));
//         }
//         printf(";");
//     }
// }


__global__ void chess_pairing(int *p, int *q, int N)
{
    // Get our global thread ID
    int i = blockIdx.x; // i is iteration index
    int Tid = threadIdx.x;

    int Index_1;
    int Index_2;
    Index_1 = (Tid+i)%(N-1);
    if(Tid != 0) Index_2 = ((N-Tid)+i-1)%(N-1);
    else Index_2 = N-1;
    p[i*(N/2)+Tid] = min(Index_1, Index_2);
    q[i*(N/2)+Tid] = max(Index_1, Index_2);
    // printf("this is in GPU\n");

}
__global__ void find_cos_sin(double *D, int *d_p, int *d_q,
                        int N, int iteration, double *d_cos,double *d_sin)
{   
    // __shared__ double CONVRG = 1e-3
    // Get our global thread ID
    // int block_id = blockIdx.x; // i is iteration index
    int id = threadIdx.x;

    int k = d_p[iteration*N/2 + id]; //row
    int l = d_q[iteration*N/2 + id]; //column

    double p = D[k*N + l];
    
    // double y = (e[l] - e[k]) / 2.0;
    double y = (D[l*N+l] - D[k*N+k]) / 2.0;
    double d = fabs(y) + sqrt(p*p + y*y);
    double r = sqrt(p*p + d*d);
    double c;
    double s;
    if((fabs(p)<cs_CONVRG) && (fabs(d)<cs_CONVRG)){
        c = 1;
        s = 0;
    }
    else{
        c = d / r;
        s = (fabs(y)/y)*(p / r);
    }
    
    
    // double t = (p*p) / d;

    // if (y < 0.0) { s = -s; t = -t; }
    d_cos[id] = c;
    d_sin[id] = s;
    // printf("this is in GPU\n");

}

__global__ void init_identity(double *d, int N)
{
// Get our global thread ID
int block_id = blockIdx.x; // i is iteration index
int Tid = threadIdx.x;
if(Tid == block_id){
    d[block_id*N + Tid] = 1;
}
else{
    d[block_id*N + Tid] = 0;
}
}

__global__ void update_columns(double *da, double *db, double *d_cos, double *d_sin, int *d_p, int *d_q,int iteration , int N)
{
// Get our global thread ID
int block_id = blockIdx.x; // i is iteration index
int Tid = threadIdx.x;

int p = d_p[iteration*N/2 + block_id];
int q = d_q[iteration*N/2 + block_id];

double co = d_cos[block_id];
double si = d_sin[block_id];

double val_p = da[p*N+Tid];
double val_q = da[q*N+Tid];

db[p*N+Tid] = co*val_p - si*val_q;
db[q*N+Tid] = si*val_p + co*val_q;
}

__global__ void update_rows(double *da, double *db, double *d_cos, double *d_sin, int *d_p, int *d_q,int iteration , int N)
{
// Get our global thread ID
int block_id = blockIdx.x; // i is iteration index
int Tid = threadIdx.x;

int p = d_p[iteration*N/2 + block_id];
int q = d_q[iteration*N/2 + block_id];

double co = d_cos[block_id];
double si = d_sin[block_id];

double val_p = da[p*N+Tid];
double val_q = da[q*N+Tid];

db[Tid*N + p] = co*val_p - si*val_q;
db[Tid*N + q] = co*val_q + si*val_p;

}

// Db is new, Da is old one
bool check_convergence(double* Da, double* Db, int N){
	double sq_diff = 0;
	for(int i=0; i<N; i++){
		double diff = Db[i*N + i] - Da[i*N + i];
		if(diff<0) sq_diff -= diff;
		else sq_diff += diff;
    
    }
    // printf("check_convergence : %.5f\n",sq_diff);

	return (sq_diff < CONVRG);
}
////copy array from dfrom to dto
__global__ void copy_darray(double *dfrom, double*dto){

    int id = blockIdx.x*blockDim.x + threadIdx.x;
    dto[id] = dfrom[id];
}

void copy_harray(double* Dfrom, double* Dto, int N){
    for(int i=0;i<N; i++){
        for(int j=0;j<N;j++){
            Dto[i*N+j] = Dfrom[i*N+j];
        }
    }
}

void host_transpose(double *d, double *dt, int N, int M)
{
    for(int i=0; i<M; i++){
		for(int j=0; j<N; j++){
			dt[j*M+i] = d[N*i + j];
		}
	}
}

__global__ void device_transpose(double *d, double *dt, int width, int height)
{
    __shared__ double block[TILE_DIM][TILE_DIM+1];

    int id_x = blockIdx.x * TILE_DIM + threadIdx.x;
    int id_y = blockIdx.y * TILE_DIM + threadIdx.y;

    if((id_x < width) && (id_y < height))
    {
        int index_d = id_y * width + id_x;
        block[threadIdx.y][threadIdx.x] = d[index_d];
    }

    __syncthreads();

    id_x = blockIdx.y * TILE_DIM + threadIdx.x;
    id_y = blockIdx.x * TILE_DIM + threadIdx.y;
    if((id_x < height) && (id_y < width))
    {
        int index_dt = id_y * height + id_x;
        dt[index_dt] = block[threadIdx.x][blockIdx.y];
    }
}

// void print_device_matrix(double *Mat, int n, int m){
//     size_t msize = (n*m)*sizeof(double);
//     double *hm = (double*)malloc(sizeof(double)*msize);
//     cudaMemcpy(hm, Mat, msize, cudaMemcpyDeviceToHost);
//     // printf("D matrix : \n");
//     printd_matrix(hm, n, m);
//     free(hm);
// }

// __global__ void matrix_multiply(double *a, double*b, double*c, int m, int n)
// {

// }

__global__ void device_matrix_multiply(double* A, double* B, double* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols)
{
    __shared__ double As[TILE_DIM][TILE_DIM];
    __shared__ double Bs[TILE_DIM][TILE_DIM];

    double CValue = 0;
    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;
  
    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

    if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows) As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
    else As[threadIdx.y][threadIdx.x] = 0.0;

    if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)  Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
    else Bs[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    for (int n = 0; n < TILE_DIM; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

    __syncthreads();
  
    }
    if (Row < CRows && Col < CCols) C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;

}


void jacobi(double* D, int n, double** evalue_out, double** evector_out){
    size_t num_size = (n*n)*sizeof(double);
    double *d_a;
    double *d_b;
    double *E;
    double *E_b;
    cudaMalloc((void **)&d_a, num_size);
    cudaMalloc((void **)&d_b, num_size);
    
    cudaMalloc((void **)&E, num_size);
    init_identity<<<n,n>>>(E,n);
    cudaMalloc((void **)&E_b, num_size);
    init_identity<<<n,n>>>(E_b,n);

    cudaMemcpy(d_a, D, num_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_b, D, num_size, cudaMemcpyHostToDevice);
    
    // float *d_c;
    int st = n;
    size_t cuda_pq = (st-1)*(st/2)*sizeof(int);
    // int *h_p = (int *)malloc(cuda_pq);
    // int *h_q = (int *)malloc(cuda_pq);
    int *d_p;
    int *d_q;
    
    cudaMalloc((void **)&d_p, cuda_pq);
    cudaMalloc((void **)&d_q, cuda_pq);
    chess_pairing<<<n-1,n/2>>>(d_p, d_q, n);

    size_t num_cs = (n/2)*sizeof(double);
    double *d_cos;
    double *d_sin;
    cudaMalloc((void **)&d_cos, num_cs);
    cudaMalloc((void **)&d_sin, num_cs);

    double *Dtemp = (double*)malloc(sizeof(double)*num_size);
    double *evalue = (double*)malloc(sizeof(double)*num_size);
    copy_harray(D, evalue, n);
    double *evector = (double*)malloc(sizeof(double)*num_size);
    cudaMemcpy(Dtemp, d_a, num_size, cudaMemcpyDeviceToHost);
    // printf("D matrix : \n"printf("evalue from jacobi: \n");;
    // printf("evalue matrix : \n");
    // printd_matrix(evalue, n, n);
    

    double* d_c_check = (double*)malloc(sizeof(double)*num_cs);
    double* d_s_check = (double*)malloc(sizeof(double)*num_cs);
    cudaDeviceSynchronize();
    bool not_converge = false;
    // int iter = 0;
    while(!not_converge){
        for(int i=0; i<n-1; i++){

            find_cos_sin<<<1,n/2>>>(d_a, d_p, d_q, n,i, d_cos, d_sin);
            cudaMemcpy(d_c_check, d_cos, num_cs, cudaMemcpyDeviceToHost);
            cudaMemcpy(d_s_check, d_sin, num_cs, cudaMemcpyDeviceToHost);
            
            cudaDeviceSynchronize();

            update_rows<<< n/2, n >>>(d_a, d_b, d_cos, d_sin, d_p, d_q, i, n);
            cudaDeviceSynchronize();
            // update_rows<<< n/2, n >>>(d_a, E, d_cos, d_sin, d_p, d_q, i, n);
            
            update_columns<<< n/2, n >>>(d_b, d_a, d_cos, d_sin, d_p, d_q, i, n);
            cudaDeviceSynchronize();
            update_columns<<< n/2, n >>>(E, E_b, d_cos, d_sin, d_p, d_q, i, n);
            cudaDeviceSynchronize();
            copy_darray<<<n, n>>>(E_b, E);
            cudaDeviceSynchronize();
            // printf("matrix d_a in %d iteration \n",i);
            // print_device_matrix(d_a, n,n);
            // printf("matrix E in %d iteration \n",i);
            // print_device_matrix(E, n,n);
            
            
        }
        cudaMemcpy(Dtemp, d_a, num_size, cudaMemcpyDeviceToHost);
        // cudaMemcpy(evector, E, num_size, cudaMemcpyDeviceToHost);

        // printf("Dtemp matrix after %d iteration : \n",iter);
        // iter++;
        // printd_matrix(Dtemp, n, n);
        not_converge = check_convergence(evalue, Dtemp, n);

        //  exchange of pointer among new and old values of pointers for eigenvalues
        double* temp = evalue;
        evalue = Dtemp;
        Dtemp = temp;

    }
    cudaMemcpy(evector, E_b, num_size, cudaMemcpyDeviceToHost);
    // printf("finally eigenvalues : \n");
    // printd_matrix(Dtemp, n, n);
    // printf("evector finally : \n");
    // printd_matrix(evector, n, n);
    *evalue_out = evalue;
    *evector_out = evector;
    
}

void host_matrix_multiply(int na, int ma,double *a,
                         int nb, int mb, double *b,
                         int nc, int mc, double *c)
{   size_t a_size = (na*ma)*sizeof(double);
    size_t b_size = (nb*mb)*sizeof(double);
    size_t c_size = (nc*mc)*sizeof(double);
    
    double *d_a;
    cudaMalloc((void **)&d_a, a_size);
    double *d_b;
    cudaMalloc((void **)&d_b, b_size);
    double *d_c;
    cudaMalloc((void **)&d_c, c_size);
    cudaMemcpy(d_a,a, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b, b_size, cudaMemcpyHostToDevice);
    
    dim3 dimGrid1((nc + TILE_DIM - 1) / TILE_DIM, (mc + TILE_DIM - 1) / TILE_DIM);
    dim3 dimBlock1(TILE_DIM, TILE_DIM);

    device_matrix_multiply<<<dimGrid1,dimBlock1>>>(d_a,d_b, d_c, ma,na,mb,nb,mc,nc);
    cudaMemcpy(c, d_c, c_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void give_sigma_U_and_VT(double *d,double *dT,double *evalue, double *evector, double *sigma,double *U, double *VT, int M, int N)
{
    vector<pair<double, int>> evals;
    for(int i=0;i<N;i++){
        evals.push_back({evalue[i*N + i],i});
    }
    sort(evals.rbegin(), evals.rend());
    // printf("evals vector : \n");
    for(int i=0;i<N;i++){
        // printf("%lf %d : ",evals[i].first, evals[i].second);
        evals.push_back({evalue[i*N + i],i});
    }
    // printf("\n");
    
    double *sigma_inv = (double *)calloc(N*M, sizeof(double));
    for(int i=0;i<N;i++){
        // int t = evals[i].second;
        sigma[i] = (double)sqrt(evals[i].first);
        sigma_inv[i*N+i] = 1/(double)sqrt(evals[i].first);
    }

    for(int i=0; i<N;i++){
        int t = evals[i].second;
        for(int j=0;j<N;j++){
            U[j*N + i] = evector[j*N+t];
        }
    }

    double *UT = (double *)malloc(N*N*sizeof(double));
    host_transpose(U, UT, N, N);
    double *v_temp = (double *)malloc(M*N*sizeof(double));
    host_matrix_multiply(N, M, sigma_inv, N, N, UT, N, M, v_temp);
    // *VT = (double *)malloc(M*M*sizeof(double));
    host_matrix_multiply( N, M, v_temp, M, N, dT, M, M, VT);

}

void do_pca(double *dT, double **D_hat_out, double **W_out, double *U, double *VT, double *sigma, int M, int N, double info_required,int *k)
{   
    // *k=0;
    double eval_sum=0;
    for(int i=0;i<N;i++){
        eval_sum += (sigma[i])*(sigma[i]);
    }
    // printf("sum calculated : %lf\n",eval_sum);
    double eval_sum_check=0;
    int hk = 0;
    for(int i=0;i<N;i++){
        hk+=1;
        eval_sum_check += (sigma[i])*(sigma[i]);
        if(100*eval_sum_check/eval_sum>info_required){
            break;
        }
    }

    // printf("value of k : %d\n", hk);
    // int hk = **k;
    double *W = (double*)malloc((hk)*N*sizeof(double));
    for(int i=0;i<N;i++){
        for(int j=0;j<hk;j++){
            W[i*hk+j] = U[i*N+j];
        }
    }
    // printf("W calculated: \n");

    double *D_hat = (double*)malloc((hk)*M*sizeof(double));
    // int factor = *k;

    host_matrix_multiply( N,M,dT,hk,N,W, hk,M,D_hat);
    // printf("finally returning \n");
    // int *k_out ;
    // *(k_out) = hk;
    *k = hk;
    // printf("k okay\n");
    *W_out = (double*)malloc((hk)*N*sizeof(double));
    *W_out = W ;
    *D_hat_out = (double*)malloc((hk)*M*sizeof(double));
    *D_hat_out = D_hat;
    // printf("finally returbed \n");
}

void go_for_svd(double *d, double *dT, int M, int N, double * sigma, double * U, double * VT, int* k,int retention,double **D_HAT)
{
    double* evalue;
    double* evector;
    evalue = (double*) malloc(sizeof(double)*(N*N));
    evector = (double*) malloc(sizeof(double)*(N*N));
    
    double* dTd;
    dTd = (double*) malloc(sizeof(double)*(N*N));
    // calculate dTd
    host_matrix_multiply(M,N,dT,N,M,d,N,N,dTd);
    // printf("multiplied matrix : \n");
    // printd_matrix(dTd,N,N);
    jacobi(dTd, N,&evalue, &evector);

    double* evtemp;
    evtemp = (double*) malloc(sizeof(double)*(N*N));
    host_transpose(evector,evtemp,N,N);
    evector = evtemp;
    

    // printf("going for sigma,U and VT\n");
    give_sigma_U_and_VT(d,dT,evalue, evector, sigma,U, VT,M,N);
    // printf("sigma matrix: \n");
    // printd_matrix(sigma, M, N);
    // printf("U matrix: \n");
    // printd_matrix(U, N, N);
    // printf("sigma matrix: \n");
    // printd_matrix(sigma, M, N);
    
    // double* D_hat;
    double* W;
    // double info_required = 80.0;
    

    // printf("going for pca\n");
    do_pca(d, D_HAT, &W, U, VT, sigma, M, N, retention,k);
    int my_k = *k;
    // printf("W matrix : \n");
    // printd_matrix(W,N,my_k);
    // printf("D_hat matrix : \n");
    // printd_matrix(*D_HAT,my_k,M);


}
void SVD_and_PCA (int M, 
    int N, 
    double* D, 
    double** U, 
    double** SIGMA, 
    double** V_T, 
    int* SIGMAm,
    int* SIGMAn, 
    double** D_HAT, 
    int *K,
    int retention) {
    // write your code here
    *SIGMAm = N;
    *SIGMAn = M;
    
    *SIGMA = (double*) calloc(N*1, sizeof(double)); 
    *U = (double*) malloc(sizeof(double)*(N*N));
    *V_T = (double*) malloc(sizeof(double)*(M*M));
    int *k = (int *)malloc(1*sizeof(int));

    double* dT;
    dT = (double*) malloc(sizeof(double)*(N*M));
    
    
    host_transpose(D, dT, N, M);
    // printf("d matrix : \n");
    // printd_matrix(d, m,n);
    // printf("dT matrix : \n");
    // printd_matrix(dT, n,m);
    double* dTd;
    dTd = (double*) malloc(sizeof(double)*(N*N));
    // host_matrix_multiply(int na, int ma,double *a,
    //     int nb, int mb, double *b,
    //     int nc, int mc, double *c)
    host_matrix_multiply(M,N,dT,N,M,D,N,N,dTd);
    // printf("multiplied matrix : \n");
    // printd_matrix(dTd,n,n);
    // jacobi(dTd, n,n,&evalue, &evector);

    
    go_for_svd(D,dT,M,N, *SIGMA, *U, *V_T,k,retention,D_HAT);
    // go_for_svd()
    *K = *k;

}

// int main( int argc, char* argv[] )
// {
//     // Size of vectors
//     int n;
//     int m;
//     char* input_file = "./data/iris_stndardized";
    
//     // Host input vectors
//     double *d;
//     read_matrix(input_file, &m, &n, &d);
//     printf("value of n and m ; %d, %d\n",n,m);
//     // double* evalue;
//     // double* evector;
//     // evalue = (double*) malloc(sizeof(double)*(n*m));
//     // evector = (double*) malloc(sizeof(double)*(n*m));
//     double* dT;
//     dT = (double*) malloc(sizeof(double)*(n*m));
//     host_transpose(d, dT, n, m);
//     // printf("d matrix : \n");
//     // printd_matrix(d, m,n);
//     // printf("dT matrix : \n");
//     // printd_matrix(dT, n,m);
//     double* dTd;
//     dTd = (double*) malloc(sizeof(double)*(n*n));
//     // host_matrix_multiply(int na, int ma,double *a,
//     //     int nb, int mb, double *b,
//     //     int nc, int mc, double *c)
//     host_matrix_multiply(m,n,dT,n,m,d,n,n,dTd);
//     // printf("multiplied matrix : \n");
//     // printd_matrix(dTd,n,n);
//     // jacobi(dTd, n,n,&evalue, &evector);

//     double* SIGMA; //N*M matrix
//     // sigma = (double*) calloc(n*m, sizeof(double)); 
//     double* U;
//     // U = (double*) malloc(sizeof(double)*(n*n));
//     double* VT;
//     // VT = (double*) malloc(sizeof(double)*(m*m));
//     // go_for_svd(d,dT,m,n, sigma, U, VT);
//     int * SIGMAn = (int *)malloc(sizeof(int));
//     int * SIGMAm = (int *)malloc(sizeof(int));
//     double *D_HAT;
//     int *K;
//     int retention = 80;
//     SVD_and_PCA (m, n, d, &U, &SIGMA, &VT, SIGMAm,SIGMAn, &D_HAT, K, retention);
//     // printd_matrix(evalue,n,n);
//     // printd_matrix(evector,n,n);
    
//     // size_t d_size = (n*n)*sizeof(double);
//     // double *a = (double*)malloc(d_size);
//     // double *b = (double*)malloc(d_size);
//     // double *c = (double*)malloc(d_size);

//     // for(int i=0;i<m;i++){
//     //     for(int j=0;j<n;j++){
//     //         a[i*n+j] = D[i*n+j];
//     //         b[i*n+j] = D[i*n+j];
//     //     }
//     // }
//     // host_matrix_multiply(n,n,a,n,n,b,n,n,c);
//     // printd_matrix(c,n,n);

//     return 0;
// }