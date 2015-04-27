/*
 * Eona Studio (c) 2015
 */

#include <cuda.h>
#ifndef CUDA_FUNC_H_
#define CUDA_FUNC_H_

typedef float (*op_func_t) (float); // device pointer function
typedef float (*op_func_dual_t) (float, float); // device pointer function (two arguments)

#define TILE_WIDTH 16 //for shared memory multiplication

__device__ float sigmoid_func (float x)
{
	return 1.f / (1.f + std::exp(-x));
}

__device__ float sigmoid_gradient_func (float x)
{
	return x * (1.f - x);
}

__device__ float sin_func (float x)
{
	return sin(x);
}

__device__ float cos_func (float x)
{
	return cos(x);
}

__device__ float tanh_func (float x)
{
	return tanh(x);
}

__device__ float tanh_gradient_func (float x)
{
	return 1.f - x * x;
}

__device__ float square_loss_func (float x, float y)
{
	float diff = x - y;
	return 0.5f * diff * diff;
}

__device__ float element_mult_func (float x, float y)
{
	return x * y;
}

__device__ float add_func (float x, float y)
{
	return x + y;
}

__device__ float subtract_func (float x, float y)
{
	return x - y;
}

//Static device function (single variable)
__device__ op_func_t cu_sigmoid_func = sigmoid_func;
__device__ op_func_t cu_sigmoid_gradient_func = sigmoid_gradient_func;
__device__ op_func_t cu_sin_func = sin_func;
__device__ op_func_t cu_cos_func = cos_func;
__device__ op_func_t cu_tanh_func = tanh_func;
__device__ op_func_t cu_tanh_gradient_func = tanh_gradient_func;

//Static device function (dual variable)
__device__ op_func_dual_t cu_square_loss_func = square_loss_func;
__device__ op_func_dual_t cu_element_mult_func = element_mult_func;
__device__ op_func_dual_t cu_add_func = add_func;
__device__ op_func_dual_t cu_subtract_func = subtract_func;


//d <- op(d)
__global__ void mat_op_kernel(float *d, int N, op_func_t op)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N) return;
	d[tid] = (*op)(d[tid]);
}

// t <- op(s)
__global__ void mat_op_kernel(float *t, float *s, int N, op_func_t op)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N) return;
	t[tid] = (*op)(s[tid]);
}

// c <- op(a, b)
__global__ void mat_op_kernel(float *c, float *a, float *b, int N, op_func_dual_t op)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N) return;
	c[tid] = (*op)(a[tid], b[tid]);
}

//__global__ void mat_sum_kernel(float *source, float sum, int N)
//{
//	int tid = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tid >= N) return;
//	atomicAdd(&sum, source[tid]);
//}

// Y = a*X
__global__ void mat_scale_kernel(float *target, float alpha, int N)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N) return;
	target[tid] = target[tid]*alpha;
}

// Y = [a]*sizeof(Y)
__global__ void mat_fill_kernel(float *target, float alpha, int N)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N) return;
	target[tid] = alpha;
}

//############naive multiplication implementation##############
__global__ void mat_multNN_kernel(float *C, float *A, float *B, int m, int n, int l, int k)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= m * k) return;
    size_t b_col = (idx / m) * n;//start index in B
    size_t a_row = (idx % m); //start index in A
    float sum = 0;
    for (int i = 0; i < n; ++i) { //a column of A
        sum += B[b_col + i] * A[a_row + i * m];
    }
    C[idx] = sum;
}

__global__ void mat_multNT_kernel(float *C, float *A, float *B, int m, int n, int l, int k)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= m * l) return;
    size_t b_row = (idx / m); //start index in B
    size_t a_row = (idx % m); //start index in A
    float sum = 0;
    for (int i = 0; i < n; ++i) { //a row of A
        sum += B[b_row + i * l] * A[a_row + i * m];
    }
    C[idx] = sum;
}

__global__ void mat_multTN_kernel(float *C, float *A, float *B, int m, int n, int l, int k)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n * k) return;
    size_t b_col = (idx / n) * l; //start index in B
    size_t a_col = (idx % n) * m; //start index in A
    float sum = 0;
    for (int i = 0; i < m; ++i) { //a column of A
        sum += B[b_col + i] * A[a_col + i];
    }
    C[idx] = sum;
}


//############Shared memory implementation, assuming 2D kernel launch##############
__global__ void mat_multNN_shared_kernel(float* C, int CRows, int CCols, float* A, int ARows, int ACols, float* B, int BRows, int BCols) {

	float CValue = 0;

    int Row = blockIdx.y*TILE_WIDTH + threadIdx.y; // which row this thread is on
    int Col = blockIdx.x*TILE_WIDTH + threadIdx.x; // which col this thread is on

    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    for (int k = 0; k < (TILE_WIDTH + ACols - 1)/TILE_WIDTH; k++) {
    	//load the tile
    	if (k*TILE_WIDTH + threadIdx.x < ACols && Row < ARows) {
    		As[threadIdx.y][threadIdx.x] = A[(k*TILE_WIDTH + threadIdx.x)*ARows + Row];
    	}
    	else
    		As[threadIdx.y][threadIdx.x] = 0.0;

    	if (k*TILE_WIDTH + threadIdx.y < BRows && Col < BCols)
    		Bs[threadIdx.y][threadIdx.x] = B[Col*BRows + k*TILE_WIDTH + threadIdx.y];
    	else
    		Bs[threadIdx.y][threadIdx.x] = 0.0;

    	__syncthreads();

    	//compute partial result
    	for (int n = 0; n < TILE_WIDTH; ++n)
    		CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

    	__syncthreads();
    }

    if (Row < CRows && Col < CCols)
    	C[(blockIdx.x * blockDim.x + threadIdx.x)*CRows + blockIdx.y*blockDim.y+threadIdx.y]=CValue;
}
//__global__ void mat_multNT_shared_kernel(float *C, float *A, float *B, int m, int n, int l, int k)
//{
//	int idx = blockDim.x * blockIdx.x + threadIdx.x;
//    if (idx >= m * l) return;
//    size_t b_row = (idx / m); //start index in B
//    size_t a_row = (idx % m); //start index in A
//    float sum = 0;
//    for (int i = 0; i < n; ++i) { //a row of A
//        sum += B[b_row + i * l] * A[a_row + i * m];
//    }
//    C[idx] = sum;
//}
//
//__global__ void mat_multTN_shared_kernel(float *C, float *A, float *B, int m, int n, int l, int k)
//{
//	int idx = blockDim.x * blockIdx.x + threadIdx.x;
//    if (idx >= n * k) return;
//    size_t b_col = (idx / n) * l; //start index in B
//    size_t a_col = (idx % n) * m; //start index in A
//    float sum = 0;
//    for (int i = 0; i < m; ++i) { //a column of A
//        sum += B[b_col + i] * A[a_col + i];
//    }
//    C[idx] = sum;
//}

#endif
