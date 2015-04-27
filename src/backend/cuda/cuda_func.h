/*
 * Eona Studio (c) 2015
 */


#ifndef CUDA_FUNC_H_
#define CUDA_FUNC_H_

typedef float (*op_func_t) (float); // device pointer function
typedef float (*op_func_dual_t) (float, float); // device pointer function (two arguments)



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

__global__ void mat_sum_kernel(float *source, float sum, int N)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N) return;
	atomicAdd(&sum, source[tid]);
}

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


#endif
