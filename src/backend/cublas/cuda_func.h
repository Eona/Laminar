/*
 * Eona Studio (c) 2015
 */


#ifndef CUDA_FUNC_H_
#define CUDA_FUNC_H_

typedef float (*op_func_t) (float); // device pointer function

__device__ float sigmoid_func (float x)
{
	return 1.f - x * x;
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
	return 1.f / (1.f + exp(-x));
}


//Static device function
__device__ op_func_t cu_sigmoid_func = sigmoid_func;
__device__ op_func_t cu_sigmoid_gradient_func = sigmoid_gradient_func;
__device__ op_func_t cu_sin_func = sin_func;
__device__ op_func_t cu_cos_func = cos_func;
__device__ op_func_t cu_tanh_func = tanh_func;
__device__ op_func_t cu_tanh_gradient_func = tanh_gradient_func;


__global__ void matOp(float *d, op_func_t op)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	d[tid] = (*op)(d[tid]);
}



#endif
