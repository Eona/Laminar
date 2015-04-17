/*
 * Eona Studio (c) 2015
 */


#ifndef CUDA_FUNC_H_
#define CUDA_FUNC_H_


typedef float (*op_func_t) (float);

__device__ float neg_func (float x)
{
	return x = -x;
}

__device__ op_func_t p_neg_func = neg_func;

__global__ void dummy_kernel(op_func_t op)
{
	printf("Result: %f\n", (*op)(1) );
}



#endif
