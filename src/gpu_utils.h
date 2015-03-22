/*
 * Eona Studio (c) 2015
 * Author: Jim Fan
 */
#ifndef gpu_utils_h__
#define gpu_utils_h__

#include "global_utils.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// To cheat the intellisense syntax highligher in VS2013
#ifdef __INTELLISENSE__
void __syncthreads();
void atomicAdd(uint*, uint);
#endif

#define CUDA_CHECK(ans) { gpu_check_error((ans), __FILE__, __LINE__); }
inline void gpu_check_error(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA_CHECK: %s in %s at %d\n", 
				cudaGetErrorString(code), file, line);
		exit(code);
	}
}

template<typename T>
void g_memcpy_vector2device(const vector<T>& h_data, T *d_data)
{
	CUDA_CHECK(
			cudaMemcpy((void *)d_data,
			(void *) &h_data[0],
			h_data.size() * sizeof(T),
			cudaMemcpyHostToDevice)
	);
}

template<typename T>
void g_memcpy_host2device(T *h_data, T *d_data, size_t size)
{
	CUDA_CHECK(
			cudaMemcpy((void *)d_data,
			(void *) h_data,
			size * sizeof(T),
			cudaMemcpyHostToDevice)
	);
}

template<typename T>
void g_memcpy_device2vector(T *d_data, const vector<T>& h_data)
{
	CUDA_CHECK(
		cudaMemcpy((void *)&h_data[0],
			(void *)d_data,
			h_data.size() * sizeof(T),
			cudaMemcpyDeviceToHost)
	);
}

template<typename T>
void g_memcpy_device2host(T *d_data, T *h_data, size_t size)
{
	CUDA_CHECK(
		cudaMemcpy((void *)h_data,
			(void *)d_data,
			size * sizeof(T),
			cudaMemcpyDeviceToHost)
	);
}

template<typename T>
T *g_malloc(size_t size)
{
	T* d_data;
	CUDA_CHECK(
		cudaMalloc((void **)&d_data, size * sizeof(T))
	);
	return d_data;
}

template<typename T>
void g_memset(T *d_data, size_t size, T value = 0)
{
	CUDA_CHECK(cudaMemset((void *)d_data, value, size * sizeof(T)));
}

template<typename T>
void g_free(T *d_data)
{
	CUDA_CHECK(cudaFree(d_data));
}

#endif // gpu_utils_h__
