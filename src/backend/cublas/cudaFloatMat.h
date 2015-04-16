#ifndef CUDA_FLOAT_MAT_H_
#define CUDA_FLOAT_MAT_H_


#include <cuda.h>
#include <curand.h>
#include <vector>


#define GPU_CHECKERROR( err ) (gpuCheckError( err, __FILE__, __LINE__ ))
static void gpuCheckError( cudaError_t err,
                          const char *file,
                          int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
               file, line );
        exit( EXIT_FAILURE );
    }
}



class CudaFloatMat
{
public:
	int DIM_X;
	int DIM_Y;
	int LEN;
	int NUM_DIM;
	vector<int> DIM_ALL;

	float *device_data;
	float *host_data;


	CudaFloatMat(){	}

	CudaFloatMat(float *d, int m, int n):
		device_data(NULL)
	{
		init_dim(m, n); //initialize matrix dimensions
		init_cuda_mem(d); //copy data to device
	}

	CudaFloatMat(int m, int n):
		device_data(NULL)
	{
		init_dim(m, n); //initialize matrix dimensions
		init_cuda_mem();
	}


	CudaFloatMat(vector<int> dim):
		device_data(NULL)
	{
		init_dim(dim); //initialize matrix dimensions
		init_cuda_mem();
	}

	void init_cuda_mem() {
		GPU_CHECKERROR(
		cudaMalloc( (void**)&device_data, LEN * sizeof(float) )
		);
		GPU_CHECKERROR(
		cudaMemset( (void**)&device_data, 0, LEN * sizeof(float) )
		);
	}


	void init_cuda_mem(float *d) {
		GPU_CHECKERROR(
		cudaMalloc( (void**)&device_data, LEN * sizeof(float) )
		);
		GPU_CHECKERROR(
		cudaMemcpy( device_data, d, LEN * sizeof(float), cudaMemcpyHostToDevice )
		);
	}

	void init_dim(int m, int n) {
		DIM_X = m;
		DIM_Y = n;
		LEN = DIM_X * DIM_Y;
		NUM_DIM = 2;
	}

	void init_dim(vector<int> dim){
		DIM_ALL = dim;
		NUM_DIM = dim.size();
		LEN = 1;
		for (int i = 0; i < dim.size(); ++i) {
			LEN *= dim[i];
		}
		if (dim.size > 1) DIM_X = dim[0];
		if (dim.size > 2) DIM_Y = dim[1];

	}

	/*
	 * Copy data to CUDA device
	 */
	float* to_device(float *d){
		GPU_CHECKERROR(
		cudaMemcpy( device_data, d, LEN * sizeof(float), cudaMemcpyHostToDevice )
		);
		return device_data;
	}

	/*
	 * Copy device data to host
	 */

	float* to_host(){
		GPU_CHECKERROR(
		cudaHostAlloc( (void**)&host_data,
					 	 LEN * sizeof(float),
					 	 cudaHostAllocDefault )//allocate host data
		);
		GPU_CHECKERROR(
		cudaMemcpy( host_data, device_data, LEN * sizeof(float), cudaMemcpyDeviceToHost )
		);
		return host_data;
	}

	~CudaFloatMat(){
		if (device_data) cudaFree(device_data);
		if (host_data) free(host_data);
	}

private:

};

#endif
