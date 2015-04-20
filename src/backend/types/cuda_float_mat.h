#ifndef CUDA_FLOAT_MAT_H_
#define CUDA_FLOAT_MAT_H_

#include <cuda.h>
#include <curand.h>
#include <vector>
#include "gpu_float_mat.h"

#define NUM_THREAD_PER_BLOCK 512;

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


class CudaFloatMat : public GPUFLoatMat
{
public:
    int LDIM; //leading dimension

	dim3 BLOCK_DIM; //block dim
	dim3 GRID_DIM; //grid dim

	float * device_data;

	CudaFloatMat(){
		device_data = NULL;
		host_data = NULL;
	}

	CudaFloatMat(float *d, int m, int n) {
		device_data = NULL;
		host_data = NULL;
		init_dim(m, n); //initialize matrix dimensions
		init_device_mem(d); //copy data to device
	}

	CudaFloatMat(int m, int n) {
		device_data = NULL;
		host_data = NULL;
		init_dim(m, n); //initialize matrix dimensions
		init_device_mem();
	}


	CudaFloatMat(std::vector<int> dim) {
		device_data = NULL;
		host_data = NULL;
		init_dim(dim); //initialize matrix dimensions
		init_device_mem();
	}


	/*
	 * Copy data to CUDA device
	 */
	void to_device(float *d) {
		GPU_CHECKERROR(
		cudaMemcpy( device_data, d, DATA_LEN, cudaMemcpyHostToDevice )
		);
	}

	/*
	 * Copy device data to host
	 */

	void to_host() {
		host_data = (float *)malloc(DATA_LEN);

		GPU_CHECKERROR(
		cudaMemcpy( host_data, device_data, DATA_LEN, cudaMemcpyDeviceToHost )
		);
	}


    cublasOperation_t getOp() {
        return op;
    }

    cublasOperation_t getOp(std::string opcode) {
        if (opcode == "C") {
        	return CUBLAS_OP_C;
        } else if (opcode == "T") {
        	return CUBLAS_OP_T;
        } else {
        	return CUBLAS_OP_N;
        }
    }

    void free_data(){
    	if (device_data) cudaFree(device_data);
		if (host_data) free(host_data);
    }

	~CudaFloatMat(){
		free_data();
	}

private:
    cublasOperation_t op;

	void init_device_mem() {
		op = CUBLAS_OP_N;

		GPU_CHECKERROR(
		cudaMalloc( (void**)&device_data, DATA_LEN )
		);
		GPU_CHECKERROR(
		cudaMemset( (void *)device_data, 0, DATA_LEN )
		);
	}


	void init_device_mem(float *d) {
		op = CUBLAS_OP_N;

		GPU_CHECKERROR(
		cudaMalloc( (void**)&device_data, DATA_LEN )
		);

		GPU_CHECKERROR(
		cudaMemcpy( device_data, d, DATA_LEN, cudaMemcpyHostToDevice )
		);
	}


	void init_dim(int m, int n) {
		GPUFLoatMat::init_dim(m, n);
        LDIM = DIM_ROW;
        BLOCK_DIM.x = NUM_THREAD_PER_BLOCK;
        GRID_DIM.x = ceil(float(LEN)/float(BLOCK_DIM.x));
	}

	void init_dim(std::vector<int> dim){
		GPUFLoatMat::init_dim(dim);
        LDIM = DIM_ROW;
        BLOCK_DIM.x = NUM_THREAD_PER_BLOCK; //number of thread per block
        GRID_DIM.x = ceil(float(LEN)/float(BLOCK_DIM.x)); //number of block
	}

	//not used
	void to_column_major(float *target, float *source){
		int c = 0;
		for (int i = 0; i < DIM_COL; ++i) {
			for (int j = 0; j < DIM_ROW; ++j) {
				target[c] = source[j*DIM_COL+i];
				c++;
			}
		}
	}

	//not used
	void to_row_major(float *target, float *source){
		int c = 0;
		for (int i = 0; i < DIM_ROW; ++i) {
			for (int j = 0; j < DIM_COL; ++j) {
				target[c] = source[j*DIM_ROW+i];
				c++;
			}
		}
	}
};

#endif
