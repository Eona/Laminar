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


class CudaFloatMat : public GPUFloatMat
{
public:
    int LDIM; //leading dimension

	dim3 BLOCK_DIM; //block dim
	dim3 GRID_DIM; //grid dim

	float * device_data;

	CudaFloatMat(){
		init_dim(1,1);
		op = CUBLAS_OP_N;
		device_data = NULL;
		host_data = NULL;
		device_data_initialized = false;
	}

	CudaFloatMat(float *d, int m, int n) {
		device_data = NULL;
		host_data = NULL;
		op = CUBLAS_OP_N;
		init_dim(m, n); //initialize matrix dimensions
		init_device_mem(d); //copy data to device
		device_data_initialized = false;
	}

	CudaFloatMat(int m, int n) {
		device_data = NULL;
		host_data = NULL;
		op = CUBLAS_OP_N;
		init_dim(m, n); //initialize matrix dimensions
		init_device_mem();
		device_data_initialized = false;
	}

	CudaFloatMat(std::vector<int> dim) {
		device_data = NULL;
		host_data = NULL;
		op = CUBLAS_OP_N;
		init_dim(dim); //initialize matrix dimensions
		init_device_mem();
		device_data_initialized = false;
	}


	void reset(int m, int n) {
		init_dim(m, n); //initialize matrix dimensions
		init_device_mem();
	}

	void reset(vector<int> dim) {
		init_dim(dim); //initialize matrix dimensions
		init_device_mem();
	}

	void reset(float * d, int m, int n) {
		init_dim(m, n); //initialize matrix dimensions
		init_device_mem(d);
	}

	/*
	 * Copy data to CUDA device
	 */
	void to_device(float *d) {
		GPU_CHECKERROR(
		cudaMemcpy( device_data, d, MEM_SIZE, cudaMemcpyHostToDevice )
		);
	}

	void copy_to_device(float *device_d) {
		GPU_CHECKERROR(
		cudaMemcpy( device_d, device_data, MEM_SIZE, cudaMemcpyDeviceToDevice )
		);
	}

	/*
	 * Copy device data to host
	 */

	void to_host() {
		if (!host_data) host_data = (float *)malloc(MEM_SIZE);

		GPU_CHECKERROR(
		cudaMemcpy( host_data, device_data, MEM_SIZE, cudaMemcpyDeviceToHost )
		);
	}

	void to_host(float * d) {
		GPU_CHECKERROR(
		cudaMemcpy( d, device_data, MEM_SIZE, cudaMemcpyDeviceToHost )
		);
	}


	void take_at(float * d, size_t offset, size_t num_float) {
		auto r = alloc_vector();
		to_host(&r[0]);
		for (int i = 0; i < num_float; ++i) {
			d[i] = r[i+offset];
		}
	}

	void fill_rand(int seed) {
		auto r = alloc_vector();
		srand (seed);
		for (int i = 0; i < LEN; ++i) {
			r[i] = 0.16 * ((double) rand() / (RAND_MAX)) - 0.08;
		}
		to_device(&r[0]);
	}

	void fill(float num) {
		auto r = alloc_vector();
		for (int i = 0; i < LEN; ++i) {
			r[i] = num;
		}
		to_device(&r[0]);
	}

	void local_transpose() {
		GPUFloatMat::local_transpose();
		LDIM = DIM_ROW;
	}

    void print_matrix(std::string msg)
    {
		auto r = alloc_vector();
		to_host(&r[0]);
        std::cout <<  msg << "\n";
        for (int i = 0; i < DIM_ROW; ++i) {
            for (int j = 0; j < DIM_COL; ++j) {
                std::cout << r[j*DIM_ROW+i] << '\t';
            }
            std::cout<<"\n";
        }
        std::cout << std::endl;
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
    	if (device_data_initialized) cudaFree(device_data);
		if (host_data) free(host_data);
    }

	virtual ~CudaFloatMat(){
		free_data();
	}

private:
    cublasOperation_t op;

	void init_device_mem() {

		GPU_CHECKERROR(
		cudaMalloc( (void**)&device_data, MEM_SIZE )
		);
		GPU_CHECKERROR(
		cudaMemset( (void *)device_data, 0, MEM_SIZE )
		);
		device_data_initialized = true;
	}


	void init_device_mem(float *d) {

		GPU_CHECKERROR(
		cudaMalloc( (void**)&device_data, MEM_SIZE )
		);

		GPU_CHECKERROR(
		cudaMemcpy( device_data, d, MEM_SIZE, cudaMemcpyHostToDevice )
		);
		device_data_initialized = true;
	}


	void init_dim(int m, int n) {
		GPUFloatMat::init_dim(m, n);
        LDIM = DIM_ROW;
        BLOCK_DIM.x = NUM_THREAD_PER_BLOCK;
        GRID_DIM.x = ceil(float(LEN)/float(BLOCK_DIM.x));
	}

	void init_dim(std::vector<int> dim) {
		GPUFloatMat::init_dim(dim);
        LDIM = DIM_ROW;
        BLOCK_DIM.x = NUM_THREAD_PER_BLOCK; //number of thread per block
        GRID_DIM.x = ceil(float(LEN)/float(BLOCK_DIM.x)); //number of block
	}

	//not used
	void to_column_major(float *target, float *source) {
		int c = 0;
		for (int i = 0; i < DIM_COL; ++i) {
			for (int j = 0; j < DIM_ROW; ++j) {
				target[c] = source[j*DIM_COL+i];
				c++;
			}
		}
	}

	//not used
	void to_row_major(float *target, float *source) {
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
