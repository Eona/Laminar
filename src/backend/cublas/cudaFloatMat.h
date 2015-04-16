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
	int DIM_ROW;
	int DIM_COL;
	int LEN;
	int NUM_DIM;
    int LDIM;
	std::vector<int> DIM_ALL;
    

	float * device_data;
	float * host_data;


	CudaFloatMat():
		device_data(NULL),
		host_data(NULL),
        op(CUBLAS_OP_N)
    {   }

	CudaFloatMat(float *d, int m, int n):
		device_data(NULL),
		host_data(NULL),
        op(CUBLAS_OP_N)
	{
		init_dim(m, n); //initialize matrix dimensions
		init_cuda_mem(d); //copy data to device
	}

	CudaFloatMat(int m, int n):
		device_data(NULL),
		host_data(NULL),
        op(CUBLAS_OP_N)
	{
		init_dim(m, n); //initialize matrix dimensions
		init_cuda_mem();
	}


	CudaFloatMat(std::vector<int> dim):
		device_data(NULL),
		host_data(NULL),
        op(CUBLAS_OP_N)
	{
		init_dim(dim); //initialize matrix dimensions
		init_cuda_mem();
	}

	void init_cuda_mem() {
		GPU_CHECKERROR(
		cudaMalloc( (void**)&device_data, LEN * sizeof(float) )
		);
		GPU_CHECKERROR(
		cudaMemset( device_data, 0, LEN * sizeof(float) )
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
		DIM_ROW = m;
		DIM_COL = n;
		LEN = DIM_ROW * DIM_COL;
		NUM_DIM = 2;
        LDIM = DIM_ROW;
	}

	void init_dim(std::vector<int> dim){
		DIM_ALL = dim;
		NUM_DIM = dim.size();
		LEN = 1;
		for (int i = 0; i < dim.size(); ++i) {
			LEN *= dim[i];
		}
		if (dim.size() > 0) DIM_ROW = dim[0];
		if (dim.size() > 1) DIM_COL = dim[1];

        LDIM = DIM_ROW;
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
    

    void print_matrix(std::string msg){
        to_host();
        std::cout << msg << "\n";
        for (int i = 0; i < DIM_ROW; ++i) {
            for (int j = 0; j < DIM_COL; ++j) {
                std::cout << host_data[i*DIM_COL+j] << '\t';
            }
            std::cout<<"\n";
        } 
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
	~CudaFloatMat(){
		if (device_data) cudaFree(device_data);
		//if (host_data) free(host_data);
	}

private:
    cublasOperation_t op;
};

#endif
