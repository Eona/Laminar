#ifndef GPU_FLOAT_MAT_H_
#define GPU_FLOAT_MAT_H_

#include <vector>

class GPUFloatMat
{
public:
	int DIM_ROW; //number of rows
	int DIM_COL; //number of columns
	int LEN; //number of data element in the buffer
	int MEM_SIZE; //size of the buffer
	int NUM_DIM; //number of dimensions

	std::vector<int> DIM_ALL; //all dimensions

	float * host_data; //host data

	GPUFloatMat(): host_data(NULL) {

	}

	GPUFloatMat(float *d, int m, int n): host_data(NULL) {
		init_dim(m, n); //initialize matrix dimensions
		init_device_mem(d); //copy data to device
	}

	GPUFloatMat(int m, int n): host_data(NULL) {
		init_dim(m, n); //initialize matrix dimensions
		init_device_mem();
	}


	GPUFloatMat(std::vector<int> dim): host_data(NULL) {
		init_dim(dim); //initialize matrix dimensions
		init_device_mem();
	}


	/*
	 * Copy data to device
	 */
	void to_device(float *d) {

	}

	/*
	 * Copy device data to host
	 */

	void to_host() {
	}


	void fill_rand(int seed) {
	}

	void fill(float num) {
	}


    void print_matrix(std::string msg) {
    }

    void free_data(){
		if (host_data) free(host_data);
    }

	~GPUFloatMat(){
		free_data();
	}

protected:
	//Initialize device memory and set the memory to zero
	void init_device_mem() {
	}

	//Initialize device memory and set the memory to input data
	void init_device_mem(float *d) {
	}

	void init_dim(int m, int n) {
		NUM_DIM = 2;
		DIM_ROW = m;
		DIM_COL = n;
		LEN = DIM_ROW * DIM_COL;
		MEM_SIZE = LEN * sizeof(float);
	}

	void init_dim(std::vector<int> dim){
		DIM_ALL = dim;
		NUM_DIM = dim.size();
		LEN = 1;

		for (int i = 0; i < dim.size(); ++i) {
			LEN *= dim[i];
		}

		if (dim.size() > 0) {
			DIM_ROW = dim[0];
		} else {
			printf("Error: The matrix has no dimension information");
		}

		if (dim.size() > 1) {
			DIM_COL = dim[1];
		} else {
			DIM_COL = 1;
		}
		MEM_SIZE = LEN * sizeof(float);
	}
};

#endif
