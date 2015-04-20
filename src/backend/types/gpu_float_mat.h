#ifndef GPU_FLOAT_MAT_H_
#define GPU_FLOAT_MAT_H_

#include <vector>

class GPUFloatMat
{
public:
	int DIM_ROW; //
	int DIM_COL;
	int LEN;
	int MEM_SIZE;
	int NUM_DIM;

	std::vector<int> DIM_ALL;

	float * host_data;

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
		float r[MEM_SIZE];
		srand (seed);
		for (int i = 0; i < LEN; ++i) {
			r[i] = (double) rand() / (RAND_MAX);
		}
		to_device(r);
	}

	void fill(float num) {
		float r[MEM_SIZE];
		for (int i = 0; i < LEN; ++i) {
			r[i] = num;
		}
		to_device(r);
	}


    void print_matrix(std::string msg) {
        to_host();
        std::cout << "\n" << msg << "\n";
        for (int i = 0; i < DIM_ROW; ++i) {
            for (int j = 0; j < DIM_COL; ++j) {
                std::cout << host_data[j*DIM_ROW+i] << '\t';
            }
            std::cout<<"\n";
        }
    }

    void free_data(){
		if (host_data) free(host_data);
    }

	~GPUFloatMat(){
		free_data();
	}

protected:

	void init_device_mem() {
	}


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
