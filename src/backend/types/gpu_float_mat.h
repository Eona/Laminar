#ifndef GPU_FLOAT_MAT_H_
#define GPU_FLOAT_MAT_H_

#include <vector>
#include "../../utils/laminar_utils.h"

class GPUFloatMat
{
public:
	int DIM_ROW; //number of rows
	int DIM_COL; //number of columns
	int LEN; //number of data element in the buffer
	int MEM_SIZE; //size of the buffer
	int NUM_DIM; //number of dimensions
	float scalar = 0; //only used in square loss
	bool isScalar = false;

	Dimension DIM_ALL; //all dimensions

	float * host_data; //host data

	GPUFloatMat(): host_data(nullptr) {

	}

	/*
	 * Copy data to device
	 */
	virtual void to_device(float *d) = 0;
	/*
	 * Copy device data to host
	 */

	virtual void to_host(float *d) = 0;

	virtual void fill_rand(int seed) = 0;

	virtual void fill(float num) = 0;

	virtual void local_transpose() {
		float * r = new float[LEN];
		float * d = new float[LEN];
		to_host(r);
		int c = 0;
		for (int i = 0; i < DIM_ROW; ++i) {
			for (int j = 0; j < DIM_COL; ++j) {
				d[c] = r[j*DIM_ROW+i];
				c++;
			}
		}
		to_device(d);
		delete[] r;
		delete[] d;
		//swap dimension
		int t = DIM_ROW;
		DIM_ROW = DIM_COL;
		DIM_COL = t;
	}


    virtual void print_matrix(std::string msg) = 0;

	virtual void perturb(size_t idx, float val) {
		float * r = new float[LEN];
		to_host(r);
		r[idx] += val;
		to_device(r);
		delete[] r;
	}

    virtual void free_data() = 0;

	virtual ~GPUFloatMat(){ }

protected:
	bool device_data_initialized;

	//Initialize device memory and set the memory to zero
	virtual void init_device_mem() = 0;

	//Initialize device memory and set the memory to input data
	virtual void init_device_mem(float *d) = 0;

	virtual void init_dim(int m, int n) {
		NUM_DIM = 2;
		DIM_ROW = m;
		DIM_COL = n;
		LEN = DIM_ROW * DIM_COL;
		MEM_SIZE = LEN * sizeof(float);
	}

	virtual void init_dim(Dimension dim){
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
