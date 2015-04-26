/*
 * Eona Studio (c)2015
 */

#ifndef OPENCL_FLOAT_MAT_H_
#define OPENCL_FLOAT_MAT_H_

#ifdef __APPLE__
#include <OpenCL/cl.h>
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl.h>
#include <CL/cl_platform.h>
#endif
#include <vector>
#include "../opencl/ocl_util.h"
#include "gpu_float_mat.h"

#define NUM_WORKER_LOCAL 512;


class OpenclFloatMat : public GPUFloatMat
{
public:

	size_t NUM_GLOBAL_WORKER; // total number of workers
	size_t NUM_LOCAL_WORKER; // number of workers per block
    
	cl_mem device_data;

	OclUtilContext* cl;



	OpenclFloatMat(){
		init_dim(1,1);
		host_data = NULL;
		device_data_initialized = false;
	}

	OpenclFloatMat(float *d, int m, int n, OclUtilContext* context) {
		cl = context;
		host_data = NULL;
		device_data_initialized = false;
		init_dim(m, n); //initialize matrix dimensions
		init_device_mem(d); //copy data to device
	}

	OpenclFloatMat(int m, int n, OclUtilContext* context) {
		cl = context;
		host_data = NULL;
		device_data_initialized = false;
		init_dim(m, n); //initialize matrix dimensions
		init_device_mem();
	}


	OpenclFloatMat(std::vector<int> dim, OclUtilContext* context) {
		cl = context;
		host_data = NULL;
		device_data_initialized = false;
		init_dim(dim); //initialize matrix dimensions
		init_device_mem();
	}

	void reset(int m, int n, OclUtilContext* context) {
		cl = context;
		init_dim(m, n); //initialize matrix dimensions
		init_device_mem();
	}

	void reset(vector<int> dim, OclUtilContext* context) {
		cl = context;
		init_dim(dim); //initialize matrix dimensions
		init_device_mem();
	}

	void reset(float * d, int m, int n, OclUtilContext* context) {
		cl = context;
		init_dim(m, n); //initialize matrix dimensions
		init_device_mem(d);
	}


	/*
	 * Copy data to device
	 */
	void to_device(float *d) {
		cl->to_device_write(device_data, d, MEM_SIZE);
	}

	/*
	 * Copy device data to host
	 */
	void to_host(float *d) {
		cl->to_host(d, device_data, MEM_SIZE);
	}


	void zero_clear() {
		cl->to_device_fill(device_data, MEM_SIZE, 0);
	}

	void fill_rand(int seed) {
		auto r= alloc_vector();
		srand (seed);
		for (int i = 0; i < LEN; ++i) {
			r[i] = 0.16*( (double) rand() / (RAND_MAX)) - 0.08;
		}
		to_device(&r[0]);
	}

	void fill(float num) {
		cl->to_device_fill(device_data, MEM_SIZE, num);
	}


    void print_matrix(std::string msg) {
		auto d= alloc_vector();
		cl->to_host(&d[0], device_data, MEM_SIZE);
        std::cout <<  msg << "\n";
        for (int i = 0; i < DIM_ROW; ++i) {
            for (int j = 0; j < DIM_COL; ++j) {
                std::cout << d[j*DIM_ROW+i] << '\t';
            }
            std::cout<<"\n";
        }
        std::cout << std::endl;
    }

    void take_at(float * d, size_t offset, size_t num_float) {
    	cl->to_host_at(d, device_data, offset*sizeof(float), num_float*sizeof(float));
    }

	virtual ~OpenclFloatMat(){
		free_data();
	}

    void free_data(){
		if (host_data) delete [] host_data;
		if (device_data_initialized) {
			clReleaseMemObject(device_data);
			device_data_initialized = false;
		}
    }


private:

	void init_device_mem() {
		device_data = cl->to_device_create_zero(MEM_SIZE);
		device_data_initialized = true;
	}


	void init_device_mem(float *d) {
		device_data = cl->to_device(d, MEM_SIZE);
		device_data_initialized = true;
	}


	void init_dim(int m, int n) {
		GPUFloatMat::init_dim(m, n);
		NUM_LOCAL_WORKER = NUM_WORKER_LOCAL; // number of workers per block
		NUM_GLOBAL_WORKER = ceil(double(LEN)/double(NUM_LOCAL_WORKER))*NUM_LOCAL_WORKER;
	}

	void init_dim(std::vector<int> dim){
		GPUFloatMat::init_dim(dim);
		NUM_LOCAL_WORKER = NUM_WORKER_LOCAL;// number of workers per block
		NUM_GLOBAL_WORKER = ceil(double(LEN)/double(NUM_LOCAL_WORKER))*NUM_LOCAL_WORKER;
	}
};

#endif
