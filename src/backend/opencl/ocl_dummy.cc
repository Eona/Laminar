/*
 * Eona Studio (c)2015
 */
#include "ocl_util.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
	float * data = new float[100];
    int DATA_SIZE = 100;
    int MEM_SIZE = DATA_SIZE*sizeof(float);

    cl_mem memobj = NULL;
    cl_int ret; //error msg

    /* Get Platform and Device Info */
    OclUtilContext cl(true);

    /* Create Memory Buffer and copy data to device */
    memobj = cl.to_device(data, MEM_SIZE);

    /*Build program from source code*/
    cl.build_program("./dummy.cl", "dummy_prog");
    /* Set OpenCL Kernel Parameters */
    cl.register_kernel("dummy_kernel", "dummy_prog");
    /* Set OpenCL Kernel Parameters */
    cl.setup_kernel("dummy_kernel", 0, sizeof(cl_mem), &memobj);
    cl.setup_kernel("dummy_kernel", 1, sizeof(int), &DATA_SIZE);

    /* Execute OpenCL Kernel */
    size_t local_ws = 100;    // Number of work-items per work-group

    size_t global_ws = ceil(double(DATA_SIZE)/double(local_ws))*local_ws;    // Total number of work-items
    cout<<global_ws<<endl;
    cl.exec_kernel("dummy_kernel", global_ws, local_ws);

    /* Copy results from the memory buffer */
    float* out = new float[MEM_SIZE];
    cl.to_host(out, memobj, MEM_SIZE);

	for (int i = 0; i < DATA_SIZE; ++i) {
		cout<<out[i];
	}

    /* Finalization */
    ret = clReleaseMemObject(memobj);


    delete[] out;
    delete[] data;

    return 0;
}
