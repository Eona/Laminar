#ifdef __APPLE__
#include <OpenCL/cl.h>
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl.h>
#include <CL/cl_platform.h>
#endif

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <math.h>
 
#define MAX_SOURCE_SIZE (0x100000)
 
using namespace std;


/*Part of the setup code comes from 
*http://www.fixstars.com/en/opencl/book/OpenCLProgrammingBook/first-opencl-program/ 
*/

void oclutil_build_program(cl_program & program, std::string filename, cl_context & context, cl_device_id device_id) {
    /*read in program text*/
    cl_int ret;
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen(filename.c_str(), "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }

    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    /* Create and build kernel program */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    assert(ret == CL_SUCCESS);
    free(source_str);

    assert(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL) == CL_SUCCESS);

    // Shows the log
    char* build_log;
    size_t log_size;
    // First call to know the proper size
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    build_log = new char[log_size+1];
    // Second call to get the log
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    build_log[log_size] = '\0';
    cout << build_log << endl;
    delete[] build_log;
}

int main(int argc, char *argv[])
{
	float * data = new float[100];
    int DATA_SIZE = 100;
    int MEM_SIZE = DATA_SIZE*sizeof(float);


    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem memobj = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_platform_id platform_id = NULL;
    cl_int ret;
 
 
    /* Get Platform and Device Info */
    ret = clGetPlatformIDs(1, &platform_id, NULL);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
 
    /* Create OpenCL context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    assert(ret == CL_SUCCESS);

    /* Create Command Queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    assert(ret == CL_SUCCESS);
 
    /* Create Memory Buffer */
    memobj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, MEM_SIZE, data, &ret);
    assert(ret == CL_SUCCESS);
 
    /*Build program from source code*/
    oclutil_build_program(program, "./greyScale.cl", context, device_id);
 
    /* Create OpenCL Kernel */
    kernel = clCreateKernel(program, "greyScale", &ret);
    assert(ret == CL_SUCCESS);
 
    /* Set OpenCL Kernel Parameters */
    assert(clSetKernelArg(kernel, 0, sizeof(cl_mem), &memobj) == CL_SUCCESS);
    assert(clSetKernelArg(kernel, 1, sizeof(int), &DATA_SIZE) == CL_SUCCESS);
 
    /* Execute OpenCL Kernel */
    const size_t local_ws = 100;    // Number of work-items per work-group
    // shrRoundUp returns the smallest multiple of local_ws bigger than size
    const size_t global_ws = ceil(double(DATA_SIZE)/double(local_ws))*local_ws;    // Total number of work-items
    cout<<global_ws<<endl;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
    assert(ret == CL_SUCCESS);
 
    /* Copy results from the memory buffer */
    float* out = new float[MEM_SIZE];
    ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0,
    MEM_SIZE, out, 0, NULL, NULL);
    assert(ret == CL_SUCCESS);
	for (int i = 0; i < DATA_SIZE; ++i) {
		cout<<out[i];
	}
 
    /* Finalization */
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(memobj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
 
    delete[] out;
    delete[] data;
 
    return 0;
}
