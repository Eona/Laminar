/*
 * Eona Studio (c) 2015
 */
#ifndef OCL_UTIL_H_
#define OCL_UTIL_H_

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
#include <unordered_map>
 
#define MAX_SOURCE_SIZE (0x100000)
 
using namespace std;

char *getCLErrorString(cl_int err) {
    switch (err) {
        case CL_SUCCESS:                          return (char *) "Success!";
        case CL_DEVICE_NOT_FOUND:                 return (char *) "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:             return (char *) "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:           return (char *) "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return (char *) "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                 return (char *) "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:               return (char *) "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:     return (char *) "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                 return (char *) "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:            return (char *) "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return (char *) "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:            return (char *) "Program build failure";
        case CL_MAP_FAILURE:                      return (char *) "Map failure";
        case CL_INVALID_VALUE:                    return (char *) "Invalid value";
        case CL_INVALID_DEVICE_TYPE:              return (char *) "Invalid device type";
        case CL_INVALID_PLATFORM:                 return (char *) "Invalid platform";
        case CL_INVALID_DEVICE:                   return (char *) "Invalid device";
        case CL_INVALID_CONTEXT:                  return (char *) "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:         return (char *) "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:            return (char *) "Invalid command queue";
        case CL_INVALID_HOST_PTR:                 return (char *) "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:               return (char *) "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return (char *) "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:               return (char *) "Invalid image size";
        case CL_INVALID_SAMPLER:                  return (char *) "Invalid sampler";
        case CL_INVALID_BINARY:                   return (char *) "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:            return (char *) "Invalid build options";
        case CL_INVALID_PROGRAM:                  return (char *) "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:       return (char *) "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:              return (char *) "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:        return (char *) "Invalid kernel definition";
        case CL_INVALID_KERNEL:                   return (char *) "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                return (char *) "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                return (char *) "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                 return (char *) "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:              return (char *) "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:           return (char *) "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:          return (char *) "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:           return (char *) "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:            return (char *) "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:          return (char *) "Invalid event wait list";
        case CL_INVALID_EVENT:                    return (char *) "Invalid event";
        case CL_INVALID_OPERATION:                return (char *) "Invalid operation";
        case CL_INVALID_GL_OBJECT:                return (char *) "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:              return (char *) "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                return (char *) "Invalid mip-map level";
        default:                                  return (char *) "Unknown";
    }
}

#define OCL_CHECKERROR( err ) (oclCheckError( err, __FILE__, __LINE__ ))
static void oclCheckError( cl_int err,
                          const char *file,
                          int line ) {
    if (err != CL_SUCCESS) {
        printf( "%s in %s at line %d\n", getCLErrorString( err ),
               file, line );
        exit( EXIT_FAILURE );
    }
}


/*Build OpenCL program from a text file*/
cl_program oclutil_build_program(std::string filename, cl_context & context, cl_device_id device_id) {
	cl_program program;

    /*read in program text*/
    cl_int ret;
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen(filename.c_str(), "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel program.\n");
        exit(1);
    }

    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    /* Create and build kernel program */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    OCL_CHECKERROR(ret);
    free(source_str);

    OCL_CHECKERROR(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

    // Shows the log
    char* build_log;
    size_t log_size;
    // First call to know the proper size
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    build_log = new char[log_size+1];
    // Second call to get the log
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    build_log[log_size] = '\0';
    cout << "OpenCL: log of compiling " <<filename << endl << build_log << endl;
    delete[] build_log;

    return program;
}


/*Create OpenCL context and other dependency*/
class OclUtilContext{
public:
    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;
    cl_platform_id platform_id;
	cl_int ret;
	cl_program program;


	std::unordered_map<std::string, cl_kernel> kernel_list; //pre-compiled kernels
	std::unordered_map<std::string, cl_program> program_list; //pre-compiled kernels

    OclUtilContext(bool use_gpu){
    	OCL_CHECKERROR(clGetPlatformIDs(1, &platform_id, NULL));
    	OCL_CHECKERROR(clGetDeviceIDs(platform_id, use_gpu?CL_DEVICE_TYPE_GPU:CL_DEVICE_TYPE_CPU, 1, &device_id, NULL));
        /* Create OpenCL context */
        context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
        OCL_CHECKERROR(ret);
        /* Create Command Queue */
        command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
        OCL_CHECKERROR(ret);
    }

    /*build a program from file*/
    void build_program(std::string filename, std::string program_name) {
    	program_list[program_name] = oclutil_build_program(filename, context, device_id);
    }

    /*register kernel to the map*/
    void register_kernel(std::string kernel_name, std::string program_name) {
        kernel_list[kernel_name] = clCreateKernel(program_list[program_name], kernel_name.c_str(), &ret);
        OCL_CHECKERROR(ret);
    }

    /*Setup the parameter of a kernel specified by kernel_name*/
    void setup_kernel(std::string kernel_name, size_t param_index, size_t param_type_size, void* param){
        OCL_CHECKERROR(clSetKernelArg(kernel_list[kernel_name], param_index, param_type_size, param));
    }

    /*Execute a kernel specified by kernel_name*/
    void exec_kernel(cl_kernel& kernel, size_t global_ws, size_t local_ws){
    	OCL_CHECKERROR(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_ws, &local_ws, 0, NULL, NULL));
	}

    /*Execute a kernel specified by kernel_name*/
    void exec_kernel(std::string kernel_name, size_t global_ws, size_t local_ws){
    	OCL_CHECKERROR(clEnqueueNDRangeKernel(command_queue, kernel_list[kernel_name], 1, NULL, &global_ws, &local_ws, 0, NULL, NULL));
	}

    /*flush the command queue*/
    void flush_queue(){
    	OCL_CHECKERROR(clFlush(command_queue));
    }

    ///////////////////Memory Functions////////////////////
    cl_mem to_device(float* d, size_t MEM_SIZE){
    	cl_mem memobj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, MEM_SIZE, d, &ret);
    	OCL_CHECKERROR(ret);
    	return memobj;
    }

    cl_mem to_device_create(size_t MEM_SIZE){
    	cl_mem memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE, NULL, &ret);
    	OCL_CHECKERROR(ret);
    	return memobj;
    }

    template<typename T>
    cl_mem to_device_create_zero(size_t MEM_SIZE){
    	cl_mem memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE, NULL, &ret);
    	OCL_CHECKERROR(ret);
    	T pattern = 0;
    	to_device_fill<T>(memobj, MEM_SIZE, pattern);
    	return memobj;
    }

    template<typename T>
    void to_device_fill(cl_mem memobj, size_t MEM_SIZE, T pattern){
    	OCL_CHECKERROR(clEnqueueFillBuffer(command_queue, memobj, &pattern, sizeof(T), 0, MEM_SIZE, 0, NULL, NULL));
    }

    void to_device_write(cl_mem& buffer, float* d, size_t MEM_SIZE){
    	OCL_CHECKERROR(clEnqueueWriteBuffer(command_queue, buffer, CL_TRUE, 0, MEM_SIZE, d, 0, NULL, NULL));
    }

    void to_host(float* out, cl_mem& memobj, size_t MEM_SIZE) {
    	OCL_CHECKERROR(clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0, MEM_SIZE, out, 0, NULL, NULL));
    }


    /*Clean up*/
    ~OclUtilContext(){
    	OCL_CHECKERROR(clFlush(command_queue));
    	OCL_CHECKERROR(clFinish(command_queue));
    	OCL_CHECKERROR(clReleaseCommandQueue(command_queue));
    	OCL_CHECKERROR(clReleaseContext(context));

    	//Delete programs
    	for ( auto it = program_list.begin(); it != program_list.end(); ++it ) {
    		OCL_CHECKERROR(clReleaseProgram(it->second));
    	}

    	//Delete kernels
    	for ( auto it = kernel_list.begin(); it != kernel_list.end(); ++it ) {
    		OCL_CHECKERROR(clReleaseKernel(it->second));
    	}
    }
};




//void get_device_info(cl_device_id device_id){
//	printf("=== %d OpenCL device(s) found on platform:\n", platforms_n);
//	for (int i=0; i<devices_n; i++)
//	{
//		char buffer[10240];
//		cl_uint buf_uint;
//		cl_ulong buf_ulong;
//		printf("  -- %d --\n", i);
//		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
//		printf("  DEVICE_NAME = %s\n", buffer);
//		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL));
//		printf("  DEVICE_VENDOR = %s\n", buffer);
//		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL));
//		printf("  DEVICE_VERSION = %s\n", buffer);
//		CL_CHECK(clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL));
//		printf("  DRIVER_VERSION = %s\n", buffer);
//		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL));
//		printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
//		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL));
//		printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
//		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL));
//		printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
//	}
//
//}

#endif
