/*
 * Eona Studio (c) 2015
 */

#ifndef OPENCL_ENGINE_H_
#define OPENCL_ENGINE_H_

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

#include "ocl_util.h"
#include "../types/opencl_float_mat.h"
#include "../types/performance_profiler.h"

#include "../../engine/engine.h"
#include "../../engine/tensor.h"
#include "../../rand_utils.h"
using namespace std;

typedef std::shared_ptr<OpenclFloatMat> OpenclFloatMatPtr;
//typedef OpenclFloatMat* OpenclFloatMatPtr;

//typedef std::shared_ptr<float> FloatPtr;

class OpenclEngine : public Engine<OpenclFloatMat>
{
public:

	OpenclEngine(GlobalTimer * g) :
		Engine<OpenclFloatMat>()
	{
		gt = g;
		timed = true;
		cl = new OclUtilContext(true);
		cout<<"Initialized context"<<endl;
		/*Build program from source*/
		cl->build_program("./mat_op_kernel.cl", "matop_prog");
		/*Register kernel functions*/
		cl->register_kernel("mat_add_kernel", "matop_prog");
		cl->register_kernel("mat_scale_kernel", "matop_prog");
		cl->register_kernel("mat_elem_mult_kernel", "matop_prog");
		cl->register_kernel("mat_sigmoid_kernel", "matop_prog");
		cl->register_kernel("mat_sigmoid_gradient_kernel", "matop_prog");
		cl->register_kernel("mat_sin_kernel", "matop_prog");
		cl->register_kernel("mat_cos_kernel", "matop_prog");
		cl->register_kernel("mat_tanh_kernel", "matop_prog");
		cl->register_kernel("mat_tanh_gradient_kernel", "matop_prog");
		cl->register_kernel("mat_square_loss_kernel", "matop_prog");
		cl->register_kernel("mat_mult_NN_kernel", "matop_prog");

//		register_create(CudaEngine::create);
//		register_opcode("t+t", CudaEngine::add);
////		register_opcode("s+s", Impl::add<S>);
//		register_opcode("t-t", CudaEngine::sub);
////		register_opcode("s-s", sub);
//		register_opcode("-t", CudaEngine::negate);
////		register_opcode("-s", negate<S>);
//		register_opcode("t*t", CudaEngine::mult);
////		register_opcode("t*s", mult<T, S>);
////		register_opcode("s*t", mult<S, T>);
////		register_opcode("s*s", mult<S, S>);
//		register_opcode("t=t", CudaEngine::assign);
////		register_opcode("s=s", assign<S>);
//
//		register_opcode("scale", CudaEngine::scale);
//		register_opcode("sin", CudaEngine::sin);
//		register_opcode("cos", CudaEngine::cos);
//		register_opcode("tanh", CudaEngine::tanh);
//		register_opcode("tanh_gradient", CudaEngine::tanh_gradient);
//		register_opcode("sigmoid", CudaEngine::sigmoid);
//		register_opcode("sigmoid_gradient", CudaEngine::sigmoid_gradient);
//		register_opcode("transpose", CudaEngine::transpose);
//		register_opcode("element_mult", CudaEngine::element_mult);
//		register_opcode("square_loss", CudaEngine::square_loss);
//
//		register_opcode("destroy", CudaEngine::destroy);
//		register_opcode("fill_rand", CudaEngine::fill_rand);
//
//		/*********** DEBUG ONLY ***********/
//		register_opcode("debug_fill", CudaEngine::debug_fill);
	}




	void create(OpenclFloatMatPtr write, vector<int> dim)
	{
		DEBUG_MSG("OpenclEngine::create dim=" << dim);
		write->reset(dim, cl);
	}

	void debug_msg(string msg, bool is_initialized)
	{
		DEBUG_MSG(("OpenclEngine::" + msg + " ->init=") << std::boolalpha << is_initialized);
	}

	/*
	 * write = alpha * Op(reads[0]) + beta * Op(reads[1])
	 */
	void addMat(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized, float alpha, float beta)
	{
	    int m = reads[0]->DIM_ROW;
	    int n = reads[0]->DIM_COL;
	    if (!is_initialized) {
	        write->reset(m, n, cl); //initialize LHS if not already
	    }

		if(timed) ScopeTimer("add", gt, m*n);

	    //Register parameters and execute kernel
//		write->print_matrix("write");
	    cl->setup_kernel("mat_add_kernel", 0, sizeof(cl_mem), &write->device_data); // C
	    cl->setup_kernel("mat_add_kernel", 1, sizeof(cl_mem), &reads[0]->device_data); //A
	    cl->setup_kernel("mat_add_kernel", 2, sizeof(cl_mem), &reads[1]->device_data); //B
	    cl->setup_kernel("mat_add_kernel", 3, sizeof(float), &alpha); //a
	    cl->setup_kernel("mat_add_kernel", 4, sizeof(float), &beta); //b
	    cl->setup_kernel("mat_add_kernel", 5, sizeof(int), &(write->LEN)); //DATA_SIZE
	    cl->exec_kernel("mat_add_kernel", write->NUM_GLOBAL_WORKER, write->NUM_LOCAL_WORKER);
	}


	/*
	 * write = alpha .* Op(reads[0]) * Op(reads[1]) + beta * write
	 */
	void multMat(vector<OpenclFloatMatPtr> reads,
				OpenclFloatMatPtr write, bool is_initialized,
				std::string kernel_name)
	{
	    int m = reads[0]->DIM_ROW;
	    int n = reads[0]->DIM_COL;
	    int k = reads[1]->DIM_COL;
	    if (!is_initialized) {
	        write->reset(m, n, cl); //initialize LHS if not already
	    }
		if(timed) ScopeTimer(kernel_name, gt, m*k);

	    //Need to re-compute number of workers
	    int NUM_LOCAL_WORKER = write->NUM_LOCAL_WORKER;
		int NUM_GLOBAL_WORKER = ceil(double(m * k)/double(NUM_LOCAL_WORKER))*NUM_LOCAL_WORKER;

	    cl->setup_kernel(kernel_name, 0, sizeof(cl_mem), &write->device_data); // C
	    cl->setup_kernel(kernel_name, 1, sizeof(cl_mem), &reads[0]->device_data); //A
	    cl->setup_kernel(kernel_name, 2, sizeof(cl_mem), &reads[1]->device_data); //B
	    cl->setup_kernel(kernel_name, 3, sizeof(int), &m); //DATA_SIZE
	    cl->setup_kernel(kernel_name, 4, sizeof(int), &n); //DATA_SIZE
	    cl->setup_kernel(kernel_name, 5, sizeof(int), &k); //DATA_SIZE
	    cl->exec_kernel(kernel_name, NUM_GLOBAL_WORKER, NUM_LOCAL_WORKER);
	    //C = a Op(A)* Op(B) + b C  -- A [mxn] B [nxk] C[mxk]
	}

	void scaleMat(vector<OpenclFloatMatPtr> reads,
				OpenclFloatMatPtr write, bool is_initialized,
				float alpha)
	{
	    int m = reads[0]->DIM_ROW;
	    int n = reads[0]->DIM_COL;
	    if (!is_initialized) {
	        write->reset(m, n, cl); //initialize LHS if not already
	    }
		if(timed) ScopeTimer("scale", gt, m*n);

	    cl->setup_kernel("mat_scale_kernel", 0, sizeof(cl_mem), &write->device_data); // Y
	    cl->setup_kernel("mat_scale_kernel", 1, sizeof(cl_mem), &reads[0]->device_data); // X
	    cl->setup_kernel("mat_scale_kernel", 2, sizeof(float), &alpha); //a
	    cl->setup_kernel("mat_scale_kernel", 3, sizeof(int), &(write->LEN)); //DATA_SIZE
	    cl->exec_kernel("mat_scale_kernel", write->NUM_GLOBAL_WORKER, write->NUM_LOCAL_WORKER);
	}

	/*
	 * assign reads[0] to write
	 */
	void assignMat(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
	    int m = reads[0]->DIM_ROW;
	    int n = reads[0]->DIM_COL;
	    if (!is_initialized) {
	        write->reset(m, n, cl); //initialize LHS if not already
	    }
		if(timed) ScopeTimer("assign", gt, m*n);

	    //y = x
	    cl->copy(write->device_data, reads[0]->device_data, reads[0]->MEM_SIZE);
	}

	void elementOp(std::string kernel_name, vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized){
	    int m = reads[0]->DIM_ROW;
	    int n = reads[0]->DIM_COL;
	    if (!is_initialized) {
	        write->reset(m, n, cl); //initialize LHS if not already
	    }
		if(timed) ScopeTimer(kernel_name, gt, m*n);
	    //y = x
	    cl->setup_kernel(kernel_name, 0, sizeof(cl_mem), &write->device_data); // Y
	    cl->setup_kernel(kernel_name, 1, sizeof(cl_mem), &reads[0]->device_data); // X
	    cl->setup_kernel(kernel_name, 2, sizeof(int), &(write->LEN)); //DATA_SIZE
	    cl->exec_kernel(kernel_name, write->NUM_GLOBAL_WORKER, write->NUM_LOCAL_WORKER);
	}


	void add(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a+b", is_initialized);
	    float alpha = 1.0f;
	    addMat(reads, write, is_initialized, alpha, alpha);
	}

	void sub(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a-b", is_initialized);

	    float alpha = 1.0f;
	    addMat(reads, write, is_initialized, alpha, -alpha);
	}

	void negate(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=-a", is_initialized);
	    //y = -y
	    scaleMat(reads, write, is_initialized, -1);
	}

	void mult(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a*b", is_initialized);
		float alpha = 1.0f;
		multMat(reads, write, is_initialized, "mat_mult_NN_kernel");
	}

	void assign(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a", is_initialized);
	    assignMat(reads, write, is_initialized);
	}

	inline void scale(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized, float* scaler)
	{
		debug_msg("scale", is_initialized);
	    //y = ay
	    scaleMat(reads, write, is_initialized, *scaler);
	}

	inline void destroy(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("destroy", is_initialized);
		reads[0]->free_data();
	}


	// standalone single-float non-linear functions
	inline void transpose(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("transpose", is_initialized);
		//TODO
	}


	inline void sigmoid(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("sigmoid", is_initialized);
		elementOp("mat_sigmoid_kernel", reads, write, is_initialized);
	}

	inline void sigmoid_gradient(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("sigmoid_gradient", is_initialized);
		elementOp("mat_sigmoid_gradient_kernel", reads, write, is_initialized);
	}

	inline void sin(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("sin", is_initialized);
		elementOp("mat_sin_kernel", reads, write, is_initialized);
	}

	inline void cos(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("cos", is_initialized);
		elementOp("mat_cos_kernel", reads, write, is_initialized);
	}

	inline void tanh(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("tanh", is_initialized);
		elementOp("mat_tanh_kernel", reads, write, is_initialized);
	}

	inline void tanh_gradient(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("tanh_gradient", is_initialized);
		elementOp("mat_tanh_gradient_kernel", reads, write, is_initialized);
	}

	inline void element_mult(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("element_mult", is_initialized);
	    int m = reads[0]->DIM_ROW;
	    int n = reads[0]->DIM_COL;
	    if (!is_initialized) {
	        write->reset(m, n, cl); //initialize LHS if not already
	    }

		if(timed) ScopeTimer("elem_mult", gt, m*n);
	    //y = x
	    cl->setup_kernel("mat_elem_mult_kernel", 0, sizeof(cl_mem), &write->device_data); // Y
	    cl->setup_kernel("mat_elem_mult_kernel", 1, sizeof(cl_mem), &reads[0]->device_data); // X
	    cl->setup_kernel("mat_elem_mult_kernel", 2, sizeof(cl_mem), &reads[1]->device_data); // X
	    cl->setup_kernel("mat_elem_mult_kernel", 3, sizeof(int), &(write->LEN)); //DATA_SIZE
	    cl->exec_kernel("mat_elem_mult_kernel", write->NUM_GLOBAL_WORKER, write->NUM_LOCAL_WORKER);
	}

	inline void square_loss(vector<OpenclFloatMatPtr> reads, float* write, bool is_initialized)
	{
		debug_msg("square_loss", is_initialized);
	    int m = reads[0]->DIM_ROW;
	    int n = reads[0]->DIM_COL;
	    OpenclFloatMat aux(m, n, cl);
	    //y = x
		if(timed) ScopeTimer("square_loss", gt, m*n);
	    cl->setup_kernel("mat_square_loss_kernel", 0, sizeof(cl_mem), &aux.device_data); // Y
	    cl->setup_kernel("mat_square_loss_kernel", 1, sizeof(cl_mem), &reads[0]->device_data); // X
	    cl->setup_kernel("mat_square_loss_kernel", 2, sizeof(cl_mem), &reads[1]->device_data); // X
	    cl->setup_kernel("mat_square_loss_kernel", 3, sizeof(int), &(aux.LEN)); //DATA_SIZE
	    cl->exec_kernel("mat_square_loss_kernel", aux.NUM_GLOBAL_WORKER, aux.NUM_LOCAL_WORKER);

	    float t[aux.MEM_SIZE];
	    aux.to_host(t);
	    for (int i = 0; i < aux.LEN; ++i) {
	    	*write += t[i];
	    }
	}

	// FIXME add contextual rand engine
	inline void fill_rand(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("fill_rand", is_initialized);
		if (!is_initialized) {
	        write->reset(write->DIM_ROW, write->DIM_COL, cl); //initialize LHS if not already
		}
		write->fill_rand(1);
	}


	/*********** DEBUG ONLY ***********/
	inline void debug_fill(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		if (!is_initialized) {
	        write->reset(write->DIM_ROW, write->DIM_COL, cl); //initialize LHS if not already
		}
		write->fill(0.66337);
	}

	~OpenclEngine()
	{
		delete cl;
	}
	OclUtilContext* cl;

private:
	bool timed;
	GlobalTimer * gt;
};

#endif /* OPENCL_ENGINE_H_ */
