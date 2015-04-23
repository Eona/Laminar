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

	OpenclEngine() :
		Engine<OpenclFloatMat>()
	{
		timed = false; //if profile time performance
		init(); //must call as the last line
	}

	OpenclEngine(GlobalTimer * g) :
		Engine<OpenclFloatMat>()
	{
		timed = true; //if profile time performance
		gt = g; //global timer
		init(); //must call as the last line
	}

	void init() {
		cl = new OclUtilContext(true, timed);
		cout<<"Initialized context"<<endl;
		/*Build program from source*/
		cl->build_program("./mat_op_kernel.cl", "matop_prog");
		/*Register kernel functions*/
		cl->register_kernel("mat_add_kernel", "matop_prog", "add");
		cl->register_kernel("mat_scale_kernel", "matop_prog", "scale");
		cl->register_kernel("mat_elem_mult_kernel", "matop_prog", "element_mult");
		cl->register_kernel("mat_sigmoid_kernel", "matop_prog", "sigmoid");
		cl->register_kernel("mat_sigmoid_gradient_kernel", "matop_prog", "sigmoid_gradient");
		cl->register_kernel("mat_sin_kernel", "matop_prog", "sin");
		cl->register_kernel("mat_cos_kernel", "matop_prog", "cos");
		cl->register_kernel("mat_tanh_kernel", "matop_prog", "tanh");
		cl->register_kernel("mat_tanh_gradient_kernel", "matop_prog", "tanh_gradient");
		cl->register_kernel("mat_square_loss_kernel", "matop_prog", "square_loss");
		cl->register_kernel("mat_mult_NN_kernel", "matop_prog", "mult_NN");
		cl->register_kernel("mat_mult_NT_kernel", "matop_prog", "mult_NT");
		cl->register_kernel("mat_mult_TN_kernel", "matop_prog", "mult_TN");
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


	    //Register parameters and execute kernel
//		write->print_matrix("write");
	    cl->setup_kernel("add", 0, sizeof(cl_mem), &write->device_data); // C
	    cl->setup_kernel("add", 1, sizeof(cl_mem), &reads[0]->device_data); //A
	    cl->setup_kernel("add", 2, sizeof(cl_mem), &reads[1]->device_data); //B
	    cl->setup_kernel("add", 3, sizeof(float), &alpha); //a
	    cl->setup_kernel("add", 4, sizeof(float), &beta); //b
	    cl->setup_kernel("add", 5, sizeof(int), &(write->LEN)); //DATA_SIZE
	    cl_ulong duration = cl->exec_kernel("add", write->NUM_GLOBAL_WORKER, write->NUM_LOCAL_WORKER);

		if(timed) gt->record_named_timer("add", duration, m*n*2);

	}


	/*
	 * write = alpha .* Op(reads[0]) * Op(reads[1]) + beta * write
	 */
	void multMat(vector<OpenclFloatMatPtr> reads,
				OpenclFloatMatPtr write, bool is_initialized,
				std::string opA, std::string opB)
	{
	    int m = reads[0]->DIM_ROW;
	    int n = reads[0]->DIM_COL;
	    int l = reads[1]->DIM_ROW;
	    int k = reads[1]->DIM_COL;

	    std::string kernel_name = "mult_" + opA + opB;
	    if (!is_initialized) {
		    if (opA == "N" && opB == "N") write->reset(m, n, cl); // A * B
		    if (opA == "N" && opB == "T") write->reset(m, l, cl); // A * B^T
		    if (opA == "T" && opB == "N") write->reset(n, k, cl); // A^T * B
	    }

	    //C = a Op(A)* Op(B) + b C  -- A [mxn] B [lxk]
	    //Need to re-compute number of workers
	    int NUM_LOCAL_WORKER = write->NUM_LOCAL_WORKER;
		int NUM_GLOBAL_WORKER = ceil(double(write->LEN)/double(NUM_LOCAL_WORKER))*NUM_LOCAL_WORKER;

	    cl->setup_kernel(kernel_name, 0, sizeof(cl_mem), &write->device_data); // C
	    cl->setup_kernel(kernel_name, 1, sizeof(cl_mem), &reads[0]->device_data); //A
	    cl->setup_kernel(kernel_name, 2, sizeof(cl_mem), &reads[1]->device_data); //B
	    cl->setup_kernel(kernel_name, 3, sizeof(int), &m); //DATA_SIZE
	    cl->setup_kernel(kernel_name, 4, sizeof(int), &n); //DATA_SIZE
	    cl->setup_kernel(kernel_name, 5, sizeof(int), &l); //DATA_SIZE
	    cl->setup_kernel(kernel_name, 6, sizeof(int), &k); //DATA_SIZE
	    cl_ulong duration = cl->exec_kernel(kernel_name, NUM_GLOBAL_WORKER, NUM_LOCAL_WORKER);

		if(timed) gt->record_named_timer("mult_"+opA+opB, duration, m*n + l*k);

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

	    cl->setup_kernel("scale", 0, sizeof(cl_mem), &write->device_data); // Y
	    cl->setup_kernel("scale", 1, sizeof(cl_mem), &reads[0]->device_data); // X
	    cl->setup_kernel("scale", 2, sizeof(float), &alpha); //a
	    cl->setup_kernel("scale", 3, sizeof(int), &(write->LEN)); //DATA_SIZE
	    cl_ulong duration = cl->exec_kernel("scale", write->NUM_GLOBAL_WORKER, write->NUM_LOCAL_WORKER);

		if(timed) gt->record_named_timer("scale", duration, m*n);
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
	    //y = x
	    cl->copy(write->device_data, reads[0]->device_data, reads[0]->MEM_SIZE);
	}

	void elementOp(std::string kernel_name, vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized){
	    int m = reads[0]->DIM_ROW;
	    int n = reads[0]->DIM_COL;
	    if (!is_initialized) {
	        write->reset(m, n, cl); //initialize LHS if not already
	    }
	    //y = x
	    cl->setup_kernel(kernel_name, 0, sizeof(cl_mem), &write->device_data); // Y
	    cl->setup_kernel(kernel_name, 1, sizeof(cl_mem), &reads[0]->device_data); // X
	    cl->setup_kernel(kernel_name, 2, sizeof(int), &(write->LEN)); //DATA_SIZE
	    cl_ulong duration = cl->exec_kernel(kernel_name, write->NUM_GLOBAL_WORKER, write->NUM_LOCAL_WORKER);
	    if(timed) gt->record_named_timer(kernel_name, duration, m*n);
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

	void multNN(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a*b", is_initialized);
		multMat(reads, write, is_initialized, "N", "N");
	}

	void multNT(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a*b", is_initialized);
		multMat(reads, write, is_initialized, "N", "T");
	}

	void multTN(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a*b", is_initialized);
		multMat(reads, write, is_initialized, "T", "N");
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
		elementOp("sigmoid", reads, write, is_initialized);
	}

	inline void sigmoid_gradient(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("sigmoid_gradient", is_initialized);
		elementOp("sigmoid_gradient", reads, write, is_initialized);
	}

	inline void sin(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("sin", is_initialized);
		elementOp("sin", reads, write, is_initialized);
	}

	inline void cos(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("cos", is_initialized);
		elementOp("cos", reads, write, is_initialized);
	}

	inline void tanh(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("tanh", is_initialized);
		elementOp("tanh", reads, write, is_initialized);
	}

	inline void tanh_gradient(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("tanh_gradient", is_initialized);
		elementOp("tanh_gradient", reads, write, is_initialized);
	}

	inline void element_mult(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("element_mult", is_initialized);
	    int m = reads[0]->DIM_ROW;
	    int n = reads[0]->DIM_COL;
	    if (!is_initialized) {
	        write->reset(m, n, cl); //initialize LHS if not already
	    }

	    //y = x
	    cl->setup_kernel("element_mult", 0, sizeof(cl_mem), &write->device_data); // Y
	    cl->setup_kernel("element_mult", 1, sizeof(cl_mem), &reads[0]->device_data); // X
	    cl->setup_kernel("element_mult", 2, sizeof(cl_mem), &reads[1]->device_data); // X
	    cl->setup_kernel("element_mult", 3, sizeof(int), &(write->LEN)); //DATA_SIZE
	    cl_ulong duration = cl->exec_kernel("element_mult", write->NUM_GLOBAL_WORKER, write->NUM_LOCAL_WORKER);
	    if(timed) gt->record_named_timer("element_mult", duration, m*n*2);
	}

	inline void square_loss(vector<OpenclFloatMatPtr> reads, float* write, bool is_initialized)
	{
		debug_msg("square_loss", is_initialized);
	    int m = reads[0]->DIM_ROW;
	    int n = reads[0]->DIM_COL;
	    OpenclFloatMat aux(m, n, cl);
	    //y = x
	    cl->setup_kernel("square_loss", 0, sizeof(cl_mem), &aux.device_data); // Y
	    cl->setup_kernel("square_loss", 1, sizeof(cl_mem), &reads[0]->device_data); // X
	    cl->setup_kernel("square_loss", 2, sizeof(cl_mem), &reads[1]->device_data); // X
	    cl->setup_kernel("square_loss", 3, sizeof(int), &(aux.LEN)); //DATA_SIZE
	    cl_ulong duration = cl->exec_kernel("square_loss", aux.NUM_GLOBAL_WORKER, aux.NUM_LOCAL_WORKER);
	    if(timed) gt->record_named_timer("square_loss", duration, m*n*2);

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
