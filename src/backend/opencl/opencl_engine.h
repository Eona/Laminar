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

#include "../../engine/engine.h"
#include "../../engine/tensor.h"
#include "../../rand_utils.h"

using namespace std;


class OpenclEngine : public Engine<OpenclFloatMat>
{
public:

	OpenclEngine() :
		Engine<OpenclFloatMat>()
	{
		cl = new OclUtilContext(true);
		cout<<"created context"<<endl;
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


	typedef std::shared_ptr<OpenclFloatMat> OpenclFloatMatPtr;
	typedef std::shared_ptr<float> FloatPtr;

	void create(OpenclFloatMatPtr write, vector<int> dim)
	{
		DEBUG_MSG("CudaImpl::create dim=" << dim);
		*write = OpenclFloatMat(dim, cl);
	}

	void debug_msg(string msg, bool is_initialized)
	{
		DEBUG_MSG(("CudaImpl::" + msg + " ->init=") << std::boolalpha << is_initialized);
	}

	/*
	 * write = alpha * Op(reads[0]) + beta * Op(reads[1])
	 */
	void addMat(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized, float alpha, float beta)
	{

	    int m = reads[0]->DIM_ROW;
	    int n = reads[0]->DIM_COL;
	    if (!is_initialized) {
	        *write = OpenclFloatMat(m, n, cl); //initialize LHS if not already
	    }
	}


	/*
	 * write = alpha .* Op(reads[0]) * Op(reads[1]) + beta * write
	 */
	void multMat(vector<OpenclFloatMatPtr> reads,
				OpenclFloatMatPtr write, bool is_initialized,
				float alpha, float beta,
				std::string opA, std::string opB)
	{
	    int m = reads[0]->DIM_ROW;
	    int n = reads[0]->DIM_COL;
	    int k = reads[1]->DIM_COL;
	    if (!is_initialized) {
	        *write = OpenclFloatMat(m, n, cl); //initialize LHS if not already
	    }

	    //C = a Op(A)* Op(B) + b C  -- A [mxn] B [nxk] C[mxk]
	}

	/*
	 * assign reads[0] to write
	 */
	void assignMat(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
	    int m = reads[0]->DIM_ROW;
	    int n = reads[0]->DIM_COL;
	    if (!is_initialized) {
	        *write = OpenclFloatMat(m, n, cl); //initialize LHS if not already
	    }
	    //y = x
	    cl->copy(write->device_data, reads[0]->device_data, reads[0]->MEM_SIZE);
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
	    //y = x
	    assignMat(reads, write, is_initialized);

	    //y = -y
	    const float alpha = -1.0f;

	}

	void mult(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a*b", is_initialized);
		float alpha = 1.0f;
		multMat(reads, write, is_initialized, alpha, 0, "N", "N");
	}

	void assign(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a", is_initialized);
	    assignMat(reads, write, is_initialized);
	}

	inline void scale(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized, float* scaler)
	{
		debug_msg("scale", is_initialized);
	    //y = x
	    assignMat(reads, write, is_initialized);
	    //y = ay
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
	}

	inline void sigmoid_gradient(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("sigmoid_gradient", is_initialized);
	}

	inline void sin(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("sin", is_initialized);
	}

	inline void cos(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("cos", is_initialized);
	}

	inline void tanh(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("tanh", is_initialized);
	}

	inline void tanh_gradient(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("tanh_gradient", is_initialized);
	}

	inline void element_mult(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("element_mult", is_initialized);
	}

	inline void square_loss(vector<OpenclFloatMatPtr> reads, float* write, bool is_initialized)
	{
		debug_msg("square_loss", is_initialized);

	}

	// FIXME add contextual rand engine
	inline void fill_rand(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("fill_rand", is_initialized);
		if (!is_initialized) {
			*write = OpenclFloatMat(reads[0]->DIM_ROW, reads[0]->DIM_COL, cl);
		}
		write->fill_rand(1);
	}


	/*********** DEBUG ONLY ***********/
	inline void debug_fill(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		if (!is_initialized) {
			*write = OpenclFloatMat(reads[0]->DIM_ROW, reads[0]->DIM_COL, cl);
		}
		write->fill(0.66337);
	}

	~OpenclEngine()
	{
		delete cl;
	}
	OclUtilContext* cl;

private:

};

#endif /* OPENCL_ENGINE_H_ */
