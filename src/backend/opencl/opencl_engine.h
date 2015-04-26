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
#include "../../utils/laminar_utils.h"
#include "../../utils/global_utils.h"
using namespace std;


typedef std::shared_ptr<OpenclFloatMat> OpenclFloatMatPtr;

class OpenclEngine : public Engine<OpenclFloatMat>
{
public:

	OpenclEngine() :
		Engine<OpenclFloatMat>()
	{
		timed = false; //if profile time performance
		init(); //must call as the last line
	}

	OpenclEngine(GlobalTimer<cl_event> * g) :
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
		cl->register_kernel("dummy", "matop_prog", "dummy");
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
		NUM_LOCAL_WORKER = cl->query_group_size();

		register_create_op(MEMFUNC_BIND_2(OpenclEngine::create));
		register_normal_op("t+t", MEMFUNC_BIND_3(OpenclEngine::add));
		register_normal_op("t-t", MEMFUNC_BIND_3(OpenclEngine::sub));
		register_normal_op("-t", MEMFUNC_BIND_3(OpenclEngine::negate));
		register_normal_op("t*t", MEMFUNC_BIND_3(OpenclEngine::multNN));
		register_normal_op("t*s", MEMFUNC_BIND_3(OpenclEngine::multTS));
		register_normal_op("s*t", MEMFUNC_BIND_3(OpenclEngine::multST));
		register_normal_op("t=t", MEMFUNC_BIND_3(OpenclEngine::assign));
		register_context_op<float>("s=const", MEMFUNC_BIND_4(OpenclEngine::assign_const));

		register_normal_op("sin", MEMFUNC_BIND_3(OpenclEngine::sin));
		register_normal_op("cos", MEMFUNC_BIND_3(OpenclEngine::cos));
		register_normal_op("tanh", MEMFUNC_BIND_3(OpenclEngine::tanh));
		register_normal_op("tanh_gradient", MEMFUNC_BIND_3(OpenclEngine::tanh_gradient));
		register_normal_op("sigmoid", MEMFUNC_BIND_3(OpenclEngine::sigmoid));
		register_normal_op("sigmoid_gradient", MEMFUNC_BIND_3(OpenclEngine::sigmoid_gradient));
		register_normal_op("transpose", MEMFUNC_BIND_3(OpenclEngine::transpose));
		register_normal_op("element_mult", MEMFUNC_BIND_3(OpenclEngine::element_mult));
		register_normal_op("square_loss", MEMFUNC_BIND_3(OpenclEngine::square_loss));

		register_normal_op("destroy", MEMFUNC_BIND_3(OpenclEngine::destroy));
		register_normal_op("zero_clear", MEMFUNC_BIND_3(OpenclEngine::zero_clear));
		register_normal_op("soft_max", MEMFUNC_BIND_3(OpenclEngine::soft_max));
		register_normal_op("label_entropy_loss", MEMFUNC_BIND_3(OpenclEngine::label_entropy_loss));
		register_normal_op("label_softmax_entropy_gradient", MEMFUNC_BIND_3(OpenclEngine::label_softmax_entropy_gradient));


		register_normal_op("fill_rand", MEMFUNC_BIND_3(OpenclEngine::fill_rand));
		register_context_op<float>("scale", MEMFUNC_BIND_4(OpenclEngine::scale));
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
	    cl_ulong duration = cl->exec_kernel("add", cl->get_global_size(write->LEN, NUM_LOCAL_WORKER), NUM_LOCAL_WORKER);

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
		    if (opA == "N" && opB == "N") write->reset(m, k, cl); // A * B
		    if (opA == "N" && opB == "T") write->reset(m, l, cl); // A * B^T
		    if (opA == "T" && opB == "N") write->reset(n, k, cl); // A^T * B
	    }

	    //C = a Op(A)* Op(B) + b C  -- A [mxn] B [lxk]
	    //Need to re-compute number of workers
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
	    cl->setup_kernel("scale", 0, sizeof(cl_mem), &write->device_data); // Y
	    cl->setup_kernel("scale", 1, sizeof(cl_mem), &reads[0]->device_data); // X
	    cl->setup_kernel("scale", 2, sizeof(float), &alpha); //a
	    cl->setup_kernel("scale", 3, sizeof(int), &(write->LEN)); //DATA_SIZE
	    cl_ulong duration = cl->exec_kernel("scale", cl->get_global_size(write->LEN, NUM_LOCAL_WORKER), NUM_LOCAL_WORKER);

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
	    cl_ulong duration = cl->exec_kernel(kernel_name, cl->get_global_size(write->LEN, NUM_LOCAL_WORKER), NUM_LOCAL_WORKER);
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
	void assign_const(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized, float constant){
	    debug_msg("c=constS", is_initialized);
	    write->scalar = constant;
	}

	void multST(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		scale(reads, write, is_initialized, reads[0]->scalar);
	}

	void multTS(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		scale(reads, write, is_initialized, reads[1]->scalar);
	}
	void assign(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a", is_initialized);
	    assignMat(reads, write, is_initialized);
	}

	inline void scale(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized, float scaler)
	{
		debug_msg("scale", is_initialized);
	    //y = ay
	    scaleMat(reads, write, is_initialized, scaler);
	}

	inline void destroy(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("destroy", is_initialized);
//		reads[0]->free_data();
	}


	// standalone single-float non-linear functions
	inline void transpose(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("transpose", is_initialized);
		//TODO
	    assignMat(reads, write, is_initialized);
	    write->local_transpose();
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
	    cl_ulong duration = cl->exec_kernel("element_mult", cl->get_global_size(write->LEN, NUM_LOCAL_WORKER), NUM_LOCAL_WORKER);
	    if(timed) gt->record_named_timer("element_mult", duration, m*n*2);
	}

	inline void square_loss(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
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
	    cl_ulong duration = cl->exec_kernel("square_loss", cl->get_global_size(aux.LEN, NUM_LOCAL_WORKER), NUM_LOCAL_WORKER);
	    if(timed) gt->record_named_timer("square_loss", duration, m*n*2);

	    float t[aux.MEM_SIZE];
	    aux.to_host(t);

	    for (int i = 0; i < aux.LEN; ++i) {
	    	write->scalar += t[i];
	    }
	}

	void zero_clear(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		write->zero_clear();
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

	inline void soft_max(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized) {

		if (!is_initialized)
	    	write->reset(reads[0]->DIM_ROW, reads[0]->DIM_COL, cl);

		float* rmat = new float[write->LEN];
		float* wmat = new float[write->LEN];
		int m = write->DIM_ROW;
		int n = write->DIM_COL;
		reads[0]->to_host(rmat);

		// Each column is a data feature vector
		// coldim is batch size


		for (int c = 0; c < n; ++c) //each col
		{
			// find max
			float mx = -1e20f;
			for (int r = 0; r < m; ++r)
				if (rmat[c*m + r] > mx)
					mx = rmat[c*m + r];

			// exp(a - mx) for all 'a'
			for (int r = 0; r < m; ++r)
				wmat[c*m + r] = std::exp(rmat[c*m + r] - mx);

			// sum last step
			float sum = 0;
			for (int r = 0; r < m; ++r)
				sum += wmat[c*m + r];

			// divide every wmat col element by sum
			for (int r = 0; r < m; ++r)
				wmat[c*m + r] /= sum;
		}

		write->to_device(wmat);

		delete [] rmat;
		delete [] wmat;
	}

	/**
	 * -log(value_at_label)
	 * @param reads a tensor of int class labels (faked as floats)
	 * @param write a scalor loss
	 */
	inline void label_entropy_loss(
			vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("label_entropy_loss", is_initialized);

		if (!is_initialized)
			write->reset(1, 1, cl);

		float * rmat = new float[reads[0]->LEN];
		float * labels = new float[reads[1]->LEN];
		reads[0]->to_host(rmat);
		reads[1]->to_host(labels);

		write->scalar = 0;
		for (int c = 0; c < reads[0]->DIM_COL; ++c)
		{
			int label = (int) labels[c];
			// value at label:
			write->scalar -= std::log(rmat[label + c * reads[0]->DIM_ROW]);
		}
	}

	/**
	 *
	 * @param reads y, vector *after* softmax
	 * @param write y - t, where t is a sparse vector with a single '1' at the correct label
	 * @param is_initialized
	 */
	inline void label_softmax_entropy_gradient(
			vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		debug_msg("label_softmax_entropy_gradient", is_initialized);

		int m = reads[0]->DIM_ROW;
		int n = reads[0]->DIM_COL;
		if (!is_initialized)
			write->reset(m, n, cl);

		float * rmat = new float[reads[0]->LEN];
		float * labels = new float[reads[1]->LEN];
		float * wmat = new float[write->LEN];

		reads[0]->to_host(rmat);
		reads[0]->to_host(wmat);// copy most values won't change
		reads[1]->to_host(labels);

		for (int c = 0; c < n; ++c)
		{
			int label = (int) labels[c];
			wmat[label + c * reads[0]->DIM_ROW] -= 1.f; // y - t (sparse)
		}

		write->to_device(wmat);
	}

	/*********** DEBUG ONLY ***********/
	inline void debug_fill(vector<OpenclFloatMatPtr> reads, OpenclFloatMatPtr write, bool is_initialized)
	{
		if (!is_initialized) {
	        write->reset(write->DIM_ROW, write->DIM_COL, cl); //initialize LHS if not already
		}
		write->fill(0.66337);
	}


	float tensor_data_at(OpenclFloatMatPtr reads, DimIndex idx) {
		int m = reads->DIM_ROW;
		int n = reads->DIM_COL;
		int i = m*idx[1] + idx[0];
		float d;
		reads->take_at(&d, i, 1);
		return d;
	}

	float scalar_data_at(OpenclFloatMatPtr reads) {
		return reads->scalar;
	}


	~OpenclEngine()
	{
		delete cl;
	}
	OclUtilContext* cl;

private:
	bool timed;
	GlobalTimer<cl_event> * gt;
	size_t NUM_LOCAL_WORKER;
};

#endif /* OPENCL_ENGINE_H_ */
