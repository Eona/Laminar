/*
 * Eona Studio (c) 2015
 */

#ifndef CUDA_ENGINE_H_
#define CUDA_ENGINE_H_

#include "../../utils/global_utils.h"
#include "../../utils/laminar_utils.h"
#include "../../utils/rand_utils.h"
#include "../../engine/engine.h"
#include "../../engine/tensor.h"
#include "../../engine/tensor_ops.h"
#include <cuda.h>
#include "cublas_v2.h"
#include "../types/cuda_float_mat.h"
#include "../types/performance_profiler.h"
#include "cuda_func.h"
#include <assert.h>
using namespace std;

#define TIME(name, size, func_call) {\
		CudaTimer t(name, gt, size);\
		if(timed) {\
			t.start();\
		}\
		func_call;\
		if(timed) {\
			t.stop();\
		}\
}


class CudaEngine : public Engine<CudaFloatMat>
{
public:
	CudaEngine() :
		Engine<CudaFloatMat>()
	{
		gt = NULL;
		timed = false;
//		log_mem = false;
		init();

	}

	CudaEngine(GlobalTimer<cudaEvent_t> * g) :
		Engine<CudaFloatMat>()
	{
		gt = g;
		timed = true;
//		log_mem = false;
		init();
	}

//	CudaEngine(GlobalTimer<cudaEvent_t> * g, MemoryMonitor * m) :
//		Engine<CudaFloatMat>()
//	{
//		gt = g;
//		mm = m;
//		timed = true;
//		log_mem = true;
//		init();
//	}

	void init() {
	    cublasCreate(&handle);
		register_create_op(MEMFUNC_BIND_2(CudaEngine::create));
		register_normal_op("t+t", MEMFUNC_BIND_3(CudaEngine::add));
		register_normal_op("t-t", MEMFUNC_BIND_3(CudaEngine::sub));
		register_normal_op("s+s", MEMFUNC_BIND_3(CudaEngine::add_scalor));
		register_normal_op("-t", MEMFUNC_BIND_3(CudaEngine::negate));
		register_normal_op("t*t", MEMFUNC_BIND_3(CudaEngine::multNN));
		register_normal_op("t*s", MEMFUNC_BIND_3(CudaEngine::multTS));
		register_normal_op("s*t", MEMFUNC_BIND_3(CudaEngine::multST));
		register_normal_op("s*s", MEMFUNC_BIND_3(CudaEngine::multSS));
		register_normal_op("t=t", MEMFUNC_BIND_3(CudaEngine::assign));
		register_context_op<float>("s=const", MEMFUNC_BIND_4(CudaEngine::assign_const));

		register_normal_op("sin", MEMFUNC_BIND_3(CudaEngine::sin));
		register_normal_op("cos", MEMFUNC_BIND_3(CudaEngine::cos));
		register_normal_op("tanh", MEMFUNC_BIND_3(CudaEngine::tanh));
		register_normal_op("tanh_gradient", MEMFUNC_BIND_3(CudaEngine::tanh_gradient));
		register_normal_op("sigmoid", MEMFUNC_BIND_3(CudaEngine::sigmoid));
		register_normal_op("sigmoid_gradient", MEMFUNC_BIND_3(CudaEngine::sigmoid_gradient));
		register_normal_op("transpose", MEMFUNC_BIND_3(CudaEngine::transpose));
		register_normal_op("element_mult", MEMFUNC_BIND_3(CudaEngine::element_mult));
		register_normal_op("square_loss", MEMFUNC_BIND_3(CudaEngine::square_loss));

		register_normal_op("destroy", MEMFUNC_BIND_3(CudaEngine::destroy));
		register_normal_op("zero_clear", MEMFUNC_BIND_3(CudaEngine::zero_clear));

		register_normal_op("fill_rand", MEMFUNC_BIND_3(CudaEngine::fill_rand));
		register_normal_op("fill_rand_prehistory", MEMFUNC_BIND_3(CudaEngine::fill_rand));
		register_normal_op("softmax", MEMFUNC_BIND_3(CudaEngine::softmax));
		register_normal_op("label_entropy_loss", MEMFUNC_BIND_3(CudaEngine::label_entropy_loss));
		register_normal_op("label_softmax_entropy_gradient", MEMFUNC_BIND_3(CudaEngine::label_softmax_entropy_gradient));

		register_context_op<float>("scale", MEMFUNC_BIND_4(CudaEngine::scale));

		register_context_op<DimIndex, float>("perturb", MEMFUNC_BIND_5(CudaEngine::perturb));

		register_context_op<lmn::ElementFillFunc<float>>(
						"fill_element", MEMFUNC_BIND_4(CudaEngine::fill_element));

	}

	typedef std::shared_ptr<CudaFloatMat> CudaFloatMatPtr;
	typedef std::shared_ptr<float> FloatPtr;

#define CHECK2() {\
	assert(reads[0]->DIM_ROW == reads[1]->DIM_ROW && \
		   reads[0]->DIM_COL == reads[1]->DIM_COL && \
		   reads[0]->DIM_ROW == write->DIM_ROW &&\
		   reads[0]->DIM_COL == write->DIM_COL); \
}

#define CHECK1() {\
		assert(reads[0]->DIM_ROW == write->DIM_ROW &&\
				reads[0]->DIM_COL == write->DIM_COL); \
}

//Execute kernel function of the form Y = op(X)
#define MATOP(name, device_func) {\
		if (!is_initialized) {\
	    	write->reset(reads[0]->DIM_ROW, reads[0]->DIM_COL);\
		}\
		op_func_t h_func;\
		cudaMemcpyFromSymbol( &h_func, device_func, sizeof( op_func_t ) );\
		CudaTimer t(name, gt, reads[0]->DIM_ROW * reads[0]->DIM_COL);\
		if(timed) {\
			t.start();\
		}\
		mat_op_kernel<<<write->GRID_DIM, write->BLOCK_DIM>>>( write->device_data, \
															  reads[0]->device_data, \
															  write->LEN, \
															  h_func ); \
	    if(timed) {\
	    	t.stop();\
	    }\
}

//Execute kernel function of the form C = op(A,B)
#define MATOP_DUAL(name, device_func) {\
		if (!is_initialized) {\
	    	write->reset(reads[0]->DIM_ROW, reads[0]->DIM_COL);\
		}\
		op_func_dual_t h_func;\
		cudaMemcpyFromSymbol( &h_func, device_func, sizeof( op_func_dual_t ) );\
		CudaTimer t(name, gt, reads[0]->DIM_ROW * reads[0]->DIM_COL * 2);\
		if(timed) {\
			t.start();\
		}\
		mat_op_kernel<<<write->GRID_DIM, write->BLOCK_DIM>>>( write->device_data, \
															  reads[0]->device_data, \
															  reads[1]->device_data, \
															  write->LEN, \
															  h_func ); \
		if(timed) {\
			t.stop();\
		}\
}


	void create(CudaFloatMatPtr write, vector<int> dim)
	{
		// DEBUG_MSG("CudaCudaEngine::create dim=" << dim);
		write->reset(dim);
	}

	void zero_clear(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		if (write->isScalar)
			write->scalar = 0;
		else
		{
			LMN_ASSERT_THROW(is_initialized,
				EngineException("CUDA zero_clear must have been inited"));
			//__global__ void mat_scale_kernel(float *target, float alpha, int N)
			mat_fill_kernel<<<write->GRID_DIM, write->BLOCK_DIM>>>( write->device_data,
																  	 0,
																  	 write->LEN
																   );
		}
	}

	void debug_msg(string msg, bool is_initialized)
	{
		// DEBUG_MSG(("CudaCudaEngine::" + msg + " ->init=") << std::boolalpha << is_initialized);
	}

	/*


	/*
	 * write = alpha .* Op(reads[0]) * Op(reads[1]) + beta * write
	 */
	void multMat(vector<CudaFloatMatPtr> reads,
				CudaFloatMatPtr write, bool is_initialized,
				float alpha, float beta,
				std::string opA, std::string opB)
	{
	    int m = reads[0]->DIM_ROW;
	    int k = reads[0]->DIM_COL;
	    int l = reads[1]->DIM_ROW;
	    int n = reads[1]->DIM_COL;

	    if (opA == "N" && opB == "N")
	    	LMN_ASSERT_THROW(k == l,
	    		EngineException("multMat dim mismatch "
	    				+ container2str(Dimension{m, k}) + " <-> "
						+ container2str(Dimension{l, n})));
	    if (opA == "N" && opB == "T")
	    	LMN_ASSERT_THROW(k == n,
	    		EngineException("multMat dim mismatch "
	    				+ container2str(Dimension{m, k}) + " <-> "
						+ container2str(Dimension{n, l})));
	    if (opA == "T" && opB == "N")
	    	LMN_ASSERT_THROW(m == l,
	    		EngineException("multMat dim mismatch "
	    				+ container2str(Dimension{k, m}) + " <-> "
						+ container2str(Dimension{l, n})));

	    if (!is_initialized) {
		    if (opA == "N" && opB == "N") write->reset(m, n); // A * B
		    if (opA == "N" && opB == "T") write->reset(m, l); // A * B^T
		    if (opA == "T" && opB == "N") write->reset(k, n); // A^T * B
		}
	    std::string name = "mult_"+opA+opB;
	    //C = a Op(A)* Op(B) + b C  -- A [mxk] B [kxn] C[mxn]
	    //handle, A_len, x, incx, y, incy

	    dim3 thread_per_block (16,16);
	    dim3 num_blocks (ceil(double(write->DIM_COL)/double(thread_per_block.x)),
	                    ceil(double(write->DIM_ROW)/double(thread_per_block.y)));

		CudaTimer t(name, gt, m*k+l*n);
		if(timed) t.start();
		if (opA == "N" && opB == "N") {
			mat_multNN_shared_kernel<<<num_blocks, thread_per_block>>>
					(write->device_data, write->DIM_ROW, write->DIM_COL,
							reads[0]->device_data, reads[0]->DIM_ROW, reads[0]->DIM_COL,
							reads[1]->device_data, reads[1]->DIM_ROW, reads[1]->DIM_COL
					);
		}

		if (opA == "T" && opB == "N") {
			//	    	TIME(name, m*k+l*n,
			mat_multTN_shared_kernel<<<num_blocks, thread_per_block>>>
					(write->device_data, write->DIM_ROW, write->DIM_COL,
							reads[0]->device_data, reads[0]->DIM_ROW, reads[0]->DIM_COL,
							reads[1]->device_data, reads[1]->DIM_ROW, reads[1]->DIM_COL
					);
			//	    		);
		}

		if (opA == "N" && opB == "T") {
			//	    	TIME(name, m*k+l*n,
			mat_multNT_shared_kernel<<<num_blocks, thread_per_block>>>
					(write->device_data, write->DIM_ROW, write->DIM_COL,
							reads[0]->device_data, reads[0]->DIM_ROW, reads[0]->DIM_COL,
							reads[1]->device_data, reads[1]->DIM_ROW, reads[1]->DIM_COL
					);
			//	    		);
		}
    	if(timed)  t.stop();
	}

	/*
	 * assign reads[0] to write
	 */
	void assignMat(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
	    int m = reads[0]->DIM_ROW;
	    int n = reads[0]->DIM_COL;
	    if (!is_initialized) {
	    	write->reset(m, n);
	    }

	    //y = x
	    //handle, x_len, x, incx, y, incy
	    TIME("assign", m*n,
	    reads[0]->copy_to_device(write->device_data);
	    );
	    // FIX: otherwise transpose dimension error
	    write->DIM_ROW = reads[0]->DIM_ROW;
	    write->DIM_COL = reads[0]->DIM_COL;
	    write->LDIM = reads[0]->LDIM;
	    CHECK1();
	}

	//helper func
	void scaleMat(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized, float scalar)
	{
	    assignMat(reads, write, is_initialized); // copy to write
	    //y = -y
		CudaTimer t("scale", gt, reads[0]->DIM_ROW * reads[0]->DIM_COL);
		if(timed) t.start();
		//__global__ void mat_scale_kernel(float *target, float alpha, int N)
		mat_scale_kernel<<<write->GRID_DIM, write->BLOCK_DIM>>>( write->device_data,
															  	 scalar,
															  	 write->LEN
															   );
		if(timed) t.stop();
	    CHECK1();
	}
	void add(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a+b", is_initialized);
	    MATOP_DUAL("add", cu_add_func);
	    CHECK2();
	}

	void add_scalor(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		reads[0]->isScalar = true;
		reads[1]->isScalar = true;
		write->isScalar = true;
		write->scalar = reads[0]->scalar + reads[1]->scalar;
	}

	void sub(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a-b", is_initialized);
	    MATOP_DUAL("subtract", cu_subtract_func);
	    CHECK2();
	}


	void negate(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=-a", is_initialized);
	    //y = x
	    scaleMat(reads, write, is_initialized, -1);
	}

	void multNN(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a*b", is_initialized);
		float alpha = 1.0f;
		multMat(reads, write, is_initialized, alpha, 0, "N", "N");
	}

	void multNT(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a*b", is_initialized);
		float alpha = 1.0f;
		multMat(reads, write, is_initialized, alpha, 0, "N", "T");
	}

	void multTN(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a*b", is_initialized);
		float alpha = 1.0f;
		multMat(reads, write, is_initialized, alpha, 0, "T", "N");
	}

	void assign(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a", is_initialized);
	    assignMat(reads, write, is_initialized);
	}

	void assign_const(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized, float constant){
	    debug_msg("c=constS", is_initialized);

	    write->isScalar = true;
	    write->scalar = constant;
	}

	void multST(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		LMN_ASSERT_THROW(reads[0]->isScalar,
				EngineException("reads[0] in s*t must be scalar"));
		// WARNING tensor is the SECOND arg of reads!!!
		scale({ reads[1] }, write, is_initialized, reads[0]->scalar);
	}

	void multTS(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		LMN_ASSERT_THROW(reads[1]->isScalar,
				EngineException("reads[1] in t*s must be scalar"));
		scale(reads, write, is_initialized, reads[1]->scalar);
	}

	void multSS(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		LMN_ASSERT_THROW(reads[0]->isScalar && reads[1]->isScalar,
				EngineException("reads[0] and [1] in s*s must both be scalar"));

		write->isScalar = true;
		write->scalar = reads[0]->scalar * reads[1]->scalar;
	}

	inline void scale(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized, float scalar)
	{
		debug_msg("scale", is_initialized);
		//y = ay
	    scaleMat(reads, write, is_initialized, scalar);
	}

	inline void destroy(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		debug_msg("destroy", is_initialized);
//		reads[0]->free_data();
	}


	// standalone single-float non-linear functions
	inline void transpose(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		debug_msg("transpose", is_initialized);
	    assignMat(reads, write, is_initialized);
	    write->local_transpose();
	}

	inline void sigmoid(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		debug_msg("sigmoid", is_initialized);
		MATOP("sigmoid", cu_sigmoid_func);
	}

	inline void sigmoid_gradient(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		debug_msg("sigmoid_gradient", is_initialized);
		MATOP("sigmoid_gradient", cu_sigmoid_gradient_func);
	}

	inline void sin(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		debug_msg("sin", is_initialized);
		MATOP("sin", cu_sin_func);
	}

	inline void cos(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		debug_msg("cos", is_initialized);
		MATOP("cos", cu_cos_func);
	}

	inline void tanh(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		debug_msg("tanh", is_initialized);
		MATOP("tanh", cu_tanh_func);
	}

	inline void tanh_gradient(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		debug_msg("tanh_gradient", is_initialized);
		MATOP("tanh_gradient", cu_tanh_gradient_func);
	}

	inline void element_mult(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		debug_msg("element_mult", is_initialized);
	    MATOP_DUAL("element_mult", cu_element_mult_func);
	}

	inline void square_loss(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		debug_msg("square_loss", is_initialized);

		CudaFloatMat aux(reads[0]->DIM_ROW, reads[0]->DIM_COL);
		op_func_dual_t h_func;
		cudaMemcpyFromSymbol( &h_func, cu_square_loss_func, sizeof( op_func_t ) );
		mat_op_kernel<<<aux.GRID_DIM, aux.BLOCK_DIM>>>( aux.device_data,
														reads[0]->device_data,
														reads[1]->device_data,
														aux.LEN,
														h_func );
//		cudaDeviceSynchronize();
//		float sum = 0;
//		mat_sum_kernel<<<aux.GRID_DIM, aux.BLOCK_DIM>>>( aux.device_data,
//														 sum,
//														 aux.LEN
//													   );
	    cublasSasum(handle, aux.LEN, aux.device_data, 1, &write->scalar);
		write->isScalar = true;
	}

	inline void softmax(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized) {

		if (!is_initialized)
	    	write->reset(reads[0]->DIM_ROW, reads[0]->DIM_COL);

		vector<float> rmat(write->LEN);
		vector<float> wmat(write->LEN);
		int m = write->DIM_ROW;
		int n = write->DIM_COL;
		reads[0]->to_host(&rmat[0]);

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

		write->to_device(&wmat[0]);
	}

	/**
	 * -log(value_at_label)
	 * @param reads a tensor of int class labels (faked as floats)
	 * @param write a scalor loss
	 */
	inline void label_entropy_loss(
			vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		debug_msg("label_entropy_loss", is_initialized);

		write->isScalar = true;

		vector<float> rmat(reads[0]->LEN);
		vector<float> labels(reads[1]->LEN);
		reads[0]->to_host(&rmat[0]);
		reads[1]->to_host(&labels[0]);

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
			vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		debug_msg("label_softmax_entropy_gradient", is_initialized);

		int m = reads[0]->DIM_ROW;
		int n = reads[0]->DIM_COL;
		if (!is_initialized)
			write->reset(m, n);

		vector<float> rmat(reads[0]->LEN);
		vector<float> labels(reads[1]->LEN);
		vector<float> wmat(write->LEN);

		reads[0]->to_host(&rmat[0]);
		reads[0]->to_host(&wmat[0]);// copy most values won't change
		reads[1]->to_host(&labels[0]);

		for (int c = 0; c < n; ++c)
		{
			int label = (int) labels[c];
			wmat[label + c * reads[0]->DIM_ROW] -= 1.f; // y - t (sparse)
		}
		write->to_device(&wmat[0]);
	}


	void fill_element(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized,
			lmn::ElementFillFunc<float> filler)
	{
		debug_msg("fill_element", is_initialized);

		assert(is_initialized);
		int m = write->DIM_ROW;
		int n = write->DIM_COL;
		vector<float> t(write->LEN);

		for (int i = 0; i < m; ++i) { // which row
			for (int j = 0; j < n; ++j) { //which col
				t[i + j * m] = filler(DimIndex {i, j});
			}
		}
		write->to_device(&t[0]);
	}

	// FIXME add contextual rand engine
	inline void fill_rand(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		debug_msg("fill_rand", is_initialized);
		if (!is_initialized) {
	    	write->reset(reads[0]->DIM_ROW, reads[0]->DIM_COL);\
		}
		write->fill_rand(DEBUG_SEED);
	}


	/*********** DEBUG ONLY ***********/
	inline void debug_fill(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		if (!is_initialized) {
	    	write->reset(reads[0]->DIM_ROW, reads[0]->DIM_COL);\
		}
		write->fill(0.66337);
	}

	inline void perturb(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized,
			DimIndex idx, float eps)
	{
		debug_msg("perturb", is_initialized);

		size_t i = idx[1] * write->DIM_ROW + idx[0]; //c*dim_row + r
		write->perturb(i, eps);
	}

	float tensor_data_at(CudaFloatMatPtr reads, DimIndex idx) {
		int m = reads->DIM_ROW;
		int n = reads->DIM_COL;
		int i = m*idx[1] + idx[0];
		float d;
		reads->take_at(&d, i, 1);
		return d;
	}

	float scalar_data_at(CudaFloatMatPtr reads)
	{
		LMN_ASSERT_THROW(reads->isScalar,
				EngineException("read in scalar_data_at must be scalar"));
		return reads->scalar;
	}

	TYPEDEF_PTR(CudaEngine);

private:
	cublasHandle_t handle;
	bool timed;
	GlobalTimer<cudaEvent_t> * gt;

//	bool log_mem;
//	MemoryMonitor * mm;
};

#endif /* CUDA_ENGINE_H_ */
