/*
 * Eona Studio (c) 2015
 */


#ifndef CUDA_ENGINE_H_
#define CUDA_ENGINE_H_

#include "../../engine/engine.h"
#include "../../engine/tensor.h"
#include "../../rand_utils.h"
#include <cuda.h>
#include "cublas_v2.h"
#include "../types/cuda_float_mat.h"
#include "../types/performance_profiler.h"
#include "cuda_func.h"
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

	CudaEngine(GlobalTimer * g) :
		Engine<CudaFloatMat>()
	{
		gt = g;
		timed = true;
	    cublasCreate(&handle);

//	    if (timed) GPU_CHECKERROR(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));//for accurate timing
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


	typedef std::shared_ptr<CudaFloatMat> CudaFloatMatPtr;
	typedef std::shared_ptr<float> FloatPtr;

	void create(CudaFloatMatPtr write, vector<int> dim)
	{
		DEBUG_MSG("CudaImpl::create dim=" << dim);
		write->reset(dim);
	}

	void debug_msg(string msg, bool is_initialized)
	{
		DEBUG_MSG(("CudaImpl::" + msg + " ->init=") << std::boolalpha << is_initialized);
	}

	/*
	 * write = alpha * Op(reads[0]) + beta * Op(reads[1])
	 */
	void addMat(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized, float alpha, float beta)
	{

	    int m = reads[0]->DIM_ROW;
	    int n = reads[0]->DIM_COL;
	    if (!is_initialized) {
	    	write->reset(m, n);
	    }
		TIME("add", m*n*2,
		cublasSgeam(handle,
	                reads[0]->getOp(), reads[1]->getOp(),
	                m, n,
	                &alpha, reads[0]->device_data, reads[0]->LDIM,
	                &beta, reads[1]->device_data, reads[1]->LDIM,
	                write->device_data, write->LDIM)
	    );
	}


	/*
	 * write = alpha .* Op(reads[0]) * Op(reads[1]) + beta * write
	 */
	void multMat(vector<CudaFloatMatPtr> reads,
				CudaFloatMatPtr write, bool is_initialized,
				float alpha, float beta,
				std::string opA, std::string opB)
	{
	    int m = reads[0]->DIM_ROW;
	    int n = reads[0]->DIM_COL;
	    int l = reads[1]->DIM_ROW;
	    int k = reads[1]->DIM_COL;
	    if (!is_initialized) {
		    if (opA == "N" && opB == "N") write->reset(m, n); // A * B
		    if (opA == "N" && opB == "T") write->reset(m, l); // A * B^T
		    if (opA == "T" && opB == "N") write->reset(n, k); // A^T * B
		}
	    std::string name = "mult_"+opA+opB;
	    //C = a Op(A)* Op(B) + b C  -- A [mxn] B [nxk] C[mxk]
	    //handle, A_len, x, incx, y, incy
	    TIME(name, m*n+l*k,
		cublasSgemm(handle,
					reads[0]->getOp(opA), reads[1]->getOp(opB),
					m, n, k,
					&alpha, reads[0]->device_data, reads[0]->LDIM,
					reads[1]->device_data, reads[1]->LDIM, &beta,
					write->device_data, write->LDIM)
	    );
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
	    cublasScopy(handle, reads[0]->LEN, reads[0]->device_data, 1, write->device_data, 1)
	    );

	}

	void add(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a+b", is_initialized);
	    float alpha = 1.0f;
	    addMat(reads, write, is_initialized, alpha, alpha);
	}

	void sub(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a-b", is_initialized);

	    float alpha = 1.0f;
	    addMat(reads, write, is_initialized, alpha, -alpha);
	}

	void negate(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=-a", is_initialized);
	    //y = x
	    assignMat(reads, write, is_initialized);

	    //y = -y
	    const float alpha = -1.0f;
	    TIME("scale", write->LEN,
	    cublasSscal(handle, write->LEN, &alpha, write->device_data, 1);
	    )
	}

	void mult(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a*b", is_initialized);
		float alpha = 1.0f;
		multMat(reads, write, is_initialized, alpha, 0, "N", "N");
	}

	void assign(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
	    debug_msg("c=a", is_initialized);
	    assignMat(reads, write, is_initialized);
	}

	inline void scale(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized, float* scaler)
	{
		debug_msg("scale", is_initialized);
	    //y = x
	    assignMat(reads, write, is_initialized);
	    //y = ay
	    TIME("scale", write->LEN,
	    cublasSscal(handle, write->LEN, scaler, write->device_data, 1);
	    )
	}

	inline void destroy(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		debug_msg("destroy", is_initialized);
		reads[0]->free_data();
	}


	// standalone single-float non-linear functions
	inline void transpose(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		debug_msg("transpose", is_initialized);
		//TODO
	}



	#define MATOP(name, device_func) {\
			if (!is_initialized) {\
		    	write->reset(reads[0]->DIM_ROW, reads[0]->DIM_COL);\
			    cublasScopy(handle, reads[0]->LEN, reads[0]->device_data, 1, write->device_data, 1);\
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


	#define MATOP_DUAL(name, device_func) {\
			if (!is_initialized) {\
		    	write->reset(reads[0]->DIM_ROW, reads[0]->DIM_COL);\
			    cublasScopy(handle, reads[0]->LEN, reads[0]->device_data, 1, write->device_data, 1);\
			}\
			op_func_dual_t h_func;\
			cudaMemcpyFromSymbol( &h_func, device_func, sizeof( op_func_t ) );\
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

	inline void square_loss(vector<CudaFloatMatPtr> reads, float* write, bool is_initialized)
	{
		debug_msg("square_loss", is_initialized);
		CudaFloatMat aux(reads[0]->DIM_ROW, reads[0]->DIM_COL);

		cublasScopy(handle, reads[0]->LEN, reads[0]->device_data, 1, aux.device_data, 1);
		op_func_dual_t h_func;
		cudaMemcpyFromSymbol( &h_func, cu_square_loss_func, sizeof( op_func_t ) );
		mat_op_kernel<<<aux.GRID_DIM, aux.BLOCK_DIM>>>( aux.device_data,
														reads[0]->device_data,
														reads[1]->device_data,
														aux.LEN,
														h_func );

	    cublasSasum(handle, aux.LEN, aux.device_data, 1, write);
	}

	// FIXME add contextual rand engine
	inline void fill_rand(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		debug_msg("fill_rand", is_initialized);
		if (!is_initialized) {
	    	write->reset(reads[0]->DIM_ROW, reads[0]->DIM_COL);\
		}
		write->fill_rand(1);
	}


	/*********** DEBUG ONLY ***********/
	inline void debug_fill(vector<CudaFloatMatPtr> reads, CudaFloatMatPtr write, bool is_initialized)
	{
		if (!is_initialized) {
	    	write->reset(reads[0]->DIM_ROW, reads[0]->DIM_COL);\
		}
		write->fill(0.66337);
	}

private:
	cublasHandle_t handle;
	bool timed;
	GlobalTimer * gt;

};

#endif /* CUDA_ENGINE_H_ */
