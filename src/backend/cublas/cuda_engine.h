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
#include "cudaFloatMat.h"
#include "cuda_func.h"
using namespace std;


class CublasHandleInstance{
public:
//    static cublasHandle_t & Instance()
//    {
//        if (!init) {
//            cublasCreate(&handle);
//            init = true;
//        }
//        return handle;
//    }
//
//    void destroy()
//    {
//        if (init) {
//            cublasDestroy(handle);
//        }
//    }

    CublasHandleInstance(){
    	cublasCreate(&handle);
    	printf("initialized!\n");
    }
    static cublasHandle_t handle;

private:

    static bool init;
};

cublasHandle_t & cublasHandleInstance() {
    static cublasHandle_t handle;
    cublasCreate(&handle);
    return handle;
}



namespace lmn {

namespace CudaImpl {

enum TensorT {
	TENSOR = 0,
	SCALOR = 1
};




template<int TensorT>
struct tensor_op {};

template<>
struct tensor_op<TENSOR>
{
	static constexpr const char *operand = "t";
};

template<>
struct tensor_op<SCALOR>
{
	static constexpr const char *operand = "s";
};

void create(CudaFloatMat* write, vector<int> dim)
{
	DEBUG_MSG("CudaImpl::create dim=" << dim);
	*write = CudaFloatMat(dim);
}

void debug_msg(string msg, bool is_initialized)
{
	DEBUG_MSG(("CudaImpl::" + msg + " ->init=") << std::boolalpha << is_initialized);
}


/*
 * write = alpha * Op(reads[0]) + beta * Op(reads[1])
 */
void addMat(vector<CudaFloatMat*> reads, CudaFloatMat* write, bool is_initialized, float alpha, float beta) 
{

    int m = reads[0]->DIM_ROW;
    int n = reads[0]->DIM_COL;
    if (!is_initialized) {
        *write = CudaFloatMat(m, n); //initialize LHS if not already
    }   
    
    cublasSgeam(cublasHandleInstance(),
                reads[0]->getOp(), reads[1]->getOp(),
                m, n,  
                &alpha, reads[0]->device_data, reads[0]->LDIM, 
                &beta, reads[1]->device_data, reads[1]->LDIM,
                write->device_data, write->LDIM);
}


/*
 * write = alpha .* Op(reads[0]) * Op(reads[1]) + beta * write
 */
void multMat(vector<CudaFloatMat*> reads,
			CudaFloatMat* write, bool is_initialized,
			float alpha, float beta,
			std::string opA, std::string opB)
{
    int m = reads[0]->DIM_ROW;
    int n = reads[0]->DIM_COL;
    int k = reads[1]->DIM_COL;
    if (!is_initialized) {
        *write = CudaFloatMat(m, k); //initialize LHS if not already
    }

    //C = a Op(A)* Op(B) + b C  -- A [mxn] B [nxk] C[mxk]
    //handle, A_len, x, incx, y, incy
    cublasSgemm(cublasHandleInstance(),
                reads[0]->getOp(opA), reads[1]->getOp(opB),
                m, n, k,
                &alpha, reads[0]->device_data, reads[0]->LDIM,
                reads[1]->device_data, reads[1]->LDIM, &beta,
                write->device_data, write->LDIM);
}

/*
 * assign reads[0] to write
 */
void assignMat(vector<CudaFloatMat*> reads, CudaFloatMat* write, bool is_initialized)
{
    int m = reads[0]->DIM_ROW;
    int n = reads[0]->DIM_COL;
    if (!is_initialized) {
        *write = CudaFloatMat(m, n); //initialize LHS if not already
    }
    //y = x
    //handle, x_len, x, incx, y, incy
    cublasScopy(cublasHandleInstance(), reads[0]->LEN, reads[0]->device_data, 1, write->device_data, 1);
}


template<int TensorT>
void add(vector<CudaFloatMat*> reads, CudaFloatMat* write, bool is_initialized)
{
    string op = tensor_op<TensorT>::operand;
    debug_msg(op + "+" + op, is_initialized);
    float alpha = 1.0f;
    addMat(reads, write, is_initialized, alpha, alpha);
}

template<int TensorT>
void sub(vector<CudaFloatMat*> reads, CudaFloatMat* write, bool is_initialized)
{
	string op = tensor_op<TensorT>::operand;
	debug_msg(op + "-" + op, is_initialized);
    float alpha = 1.0f;
    addMat(reads, write, is_initialized, alpha, -alpha);
}

template<int TensorT>
void negate(vector<CudaFloatMat*> reads, CudaFloatMat* write, bool is_initialized)
{
	string op = tensor_op<TensorT>::operand;
	debug_msg("-" + op, is_initialized);

    //y = x
    assignMat(reads, write, is_initialized);

    //y = -y
    const float alpha = -1.0f;
    cublasSscal(cublasHandleInstance(), write->LEN, &alpha, write->device_data, 1);

}

template<int TensorT1, int TensorT2>
void mult(vector<CudaFloatMat*> reads, CudaFloatMat* write, bool is_initialized)
{
	string op1 = tensor_op<TensorT1>::operand;
	string op2 = tensor_op<TensorT2>::operand;
	debug_msg(op1 + "*" + op2, is_initialized);

	float alpha = 1.0f;
	multMat(reads, write, is_initialized, alpha, 0, "N", "N");
}

template<int TensorT>
void assign(vector<CudaFloatMat*> reads, CudaFloatMat* write, bool is_initialized)
{
	string op = tensor_op<TensorT>::operand;
	debug_msg(op + "=" + op, is_initialized);

    assignMat(reads, write, is_initialized);
}


inline void destroy(vector<CudaFloatMat*> reads, CudaFloatMat* write, bool is_initialized)
{
	debug_msg("destroy", is_initialized);
	reads[0]->free_data();
}


// standalone single-float non-linear functions
inline void transpose(vector<CudaFloatMat*> reads, CudaFloatMat* write, bool is_initialized)
{
	debug_msg("transpose", is_initialized);
	//TODO
}

#define MATOP(device_func) {\
		if (!is_initialized) *write = CudaFloatMat(reads[0]->DIM_ROW, reads[0]->DIM_COL);\
		cublasScopy(cublasHandleInstance(), reads[0]->LEN, reads[0]->device_data, 1, write->device_data, 1);\
		op_func_t h_func;\
		cudaMemcpyFromSymbol( &h_func, device_func, sizeof( op_func_t ) );\
		mat_op_kernel<<<write->GRID_DIM, write->BLOCK_DIM>>>( write->device_data, \
															  reads[0]->device_data, \
															  h_func ); \
}


#define MATOP_DUAL(device_func) {\
		if (!is_initialized) *write = CudaFloatMat(reads[0]->DIM_ROW, reads[0]->DIM_COL);\
		cublasScopy(cublasHandleInstance(), reads[0]->LEN, reads[0]->device_data, 1, write->device_data, 1);\
		op_func_dual_t h_func;\
		cudaMemcpyFromSymbol( &h_func, device_func, sizeof( op_func_t ) );\
		mat_op_kernel<<<write->GRID_DIM, write->BLOCK_DIM>>>( write->device_data, \
															  reads[0]->device_data, \
															  reads[1]->device_data, \
															  h_func ); \
}


inline void sigmoid(vector<CudaFloatMat*> reads, CudaFloatMat* write, bool is_initialized)
{
	debug_msg("sigmoid", is_initialized);
	MATOP(cu_sigmoid_func);
}

inline void sigmoid_gradient(vector<CudaFloatMat*> reads, CudaFloatMat* write, bool is_initialized)
{
	debug_msg("sigmoid_gradient", is_initialized);
	MATOP(cu_sigmoid_gradient_func);
}

inline void sin(vector<CudaFloatMat*> reads, CudaFloatMat* write, bool is_initialized)
{
	debug_msg("sin", is_initialized);
	MATOP(cu_sin_func);
}

inline void cos(vector<CudaFloatMat*> reads, CudaFloatMat* write, bool is_initialized)
{
	debug_msg("cos", is_initialized);
	MATOP(cu_cos_func);
}

inline void tanh(vector<CudaFloatMat*> reads, CudaFloatMat* write, bool is_initialized)
{
	debug_msg("tanh", is_initialized);
	MATOP(cu_tanh_func);
}

inline void tanh_gradient(vector<CudaFloatMat*> reads, CudaFloatMat* write, bool is_initialized)
{
	debug_msg("tanh_gradient", is_initialized);
	MATOP(cu_tanh_gradient_func);
}

inline void element_mult(vector<CudaFloatMat*> reads, CudaFloatMat* write, bool is_initialized)
{
	debug_msg("element_mult", is_initialized);
    MATOP_DUAL(cu_element_mult_func);
}

inline void square_loss(vector<CudaFloatMat*> reads, CudaFloatMat* write, bool is_initialized)
{
	debug_msg("square_loss", is_initialized);
    MATOP_DUAL(cu_square_loss_func);
}

// FIXME add contextual rand engine
inline void fill_rand(vector<CudaFloatMat*> reads, CudaFloatMat* write, bool is_initialized)
{
	debug_msg("fill_rand", is_initialized);
	//*write = FakeRand::instance_connection()();
	//DEBUG_MSG("rand? " << *write);
}


/*********** DEBUG ONLY ***********/
inline void debug_fill(vector<CudaFloatMat*> reads, CudaFloatMat* write, bool is_initialized)
{
	debug_msg("debug_fill", is_initialized);
	//*write = 0.66337;
}



} // end of DummyImpl::
} // end of lmn::



//class CudaEngine : public Engine<float>
//{
//public:
//	CudaEngine() :
//		Engine<float>()
//	{
//		namespace Impl = lmn::CudaImpl;
//		const int T = Impl::TENSOR;
//		const int S = Impl::SCALOR;
//		register_create(Impl::create);
//		register_opcode("t+t", Impl::add<T>);
//		register_opcode("s+s", Impl::add<S>);
//		register_opcode("t-t", Impl::sub<T>);
//		register_opcode("s-s", Impl::sub<S>);
//		register_opcode("-t", Impl::negate<T>);
//		register_opcode("-s", Impl::negate<S>);
//		register_opcode("t*t", Impl::mult<T, T>);
//		register_opcode("t*s", Impl::mult<T, S>);
//		register_opcode("s*t", Impl::mult<S, T>);
//		register_opcode("s*s", Impl::mult<S, S>);
//		register_opcode("t=t", Impl::assign<T>);
//		register_opcode("s=s", Impl::assign<S>);
//
//		register_opcode("sin", Impl::sin);
//		register_opcode("cos", Impl::cos);
//		register_opcode("tanh", Impl::tanh);
//		register_opcode("tanh_gradient", Impl::tanh_gradient);
//		register_opcode("sigmoid", Impl::sigmoid);
//		register_opcode("sigmoid_gradient", Impl::sigmoid_gradient);
//		register_opcode("transpose", Impl::transpose);
//		register_opcode("element_mult", Impl::element_mult);
//		register_opcode("square_loss", Impl::square_loss);
//
//		register_opcode("destroy", Impl::destroy);
//		register_opcode("fill_rand", Impl::fill_rand);
//
//		/*********** DEBUG ONLY ***********/
//		register_opcode("debug_fill", Impl::debug_fill);
//	}
//};

#endif /* CUDA_ENGINE_H_ */
