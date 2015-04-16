/*
 * Eona Studio (c) 2015
 */


#ifndef DUMMY_ENGINE_H_
#define DUMMY_ENGINE_H_

#include "engine.h"
#include "tensor.h"
#include "../rand_utils.h"

namespace lmn {

namespace DummyImpl {

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

void create(float* write, Dimension dim)
{
	DEBUG_MSG("DummyImpl::create dim=" << dim);
	*write = 0;
}

void debug_msg(string msg, bool is_initialized)
{
	DEBUG_MSG(("DummyImpl::" + msg + " ->init=") << std::boolalpha << is_initialized);
}

template<int TensorT>
void add(vector<float*> reads, float* write, bool is_initialized)
{
	string op = tensor_op<TensorT>::operand;
	debug_msg(op + "+" + op, is_initialized);
	*write = *reads[0] + *reads[1];
}

template<int TensorT>
void sub(vector<float*> reads, float* write, bool is_initialized)
{
	string op = tensor_op<TensorT>::operand;
	debug_msg(op + "-" + op, is_initialized);
	*write = *reads[0] - *reads[1];
}

template<int TensorT>
void negate(vector<float*> reads, float* write, bool is_initialized)
{
	string op = tensor_op<TensorT>::operand;
	debug_msg("-" + op, is_initialized);
	*write = - (*reads[0]);
}

template<int TensorT1, int TensorT2>
void mult(vector<float*> reads, float* write, bool is_initialized)
{
	string op1 = tensor_op<TensorT1>::operand;
	string op2 = tensor_op<TensorT2>::operand;
	debug_msg(op1 + "*" + op2, is_initialized);
	*write = (*reads[0]) * (*reads[1]);
}

template<int TensorT>
void assign(vector<float*> reads, float* write, bool is_initialized)
{
	string op = tensor_op<TensorT>::operand;
	debug_msg(op + "=" + op, is_initialized);
	*write = *reads[0];
}

inline void destroy(vector<float*> reads, float* write, bool is_initialized)
{
	debug_msg("destroy", is_initialized);
}


// standalone single-float non-linear functions
inline void transpose(vector<float *> reads, float *write, bool is_initialized)
{
	debug_msg("transpose", is_initialized);
	float r = *reads[0];
	*write = *reads[0];
}

inline void sigmoid(vector<float *> reads, float *write, bool is_initialized)
{
	debug_msg("sigmoid", is_initialized);
	float r = *reads[0];
	*write = 1.f / (1.f + exp(-r));
}

inline void sigmoid_gradient(vector<float *> reads, float *write, bool is_initialized)
{
	debug_msg("sigmoid_gradient", is_initialized);
	float r = *reads[0];
	*write = r * (1.f - r);
}

inline void sin(vector<float *> reads, float *write, bool is_initialized)
{
	debug_msg("sin", is_initialized);
	float r = *reads[0];
	*write = std::sin(r);
}

inline void cos(vector<float *> reads, float *write, bool is_initialized)
{
	debug_msg("cos", is_initialized);
	float r = *reads[0];
	*write = std::cos(r);
}

inline void tanh(vector<float *> reads, float *write, bool is_initialized)
{
	debug_msg("tanh", is_initialized);
	float r = *reads[0];
	*write = std::tanh(r);
}

inline void tanh_gradient(vector<float *> reads, float *write, bool is_initialized)
{
	debug_msg("tanh_gradient", is_initialized);
	float r = *reads[0];
	*write = 1.f - r * r;
}

inline void element_mult(vector<float *> reads, float *write, bool is_initialized)
{
	debug_msg("element_mult", is_initialized);
	*write = (*reads[0]) * (*reads[1]);
}

inline void square_loss(vector<float *> reads, float *write, bool is_initialized)
{
	debug_msg("square_loss", is_initialized);
	float diff = *reads[0] - *reads[1];
	*write = 0.5f * diff * diff;
}

// FIXME add contextual rand engine
inline void fill_rand(vector<float *> reads, float *write, bool is_initialized)
{
	debug_msg("fill_rand", is_initialized);
	*write = FakeRand::instance_connection()();
}

// For gradient checking
inline void perturb(vector<float *> reads, float *write, bool is_initialized,
		DimIndex idx, float eps)
{
	debug_msg("perturb", is_initialized);
	*write += eps;
}

/*********** DEBUG ONLY ***********/
inline void debug_fill(vector<float *> reads, float *write, bool is_initialized)
{
	debug_msg("debug_fill", is_initialized);
	*write = 0.66337;
}



} // end of DummyImpl::
} // end of lmn::

class DummyEngine : public Engine<float>
{
public:
	DummyEngine() :
		Engine<float>()
	{
		namespace Impl = lmn::DummyImpl;
		const int T = Impl::TENSOR;
		const int S = Impl::SCALOR;

		register_create_op(Impl::create);
		register_normal_op("t+t", Impl::add<T>);
		register_normal_op("s+s", Impl::add<S>);
		register_normal_op("t-t", Impl::sub<T>);
		register_normal_op("s-s", Impl::sub<S>);
		register_normal_op("-t", Impl::negate<T>);
		register_normal_op("-s", Impl::negate<S>);
		register_normal_op("t*t", Impl::mult<T, T>);
		register_normal_op("t*s", Impl::mult<T, S>);
		register_normal_op("s*t", Impl::mult<S, T>);
		register_normal_op("s*s", Impl::mult<S, S>);
		register_normal_op("t=t", Impl::assign<T>);
		register_normal_op("s=s", Impl::assign<S>);

		register_normal_op("sin", Impl::sin);
		register_normal_op("cos", Impl::cos);
		register_normal_op("tanh", Impl::tanh);
		register_normal_op("tanh_gradient", Impl::tanh_gradient);
		register_normal_op("sigmoid", Impl::sigmoid);
		register_normal_op("sigmoid_gradient", Impl::sigmoid_gradient);
		register_normal_op("transpose", Impl::transpose);
		register_normal_op("element_mult", Impl::element_mult);
		register_normal_op("square_loss", Impl::square_loss);

		register_normal_op("destroy", Impl::destroy);
		register_normal_op("fill_rand", Impl::fill_rand);

		register_context_op<DimIndex, float>("perturb", Impl::perturb);

		/*********** DEBUG ONLY ***********/
		register_normal_op("debug_fill", Impl::debug_fill);
	}
};

#endif /* DUMMY_ENGINE_H_ */
