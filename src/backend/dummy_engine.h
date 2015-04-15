/*
 * Eona Studio (c) 2015
 */


#ifndef DUMMY_ENGINE_H_
#define DUMMY_ENGINE_H_

#include "../engine/engine.h"
#include "../engine/tensor.h"
#include "../rand_utils.h"

namespace lmn { //what is this?

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

void create(float* write, vector<int> dim)
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
	DEBUG_MSG("rand? " << *write);
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
		register_create(Impl::create);
		register_opcode("t+t", Impl::add<T>);
		register_opcode("s+s", Impl::add<S>);
		register_opcode("t-t", Impl::sub<T>);
		register_opcode("s-s", Impl::sub<S>);
		register_opcode("-t", Impl::negate<T>);
		register_opcode("-s", Impl::negate<S>);
		register_opcode("t*t", Impl::mult<T, T>);
		register_opcode("t*s", Impl::mult<T, S>);
		register_opcode("s*t", Impl::mult<S, T>);
		register_opcode("s*s", Impl::mult<S, S>);
		register_opcode("t=t", Impl::assign<T>);
		register_opcode("s=s", Impl::assign<S>);

		register_opcode("sin", Impl::sin);
		register_opcode("cos", Impl::cos);
		register_opcode("tanh", Impl::tanh);
		register_opcode("tanh_gradient", Impl::tanh_gradient);
		register_opcode("sigmoid", Impl::sigmoid);
		register_opcode("sigmoid_gradient", Impl::sigmoid_gradient);
		register_opcode("transpose", Impl::transpose);
		register_opcode("element_mult", Impl::element_mult);
		register_opcode("square_loss", Impl::square_loss);

		register_opcode("destroy", Impl::destroy);
		register_opcode("fill_rand", Impl::fill_rand);

		/*********** DEBUG ONLY ***********/
		register_opcode("debug_fill", Impl::debug_fill);
	}
};

#endif /* DUMMY_ENGINE_H_ */
