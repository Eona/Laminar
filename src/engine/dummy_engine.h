/*
 * Eona Studio (c) 2015
 */


#ifndef DUMMY_ENGINE_H_
#define DUMMY_ENGINE_H_

#include "engine.h"
#include "tensor.h"

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

void create(float* write, vector<int> dim)
{
	DEBUG_MSG("DummyImpl::create dim=" << dim);
	*write = 0;
}

template<int TensorT>
void add(vector<float*> reads, float* write, bool is_initialized)
{
	string op = tensor_op<TensorT>::operand;
	DEBUG_MSG(("DummyImpl::" + op + "+" + op + " ->init=") << std::boolalpha << is_initialized);
	*write = *reads[0] + *reads[1];
}

template<int TensorT>
void sub(vector<float*> reads, float* write, bool is_initialized)
{
	string op = tensor_op<TensorT>::operand;
	DEBUG_MSG(("DummyImpl::" + op + "-" + op + " ->init=") << std::boolalpha << is_initialized);
	*write = *reads[0] - *reads[1];
}

template<int TensorT1, int TensorT2>
void mult(vector<float*> reads, float* write, bool is_initialized)
{
	string op1 = tensor_op<TensorT1>::operand;
	string op2 = tensor_op<TensorT2>::operand;
	DEBUG_MSG(("DummyImpl::" + op1 + "*" + op2 + " ->init=") << std::boolalpha << is_initialized);
	*write = (*reads[0]) * (*reads[1]);
}

template<int TensorT>
void assign(vector<float*> reads, float* write, bool is_initialized)
{
	string op = tensor_op<TensorT>::operand;
	DEBUG_MSG(("DummyImpl::" + op + "=" + op + " ->init=") << std::boolalpha << is_initialized);
	*write = *reads[0];
}

void destroy(vector<float*> reads, float* write, bool is_initialized)
{
	DEBUG_MSG("DummyImpl::destroy ->init=" << std::boolalpha << is_initialized);
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
		register_opcode("t*t", Impl::mult<T, T>);
		register_opcode("t*s", Impl::mult<T, S>);
		register_opcode("s*t", Impl::mult<S, T>);
		register_opcode("s*s", Impl::mult<S, S>);
		register_opcode("t=t", Impl::assign<T>);
		register_opcode("s=s", Impl::assign<S>);
		register_opcode("destroy", Impl::destroy);
	}
};

#endif /* DUMMY_ENGINE_H_ */
