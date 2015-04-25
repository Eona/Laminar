/*
 * Eona Studio (c) 2015
 */


#ifndef DUMMY_ENGINE_H_
#define DUMMY_ENGINE_H_

#include "../../engine/engine.h"
#include "../../engine/tensor.h"
#include "../../utils/rand_utils.h"

#define DUMMY_DEBUG false

namespace lmn {

typedef std::shared_ptr<float> FloatPtr;

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

void create(FloatPtr write, Dimension dim)
{
#if DUMMY_DEBUG
	DEBUG_MSG("Dummy::create dim=" << dim);
#endif
	*write = 0;
}

void debug_msg(string msg, bool is_initialized)
{
#if DUMMY_DEBUG
	DEBUG_MSG(("Dummy::" + msg + " ->init=") << std::boolalpha << is_initialized);
#endif
}

template<int TensorT>
void add(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	string op = tensor_op<TensorT>::operand;
	debug_msg(op + "+" + op, is_initialized);
	*write = *reads[0] + *reads[1];
}

template<int TensorT>
void sub(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	string op = tensor_op<TensorT>::operand;
	debug_msg(op + "-" + op, is_initialized);
	*write = *reads[0] - *reads[1];
}

template<int TensorT>
void negate(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	string op = tensor_op<TensorT>::operand;
	debug_msg("-" + op, is_initialized);
	*write = - (*reads[0]);
}

template<int TensorT1, int TensorT2>
void mult(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	string op1 = tensor_op<TensorT1>::operand;
	string op2 = tensor_op<TensorT2>::operand;
	debug_msg(op1 + "*" + op2, is_initialized);
	*write = (*reads[0]) * (*reads[1]);
}

void scale(vector<FloatPtr> reads, FloatPtr write, bool is_initialized, float scalarContext)
{
	debug_msg("scale * " + to_str(scalarContext), is_initialized);
	*write = scalarContext * (*reads[0]);
}

template<int TensorT>
void assign(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	string op = tensor_op<TensorT>::operand;
	debug_msg(op + "=" + op, is_initialized);
	*write = *reads[0];
}

/**
 * Assign a constant float to a Scalar
 * OpContext<float>
 */
void assign_const(vector<FloatPtr> reads, FloatPtr write, bool is_initialized, float constant)
{
	debug_msg("s=const", is_initialized);
	*write = constant;
}

inline void destroy(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	debug_msg("destroy", is_initialized);
}


// standalone single-float non-linear functions
inline void transpose(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	debug_msg("transpose", is_initialized);
	float r = *reads[0];
	*write = *reads[0];
}

inline void sigmoid(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	debug_msg("sigmoid", is_initialized);
	float r = *reads[0];
	*write = 1.f / (1.f + exp(-r));
}

inline void sigmoid_gradient(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	debug_msg("sigmoid_gradient", is_initialized);
	float r = *reads[0];
	*write = r * (1.f - r);
}

inline void sin(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	debug_msg("sin", is_initialized);
	float r = *reads[0];
	*write = std::sin(r);
}

inline void cos(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	debug_msg("cos", is_initialized);
	float r = *reads[0];
	*write = std::cos(r);
}

inline void tanh(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	debug_msg("tanh", is_initialized);
	float r = *reads[0];
	*write = std::tanh(r);
}

inline void tanh_gradient(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	debug_msg("tanh_gradient", is_initialized);
	float r = *reads[0];
	*write = 1.f - r * r;
}

inline void element_mult(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	debug_msg("element_mult", is_initialized);
	*write = (*reads[0]) * (*reads[1]);
}

inline void square_loss(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	debug_msg("square_loss", is_initialized);
	float diff = *reads[0] - *reads[1];
	*write = 0.5f * diff * diff;
}

inline void zero_clear(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	debug_msg("zero_clear", is_initialized);
	*write = 0;
}

inline void set_value(vector<FloatPtr> reads, FloatPtr write, bool is_initialized,
		DimIndex idx, float val)
{
	debug_msg("set_value: " + container2str(idx) + "=" + to_str(val), is_initialized);
	*write = val;
}


// FIXME add contextual rand engine
inline void fill_rand(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	LMN_ASSERT_THROW(is_initialized,
		EngineException("DummyEngine: fill_rand must have been initialized"));
	debug_msg("fill_rand", is_initialized);
	*write = FakeRand::instance_connection()();
}

inline void fill_rand_prehistory(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	LMN_ASSERT_THROW(is_initialized,
		EngineException("DummyEngine: fill_rand_prehistory must have been initialized"));
	debug_msg("fill_rand_prehistory", is_initialized);
	*write = FakeRand::instance_prehistory()();
}

inline void fill_element(vector<FloatPtr> reads, FloatPtr write, bool is_initialized,
		lmn::ElementFillFunc<float> filler)
{
	LMN_ASSERT_THROW(is_initialized,
		EngineException("DummyEngine: fill_element must have been initialized"));

	debug_msg("fill_element", is_initialized);

	*write = filler({0});
}

// For gradient checking
inline void perturb(vector<FloatPtr> reads, FloatPtr write, bool is_initialized,
		DimIndex idx, float eps)
{
	debug_msg("perturb", is_initialized);
	*write += eps;
}

/*********** DEBUG ONLY ***********/

inline void debug_context_tmp(vector<FloatPtr> reads, FloatPtr write, bool is_initialized, string x, float y, std::pair<char, int> z)
{
	DEBUG_MSG("DEBUG_CONTEXT executed: "
		<< "string=" << x << " float=" << y
		<< " pair=<" << z.first << ", " << z.second << ">");
}

} // end of DummyImpl::
} // end of lmn::

class DummyEngine :
	public Engine<float>
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
		register_context_op<float>("s=const", Impl::assign_const);

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
		register_normal_op("zero_clear", Impl::zero_clear);

		register_normal_op("fill_rand", Impl::fill_rand);
		register_normal_op("fill_rand_prehistory", Impl::fill_rand_prehistory);
		register_context_op<lmn::ElementFillFunc<float>>(
						"fill_element", Impl::fill_element);

		register_context_op<DimIndex, float>("perturb", Impl::perturb);
		register_context_op<DimIndex, float>("set_value", Impl::set_value);
		register_context_op<float>("scale", Impl::scale);

		/*********** DEBUG ONLY ***********/
		register_context_op<string, float, std::pair<char, int>>("debug_context_tmp", Impl::debug_context_tmp);
	}

	/**
	 * Implements element retrieval
	 */
	float tensor_data_at(lmn::FloatPtr f, DimIndex)
	{
		return *f;
	}

	float scalar_data_at(lmn::FloatPtr f)
	{
		return *f;
	}
};

#endif /* DUMMY_ENGINE_H_ */
