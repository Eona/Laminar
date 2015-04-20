/*
 * Eona Studio (c) 2015
 */

#ifndef BACKEND_VECMAT_VECMAT_ENGINE_H_
#define BACKEND_VECMAT_VECMAT_ENGINE_H_

#include "../../engine/engine.h"
#include "../../engine/tensor.h"
#include "../../engine/tensor_ops.h"
#include "../vecmat/vecmat.h"

#define VECTORMAT_DEBUG false

namespace lmn {

typedef Vecmat<float> Vecmatf;
typedef std::shared_ptr<Vecmatf> VecmatfPtr;

namespace VecmatImpl {

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

void create(VecmatfPtr write, Dimension dim)
{
#if VECTORMAT_DEBUG
	DEBUG_MSG("VectorMat::create dim=" << dim);
#endif
	write->new_zeros(dim[0], dim[1]);
}

void debug_msg(string msg, bool is_initialized)
{
#if VECTORMAT_DEBUG
	DEBUG_MSG(("VectorMat::" + msg + " ->init=") << std::boolalpha << is_initialized);
#endif
}

template<int TensorT>
void add(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	string op = tensor_op<TensorT>::operand;
	debug_msg(op + "+" + op, is_initialized);

	if (!is_initialized)
		// WARNING in unoptimized version, x += a translates to
		// x = x + a, where reads[0] == write == x and reads[1] == a
		// if x is null created (as often in layer gradient += ops)
		// reads[0] will be empty. So here 'write' should be initialized with reads[1] size.
 		write->new_zeros(reads[1]);

	*write = *reads[0] + *reads[1];
}

template<int TensorT>
void sub(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	string op = tensor_op<TensorT>::operand;
	debug_msg(op + "-" + op, is_initialized);

	if (!is_initialized)
		write->new_zeros(reads[0]);

	*write = *reads[0] - *reads[1];
}

template<int TensorT>
void negate(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	string op = tensor_op<TensorT>::operand;
	debug_msg("-" + op, is_initialized);

	if (!is_initialized)
		write->new_zeros(reads[0]);

	*write = - (*reads[0]);
}

template<int TensorT1, int TensorT2>
void mult(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	string op1 = tensor_op<TensorT1>::operand;
	string op2 = tensor_op<TensorT2>::operand;
	debug_msg(op1 + "*" + op2, is_initialized);

	if (!is_initialized)
		write->new_zeros(reads[0]->row(), reads[1]->col());

	*write = (*reads[0]) * (*reads[1]);
}

void scale(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized, float scalorContext)
{
	debug_msg("scale * " + to_str(scalorContext), is_initialized);

	if (!is_initialized)
		write->new_zeros(reads[0]);

	*write = reads[0]->scale(scalorContext);
}

template<int TensorT>
void assign(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	string op = tensor_op<TensorT>::operand;
	debug_msg(op + "=" + op, is_initialized);

	if (!is_initialized)
		write->new_zeros(reads[0]);

	*write = *reads[0];
}

inline void destroy(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	debug_msg("destroy", is_initialized);
}


// standalone single-float non-linear functions
inline void transpose(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	debug_msg("transpose", is_initialized);

	if (!is_initialized)
		write->new_zeros(reads[0]->col(), reads[0]->row());

	*write = reads[0]->transpose();
}

/**
 * Helper for element-wise unary function application
 */
template<typename UnaryFunc>
inline void element_apply(VecmatfPtr read, VecmatfPtr write, UnaryFunc func)
{
	Vecmatf& rmat = *read;
	Vecmatf& wmat = *write;

	wmat.assert_same_dim(rmat, "element_apply read VS write addr");

	for (int r = 0; r < rmat.row(); ++r)
		for (int c = 0; c < rmat.col(); ++c)
			wmat(r, c) = func(rmat(r, c));
}

inline void sigmoid(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	debug_msg("sigmoid", is_initialized);

	if (!is_initialized)
		write->new_zeros(reads[0]);

	element_apply(reads[0], write,
			[](float x) { return 1.f / (1.f + exp(-x)); });
}

inline void sigmoid_gradient(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	debug_msg("sigmoid_gradient", is_initialized);

	if (!is_initialized)
		write->new_zeros(reads[0]);

	element_apply(reads[0], write,
			[](float x) { return x * (1.f - x); });
}

inline void sin(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	debug_msg("sin", is_initialized);

	if (!is_initialized)
		write->new_zeros(reads[0]);

	element_apply(reads[0], write,
			static_cast<float (*)(float)>(std::sin));
}

inline void cos(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	debug_msg("cos", is_initialized);

	if (!is_initialized)
		write->new_zeros(reads[0]);

	element_apply(reads[0], write,
			static_cast<float (*)(float)>(std::cos));
}

inline void tanh(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	debug_msg("tanh", is_initialized);

	if (!is_initialized)
		write->new_zeros(reads[0]);

	element_apply(reads[0], write,
			static_cast<float (*)(float)>(std::tanh));
}

inline void tanh_gradient(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	debug_msg("tanh_gradient", is_initialized);

	if (!is_initialized)
		write->new_zeros(reads[0]);

	element_apply(reads[0], write,
			[](float x) { return 1.f - x * x; });
}

inline void element_mult(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	debug_msg("element_mult", is_initialized);

	if (!is_initialized)
		write->new_zeros(reads[0]);

	Vecmatf& rmat1 = *reads[0];
	Vecmatf& rmat2 = *reads[1];
	Vecmatf& wmat = *write;

	rmat1.assert_same_dim(rmat2, "element_mult reads[0] VS reads[1] addr");
	wmat.assert_same_dim(rmat2, "element_mult reads[1] VS write addr");

	for (int r = 0; r < wmat.row(); ++r)
		for (int c = 0; c < wmat.col(); ++c)
			wmat(r, c) = rmat1(r, c) * rmat2(r, c);
}

/**
 * @param reads
 * @param write a scalor that's the sum of all square differences
 * @param is_initialized
 */
inline void square_loss(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	debug_msg("square_loss", is_initialized);

	if (!is_initialized)
		write->new_zeros(1, 1);

	Vecmatf& rmat1 = *reads[0];
	Vecmatf& rmat2 = *reads[1];
	Vecmatf& wmat = *write;

	rmat1.assert_same_dim(rmat2, "square_loess reads[0] VS reads[1] addr");

	wmat(0, 0) = 0;
	for (int r = 0; r < rmat1.row(); ++r)
		for (int c = 0; c < rmat1.col(); ++c)
		{
			float diff = rmat1(r, c) - rmat2(r, c);
			wmat(0, 0) += 0.5f * diff * diff;
		}
}

inline void zero_clear(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	debug_msg("zero_clear", is_initialized);

	// FIXME loss layer output might be zero cleared without being initialized
//	assert_throw<EngineException>(is_initialized,
//		"VecmatEngine: calling zero_clear on uninitialized write addr");

	if (is_initialized)
		write->zero_clear();
}

inline void set_value(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized,
		DimIndex idx, float val)
{
	debug_msg("set_value: " + container2str(idx) + "=" + to_str(val), is_initialized);

	assert_throw<EngineException>(is_initialized,
		"VecmatEngine: calling set_value on uninitialized write addr");

	write->at(idx) = val;
}


// FIXME add contextual rand engine
inline void fill_rand(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	assert_throw<EngineException>(is_initialized,
		"VecmatEngine: calling fill_rand on uninitialized write addr");

	debug_msg("fill_rand", is_initialized);

	write->fill([](int i, int j) {
		return FakeRand::instance_connection()();
	});
}

inline void fill_rand_prehistory(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	assert_throw<EngineException>(is_initialized,
		"VecmatEngine: calling fill_rand_prehistory on uninitialized write addr");

	debug_msg("fill_rand_prehistory", is_initialized);

	write->fill([](int i, int j) {
		return FakeRand::instance_prehistory()();
	});
}

// For gradient checking
inline void perturb(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized,
		DimIndex idx, float eps)
{
	debug_msg("perturb", is_initialized);
	assert_throw<EngineException>(is_initialized,
		"VecmatEngine: calling perturb on uninitialized write addr");

	write->at(idx) += eps;
}


} // end of DummyImpl::
} // end of lmn::


class VecmatEngine :
	public Engine<lmn::Vecmatf>,
	public ElementInspection<lmn::Vecmatf, float>
{
public:
	VecmatEngine() :
		Engine()
	{
		namespace Impl = lmn::VecmatImpl;
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
//		register_normal_op("t*s", Impl::mult<T, S>);
//		register_normal_op("s*t", Impl::mult<S, T>);
//		register_normal_op("s*s", Impl::mult<S, S>);
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
		register_normal_op("zero_clear", Impl::zero_clear);

		register_normal_op("fill_rand", Impl::fill_rand);
		register_normal_op("fill_rand_prehistory", Impl::fill_rand_prehistory);
		register_context_op<DimIndex, float>("perturb", Impl::perturb);
		register_context_op<DimIndex, float>("set_value", Impl::set_value);
		register_context_op<float>("scale", Impl::scale);
	}

	float element_at(lmn::VecmatfPtr vecmat, DimIndex idx)
	{
		assert_throw<EngineException>(!vecmat->is_empty(),
			"VecmatEngine: element_at() called on null matrix");

		return vecmat->at(idx);
	}
};


#endif /* BACKEND_VECMAT_VECMAT_ENGINE_H_ */
