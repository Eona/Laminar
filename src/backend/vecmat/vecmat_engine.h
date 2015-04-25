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

class VecmatEngineException: public EngineException {
public:
    VecmatEngineException(const std::string& msg):
    	EngineException(msg)
	{}

    virtual std::string error_header() const
    {
    	return "VecmatEngine error";
    }
};

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

void debug_assert_init(string msg, bool is_initialized)
{
	LMN_ASSERT_THROW(is_initialized,
		VecmatEngineException("calling "+msg+" on uninitialized write addr"));
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

void mult_t_t(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	debug_msg("t*t", is_initialized);

	if (!is_initialized)
		write->new_zeros(reads[0]->row(), reads[1]->col());

	*write = (*reads[0]) * (*reads[1]);
}

void mult_t_s(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	debug_msg("t*s", is_initialized);

	if (!is_initialized)
		write->new_zeros(reads[0]);

	LMN_ASSERT_THROW(reads[1]->dim() == (Dimension {1, 1}),
		VecmatEngineException("t*s Scalar reads[1] should have dimension [1, 1]\n"
				"But now it's " + container2str(reads[1]->dim())));

	*write = reads[0]->scale(reads[1]->at({0, 0}));
}

void mult_s_t(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	debug_msg("s*t", is_initialized);

	if (!is_initialized)
		write->new_zeros(reads[1]);

	LMN_ASSERT_THROW(reads[0]->dim() == (Dimension {1, 1}),
		VecmatEngineException("s*t Scalar reads[0] should have dimension [1, 1]\n"
				"But now it's " + container2str(reads[0]->dim())));

	*write = reads[1]->scale(reads[0]->at({0, 0}));
}

void scale(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized, float scalarContext)
{
	debug_msg("scale * " + to_str(scalarContext), is_initialized);

	LMN_ASSERT_THROW(!reads[0]->is_empty(),
		VecmatEngineException("calling scale on unintialized reads[0] addr"));

	if (!is_initialized)
		write->new_zeros(reads[0]);

	*write = reads[0]->scale(scalarContext);
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

/**
 * Assign a float constant to Scalar
 */
void assign_const(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized, float constant)
{
	debug_msg("s=const", is_initialized);

	if (!is_initialized)
		write->new_zeros(1, 1);

	write->at({0, 0}) = constant;
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
 * @param write a scalar that's the sum of all square differences
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

	rmat1.assert_same_dim(rmat2, "square_loss reads[0] VS reads[1] addr");

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
//	debug_assert_init("zero_clear", is_initialized);

	if (is_initialized)
		write->zero_clear();
}

inline void set_value(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized,
		DimIndex idx, float val)
{
	debug_msg("set_value: " + container2str(idx) + "=" + to_str(val), is_initialized);
	debug_assert_init("set_value", is_initialized);

	write->at(idx) = val;
}


// FIXME add contextual rand engine
inline void fill_rand(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	debug_msg("fill_rand", is_initialized);
	debug_assert_init("fill_rand", is_initialized);

	write->fill([](int i, int j) {
		return FakeRand::instance_connection()();
	});
}

inline void fill_rand_prehistory(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized)
{
	debug_msg("fill_rand_prehistory", is_initialized);
	debug_assert_init("fill_rand_prehistory", is_initialized);

	write->fill([](int i, int j) {
		return FakeRand::instance_prehistory()();
	});
}

inline void fill_element(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized,
		lmn::ElementFillFunc<float> filler)
{
	debug_msg("fill_element", is_initialized);
	debug_assert_init("fill_element", is_initialized);

	write->fill([filler](int i, int j) {
		return filler(DimIndex {i, j});
	});
}

// For gradient checking
inline void perturb(vector<VecmatfPtr> reads, VecmatfPtr write, bool is_initialized,
		DimIndex idx, float eps)
{
	debug_msg("perturb", is_initialized);
	debug_assert_init("perturb", is_initialized);

	write->at(idx) += eps;
}


} // end of DummyImpl::
} // end of lmn::


class VecmatEngine :
	public Engine<lmn::Vecmatf, float>
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
		register_normal_op("t*t", Impl::mult_t_t);
		register_normal_op("t*s", Impl::mult_t_s);
		register_normal_op("s*t", Impl::mult_s_t);
//		register_normal_op("s*s", Impl::mult<S, S>);
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
	}

	/**
	 * Implements element retrieval
	 */
	float tensor_data_at(lmn::VecmatfPtr vecmat, DimIndex idx)
	{
		LMN_ASSERT_THROW(!vecmat->is_empty(),
			EngineException("VecmatEngine: scalar_at() called on null matrix"));

		return vecmat->at(idx);
	}

	float scalar_data_at(lmn::VecmatfPtr vecmat)
	{
		LMN_ASSERT_THROW(!vecmat->is_empty(),
			EngineException("VecmatEngine: scalar_at() called on null matrix"));

		LMN_ASSERT_THROW(vecmat->dim() == (Dimension {1, 1}),
			EngineException("VecmatEngine: scalar_at() called on wrong dimension:\n"
					+ container2str(vecmat->dim()) + " while [1, 1] expected."));

		return vecmat->at({0, 0});
	}
};


#endif /* BACKEND_VECMAT_VECMAT_ENGINE_H_ */
