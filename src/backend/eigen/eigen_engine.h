/*
 * Eona Studio (c) 2015
 */

#ifndef BACKEND_EIGEN_EIGEN_ENGINE_H_
#define BACKEND_EIGEN_EIGEN_ENGINE_H_

#include <Eigen/Dense>
#include "../../engine/engine.h"
#include "../../engine/tensor.h"
#include "../../engine/tensor_ops.h"
#include "../../utils/rand_utils.h"

using namespace Eigen;

class EigenEngineException: public EngineException {
public:
    EigenEngineException(const std::string& msg):
    	EngineException(msg)
	{}

    virtual std::string error_header() const
    {
    	return "EigenEngine error";
    }
};

namespace lmn {

typedef std::shared_ptr<MatrixXf> EigenfPtr;

namespace EigenImpl {

MatrixXf create(int row, int col)
{
	// WARNING plain Eigen ctor does NOT initialize to zero!!!
	return MatrixXf::Zero(row, col);
}

MatrixXf create(Dimension dim)
{
	return MatrixXf::Zero(dim[0], dim[1]);
}

MatrixXf create(EigenfPtr mat)
{
	return MatrixXf::Zero(mat->rows(), mat->cols());
}

Dimension dim(EigenfPtr mat)
{
	return Dimension { (int) mat->rows(), (int) mat->cols() };
}

// omit args for Scalar
float& at(EigenfPtr mat, int r = 0, int c = 0)
{
	return (*mat)(r, c);
}

float& at(EigenfPtr mat, Dimension dim)
{
	return (*mat)(dim[0], dim[1]);
}

void create_op(EigenfPtr write, Dimension dim)
{
	*write = create(dim);
}

void debug_assert_init(string msg, bool is_initialized)
{
	LMN_ASSERT_THROW(is_initialized,
		EigenEngineException("calling "+msg+" on uninitialized write addr"));
}

void add(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		// WARNING in unoptimized version, x += a translates to
		// x = x + a, where reads[0] == write == x and reads[1] == a
		// if x is null created (as often in layer gradient += ops)
		// reads[0] will be empty. So here 'write' should be initialized with reads[1] size.
		*write = create(reads[1]);

	*write = *reads[0] + *reads[1];
}

void sub(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		*write = create(reads[1]);

	*write = *reads[0] - *reads[1];
}

void negate(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		*write = create(reads[0]);

	*write = - (*reads[0]);
}

void mult_t_t(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		*write = create(reads[0]->rows(), reads[1]->cols());

	*write = (*reads[0]) * (*reads[1]);
}

void mult_t_s(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		*write = create(reads[0]);

	LMN_ASSERT_THROW(dim(reads[1]) == (Dimension {1, 1}),
		EigenEngineException("t*s Scalar reads[1] should have dimension [1, 1]\n"
				"But now it's " + container2str(dim(reads[1]))));

	*write = *reads[0] * at(reads[1]);
}

void mult_s_t(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		*write = create(reads[1]);

	LMN_ASSERT_THROW(dim(reads[0]) == (Dimension {1, 1}),
		EigenEngineException("s*t Scalar reads[0] should have dimension [1, 1]\n"
				"But now it's " + container2str(dim(reads[0]))));

	*write = *reads[1] * at(reads[0]);
}

void mult_s_s(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		*write = create(1, 1);

	LMN_ASSERT_THROW(dim(reads[0]) == (Dimension {1, 1})
			&& dim(reads[1]) == (Dimension {1, 1}),
		EigenEngineException("s*s Scalar both reads[0] and [1] dim should be [1, 1]\n"
				"But now they're " + container2str(dim(reads[0])) +
				" and " + container2str(dim(reads[1]))));

	at(write) = at(reads[0]) * at(reads[1]);
}

void scale(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized, float scalarContext)
{
	LMN_ASSERT_THROW(reads[0]->size() != 0,
		EigenEngineException("calling scale on unintialized reads[0] addr"));

	if (!is_initialized)
		*write = create(reads[0]);

	*write = *reads[0] * scalarContext;
}

void assign(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		*write = create(reads[0]);

	*write = *reads[0];
}

/**
 * Assign a float constant to Scalar
 */
void assign_const(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized, float constant)
{
	if (!is_initialized)
		*write = create(1, 1);

	at(write) = constant;
}

inline void destroy(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
}

// standalone single-float non-linear functions
inline void transpose(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		*write = create(reads[0]->cols(), reads[0]->rows());

	*write = reads[0]->transpose();
}

/**
 * Helper for element-wise unary function application
 */
template<typename UnaryFunc>
inline void element_apply(EigenfPtr read, EigenfPtr write, UnaryFunc func)
{
	MatrixXf& rmat = *read;
	MatrixXf& wmat = *write;

	for (int r = 0; r < rmat.rows(); ++r)
		for (int c = 0; c < rmat.cols(); ++c)
			wmat(r, c) = func(rmat(r, c));
}

inline void sigmoid(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		*write = create(reads[0]);

	element_apply(reads[0], write,
			[](float x) { return 1.f / (1.f + exp(-x)); });
}

inline void sigmoid_gradient(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		*write = create(reads[0]);

	element_apply(reads[0], write,
			[](float x) { return x * (1.f - x); });
}

inline void sin(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		*write = create(reads[0]);

	element_apply(reads[0], write,
			static_cast<float (*)(float)>(std::sin));
}

inline void cos(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		*write = create(reads[0]);

	element_apply(reads[0], write,
			static_cast<float (*)(float)>(std::cos));
}

inline void tanh(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		*write = create(reads[0]);

	element_apply(reads[0], write,
			static_cast<float (*)(float)>(std::tanh));
}

inline void tanh_gradient(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		*write = create(reads[0]);

	element_apply(reads[0], write,
			[](float x) { return 1.f - x * x; });
}

inline void element_mult(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		*write = create(reads[0]);

	MatrixXf& rmat1 = *reads[0];
	MatrixXf& rmat2 = *reads[1];
	MatrixXf& wmat = *write;

	for (int r = 0; r < wmat.rows(); ++r)
		for (int c = 0; c < wmat.cols(); ++c)
			wmat(r, c) = rmat1(r, c) * rmat2(r, c);
}

/**
 * @param reads
 * @param write a scalar that's the sum of all square differences
 * @param is_initialized
 */
inline void square_loss(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		*write = create(1, 1);

	MatrixXf& rmat1 = *reads[0];
	MatrixXf& rmat2 = *reads[1];
	MatrixXf& loss = *write;

	loss(0, 0) = 0;
	for (int r = 0; r < rmat1.rows(); ++r)
		for (int c = 0; c < rmat1.cols(); ++c)
		{
			float diff = rmat1(r, c) - rmat2(r, c);
			loss(0, 0) += 0.5f * diff * diff;
		}
}

inline void clip(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	debug_assert_init("clip", is_initialized);

	auto clipper = [](float x) ->float
	{
		if (x != x) return 0; // NaN
		if (x < -1) return -1;
		else if (x > 1) return 1;
		else return x;
	};

	element_apply(reads[0], write, clipper);
}

/**
 * Max-exp trick for numerical stability
 * 1) find max in each column vector
 * 2) subtract max from every element in this column
 * 3) exp every element
 * 4) sum step (3)
 * 5) divide every element by step (4)
 */
inline void softmax(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		*write = create(reads[0]);

	MatrixXf& rmat = *reads[0];
	MatrixXf& wmat = *write;

	// Each column is a data feature vector
	// coldim is batch size
	for (int c = 0; c < rmat.cols(); ++c)
	{
		// find max
		float mx = -1e20f;
		for (int r = 0; r < rmat.rows(); ++r)
			if (rmat(r, c) > mx)
				mx = rmat(r, c);

		// exp(a - mx) for all 'a'
		for (int r = 0; r < rmat.rows(); ++r)
			wmat(r, c) = std::exp((float) rmat(r, c) - mx);

		// sum last step
		float sum = 0;
		for (int r = 0; r < wmat.rows(); ++r)
			sum += wmat(r, c);

		// divide every wmat col element by sum
		for (int r = 0; r < wmat.rows(); ++r)
			wmat(r, c) /= sum;
	}
}

/**
 * -log(value_at_label)
 * @param reads a tensor of int class labels (faked as floats)
 * @param write a scalor loss
 */
inline void label_entropy_loss(
		vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		*write = create(1, 1);

	MatrixXf& rmat = *reads[0];
	MatrixXf& labels = *reads[1];

	MatrixXf& loss = *write;

	LMN_ASSERT_THROW(rmat.cols() == labels.cols(),
			EigenEngineException("label_entropy_loss input mat coldim doesn't match labels coldim"));

	loss(0, 0) = 0;
	for (int c = 0; c < rmat.cols(); ++c)
	{
		int label = (int) labels(0, c);

		LMN_ASSERT_THROW(0 <= label && label < rmat.rows(),
			EigenEngineException("label_softmax_entropy_gradient label value error\n" +
				to_str(label) + " should < rowdim " + to_str(rmat.rows())));

		// value at label:
		loss(0, 0) += -std::log((float)rmat(label, c));
	}
}

/**
 *
 * @param reads y, vector *after* softmax
 * @param write y - t, where t is a sparse vector with a single '1' at the correct label
 * @param is_initialized
 */
inline void label_softmax_entropy_gradient(
		vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	if (!is_initialized)
		*write = create(reads[0]);

	MatrixXf& rmat = *reads[0];
	MatrixXf& labels = *reads[1];

	MatrixXf& wmat = *write;

	LMN_ASSERT_THROW(rmat.cols() == labels.cols(),
			EigenEngineException(
				"label_softmax_entropy_gradient input mat coldim doesn't match labels coldim"));

	wmat = rmat; // copy most values won't change

	for (int c = 0; c < rmat.cols(); ++c)
	{
		int label = (int) labels(0, c);

		LMN_ASSERT_THROW(0 <= label && label < rmat.rows(),
			EigenEngineException("label_softmax_entropy_gradient label value error\n" +
				to_str(label) + " should < rowdim " + to_str(rmat.rows())));

		wmat(label, c) -= 1.f; // y - t (sparse)
	}
}


inline void zero_clear(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
// FIXME loss layer output might be zero cleared without being initialized
//	debug_assert_init("zero_clear", is_initialized);

//	if (is_initialized)
	write->setZero();
}

inline void set_value(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized,
		DimIndex idx, float val)
{
	debug_assert_init("set_value", is_initialized);

	at(write, idx) = val;
}


// FIXME add contextual rand engine
inline void fill_rand(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized)
{
	debug_assert_init("fill_rand", is_initialized);

	MatrixXf& wmat = *write;
	UniformRand<float> rnd(-0.08f, 0.08f);

	for (int c = 0; c < wmat.cols(); ++c)
		for (int r = 0; r < wmat.rows(); ++r)
			wmat(r, c) = rnd();
}

inline void fill_element(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized,
		lmn::ElementFillFunc<float> filler)
{
	debug_assert_init("fill_element", is_initialized);
	MatrixXf& wmat = *write;

	for (int c = 0; c < wmat.cols(); ++c)
		for (int r = 0; r < wmat.rows(); ++r)
			wmat(r, c) = filler(Dimension {r, c});
}

// For gradient checking
inline void perturb(vector<EigenfPtr> reads, EigenfPtr write, bool is_initialized,
		DimIndex idx, float eps)
{
	debug_assert_init("perturb", is_initialized);

	at(write, idx) += eps;
}


} // end of EigenImpl::
} // end of lmn::


class EigenEngine :
	public Engine<MatrixXf, float>
{
public:
	EigenEngine() :
		Engine<MatrixXf, float>()
	{
		namespace Impl = lmn::EigenImpl;

		register_create_op(Impl::create_op);

		register_normal_op("t+t", Impl::add);
		register_normal_op("s+s", Impl::add);
		register_normal_op("t-t", Impl::sub);
		register_normal_op("s-s", Impl::sub);
		register_normal_op("-t", Impl::negate);
		register_normal_op("-s", Impl::negate);
		register_normal_op("t*t", Impl::mult_t_t);
		register_normal_op("t*s", Impl::mult_t_s);
		register_normal_op("s*t", Impl::mult_s_t);
		register_normal_op("s*s", Impl::mult_s_s);
		register_normal_op("t=t", Impl::assign);
		register_normal_op("s=s", Impl::assign);
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
		register_normal_op("softmax", Impl::softmax);
		register_normal_op("clip", Impl::clip);
		register_normal_op("label_entropy_loss", Impl::label_entropy_loss);
		register_normal_op("label_softmax_entropy_gradient", Impl::label_softmax_entropy_gradient);

		register_normal_op("destroy", Impl::destroy);
		register_normal_op("zero_clear", Impl::zero_clear);

		register_normal_op("fill_rand", Impl::fill_rand);
		register_normal_op("fill_rand_prehistory", Impl::fill_rand);
		register_context_op<lmn::ElementFillFunc<float>>(
						"fill_element", Impl::fill_element);

		register_context_op<DimIndex, float>("perturb", Impl::perturb);
		register_context_op<DimIndex, float>("set_value", Impl::set_value);
		register_context_op<float>("scale", Impl::scale);
	}

	/**
	 * Implements element retrieval
	 */
	float tensor_data_at(lmn::EigenfPtr mat, DimIndex idx)
	{
		namespace Impl = lmn::EigenImpl;
		LMN_ASSERT_THROW(mat->size() != 0,
			EigenEngineException("scalar_at() called on null matrix"));

		return Impl::at(mat, idx);
	}

	float scalar_data_at(lmn::EigenfPtr mat)
	{
		namespace Impl = lmn::EigenImpl;
		LMN_ASSERT_THROW(mat->size() != 0,
			EigenEngineException("scalar_at() called on null matrix"));

		LMN_ASSERT_THROW(Impl::dim(mat) == (Dimension {1, 1}),
			EigenEngineException("scalar_at() called on wrong dimension:\n"
					+ container2str(Impl::dim(mat)) + " while [1, 1] expected."));

		return Impl::at(mat);
	}
};

#endif /* BACKEND_EIGEN_EIGEN_ENGINE_H_ */
