/*
 * Eona Studio (c) 2015
 */

#ifndef BACKEND_VECTOR_VECTOR_MAT_H_
#define BACKEND_VECTOR_VECTOR_MAT_H_

#include "../../global_utils.h"

class VectorMatException: public LaminarException {
public:
    VectorMatException(const string& msg):
    	LaminarException(msg)
	{}

    virtual string error_header() const
    {
    	return "VectorMat error";
    }
};


template<typename FloatT>
class VectorMat
{
public:
	VectorMat() {}

	VectorMat(int row, int col)
	{
		mat.resize(row);
		for (int r = 0; r < row; ++r)
			mat[r].resize(col);
	}

	// Copy ctor
	VectorMat(const VectorMat& other)
	{
		check_dim(other, "copy ctor");
		this->mat = other.mat;
	}

	// Copy assignment
	VectorMat& operator=(const VectorMat& other)
	{
		check_dim(other, "copy assign");
		this->mat = other.mat;
		return *this;
	}

	// Move ctor
	VectorMat(VectorMat&& other)
	{
		check_dim(other, "move ctor");
		this->mat = std::move(other.mat);
	}

	// Move assignment
	VectorMat& operator=(VectorMat&& other)
	{
		check_dim(other, "move assign");
		this->mat = std::move(other.mat);
		return *this;
	}

	vector<FloatT>& operator[](int row)
	{
		return mat[row];
	}

	int row() const
	{
		return mat.size();
	}

	int col() const
	{
		return mat[0].size();
	}

	bool is_initialized() const
	{
		return row() != 0 && col() != 0;
	}

	VectorMat operator+(const VectorMat& rhs)
	{
		check_dim(rhs, "addition");

		VectorMat ans(row(), col());
		for (int r = 0; r < row(); ++r)
			for (int c = 0; c < col(); ++c)
				ans[r][c] = this->mat[r][c] + rhs[r][c];

		return ans;
	}

	VectorMat operator-(const VectorMat& rhs)
	{
		check_dim(rhs, "subtraction");

		VectorMat ans(row(), col());
		for (int r = 0; r < row(); ++r)
			for (int c = 0; c < col(); ++c)
				ans[r][c] = this->mat[r][c] - rhs[r][c];

		return ans;
	}

	VectorMat scale(FloatT scalor)
	{
		VectorMat ans(row(), col());
		for (int r = 0; r < row(); ++r)
			for (int c = 0; c < col(); ++c)
				ans[r][c] = this->mat[r][c] * scalor;

		return ans;
	}

	// Negation
	VectorMat operator-()
	{
		return this->scale(FloatT(-1));
	}

	VectorMat operator*(const VectorMat& rhs)
	{
		assert_throw(this->col() == rhs.row(),
			VectorMatException("multiplication dimension mismatch"));

		VectorMat ans(this->row(), rhs.col());

		for(int i = 0; i < this->row(); ++i)
		  for(int j = 0; j < rhs.col(); ++j)
			 for(int k = 0; k < this->col(); ++k)
				ans[i][j] += this->mat[i][k] * rhs[k][j];

		return ans;
	}

	VectorMat transpose()
	{
		VectorMat ans(col(), row());
		for (int r = 0; r < row(); ++r)
			for (int c = 0; c < col(); ++c)
				ans[c][r] = this->mat[r][c];

		return ans;
	}

private:
	vector<vector<FloatT>> mat;

	void check_dim(const VectorMat& other, string msg)
	{
		assert_throw(this->row() == other.row()
			&& this->col() == other.col(),
			VectorMatException(msg + " dimension mismatch"));
	}
};

namespace lmn {

namespace VectorImpl {

typedef std::shared_ptr<float> FloatPtr;

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

void scale(vector<FloatPtr> reads, FloatPtr write, bool is_initialized, float scalorContext)
{
	debug_msg("scale * " + to_str(scalorContext), is_initialized);
	*write = scalorContext * (*reads[0]);
}

template<int TensorT>
void assign(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	string op = tensor_op<TensorT>::operand;
	debug_msg(op + "=" + op, is_initialized);
	*write = *reads[0];
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

inline void clear(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	debug_msg("clear", is_initialized);
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
	assert_throw(is_initialized,
		EngineException("DummyEngine: fill_rand must have been initialized"));
	debug_msg("fill_rand", is_initialized);
	*write = FakeRand::instance_connection()();
}

inline void fill_rand_prehistory(vector<FloatPtr> reads, FloatPtr write, bool is_initialized)
{
	assert_throw(is_initialized,
		EngineException("DummyEngine: fill_rand_prehistory must have been initialized"));
	debug_msg("fill_rand_prehistory", is_initialized);
	if (write == nullptr)
		cout << "BAD bAD bAD" << endl;
	*write = FakeRand::instance_prehistory()();
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

#endif /* BACKEND_VECTOR_VECTOR_MAT_H_ */
