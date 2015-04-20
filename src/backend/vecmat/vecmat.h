/*
 * Eona Studio (c) 2015
 */

#ifndef BACKEND_VECMAT_VECMAT_H_
#define BACKEND_VECMAT_VECMAT_H_

#include "../../utils/global_utils.h"

class VecmatException: public LaminarException {
public:
    VecmatException(const string& msg):
    	LaminarException(msg)
	{}

    virtual string error_header() const
    {
    	return "VectorMat error";
    }
};


template<typename FloatT = float>
class Vecmat
{
public:
	Vecmat() {}

	Vecmat(int row, int col)
	{
		mat.resize(row);
		for (int r = 0; r < row; ++r)
			mat[r].resize(col);
	}

	Vecmat(std::initializer_list<std::initializer_list<FloatT>> initer)
	{
		mat.insert(mat.end(), initer.begin(), initer.end());
	}

	// Copy ctor
	Vecmat(const Vecmat& other) :
		mat(other.mat)
	{ }

	TYPEDEF_PTR(Vecmat<FloatT>);

	// Copy assignment
	Vecmat& operator=(const Vecmat& other)
	{
		assert_throw<VecmatException>(!is_empty(),
			"\nShouldn't copy assign to a default constructed "
					"empty matrix. \nUse 'mat.new_zeros()' first.");

		assert_same_dim(other, "copy assign");
		this->mat = other.mat;
		return *this;
	}

	// Move ctor
	Vecmat(Vecmat&& other) :
		mat(std::move(other.mat))
	{ }

	// Move assignment
	Vecmat& operator=(Vecmat&& other)
	{
		assert_throw<VecmatException>(!is_empty(),
			"\nShouldn't move assign to a default constructed "
					"empty matrix. \nUse 'mat.new_zeros()' first.");

		assert_same_dim(other, "move assign");
		this->mat = std::move(other.mat);
		return *this;
	}

	// Only for default constructed matrix
	void new_zeros(int row, int col)
	{
		assert_throw<VecmatException>(is_empty(),
			"\nalloc_size() should only be used with default "
					"constructed empty matrix.");

		mat.resize(row);
		for (int r = 0; r < row; ++r)
			mat[r].resize(col);
	}

	/**
	 * Make a zero matrix of the same size as 'other'
	 */
	void new_zeros(const Vecmat& other)
	{
		assert_throw<VecmatException>(!other.is_empty(),
			"\nalloc_size(other) the other matrix cannot be empty.");

		this->new_zeros(other.row(), other.col());
	}

	/**
	 * Make a zero matrix of the same size as 'other'
	 */
	void new_zeros(Vecmat::Ptr other)
	{
		this->new_zeros(*other);
	}

	/**
	 * Set all entries to 0
	 */
	void zero_clear()
	{
		for (int r = 0; r < row(); ++r)
			for (int c = 0; c < col(); ++c)
				mat[r][c] = FloatT(0);
	}

	vector<FloatT>& operator[](int row)
	{
		return mat[row];
	}

	FloatT& operator()(int row, int col)
	{
		return mat[row][col];
	}

	FloatT operator()(int row, int col) const
	{
		return mat[row][col];
	}

	FloatT& at(DimIndex idx)
	{
		return mat[idx[0]][idx[1]];
	}

	FloatT at(DimIndex idx) const
	{
		return mat[idx[0]][idx[1]];
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

	Vecmat operator+(const Vecmat& rhs)
	{
		assert_same_dim(rhs, "addition");

		Vecmat ans(row(), col());
		for (int r = 0; r < row(); ++r)
			for (int c = 0; c < col(); ++c)
				ans[r][c] = this->mat[r][c] + rhs(r, c);

		return ans;
	}

	Vecmat operator-(const Vecmat& rhs)
	{
		assert_same_dim(rhs, "subtraction");

		Vecmat ans(row(), col());
		for (int r = 0; r < row(); ++r)
			for (int c = 0; c < col(); ++c)
				ans[r][c] = this->mat[r][c] - rhs(r, c);

		return ans;
	}

	Vecmat scale(FloatT scalor)
	{
		Vecmat ans(row(), col());
		for (int r = 0; r < row(); ++r)
			for (int c = 0; c < col(); ++c)
				ans[r][c] = this->mat[r][c] * scalor;

		return ans;
	}

	Vecmat operator*(FloatT scalor)
	{
		return this->scale(scalor);
	}

	// Negation
	Vecmat operator-()
	{
		return this->scale(FloatT(-1));
	}

	Vecmat operator*(const Vecmat& rhs)
	{
		assert_throw<VecmatException>(this->col() == rhs.row(),
				"multiplication dimension mismatch");

		Vecmat ans(this->row(), rhs.col());

		for(int i = 0; i < this->row(); ++i)
		  for(int j = 0; j < rhs.col(); ++j)
			 for(int k = 0; k < this->col(); ++k)
				ans[i][j] += this->mat[i][k] * rhs(k, j);

		return ans;
	}

	Vecmat transpose()
	{
		Vecmat ans(col(), row());
		for (int r = 0; r < row(); ++r)
			for (int c = 0; c < col(); ++c)
				ans[c][r] = this->mat[r][c];

		return ans;
	}

	/**
	 * Fill the matrix with a generator f(r, c)
	 */
	void fill(std::function<FloatT(int, int)> gen)
	{
		assert_throw<VecmatException>(!this->is_empty(),
				"cannot fill emptry matrix");

		for (int r = 0; r < row(); ++r)
			for (int c = 0; c < col(); ++c)
				mat[r][c] = gen(r, c);
	}

	bool is_empty() const
	{
		return row() == 0;
	}

	void assert_same_dim(const Vecmat& other, string msg)
	{
		assert_throw<VecmatException>(this->row() == other.row()
			&& this->col() == other.col(),
			msg + " dimension mismatch");
	}

private:
	vector<vector<FloatT>> mat;
};

template<typename FloatT>
std::ostream& operator<<(std::ostream& os, Vecmat<FloatT> mat)
{
	os << "[";
	for (int r = 0; r < mat.row(); ++r)
	{
		os << mat[r];
		if (r != mat.row() - 1)
			os << ",\n";
	}
	return os << "]";
}

#endif /* BACKEND_VECMAT_VECMAT_H_ */
