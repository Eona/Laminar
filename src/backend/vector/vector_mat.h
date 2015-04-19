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

	VectorMat(std::initializer_list<std::initializer_list<FloatT>> initer)
	{
		mat.insert(mat.end(), initer.begin(), initer.end());
	}

	// Copy ctor
	VectorMat(const VectorMat& other) :
		mat(other.mat)
	{ }

	// Copy assignment
	VectorMat& operator=(const VectorMat& other)
	{
		assert_throw(!is_empty(),
			VectorMatException("\nShouldn't copy assign to a default constructed "
					"empty matrix. \nUse 'mat.alloc()' instead."));

		check_dim(other, "copy assign");
		this->mat = other.mat;
		return *this;
	}

	// Move ctor
	VectorMat(VectorMat&& other) :
		mat(std::move(other.mat))
	{ }

	// Move assignment
	VectorMat& operator=(VectorMat&& other)
	{
		assert_throw(!is_empty(),
			VectorMatException("\nShouldn't move assign to a default constructed "
					"empty matrix. \nUse 'mat.alloc()' instead."));

		check_dim(other, "move assign");
		this->mat = std::move(other.mat);
		return *this;
	}

	// Only for default constructed matrix
	void alloc(int row, int col)
	{
		assert_throw(is_empty(),
			VectorMatException("\nalloc() should only be used with default "
					"constructed empty matrix."));

		mat.resize(row);
		for (int r = 0; r < row; ++r)
			mat[r].resize(col);
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
				ans[r][c] = this->mat[r][c] + rhs(r, c);

		return ans;
	}

	VectorMat operator-(const VectorMat& rhs)
	{
		check_dim(rhs, "subtraction");

		VectorMat ans(row(), col());
		for (int r = 0; r < row(); ++r)
			for (int c = 0; c < col(); ++c)
				ans[r][c] = this->mat[r][c] - rhs(r, c);

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
				ans[i][j] += this->mat[i][k] * rhs(k, j);

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

	bool is_empty()
	{
		return row() == 0;
	}
};

template<typename FloatT>
ostream& operator<<(ostream& os, VectorMat<FloatT> mat)
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

#endif /* BACKEND_VECTOR_VECTOR_MAT_H_ */
