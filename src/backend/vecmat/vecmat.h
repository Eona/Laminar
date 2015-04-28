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

	/**
	 * Initialize with a generator function
	 */
	Vecmat(int row, int col, std::function<FloatT(int, int)> gen) :
		Vecmat(row, col)
	{
		this->fill(gen);
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
		LMN_ASSERT_THROW(!is_empty(),
			VecmatException("\nShouldn't copy assign to a default constructed "
					"empty matrix. \nUse 'mat.new_zeros()' first."));

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
		LMN_ASSERT_THROW(!is_empty(),
			VecmatException("\nShouldn't move assign to a default constructed "
							"empty matrix. \nUse 'mat.new_zeros()' first."));

		assert_same_dim(other, "move assign");
		this->mat = std::move(other.mat);
		return *this;
	}

	// Only for default constructed matrix
	void new_zeros(int row, int col)
	{
		LMN_ASSERT_THROW(is_empty(),
			VecmatException("\nalloc_size() should only be used with "
							"default constructed empty matrix."));

		mat.resize(row);
		for (int r = 0; r < row; ++r)
			mat[r].resize(col);
	}

	/**
	 * Make a zero matrix of the same size as 'other'
	 */
	void new_zeros(const Vecmat& other)
	{
		LMN_ASSERT_THROW(!other.is_empty(),
			VecmatException("\nalloc_size(other) the other matrix cannot be empty."));

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
/*		LMN_ASSERT_THROW(0<= row && row < this->row() && 0 <= col && col < this->col(),
			VecmatException("access out of range\n"
				"Dim = " + container2str(Dimension{this->row(), this->col()}) +
				"\naccessor = " + container2str(Dimension{row, col})))*/
		return mat[row][col];
	}

	FloatT operator()(int row, int col) const
	{
/*		LMN_ASSERT_THROW(0<= row && row < this->row() && 0 <= col && col < this->col(),
			VecmatException("access out of range\n"
				"Dim = " + container2str(Dimension{this->row(), this->col()}) +
				"\naccessor = " + container2str(Dimension{row, col})))*/
		return mat[row][col];
	}

	FloatT& at(DimIndex idx)
	{
/*		LMN_ASSERT_THROW(idx[0] < this->row() && idx[1] < this->col(),
			VecmatException("access out of range\n"
				"Dim = " + container2str(Dimension{this->row(), this->col()}) +
				"\naccessor = " + container2str(idx)))*/
		return mat[idx[0]][idx[1]];
	}

	FloatT at(DimIndex idx) const
	{
/*		LMN_ASSERT_THROW(idx[0] < this->row() && idx[1] < this->col(),
			VecmatException("access out of range\n"
				"Dim = " + container2str(Dimension{this->row(), this->col()}) +
				"\naccessor = " + container2str(idx)))*/
		return mat[idx[0]][idx[1]];
	}

	int row() const
	{
		return mat.size();
	}

	int col() const
	{
		LMN_ASSERT_THROW(!mat.empty(),
			VecmatException("col() error: mat cannot be empty"));
		return mat[0].size();
	}

	Dimension dim() const
	{
		return is_empty() ? Dimension{} :
				Dimension{ row(), col() };
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

	Vecmat scale(FloatT scalar)
	{
		Vecmat ans(row(), col());
		for (int r = 0; r < row(); ++r)
			for (int c = 0; c < col(); ++c)
				ans[r][c] = this->mat[r][c] * scalar;

		return ans;
	}

	Vecmat operator*(FloatT scalar)
	{
		return this->scale(scalar);
	}

	// Negation
	Vecmat operator-()
	{
		return this->scale(FloatT(-1));
	}

	Vecmat operator*(const Vecmat& rhs)
	{
		LMN_ASSERT_THROW(this->is_initialized()
				&& this->col() == rhs.row(),
			VecmatException("multiplication dimension mismatch\n"
				+ dims_errmsg(rhs)));

		Vecmat ans(this->row(), rhs.col());

		for(int i = 0; i < this->row(); ++i)
		  for(int j = 0; j < rhs.col(); ++j)
			 for(int k = 0; k < this->col(); ++k)
				ans[i][j] += this->mat[i][k] * rhs(k, j);

		return ans;
	}

	/**
	 * Compare entry by entry with tolerance
	 * @return is same
	 */
	bool equals(const Vecmat& other, FloatT eps = 1e-6) const
	{
		assert_same_dim(other, "entry-by-entry equals");

		for (int r = 0; r < row(); ++r)
			for (int c = 0; c < col(); ++c)
				if (std::abs(mat[r][c] - other(r, c)) > eps)
					return false;

		return true;
	}

	bool operator==(const Vecmat& other) const
	{
		return this->equals(other, FloatT(1e-6));
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
	 * NOTE must be column major to work with VecmatDataManager
	 */
	void fill(std::function<FloatT(int, int)> gen)
	{
		LMN_ASSERT_THROW(this->is_initialized(),
				VecmatException("cannot fill emptry matrix"));

		for (int c = 0; c < col(); ++c)
			for (int r = 0; r < row(); ++r)
				mat[r][c] = gen(r, c);
	}

	bool is_empty() const
	{
		return row() == 0;
	}

	bool is_initialized() const
	{
		return row() != 0;
	}

	void assert_same_dim(const Vecmat& other, string msg) const
	{
		LMN_ASSERT_THROW(this->is_initialized()
			&& other.is_initialized()
			&& this->row() == other.row()
			&& this->col() == other.col(),

			VecmatException(msg + " dimension mismatch\n" + dims_errmsg(other)));
	}

	explicit operator string() const
	{
		std::ostringstream os;
		os << "[";
		for (int r = 0; r < row(); ++r)
		{
			os << mat[r];
			if (r != row() - 1)
				os << ",\n";
		}
		os << "]";
		return os.str();
	}

	/**
	 * Helper for dimension mismatch error message
	 */
	string dims_errmsg(const Vecmat& other) const
	{
		return container2str(this->dim())
				+ " <-> " + container2str(other.dim());
	}

private:
	vector<vector<FloatT>> mat;
};

template<typename FloatT>
std::ostream& operator<<(std::ostream& os, Vecmat<FloatT> mat)
{
	return os << string(mat);
}

#endif /* BACKEND_VECMAT_VECMAT_H_ */
