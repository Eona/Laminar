/*
 * Eona Studio (c) 2015
 */

#ifndef TEST_EIGEN_HELPER_H_
#define TEST_EIGEN_HELPER_H_

#include "../backend/eigen/eigen_engine.h"
#include "rand_dataman.h"
#include "test_utils.h"

/**
 * Eigen
 */
struct EigenRandDataManager :
		public RandDataManager<MatrixXf>
{
	EigenRandDataManager(EngineBase::Ptr engine,
					int inputDim, int targetDim, int batchSize,
					int targetLabelClasses = 0) :
		RandDataManager<MatrixXf>(engine,
				inputDim, targetDim, batchSize,
				targetLabelClasses)
	{ }

protected:
	// subclass handles actual data load
	void alloc_zeros(DataPtr write, int rowdim, int coldim)
	{
		*write = MatrixXf(rowdim, coldim);
	}

	// one batch of image (28 * 28 * batchSize)
	void load_data(DataPtr write, vector<float>& data)
	{
		MatrixXf& wmat = *write;
		int i = 0;
		for (int c = 0; c < wmat.cols(); ++c)
			for (int r = 0; r < wmat.rows(); ++r)
				wmat(r, c) = data[i++];
	}
};


#endif /* TEST_EIGEN_HELPER_H_ */
