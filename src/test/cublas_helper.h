/*
 * Eona Studio (c) 2015
 */

#ifndef TEST_CUBLAS_HELPER_H_
#define TEST_CUBLAS_HELPER_H_

#include "../backend/cublas/cublas_engine.h"
#include "rand_dataman.h"
#include "test_utils.h"

/**
 * Cublas
 */
struct CublasRandDataManager :
		public RandDataManager<CudaFloatMat>
{
	CublasRandDataManager(EngineBase::Ptr engine,
					int inputDim, int targetDim, int batchSize,
					int targetLabelClasses = 0) :
		RandDataManager<CudaFloatMat>(engine,
				inputDim, targetDim, batchSize,
				targetLabelClasses)
	{ }

protected:
	// subclass handles actual data load
	void alloc_zeros(DataPtr write, int rowdim, int coldim)
	{
		write->reset(rowdim, coldim);
	}

	// one batch of image (28 * 28 * batchSize)
	void load_data(DataPtr write, vector<float>& data)
	{
		write->to_device(&data[0]);
	}
};



#endif /* TEST_CUBLAS_HELPER_H_ */
