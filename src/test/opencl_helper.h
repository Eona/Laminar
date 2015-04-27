/*
 * Eona Studio (c) 2015
 */

#ifndef TEST_OPENCL_HELPER_H_
#define TEST_OPENCL_HELPER_H_

#include "../backend/opencl/opencl_engine.h"
#include "rand_dataman.h"
#include "test_utils.h"

struct OpenclRandDataManager :
		public RandDataManager<OpenclFloatMat>
{
	OpenclRandDataManager(EngineBase::Ptr engine,
					int inputDim, int targetDim, int batchSize,
					int targetLabelClasses = 0) :
		RandDataManager<OpenclFloatMat>(engine,
				inputDim, targetDim, batchSize,
				targetLabelClasses),
		cl(EngineBase::cast<OpenclEngine>(engine)->cl)
	{ }

protected:
	OclUtilContext* cl;

	// subclass handles actual data load
	void alloc_zeros(DataPtr write, int rowdim, int coldim)
	{
		write->reset(rowdim, coldim, cl);
	}

	// one batch of image (28 * 28 * batchSize)
	void load_data(DataPtr write, vector<float>& data)
	{
		write->to_device(&data[0]);
	}
};


#endif /* TEST_OPENCL_HELPER_H_ */
