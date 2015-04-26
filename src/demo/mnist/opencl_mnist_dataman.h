/*
 * Eona Studio (c) 2015
 */

#ifndef DEMO_MNIST_CUBLAS_MNIST_DATAMAN_H_
#define DEMO_MNIST_CUBLAS_MNIST_DATAMAN_H_

#include "mnist_dataman.h"
#include "../../backend/opencl/opencl_engine.h"

struct OpenclMnistDataManager :
		public MnistDataManager<OpenclFloatMat>
{
	OpenclMnistDataManager(EngineBase::Ptr engine,
					int batchSize,
					string mnistDataDir) :
		MnistDataManager<CudaFloatMat>(engine, batchSize, mnistDataDir),
		cl(EngineBase::cast<OpenclEngine>(engine)->cl)
	{}

protected:
	OclUtilContext* cl;

	// subclass handles actual data load
	void alloc_zeros(DataPtr write, int rowdim, int coldim)
	{
		write->reset(rowdim, coldim, cl);
	}

	// one batch of image (28 * 28 * batchSize)
	void load_data(DataPtr write, vector<float>& imageBatch)
	{
		write->to_device(&imageBatch[0]);
	}
};


#endif /* DEMO_MNIST_CUBLAS_MNIST_DATAMAN_H_ */
