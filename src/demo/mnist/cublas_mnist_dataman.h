/*
 * Eona Studio (c) 2015
 */

#ifndef DEMO_MNIST_CUBLAS_MNIST_DATAMAN_H_
#define DEMO_MNIST_CUBLAS_MNIST_DATAMAN_H_

#include "mnist_dataman.h"
#include "../../backend/cublas/cublas_engine.h"

struct CublasMnistDataManager :
		public MnistDataManager<CudaFloatMat>
{
	CublasMnistDataManager(EngineBase::Ptr engine,
					int batchSize,
					string mnistDataDir) :
		MnistDataManager<CudaFloatMat>(engine, batchSize, mnistDataDir)
	{ }

protected:
	// subclass handles actual data load
	void alloc_zeros(DataPtr write, int rowdim, int coldim)
	{
		write->reset(rowdim, coldim);
	}

	// one batch of image (28 * 28 * batchSize)
	void load_data(DataPtr write, vector<float>& imageBatch)
	{
		write->to_device(&imageBatch[0]);
	}
};


#endif /* DEMO_MNIST_CUBLAS_MNIST_DATAMAN_H_ */
