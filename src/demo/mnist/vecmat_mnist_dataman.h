/*
 * Eona Studio (c) 2015
 */

#ifndef DEMO_MNIST_CUBLAS_MNIST_DATAMAN_H_
#define DEMO_MNIST_CUBLAS_MNIST_DATAMAN_H_

#include "mnist_dataman.h"
#include "../../backend/vecmat/vecmat_engine.h"

struct VecmatMnistDataManager :
		public MnistDataManager<lmn::Vecmatf>
{
	VecmatMnistDataManager(EngineBase::Ptr engine,
					int batchSize,
					string mnistDataDir) :
		MnistDataManager<lmn::Vecmatf>(engine, batchSize, mnistDataDir)
	{ }

protected:
	// subclass handles actual data load
	void alloc_zeros(DataPtr write, int rowdim, int coldim)
	{
		write->new_zeros(rowdim, coldim);
	}

	// one batch of image (28 * 28 * batchSize)
	void load_data(DataPtr write, vector<float>& imageBatch)
	{
		write->fill([&](int r, int c) {
			return imageBatch[r + c * MNIST_INPUT_DIM];
		});
	}
};


#endif /* DEMO_MNIST_CUBLAS_MNIST_DATAMAN_H_ */
