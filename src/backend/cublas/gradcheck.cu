/*
 * Eona Studio (c) 2015
 */

#include "cuda_engine.h"
#include "../rand_dataman.h"
#include "../../connection.h"
#include "../../full_connection.h"
#include "../../gated_connection.h"
#include "../../activation_layer.h"
#include "../../bias_layer.h"
#include "../../loss_layer.h"
#include "../../parameter.h"
#include "../../network.h"
#include "../../rnn.h"
#include "../../lstm.h"
#include "../../gradient_check.h"

struct CublasRandDataManager :
		public RandDataManager<CudaFloatMat>
{
	CublasRandDataManager(EngineBase::Ptr engine,
					int inputDim, int targetDim, int batchSize) :
		RandDataManager<CudaFloatMat>(engine, inputDim, targetDim, batchSize)
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

int main()
{
	const int INPUT_DIM = 3;
	const int TARGET_DIM = 4;
	const int BATCH_SIZE = 2;

	auto engine = EngineBase::make<CudaEngine>();

	auto dataman = DataManagerBase::make<CublasRandDataManager>(
					engine, INPUT_DIM, TARGET_DIM, BATCH_SIZE);

	auto l1 = Layer::make<ConstantLayer>(INPUT_DIM);

	auto l2 = Layer::make<SigmoidLayer>(7);

	auto l3 = Layer::make<TanhLayer>(4);

	auto l4 = Layer::make<SquareLossLayer>(TARGET_DIM);

	ForwardNetwork net(engine, dataman);

	net.add_layer(l1);
	net.new_connection<FullConnection>(l1, l2);
	net.add_layer(l2);
	net.new_connection<FullConnection>(l2, l3);
	net.add_layer(l3);
	net.new_connection<FullConnection>(l3, l4);
	net.add_layer(l4);

//	net.execute("initialize");
//	net.execute("load_input");
//	net.execute("load_target");
//	net.execute("forward");
//	net.execute("backward");

	gradient_check<CudaEngine, CublasRandDataManager>(net, 1e-2f, 0.8f);
}
