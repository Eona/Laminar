/*
 * Eona Studio (c) 2015
 */

#include "../backend/opencl/opencl_engine.h"
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

TEST(OpenclForward, Simple)
{
	const int INPUT_DIM = 3;
	const int TARGET_DIM = 4;
	const int BATCH_SIZE = 2;

	auto engine = EngineBase::make<OpenclEngine>();

	auto dataman = DataManagerBase::make<OpenclRandDataManager>(
					engine, INPUT_DIM, TARGET_DIM, BATCH_SIZE);

	auto l1 = Layer::make<ConstantLayer>(INPUT_DIM);

//	auto l2 = Layer::make<SigmoidLayer>(3);

	auto l3 = Layer::make<CosineLayer>(7);

	auto l4 = Layer::make<SquareLossLayer>(TARGET_DIM);

	ForwardNetwork net(engine, dataman);

	net.add_layer(l1);
//	net.new_connection<FullConnection>(l1, l3);
//	net.new_connection<FullConnection>(l1, l2);
//	net.add_layer(l2);
//	net.new_connection<FullConnection>(l2, l3);
//	net.add_layer(l3);
//	net.new_connection<FullConnection>(l3, l4);
	net.new_connection<FullConnection>(l1, l4);
	net.add_layer(l4);

//	net.execute("initialize");
//	net.execute("load_input");
//	net.execute("load_target");
//	net.execute("forward");
//	net.execute("backward");

	gradient_check<OpenclEngine, OpenclRandDataManager>(net, 1e-2f, 0.8f);
}

TEST(OpenclForward, Softmax)
{
	const int INPUT_DIM = 3;
	const int TARGET_DIM = 4;
	const int BATCH_SIZE = 2;

	auto engine = EngineBase::make<OpenclEngine>();

	auto dataman = DataManagerBase::make<OpenclRandDataManager>(
					engine, INPUT_DIM, 1, BATCH_SIZE,
					// number of target label classes (classification task)
					TARGET_DIM);

	auto l1 = Layer::make<ConstantLayer>(INPUT_DIM);

	auto l2 = Layer::make<SigmoidLayer>(4);

	auto l3 = Layer::make<TanhLayer>(5);

	auto l4 = Layer::make<LabelSoftmaxEntropyLayer>(TARGET_DIM);

	ForwardNetwork net(engine, dataman);

	net.add_layer(l1);
	net.new_connection<FullConnection>(l1, l2);
	net.add_layer(l2);
	net.new_connection<FullConnection>(l2, l3);
	net.add_layer(l3);
	net.new_connection<FullConnection>(l3, l4);
	net.add_layer(l4);

	gradient_check<OpenclEngine, OpenclRandDataManager>(net, 1e-2f, 0.8f);
}
