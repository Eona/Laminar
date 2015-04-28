/*
 * Eona Studio (c) 2015
 */

#include "opencl_helper.h"

TEST(OpenclLSTM, Composite)
{
	const int HISTORY = 5;
	const int INPUT_DIM = 2;
	const int TARGET_DIM = 4;
	const int BATCH = 2;

	auto inLayer = Layer::make<ConstantLayer>(INPUT_DIM);
	auto lossLayer = Layer::make<SquareLossLayer>(TARGET_DIM);

	auto engine = EngineBase::make<OpenclEngine>();
	auto dataman = DataManagerBase::make<OpenclRandDataManager>(
			engine, INPUT_DIM, TARGET_DIM, BATCH);

	RecurrentNetwork net(engine, dataman, HISTORY);

	net.add_layer(inLayer);

	auto lstmComposite =
			Composite<RecurrentNetwork>::make<LstmComposite>(inLayer, 3);
	// or you can add the object directly:
	// auto lstmCompositeObject =
	//		Composite<RecurrentNetwork>::create<LstmComposite>(inLayer, 7);

	net.add_composite(lstmComposite);
	net.new_connection<FullConnection>(lstmComposite->out_layer(), lossLayer);
	net.add_layer(lossLayer);

	gradient_check<OpenclEngine, OpenclRandDataManager>(net, 1e-2f, 1.f);
}


TEST(OpenclLSTM, Softmax)
{
	const int HISTORY = 5;
	const int INPUT_DIM = 2;
	const int TARGET_DIM = 4;
	const int BATCH = 2;

	auto inLayer = Layer::make<ConstantLayer>(INPUT_DIM);
	auto lossLayer = Layer::make<LabelSoftmaxEntropyLayer>(TARGET_DIM);

	auto engine = EngineBase::make<OpenclEngine>();
	auto dataman = DataManagerBase::make<OpenclRandDataManager>(
			engine, INPUT_DIM, 1, BATCH,
			// classification task
			TARGET_DIM);

	RecurrentNetwork net(engine, dataman, HISTORY);

	net.add_layer(inLayer);

	auto lstmComposite =
			Composite<RecurrentNetwork>::make<LstmComposite>(inLayer, 3);
	// or you can add the object directly:
	// auto lstmCompositeObject =
	//		Composite<RecurrentNetwork>::create<LstmComposite>(inLayer, 7);
	net.add_composite(lstmComposite);
	net.new_connection<FullConnection>(lstmComposite->out_layer(), lossLayer);
	net.add_layer(lossLayer);

	gradient_check<OpenclEngine, OpenclRandDataManager>(net, 1e-2f, 1.f);
}
