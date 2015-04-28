/*
 * Eona Studio (c) 2015
 */

#include "cublas_helper.h"

FakeRand& rand_conn = FakeRand::instance_connection();
FakeRand& rand_prehis = FakeRand::instance_prehistory();
FakeRand& rand_input = FakeRand::instance_input();
FakeRand& rand_target = FakeRand::instance_target();


TEST(CublasLSTM, Composite)
{
	const int HISTORY = 5;
	const int INPUT_DIM = 2;
	const int TARGET_DIM = 4;
	const int BATCH = 2;

	auto inLayer = Layer::make<ConstantLayer>(INPUT_DIM);
	auto lossLayer = Layer::make<SquareLossLayer>(TARGET_DIM);

	auto engine = EngineBase::make<CublasEngine>();
	auto dataman = DataManagerBase::make<CublasRandDataManager>(
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

	gradient_check<CublasEngine, CublasRandDataManager>(net, 1e-2f, 2.f);
}


TEST(CublasLSTM, Softmax)
{
	const int HISTORY = 5;
	const int INPUT_DIM = 2;
	const int TARGET_DIM = 4;
	const int BATCH = 2;

	auto inLayer = Layer::make<ConstantLayer>(INPUT_DIM);
	auto lossLayer = Layer::make<LabelSoftmaxEntropyLayer>(TARGET_DIM);

	auto engine = EngineBase::make<CublasEngine>();
	auto dataman = DataManagerBase::make<CublasRandDataManager>(
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

	gradient_check<CublasEngine, CublasRandDataManager>(net, 1e-2f, 2.f);
}
