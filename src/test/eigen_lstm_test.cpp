/*
 * Eona Studio (c) 2015
 */

#include "eigen_helper.h"

TEST(EigenLSTM, Double)
{
	// Pass in global parameter to lmn::fill_rand()
	// uniformly random initialize connections: -1.f to +1.f
	FakeRand::instance_passin().set_rand_seq({1.5f});

	const int HISTORY = 5;
	const int INPUT_DIM = 2;
	const int TARGET_DIM = 4;
	const int BATCH = 2;

	auto inLayer = Layer::make<ConstantLayer>(INPUT_DIM);
	auto lossLayer = Layer::make<SquareLossLayer>(TARGET_DIM);

	auto engine = EngineBase::make<EigenEngine>();
	auto dataman = DataManagerBase::make<EigenRandDataManager>(
			engine, INPUT_DIM, TARGET_DIM, BATCH);

	RecurrentNetwork net(engine, dataman, HISTORY);

	net.add_layer(inLayer);

	// pointer composite semantics
	auto lstm1 = Composite<RecurrentNetwork>::make<LstmComposite>(inLayer, 4);
	net.add_composite(lstm1);

	// object composite semantics
	auto lstm2 = Composite<RecurrentNetwork>::create<LstmComposite>(lstm1->out_layer(), 7);
	net.add_composite(lstm2);
	auto lstm3 = Composite<RecurrentNetwork>::create<LstmComposite>(lstm2.out_layer(), 3);
	net.add_composite(lstm3);

	net.new_connection<FullConnection>(lstm3.out_layer(), lossLayer);
	net.add_layer(lossLayer);

	gradient_check<EigenEngine, EigenRandDataManager>(net, 1e-2f, 2.f);
}

/**
 * Fancy "Diamond" LSTM network
 */
TEST(EigenLSTM, Diamond)
{
	// Pass in global parameter to lmn::fill_rand()
	// uniformly random initialize connections: -1.f to +1.f
	FakeRand::instance_passin().set_rand_seq({1.f});

	const int HISTORY = 5;
	const int INPUT_DIM = 2;
	const int TARGET_DIM = 4;
	const int BATCH = 2;

	auto inLayer = Layer::make<ConstantLayer>(INPUT_DIM);
	auto prelossLayer = Layer::make<CosineLayer>(4);
	auto lossLayer = Layer::make<LabelSoftmaxEntropyLayer>(TARGET_DIM);

	auto engine = EngineBase::make<EigenEngine>();
	auto dataman = DataManagerBase::make<EigenRandDataManager>(
			engine, INPUT_DIM, 1, BATCH, TARGET_DIM);

	RecurrentNetwork net(engine, dataman, HISTORY);

	net.add_layer(inLayer);

	// pointer composite semantics
	auto lstm1 = Composite<RecurrentNetwork>::make<LstmComposite>(inLayer, 4);
	net.add_composite(lstm1);

	// object composite semantics
	auto lstm2 = Composite<RecurrentNetwork>::create<LstmComposite>(lstm1->out_layer(), 7);
	auto lstm3 = Composite<RecurrentNetwork>::create<LstmComposite>(lstm1->out_layer(), 3);

	net.add_composite(lstm2);
	net.add_composite(lstm3);

	net.new_connection<FullConnection>(lstm2.out_layer(), prelossLayer);
	net.new_connection<FullConnection>(lstm3.out_layer(), prelossLayer);
	net.add_layer(prelossLayer);

	net.new_connection<FullConnection>(prelossLayer, lossLayer);
	net.add_layer(lossLayer);

	gradient_check<EigenEngine, EigenRandDataManager>(net, 1e-2f, 2.f);
}
