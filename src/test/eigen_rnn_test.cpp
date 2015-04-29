/*
 * Eona Studio (c) 2015
 */

#include "eigen_helper.h"

TEST(EigenRNN, Simple)
{
	const int HISTORY = 5;
	const int INPUT_DIM = 3;
	const int TARGET_DIM = 2;
	const int BATCH = 2;

	auto l1 = Layer::make<ConstantLayer>(INPUT_DIM);
	auto l2 = Layer::make<TanhLayer>(2);
	auto l3 = Layer::make<SigmoidLayer>(3);
	auto l4 = Layer::make<LabelSoftmaxEntropyLayer>(TARGET_DIM);

	// Naming: c<in><out>_<skip>
	auto c12 = Connection::make<FullConnection>(l1, l2);
	auto c23 = Connection::make<FullConnection>(l2, l3);
	auto c34 = Connection::make<FullConnection>(l3, l4);

	auto c22_1 = Connection::make<FullConnection>(l2, l2);
	auto c23_1 = Connection::make<FullConnection>(l2, l3);
	auto c33_1 = Connection::make<FullConnection>(l3, l3);

	auto engine = EngineBase::make<EigenEngine>();
	auto dataman = DataManagerBase::make<EigenRandDataManager>(
			engine, INPUT_DIM, 1, BATCH, TARGET_DIM);

	RecurrentNetwork net(engine, dataman, HISTORY);

	net.add_layer(l1);
	net.add_recur_connection(c22_1);
	net.add_connection(c12);

	net.add_layer(l2);

	net.add_recur_connection(c23_1);
	net.add_recur_connection(c33_1);
	net.add_connection(c23);

	net.add_layer(l3);
	net.add_connection(c34);
	net.add_layer(l4);

	gradient_check<EigenEngine, EigenRandDataManager, float>(net, 1e-2, 2.f);
}

TEST(EigenRNN, TemporalSkip)
{
	const int HISTORY = 5;
	const int INPUT_DIM = 2;
	const int TARGET_DIM = 4;
	const int BATCH = 2;

	auto l1 = Layer::make<ConstantLayer>(INPUT_DIM);
	auto l2 = Layer::make<SigmoidLayer>(3);
	auto l3 = Layer::make<CosineLayer>(2);
	auto l4 = Layer::make<LabelSoftmaxEntropyLayer>(TARGET_DIM);

	// NOTE IMPORTANT RULE
	// For recurrent linear connection conn[layer(alpha) => layer(beta)]
	// Must be added before you add layer(beta). alpha doesn't matter

	// Naming: c<in><out>_<skip>
	auto c12 = conn_full(l1, l2);
	auto c23 = conn_full(l2, l3);
	auto c34 = conn_full(l3, l4);

	auto c22_1 = conn_full(l2, l2);
	auto c22_3 = conn_full(l2, l2);
	auto c23_1 = conn_full(l2, l3);
	auto c23_2 = conn_full(l2, l3);
	auto c32_3 = conn_full(l3, l2);
	auto c33_1 = conn_full(l3, l3);
	auto c33_2 = conn_full(l3, l3);

	auto engine = EngineBase::make<EigenEngine>();
	auto dataman = DataManagerBase::make<EigenRandDataManager>(
			engine, INPUT_DIM, 1, BATCH, TARGET_DIM);

	RecurrentNetwork net(engine, dataman, HISTORY);

	net.init_max_temporal_skip(3); // or Layer::UNLIMITED_TEMPORAL_SKIP

	net.add_layer(l1);

	net.add_connection(c12);
	net.add_recur_connection(c22_1);
	net.add_recur_connection(c22_3, 3);
	net.add_recur_connection(c32_3, 3);

	net.add_layer(l2);

	net.add_connection(c23);
	net.add_recur_connection(c23_1);
	net.add_recur_connection(c23_2, 2);
	net.add_recur_connection(c33_1);
	net.add_recur_connection(c33_2, 2);

	net.add_layer(l3);
	net.add_connection(c34);
	net.add_layer(l4);

	gradient_check<EigenEngine, EigenRandDataManager, float>(net, 1e-2, 2.f);
}


TEST(EigenRNN, TemporalSkipBias)
{

	const int HISTORY = 5;
	const int INPUT_DIM = 2;
	const int TARGET_DIM = 4;
	const int BATCH = 2;

	auto l1 = Layer::make<ConstantLayer>(INPUT_DIM);
	auto l2 = Layer::make<SigmoidLayer>(3);
	auto l3 = Layer::make<CosineLayer>(2);
	auto l4 = Layer::make<SquareLossLayer>(TARGET_DIM);

	// NOTE IMPORTANT RULE
	// For recurrent linear connection conn[layer(alpha) => layer(beta)]
	// Must be added before you add layer(beta). alpha doesn't matter

	auto engine = EngineBase::make<EigenEngine>();
	auto dataman = DataManagerBase::make<EigenRandDataManager>(
			engine, INPUT_DIM, TARGET_DIM, BATCH);

	RecurrentNetwork net(engine, dataman, HISTORY);

	net.init_max_temporal_skip(3); // or Layer::UNLIMITED_TEMPORAL_SKIP

	net.add_layer(l1);

	net.new_connection<FullConnection>(l1, l2);

	net.new_recur_connection<FullConnection>(l2, l2);
	net.new_recur_skip_connection<FullConnection>(3, l2, l2);
	net.new_recur_skip_connection<FullConnection>(3, l3, l2);

	net.new_bias_layer(l2);
	net.add_layer(l2);

	net.new_connection<FullConnection>(l2, l3);
	net.new_recur_connection<FullConnection>(l2, l3);
	net.new_recur_skip_connection<FullConnection>(2, l2, l3);
	net.new_recur_skip_connection<FullConnection>(1, l3, l3);
	net.new_recur_skip_connection<FullConnection>(2, l3, l3);

	net.new_bias_layer(l3);
	net.add_layer(l3);

	net.new_connection<FullConnection>(l3, l4);

	net.new_bias_layer(l4);
	net.add_layer(l4);

	gradient_check<EigenEngine, EigenRandDataManager, float>(net, 1e-2, 2.f);
}


TEST(EigenRNN, GatedTanhConnection)
{
	const int HISTORY = 5;
	const int INPUT_DIM = 3;
	const int TARGET_DIM = 1;
	const int BATCH = 3;

	auto l1 = Layer::make<ConstantLayer>(INPUT_DIM);

	// NOTE layers engaged in a gate must have the same dims
	auto l2 = Layer::make<ScalarLayer>(TARGET_DIM, 2.3f);
	auto l3 = Layer::make<CosineLayer>(TARGET_DIM); // gate

	auto l4 = Layer::make<SquareLossLayer>(TARGET_DIM);

	auto g234 = Connection::make<GatedTanhConnection>(l2, l3, l4);
	auto g234_1 = Connection::make<GatedTanhConnection>(l2, l3, l4);
	auto g234_2 = Connection::make<GatedTanhConnection>(l2, l3, l4);

	auto engine = EngineBase::make<EigenEngine>();
	auto dataman = DataManagerBase::make<EigenRandDataManager>(
			engine, INPUT_DIM, TARGET_DIM, BATCH);

	RecurrentNetwork net(engine, dataman, HISTORY, 2);

	net.add_layer(l1);
	net.new_connection<FullConnection>(l1, l2);
	net.add_layer(l2);
	net.new_connection<FullConnection>(l1, l3);
	net.add_layer(l3);
	net.add_connection(g234);
	net.add_recur_connection(g234_1);
	net.add_recur_connection(g234_2, 2);
	net.add_layer(l4);

	gradient_check<EigenEngine, EigenRandDataManager>(net, 1e-2f, 2.f);
}


TEST(EigenRNN, GatedTanhBias)
{
	const int HISTORY = 5;
	const int INPUT_DIM = 3;
	const int TARGET_DIM = 2;
	const int BATCH = 4;

	auto l1 = Layer::make<ConstantLayer>(INPUT_DIM);

	// NOTE layers engaged in a gate must have the same dims
	auto l2 = Layer::make<CosineLayer>(TARGET_DIM);
	auto l3 = Layer::make<TanhLayer>(TARGET_DIM); // gate

	auto l4 = Layer::make<LabelSoftmaxEntropyLayer>(TARGET_DIM);

	auto g234 = Connection::make<GatedTanhConnection>(l2, l3, l4);
	auto g234_1 = Connection::make<GatedTanhConnection>(l2, l3, l4);
	auto g234_2 = Connection::make<GatedTanhConnection>(l2, l3, l4);

	auto engine = EngineBase::make<EigenEngine>();
	auto dataman = DataManagerBase::make<EigenRandDataManager>(
			engine, INPUT_DIM, 1, BATCH, TARGET_DIM);

	RecurrentNetwork net(engine, dataman, HISTORY, 2);

	net.add_layer(l1);
	net.new_connection<FullConnection>(l1, l2);
	net.new_bias_layer(l2);
	net.add_layer(l2);
	net.new_connection<FullConnection>(l1, l3);
	net.new_bias_layer(l3);
	net.add_layer(l3);
	net.add_connection(g234);
	net.add_recur_connection(g234_1);
	net.add_recur_connection(g234_2, 2);
	net.new_bias_layer(l4);
	net.add_layer(l4);

	gradient_check<EigenEngine, EigenRandDataManager>(net, 1e-2f, 2.f);
}
