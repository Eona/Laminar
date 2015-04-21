/*
 * Eona Studio (c) 2015
 */

#include "test_utils.h"

FakeRand& rand_conn = FakeRand::instance_connection();
FakeRand& rand_prehis = FakeRand::instance_prehistory();
FakeRand& rand_input = FakeRand::instance_input();
FakeRand& rand_target = FakeRand::instance_target();

TEST(DummyRNN, Simple)
{
	rand_conn.set_rand_seq(vector<float> {
		0.543, 0.44, 1.47, 1.64, 1.31, -0.616
	});
	rand_prehis.set_rand_seq(vector<float> {
		.7
	});

	vector<float> inputSeq { 1.2, -0.9, 0.57, -1.47, -3.08 };
	vector<float> targetSeq { 1.39, 0.75, -0.45, -0.11, 1.55 };

	rand_input.set_rand_seq(inputSeq);
	rand_target.set_rand_seq(targetSeq);

	auto l1 = Layer::make<ConstantLayer>(DUMMY_DIM);
	auto l2 = Layer::make<SigmoidLayer>(DUMMY_DIM);
	auto l3 = Layer::make<SigmoidLayer>(DUMMY_DIM);
	auto l4 = Layer::make<SquareLossLayer>(DUMMY_DIM);

	// Naming: c<in><out>_<skip>
	auto c12 = Connection::make<FullConnection>(l1, l2);
	auto c23 = Connection::make<FullConnection>(l2, l3);
	auto c34 = Connection::make<FullConnection>(l3, l4);

	auto c22_1 = Connection::make<FullConnection>(l2, l2);
	auto c23_1 = Connection::make<FullConnection>(l2, l3);
	auto c33_1 = Connection::make<FullConnection>(l3, l3);

	auto dummyEng = EngineBase::make<DummyEngine>();
	auto dummyData = DataManagerBase::make<DummyDataManager>(dummyEng);

	RecurrentNetwork net(dummyEng, dummyData, inputSeq.size());

	net.add_layer(l1);
	net.add_recurrent_connection(c22_1);
	net.add_connection(c12);

	net.add_layer(l2);

	net.add_recurrent_connection(c23_1);
	net.add_recurrent_connection(c33_1);
	net.add_connection(c23);

	net.add_layer(l3);
	net.add_connection(c34);
	net.add_layer(l4);
/*
	RecurrentNetwork net;
	net.set_input(input);
	net.set_target(target);
	net.add_layer(l1);
	net.new_connection<FullConnection>(l1, l2);
	net.new_recurrent_connection<FullConnection>(l2, l2);
	net.add_layer(l2);
	net.new_recurrent_connection<FullConnection>(l2, l3);
	net.new_connection<FullConnection>(l2, l3);
	net.new_recurrent_connection<FullConnection>(l3, l3);
	net.add_layer(l3);
	net.new_connection<FullConnection>(l3, l4);
	net.add_layer(l4);
*/
	gradient_check<DummyEngine, DummyDataManager, float>(net, 1e-2, 1);
}

TEST(DummyRNN, TemporalSkip)
{
	rand_conn.set_rand_seq(vector<float> {
		0.91, 1.329, -0.525, 1.724, 1.613, -0.864, 0.543, 0.59, -0.819, -0.938
	});

	rand_prehis.set_rand_seq(vector<float> {
		.3
	});

//	rand_conn.gen_uniform_rand(10, -1, 2);
//	rand_conn.print_rand_seq();

	vector<float> inputSeq { 1.2, -0.9, 0.57, -1.47, -3.08, 1.2, .31, -2.33, -0.89 };
	vector<float> targetSeq { 1.39, 0.75, -0.45, -0.11, 1.55, -.44, 2.39, 1.72, -3.06 };

	rand_input.set_rand_seq(inputSeq);
	rand_target.set_rand_seq(targetSeq);

	auto l1 = Layer::make<ConstantLayer>(DUMMY_DIM);
	auto l2 = Layer::make<SigmoidLayer>(DUMMY_DIM);
	auto l3 = Layer::make<CosineLayer>(DUMMY_DIM);
	auto l4 = Layer::make<SquareLossLayer>(DUMMY_DIM);

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

	auto dummyEng = EngineBase::make<DummyEngine>();
	auto dummyData = DataManagerBase::make<DummyDataManager>(dummyEng);

	RecurrentNetwork net(dummyEng, dummyData, inputSeq.size());
	net.init_max_temporal_skip(3); // or Layer::UNLIMITED_TEMPORAL_SKIP

	net.add_layer(l1);

	net.add_connection(c12);
	net.add_recurrent_connection(c22_1);
	net.add_recurrent_connection(c22_3, 3);
	net.add_recurrent_connection(c32_3, 3);

	net.add_layer(l2);

	net.add_connection(c23);
	net.add_recurrent_connection(c23_1);
	net.add_recurrent_connection(c23_2, 2);
	net.add_recurrent_connection(c33_1);
	net.add_recurrent_connection(c33_2, 2);

	net.add_layer(l3);
	net.add_connection(c34);
	net.add_layer(l4);

/*	net.add_layer(l1);

	net.new_recurrent_connection<FullConnection>(l2, l2);
	net.new_recurrent_skip_connection<FullConnection>(3, l2, l2);
	net.new_recurrent_skip_connection<FullConnection>(3, l3, l2);
	net.new_connection<FullConnection>(l1, l2);

	net.add_layer(l2);

	net.new_connection<FullConnection>(l2, l3);
	net.new_recurrent_skip_connection<FullConnection>(2, l2, l3);
	net.new_recurrent_connection<FullConnection>(l2, l3);
	net.new_recurrent_connection<FullConnection>(l3, l3);
	net.new_recurrent_skip_connection<FullConnection>(2, l3, l3);

	net.add_layer(l3);
	net.new_connection<FullConnection>(l3, l4);
	net.add_layer(l4);*/

	gradient_check<DummyEngine, DummyDataManager, float>(net, 1e-2, 1);
}

TEST(DummyRNN, GatedConnection)
{
	rand_conn.set_rand_seq(vector<float> {
		0.559, -0.259, -0.736, -0.317, 0.306, -0.509, 0.569, -0.713, -0.83, -0.401
	});

//	rand_conn.gen_uniform_rand(10, -1, 1);
//	rand_conn.print_rand_seq();

	rand_prehis.set_rand_seq(vector<float> {
		.3
	});

	vector<float> inputSeq { 1.2, -0.9, 0.57, -1.47, -3.08, 1.2, .31, -2.33, -0.89 };
	vector<float> targetSeq { 1.39, 0.75, -0.45, -0.11, 1.55, -.44, 2.39, 1.72, -3.06 };

	rand_input.set_rand_seq(inputSeq);
	rand_target.set_rand_seq(targetSeq);

	auto l1 = Layer::make<ConstantLayer>(DUMMY_DIM);
	auto l2 = Layer::make<SigmoidLayer>(DUMMY_DIM);
	auto l3 = Layer::make<CosineLayer>(DUMMY_DIM); // gate
	auto l4 = Layer::make<SquareLossLayer>(DUMMY_DIM);

	// NOTE IMPORTANT RULE
	// For recurrent gated connection conn[layer(alpha), layer(gate) => layer(beta)]
	// where alpha is t-1 (or more) and gate/beta are the current t
	// Must be added after you add gate & alpha, and before beta.
	// If recurrent, alpha doesn't necessary need to precede this connection
	// (because layer alpha lives in the past)

	// Naming: c<in><out>_<skip>
	// g<in><gate><out>_<skip>
	auto c12 = conn_full(l1, l2);
	auto c13 = conn_full(l1, l3);

	auto g234 = Connection::make<GatedConnection>(l2, l3, l4);
	auto g234_1 = Connection::make<GatedConnection>(l2, l3, l4);
	auto g234_2 = Connection::make<GatedConnection>(l2, l3, l4);

	auto dummyEng = EngineBase::make<DummyEngine>();
	auto dummyData = DataManagerBase::make<DummyDataManager>(dummyEng);

	RecurrentNetwork net(dummyEng, dummyData, inputSeq.size(), 2);

	net.add_layer(l1);

	net.add_connection(c13);
	net.add_layer(l3);

	net.add_connection(c12);
	net.add_layer(l2);

	net.add_connection(g234);
	net.add_recurrent_connection(g234_1);
	net.add_recurrent_connection(g234_2, 2);

	net.add_layer(l4);

	gradient_check<DummyEngine, DummyDataManager>(net, 1e-2f, 1.f);
}


TEST(DummyRNN, GatedTanhConnection)
{
	rand_conn.set_rand_seq(vector<float> {
			.798, 0.617
	});

	rand_prehis.set_rand_seq(vector<float> {
		.3
	});

	vector<float> inputSeq { 1.2, -0.9, 0.57, -1.47, -3.08 };
	vector<float> targetSeq { 1.39, 0.75, -0.45, -0.11, 1.55 };

	rand_input.set_rand_seq(inputSeq);
	rand_target.set_rand_seq(targetSeq);

	auto l1 = Layer::make<ConstantLayer>(DUMMY_DIM);
	auto l2 = Layer::make<ScalorLayer>(DUMMY_DIM, 1.3f);
	auto l3 = Layer::make<CosineLayer>(DUMMY_DIM); // gate
	auto l4 = Layer::make<SquareLossLayer>(DUMMY_DIM);

	auto c12 = conn_full(l1, l2);
	auto c13 = conn_full(l1, l3);
	auto c24 = conn_full(l2, l4);

	auto g234_1 = Connection::make<GatedTanhConnection>(l2, l3, l4);
	auto g234_2 = Connection::make<GatedTanhConnection>(l2, l3, l4);

	auto dummyEng = EngineBase::make<DummyEngine>();
	auto dummyData = DataManagerBase::make<DummyDataManager>(dummyEng);

	RecurrentNetwork net(dummyEng, dummyData, inputSeq.size(), 2);

	net.add_layer(l1);
	net.add_connection(c13);
	net.add_layer(l3);
	net.add_connection(c12);
	net.add_layer(l2);
	net.add_connection(c24);
	net.add_recurrent_connection(g234_1);
	net.add_recurrent_connection(g234_2, 2);
	net.add_layer(l4);

	gradient_check<DummyEngine, DummyDataManager>(net, 1e-2f, 1.f);
}
