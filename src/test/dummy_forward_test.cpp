/*
 * Eona Studio (c) 2015
 */

#include "test_utils.h"

FakeRand& rand_conn = FakeRand::instance_connection();
FakeRand& rand_input = FakeRand::instance_input();
FakeRand& rand_target = FakeRand::instance_target();

TEST(DummyForward, Diamond)
{
	rand_conn.set_rand_seq(
			vector<float> { 2.51, 5.39, 5.80, -2.96, -2.73, -2.4, 0.55, -.47 });

	rand_input.set_rand_seq(
			vector<float> { 0.2 });

	rand_target.set_rand_seq(
			vector<float> { 1.5 });

	auto dummyEng = EngineBase::make<DummyEngine>();
	auto dummyData = DataManagerBase::make<DummyDataManager>(dummyEng);

	auto l1 = Layer::make<ConstantLayer>(DUMMY_DIM);
	auto l2_1 = Layer::make<ScalorLayer>(DUMMY_DIM, 1.7f);
	auto l2_2 = Layer::make<CosineLayer>(DUMMY_DIM);
	auto l3_1 = Layer::make<SigmoidLayer>(DUMMY_DIM);
	auto l3_2 = Layer::make<ScalorLayer>(DUMMY_DIM, -2.3f);
	auto l4 = Layer::make<SquareLossLayer>(DUMMY_DIM);

	ForwardNetwork net(dummyEng, dummyData);

	net.add_layer(l1);
	net.add_connection(Connection::make<FullConnection>(l1, l2_1));
	net.add_connection(Connection::make<FullConnection>(l1, l2_2));
	// same as add_connection(make_connection<>)
	net.new_connection<FullConnection>(l1, l3_1);
	net.new_connection<FullConnection>(l1, l3_2);
	net.new_connection<FullConnection>(l1, l4);
	net.add_layer(l2_1);
	net.add_layer(l2_2);
	net.new_connection<FullConnection>(l2_1, l3_1);
	net.new_connection<FullConnection>(l2_1, l3_2);
	net.new_connection<FullConnection>(l2_1, l4);
	net.new_connection<FullConnection>(l2_2, l3_2);
	net.new_connection<FullConnection>(l2_2, l3_1);
	net.new_connection<FullConnection>(l2_2, l4);
	net.add_layer(l3_1);
	net.add_layer(l3_2);
	net.new_connection<FullConnection>(l3_1, l4);
	net.new_connection<FullConnection>(l3_2, l4);
	net.add_layer(l4);

	gradient_check<DummyEngine, DummyDataManager>(net, 1e-2f, 0.3f);
}


