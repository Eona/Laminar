/*
 * Eona Studio (c) 2015
 */

#include "test_utils.h"

FakeRand& rand_conn = FakeRand::instance_connection();
FakeRand& rand_input = FakeRand::instance_input();
FakeRand& rand_target = FakeRand::instance_target();

TEST(VecmatForward, Diamond)
{
	rand_conn.set_rand_seq(vector<float> {
		0.869, -0.764, -0.255, 0.771, -0.913, 0.294, -0.957, 0.958, -0.388, -0.184,
		0.922, 0.434, 0.217, 0.655, 0.707, 0.655, 0.368, -0.383, -0.838,
		0.638, -0.706, 0.429, -0.72, -0.439, 0.429, -0.977, 0.858, -0.937,
		0.381, -0.973, 0.764, -0.776, 0.907, 0.483, -0.573, -0.728, 0.587,
		0.102, -0.763, 0.939, 0.876, 0.195, 0.423, 0.0761, -0.364, 0.0478,
		0.558, 0.0241, -0.13, 0.591, -0.294, -0.762, 0.741, 0.0955, 0.784,
		0.398, 0.475, -0.199, -0.533, -0.483, -0.939, -0.344
	});
//	rand_conn.gen_uniform_rand(62, -1, 1);
//	rand_conn.print_rand_seq();

	rand_input.set_rand_seq(vector<float> {
		0.276, 2.54, 2.27, 2.81, -0.0979, 0.205
	});
//	rand_input.gen_uniform_rand(6, -1, 3);
//	rand_input.print_rand_seq();

	rand_target.set_rand_seq(vector<float> {
		0.457, -0.516, -0.312, 0.126
	});
//	rand_target.gen_uniform_rand(4, -1, 3);
//	rand_target.print_rand_seq();

	const int INPUT_DIM = 3;
	const int TARGET_DIM = 2;
	const int BATCH_SIZE = 2;

	auto engine = EngineBase::make<VecmatEngine>();
	auto dataman = DataManagerBase::make<VecmatDataManager>(
					engine, INPUT_DIM, TARGET_DIM, BATCH_SIZE);

	auto l1 = Layer::make<ConstantLayer>(INPUT_DIM);

	auto l2_1 = Layer::make<ScalorLayer>(1, 1.7f);
	auto l2_2 = Layer::make<CosineLayer>(3);
	auto l3_1 = Layer::make<SigmoidLayer>(2);
	auto l3_2 = Layer::make<ScalorLayer>(2, -2.3f);

	auto l4 = Layer::make<SquareLossLayer>(TARGET_DIM);

	ForwardNetwork net(engine, dataman);

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

	gradient_check<VecmatEngine, VecmatDataManager>(net, 1e-2f, 0.8f);
}


