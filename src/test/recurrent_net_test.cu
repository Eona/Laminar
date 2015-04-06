/*
 * Eona Studio (c) 2015
 */

#include "test_utils.h"

TEST(RecurrentNet, Simple)
{
	FakeRand::instance_connection().set_rand_seq(vector<float> {
		0.543, 0.44, 1.47, 1.64, 1.31, -0.616
	});
	FakeRand::instance_prehistory().set_rand_seq(vector<float> {
		.7
	});
	FakeRand::instance_connection().use_fake_seq();
//	FakeRand::instance_connection().use_uniform_rand(-1, 2);
//	FakeRand::instance_connection().set_rand_display(true);

	vector<float> input { 1.2, -0.9, 0.57, -1.47, -3.08 };
	vector<float> target { 1.39, 0.75, -0.45, -0.11, 1.55 };

	auto l1 = Layer::make<LinearLayer>();
	auto l2 = Layer::make<SigmoidLayer>();
	auto l3 = Layer::make<SigmoidLayer>();
	auto l4 = Layer::make<SquareLossLayer>();

	// Naming: c<in><out>_<skip>
	auto c12 = Connection::make<LinearConnection>(l1, l2);
	auto c23 = Connection::make<LinearConnection>(l2, l3);
	auto c34 = Connection::make<LinearConnection>(l3, l4);

	auto c22_1 = Connection::make<LinearConnection>(l2, l2);
	auto c23_1 = Connection::make<LinearConnection>(l2, l3);
	auto c33_1 = Connection::make<LinearConnection>(l3, l3);

	RecurrentNetwork net;
	net.set_input(input);
	net.set_target(target);

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
	net.new_connection<LinearConnection>(l1, l2);
	net.new_recurrent_connection<LinearConnection>(l2, l2);
	net.add_layer(l2);
	net.new_recurrent_connection<LinearConnection>(l2, l3);
	net.new_connection<LinearConnection>(l2, l3);
	net.new_recurrent_connection<LinearConnection>(l3, l3);
	net.add_layer(l3);
	net.new_connection<LinearConnection>(l3, l4);
	net.add_layer(l4);
*/
	gradient_check(net, 1e-2, 1);
}

TEST(RecurrentNet, TemporalSkip)
{
	FakeRand::instance_connection().set_rand_seq(vector<float> {
		0.163, 1.96, 1.09, 0.516, -0.585, 0.776, 1, -0.301, -0.167, 0.732
	});

//	FakeRand::instance_connection().use_uniform_rand(-1, 2);
//	FakeRand::instance_connection().set_rand_display(true);
	FakeRand::instance_connection().use_fake_seq();

	FakeRand::instance_prehistory().set_rand_seq(vector<float> {
		.3
	});

	vector<float> input { 1.2, -0.9, 0.57, -1.47, -3.08, 1.2, .31, -2.33, -0.89 };
	vector<float> target { 1.39, 0.75, -0.45, -0.11, 1.55, -.44, 2.39, 1.72, -3.06 };

	auto l1 = Layer::make<LinearLayer>();
	auto l2 = Layer::make<SigmoidLayer>();
	auto l3 = Layer::make<CosineLayer>();
	auto l4 = Layer::make<SquareLossLayer>();

	// NOTE IMPORTANT RULE
	// For recurrent linear connection conn[layer(alpha) => layer(beta)]
	// Must be added before you add layer(beta).
	// Ideally you should add layer(alpha) before you add conn, but doesn't matter.

	// Naming: c<in><out>_<skip>
	auto c12 = Connection::make<LinearConnection>(l1, l2);
	auto c23 = Connection::make<LinearConnection>(l2, l3);
	auto c34 = Connection::make<LinearConnection>(l3, l4);

	auto c22_1 = Connection::make<LinearConnection>(l2, l2);
	auto c22_3 = Connection::make<LinearConnection>(l2, l2);
	auto c23_1 = Connection::make<LinearConnection>(l2, l3);
	auto c23_2 = Connection::make<LinearConnection>(l2, l3);
	auto c32_3 = Connection::make<LinearConnection>(l3, l2);
	auto c33_1 = Connection::make<LinearConnection>(l3, l3);
	auto c33_2 = Connection::make<LinearConnection>(l3, l3);

	RecurrentNetwork net;
	net.set_input(input);
	net.set_target(target);
	net.set_max_temporal_skip(3); // or Layer::UNLIMITED_TEMPORAL_SKIP

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

/*
	net.add_layer(l1);

	net.new_recurrent_connection<LinearConnection>(l2, l2);
	net.new_recurrent_skip_connection<LinearConnection>(3, l2, l2);
	net.new_recurrent_skip_connection<LinearConnection>(3, l3, l2);
	net.new_connection<LinearConnection>(l1, l2);

	net.add_layer(l2);

	net.new_connection<LinearConnection>(l2, l3);
	net.new_recurrent_skip_connection<LinearConnection>(2, l2, l3);
	net.new_recurrent_connection<LinearConnection>(l2, l3);
	net.new_recurrent_connection<LinearConnection>(l3, l3);
	net.new_recurrent_skip_connection<LinearConnection>(2, l3, l3);

	net.add_layer(l3);
	net.new_connection<LinearConnection>(l3, l4);
	net.add_layer(l4);
*/

	gradient_check(net, 1e-2, 1);

	net.reset();
	for (int i = 0; i < net.input.size(); ++i)
		net.forward_prop();
	for (int i = 0; i < net.input.size(); ++i)
		net.backward_prop();

/*	for (ConnectionPtr c : { c12, c23, c34, c22_1, c22_3, c23_1, c23_2, c32_3, c33_1, c33_2 })
		cout << std::setprecision(4) << Connection::cast<LinearConnection>(c)->gradient << "  ";
	cout << endl;
	for (LayerPtr l : { l2, l3 })
		cout << std::setprecision(4) << static_cast<ParamContainerPtr>(net.prehistoryLayerMap[l])->paramGradients << "  ";
	cout << endl;*/
}


TEST(RecurrentNet, GatedConnection)
{
	FakeRand::instance_connection().set_rand_seq(vector<float> {
			0.163, 1.96, 1.09, 0.516, -0.585, 0.776, 1, -0.301, -0.167, 0.732
	});

	FakeRand::instance_connection().use_fake_seq();

	FakeRand::instance_prehistory().set_rand_seq(vector<float> {
		.3
	});

	vector<float> input { 1.2, -0.9, 0.57, -1.47, -3.08, 1.2, .31, -2.33, -0.89 };
	vector<float> target { 1.39, 0.75, -0.45, -0.11, 1.55, -.44, 2.39, 1.72, -3.06 };

	auto l1 = Layer::make<LinearLayer>();
	auto l2 = Layer::make<SigmoidLayer>();
	auto l3 = Layer::make<CosineLayer>(); // gate
	auto l4 = Layer::make<SquareLossLayer>();

	// NOTE IMPORTANT RULE
	// For recurrent gated connection conn[layer(alpha), layer(gate) => layer(beta)]
	// where alpha is t-1 (or more) and gate/beta are the current t
	// Must be added after you add gate & alpha, and before beta.

	// Naming: c<in><out>_<skip>
	// g<in><gate><out>_<skip>
	auto c12 = Connection::make<LinearConnection>(l1, l2);
	auto c13 = Connection::make<LinearConnection>(l1, l3);

	auto g234_1 = Connection::make<GatedConnection>(l2, l3, l4);
	auto g234_2 = Connection::make<GatedConnection>(l2, l3, l4);

	RecurrentNetwork net;
	net.set_input(input);
	net.set_target(target);
	net.set_max_temporal_skip(3);

	net.add_layer(l1);

	net.add_connection(c13);
	net.add_layer(l3);

	net.add_connection(c12);
	net.add_layer(l2);

	net.add_recurrent_connection(g234_1);
	net.add_recurrent_connection(g234_2, 2);

	net.add_layer(l4);

	gradient_check(net, 1e-2, 1);
}
