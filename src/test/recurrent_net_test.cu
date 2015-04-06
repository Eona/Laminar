/*
 * Eona Studio (c) 2015
 */

#include "test_utils.h"

TEST(RecurrentNet, Simple)
{
	FakeRand::instance().set_rand_seq(vector<float> {
		0.527, 1.54, 1.78, 1.45, 2.56, -0.874, -0.861, -0.0736
	});

	vector<float> input { 1.2, -0.9, 0.57, -1.47, -3.08 };
	vector<float> target { 1.39, 0.75, -0.45, -0.11, 1.55 };

	auto l1 = Layer::make<LinearLayer>();
	auto l2 = Layer::make<SigmoidLayer>();
	auto l3 = Layer::make<SigmoidLayer>();
	auto l4 = Layer::make<SquareLossLayer>();

	RecurrentNetwork net;
	net.set_input(input);
	net.set_target(target);

	net.add_layer(l1);
	net.new_connection<LinearConnection>(l1, l2);
	net.add_layer(l2);
	net.new_connection<LinearConnection>(l2, l3);
	net.add_layer(l3);
	net.new_connection<LinearConnection>(l3, l4);
	net.add_layer(l4);

	net.new_recurrent_connection<LinearConnection>(l2, l2);
	net.new_recurrent_connection<LinearConnection>(l2, l3);
	net.new_recurrent_connection<LinearConnection>(l3, l3);

	gradient_check(net, 1e-2, 1);
}

TEST(RecurrentNet, TemporalSkip)
{
	FakeRand::instance().set_rand_seq(vector<float> {
		1.39, 0.27, 1.1, -0.317, -1.41, 0.0649, 0.777, 3.58, 0.18, 1.72, -0.963, 1.46, -0.685, -0.0132, 0.471, 0.783, 0.19
	});

	vector<float> input { 1.2, -0.9, 0.57, -1.47, -3.08, 1.2, .31, -2.33, -0.89 };
	vector<float> target { 1.39, 0.75, -0.45, -0.11, 1.55, -.44, 2.39, 1.72, -3.06 };

	auto l1 = Layer::make<LinearLayer>();
	auto l2 = Layer::make<SigmoidLayer>();
	auto l3 = Layer::make<CosineLayer>();
	auto l4 = Layer::make<SquareLossLayer>();

	RecurrentNetwork net;
	net.set_input(input);
	net.set_target(target);
	net.set_max_temporal_skip(3);

	net.add_layer(l1);
	net.new_connection<LinearConnection>(l1, l2);
	net.add_layer(l2);
	net.new_connection<LinearConnection>(l2, l3);
	net.add_layer(l3);
	net.new_connection<LinearConnection>(l3, l4);
	net.add_layer(l4);

	net.new_recurrent_connection<LinearConnection>(l2, l2);
	net.new_recurrent_skip_connection<LinearConnection>(3, l2, l2);
	net.new_recurrent_skip_connection<LinearConnection>(2, l2, l3);
	net.new_recurrent_connection<LinearConnection>(l2, l3);
	net.new_recurrent_skip_connection<LinearConnection>(3, l3, l2);
	net.new_recurrent_connection<LinearConnection>(l3, l3);
	net.new_recurrent_skip_connection<LinearConnection>(2, l3, l3);

	gradient_check(net, 1e-2, 1);
}
