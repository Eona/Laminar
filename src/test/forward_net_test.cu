/*
 * Eona Studio (c) 2015
 */

#include "test_utils.h"

TEST(ForwardNet, FourLayerInterconnected)
{
	FakeRand::instance().set_rand_seq(
			vector<float> { 2.51, 5.39, 5.80, -2.96, -2.73, -2.4, 0.55, -.47 });

	float input = 0.2;
	float target = 1.5;

	auto l1 = make_layer<LinearLayer>();
	auto l2_1 = make_layer<LinearLayer>(1.7f);
	auto l2_2 = make_layer<CosineLayer>();
	auto l3_1 = make_layer<SigmoidLayer>();
	auto l3_2 = make_layer<LinearLayer>(-2.3f);
	auto l4 = make_layer<SquareLossLayer>();

	ForwardNetwork net;
	net.set_input(input);
	net.set_target(target);

	net.add_layer(l1);
	net.add_connection(make_connection<LinearConnection>(l1, l2_1));
	net.add_connection(make_connection<LinearConnection>(l1, l2_2));
	// same as add_connection(make_connection<>)
	net.new_connection<LinearConnection>(l1, l3_1);
	net.new_connection<LinearConnection>(l1, l3_2);
	net.new_connection<LinearConnection>(l1, l4);
	net.add_layer(l2_1);
	net.add_layer(l2_2);
	net.new_connection<LinearConnection>(l2_1, l3_1);
	net.new_connection<LinearConnection>(l2_1, l3_2);
	net.new_connection<LinearConnection>(l2_1, l4);
	net.new_connection<LinearConnection>(l2_2, l3_2);
	net.new_connection<LinearConnection>(l2_2, l3_1);
	net.new_connection<LinearConnection>(l2_2, l4);
	net.add_layer(l3_1);
	net.add_layer(l3_2);
	net.new_connection<LinearConnection>(l3_1, l4);
	net.new_connection<LinearConnection>(l3_2, l4);
	net.add_layer(l4);

	gradient_check(net, 1e-2, 0.1);
}


