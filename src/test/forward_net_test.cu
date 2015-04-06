/*
 * Eona Studio (c) 2015
 */

#include "test_utils.h"

TEST(ForwardNet, Interconnected)
{
	FakeRand::instance_connection().set_rand_seq(
			vector<float> { 2.51, 5.39, 5.80, -2.96, -2.73, -2.4, 0.55, -.47 });

	float input = 0.2;
	float target = 1.5;

	auto l1 = Layer::make<LinearLayer>();
	auto l2_1 = Layer::make<LinearLayer>(1.7f);
	auto l2_2 = Layer::make<CosineLayer>();
	auto l3_1 = Layer::make<SigmoidLayer>();
	auto l3_2 = Layer::make<LinearLayer>(-2.3f);
	auto l4 = Layer::make<SquareLossLayer>();

	ForwardNetwork net;
	net.set_input(input);
	net.set_target(target);

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

	gradient_check(net, 1e-2, 0.3);
}


