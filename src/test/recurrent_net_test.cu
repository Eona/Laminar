/*
 * Eona Studio (c) 2015
 */

#include "test_utils.h"

TEST(RecurrentNet, TwoLayer)
{
	FakeRand::instance().set_rand_seq(
			vector<float> { 2.51, 5.39, 5.80, -2.96, -2.73, -2.4 });

	vector<float> input { 1.2, -0.9, 0.57, -1.47, -3.08 };
	vector<float> target { 1.39, 0.75, -0.45, -0.11, 1.55 };

	auto l1 = make_layer<LinearLayer>();
	auto l2 = make_layer<SigmoidLayer>();
	auto l3 = make_layer<SigmoidLayer>();
	auto l4 = make_layer<SquareLossLayer>();

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
