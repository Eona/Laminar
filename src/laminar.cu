/*
 * Eona Studio (c)2015
 */
#include "global_utils.h"
#include "timer.h"
#include "connection.h"
#include "transfer_layer.h"
#include "loss_layer.h"
#include "parameter.h"
#include "lstm_layer.h"
#include "network.h"
#include "gradient_check.h"

int main(int argc, char **argv)
{
	FakeRand::instance().set_rand_seq(vector<float> {
		3.16, 2.90, -0.93, 3.75, 0.48, 1.90, -0.13, 3.75, -0.78, 1.28
	});

	vector<float> input { 1.2, -0.9, 0.57, -1.47, -3.08, 1.2, .31, -2.33, -0.89 };
	vector<float> target { 1.39, 0.75, -0.45, -0.11, 1.55, -.44, 2.39, 1.72, -3.06 };

	auto l1 = Layer::make<LinearLayer>();
	auto l2 = Layer::make<SigmoidLayer>();
	auto l3 = Layer::make<SigmoidLayer>(); // gate
	auto l4 = Layer::make<SquareLossLayer>();

	RecurrentNetwork net;
	net.set_input(input);
	net.set_target(target);
	net.set_max_temporal_skip(1);

	net.add_layer(l1);
	net.new_connection<LinearConnection>(l1, l2);
	net.new_connection<LinearConnection>(l1, l3);
	net.add_layer(l2);
	net.add_layer(l3);
//	net.new_connection<LinearConnection>(l2, l4);
//	net.new_connection<LinearConnection>(l3, l4);
	net.new_recurrent_connection<GatedConnection>(l2, l3, l4);
	net.add_layer(l4);

//	net.new_recurrent_connection<GatedConnection>(l2, l3, l4);

/*
	net.reset();
	net.forward_prop();
	cout << net << endl;
	net.forward_prop();
	cout << net << endl;
	net.forward_prop();
	cout << net << endl;
	net.forward_prop();
	cout << net << endl;
*/
	gradient_check(net, 1e-2, 1);
}
