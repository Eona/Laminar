/*
 * Eona Studio (c)2015
 */
#include "global_utils.h"
#include "timer.h"
#include "connection.h"
#include "recurrent_layer.h"
#include "transfer_layer.h"
#include "loss_layer.h"
#include "parameter.h"
#include "lstm_layer.h"
#include "network.h"
#include "gradient_check.h"

int main(int argc, char **argv)
{
	vector<float> input { 0.2, 0.3, 0.5 };
	vector<float> target { 1.3, 1.5, 2.2 };

	auto l1 = make_layer<LinearLayer>();
	auto l2 = make_layer<SigmoidLayer>();
	auto l3 = make_layer<SquareLossLayer>();

	RecurrentNetwork net;
	net.set_input(input);
	net.set_target(target);

	net.add_layer(l1);
	net.new_connection<LinearConnection>(l1, l2);
	net.add_layer(l2);
	net.new_connection<LinearConnection>(l2, l3);
	net.add_layer(l3);

	net.new_recurrent_connection<LinearConnection>(l2, l2);

	net.assemble();
	net.forward_prop();
	net.forward_prop();
	net.forward_prop();
	net.backward_prop();
	net.backward_prop();
	net.backward_prop();
	cout << net << endl;
//	gradient_check(net, 1e-2);
}
