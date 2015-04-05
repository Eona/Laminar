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
	vector<float> input { 1.2, -0.9, 0.57, -1.47, -3.08 };
	vector<float> target { 1.39, 0.75, -0.45, -0.11, 1.55 };

	auto l1 = Layer::make<LinearLayer>();
	auto l2 = Layer::make<LstmLayer>();
//	auto l3 = make_layer<SigmoidLayer>();
	auto l4 = Layer::make<SquareLossLayer>();

	RecurrentNetwork net;
	net.set_input(input);
	net.set_target(target);

	net.add_layer(l1);
	net.new_connection<LinearConnection>(l1, l2);
	net.add_layer(l2);
//	net.new_connection<LinearConnection>(l2, l3);
//	net.add_layer(l3);
	net.new_connection<LinearConnection>(l2, l4);
	net.add_layer(l4);

	cout << LstmLayer::_W_ci << endl;

//	net.new_recurrent_connection<LinearConnection>(l2, l3);
//	net.new_recurrent_connection<LinearConnection>(l3, l3);

//	net.assemble();
//	net.forward_prop();
//	net.forward_prop();
//	net.forward_prop();
//
//	net.backward_prop();
//	cout << net << endl;
//	net.backward_prop();
//	cout << net << endl;
//	net.backward_prop();
//	cout << net << endl;
//	cout << "Total loss = " << net.lossLayer->totalLoss << endl;
//	gradient_check(net, 1e-2);
}
