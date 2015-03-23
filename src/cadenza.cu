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

int main(int argc, char **argv)
{
	float input = 0.2;
	float target = 1.5;

	auto l1 = make_layer<LinearLayer>();
	auto s2 = make_layer<SigmoidLayer>();
	auto l3 = make_layer<LinearLayer>();
	auto sq4 = make_layer<SquareLossLayer>();

	ForwardNetwork net;
	net.set_input(input);
	net.set_target(target);

	net.add_layer(l1);
	net.add_connection(make_connection<LinearConnection>(l1, s2));
	net.add_layer(s2);
	net.add_connection(make_connection<LinearConnection>(s2, l3));
	net.add_layer(l3);
	net.add_connection(make_connection<ConstantConnection>(l3, sq4));
	net.add_layer(sq4);

	gradient_check(net, 1e-3f);

//	cout << net << endl;
}
