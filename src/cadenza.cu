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
	net.add_connection(make_connection<LinearConnection>(l1, l3_1));
	net.add_connection(make_connection<LinearConnection>(l1, l3_2));
	net.add_connection(make_connection<LinearConnection>(l1, l4));
	net.add_layer(l2_1);
	net.add_layer(l2_2);
	net.add_connection(make_connection<LinearConnection>(l2_1, l3_1));
	net.add_connection(make_connection<LinearConnection>(l2_1, l3_2));
	net.add_connection(make_connection<LinearConnection>(l2_1, l4));
	net.add_connection(make_connection<LinearConnection>(l2_2, l3_2));
	net.add_connection(make_connection<LinearConnection>(l2_2, l3_1));
	net.add_connection(make_connection<LinearConnection>(l2_2, l4));
	net.add_layer(l3_1);
	net.add_layer(l3_2);
	net.add_connection(make_connection<LinearConnection>(l3_1, l4));
	net.add_connection(make_connection<LinearConnection>(l3_2, l4));
	net.add_layer(l4);

	gradient_check(net, 1e-2);
}
