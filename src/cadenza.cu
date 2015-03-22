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

	auto l1 = makeLayer<LinearLayer>(input);
	auto s2 = makeLayer<SigmoidLayer>(0);
	auto l3 = makeLayer<LinearLayer>(0);
	auto sq4 = makeLayer<SquareErrorLayer>(0, target);

	ForwardNetwork net;
	net.addLayer(l1);
	net.addConnection(makeConnection<LinearConnection>(l1, s2));
	net.addLayer(s2);
	net.addConnection(makeConnection<LinearConnection>(s2, l3));
	net.addLayer(l3);
	net.addConnection(makeConnection<ConstantConnection>(l3, sq4));
	net.addLayer(sq4);

	net.forward_prop();
	net.backward_prop();

	cout << net << endl;
}
