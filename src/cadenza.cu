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
	LinearLayer l1(input);
	SigmoidLayer s2(0);
	LinearLayer l3(0);
	SquareErrorLayer sq4(0, 1.5f);
	LinearConnection conn12(l1, s2);
	LinearConnection conn23(s2, l3);
	ConstantConnection conn34(l3, sq4);

	ForwardNetwork net {&l1, &conn12, &s2, &conn23, &l3, &conn34, &sq4};

	net.forward_prop();
	net.backward_prop();

	cout << net << endl;
}
