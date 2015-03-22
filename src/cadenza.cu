/*
 * Eona Studio (c)2015
 */
#include "global_utils.h"
#include "timer.h"
#include "connection.h"
#include "input_layer.h"
#include "output_layer.h"
#include "recurrent_layer.h"
#include "transfer_layer.h"
#include "loss_layer.h"
#include "parameter.h"
#include "lstm_layer.h"

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

	l1.forward();
	conn12.forward();
	s2.forward();
	conn23.forward();
	l3.forward();
	conn34.forward();
	sq4.forward();

	sq4.backward();
	conn34.backward();
	l3.backward();
	conn23.backward();
	s2.backward();
	conn12.backward();
	l1.backward();

	cout << sq4.outValue << endl;
	cout << l1.inGradient << endl;
}
