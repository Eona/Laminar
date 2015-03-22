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
	LinearConnection conn12(l1, s2);
	LinearConnection conn23(s2, l3);

	l1.forward();
	conn12.forward();
	s2.forward();
	conn23.forward();
	l3.forward();

	cout << l3.outValue << endl;
}
