/*
 * Eona Studio (c)2015
 */
#include "ocl_util.h"
#include <iostream>
#include "opencl_engine.h"

using namespace std;
int main(int argc, char **argv)
{

	//create testcases
	float t1[9] = {1.1, 7.8, 5.9, 3.0, 2, 5, 6, 10, 5};
	float t2[9] = {0.1, 6.8, 4.9, 2.0, 1, 4, 5, 9, 4};
	float t3[6] = {1.1, 7.8, 5.9, 3.0, 2, 5};

	vector<int> dim = {3,3};
	OpenclEngine oe;
	OpenclFloatMatPtr m1 (new OpenclFloatMat(t1, 3, 3, oe.cl));
	OpenclFloatMatPtr m2 (new OpenclFloatMat(t2, 3, 3, oe.cl));
	OpenclFloatMatPtr m3 (new OpenclFloatMat(t3, 2, 3, oe.cl));
	OpenclFloatMatPtr out(new OpenclFloatMat());


    m1->print_matrix("m1");
    m2->print_matrix("m2");
    m3->print_matrix("m3");
//
	std::vector<OpenclFloatMatPtr> v;
	v.push_back(m1);
	v.push_back(m2);
//
	std::vector<OpenclFloatMatPtr> v1;
	v1.push_back(m3);
	v1.push_back(m1);
//

	oe.create(out, dim);

	oe.sub(v, out, true);
	out->print_matrix("m1 - m2");

	oe.add(v, out, true);
	out->print_matrix("m1 + m2");

	oe.negate(v, out, true);
	out->print_matrix("-m1");
	oe.negate(v, out, true);
	out->print_matrix("-m1");

//	oe.mult(v, out, true);
//	out->print_matrix("m1 * m2");
//
//	oe.mult(v1, out, false);
//	out->print_matrix("m3 * m1");

	oe.assign(v1, out, false);
	out->print_matrix("m3 -> out");

	oe.sigmoid(v, out, false);
	out->print_matrix("sigmod(m1)");

	oe.sigmoid_gradient(v, out, true);
	out->print_matrix("sigmoid_gradient(m1)");

	oe.sin(v, out, true);
	out->print_matrix("sin(m1)");

	oe.cos(v, out, true);
	out->print_matrix("cos(m1)");

	oe.tanh(v, out, true);
	out->print_matrix("tanh(m1)");

	oe.tanh_gradient(v, out, true);
	out->print_matrix("tanh_gradient(m1)");

	oe.element_mult(v, out, true);
	out->print_matrix("m1 .* m2");

//	float loss;
//	oe.square_loss(v, &loss, true);
//	cout<<"loss: "<<loss<<endl;

	oe.fill_rand(v, out, true);
	out->print_matrix("rand");

	oe.debug_fill(v, out, true);
	out->print_matrix("0.66337");
}
