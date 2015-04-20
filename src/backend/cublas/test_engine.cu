/*
 * Eona Studio (c)2015
 */
#include "cuda_engine.h"
#include <iostream>

using namespace std;
typedef std::shared_ptr<CudaFloatMat> CudaFloatMatPtr;
int main(int argc, char **argv)
{

	//create testcases
	float t1[9] = {1.1, 7.8, 5.9, 3.0, 2, 5, 6, 10, 5};
	float t2[9] = {0.1, 6.8, 4.9, 2.0, 1, 4, 5, 9, 4};
	float t3[6] = {1.1, 7.8, 5.9, 3.0, 2, 5};


	CudaFloatMatPtr m1 (new CudaFloatMat(t1, 3, 3));
	CudaFloatMatPtr m2 (new CudaFloatMat(t2, 3, 3));
	CudaFloatMatPtr m3 (new CudaFloatMat(t3, 2, 3));
	CudaFloatMatPtr out(new CudaFloatMat());
    m1->print_matrix("m1");
    m2->print_matrix("m2");
    m3->print_matrix("m3");

	std::vector<CudaFloatMatPtr> v;
	v.push_back(m1);
	v.push_back(m2);

	std::vector<CudaFloatMatPtr> v1;
	v1.push_back(m3);
	v1.push_back(m1);
	CudaEngine ce;
	ce.add(v, out, false);
	out->print_matrix("m1 + m2");

	ce.sub(v, out, true);
	out->print_matrix("m1 - m2");

	ce.negate(v, out, true);
	out->print_matrix("-m1");

	ce.mult(v, out, true);
	out->print_matrix("m1 * m2");

	ce.mult(v1, out, false);
	out->print_matrix("m3 * m1");

	ce.assign(v1, out, true);
	out->print_matrix("m3 -> out");

	ce.sigmoid(v, out, false);
	out->print_matrix("sigmod(m1)");

	ce.sigmoid_gradient(v, out, true);
	out->print_matrix("sigmoid_gradient(m1)");

	ce.sin(v, out, true);
	out->print_matrix("sin(m1)");

	ce.cos(v, out, true);
	out->print_matrix("cos(m1)");

	ce.tanh(v, out, true);
	out->print_matrix("tanh(m1)");

	ce.tanh_gradient(v, out, true);
	out->print_matrix("tanh_gradient(m1)");

	ce.element_mult(v, out, true);
	out->print_matrix("m1 .* m2");

	float loss;
	ce.square_loss(v, &loss, true);
	cout<<"loss: "<<loss<<endl;

	ce.fill_rand(v, out, true);
	out->print_matrix("rand");

	ce.debug_fill(v, out, true);
	out->print_matrix("0.66337");
}
