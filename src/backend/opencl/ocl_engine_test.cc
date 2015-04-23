/*
 * Eona Studio (c)2015
 */
#include "ocl_util.h"
#include <iostream>
#include "opencl_engine.h"

using namespace std;
int main(int argc, char **argv)
{
	GlobalTimer gt;
	OpenclEngine engine(&gt);

	//create testcases

//	float t1[9] = {1.1, 7.8, 5.9, 3.0, 2, 5, 6, 10, 5};
//	float t2[9] = {0.1, 6.8, 4.9, 2.0, 1, 4, 5, 9, 4};
//	float t3[6] = {1.1, 7.8, 5.9, 3.0, 2, 5};
//	float t4[8] = {1.1, 7.8, 5.9, 3.0, 2, 5, 6, 10};
//	float t5[8] = {1.1, 7.8, 5.9, 3.0, 2, 5, 6, 10};
//
//
//	OpenclFloatMatPtr m1 (new OpenclFloatMat(t1, 3, 3, engine.cl));
//	OpenclFloatMatPtr m2 (new OpenclFloatMat(t2, 3, 3, engine.cl));
//	OpenclFloatMatPtr m3 (new OpenclFloatMat(t3, 2, 3, engine.cl));
//	OpenclFloatMatPtr m4 (new OpenclFloatMat(t4, 4, 2, engine.cl));
//	OpenclFloatMatPtr m5 (new OpenclFloatMat(t5, 4, 2, engine.cl));
//	OpenclFloatMatPtr out(new OpenclFloatMat());
//	vector<int> dim = {3,3};
//	engine.create(out, dim);

	vector<int> dim = {1000, 1000};

	OpenclFloatMatPtr m1 (new OpenclFloatMat());
	OpenclFloatMatPtr m2 (new OpenclFloatMat());
	OpenclFloatMatPtr m3 (new OpenclFloatMat());
	OpenclFloatMatPtr out(new OpenclFloatMat());
	engine.create(m1, dim);
	engine.create(m2, dim);
	engine.create(m3, dim);
	engine.create(out, dim);

	std::vector<OpenclFloatMatPtr> rv;
	engine.fill_rand(rv, m1, true);
	engine.fill_rand(rv, m2, true);
	engine.fill_rand(rv, m3, true);

	engine.debug_fill(rv, m1, true);
	engine.debug_fill(rv, m2, true);
	engine.debug_fill(rv, m3, true);
//    m1->print_matrix("m1");
//    m2->print_matrix("m2");
//    m3->print_matrix("m3");
//    m4->print_matrix("m4");

//

	std::vector<OpenclFloatMatPtr> v;
	v.push_back(m1);
	v.push_back(m2);
//
	std::vector<OpenclFloatMatPtr> v1;
	v1.push_back(m3);
	v1.push_back(m1);

//	std::vector<OpenclFloatMatPtr> v2;
//	v2.push_back(m1);
//	v2.push_back(m3);
//
//	std::vector<OpenclFloatMatPtr> v3;
//	v3.push_back(m4);
//	v3.push_back(m5);
//
	for (int i = 0; i < 10; ++i){
	engine.sub(v, out, true);
//	out->print_matrix("m1 - m2");

	engine.add(v, out, true);
//	out->print_matrix("m1 + m2");

//	out->print_matrix("-m1");
	engine.negate(v, out, true);
//	out->print_matrix("-m1");

	engine.multNN(v, out, true);
//	out->print_matrix("m1 * m2");
//	float buffer[10];
//	out->take_at(buffer, dim[0]*dim[1]-10, 10);
//	for (int i = 0; i < 10; ++i){
//		cout<<buffer[i]<<" ";
//	}
//	cout<<"\n";
//
//	engine.multNN(v, out, false);
//	out->print_matrix("m3 * m1");

	engine.multNT(v, out, true);
//	out->print_matrix("m1 * T(m3)");

	engine.multTN(v, out, true);
//	out->print_matrix("T(m4) * m5");

	engine.assign(v1, out, true);
//	out->print_matrix("m3 -> out");

	engine.sigmoid(v, out, true);
//	out->print_matrix("sigmod(m1)");

	engine.sigmoid_gradient(v, out, true);
//	out->print_matrix("sigmoid_gradient(m1)");

	engine.sin(v, out, true);
//	out->print_matrix("sin(m1)");

	engine.cos(v, out, true);
//	out->print_matrix("cos(m1)");

	engine.tanh(v, out, true);
//	out->print_matrix("tanh(m1)");

	engine.tanh_gradient(v, out, true);
//	out->print_matrix("tanh_gradient(m1)");

	engine.element_mult(v, out, true);
//	out->print_matrix("m1 .* m2");

//	float loss;
//	engine.square_loss(v, &loss, true);
//	cout<<"loss: "<<loss<<endl;

//	engine.fill_rand(v, out, true);
//	out->print_matrix("rand");
//
//	engine.debug_fill(v, out, true);
//	out->print_matrix("0.66337");
	}

	gt.print_stats(GlobalTimer::Nanosec, "test");
	out->free_data();
}
