/*
 * Eona Studio (c)2015
 */
#include "cuda_engine.h"
#include <iostream>

using namespace std;
class testMat{
public:
	testMat(float * d, int m, int n){
		mat = (float *) malloc( sizeof(float) * m * n );
		memcpy(mat, d, sizeof(float) * m * n);
		this->m = m;
		this->n = n;
	}

	testMat(int m, int n){
		mat = (float *) malloc( sizeof(float) * m * n );
		this->m = m;
		this->n = n;
	}

	testMat(){
		mat = 0;
		m = 0;
		n = 0;
	}
	float * mat;
	int m;
	int n;
	~testMat(){
		free(mat);
	}
};


void readMat(std::string infile, testMat * mat) {
	ifstream f(infile);
	int m,n;
	f >> m >> n;
	*mat = testMat(m,n);

	for (int i = 0; i < m*n; i++){
		float a;
		f >> a;
		mat->mat[i] = a;
	}
}

void testAdd()
{

}

void testSub()
{

}

int main(int argc, char **argv)
{
	//printf("hello world\n");


	//create testcases
	float t1[9] = {1.1, 7.8, 5.9, 3.0, 2, 5, 6, 10, 5};
	float t2[9] = {0.1, 6.8, 4.9, 2.0, 1, 4, 5, 9, 4};
	float t3[6] = {1.1, 7.8, 5.9, 3.0, 2, 5};

//	testMat d1(t1, 3, 3);
//	testMat d2(t2, 3, 3);
//	testMat d3(t3, 2, 3);
//	testMat d4;




	CudaFloatMat m1(t1, 3, 3);
	CudaFloatMat m2(t2, 3, 3);
	CudaFloatMat m3(t3, 2, 3);
	CudaFloatMat out;
    m1.print_matrix("m1");
    m2.print_matrix("m2");
    m3.print_matrix("m3");

	std::vector<CudaFloatMat*> v;
	v.push_back(&m1);
	v.push_back(&m2);

	std::vector<CudaFloatMat*> v1;
	v1.push_back(&m3);
	v1.push_back(&m1);

//	lmn::CudaImpl::add<0>(v, &out, false);
//	out.print_matrix("m1 + m2");
//
//	lmn::CudaImpl::sub<0>(v, &out, true);
//	out.print_matrix("m1 - m2");
//
//	lmn::CudaImpl::negate<0>(v, &out, true);
//	out.print_matrix("-m1");
//
//	lmn::CudaImpl::mult<0, 0>(v, &out, true);
//	out.print_matrix("m1 * m2");
//
//	lmn::CudaImpl::mult<0, 0>(v1, &out, false);
//	out.print_matrix("m3 * m1");
//
//	lmn::CudaImpl::assign<0>(v1, &out, true);
//	out.print_matrix("m3 -> out");

	lmn::CudaImpl::sigmoid(v, &out, false);
	out.print_matrix("sigmod(m1)");

	lmn::CudaImpl::sigmoid_gradient(v, &out, false);
	out.print_matrix("sigmoid_gradient(m1)");

	lmn::CudaImpl::sin(v, &out, false);
	out.print_matrix("sin(m1)");

	lmn::CudaImpl::cos(v, &out, false);
	out.print_matrix("cos(m1)");

	lmn::CudaImpl::tanh(v, &out, false);
	out.print_matrix("tanh(m1)");

	lmn::CudaImpl::tanh_gradient(v, &out, false);
	out.print_matrix("tanh_gradient(m1)");
}
