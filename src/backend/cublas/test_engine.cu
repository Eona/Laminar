/*
 * Eona Studio (c)2015
 */
#include "cuda_engine.h"
#include <iostream>
int main(int argc, char **argv)
{
	//printf("hello world\n");
	float a[6] = {1,2,3,4,5,6};
	CudaFloatMat m1(a,3,2);
	CudaFloatMat m2(a,3,2);
	CudaFloatMat m3;
    m1.print_matrix("m1");
    m2.print_matrix("m2");

	std::vector<CudaFloatMat*> v;
	v.push_back(&m1);
	v.push_back(&m2);

	lmn::CudaImpl::add<0>(v, &m3, false);
    m3.print_matrix("m3");
}
