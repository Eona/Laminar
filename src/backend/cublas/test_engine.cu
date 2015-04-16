/*
 * Eona Studio (c)2015
 */
#include "cuda_engine.h"

int main(int argc, char **argv)
{
	printf("hello world\n");
	float a[4] = {1,2,4,5};
	CudaFloatMat m1(a,2,2);
	CudaFloatMat m2(a,2,2);
	CudaFloatMat m3(2,2);

	std::vector<CudaFloatMat*> v;
	v.push_back(&m1);
	v.push_back(&m2);

	lmn::CudaImpl::add<0>(v, &m3, true);
}
