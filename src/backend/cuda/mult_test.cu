/*
 * Eona Studio (c)2015
 */
#include <iostream>
#include "cublas_engine.h"

using namespace std;
typedef std::shared_ptr<CudaFloatMat> CudaFloatMatPtr;
int main(int argc, char **argv)
{
	GlobalTimer<cudaEvent_t> gt;
	CublasEngine engine(&gt);

	CudaFloatMatPtr m1 (new CudaFloatMat());
	CudaFloatMatPtr m2 (new CudaFloatMat());
	CudaFloatMatPtr m3 (new CudaFloatMat());
	CudaFloatMatPtr out(new CudaFloatMat());
	engine.create(m1, {10, 20});
	engine.create(m2, {20, 30});
	engine.create(m3, {10, 100});
	engine.create(out, dim);

	std::vector<CudaFloatMatPtr> rv;
	engine.fill_rand(rv, m1, true);
	engine.fill_rand(rv, m2, true);
	engine.fill_rand(rv, m3, true);


	std::vector<CudaFloatMatPtr> v;
	v.push_back(m1);
	v.push_back(m2);
	std::vector<CudaFloatMatPtr> v1;
	v1.push_back(m3);
	v1.push_back(m1);

	engine.multNN(v, out, true);
	engine.multNN(v1, out, false);
}
