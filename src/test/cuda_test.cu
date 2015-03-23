#include "test_utils.h"

__global__ void testkernel()
{
	double p = threadIdx.x + 66;
	for (int i = 0; i < 30000000; ++i)
		p += i / p - std::sqrt(p);

	printf("thread %d; block %d\n", threadIdx.x, blockIdx.x);
}

TEST(CudaTest, Run)
{
	testkernel<<<3, 4>>>();

}
