/*
 * test.cpp
 * (c) 2015
 * Author: Jim Fan
 * See below link for how to support C++11 in eclipse
 * http://scrupulousabstractions.tumblr.com/post/36441490955/eclipse-mingw-builds
 */
#include "timer.h"

#ifdef is_CUDA
__global__ void testkernel()
{
	double p = threadIdx.x + 66;
	for (int i = 0; i < 30000000; ++i)
		p += i / p - std::sqrt(p);

	printf("thread %d; block %d\n", threadIdx.x, blockIdx.x);
}
#endif

int main(int argc, char **argv)
{
#ifdef is_CPP_11
	cout << "C++ 11 supported!" << endl;
#else
	cout << "C++ 11 NOT supported!" << endl;
#endif
#ifdef is_CPP_11
	vector<int> a {115, 113, 111, 110, 112};
	cout << "[";
	for (auto i : a)
		cout << i << " ";
	cout << "]";
#else
	vector<int> a;
	a.push_back(5); a.push_back(1); a.push_back(4); a.push_back(9); a.push_back(0);
	cout << a << endl;
#endif

double p = 66;
GpuTimer t;
t.start();
testkernel<<< 3, 4 >>>();
t.setResolution(Timer::Microsec).printElapsed();
}
