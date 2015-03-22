/*
 * test.cpp
 * (c) 2015
 * Author: Jim Fan
 * See below link for how to support C++11 in eclipse
 * http://scrupulousabstractions.tumblr.com/post/36441490955/eclipse-mingw-builds
 */
#include "timer.h"
#include "input_layer.h"
#include "output_layer.h"
#include "recurrent_layer.h"
#include "connection.h"
#include "transfer_layer.h"
#include "loss_layer.h"
#include "parameter.h"
#include "lstm_layer.h"

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

    double p = 66;
    GpuTimer t;
    t.start();
    testkernel<<< 3, 4 >>>();

    t.setResolution(Timer::Microsec).printElapsed();
}
