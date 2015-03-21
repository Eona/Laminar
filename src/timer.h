/*
 * mytimer.h
 * (c) 2015
 * Author: Jim Fan
 */
#ifndef MYTIMER_H_
#define MYTIMER_H_

#include "utils.h"
#include <ctime>
#ifdef is_CPP_11
#include <chrono>
#endif

class Timer
{
public:
	enum Resolution
	{
		Sec, Millisec, Microsec
	};

	Timer(Resolution scale) : scale(scale)
	{ }

	virtual ~Timer() {};

	virtual void start() = 0;

	ulong elapsed()
	{
		// std::round doesn't work in C++98
		return ::round(_elapsed_sec() * (scale == Sec ? 1 :
					scale == Millisec ? 1000 :
					scale == Microsec ? 1e6 : 1));
	}

	void printElapsed(string msg = "")
	{
		if (msg != "")
			msg += ": ";
		string scaleName = scale == Sec ? "seconds" :
					scale == Millisec ? "milliseconds" :
					scale == Microsec ? "microseconds" : "";

		cout << msg << elapsed() << " " << scaleName << " elapsed" << endl;
	}

	Timer& setResolution(Resolution scale)
	{
		this->scale = scale;
		return *this;
	}

	virtual void showTime() {};

protected:
	// must return in seconds
	virtual double _elapsed_sec() = 0;

	Resolution scale;
};

#ifdef is_CPP_11
class CpuTimer : public Timer
{
public:
	CpuTimer(Resolution scale = Millisec) : Timer(scale)
	{ }

	~CpuTimer() {}

	void start()
	{
		start_time = std::chrono::system_clock::now();
	}

	void showTime()
	{
		time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
		cout << ctime(&t) << endl;
	}

private:
	std::chrono::time_point<std::chrono::system_clock> start_time, end_time;

protected:
	double _elapsed_sec()
	{
		end_time = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end_time - start_time;
		return elapsed_seconds.count();
	}
};

#else // doesn't support C++11
class CpuTimer : public Timer
{
public:
	CpuTimer(Resolution scale = Millisec) :
		Timer(scale), startTime(0), stopTime(0)
	{
	}

	~CpuTimer() {}

	void start()
	{
		startTime = clock();
	}

private:
	clock_t startTime, stopTime;

protected:
	double _elapsed_sec()
	{
		stopTime = clock();
		return (double)(stopTime - startTime) / CLOCKS_PER_SEC;
	}
};

#endif

/************************************/
/****** GPU timer by CudaEvent API ******/
#ifdef is_CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
class GpuTimer : public Timer
{
public:
	GpuTimer(Resolution scale = Millisec) : Timer(scale)
	{
		cudaEventCreate(&startTime);
		cudaEventCreate(&stopTime);
	}

	~GpuTimer()
	{
		cudaEventDestroy(startTime);
		cudaEventDestroy(stopTime);
	}

	void start()
	{
		cudaEventRecord(startTime, 0);
	}

private:
	cudaEvent_t startTime;
	cudaEvent_t stopTime;

protected:
	double _elapsed_sec()
	{
		cudaEventRecord(stopTime, 0);

		float elapsed;
		cudaEventSynchronize(stopTime);
		cudaEventElapsedTime(&elapsed, startTime, stopTime);
		return elapsed / 1000.0;
	}
};
#endif /* GPU timer */

#endif /* MYTIMER_H_ */
