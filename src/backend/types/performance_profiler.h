/*
 * Eona Studio (c)2015
 */

#ifndef PERFORMANCE_PROFILER_H_
#define PERFORMANCE_PROFILER_H_


#include <chrono>
#include <string>
#include <iostream>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl.h>
#include <CL/cl_platform.h>
#endif

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::nanoseconds nanoseconds;


struct TimeEntry{
	uint64_t time;//time in nanoseconds
	size_t data_size;
	TimeEntry() {
		time = 0;
		data_size = 0;
	}
};

//Generic global timer
class GlobalTimer{

public:
	enum Resolution
	{
		Sec, Millisec, Microsec, Nanosec
	};

	Clock::time_point t0; //unused
	uint64_t global_duration;
	std::unordered_map<std::string, TimeEntry> named_timers;

	GlobalTimer() {
		global_duration = 0; //unused
	}

	void record_named_timer (std::string timer_name, uint64_t time_incre_ns, size_t data_size){
		if ( named_timers.find (timer_name) == named_timers.end()) {
			named_timers[timer_name] = TimeEntry();
		}
		named_timers[timer_name].time += time_incre_ns;
		named_timers[timer_name].data_size += data_size;
	}

	uint64_t to_time_scale(Resolution res, uint64_t duration) {
		if (res == Sec) duration /= 1e9;
		if (res == Millisec) duration /= 1e6;
		if (res == Microsec) duration /= 1e3;

		return duration;
	}

	void print_stats(Resolution res) {
//		Clock::time_point t1 = Clock::now();
//		nanoseconds global_duration = std::chrono::duration_cast<nanoseconds>(t1 - t0);
//		named_timers["global"].time += global_duration;
//		named_timers["global"].data_size = 1;


    	for ( auto it = named_timers.begin(); it != named_timers.end(); ++it ) {
    		uint64_t time = to_time_scale(res, it->second.time);
    		cout<<it->first << ": " << time << ", " << it->second.data_size << ", " <<endl;
    	}
	}


	~GlobalTimer(){
	}
};


class LocalTimer {
protected:
	std::string name;
	GlobalTimer * gt;
	size_t data_size;
};

/*
 * Scoped timer using std::chrono, only good for GPU timing
 */
class ScopeTimer: public LocalTimer{
public:
	Clock::time_point t0;


	ScopeTimer(std::string msg){
		gt = NULL;
		data_size = 1;
		name = msg;
		t0 = Clock::now();
	}

	ScopeTimer(std::string msg, GlobalTimer* g) {
		gt = g;
		data_size = 1;
		name = msg;
		t0 = Clock::now();
	}

	ScopeTimer(std::string msg, GlobalTimer* g, size_t n) {
		gt = g;
		data_size = n;
		name = msg;
		t0 = Clock::now();
	}

	~ScopeTimer(){
		Clock::time_point t1 = Clock::now();
		nanoseconds ns = std::chrono::duration_cast<nanoseconds>(t1 - t0);
		if (gt) {
			gt->record_named_timer(name, ns.count(), data_size);
		}
	}
};

class CudaTimer: public LocalTimer{
public:

	CudaTimer(std::string msg) {
		gt = NULL;
		data_size = 1;
		name = msg;
		cudaEventCreate(&startTime);
		cudaEventCreate(&stopTime);
	}

	CudaTimer(std::string msg, GlobalTimer* g) {
		gt = g;
		data_size = 1;
		name = msg;
		cudaEventCreate(&startTime);
		cudaEventCreate(&stopTime);
	}

	CudaTimer(std::string msg, GlobalTimer* g, size_t n) {
		gt = g;
		data_size = n;
		name = msg;
		cudaEventCreate(&startTime);
		cudaEventCreate(&stopTime);
	}

	void start()
	{
		cudaEventRecord(startTime, 0);
	}

	float stop()
	{
		cudaEventRecord(stopTime, 0);
		float elapsed;
		cudaEventSynchronize(stopTime);
		cudaEventElapsedTime(&elapsed, startTime, stopTime);

		elapsed *= 1e6; //convert to nanoseconds
		if (gt) {
			gt->record_named_timer(name, (uint64_t)elapsed, data_size);
		}
		return elapsed;
	}


private:
	cudaEvent_t startTime;
	cudaEvent_t stopTime;

};


#endif
