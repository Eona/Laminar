/*
 * Eona Studio (c)2015
 */

#ifndef PERFORMANCE_PROFILER_H_
#define PERFORMANCE_PROFILER_H_


#include <chrono>
#include <string>
#include <iostream>
#include <fstream>

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
//#include "../../gpu_utils.h"

using namespace std;
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::nanoseconds nanoseconds;
typedef std::chrono::microseconds microseconds;

struct TimerEntry{
	uint64_t time;//time in nanoseconds
	size_t data_size;
	Clock::time_point time_stamp;
	size_t op_index;

//	TimerEntry() {
//		time = 0;
//		data_size = 0;
//		op_index = 0;
//		time_stamp = Clock::now();
//	}
	TimerEntry(uint64_t t, size_t d, size_t index) {
		time = t;
		data_size = d;
		op_index = index;
		time_stamp = Clock::now();
	}
};

//Generic global timer
class GlobalTimer{

public:
	enum Resolution
	{
		Sec, Millisec, Microsec, Nanosec
	};

	Clock::time_point t0;
	uint64_t global_duration;
	std::unordered_map<std::string, vector<TimerEntry>> named_timers;
	size_t op_counter;

	GlobalTimer() {
		global_duration = 0; //unused
		t0 = Clock::now();
		op_counter = 0;
	}

	void record_named_timer (std::string timer_name, uint64_t time_incre_ns, size_t data_size){
		if ( named_timers.find (timer_name) == named_timers.end()) {
			named_timers[timer_name] = vector<TimerEntry>();
		}
		TimerEntry e(time_incre_ns, data_size, op_counter++);
		named_timers[timer_name].push_back(e);
	}

	uint64_t to_time_scale(Resolution res, uint64_t duration) {
		if (res == Sec) duration /= 1e9;
		if (res == Millisec) duration /= 1e6;
		if (res == Microsec) duration /= 1e3;

		return duration;
	}

	void print_stats(Resolution res, std::string exp_name) {
//		Clock::time_point t1 = Clock::now();
//		nanoseconds global_duration = std::chrono::duration_cast<nanoseconds>(t1 - t0);
//		named_timers["global"].time += global_duration;
//		named_timers["global"].data_size = 1;

    	for ( auto it = named_timers.begin(); it != named_timers.end(); ++it ) {
    		ofstream outfile;
    		outfile.open("../experiment/" + exp_name  + "/" + it->first + ".csv");
    		int sum_data = 0;
    		uint64_t sum_time = 0;
			cout<<it->first << ": "<<endl;
    		for (auto entry: it->second) {
        		uint64_t t = to_time_scale(res, entry.time); //task duration
        		uint64_t stamp = to_time_scale(Millisec, std::chrono::duration_cast<nanoseconds>(entry.time_stamp - t0).count());
        		sum_time += t;
        		sum_data += entry.data_size;
        		outfile<<stamp << ","<< entry.op_index << "," << t << "," <<entry.data_size<<endl;
//        		cout << t << ", " << entry.data_size<< " at "<< stamp << endl;
    		}
    		outfile.close();
    		cout << "Computation through put" <<(double)sum_data/(double)sum_time<<"\n\n";
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
    	cudaDeviceSynchronize();
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
