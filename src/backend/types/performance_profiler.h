/*
 * Eona Studio (c)2015
 */

#ifndef PERFORMANCE_PROFILER_H_
#define PERFORMANCE_PROFILER_H_


#include <chrono>
#include <string>
#include <iostream>
using namespace std;
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::nanoseconds nanoseconds;

struct TimeEntry{
	nanoseconds time;
	size_t data_size;
	TimeEntry() {
		time = Clock::now() - Clock::now();
		data_size = 0;
	}
};

class GlobalTimer{
public:
	Clock::time_point t0;
	nanoseconds global_duration;
	std::string msg;
	std::unordered_map<std::string, TimeEntry> named_timers;

	GlobalTimer() {
//		named_timers["global"] = TimeEntry();
//		t0 = Clock::now();
	}

	void increment_named_timer (std::string timer_name, nanoseconds time_incre, size_t data_size){
		if ( named_timers.find (timer_name) == named_timers.end()) {
			named_timers[timer_name] = TimeEntry();
		}
		named_timers[timer_name].time += time_incre;
		named_timers[timer_name].data_size += data_size;
	}

	void print_stats(int unit_data) {
//		Clock::time_point t1 = Clock::now();
//		nanoseconds global_duration = std::chrono::duration_cast<nanoseconds>(t1 - t0);
//		named_timers["global"].time += global_duration;
//		named_timers["global"].data_size = 1;
    	for ( auto it = named_timers.begin(); it != named_timers.end(); ++it ) {
    		cout<<it->first << ": " << it->second.time.count() << ", " << it->second.data_size << ", " << unit_data*double(it->second.time.count())/double(it->second.data_size)<<endl;
    	}
	}


	~GlobalTimer(){
	}
};

class ScopeTimer{
public:
	Clock::time_point t0;
	std::string name;
	GlobalTimer * gt;
	size_t data_size;

	ScopeTimer(std::string msg) {
		t0 = Clock::now();
		name = msg;
		gt = NULL;
		data_size = 1;
	}

	ScopeTimer(std::string msg, GlobalTimer* g) {
		t0 = Clock::now();
		name = msg;
		gt = g;
		data_size = 1;
	}

	ScopeTimer(std::string msg, GlobalTimer* g, size_t n) {
		t0 = Clock::now();
		name = msg;
		gt = g;
		data_size = n;
	}

	~ScopeTimer(){
		Clock::time_point t1 = Clock::now();
		nanoseconds ns = std::chrono::duration_cast<nanoseconds>(t1 - t0);
		if (gt) {
			gt->increment_named_timer(name, ns, data_size);
		}
//		cout << msg << ns.count() << " nanoseconds"<<'\n';
	}
};


#endif
