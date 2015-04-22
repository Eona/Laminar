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

class ScopeTimer{
public:
	Clock::time_point t0;
	std::string msg;
	ScopeTimer(std::string msg) {
		t0 = Clock::now();
		this->msg = msg;
	}
	~ScopeTimer(){
		Clock::time_point t1 = Clock::now();
		nanoseconds ms = std::chrono::duration_cast<nanoseconds>(t1 - t0);
		cout << msg << ms.count() << " nanoseconds"<<'\n';
	}
};


#endif
