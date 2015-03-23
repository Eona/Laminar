/*
 * Eona Studio (c) 2015
 */

#ifndef RAND_UTILS_H_
#define RAND_UTILS_H_

#include "global_utils.h"
#include <random>
#include <chrono>
using std::default_random_engine;
using std::uniform_real_distribution;

template<typename FloatT>
class UniformRand
{
public:
	UniformRand(float low, float high, ulong initialSeed)
	: seed(initialSeed),
	  generator(seed),
	  distribution(uniform_real_distribution<FloatT>{low, high})
	{}

	UniformRand(float low, float high)
	: UniformRand(low, high, generateSeed()) {}

	FloatT operator() ()
	{
		return distribution(generator);
	}

	static ulong generateSeed()
	{
		return std::chrono::system_clock::now().time_since_epoch().count();
	}

	void setSeed(ulong seed)
	{
		this->seed = seed;
		generator.seed(seed);
	}

	void setSeed()
	{
		setSeed(generateSeed());
	}

	ulong getSeed() { return seed; }

private:
	ulong seed;
	default_random_engine generator;
	uniform_real_distribution<FloatT> distribution;
};

#endif /* RAND_UTILS_H_ */
