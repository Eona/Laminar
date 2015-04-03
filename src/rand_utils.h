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

// Set the same seed for reproducibility
#define DEBUG_SEED 388011773L

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

/**
 * DEBUG ONLY. Singleton pattern
 */
template<typename FloatT>
class FakeRand
{
private:
	// fake args
	FakeRand() :
		preset { 3.24, -1.18, 0.47, 1.35, -0.62, 0.57, -1.25 }
	{ }

	// disable copying and assignment
	FakeRand(const FakeRand&) =delete;
	FakeRand& operator=(const FakeRand&) =delete;

	vector<FloatT> preset;
	int i = 0;

public:
	static FakeRand& instance()
	{
		static FakeRand rnd;
		return rnd;
	}

	FloatT operator() ()
	{
		if (i >= preset.size())
		{
			cerr << "Fake random generator runs out. Start from beginning." << endl;
			i = 0;
		}
		return preset[i++];
	}
};

#endif /* RAND_UTILS_H_ */
