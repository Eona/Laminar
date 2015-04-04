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
class FakeRand
{
private:
	// fake args
	FakeRand() :
//		randSeq { 0.24, -1.18, 0.47, 1.35, -0.62, 0.57, -1.25, -.88 }
		randSeq { 2.51, 5.39, 5.80, -2.96, -2.73, -2.4, 0.55, -.47 }
	{ }

	// disable copying and assignment
	FakeRand(const FakeRand&) =delete;
	FakeRand& operator=(const FakeRand&) =delete;

	vector<float> randSeq;
	int i = 0;

public:
	static FakeRand& instance()
	{
		static FakeRand rnd;
		return rnd;
	}

	/**
	 * Manually set the internal 'random sequence'
	 */
	void set_rand_seq(vector<float>& _randSeq)
	{
		randSeq = _randSeq;
	}
	void set_rand_seq(vector<float>&& _randSeq)
	{
		randSeq = _randSeq;
	}

	float operator() ()
	{
		if (i >= randSeq.size())
			i = 0;
		return randSeq[i++];
	}
};

#endif /* RAND_UTILS_H_ */
