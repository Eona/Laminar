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

/*********** DEBUG ONLY ***********/
/**
 * Manually set the sequence
 */
class FakeRand
{
private:
	// fake args
	FakeRand() :
		randSeq { 2.51, 5.39, 5.80, -2.96, -2.73, -2.4, 0.55, -.47 }
	{ }

	// disable copying and assignment
	FakeRand(const FakeRand&) =delete;
	FakeRand& operator=(const FakeRand&) =delete;

	vector<float> randSeq;
	int i = 0;

	default_random_engine generator;
	uniform_real_distribution<float> distribution;

	bool isDisplay = false;

public:
// Separate instances with independent sequences
#define GenFakeRandInstance(name) \
	static FakeRand& instance_##name() \
	{ \
		static FakeRand rnd; \
		return rnd; \
	}

	GenFakeRandInstance(connection);
	GenFakeRandInstance(prehistory);

	/**
	 * Manually set the internal 'random sequence'
	 */
	void set_rand_seq(vector<float>& _randSeq)
	{
		i = 0;
		randSeq = _randSeq;
	}
	void set_rand_seq(vector<float>&& _randSeq)
	{
		i = 0;
		randSeq = _randSeq;
	}

	float operator() ()
	{
		if (i < 0)
		{
			float r = distribution(generator);
			if (isDisplay)
				cout << std::setprecision(3) << r << ", "; // sample a good unit test
			return r;
		}

		if (i >= randSeq.size())
			i = 0;
		return randSeq[i++];
	}

	void use_uniform_rand(float low, float high)
	{
		i = -1;
		generator = default_random_engine(UniformRand<float>::generateSeed());
		distribution = uniform_real_distribution<float>{low, high};
	}

	void use_fake_seq()
	{
		i = 0;
	}

	void is_rand_displayed(bool isDisplay)
	{
		this->isDisplay = isDisplay;
	}
};

#endif /* RAND_UTILS_H_ */
