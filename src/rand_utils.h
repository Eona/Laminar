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
	: UniformRand(low, high, generate_seed()) {}

	FloatT operator() ()
	{
		return distribution(generator);
	}

	static ulong generate_seed()
	{
		return std::chrono::system_clock::now().time_since_epoch().count();
	}

	void set_seed(ulong seed)
	{
		this->seed = seed;
		generator.seed(seed);
	}

	void set_seed()
	{
		set_seed(generate_seed());
	}

	ulong get_seed() { return seed; }

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
	FakeRand(string _name) :
		randSeq { 2.51, 5.39, 5.80, -2.96, -2.73, -2.4, 0.55, -.47 },
		name(_name)
	{ }

	// disable copying and assignment
	FakeRand(const FakeRand&) =delete;
	FakeRand& operator=(const FakeRand&) =delete;

	vector<float> randSeq;
	int i = 0;

	string name;

public:
// Separate instances with independent sequences
#define GenFakeRandInstance(name) \
	static FakeRand& instance_##name() \
	{ \
		static FakeRand rnd(STRINGFY(name)); \
		return rnd; \
	}

	GenFakeRandInstance(connection);
	GenFakeRandInstance(prehistory);
	GenFakeRandInstance(input);
	GenFakeRandInstance(target);

	/**
	 * Manually set the internal 'random sequence'
	 */
	void set_rand_seq(vector<float> _randSeq)
	{
		i = 0;
		randSeq = _randSeq;
	}

	float operator() ()
	{
		if (i >= randSeq.size())
			i = 0;
		return randSeq[i++];
	}

	int size() const
	{
		return randSeq.size();
	}

	/**
	 * Change internal randSeq value
	 */
	float& operator[](int i)
	{
		return this->randSeq[i];
	}

	void gen_uniform_rand(int seqLength, float low, float high)
	{
		i = 0;
		default_random_engine generator(UniformRand<float>::generate_seed());
		uniform_real_distribution<float> distribution{low, high};

		this->randSeq.clear();
		for (int s = 0; s < seqLength; ++s)
			this->randSeq.push_back(distribution(generator));
	}

	void reset_seq() { i = 0; }

	void print_rand_seq()
	{
		cout << std::setprecision(3) << "Seq(" << name << ") = ";
		for (int j = 0; j < size(); ++j)
		{
			cout << randSeq[j];
			if (j != size() - 1)
				cout << ", ";
		}
	}
};

#endif /* RAND_UTILS_H_ */
