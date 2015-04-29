/*
 * Eona Studio (c) 2015
 */

#ifndef UTILS_RAND_UTILS_H_
#define UTILS_RAND_UTILS_H_

#include <random>
#include <chrono>

#include "global_utils.h"
using std::default_random_engine;
using std::uniform_real_distribution;

// Set the same seed for reproducibility
static constexpr const ulong DEBUG_SEED = 388011773L;

template<typename FloatT>
class UniformRand
{
public:
	UniformRand(FloatT low, FloatT high, ulong initialSeed)
	: seed(initialSeed),
	  generator(seed),
	  distribution(uniform_real_distribution<FloatT>{low, high})
	{}

	UniformRand(FloatT low, FloatT high)
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
		name(_name)
	{ }

	// disable copying and assignment
	FakeRand(const FakeRand&) =delete;
	FakeRand& operator=(const FakeRand&) =delete;

	vector<float> randSeq;
	int i = 0;

	string name;

	bool isCircularWrap = true;

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
	// passing global float across files, debugging workaround
	GenFakeRandInstance(passin);

	/**
	 * Manually set the internal 'random sequence'
	 */
	void set_rand_seq(vector<float> randSeq_)
	{
		i = 0;
		randSeq = randSeq_;
	}

	vector<float> get_rand_seq() const
	{
		return this->randSeq;
	}

	float operator() ()
	{
		LMN_ASSERT_THROW(size() != 0,
			LaminarException("FakeRand no internal sequence set."));

		if (i >= size())
		{
			if (isCircularWrap)
				i = 0;
			else
				throw LaminarException("FakeRand instance " + name +
					" internal sequence depleted and isCircularWrap = false");
		}

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
		LMN_ASSERT_THROW(i < size(),
			LaminarException("FakeRand instance " + name + " operator[] out of range: "
					+ to_str(i) + " requested but internal size = " + to_str(size())));

		return randSeq[i];
	}

	void gen_uniform_rand(int seqLength, float low, float high, ulong seed)
	{
		i = 0;
		default_random_engine generator(seed);
		uniform_real_distribution<float> distribution{low, high};

		this->randSeq.clear();
		for (int s = 0; s < seqLength; ++s)
			this->randSeq.push_back(distribution(generator));
	}

	void gen_uniform_rand(int seqLength, float low, float high)
	{
		gen_uniform_rand(seqLength, low, high, UniformRand<float>::generate_seed());
	}

	// Position of the internal rand seq
	int current_position()
	{
		return this->i;
	}

	/**
	 * Reset i to 0 when the internal seq runs out
	 * Default: true
	 */
	void set_circular_wrap(bool isCircularWrap)
	{
		this->isCircularWrap = isCircularWrap;
	}

	void reset_seq() { i = 0; }

	void print_rand_seq()
	{
		cout << std::setprecision(3) << "Seq(" << name << ") = ";
		// 10 numbers per line break
		int lineBreak = 0;
		for (int j = 0; j < size(); ++j, ++lineBreak)
		{
			cout << randSeq[j];
			if (j != size() - 1)
			{
				cout << ", ";
				if (lineBreak == 9)
				{
					cout << "\n";
					lineBreak = 0;
				}
			}
			else
				cout << "\n";
		}
	}
};

#endif /* UTILS_RAND_UTILS_H_ */
