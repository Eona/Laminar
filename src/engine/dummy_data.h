/*
 * Eona Studio (c) 2015
 */

#ifndef ENGINE_DUMMY_DATA_H_
#define ENGINE_DUMMY_DATA_H_

#include "data_manager.h"
#include "../rand_utils.h"

class DummyDataManager : public DataManager<float>
{
public:
	typedef std::shared_ptr<float> DataPtr;

	DummyDataManager(EngineBase::Ptr engine) :
		DataManager<float>(engine)
	{}

	void load_input(DataPtr write, bool is_initialized)
	{
		*write = input_rand();
	}

	void load_target(DataPtr write, bool is_initialized)
	{
		*write = target_rand();
	}

	void start_new_epoch()
	{
		input_rand.reset_seq();
		target_rand.reset_seq();
	}

	// FIXME
	void start_new_sequence()
	{
	}

	/**
	 * Debug only, for gradient check
	 * Perturbs the internal 'random sequence', doesn't upload any instruction
	 */
	void perturb_input(int idx, float eps)
	{
		this->lastPerturbedIdx = idx;
		this->lastEps = eps;
		input_rand[idx] += eps;
	}
	/**
	 * Debug only, for gradient check
	 */
	void restore_last_input()
	{
		input_rand[lastPerturbedIdx] -= lastEps;
	}

private:
	FakeRand& input_rand = FakeRand::instance_input();
	FakeRand& target_rand = FakeRand::instance_target();

	// for restoring perturbed input
	int lastPerturbedIdx;
	float lastEps;
};

#endif /* ENGINE_DUMMY_DATA_H_ */
