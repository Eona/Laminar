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
	DummyDataManager(EngineBase::Ptr engine) :
		DataManager<float>(engine)
	{}

	void load_input(float *write, bool is_initialized)
	{
		*write = input_rand();
	}

	void load_target(float *write, bool is_initialized)
	{
		*write = target_rand();
	}

	void start_new_epoch()
	{
		input_rand.reset_seq();
		target_rand.reset_seq();
	}

private:
	FakeRand& input_rand = FakeRand::instance_input();
	FakeRand& target_rand = FakeRand::instance_target();
};

#endif /* ENGINE_DUMMY_DATA_H_ */
