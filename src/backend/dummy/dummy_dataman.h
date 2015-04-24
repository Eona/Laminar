/*
 * Eona Studio (c) 2015
 */

#ifndef ENGINE_DUMMY_DATA_H_
#define ENGINE_DUMMY_DATA_H_

#include "../../engine/data_manager.h"
#include "../../utils/laminar_utils.h"
#include "../../utils/rand_utils.h"

class DummyDataManager :
		public DataManager<float>,
		public GradientCheckable<>
{
public:
	typedef std::shared_ptr<float> DataPtr;

	DummyDataManager(EngineBase::Ptr engine) :
		DataManager<float>(engine)
	{}

	bool load_input(DataPtr write, bool is_initialized, LearningStage)
	{
		*write = input_rand();
		return false;
	}

	void load_target(DataPtr write, bool is_initialized, LearningStage)
	{
		*write = target_rand();
	}

	Dimension input_dim() const
	{
		return {1, 1};
	}

	Dimension target_dim() const
	{
		return {1, 1};
	}

	int batch_size() const
	{
		return 1;
	}

	void reset_epoch(LearningStage)
	{
		input_rand.reset_seq();
		target_rand.reset_seq();
	}

	/*********** Gradient checking ***********/
	/**
	 * GradientCheckable<float> interface
	 * Because dummy is only a single float, dimIdx is ignored
	 */
	virtual void gradient_check_perturb_impl(
			int changeItem, DimIndex dimIdx, float eps)
	{
		input_rand[changeItem] += eps;
	}

	/**
	 * GradientCheckable<float> interface
	 * Because dummy is only a single float, dimIdx is ignored
	 */
	virtual void gradient_check_restore_impl(
			int lastChangeItem, DimIndex lastDimIdx, float lastEps)
	{
		input_rand[lastChangeItem] -= lastEps;
	}

private:
	FakeRand& input_rand = FakeRand::instance_input();
	FakeRand& target_rand = FakeRand::instance_target();
};

#endif /* ENGINE_DUMMY_DATA_H_ */
