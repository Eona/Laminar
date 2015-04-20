/*
 * Eona Studio (c) 2015
 */

#ifndef BACKEND_VECMAT_VECMAT_DATAMAN_H_
#define BACKEND_VECMAT_VECMAT_DATAMAN_H_

#include "../../engine/data_manager.h"
#include "../../utils/rand_utils.h"
#include "../vecmat/vecmat_engine.h"

class VecmatDataManager :
		public DataManager<lmn::Vecmatf>
//		public GradientCheckable<>
{
public:
	typedef lmn::VecmatfPtr DataPtr;

	VecmatDataManager(EngineBase::Ptr engine, int inputDim_, int targetDim_, int batchSize_) :
		DataManager(engine),
		inputDim(inputDim_), targetDim(targetDim_), batchSize(batchSize_)
	{}

	void load_input(DataPtr write, bool is_initialized)
	{
		if (!is_initialized)
			write->new_zeros(inputDim, batchSize);

		write->fill([&](int i, int j) {
			return input_rand();
		});
	}

	void load_target(DataPtr write, bool is_initialized)
	{
		if (!is_initialized)
			write->new_zeros(targetDim, batchSize);

		write->fill([&](int i, int j) {
			return target_rand();
		});
	}

	void start_new_epoch()
	{
		input_rand.reset_seq();
		target_rand.reset_seq();
	}

	void start_new_sequence()
	{
		input_rand.reset_seq();
		target_rand.reset_seq();
	}

	Dimension input_dim() const
	{
		return { this->inputDim, this->batchSize };
	}

	Dimension target_dim() const
	{
		return { this->targetDim, this->batchSize };
	}

	int batch_size() const
	{
		return this->batchSize;
	}

	/*********** Gradient checking ***********/
	/** TODO
	 * GradientCheckable<float> interface
	 */

private:
	int inputDim;
	int targetDim;
	int batchSize;

	FakeRand& input_rand = FakeRand::instance_input();
	FakeRand& target_rand = FakeRand::instance_target();

	// for restoring perturbed input
	int lastPerturbedIdx;
	float lastEps;
};

#endif /* BACKEND_VECMAT_VECMAT_DATAMAN_H_ */
