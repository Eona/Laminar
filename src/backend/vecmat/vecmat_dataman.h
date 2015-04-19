/*
 * Eona Studio (c) 2015
 */

#ifndef BACKEND_VECMAT_VECMAT_DATAMAN_H_
#define BACKEND_VECMAT_VECMAT_DATAMAN_H_

#include "../../engine/data_manager.h"
#include "../../utils/rand_utils.h"
#include "../vecmat/vecmat_engine.h"

class VecmatDataManager : public DataManager<lmn::VecmatImpl::Vecmatf>
{
public:
	typedef lmn::VecmatImpl::VecmatfPtr DataPtr;

	VecmatDataManager(EngineBase::Ptr engine, int inputDim_, int batchSize_) :
		DataManager(engine),
		inputDim(inputDim_), batchSize(batchSize_)
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
			write->new_zeros(inputDim, batchSize);

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

	int input_dim() const
	{
		return this->inputDim;
	}

	int batch_size() const
	{
		return this->batchSize;
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
	int inputDim;
	int batchSize;

	FakeRand& input_rand = FakeRand::instance_input();
	FakeRand& target_rand = FakeRand::instance_target();

	// for restoring perturbed input
	int lastPerturbedIdx;
	float lastEps;
};

#endif /* BACKEND_VECMAT_VECMAT_DATAMAN_H_ */
