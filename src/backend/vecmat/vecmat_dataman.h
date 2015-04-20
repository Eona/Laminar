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

	VecmatDataManager(EngineBase::Ptr engine, int inputDim_, int batchSize_) :
		DataManager(engine),
		dataDim(inputDim_), batchSize(batchSize_)
	{}

	void load_input(DataPtr write, bool is_initialized)
	{
		if (!is_initialized)
			write->new_zeros(dataDim, batchSize);

		write->fill([&](int i, int j) {
			return input_rand();
		});
	}

	void load_target(DataPtr write, bool is_initialized)
	{
		if (!is_initialized)
			write->new_zeros(dataDim, batchSize);

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
		return { this->dataDim, this->batchSize };
	}

	Dimension target_dim() const
	{
		return input_dim();
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
	int dataDim;
	int batchSize;

	FakeRand& input_rand = FakeRand::instance_input();
	FakeRand& target_rand = FakeRand::instance_target();

	// for restoring perturbed input
	int lastPerturbedIdx;
	float lastEps;
};

#endif /* BACKEND_VECMAT_VECMAT_DATAMAN_H_ */
