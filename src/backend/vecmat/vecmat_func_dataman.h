/*
 * Eona Studio (c) 2015
 */

#ifndef BACKEND_VECMAT_VECMAT_RAND_DATAMAN_H_
#define BACKEND_VECMAT_VECMAT_RAND_DATAMAN_H_

#include "../../engine/data_manager.h"
#include "../../utils/rand_utils.h"
#include "../vecmat/vecmat_engine.h"

/**
 * Functional data manager: the target tensor is an artificially constructed
 * vector function applied to the input tensor
 */
class VecmatFuncDataManager :
		public DataManager<lmn::Vecmatf>,
		public GradientCheckable<>
{
public:
	typedef lmn::VecmatfPtr DataPtr;

	VecmatFuncDataManager(EngineBase::Ptr engine, int inputDim_, int targetDim_, int batchSize_) :
		DataManager(engine),
		inputDim(inputDim_), targetDim(targetDim_), batchSize(batchSize_),
		indexer(Dimension { inputDim, batchSize }) // for gradient check debugging
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

	}

	void start_new_epoch()
	{
		input_rand.reset_seq();
	}

	int current_epoch()
	{

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

protected:
	/*********** Gradient checking ***********/
	/**
	 * GradientCheckable<float> interface
	 */
	virtual void gradient_check_perturb_impl(
				int changeItem, DimIndex dimIdx, float eps)
	{
		// assume input_rand is fully reset
		// calculate the 1D index inside input_rand
		input_rand[changeItem * inputDim * batchSize
			+ this->indexer.linearize(dimIdx)] += eps;
	}

	virtual void gradient_check_restore_impl(
			int lastChangeItem, DimIndex lastDimIdx, float lastEps)
	{
		input_rand[lastChangeItem * inputDim * batchSize
			+ this->indexer.linearize(lastDimIdx)] -= lastEps;
	}

private:
	int inputDim;
	int targetDim;
	int batchSize;

	FakeRand& input_rand = FakeRand::instance_input();

	// for restoring perturbed input
	DimIndexEnumerator indexer;
};

#endif /* BACKEND_VECMAT_VECMAT_RAND_DATAMAN_H_ */
