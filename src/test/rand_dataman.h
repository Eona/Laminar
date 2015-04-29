/*
 * Eona Studio (c) 2015
 */

#ifndef BACKEND_RAND_DATAMAN_H_
#define BACKEND_RAND_DATAMAN_H_

#include "../engine/data_manager.h"
#include "../engine/engine.h"
#include "../utils/rand_utils.h"
#include "../utils/laminar_utils.h"

// TODO sequence rand data support
template<typename DataT>
class RandDataManager :
		public DataManager<DataT>
//		public GradientCheckable<>
{
public:
	typedef std::shared_ptr<DataT> DataPtr;

	RandDataManager(EngineBase::Ptr engine,
			int inputDim, int targetDim, int batchSize,
			int targetLabelClasses = 0) :
		DataManager<DataT>(engine),
		inputDim(inputDim), targetDim(targetDim), batchSize(batchSize),
		genRandInput(inputDim * batchSize),
		genRandTarget(targetDim * batchSize),
		indexer(Dimension { inputDim, batchSize }) // for gradient check debugging
	{
		UniformRand<float> unirand(-1, 1);
		for (float& f : genRandInput)
			f = unirand();

		if (targetLabelClasses > 0)
		{
			for (float& f : genRandTarget)
				f = rand() % targetLabelClasses;
		}
		else
		{
			for (float& f : genRandTarget)
				f = unirand();
		}
	}

	virtual ~RandDataManager() {}

	void load_input(DataPtr write, bool is_initialized, LearningPhase)
	{
		if (!is_initialized)
			this->alloc_zeros(write, inputDim, batchSize);

		this->load_data(write, genRandInput);
	}

	void load_target(DataPtr write, bool is_initialized, LearningPhase)
	{
		if (!is_initialized)
			this->alloc_zeros(write, targetDim, batchSize);

		this->load_data(write, genRandTarget);
	}

	bool prepare_next_batch_impl(LearningPhase)
	{
		return false;
	}

	void reset_epoch_impl(LearningPhase)
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
	// subclass handles actual data load
	virtual void alloc_zeros(DataPtr write, int rowdim, int coldim) = 0;

	virtual void load_data(DataPtr write, vector<float>&) = 0;

	/*********** Gradient checking ***********/
	/**
	 * GradientCheckable<float> interface
	 */
/*	virtual void gradient_check_perturb_impl(
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
	}*/

private:
	int inputDim;
	int targetDim;
	int batchSize;

	vector<float> genRandInput;
	vector<float> genRandTarget;

	// for restoring perturbed input
	DimIndexEnumerator indexer;
};

#endif /* BACKEND_RAND_DATAMAN_H_ */
