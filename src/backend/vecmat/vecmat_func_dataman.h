/*
 * Eona Studio (c) 2015
 */

#ifndef BACKEND_VECMAT_VECMAT_FUNC_DATAMAN_H_
#define BACKEND_VECMAT_VECMAT_FUNC_DATAMAN_H_

#include "../../engine/data_manager.h"
#include "../../utils/rand_utils.h"
#include "../vecmat/vecmat_engine.h"

/**
 * Functional data manager: the target tensor is an artificially constructed
 * vector function applied to the input tensor
 */
class VecmatFuncDataManager :
		public DataManager<lmn::Vecmatf>
//		public GradientCheckable<>
{
public:
	using Vecmatf = lmn::Vecmatf;
	typedef lmn::VecmatfPtr DataPtr;

	/**
	 * @param randLow lower bound of random input uniform distr
	 * @param randHigh higher bound of random input uniform distr
	 */
	VecmatFuncDataManager(EngineBase::Ptr engine,
			int inputDim, int targetDim, int batchSize,
			std::function<void(const Vecmatf& input, Vecmatf& target)> learnableFunc,
			int trainingSize, int validationSize, int testSize,
			float randLow, float randHigh, ulong seed = DEBUG_SEED) :
		DataManager(engine),
		inputDim(inputDim), targetDim(targetDim), batchSize(batchSize),
		streamSizes({ trainingSize, validationSize, testSize }),
		streamPos( {0, 0, 0} ),
		unirand(randLow, randHigh, seed),
		indexer(Dimension { inputDim, batchSize }) // for gradient check debugging
	{
		// Generate training/validation/testing inputs and targets
		for (int phase = 0; phase < LEARNING_PHASE_N; ++phase)
		{
			for (int ssize = 0; ssize < streamSizes[phase]; ++ssize)
			{
				Vecmatf inputMat(inputDim, batchSize, [this](int, int) -> float {
							return this->unirand();
						});

				inputStreams[phase].push_back(inputMat);

				Vecmatf targetMat(targetDim, batchSize);

				learnableFunc(inputMat, targetMat);

				targetStreams[phase].push_back(std::move(targetMat));
			}
		}
	}

	void load_input(DataPtr write, bool is_initialized, LearningPhase learnPhase)
	{
		if (!is_initialized)
			write->new_zeros(inputDim, batchSize);

		int phase = to_int(learnPhase);
		LMN_ASSERT_THROW(streamPos[phase] < streamSizes[phase],
				DataException("load_input stream position out of bound"));

		*write = inputStreams[phase][streamPos[phase]];
	}

	void load_target(DataPtr write, bool is_initialized, LearningPhase learnPhase)
	{
		if (!is_initialized)
			write->new_zeros(targetDim, batchSize);

		int phase = to_int(learnPhase);
		LMN_ASSERT_THROW(streamPos[phase] < streamSizes[phase],
				DataException("load_target stream position out of bound"));

		*write = targetStreams[phase][streamPos[phase]];
	}

	bool prepare_next_batch_impl(LearningPhase learnPhase)
	{
		int phase = to_int(learnPhase);
		// proceed to the next item in stream
		++ streamPos[phase];

		// If point to last in stream, end of epoch = true
		return streamPos[phase] == streamSizes[phase];
	}

	void reset_epoch_impl(LearningPhase phase)
	{
		this->streamPos[to_int(phase)] = 0;
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
	// Helper: convert LearningPhase to int
	static int to_int(LearningPhase phase)
	{
		return (int) enum2integral(phase);
	}

	/*********** Gradient checking ***********/
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

	// Training/Validation/Testing sample sizes
	std::array<int, LEARNING_PHASE_N> streamSizes;

	// current position in input/target stream
	std::array<int, LEARNING_PHASE_N> streamPos;

	// accessed by enum2integral(Training/Validation/Testing)
	std::array<vector<Vecmatf>, LEARNING_PHASE_N> inputStreams, targetStreams;

	UniformRand<float> unirand;

	// for restoring perturbed input
	DimIndexEnumerator indexer;
};

#endif /* BACKEND_VECMAT_VECMAT_FUNC_DATAMAN_H_ */
