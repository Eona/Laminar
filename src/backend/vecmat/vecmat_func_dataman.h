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
		for (int stage = 0; stage < STAGE_N; ++stage)
		{
			for (int ssize = 0; ssize < streamSizes[stage]; ++ssize)
			{
				Vecmatf inputMat(inputDim, batchSize, [this](int, int) -> float {
							return this->unirand();
						});

				inputStreams[stage].push_back(inputMat);

				Vecmatf targetMat(targetDim, batchSize);

				learnableFunc(inputMat, targetMat);
/*
				LMN_ASSERT_THROW(targetMat.dim() == (Dimension {targetDim, batchSize}),
						DataException("Target matrix produced by the provided learnable "
								"function doesn't have the right dimension "
								+ to_str(targetDim) + " x " + to_str(batchSize)));
*/

				targetStreams[stage].push_back(std::move(targetMat));
			}
		}
	}

	void load_input(DataPtr write, bool is_initialized, LearningStage learningStage)
	{
		if (!is_initialized)
			write->new_zeros(inputDim, batchSize);

		int stage = to_int(learningStage);
		LMN_ASSERT_THROW(streamPos[stage] < streamSizes[stage],
				DataException("load_input stream position out of bound"));

		*write = inputStreams[stage][streamPos[stage]];
	}

	void load_target(DataPtr write, bool is_initialized, LearningStage learningStage)
	{
		if (!is_initialized)
			write->new_zeros(targetDim, batchSize);

		int stage = to_int(learningStage);
		LMN_ASSERT_THROW(streamPos[stage] < streamSizes[stage],
				DataException("load_target stream position out of bound"));

		*write = targetStreams[stage][streamPos[stage]];
	}

	bool prepare_next_batch_impl(LearningStage learningStage)
	{
		int stage = to_int(learningStage);
		// proceed to the next item in stream
		++ streamPos[stage];

		// If point to last in stream, end of epoch = true
		return streamPos[stage] == streamSizes[stage];
	}

	void reset_epoch_impl(LearningStage stage)
	{
		this->streamPos[to_int(stage)] = 0;
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
	// Helper: convert LearningStage to int
	static int to_int(LearningStage stage)
	{
		return (int) enum2integral(stage);
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
	// 3 learning stages
	static constexpr const int STAGE_N = 3;

	int inputDim;
	int targetDim;
	int batchSize;

	// Training/Validation/Testing sample sizes
	std::array<int, STAGE_N> streamSizes;

	// current position in input/target stream
	std::array<int, STAGE_N> streamPos;

	// accessed by enum2integral(Training/Validation/Testing)
	std::array<vector<Vecmatf>, STAGE_N> inputStreams, targetStreams;

	UniformRand<float> unirand;

	// for restoring perturbed input
	DimIndexEnumerator indexer;
};

#endif /* BACKEND_VECMAT_VECMAT_FUNC_DATAMAN_H_ */
