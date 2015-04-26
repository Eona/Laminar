/*
 * Eona Studio (c) 2015
 */

#ifndef MNIST_DATAMAN_H_
#define MNIST_DATAMAN_H_

#include "../../engine/data_manager.h"
#include "../../utils/laminar_utils.h"
#include "mnist_parser.h"

template<typename DataT>
class MnistDataManager :
		public DataManager<DataT>
{
public:
	typedef std::shared_ptr<DataT> DataPtr;

	static const constexpr int MNIST_INPUT_DIM = 28 * 28;
	static const constexpr int MNIST_TRAINING_SIZE = 60000;
	static const constexpr int MNIST_TESTING_SIZE = 10000;

	/**
	 * @param randLow lower bound of random input uniform distr
	 * @param randHigh higher bound of random input uniform distr
	 */
	MnistDataManager(EngineBase::Ptr engine,
					int batchSize,
					string mnistDataDir) :
		DataManager<DataT>(engine),
		batchSize(batchSize),
		streamSizes({ MNIST_TRAINING_SIZE / batchSize, 0, MNIST_TESTING_SIZE / batchSize }),
		streamPos( {0, 0, 0} )
	{
		inputStreams[enum2integral(LearningPhase::Training)] =
				read_mnist_image(dataDir + "/" + MNIST_TRAIN_IMAGE_FILE, 0, batchSize, true);

		inputStreams[enum2integral(LearningPhase::Testing)] =
				read_mnist_image(dataDir + "/" + MNIST_TEST_IMAGE_FILE, 0, batchSize, true);

		targetStreams[enum2integral(LearningPhase::Training)] =
				read_mnist_label(dataDir + "/" + MNIST_TRAIN_LABEL_FILE, 0, batchSize);

		targetStreams[enum2integral(LearningPhase::Testing)] =
				read_mnist_label(dataDir + "/" + MNIST_TEST_LABEL_FILE, 0, batchSize);
	}

	virtual ~MnistDataManager() {}

	void load_input(DataPtr write, bool is_initialized, LearningPhase learnPhase)
	{
		LMN_ASSERT_THROW(learnPhase != LearningPhase::Validation,
				DataException("MNIST load_input: no validation data"));

		if (!is_initialized)
			this->alloc_zeros(write, MNIST_INPUT_DIM, batchSize);

		int phase = to_int(learnPhase);
		LMN_ASSERT_THROW(streamPos[phase] < streamSizes[phase],
				DataException("load_input stream position out of bound"));

		this->load_data(write, inputStreams[phase][streamPos[phase]]);
	}

	void load_target(DataPtr write, bool is_initialized, LearningPhase learnPhase)
	{
		LMN_ASSERT_THROW(learnPhase != LearningPhase::Validation,
				DataException("MNIST load_target: no validation data"));

		if (!is_initialized)
			// just a row of labels
			this->alloc_zeros(write, 1, batchSize);

		int phase = to_int(learnPhase);
		LMN_ASSERT_THROW(streamPos[phase] < streamSizes[phase],
				DataException("load_target stream position out of bound"));

		this->load_data(write, targetStreams[phase][streamPos[phase]]);
	}

	bool prepare_next_batch_impl(LearningPhase learnPhase)
	{
		LMN_ASSERT_THROW(learnPhase != LearningPhase::Validation,
				DataException("MNIST prepare_next_batch: no validation data"));

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
		return { MNIST_INPUT_DIM, this->batchSize };
	}

	Dimension target_dim() const
	{
		// just a row of labels
		return { 1, this->batchSize };
	}

	int batch_size() const
	{
		return this->batchSize;
	}

protected:
	// subclass handles actual data load
	virtual void alloc_zeros(DataPtr write, int rowdim, int coldim) = 0;

	virtual void load_data(DataPtr write, vector<float>&) = 0;

private:
	int batchSize;

	// Training/Validation/Testing sample sizes
	std::array<int, LEARNING_PHASE_N> streamSizes;

	// current position in input/target stream
	std::array<int, LEARNING_PHASE_N> streamPos;

	// accessed by enum2integral(Training/Validation/Testing)
	// each inner vector<float> is a batch of images/labels
	std::array<vector<vector<float>>, LEARNING_PHASE_N> inputStreams, targetStreams;
};

#endif /* MNIST_DATAMAN_H_ */
