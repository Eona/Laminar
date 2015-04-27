/*
 * Eona Studio (c) 2015
 */

#ifndef DEMO_NLP_CORPUS_DATAMAN_H_
#define DEMO_NLP_CORPUS_DATAMAN_H_

#include "../../engine/data_manager.h"
#include "../../utils/laminar_utils.h"
#include "corpus_loader.h"

template<typename DataT>
class CorpusDataManager :
		public DataManager<DataT>
{
public:
	typedef std::shared_ptr<DataT> DataPtr;

	/**
	 * @param randLow lower bound of random input uniform distr
	 * @param randHigh higher bound of random input uniform distr
	 */
	CorpusDataManager(EngineBase::Ptr engine,
					int batchSize,
					int historyLength,
					string corpusFileName) :
		DataManager<DataT>(engine),
		batchSize(batchSize),
		historyLength(historyLength),
		streamSizes({ 0, 0, 0 }),
		streamPos( { 0, 0, 0 } ),
		corpus(corpusFileName)
	{
		// Round the corpus to the nearest thousand
		int totalSize = corpus.size();
		totalSize -= totalSize % 1000;

		// Divide training/validation by 3:1
		int total_1000 = totalSize / 1000;
		training_1000 = int(0.75 * total_1000);

		int trainingSize = training_1000 * 1000;
		int validationSize = totalSize - trainingSize;

		vector<vector<float>> = corpus.load_batch(trainingSize)

		inputStreams[enum2integral(LearningPhase::Training)] =

				read_mnist_image(mnistDataDir + "/" + MNIST_TRAIN_IMAGE_FILE, 0, batchSize, true);

		inputStreams[enum2integral(LearningPhase::Validation)] =
				read_mnist_image(mnistDataDir + "/" + MNIST_TEST_IMAGE_FILE, 0, batchSize, true);

		targetStreams[enum2integral(LearningPhase::Training)] =
				read_mnist_label(mnistDataDir + "/" + MNIST_TRAIN_LABEL_FILE, 0, batchSize);

		targetStreams[enum2integral(LearningPhase::Testing)] =
				read_mnist_label(mnistDataDir + "/" + MNIST_TEST_LABEL_FILE, 0, batchSize);
	}

	virtual ~CorpusDataManager() {}

	void load_input(DataPtr write, bool is_initialized, LearningPhase learnPhase)
	{
		LMN_ASSERT_THROW(learnPhase != LearningPhase::Testing,
				DataException("Corpus load_input: no testing data"));

		if (!is_initialized)
			this->alloc_zeros(write, CORPUS_ONE_HOT_DIM, batchSize);

		int phase = to_int(learnPhase);
		LMN_ASSERT_THROW(streamPos[phase] < streamSizes[phase],
				DataException("load_input stream position out of bound"));

		this->load_data(write, inputStreams[phase][streamPos[phase]]);
	}

	void load_target(DataPtr write, bool is_initialized, LearningPhase learnPhase)
	{
		LMN_ASSERT_THROW(learnPhase != LearningPhase::Testing,
				DataException("Corpus load_target: no testing data"));

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

	// Helper: convert LearningPhase to int
	static int to_int(LearningPhase phase)
	{
		return (int) enum2integral(phase);
	}

private:
	int batchSize;
	int historyLength;

	// Training/Validation/Testing sample sizes
	std::array<int, LEARNING_PHASE_N> streamSizes;

	// current position in input/target stream
	std::array<int, LEARNING_PHASE_N> streamPos;

	// accessed by enum2integral(Training/Validation/Testing)
	// each inner vector<float> is a batch of images/labels
	std::array<vector<vector<float>>, LEARNING_PHASE_N> inputStreams, targetStreams;

	CorpusLoader corpus;
};



#endif /* DEMO_NLP_CORPUS_DATAMAN_H_ */
