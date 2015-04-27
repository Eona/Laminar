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
		corpus(corpusFileName),
		// prepare reusable one-hot memory
		onehotTensor(CORPUS_ONE_HOT_DIM * batchSize),
		lastTarget(batchSize)
	{
		// Round the corpus to the nearest thousand
		int totalSize = corpus.size();
		const int historyBatchUnit = historyLength * batchSize;

		// discard the last few short sequences
		totalSize -= totalSize % historyBatchUnit;
		DEBUG_MSG("total size", totalSize);

		// index to LEARNING_PHASE_N
		const int TRAIN = to_int(LearningPhase::Training);
		const int VALID = to_int(LearningPhase::Validation);

		// Divide training/validation by 3:1
		streamSizes[TRAIN] = int(0.75 * (totalSize / historyBatchUnit)) * historyBatchUnit;
		streamSizes[VALID] = totalSize - streamSizes[TRAIN];

		DEBUG_MSG("train size", streamSizes[TRAIN]);
		DEBUG_MSG("valid size", streamSizes[VALID]);
/*
		LMN_ASSERT_THROW(streamSizes[TRAIN] % (historyLength * batchSize) == 0,
				DataException("training size must be divisible by historyLength * batchSize"));
		LMN_ASSERT_THROW(streamSizes[VALID] % (historyLength * batchSize) == 0,
				DataException("validation size must be divisible by historyLength * batchSize"));
*/
		for (int PHASE : { TRAIN, VALID })
		{
			// [[history], [history], [history] ...]
			// segments.size() == number of history sequences
			// segments[i].length == historyLength
			auto segments = corpus.load_segment(streamSizes[PHASE]/historyLength, historyLength);

			int numberOfBatch = segments.size() / batchSize;

			// input stream is continuous, no break of history, i.e. no historyLength
			// each batch is a batch of int labels, will be converted to one-hot matrix later
			// suppose we have history segments [[seg1], [seg2] ... [seg12]]
			// if batch size is 3, then stide = 4, the batches in input stream will look like:
			// [{seg1, seg5, seg9}, {seg2, seg6, seg10}, {seg3, seg7, seg11}, {seg4, seg8, seg12}
			// this arrangment ensures seg2 'follows' seg1 in the next batch
			vector<vector<float>> dataStream;
			int stride = numberOfBatch;
			DEBUG_MSG("number of batches", numberOfBatch);

			for (int batchIdx = 0; batchIdx < numberOfBatch; ++batchIdx)
			{
				for (int h = 0; h < historyLength; ++h)
				{
					vector<float> oneBatchData(batchSize);
					for (int b = 0; b < batchSize; ++b)
					{
						oneBatchData[b] = segments[batchIdx + b * stride][h];
					}
					// push packed batch to inputStreams
					dataStream.push_back(std::move(oneBatchData));
				}
			}
			inputStreams[PHASE] = std::move(dataStream);
		}

		// the last target (at the very end, no next char) will be all spaces
		std::fill(lastTarget.begin(), lastTarget.end(), CorpusLoader::char2code(' '));
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

		// One-hot encoding column major
		vector<float>& labels = inputStreams[phase][streamPos[phase]];

		std::fill(onehotTensor.begin(), onehotTensor.end(), 0);
		for (int b = 0; b < batchSize; ++b)
			onehotTensor[CORPUS_ONE_HOT_DIM * b + (int) labels[b]] = 1.f;

		this->load_data(write, onehotTensor);
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

		// target is always the next char (unless at the very end, fill target as '<space>')
		this->load_data(write,
				streamPos[phase] != streamSizes[phase] - 1 ?
					inputStreams[phase][streamPos[phase] + 1]:
					lastTarget);
	}

	bool prepare_next_batch_impl(LearningPhase learnPhase)
	{
		LMN_ASSERT_THROW(learnPhase != LearningPhase::Testing,
				DataException("Corpus prepare_next_batch: no testing data"));

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
		return { CORPUS_ONE_HOT_DIM, this->batchSize };
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

	CorpusLoader corpus;

	vector<float> onehotTensor; // reuse to encode one-hot
	// At the very end of sequence, when no next char, assume to be space
	vector<float> lastTarget;

public: // for debugging
	// accessed by enum2integral(Training/Validation/Testing)
	// target stream is just next char of inputStream
	std::array<vector<vector<float>>, LEARNING_PHASE_N> inputStreams;
};



#endif /* DEMO_NLP_CORPUS_DATAMAN_H_ */
