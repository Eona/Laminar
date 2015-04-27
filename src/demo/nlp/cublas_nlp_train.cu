/*
 * Eona Studio (c) 2015
 */

#include "../../full_connection.h"
#include "../../gated_connection.h"
#include "../../loss_layer.h"
#include "../../activation_layer.h"
#include "../../lstm.h"
#include "../../bias_layer.h"
#include "../../parameter.h"
#include "../../network.h"
#include "../../rnn.h"
#include "../../learning_session.h"
#include "../../utils/rand_utils.h"

#include "../../backend/cublas/cublas_engine.h"
#include "corpus_loader.h"
#include "corpus_dataman.h"

struct CublasCorpusDataManager :
		public CorpusDataManager<CudaFloatMat>
{
	CublasCorpusDataManager(EngineBase::Ptr engine,
					int batchSize,
					int historyLength,
					string corpusFileName) :
		CorpusDataManager<CudaFloatMat>(
				engine, batchSize, historyLength, corpusFileName)
	{ }

protected:
	// subclass handles actual data load
	void alloc_zeros(DataPtr write, int rowdim, int coldim)
	{
		write->reset(rowdim, coldim);
	}

	void load_data(DataPtr write, vector<float>& data)
	{
		write->to_device(&data[0]);
	}
};

struct BatchObserver :
		public Observer<RecurrentNetwork>
{
	virtual void observe_impl(std::shared_ptr<RecurrentNetwork>, LearningState::Ptr state)
	{
		DEBUG_MSG("Minibatch", state->batchInEpoch);
		DEBUG_MSG("Training loss", state->trainingLoss);
	}

	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(BatchObserver)
};


int main(int argc, char **argv)
{
/*
	CorpusLoader corpus("../data/corpus/shakespeare_hamlet.dat");
	int s = corpus.size();
	DEBUG_MSG(s);

	s -= s % 100;
	for (int b = 0; b < s / 1000 - 1; ++b)
		corpus.load(1000);

	DEBUG_MSG(CorpusLoader::code2str(corpus.load(1000)));
*/
	const string CORPUS_MIDSUMMER = "../data/corpus/shakespeare_midsummer.dat";
	const string CORPUS_UNDER_SEA = "../data/corpus/verne_under_sea.dat";

	const int INPUT_DIM = CORPUS_ONE_HOT_DIM;
	const int TARGET_DIM = CORPUS_ONE_HOT_DIM;
	const int LSTM_DIM = 128;
	const int BATCH_SIZE = 10;
	const int HISTORY_LENGTH = 100;
	const int MAX_EPOCH = 100;

	auto engine = EngineBase::make<CublasEngine>();
	auto dataman = DataManagerBase::make<CublasCorpusDataManager>(
				engine, BATCH_SIZE, HISTORY_LENGTH, CORPUS_UNDER_SEA);

	auto inLayer = Layer::make<ConstantLayer>(INPUT_DIM);
	auto lossLayer = Layer::make<LabelSoftmaxEntropyLayer>(TARGET_DIM);

	auto net = RecurrentNetwork::make(engine, dataman, HISTORY_LENGTH);

	net->add_layer(inLayer);
	auto lstmComposite =
		Composite<RecurrentNetwork>::make<LstmComposite>(inLayer, LSTM_DIM);

	net->add_composite(lstmComposite);
	net->new_connection<FullConnection>(lstmComposite->out_layer(), lossLayer);
	net->add_layer(lossLayer);

	auto opm = Optimizer::make<SimpleSGD>(0.01);
	auto eval = NoMetricEvaluator<CublasEngine>::make(net);
	auto stopper = StopCriteria::make<MaxEpochStopper>(MAX_EPOCH);
	auto ser = NullSerializer::make();
	auto evalsched = EpochIntervalSchedule::make(0, 1);
	auto obv = BatchObserver::make();

	auto session = new_learning_session(net, opm, eval, stopper, ser, evalsched, obv);

	session->initialize();
	session->train();
}


