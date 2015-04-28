/*
 * Eona Studio (c) 2015
 */

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


int main(int argc, char **argv)
{
	float lr = 0.0001;
	float moment = 0.9;
	if (argc == 3)
	{
		lr = atof(argv[1]);
		moment = atof(argv[2]);
	}
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
	const int LSTM_DIM = 230;
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

//	auto sigLayer1 = Layer::make<SigmoidLayer>(LSTM_DIM);
//	auto sigLayer2 = Layer::make<SigmoidLayer>(LSTM_DIM);
//	net->new_connection<FullConnection>(inLayer, sigLayer1);
//	net->new_recur_connection<FullConnection>(sigLayer1, sigLayer1);
//	net->new_bias_layer(sigLayer1);
//	net->add_layer(sigLayer1);
//	net->new_connection<FullConnection>(sigLayer1, sigLayer2);
//	net->new_recur_connection<FullConnection>(sigLayer2, sigLayer2);
//	net->new_bias_layer(sigLayer2);
//	net->add_layer(sigLayer2);
//	net->new_connection<FullConnection>(sigLayer2, lossLayer);

	auto lstmComposite =
		Composite<RecurrentNetwork>::make<LstmComposite>(inLayer, LSTM_DIM);
	net->add_composite(lstmComposite);
	net->new_connection<FullConnection>(lstmComposite->out_layer(), lossLayer);

	net->add_layer(lossLayer);

//	auto opm = Optimizer::make<SimpleSGD>(lr);
	auto opm = Optimizer::make<ClippedMomentumGD>(lr, moment);
	auto eval = NoMetricEvaluator<CublasEngine>::make(net);
	auto stopper = StopCriteria::make<MaxEpochStopper>(MAX_EPOCH);
	auto ser = NullSerializer::make();
	auto evalsched = EpochIntervalSchedule::make(1, 0);
	auto obv = MinibatchObserver::make();

	auto session = new_learning_session(net, opm, eval, stopper, ser, evalsched, obv);

//	net->execute("initialize");
//	net->execute("load_input");
//	net->execute("load_target");
//
//	auto inveri = engine->read_memory(net->layers[0]->in_value_ptr(0));
//	auto tarveri = engine->read_memory(net->lossLayer->target_value(0));

	session->initialize();
	session->train();
}


