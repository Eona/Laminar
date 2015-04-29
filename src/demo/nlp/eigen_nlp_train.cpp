/*
 * Eona Studio (c) 2015
 */

#include "../../backend/eigen/eigen_engine.h"
#include "corpus_loader.h"
#include "corpus_dataman.h"

struct EigenCorpusDataManager :
		public CorpusDataManager<MatrixXf>
{
	EigenCorpusDataManager(EngineBase::Ptr engine,
					int batchSize,
					int historyLength,
					string corpusFileName) :
		CorpusDataManager<MatrixXf>(
				engine, batchSize, historyLength, corpusFileName)
	{ }

protected:
	// subclass handles actual data load
	void alloc_zeros(DataPtr write, int rowdim, int coldim)
	{
		*write = MatrixXf::Zero(rowdim, coldim);
	}

	void load_data(DataPtr write, vector<float>& data)
	{
		int i = 0;
		for (int c = 0; c < batch_size(); ++c)
			for (int r = 0; r < data.size() / batch_size(); ++r)
				(*write)(r, c) = data[i++];
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

	const string CORPUS_MIDSUMMER = "../data/corpus/shakespeare_midsummer.dat";

	const int INPUT_DIM = CORPUS_ONE_HOT_DIM;
	const int TARGET_DIM = CORPUS_ONE_HOT_DIM;
	const int LSTM_DIM = 230;
	const int BATCH_SIZE = 10;
	const int HISTORY_LENGTH = 100;
	const int MAX_EPOCH = 100;

	auto engine = EngineBase::make<EigenEngine>();
	auto dataman = DataManagerBase::make<EigenCorpusDataManager>(
				engine, BATCH_SIZE, HISTORY_LENGTH, CORPUS_MIDSUMMER);

	auto inLayer = Layer::make<ConstantLayer>(INPUT_DIM);
	auto lossLayer = Layer::make<LabelSoftmaxEntropyLayer>(TARGET_DIM);

	auto net = RecurrentNetwork::make(engine, dataman, HISTORY_LENGTH);

	net->add_layer(inLayer);

	auto lstmComposite =
		Composite<RecurrentNetwork>::make<LstmComposite>(inLayer, LSTM_DIM);
	net->add_composite(lstmComposite);
	net->new_connection<FullConnection>(lstmComposite->out_layer(), lossLayer);

	net->add_layer(lossLayer);

	auto opm = Optimizer::make<ClippedMomentumGD>(lr, moment);
	auto eval = NoMetricEvaluator<EigenEngine>::make(net);
	auto stopper = StopCriteria::make<MaxEpochStopper>(MAX_EPOCH);
	auto ser = NullSerializer::make();
	auto evalsched = EpochIntervalSchedule::make(1, 0);
	auto obv = MinibatchObserver::make();

	auto session = new_learning_session(net, opm, eval, stopper, ser, evalsched, obv);

	session->initialize();
	session->train();
}


