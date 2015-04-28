/*
 * Eona Studio (c) 2015
 */

#include "../../backend/vecmat/vecmat_engine.h"
#include "../../utils/rand_utils.h"
#include "corpus_loader.h"
#include "corpus_dataman.h"

struct VecmatCorpusDataManager :
		public CorpusDataManager<lmn::Vecmatf>
{
	VecmatCorpusDataManager(EngineBase::Ptr engine,
					int batchSize,
					int historyLength,
					string corpusFileName) :
		CorpusDataManager<lmn::Vecmatf>(
				engine, batchSize, historyLength, corpusFileName)
	{ }

protected:
	// subclass handles actual data load
	void alloc_zeros(DataPtr write, int rowdim, int coldim)
	{
		write->new_zeros(rowdim, coldim);
	}

	void load_data(DataPtr write, vector<float>& data)
	{
		int i = 0;
		write->fill([&](int r, int c) {
			return data[i++];
		});
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
	const int LSTM_DIM = 128;
	const int BATCH_SIZE = 10;
	const int HISTORY_LENGTH = 100;
	const int MAX_EPOCH = 100;

	FakeRand::instance_connection().gen_uniform_rand(1e5, -0.08f, 0.08f, DEBUG_SEED);

	auto engine = EngineBase::make<VecmatEngine>();
	auto dataman = DataManagerBase::make<VecmatCorpusDataManager>(
				engine, BATCH_SIZE, HISTORY_LENGTH, CORPUS_UNDER_SEA);

/*	auto& streams = dataman->inputStreams;

	int MAX_PRINT = 500;
	for (auto& stream : streams)
	{
		DEBUG_TITLE("NEXT PHASE");
		int i = 0;
		for (int s = 0; s < stream.size(); ++s)
		{
			if (++i > MAX_PRINT)
				break;
			if (s % HISTORY_LENGTH == 0 )
					DEBUG_TITLE("next seq");
			DEBUG_MSG(CorpusLoader::code2str<float>(stream[s]));
		}
	}*/

	auto inLayer = Layer::make<ConstantLayer>(INPUT_DIM);
	auto lossLayer = Layer::make<LabelSoftmaxEntropyLayer>(TARGET_DIM);

	auto net = RecurrentNetwork::make(engine, dataman, HISTORY_LENGTH);

	net->add_layer(inLayer);
	auto lstmComposite =
		Composite<RecurrentNetwork>::make<LstmComposite>(inLayer, LSTM_DIM);

	net->add_composite(lstmComposite);
	net->new_connection<FullConnection>(lstmComposite->out_layer(), lossLayer);
	net->add_layer(lossLayer);

	auto opm = Optimizer::make<MomentumGD>(lr, moment);
	auto eval = NoMetricEvaluator<VecmatEngine>::make(net);
	auto stopper = StopCriteria::make<MaxEpochStopper>(MAX_EPOCH);
	auto ser = NullSerializer::make();
	auto evalsched = EpochIntervalSchedule::make(0, 1);
	auto obv = MinibatchObserver::make();

	auto session = new_learning_session(net, opm, eval, stopper, ser, evalsched, obv);

	session->initialize();
	session->train();
}


