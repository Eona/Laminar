/*
 * Eona Studio (c) 2015
 */

#include "../../full_connection.h"
#include "../../loss_layer.h"
#include "../../activation_layer.h"
#include "../../bias_layer.h"
#include "../../parameter.h"
#include "../../network.h"
#include "../../learning_session.h"
#include "../../utils/rand_utils.h"

#include "../../backend/vecmat/vecmat_engine.h"
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

	// one batch of image (28 * 28 * batchSize)
	void load_data(DataPtr write, vector<float>& imageBatch)
	{
//		write->fill([&](int r, int c) {
//			return imageBatch[r + c * MNIST_INPUT_DIM];
//		});
	}
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

	const int INPUT_DIM = 28 * 28;
	const int TARGET_DIM = 10;
	const int BATCH_SIZE = 10;
	const int HISTORY_LENGTH = 100;
	const int MAX_EPOCH = 100;
//
//	FakeRand::instance_connection().gen_uniform_rand(
//					300*300 + 300*764 + 10*300, -0.08f, 0.08f, DEBUG_SEED);

	auto engine = EngineBase::make<VecmatEngine>();
	auto dataman = DataManagerBase::make<VecmatCorpusDataManager>(
//			engine, BATCH_SIZE, HISTORY_LENGTH, "../data/corpus/dummy.dat");
			engine, BATCH_SIZE, HISTORY_LENGTH, argv[1]);

	auto& streams = dataman->inputStreams;

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
	}


//	auto linput = Layer::make<ConstantLayer>(INPUT_DIM);
//	auto lhidden1 = Layer::make<SigmoidLayer>(300);
//	auto lhidden2 = Layer::make<SigmoidLayer>(300);
//	auto lloss = Layer::make<LabelSoftmaxEntropyLayer>(TARGET_DIM);
//
//	auto net = ForwardNetwork::make(engine, dataman);
//	net->add_layer(linput);
//	net->new_connection<FullConnection>(linput, lhidden1);
////	net->new_bias_layer(lhidden1);
//	net->add_layer(lhidden1);
//	net->new_connection<FullConnection>(lhidden1, lhidden2);
////	net->new_bias_layer(lhidden2);
//	net->add_layer(lhidden2);
//	net->new_connection<FullConnection>(lhidden2, lloss);
//	net->add_layer(lloss);
//
//	auto opm = Optimizer::make<SimpleSGD>(0.01);
//	auto eval = NoMetricEvaluator<VecmatEngine>::make(net);
//	auto stopper = StopCriteria::make<MaxEpochStopper>(MAX_EPOCH);
//	auto ser = NullSerializer::make();
//	auto evalsched = EpochIntervalSchedule::make(0, 1);
//	auto obv = NullObserver::make();
//
//	auto session = new_learning_session(net, opm, eval, stopper, ser, evalsched, obv);
//
//	session->initialize();
//	session->train();
}


