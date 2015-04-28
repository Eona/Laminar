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

#include "mnist_dataman.h"
#include "../../backend/vecmat/vecmat_engine.h"

struct VecmatMnistDataManager :
		public MnistDataManager<lmn::Vecmatf>
{
	VecmatMnistDataManager(EngineBase::Ptr engine,
					int batchSize,
					string mnistDataDir) :
		MnistDataManager<lmn::Vecmatf>(engine, batchSize, mnistDataDir)
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
		int i = 0;
		write->fill([&](int r, int c) {
			return imageBatch[i++];
		});
	}
};


int main(int argc, char **argv)
{
	const int INPUT_DIM = 28 * 28;
	const int TARGET_DIM = 10;
	const int BATCH_SIZE = 50;
	const int MAX_EPOCH = 100;

	FakeRand::instance_connection().gen_uniform_rand(
			300*300 + 300*764 + 10*300, -0.08f, 0.08f, DEBUG_SEED);

	auto engine = EngineBase::make<VecmatEngine>();
	auto dataman =
		DataManagerBase::make<VecmatMnistDataManager>(engine, BATCH_SIZE, "../data/mnist");

	auto linput = Layer::make<ConstantLayer>(INPUT_DIM);
	auto lhidden1 = Layer::make<SigmoidLayer>(300);
	auto lhidden2 = Layer::make<SigmoidLayer>(300);
	auto lloss = Layer::make<LabelSoftmaxEntropyLayer>(TARGET_DIM);

	auto net = ForwardNetwork::make(engine, dataman);
	net->add_layer(linput);
	net->new_connection<FullConnection>(linput, lhidden1);
//	net->new_bias_layer(lhidden1);
	net->add_layer(lhidden1);
	net->new_connection<FullConnection>(lhidden1, lhidden2);
//	net->new_bias_layer(lhidden2);
	net->add_layer(lhidden2);
	net->new_connection<FullConnection>(lhidden2, lloss);
	net->add_layer(lloss);

	auto opm = Optimizer::make<SimpleSGD>(0.01);
	auto eval = NoMetricEvaluator<VecmatEngine>::make(net);
	auto stopper = StopCriteria::make<MaxEpochStopper>(MAX_EPOCH);
	auto ser = NullSerializer::make();
	auto evalsched = EpochIntervalSchedule::make(0, 1);
	auto obv = NullObserver::make();

	auto session = new_learning_session(net, opm, eval, stopper, ser, evalsched, obv);

	session->initialize();
	session->train();
}

