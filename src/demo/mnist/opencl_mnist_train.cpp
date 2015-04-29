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

#include "mnist_dataman.h"
#include "../../backend/opencl/opencl_engine.h"

// WARNING OpenCL program must run in the same relative directly w.r.t. the kernel
// For example, if a header refers to "./XXX_kernel.cl", and the header is in A/,
// if you compile the executable to dir B/, you'll have to move the executable
// to dir A/ and execute. Otherwise "invalid kernel name"

struct OpenclMnistDataManager :
		public MnistDataManager<OpenclFloatMat>
{
	OpenclMnistDataManager(EngineBase::Ptr engine,
					int batchSize,
					string mnistDataDir) :
		MnistDataManager<OpenclFloatMat>(engine, batchSize, mnistDataDir),
		cl(EngineBase::cast<OpenclEngine>(engine)->cl)
	{}

protected:
	OclUtilContext* cl;

	// subclass handles actual data load
	void alloc_zeros(DataPtr write, int rowdim, int coldim)
	{
		write->reset(rowdim, coldim, cl);
	}

	// one batch of image (28 * 28 * batchSize)
	void load_data(DataPtr write, vector<float>& imageBatch)
	{
		write->to_device(&imageBatch[0]);
	}
};


int main(int argc, char **argv)
{
	const int INPUT_DIM = 28 * 28;
	const int TARGET_DIM = 10;
	const int BATCH_SIZE = 50;
	const int MAX_EPOCH = 100;

	auto engine = EngineBase::make<OpenclEngine>();
	auto dataman =
		DataManagerBase::make<OpenclMnistDataManager>(engine, BATCH_SIZE, "../data/mnist");

	auto linput = Layer::make<ConstantLayer>(INPUT_DIM);
	auto lhidden1 = Layer::make<SigmoidLayer>(300);
	auto lhidden2 = Layer::make<SigmoidLayer>(200);
	auto lhidden3 = Layer::make<SigmoidLayer>(200);
	auto lloss = Layer::make<LabelSoftmaxEntropyLayer>(TARGET_DIM);

	auto net = ForwardNetwork::make(engine, dataman);
	net->add_layer(linput);
	net->new_connection<FullConnection>(linput, lhidden1);
	net->new_bias_layer(lhidden1);
	net->add_layer(lhidden1);
	net->new_connection<FullConnection>(lhidden1, lhidden2);
	net->new_bias_layer(lhidden2);
	net->add_layer(lhidden2);
	net->new_connection<FullConnection>(lhidden2, lhidden3);
	net->new_bias_layer(lhidden3);
	net->add_layer(lhidden3);
	net->new_connection<FullConnection>(lhidden3, lloss);
	net->add_layer(lloss);

//	net->execute("initialize");
//	net->execute("load_input");
//	net->execute("load_target");
//	net->execute("forward");
//	net->execute("backward");
//	net->execute("zero_clear");

	auto opm = Optimizer::make<SimpleSGD>(0.01);
	auto eval = NoMetricEvaluator<OpenclEngine>::make(net);
	auto stopper = StopCriteria::make<MaxEpochStopper>(MAX_EPOCH);
	auto ser = NullSerializer::make();
	auto evalsched = EpochIntervalSchedule::make(0, 1);
	auto obv = NullObserver::make();

	auto session = new_learning_session(net, opm, eval, stopper, ser, evalsched, obv);

	session->initialize();
	session->train();
}

