/*
 * Eona Studio (c) 2015
 */

#include "../../full_connection.h"
#include "../../loss_layer.h"
#include "../../activation_layer.h"
#include "../../backend/cublas/cublas_engine.h"
#include "../../bias_layer.h"
#include "../../parameter.h"
#include "../../network.h"
#include "../../learning_session.h"

#include "mnist_dataman.h"
#include "../../backend/cublas/cublas_engine.h"

struct CublasMnistDataManager :
		public MnistDataManager<CudaFloatMat>
{
	CublasMnistDataManager(EngineBase::Ptr engine,
					int batchSize,
					string mnistDataDir) :
		MnistDataManager<CudaFloatMat>(engine, batchSize, mnistDataDir)
	{ }

protected:
	// subclass handles actual data load
	void alloc_zeros(DataPtr write, int rowdim, int coldim)
	{
		write->reset(rowdim, coldim);
	}

	// one batch of image (28 * 28 * batchSize)
	void load_data(DataPtr write, vector<float>& imageBatch)
	{
		write->to_device(&imageBatch[0]);
	}
};


struct MnistAccuracyEvaluator : public Evaluator<CublasEngine, float>
{
	MnistAccuracyEvaluator(Network::Ptr net) :
		Evaluator<CublasEngine, float>(net)
	{ }

	virtual ~MnistAccuracyEvaluator() {}

	GEN_CONCRETE_MAKEPTR_STATIC_MEMBER(MnistAccuracyEvaluator)

protected:
	int total;
	int correct;

	/*********** defaults ***********/
	virtual void start_metric(Network::Ptr net, CublasEngine::Ptr engine, LearningPhase phase)
	{
		LMN_ASSERT_THROW(phase == LearningPhase::Testing,
			LearningException("MnistAccuracyEvaluator can only evaluate metric in Testing phase"));

		// clear stats and prepare for metric update
		total = 0;
		correct = 0;
	}

	virtual void update_metric(Network::Ptr net, CublasEngine::Ptr engine, LearningPhase)
	{
		// 10-x-batch matrix, each column is a distribution
		Tensor::Ptr outprob = net->lossLayer->in_value_ptr(0);
		auto distrMat = engine->read_memory(outprob);

		int r = distrMat->DIM_ROW;
		int c = distrMat->DIM_COL;
		vector<float> hostcopy (r * c);
		distrMat->to_host(&hostcopy[0]);

		auto& labelTensor = net->lossLayer->target_value(0);
		auto labels = engine->read_memory(labelTensor);
		vector<float> hostlabel(1 * c);
		labels->to_host(&hostlabel[0]);

		for (int batch = 0; batch < c; ++batch)
		{
			// each batch find the largest value index
			// that's the predicted label
			float mx = -1e20f;
			float predictedLabel = 0;
			auto iter = hostcopy.begin() + batch * 10;
			for (int i = 0; i < 10; ++i)
			{
				if (iter[i] > mx)
				{
					mx = iter[i];
					predictedLabel = i;
				}
			}

			// prediction complete, compare with actual
			if (int(hostlabel[batch]) == predictedLabel)
				correct += 1;
			total += 1;
		}
	}

	virtual float summarize_metric(Network::Ptr, CublasEngine::Ptr, LearningPhase)
	{
		return (float) correct / total;
	}
};

int main(int argc, char **argv)
{
	float lr = 0.002;
	float moment = 0.96;
	if (argc >= 3)
	{
		lr = atof(argv[1]);
		moment = atof(argv[2]);
	}

	const int INPUT_DIM = 28 * 28;
	const int TARGET_DIM = 10;
	const int BATCH_SIZE = 50;
	const int MAX_EPOCH = 100;

	auto engine = EngineBase::make<CublasEngine>();
	auto dataman =
		DataManagerBase::make<CublasMnistDataManager>(engine, BATCH_SIZE, "../data/mnist");

	auto linput = Layer::make<ConstantLayer>(INPUT_DIM);
	auto lhidden1 = Layer::make<SigmoidLayer>(300);
	auto lhidden2 = Layer::make<SigmoidLayer>(300);
	auto lloss = Layer::make<LabelSoftmaxEntropyLayer>(TARGET_DIM);

	auto net = ForwardNetwork::make(engine, dataman);
	net->add_layer(linput);
	net->new_connection<FullConnection>(linput, lhidden1);
	net->new_bias_layer(lhidden1);
	net->add_layer(lhidden1);
	net->new_connection<FullConnection>(lhidden1, lhidden2);
	net->new_bias_layer(lhidden2);
	net->add_layer(lhidden2);
	net->new_connection<FullConnection>(lhidden2, lloss);
	net->add_layer(lloss);

//	auto opm = Optimizer::make<SimpleSGD>(0.02);
//	auto opm = Optimizer::make<MomentumGD>(lr, moment);
	auto opm = Optimizer::make<NesterovMomentum>(lr, moment);
	auto eval = MnistAccuracyEvaluator::make(net);
	auto stopper = StopCriteria::make<MaxEpochStopper>(MAX_EPOCH);
	auto ser = NullSerializer::make();
	auto evalsched = EpochIntervalSchedule::make(0, 1);
	auto obv = NullObserver::make();

	auto session = new_learning_session(net, opm, eval, stopper, ser, evalsched, obv);

	session->initialize();
	session->train();
}

