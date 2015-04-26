/*
 * Eona Studio (c)2015
 */
#include "connection.h"
#include "full_connection.h"
#include "gated_connection.h"
#include "loss_layer.h"
#include "activation_layer.h"
#include "bias_layer.h"
#include "parameter.h"
#include "network.h"
#include "lstm.h"
#include "rnn.h"
#include "learning_session.h"
#include "optimizer.h"
#include "gradient_check.h"

#include "engine/engine.h"
#include "engine/tensor.h"
#include "engine/tensor_ops.h"

#include "backend/dummy/dummy_engine.h"
#include "backend/dummy/dummy_dataman.h"
#include "backend/vecmat/vecmat_engine.h"
#include "backend/vecmat/vecmat_rand_dataman.h"
#include "backend/vecmat/vecmat_func_dataman.h"
#include "backend/cublas/cuda_engine.h"
#include "backend/types/cuda_float_mat.h"
#include "backend/opencl/opencl_engine.h"
#include "utils/global_utils.h"
#include "utils/timer.h"

#include "demo/mnist/mnist_parser.h"

FakeRand& rand_conn = FakeRand::instance_connection();
FakeRand& rand_prehis = FakeRand::instance_prehistory();
FakeRand& rand_input = FakeRand::instance_input();
FakeRand& rand_target = FakeRand::instance_target();

#define conn_full Connection::make<FullConnection>
#define conn_const Connection::make<ConstantConnection>
#define conn_gated Connection::make<GatedConnection>

template<typename EngineT>
struct PrintGradient : public Observer<Network>
{
	int maxEpoch;

	PrintGradient(int maxEpoch) :
		maxEpoch(maxEpoch)
	{}

	void observe(Network::Ptr net, LearningState::Ptr state)
	{
		if (false)
		if (state->batchInEpoch == 0 && state->epoch == maxEpoch - 1)
		{
			auto params = net->param_containers();
			DEBUG_TITLE("param gradient");
			for (int i = 0; i < params.size(); ++i)
				DEBUG_MSG(*net->get_engine<EngineT>()->read_memory(params[i]->param_gradient(0)));
			DEBUG_TITLE("param values");
			for (int i = 0; i < params.size(); ++i)
				DEBUG_MSG(*net->get_engine<EngineT>()->read_memory(params[i]->param_value(0)));
		}
	}
};

int main(int argc, char **argv)
{
	auto images = read_mnist_image(string("../data/mnist/") + MnistTrainImageFile, 10, 3, false);
	auto mnlabels = read_mnist_label(string("../data/mnist/") + MnistTrainLabelFile, 100);

	lmn::Vecmatf mat(28, 28);
	mat.fill([&](int r, int c) {
		return images[1][28 * r + c];
	});
	DEBUG_MSG(mat);
	mat.fill([&](int r, int c) {
		return images[1][2*28*28 + 28 * r + c];
	});
	DEBUG_MSG(mat);

	DEBUG_MSG(mnlabels);

	using namespace lmn::VecmatImpl;
	using lmn::Vecmatf;

	Vecmat<float> A = {
		{2.1, -1.2},
		{-3.3, .4},
		{1.65, -.7},
		{-1.33, .57}
	};

	Vecmat<float> A2 = {
			{1.1, 3, 6},
			{7.8, 2, 10},
			{5.9, 5, 5}
	};

	Vecmat<float> B = {
			{1, 2, 0}
	};

	auto a = std::make_shared<Vecmatf>(A2);
	auto ans = std::make_shared<Vecmatf>(A2.row(), A2.col());
	auto labels = std::make_shared<Vecmatf>(B);

	auto scalar = std::make_shared<Vecmatf>(1, 1);

	softmax({a}, ans, true);
	label_entropy_loss({ans, labels}, scalar, true);
	label_softmax_entropy_gradient({ans, labels}, ans, true);

	DEBUG_MSG(*scalar);
	DEBUG_MSG(*ans);

	/************************************/
	auto engine = std::make_shared<OpenclEngine>();
	Tensor t1(engine, {4, 7});
	Tensor t2(engine, {7, 9});

	lmn::fill_rand(t1);
	lmn::fill_rand(t2);
	engine->flush_execute();
	auto mem1 = engine->read_memory(t1);
	auto mem2 = engine->read_memory(t2);
	mem1->print_matrix("mem1");
	mem2->print_matrix("mem2");


	Tensor t3 = t1 * t2;

	engine->flush_execute();

	auto mem = engine->read_memory(t3);
	mem->print_matrix("fei shen");

/*
//	const int HISTORY = 5;
	const int INPUT_DIM = 4;
	const int TARGET_DIM = 3;
	const int BATCH = 2;
	const int MAX_EPOCH = 1000;

	rand_conn.gen_uniform_rand(90, -0.1, 0.1, DEBUG_SEED); //rand_conn.print_rand_seq();
//
//	rand_prehis.gen_uniform_rand(30, -.5, .5); //rand_prehis.print_rand_seq();
//
//	rand_input.gen_uniform_rand(20, -1, 1); //rand_input.print_rand_seq();
//
//	rand_target.gen_uniform_rand(40, -1, 1); //rand_target.print_rand_seq();

	auto learnableFunc = [](const lmn::Vecmatf& in, lmn::Vecmatf& out) {
		// Each column is a batch
		for (int c = 0; c < in.col(); ++c)
		{
//			out(0, c) = sin(in(0, c)) + cos(in(1, c));
//			out(1, c) = cos(in(1, c)) + sin(in(2, c));
//			out(2, c) = 2 * sin(in(2, c)) - cos(in(3, c));
			out(0, c) = cos(in(0, c)) * in(1, c) + in(2, c);
			out(1, c) = in(1, c) * sin(in(2, c)) + in(3, c);
			out(2, c) = in(2, c) * in(3, c) + sin(in(0, c));
		}
	};

	auto engine = EngineBase::make<VecmatEngine>();
	auto dataman = DataManagerBase::make<VecmatFuncDataManager>(
						engine, INPUT_DIM, TARGET_DIM, BATCH,
						learnableFunc,
						100, 20, 10,
						-1.f, 1.f);

	auto linput = Layer::make<ConstantLayer>(INPUT_DIM);
	auto l2 = Layer::make<SigmoidLayer>(100);
	auto l3 = Layer::make<SigmoidLayer>(100);
	auto lloss = Layer::make<SquareLossLayer>(TARGET_DIM);

	auto net = ForwardNetwork::make(engine, dataman);
	net->add_layer(linput);
	net->new_connection<FullConnection>(linput, l2);
	net->new_bias_layer(l2);
	net->add_layer(l2);
	net->new_connection<FullConnection>(l2, l3);
	net->new_bias_layer(l3);
	net->add_layer(l3);
	net->new_connection<FullConnection>(l3, lloss);
	net->add_layer(lloss);

	auto opm = Optimizer::make<SGD>(0.3);
	auto eval = NoMetricEvaluator<VecmatEngine>::make(net);
	auto stopper = StopCriteria::make<MaxEpochStopper>(MAX_EPOCH);
	auto ser = NullSerializer::make();
	auto sched = EpochIntervalSchedule::make(1, 1);

	auto session = new_learning_session(net, opm, eval, stopper, ser, sched,
			std::make_shared<PrintGradient<VecmatEngine>>(MAX_EPOCH));

	session->initialize();

	session->train();
*/


	/*Tensor t1(dummyEng, { 2, 3 });
	Tensor t2(dummyEng, {5, 7});
	Tensor t3 = t1 + t2;
	Scalar s1(dummyEng);
	Scalar s2(dummyEng);

//	t1 -= t2;
//	t1 += t2;
//	t1 *= t2;
//	t1 *= s2;
//	s1 *= s2;
//	s1 += s2;
//	s1 -= s2;

	cout << "t3 " << t3.addr << endl;
	t3 = t3 + t3 - t1;
	cout << "t3 " << t3.addr << endl;
	t1 = t3 + 6.6f*t1 + t3;
	t3 = t1 * 3.5f;
	t1 *= 100.88f;
	cout << "t3 " << t3.addr << endl;

	dummyEng->print_routines();
	dummyEng->flush_execute();*/


/*	dummyEng->print_instructions();
	print_title("optimize");
	dummyEng->eliminate_temporary();
	dummyEng->print_instructions();

	for (auto pr : dummyEng->memoryPool.dimensions)
		DEBUG_MSG(pr.first << " " << pr.second);

	print_title("Graph");
	dummyEng->construct_graph();
	dummyEng->print_graph();*/

	DEBUG_TITLE("DONE");
}
