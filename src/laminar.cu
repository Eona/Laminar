/*
 * Eona Studio (c)2015
 */
#include "connection.h"
#include "full_connection.h"
#include "gated_connection.h"
#include "loss_layer.h"
#include "activation_layer.h"
#include "backend/cublas/cublas_engine.h"
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
//
//#include "backend/dummy/dummy_engine.h"
//#include "backend/dummy/dummy_dataman.h"
#include "backend/vecmat/vecmat_engine.h"
//#include "backend/vecmat/vecmat_rand_dataman.h"
//#include "backend/vecmat/vecmat_func_dataman.h"
//#include "backend/types/cuda_float_mat.h"
//#include "backend/opencl/opencl_engine.h"
#include "backend/types/performance_profiler.h"
#include "backend/eigen/eigen_engine.h"
#include "utils/global_utils.h"
#include "utils/timer.h"

#include "demo/mnist/mnist_parser.h"

#include <Eigen/Dense>
using namespace Eigen;

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


/*	auto images = read_mnist_image(string("../data/mnist/") + MNIST_TRAIN_IMAGE_FILE, 10, 3, true);
	auto mnlabels = read_mnist_label(string("../data/mnist/") + MNIST_TRAIN_LABEL_FILE, 10, 3);

	lmn::Vecmatf mat(28, 28);
	mat.fill([&](int r, int c) {
		return images[1][28 * r + c];
	});
	DEBUG_MSG(mat);
	mat.fill([&](int r, int c) {
		return images[1][2*28*28 + 28 * r + c];
	});
	DEBUG_MSG(mat);

	DEBUG_MSG(mnlabels[0]);
	DEBUG_MSG(mnlabels[1]);
	DEBUG_MSG(mnlabels[2]);*/

	/************************************/
/*	auto engine = std::make_shared<OpenclEngine>();
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
	mem->print_matrix("fei shen");*/


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
