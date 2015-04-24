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
#include "utils/global_utils.h"
#include "utils/timer.h"

FakeRand& rand_conn = FakeRand::instance_connection();
FakeRand& rand_prehis = FakeRand::instance_prehistory();
FakeRand& rand_input = FakeRand::instance_input();
FakeRand& rand_target = FakeRand::instance_target();

#define conn_full Connection::make<FullConnection>
#define conn_const Connection::make<ConstantConnection>
#define conn_gated Connection::make<GatedConnection>

struct Dudu
{
	Dudu() :
		initGuard("Dudu")
	{ }

	void initialize()
	{
		initGuard.initialize(false);
	}

	void before_init()
	{
		initGuard.assert_before_initialize<ComponentException>("before_dudu");
	}

	void after_init()
	{
		initGuard.assert_after_initialize<EngineException>("after_dudu");
	}

private:
	InitializeGuard<LaminarException> initGuard;
};

int main(int argc, char **argv)
{
	Vecmat<float> A = {
		{9, -2},
		{-3, 4},
		{5, -7}
	};

	Vecmat<float> A2 = {
		{3, 0},
		{-2, 4},
		{10, -7}
	};

	Vecmat<float> B = {
		{-3, 0, 9, 11},
		{-2, -6, 1, 7}
	};

//	const int HISTORY = 5;
	const int INPUT_DIM = 2;
	const int TARGET_DIM = 4;
	const int BATCH = 2;


	rand_conn.gen_uniform_rand(90, -1.5, 1.5); //rand_conn.print_rand_seq();

	rand_prehis.gen_uniform_rand(30, -.5, .5); //rand_prehis.print_rand_seq();

	rand_input.gen_uniform_rand(20, -1, 1); //rand_input.print_rand_seq();

	rand_target.gen_uniform_rand(40, -1, 1); //rand_target.print_rand_seq();

	auto engine = EngineBase::make<VecmatEngine>();
	auto dataman = DataManagerBase::make<VecmatRandDataManager>(
						engine, INPUT_DIM, TARGET_DIM, BATCH);

	auto l1 = Layer::make<ConstantLayer>(INPUT_DIM);
	auto l2 = Layer::make<SigmoidLayer>(5);
	auto l3 = Layer::make<SquareLossLayer>(TARGET_DIM);

	auto net = ForwardNetwork::make(engine, dataman);
	net->add_layer(l1);
	net->new_connection<FullConnection>(l1, l2);
	net->new_bias_layer(l2);
	net->add_layer(l2);
	net->new_connection<FullConnection>(l2, l3);
	net->add_layer(l3);


	auto opm = Optimizer::make<SGD>(2.f);
	auto eval = Evaluator<VecmatEngine>::make(net);
	auto stopper = StopCriteria::make<EpochStopCriteria>();
	auto ser = NullSerializer::make();

	LearningSession session(net, opm, eval, stopper, ser);

	session.initialize();

	auto params = net->get_param_containers();
	DEBUG_MSG(*engine->read_memory(params[0]->param_value_ptr(0)));

	session.train();

	DEBUG_TITLE("After SGD");
	DEBUG_MSG("its gradient:");
	DEBUG_MSG(*engine->read_memory(params[0]->param_gradient_ptr(0)));
	DEBUG_MSG("its new value:");
	DEBUG_MSG(*engine->read_memory(params[0]->param_value_ptr(0)));

/*	net.upload("initialize");
	net.upload("forward");
	net.upload("backward");

	net.compile();

	dummyEng->print_routines();

	DEBUG_TITLE("EXECUTE");
	net.execute("initialize");
	net.execute("forward");
	net.execute("backward");

	cout << dummyEng->read_memory(net.lossLayer->total_loss()) << "\n";*/

	/*Tensor t1(dummyEng, { 2, 3 });
	Tensor t2(dummyEng, {5, 7});
	Tensor t3 = t1 + t2;
	Scalor s1(dummyEng);
	Scalor s2(dummyEng);

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
