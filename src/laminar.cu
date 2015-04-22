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
#include "gradient_check.h"

#include "engine/engine.h"
#include "engine/tensor.h"
#include "engine/tensor_ops.h"

#include "backend/dummy/dummy_engine.h"
#include "backend/dummy/dummy_dataman.h"
#include "backend/vecmat/vecmat_dataman.h"
#include "backend/vecmat/vecmat_engine.h"
#include "utils/global_utils.h"
#include "utils/timer.h"

FakeRand& rand_conn = FakeRand::instance_connection();
FakeRand& rand_prehis = FakeRand::instance_prehistory();
FakeRand& rand_input = FakeRand::instance_input();
FakeRand& rand_target = FakeRand::instance_target();

#define conn_full Connection::make<FullConnection>
#define conn_const Connection::make<ConstantConnection>
#define conn_gated Connection::make<GatedConnection>

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

	DEBUG_MSG(A);
	DEBUG_MSG(B);
	DEBUG_MSG("A + A2\n" << A+A2);
	DEBUG_MSG("A - A2\n" << A-A2);
	DEBUG_MSG("-A\n" << -A);
	DEBUG_MSG("A * B\n" << A*B);
	DEBUG_MSG("A t\n" << A.transpose());

	rand_conn.gen_uniform_rand(90, -1.5, 1.5); //rand_conn.print_rand_seq();

	rand_prehis.gen_uniform_rand(30, -.5, .5); //rand_prehis.print_rand_seq();

	rand_input.gen_uniform_rand(20, -1, 1); //rand_input.print_rand_seq();

	rand_target.gen_uniform_rand(40, -1, 1); //rand_target.print_rand_seq();

	const int HISTORY = 5;
	const int INPUT_DIM = 2;
	const int TARGET_DIM = 4;
	const int BATCH = 2;

	auto inLayer = Layer::make<ConstantLayer>(INPUT_DIM);
	auto lossLayer = Layer::make<SquareLossLayer>(TARGET_DIM);

	auto engine = EngineBase::make<VecmatEngine>();
	auto dataman = DataManagerBase::make<VecmatDataManager>(
					engine, INPUT_DIM, TARGET_DIM, BATCH);

	auto net = Network::make<RecurrentNetwork>(engine, dataman, HISTORY);

	net->add_layer(inLayer);

	auto lstmComposite =
			Composite<RecurrentNetwork>::create<LstmComposite>(inLayer, 3);

	net->add_composite(lstmComposite);
	net->new_connection<FullConnection>(lstmComposite.out_layer(), lossLayer);
	net->add_layer(lossLayer);

	LearningSession<RecurrentNetwork> session(net);

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
