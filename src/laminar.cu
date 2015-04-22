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

//static constexpr const int DUMMY_DIM = 666;

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

	rand_conn.set_rand_seq(vector<float> {
		0.869, -0.764, -0.255, 0.771, -0.913, 0.294, -0.957, 0.958, -0.388, -0.184,
		0.922, 0.434, 0.217, 0.655, 0.707, 0.655, 0.368, -0.383, -0.838,
		0.638, -0.706, 0.429, -0.72, -0.439, 0.429, -0.977, 0.858, -0.937,
		0.381, -0.973, 0.764, -0.776, 0.907, 0.483, -0.573, -0.728, 0.587,
		0.102, -0.763, 0.939, 0.876, 0.195, 0.423, 0.0761, -0.364, 0.0478,
		0.558, 0.0241, -0.13, 0.591, -0.294, -0.762, 0.741, 0.0955, 0.784,
		0.398, 0.475, -0.199, -0.533, -0.483, -0.939, -0.344
	});
	rand_conn.gen_uniform_rand(62, -.5, .5);
	rand_conn.print_rand_seq();

	rand_input.set_rand_seq(vector<float> {
		0.276, 2.54, 2.27, 2.81, -0.0979, 0.205
	});
	rand_input.gen_uniform_rand(100, -.5, .5);
	rand_input.print_rand_seq();

	rand_target.set_rand_seq(vector<float> {
		0.457, -0.516, -0.312, 0.126
	});
	rand_target.gen_uniform_rand(100, -.5, .5);
	rand_target.print_rand_seq();

	const int INPUT_DIM = 7;
	const int TARGET_DIM = 13;
	const int BATCH_SIZE = 3;

	auto engine = EngineBase::make<VecmatEngine>();
	auto dataman = DataManagerBase::make<VecmatDataManager>(
					engine, INPUT_DIM, TARGET_DIM, BATCH_SIZE);

	auto l1 = Layer::make<ConstantLayer>(INPUT_DIM);

	auto l2_1 = Layer::make<ScalorLayer>(1, 1.7f);
	auto l2_1_bias = Layer::make<BiasLayer>();
	auto l2_2 = Layer::make<CosineLayer>(3);
	auto l2_2_bias = Layer::make<BiasLayer>();
	auto l3_1 = Layer::make<SigmoidLayer>(2);
	auto l3_1_bias = Layer::make<BiasLayer>();
	auto l3_2 = Layer::make<ScalorLayer>(2, -2.3f);
	auto l3_2_bias = Layer::make<BiasLayer>();

	auto l4 = Layer::make<SquareLossLayer>(TARGET_DIM);

	ForwardNetwork net(engine, dataman);

	net.add_layer(l1);
	net.add_connection(Connection::make<FullConnection>(l1, l2_1));
	net.add_connection(Connection::make<FullConnection>(l1, l2_2));
	// same as add_connection(make_connection<>)
	net.new_connection<FullConnection>(l1, l3_1);
	net.new_connection<FullConnection>(l1, l3_2);
	net.new_connection<FullConnection>(l1, l4);
	net.add_layer(l2_1_bias);
	net.new_connection<FullConnection>(l2_1_bias, l2_1);
	net.add_layer(l2_2_bias);
	net.new_connection<FullConnection>(l2_2_bias, l2_2);
	net.add_layer(l2_1);
	net.add_layer(l2_2);
	net.new_connection<FullConnection>(l2_1, l3_1);
	net.new_connection<FullConnection>(l2_1, l3_2);
	net.new_connection<FullConnection>(l2_1, l4);
	net.new_connection<FullConnection>(l2_2, l3_2);
	net.new_connection<FullConnection>(l2_2, l3_1);
	net.new_connection<FullConnection>(l2_2, l4);
	net.add_layer(l3_1_bias);
	net.new_connection<FullConnection>(l3_1_bias, l3_1);
	net.add_layer(l3_2_bias);
	net.new_connection<FullConnection>(l3_2_bias, l3_2);
	net.add_layer(l3_1);
	net.add_layer(l3_2);
	net.new_connection<FullConnection>(l3_1, l4);
	net.new_connection<FullConnection>(l3_2, l4);
	net.add_layer(l4);

	gradient_check<VecmatEngine, VecmatDataManager>(net, 1e-2f, 0.8f);


	/*auto dummyEng = EngineBase::make<DummyEngine>();

	auto dummyData = DataManagerBase::make<DummyDataManager>(dummyEng);

	ForwardNetwork net(dummyEng, dummyData);

	auto l1 = Layer::make<ConstantLayer>(1);
	auto l2 = Layer::make<SigmoidLayer>(5);
	auto l3 = Layer::make<SquareLossLayer>(1);

	net.add_layer(l1);
	net.new_connection<FullConnection>(l1, l2);
	net.add_layer(l2);
	net.new_connection<FullConnection>(l2, l3);
	net.add_layer(l3);

	gradient_check(net);*/

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
