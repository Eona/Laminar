/*
 * Eona Studio (c)2015
 */
#include "global_utils.h"
#include "timer.h"
#include "connection.h"
#include "full_connection.h"
#include "gated_connection.h"
#include "activation_layer.h"
#include "backend/dummy/dummy_dataman.h"
#include "loss_layer.h"
#include "parameter.h"
#include "lstm.h"
#include "network.h"
#include "gradient_check.h"

#include "engine/engine.h"
#include "engine/tensor.h"
#include "engine/tensor_ops.h"

#include "backend/dummy/dummy_engine.h"
#include "backend/vector/vector_engine.h"
#include "backend/vector/vector_mat.h"

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
	VectorMat<float> A = {
		{9, -2},
		{-3, 4},
		{5, -7}
	};

	VectorMat<float> A2 = {
		{3, 0},
		{-2, 4},
		{10, -7}
	};

	VectorMat<float> B = {
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


	auto eng = EngineBase::make<VectorEngine>();

	auto get = [eng] (const TensorBase& t) {
		return *eng->read_memory(t);
	};

	auto exec = [eng] () { return eng->flush_execute(); };

	Tensor t1(eng, {3, 4});
	Tensor t2(eng, {4, 2});
	lmn::fill_rand(t1);
	lmn::fill_rand(t2);

	t1 = lmn::sigmoid(t1);
	t2 = lmn::tanh_gradient(t2);
	Tensor t3 = t1 * t2;

	eng->eliminate_temporary();
	auto routine = exec();
	DEBUG_MSG(get(t3));

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
