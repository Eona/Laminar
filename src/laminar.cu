/*
 * Eona Studio (c)2015
 */
#include "global_utils.h"
#include "timer.h"
#include "connection.h"
#include "full_connection.h"
#include "gated_connection.h"
#include "activation_layer.h"
#include "backend/dummy/dummy_data.h"
#include "loss_layer.h"
#include "parameter.h"
#include "lstm.h"
#include "network.h"
#include "gradient_check.h"
#include "engine/engine.h"
#include "engine/tensor.h"
#include "engine/tensor_ops.h"
#include "backend/dummy/dummy_engine.h"

FakeRand& rand_conn = FakeRand::instance_connection();
FakeRand& rand_prehis = FakeRand::instance_prehistory();
FakeRand& rand_input = FakeRand::instance_input();
FakeRand& rand_target = FakeRand::instance_target();

#define conn_full Connection::make<FullConnection>
#define conn_const Connection::make<ConstantConnection>
#define conn_gated Connection::make<GatedConnection>

static constexpr const int DUMMY_DIM = 666;

int main(int argc, char **argv)
{
	rand_conn.set_rand_seq(vector<float> {
				.798, 0.617
		});

		rand_prehis.set_rand_seq(vector<float> {
			.3
		});

		vector<float> inputSeq { 1.2, -0.9, 0.57, -1.47, -3.08 };
		vector<float> targetSeq { 1.39, 0.75, -0.45, -0.11, 1.55 };

		rand_input.set_rand_seq(inputSeq);
		rand_target.set_rand_seq(targetSeq);

		auto l1 = Layer::make<ConstantLayer>(DUMMY_DIM);
		auto l2 = Layer::make<ScalorLayer>(DUMMY_DIM, 1.3f);
		auto l3 = Layer::make<CosineLayer>(DUMMY_DIM); // gate
		auto l4 = Layer::make<SquareLossLayer>(DUMMY_DIM);

		auto c12 = conn_full(l1, l2);
		auto c13 = conn_full(l1, l3);

		auto g234_1 = Connection::make<GatedTanhConnection>(l2, l3, l4);
		auto g234_2 = Connection::make<GatedTanhConnection>(l2, l3, l4);

		auto dummyEng = EngineBase::make<DummyEngine>();
		auto dummyData = DataManagerBase::make<DummyDataManager>(dummyEng);

		RecurrentNetwork net(dummyEng, dummyData, inputSeq.size(), 2);

		net.add_layer(l1);
		net.add_connection(c13);
		net.add_layer(l3);
		net.add_connection(c12);
		net.add_layer(l2);
		net.add_recurrent_connection(g234_1);
		net.add_recurrent_connection(g234_2, 2);
		net.add_layer(l4);

		net.execute("initialize");
		net.execute("forward");
		net.execute("backward");

		net.recompile("forward");


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

/*	vector<float> LSTM_CONNECTION_WEIGHTS {
		-0.904, 0.312, -0.944, 1.34, -2.14, -1.69, -2.88, -0.889, -2.28, -0.414, -2.07
	};
	vector<float> LSTM_PREHISTORY {
		.3, -.47
	};

	rand_conn.set_rand_seq(LSTM_CONNECTION_WEIGHTS);
//	rand_conn.use_uniform_rand(-3, 2); rand_conn.set_rand_display(true);
	rand_conn.use_fake_seq();

	rand_prehis.set_rand_seq(LSTM_PREHISTORY);

	rand_prehis.set_rand_seq(LSTM_PREHISTORY);

	vector<float> input {
		1.2, -0.9, 0.57, -1.47, -3.08, 1.2, .31, -2.33, -0.89
	};
	vector<float> target {
		1.39, 0.75, -0.45, -0.11, 1.55, -.44, 2.39, 1.72, -3.06
	};

//	rand_input.use_uniform_rand(-2, 2); rand_target.use_uniform_rand(-2, 2);
//	rand_input.set_rand_display(true); rand_target.set_rand_display(true);
//	vec_apply(input, rand_input); cout << endl; vec_apply(target, rand_target);
	auto inLayer = Layer::make<ConstantLayer>();
	auto lossLayer = Layer::make<SquareLossLayer>();

	RecurrentNetwork net;
	net.set_input(input);
	net.set_target(target);
	net.set_max_temporal_skip(1);

	net.add_layer(inLayer);

	auto lstmComp = Composite<RecurrentNetwork>::create<LstmComposite>(inLayer);

	net.add_composite(lstmComp);

	net.new_connection<ConstantConnection>(lstmComp.out_layer(), lossLayer);

	net.add_layer(lossLayer);

	net.assemble();
	for (int i = 0; i < input.size(); ++i)
		net.forward_prop();
	cout << net.lossLayer->outValues << endl;

	cout << *(lstmComp["forget-gate"]) << "\n";

	gradient_check(net, 1e-2, 1);

	RecurrentNetwork lstm;
	lstm.set_input(input);
	lstm.set_target(target);
	lstm.set_max_temporal_skip(1);

	inLayer = Layer::make<ConstantLayer>();

	auto lstmLayer = Layer::make<LstmDebugLayer>(LSTM_CONNECTION_WEIGHTS, LSTM_PREHISTORY);

	lossLayer = Layer::make<SquareLossLayer>();

	lstm.add_layer(inLayer);
	lstm.new_connection<ConstantConnection>(inLayer, lstmLayer);
	lstm.add_layer(lstmLayer);
	lstm.new_connection<ConstantConnection>(lstmLayer, lossLayer);
	lstm.add_layer(lossLayer);

	lstm.assemble();
	for (int i = 0; i < input.size(); ++i)
		lstm.forward_prop();
	cout << lstm.lossLayer->outValues << endl;*/

}
