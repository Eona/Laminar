/*
 * Eona Studio (c)2015
 */
#include "global_utils.h"
#include "timer.h"
#include "connection.h"
#include "full_connection.h"
#include "gated_connection.h"
#include "transfer_layer.h"
#include "loss_layer.h"
#include "parameter.h"
#include "lstm.h"
#include "network.h"
#include "gradient_check.h"
#include "engine/engine.h"
#include "engine/tensor.h"
#include "engine/dummy_engine.h"

FakeRand& rand_conn = FakeRand::instance_connection();
FakeRand& rand_prehis = FakeRand::instance_prehistory();
FakeRand& rand_input = FakeRand::instance_input();
FakeRand& rand_target = FakeRand::instance_target();

#define make_full Connection::make<FullConnection>
#define make_gated Connection::make<GatedConnection>

int main(int argc, char **argv)
{
	auto dummyEng = EngineBase::make<DummyEngine>();

	Tensor t1(dummyEng);
	Tensor t2(dummyEng);
	Tensor t3 = t1 + t2;
	cout << "t3 " << t3.addr << endl;
	t3 = t3 + t3 ;
	cout << "t3 " << t3.addr << endl;
	t1 = t3 + t1;
	t3 = t1;
	cout << "t3 " << t3.addr << endl;

	dummyEng->print_instructions();

	cout << endl;
	print_title("After elimination");
	dummyEng->eliminate_temporary();
	dummyEng->print_instructions();

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
