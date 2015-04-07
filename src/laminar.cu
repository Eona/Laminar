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

FakeRand& rand_conn = FakeRand::instance_connection();
FakeRand& rand_prehis = FakeRand::instance_prehistory();
FakeRand& rand_input = FakeRand::instance_input();
FakeRand& rand_target = FakeRand::instance_target();

#define make_full Connection::make<FullConnection>
#define make_gated Connection::make<GatedConnection>

int main(int argc, char **argv)
{/*********** FAKE_RAND ***********/
	vector<float> LSTM_CONNECTION_WEIGHTS {
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

	auto forgetGate = Layer::make<SigmoidLayer>();
	auto inputGate = Layer::make<SigmoidLayer>();
	auto cellHatLayer = Layer::make<TanhLayer>();
	auto cellLayer = Layer::make<ConstantLayer>();
	auto outputGate = Layer::make<SigmoidLayer>();
	auto outLayer = Layer::make<ConstantLayer>();

	auto lossLayer = Layer::make<SquareLossLayer>();

	// Naming: c<in><out>_<skip>, or gated: g<in><gate><out>_<skip>
	auto c_in_inputGate = make_full(inLayer, inputGate);
	auto c_outLast_inputGate = make_full(outLayer, inputGate);
	auto c_cellLast_inputGate = make_full(cellLayer, inputGate);

	auto c_in_forgetGate = make_full(inLayer, forgetGate);
	auto c_outLast_forgetGate = make_full(outLayer, forgetGate);
	auto c_cellLast_forgetGate = make_full(cellLayer, forgetGate);

	auto c_in_cellHat = make_full(inLayer, cellHatLayer);
	auto c_outLast_cellHat = make_full(outLayer, cellHatLayer);

	auto g_cellHat_inputGate_cell = make_gated(cellHatLayer, inputGate, cellLayer);
	auto g_cellLast_forgetGate_cell = make_gated(cellLayer, forgetGate, cellLayer);

	auto c_in_outputGate = make_full(inLayer, outputGate);
	auto c_outLast_outputGate = make_full(outLayer, outputGate);
	auto c_cell_outputGate = make_full(cellLayer, outputGate);

	auto g_cell_outputGate_out = Connection::make<GatedTanhConnection>(cellLayer, outputGate, outLayer);

	auto c_out_loss = Connection::make<ConstantConnection>(outLayer, lossLayer);

	vector<ConnectionPtr> fullConns {
		c_in_inputGate,
		c_outLast_inputGate,
		c_cellLast_inputGate,
		c_in_forgetGate,
		c_outLast_forgetGate,
		c_cellLast_forgetGate,
		c_in_cellHat,
		c_outLast_cellHat,
		c_in_outputGate,
		c_outLast_outputGate,
		c_cell_outputGate
	};

	RecurrentNetwork net;
	net.set_input(input);
	net.set_target(target);
	net.set_max_temporal_skip(1);

	net.add_layer(inLayer);

	net.add_connection(c_in_inputGate);
	net.add_recurrent_connection(c_outLast_inputGate);
	net.add_recurrent_connection(c_cellLast_inputGate);
	net.add_layer(inputGate);

	net.add_connection(c_in_forgetGate);
	net.add_recurrent_connection(c_outLast_forgetGate);
	net.add_recurrent_connection(c_cellLast_forgetGate);
	net.add_layer(forgetGate);

	net.add_connection(c_in_cellHat);
	net.add_recurrent_connection(c_outLast_cellHat);

	net.add_layer(cellHatLayer);

	net.add_connection(g_cellHat_inputGate_cell);
	net.add_recurrent_connection(g_cellLast_forgetGate_cell);

	net.add_layer(cellLayer);

	net.add_connection(c_in_outputGate);
	net.add_recurrent_connection(c_outLast_outputGate);
	net.add_connection(c_cell_outputGate);

	net.add_layer(outputGate);

	net.add_connection(g_cell_outputGate_out);

	net.add_layer(outLayer);

	net.add_connection(c_out_loss);

	net.add_layer(lossLayer);

	net.assemble();
	for (int i = 0; i < input.size(); ++i)
		net.forward_prop();
	cout << net.lossLayer->outValues << endl;

	gradient_check(net, 1e-2, 1);

	/************************************/
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
	cout << lstm.lossLayer->outValues << endl;

}
