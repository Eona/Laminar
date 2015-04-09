/*
 * Eona Studio (c) 2015
 */

#include "test_utils.h"

FakeRand& rand_conn = FakeRand::instance_connection();
FakeRand& rand_prehis = FakeRand::instance_prehistory();
FakeRand& rand_input = FakeRand::instance_input();
FakeRand& rand_target = FakeRand::instance_target();

TEST(RecurrentNet, Simple)
{
	rand_conn.set_rand_seq(vector<float> {
		0.543, 0.44, 1.47, 1.64, 1.31, -0.616
	});
	rand_prehis.set_rand_seq(vector<float> {
		.7
	});
	rand_conn.use_fake_seq();
//	rand_conn.use_uniform_rand(-1, 2);
//	rand_conn.set_rand_display(true);

	vector<float> input { 1.2, -0.9, 0.57, -1.47, -3.08 };
	vector<float> target { 1.39, 0.75, -0.45, -0.11, 1.55 };

	auto l1 = Layer::make<ConstantLayer>();
	auto l2 = Layer::make<SigmoidLayer>();
	auto l3 = Layer::make<SigmoidLayer>();
	auto l4 = Layer::make<SquareLossLayer>();

	// Naming: c<in><out>_<skip>
	auto c12 = Connection::make<FullConnection>(l1, l2);
	auto c23 = Connection::make<FullConnection>(l2, l3);
	auto c34 = Connection::make<FullConnection>(l3, l4);

	auto c22_1 = Connection::make<FullConnection>(l2, l2);
	auto c23_1 = Connection::make<FullConnection>(l2, l3);
	auto c33_1 = Connection::make<FullConnection>(l3, l3);

	RecurrentNetwork net;
	net.set_input(input);
	net.set_target(target);

	net.add_layer(l1);
	net.add_recurrent_connection(c22_1);
	net.add_connection(c12);

	net.add_layer(l2);

	net.add_recurrent_connection(c23_1);
	net.add_recurrent_connection(c33_1);
	net.add_connection(c23);

	net.add_layer(l3);
	net.add_connection(c34);
	net.add_layer(l4);
/*
	RecurrentNetwork net;
	net.set_input(input);
	net.set_target(target);
	net.add_layer(l1);
	net.new_connection<LinearConnection>(l1, l2);
	net.new_recurrent_connection<LinearConnection>(l2, l2);
	net.add_layer(l2);
	net.new_recurrent_connection<LinearConnection>(l2, l3);
	net.new_connection<LinearConnection>(l2, l3);
	net.new_recurrent_connection<LinearConnection>(l3, l3);
	net.add_layer(l3);
	net.new_connection<LinearConnection>(l3, l4);
	net.add_layer(l4);
*/
	gradient_check(net, 1e-2, 1);
}

TEST(RecurrentNet, TemporalSkip)
{
	rand_conn.set_rand_seq(vector<float> {
		0.163, 1.96, 1.09, 0.516, -0.585, 0.776, 1, -0.301, -0.167, 0.732
	});

//	rand_conn.use_uniform_rand(-1, 2);
//	rand_conn.set_rand_display(true);
	rand_conn.use_fake_seq();

	rand_prehis.set_rand_seq(vector<float> {
		.3
	});

	vector<float> input { 1.2, -0.9, 0.57, -1.47, -3.08, 1.2, .31, -2.33, -0.89 };
	vector<float> target { 1.39, 0.75, -0.45, -0.11, 1.55, -.44, 2.39, 1.72, -3.06 };

	auto l1 = Layer::make<ConstantLayer>();
	auto l2 = Layer::make<SigmoidLayer>();
	auto l3 = Layer::make<CosineLayer>();
	auto l4 = Layer::make<SquareLossLayer>();

	// NOTE IMPORTANT RULE
	// For recurrent linear connection conn[layer(alpha) => layer(beta)]
	// Must be added before you add layer(beta). alpha doesn't matter

	// Naming: c<in><out>_<skip>
	auto c12 = conn_full(l1, l2);
	auto c23 = conn_full(l2, l3);
	auto c34 = conn_full(l3, l4);

	auto c22_1 = conn_full(l2, l2);
	auto c22_3 = conn_full(l2, l2);
	auto c23_1 = conn_full(l2, l3);
	auto c23_2 = conn_full(l2, l3);
	auto c32_3 = conn_full(l3, l2);
	auto c33_1 = conn_full(l3, l3);
	auto c33_2 = conn_full(l3, l3);

	RecurrentNetwork net;
	net.set_input(input);
	net.set_target(target);
	net.set_max_temporal_skip(3); // or Layer::UNLIMITED_TEMPORAL_SKIP

	net.add_layer(l1);

	net.add_connection(c12);
	net.add_recurrent_connection(c22_1);
	net.add_recurrent_connection(c22_3, 3);
	net.add_recurrent_connection(c32_3, 3);

	net.add_layer(l2);

	net.add_connection(c23);
	net.add_recurrent_connection(c23_1);
	net.add_recurrent_connection(c23_2, 2);
	net.add_recurrent_connection(c33_1);
	net.add_recurrent_connection(c33_2, 2);

	net.add_layer(l3);
	net.add_connection(c34);
	net.add_layer(l4);

/*
	net.add_layer(l1);

	net.new_recurrent_connection<LinearConnection>(l2, l2);
	net.new_recurrent_skip_connection<LinearConnection>(3, l2, l2);
	net.new_recurrent_skip_connection<LinearConnection>(3, l3, l2);
	net.new_connection<LinearConnection>(l1, l2);

	net.add_layer(l2);

	net.new_connection<LinearConnection>(l2, l3);
	net.new_recurrent_skip_connection<LinearConnection>(2, l2, l3);
	net.new_recurrent_connection<LinearConnection>(l2, l3);
	net.new_recurrent_connection<LinearConnection>(l3, l3);
	net.new_recurrent_skip_connection<LinearConnection>(2, l3, l3);

	net.add_layer(l3);
	net.new_connection<LinearConnection>(l3, l4);
	net.add_layer(l4);
*/

	gradient_check(net, 1e-2, 1);

	net.reset();
	for (int i = 0; i < net.input.size(); ++i)
		net.forward_prop();
	for (int i = 0; i < net.input.size(); ++i)
		net.backward_prop();

/*	for (ConnectionPtr c : { c12, c23, c34, c22_1, c22_3, c23_1, c23_2, c32_3, c33_1, c33_2 })
		cout << std::setprecision(4) << Connection::cast<LinearConnection>(c)->gradient << "  ";
	cout << endl;
	for (LayerPtr l : { l2, l3 })
		cout << std::setprecision(4) << static_cast<ParamContainerPtr>(net.prehistoryLayerMap[l])->paramGradients << "  ";
	cout << endl;*/
}


TEST(RecurrentNet, GatedConnection)
{
	rand_conn.set_rand_seq(vector<float> {
			0.163, 1.96, 1.09, 0.516, -0.585, 0.776, 1, -0.301, -0.167, 0.732
	});

	rand_conn.use_fake_seq();

	rand_prehis.set_rand_seq(vector<float> {
		.3
	});

	vector<float> input { 1.2, -0.9, 0.57, -1.47, -3.08, 1.2, .31, -2.33, -0.89 };
	vector<float> target { 1.39, 0.75, -0.45, -0.11, 1.55, -.44, 2.39, 1.72, -3.06 };

	auto l1 = Layer::make<ConstantLayer>();
	auto l2 = Layer::make<SigmoidLayer>();
	auto l3 = Layer::make<CosineLayer>(); // gate
	auto l4 = Layer::make<SquareLossLayer>();

	// NOTE IMPORTANT RULE
	// For recurrent gated connection conn[layer(alpha), layer(gate) => layer(beta)]
	// where alpha is t-1 (or more) and gate/beta are the current t
	// Must be added after you add gate & alpha, and before beta.
	// If recurrent, alpha doesn't necessary need to precede this connection
	// (because layer alpha lives in the past)

	// Naming: c<in><out>_<skip>
	// g<in><gate><out>_<skip>
	auto c12 = conn_full(l1, l2);
	auto c13 = conn_full(l1, l3);

	auto g234_1 = Connection::make<GatedConnection>(l2, l3, l4);
	auto g234_2 = Connection::make<GatedConnection>(l2, l3, l4);

	RecurrentNetwork net;
	net.set_input(input);
	net.set_target(target);
	net.set_max_temporal_skip(2);

	net.add_layer(l1);

	net.add_connection(c13);
	net.add_layer(l3);

	net.add_connection(c12);
	net.add_layer(l2);

	net.add_recurrent_connection(g234_1);
	net.add_recurrent_connection(g234_2, 2);

	net.add_layer(l4);

	gradient_check(net, 1e-2, 1);
}


TEST(RecurrentNet, GatedTanhConnection)
{
	rand_conn.set_rand_seq(vector<float> {
			.798, 0.617
	});
	rand_conn.use_fake_seq();

	rand_prehis.set_rand_seq(vector<float> {
		.3
	});

	vector<float> input { 1.2, -0.9, 0.57, -1.47, -3.08 };
	vector<float> target { 1.39, 0.75, -0.45, -0.11, 1.55 };

	auto l1 = Layer::make<ConstantLayer>();
	auto l2 = Layer::make<ScalorLayer>(1.3f);
	auto l3 = Layer::make<CosineLayer>(); // gate
	auto l4 = Layer::make<SquareLossLayer>();

	auto c12 = conn_full(l1, l2);
	auto c13 = conn_full(l1, l3);

	auto g234_1 = Connection::make<GatedTanhConnection>(l2, l3, l4);
	auto g234_2 = Connection::make<GatedTanhConnection>(l2, l3, l4);

	RecurrentNetwork net;
	net.set_input(input);
	net.set_target(target);
	net.set_max_temporal_skip(2);

	net.add_layer(l1);
	net.add_connection(c13);
	net.add_layer(l3);
	net.add_connection(c12);
	net.add_layer(l2);
	net.add_recurrent_connection(g234_1);
	net.add_recurrent_connection(g234_2, 2);
	net.add_layer(l4);

	gradient_check(net, 1e-2, 1);
}


TEST(RecurrentNet, LSTM)
{
	/*********** FAKE_RAND ***********/
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

	vector<float> input {
		1.2, -0.9, 0.57, -1.47, -3.08, 1.2, .31, -2.33, -0.89
	};
	vector<float> target {
		1.39, 0.75, -0.45, -0.11, 1.55, -.44, 2.39, 1.72, -3.06
	};

//	rand_input.use_uniform_rand(-2, 2); rand_target.use_uniform_rand(-2, 2);
//	rand_input.set_rand_display(true); rand_target.set_rand_display(true);
//	vec_apply(input, rand_input); cout << endl; vec_apply(target, rand_target);

	/*********** LSTM layers ***********/
	auto inLayer = Layer::make<ConstantLayer>();

	auto forgetGate = Layer::make<SigmoidLayer>();
	auto inputGate = Layer::make<SigmoidLayer>();
	auto cellHatLayer = Layer::make<TanhLayer>();
	auto cellLayer = Layer::make<ConstantLayer>();
	auto outputGate = Layer::make<SigmoidLayer>();
	auto outLayer = Layer::make<ConstantLayer>();

	auto lossLayer = Layer::make<SquareLossLayer>();

	/*********** LSTM connections ***********/
	// Naming: c<in><out>_<skip>, or gated: g<in><gate><out>_<skip>
	auto c_in_inputGate = conn_full(inLayer, inputGate);
	auto c_outLast_inputGate_1 = conn_full(outLayer, inputGate);
	auto c_cellLast_inputGate_1 = conn_full(cellLayer, inputGate);

	auto c_in_forgetGate = conn_full(inLayer, forgetGate);
	auto c_outLast_forgetGate_1 = conn_full(outLayer, forgetGate);
	auto c_cellLast_forgetGate_1 = conn_full(cellLayer, forgetGate);

	auto c_in_cellHat = conn_full(inLayer, cellHatLayer);
	auto c_outLast_cellHat_1 = conn_full(outLayer, cellHatLayer);

	auto g_cellHat_inputGate_cell = conn_gated(cellHatLayer, inputGate, cellLayer);
	auto g_cellLast_forgetGate_cell_1 = conn_gated(cellLayer, forgetGate, cellLayer);

	auto c_in_outputGate = conn_full(inLayer, outputGate);
	auto c_outLast_outputGate_1 = conn_full(outLayer, outputGate);
	auto c_cell_outputGate = conn_full(cellLayer, outputGate);

	auto g_cell_outputGate_out = Connection::make<GatedTanhConnection>(cellLayer, outputGate, outLayer);

	auto c_out_loss = conn_const(outLayer, lossLayer);

	vector<ConnectionPtr> lstmFullConnections {
		c_in_inputGate,
		c_outLast_inputGate_1,
		c_cellLast_inputGate_1,
		c_in_forgetGate,
		c_outLast_forgetGate_1,
		c_cellLast_forgetGate_1,
		c_in_cellHat,
		c_outLast_cellHat_1,
		c_in_outputGate,
		c_outLast_outputGate_1,
		c_cell_outputGate
	};

	/*********** Construct the network ***********/
	// order of topology see "NOTE" sections in simple RecurrentNet gtests.
	RecurrentNetwork net;
	net.set_input(input);
	net.set_target(target);
	net.set_max_temporal_skip(1);

	net.add_layer(inLayer);

	net.add_connection(c_in_inputGate);
	net.add_recurrent_connection(c_outLast_inputGate_1);
	net.add_recurrent_connection(c_cellLast_inputGate_1);
	net.add_layer(inputGate);

	net.add_connection(c_in_forgetGate);
	net.add_recurrent_connection(c_outLast_forgetGate_1);
	net.add_recurrent_connection(c_cellLast_forgetGate_1);
	net.add_layer(forgetGate);

	net.add_connection(c_in_cellHat);
	net.add_recurrent_connection(c_outLast_cellHat_1);

	net.add_layer(cellHatLayer);

	net.add_connection(g_cellHat_inputGate_cell);
	net.add_recurrent_connection(g_cellLast_forgetGate_cell_1);

	net.add_layer(cellLayer);

	net.add_connection(c_in_outputGate);
	net.add_recurrent_connection(c_outLast_outputGate_1);
	net.add_connection(c_cell_outputGate);

	net.add_layer(outputGate);

	net.add_connection(g_cell_outputGate_out);

	net.add_layer(outLayer);
	net.add_connection(c_out_loss);
	net.add_layer(lossLayer);

	/*********** Gradient check ***********/
	gradient_check(net, 1e-2, 1);

	/*********** Use hard-coded LSTM ***********/
	RecurrentNetwork lstmDebugNet;
	lstmDebugNet.set_input(input);
	lstmDebugNet.set_target(target);

	auto l0 = Layer::make<ConstantLayer>();
	auto lstmLayer = Layer::make<LstmDebugLayer>(LSTM_CONNECTION_WEIGHTS, LSTM_PREHISTORY);
	auto l1 = Layer::make<SquareLossLayer>();

	lstmDebugNet.add_layer(l0);
	lstmDebugNet.new_connection<ConstantConnection>(l0, lstmLayer);
	lstmDebugNet.add_layer(lstmLayer);
	lstmDebugNet.new_connection<ConstantConnection>(lstmLayer, l1);
	lstmDebugNet.add_layer(l1);

	lstmDebugNet.assemble();
	for (int i = 0; i < input.size(); ++i)
		lstmDebugNet.forward_prop();

	/*********** Output check against lstmDebugNet ***********/
	net.reset();
	for (int i = 0; i < input.size(); ++i)
		net.forward_prop();

	for (int i = 0; i < input.size(); ++i)
		EXPECT_NEAR(net.lossLayer->outValues[i],
				lstmDebugNet.lossLayer->outValues[i],
				1e-4) << "LSTM output doesn't agree with LstmDebugLayer";

/*
	for (ConnectionPtr c : fullConns)
		cout << std::setprecision(4) << Connection::cast<FullConnection>(c)->param << "  \n";
	cout << endl;

	net.reset();
	for (int i = 0; i < net.input.size(); ++i)
	{
		net.forward_prop();
//		DEBUG_MSG(net);
	}
	DEBUG_MSG("BACKWARD");
	for (int i = 0; i < net.input.size(); ++i)
	{
		net.backward_prop();
//		DEBUG_MSG(net);
	}

	for (ConnectionPtr c : fullConns)
		cout << std::setprecision(4) << Connection::cast<FullConnection>(c)->gradient << "  \n";
	cout << endl;

	for (LayerPtr l : { forgetGate, inputGate, cellHatLayer, cellLayer, outputGate, outLayer })
	{
		if (key_exists(net.prehistoryLayerMap, l))
		cout << std::setprecision(4) << static_cast<ParamContainerPtr>(net.prehistoryLayerMap[l])->paramGradients << "  ";
	}
	cout << endl;
*/
}

TEST(Composite, LSTM)
{
	// same fake params as RecurrentNet.LSTM test
	vector<float> LSTM_CONNECTION_WEIGHTS {
		-0.236, -0.648, -0.669, 0.76, -0.607, 0.323, -0.932, -0.737, 0.315, -0.109, 0.764
	};
	vector<float> LSTM_PREHISTORY {
		.3, -.47
	};
	rand_conn.set_rand_seq(LSTM_CONNECTION_WEIGHTS);
	rand_prehis.set_rand_seq(LSTM_PREHISTORY);
	vector<float> input {
		1.2, -0.9, 0.57, -1.47, 2.2
	};
	vector<float> target {
		1.39, 0.75, -0.45, -0.11, 1.9
	};

	auto inLayer = Layer::make<ConstantLayer>();

	auto lossLayer = Layer::make<SquareLossLayer>();

	RecurrentNetwork net;
	net.set_input(input);
	net.set_target(target);
	net.set_max_temporal_skip(1);

	net.add_layer(inLayer);

	auto lstmComposite = Composite<RecurrentNetwork>::make<LstmComposite>(inLayer);

	net.add_composite<RecurrentNetwork>(lstmComposite);

	net.new_connection<ConstantConnection>(lstmComposite->out_layer(), lossLayer);

	net.add_layer(lossLayer);

	gradient_check(net, 1e-2, 1);

}
