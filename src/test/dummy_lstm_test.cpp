/*
 * Eona Studio (c) 2015
 */

#include "test_utils.h"

FakeRand& rand_conn = FakeRand::instance_connection();
FakeRand& rand_prehis = FakeRand::instance_prehistory();
FakeRand& rand_input = FakeRand::instance_input();
FakeRand& rand_target = FakeRand::instance_target();


TEST(DummyLSTM, LSTM)
{
	/********** FAKE_RAND **********/
	rand_conn.set_rand_seq(vector<float>{
		-0.904, 0.312, -0.944, 1.34, -2.14, -1.69, -2.88, -0.889, -2.28, -0.414, -2.07
	});

	rand_prehis.set_rand_seq(vector<float>{
		// should be same, otherwise initialization order might change
		.3, .3
	});

	vector<float> inputSeq {
		1.2, -0.9, 0.57, -1.47, -3.08, 1.2, .31, -2.33, -0.89
	};
	vector<float> targetSeq {
		1.39, 0.75, -0.45, -0.11, 1.55, -.44, 2.39, 1.72, -3.06
	};

	rand_input.set_rand_seq(inputSeq);
	rand_target.set_rand_seq(targetSeq);

	/********** LSTM layers **********/
	auto inLayer = Layer::make<ConstantLayer>(DUMMY_DIM);

	auto forgetGate = Layer::make<SigmoidLayer>(DUMMY_DIM);
	auto inputGate = Layer::make<SigmoidLayer>(DUMMY_DIM);
	auto cellhatLayer = Layer::make<TanhLayer>(DUMMY_DIM);
	auto cellLayer = Layer::make<ConstantLayer>(DUMMY_DIM);
	auto outputGate = Layer::make<SigmoidLayer>(DUMMY_DIM);
	auto outLayer = Layer::make<ConstantLayer>(DUMMY_DIM);

	auto lossLayer = Layer::make<SquareLossLayer>(DUMMY_DIM);

	/********** LSTM connections **********/
	// Naming: c<in><out>_<skip>, or gated: g<in><gate><out>_<skip>
	auto c_in_inputGate = conn_full(inLayer, inputGate);
	auto c_outLast_inputGate_1 = conn_full(outLayer, inputGate);
	auto c_cellLast_inputGate_1 = conn_full(cellLayer, inputGate);

	auto c_in_forgetGate = conn_full(inLayer, forgetGate);
	auto c_outLast_forgetGate_1 = conn_full(outLayer, forgetGate);
	auto c_cellLast_forgetGate_1 = conn_full(cellLayer, forgetGate);

	auto c_in_cellhat = conn_full(inLayer, cellhatLayer);
	auto c_outLast_cellhat_1 = conn_full(outLayer, cellhatLayer);

	auto g_cellhat_inputGate_cell = conn_gated(cellhatLayer, inputGate, cellLayer);
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
		c_in_cellhat,
		c_outLast_cellhat_1,
		c_in_outputGate,
		c_outLast_outputGate_1,
		c_cell_outputGate
	};

	/********** Construct the network **********/
	// order of topology see "NOTE" sections in simple RecurrentNet gtests.
	auto dummyEng = EngineBase::make<DummyEngine>();
	auto dummyData = DataManagerBase::make<DummyDataManager>(dummyEng);

	RecurrentNetwork net(dummyEng, dummyData, inputSeq.size(), 1);

	net.add_layer(inLayer);

	net.add_connection(c_in_inputGate);
	net.add_recur_connection(c_outLast_inputGate_1);
	net.add_recur_connection(c_cellLast_inputGate_1);
	net.new_bias_layer(inputGate);
	net.add_layer(inputGate);

	net.add_connection(c_in_forgetGate);
	net.add_recur_connection(c_outLast_forgetGate_1);
	net.add_recur_connection(c_cellLast_forgetGate_1);
	net.new_bias_layer(forgetGate);
	net.add_layer(forgetGate);

	net.add_connection(c_in_cellhat);
	net.add_recur_connection(c_outLast_cellhat_1);

	net.new_bias_layer(cellhatLayer);
	net.add_layer(cellhatLayer);

	net.add_connection(g_cellhat_inputGate_cell);
	net.add_recur_connection(g_cellLast_forgetGate_cell_1);

	net.add_layer(cellLayer);

	net.add_connection(c_in_outputGate);
	net.add_recur_connection(c_outLast_outputGate_1);
	net.add_connection(c_cell_outputGate);

	net.new_bias_layer(outputGate);
	net.add_layer(outputGate);

	net.add_connection(g_cell_outputGate_out);

	net.add_layer(outLayer);
	net.add_connection(c_out_loss);
	net.add_layer(lossLayer);

	/********** Gradient check **********/
	gradient_check<DummyEngine, DummyDataManager>(net, 1e-2f, 1.f);

	/********** Use hard-coded LSTM **********/
	auto dummyEng2 = EngineBase::make<DummyEngine>();
	auto dummyData2 = DataManagerBase::make<DummyDataManager>(dummyEng2);
	dummyData2->start_new_epoch();

	RecurrentNetwork lstmDebugNet(dummyEng2, dummyData2, inputSeq.size(), 1);

	auto l0 = Layer::make<ConstantLayer>(DUMMY_DIM);
	auto lstmLayer = Layer::make<LstmDebugLayer>(
				// LSTM dim, inLayeyDim, batchSize, all set to 1
				DUMMY_DIM, DUMMY_DIM, DUMMY_DIM);
	auto l1 = Layer::make<SquareLossLayer>(DUMMY_DIM);

	lstmDebugNet.add_layer(l0);
	lstmDebugNet.new_connection<ConstantConnection>(l0, lstmLayer);
	lstmDebugNet.add_layer(lstmLayer);
	lstmDebugNet.new_connection<ConstantConnection>(lstmLayer, l1);
	lstmDebugNet.add_layer(l1);

	// Manually reset
	rand_conn.reset_seq();
	rand_prehis.reset_seq();
	lstmDebugNet.execute("initialize");
	lstmDebugNet.execute("load_input"); lstmDebugNet.execute("load_target");
	lstmDebugNet.execute("forward");

	/********* Output check against lstmDebugNet *********/
	net.execute("forward");

	vector<float> netOutput;
	vector<float> lstmDebugOutput;
	for (int t = 0; t < net.history_length(); ++t)
	{
		// lossLayer only propagates to inValue, outValue is left blank
		netOutput.push_back(*dummyEng->read_memory(net.lossLayer->in_value(t)));
		lstmDebugOutput.push_back(*dummyEng2->read_memory(lstmDebugNet.lossLayer->in_value(t)));
	}

	cout << "Net output: " << netOutput << endl;
	cout << "LSTM debug output: " << lstmDebugOutput << endl;

	for (int t = 0; t < net.history_length(); ++t)
		EXPECT_NEAR(netOutput[t], lstmDebugOutput[t], 1e-6)
			<< "LSTM output doesn't agree with LstmDebugLayer";
}


TEST(DummyLSTM, Composite)
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

	vector<float> inputSeq {
		1.2, -0.9, 0.57, -1.47, 2.2
	};
	vector<float> targetSeq {
		1.39, 0.75, -0.45, -0.11, 1.9
	};

	rand_input.set_rand_seq(inputSeq);
	rand_target.set_rand_seq(targetSeq);

	auto inLayer = Layer::make<ConstantLayer>(DUMMY_DIM);
	auto lossLayer = Layer::make<SquareLossLayer>(DUMMY_DIM);

	auto dummyEng = EngineBase::make<DummyEngine>();
	auto dummyData = DataManagerBase::make<DummyDataManager>(dummyEng);

	RecurrentNetwork net(dummyEng, dummyData, inputSeq.size(), 1);

	net.add_layer(inLayer);

	auto lstmComposite = Composite<RecurrentNetwork>::make<LstmComposite>(inLayer, DUMMY_DIM);
	// or you can add the object directly:
	// auto lstmCompositeObject = Composite<RecurrentNetwork>::create<LstmComposite>(inLayer);

	net.add_composite(lstmComposite);

	net.new_connection<ConstantConnection>(lstmComposite->out_layer(), lossLayer);

	net.add_layer(lossLayer);

	gradient_check<DummyEngine, DummyDataManager>(net, 1e-2f, 1.f);
}
