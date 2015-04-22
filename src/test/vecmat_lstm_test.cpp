/*
 * Eona Studio (c) 2015
 */

#include "test_utils.h"

FakeRand& rand_conn = FakeRand::instance_connection();
FakeRand& rand_prehis = FakeRand::instance_prehistory();
FakeRand& rand_input = FakeRand::instance_input();
FakeRand& rand_target = FakeRand::instance_target();


TEST(VecmatLSTM, Composite)
{
	rand_conn.set_rand_seq(vector<float> {
		1.01, -0.823, 1.36, 0.0479, 0.849, -0.0118, -0.828, -0.605, -1.12, -0.165,
		-0.706, -1.49, 0.0703, -0.398, -0.901, 0.0867, -1.5, 0.813, -1.34,
		-0.525, 1.03, -0.345, 0.14, -1.43, 1.16, 0.0451, 1.47, 0.493,
		-0.928, 1.41, 1.38, 0.116, -1.23, -0.709, 0.295, 1.05, -0.722,
		0.244, 0.921, -0.872, 0.809, 0.663, -0.523, 0.803, -1.43, 0.0162,
		-0.144, 0.736, -0.62, 1.32, 1.33, -0.516, 0.652, 0.0834, 0.838,
		1.32, -1.34, -1.1, -1.43, -1.45, 0.866, 0.47, -0.0979, 0.889,
		0.511, -0.959, 0.937, 0.19, -0.993, -0.374, -0.0723, -0.206, -0.00643,
		-0.0353, -1.48, 1.22, 1.1, -0.908, -0.531, 1.31, 0.536, 0.998,
		1.19, -0.67, 0.0238, 0.461, -1.14, 0.242, -0.912, -1.46
	});
//	rand_conn.gen_uniform_rand(90, -1.5, 1.5);
//	rand_conn.print_rand_seq();

	rand_prehis.set_rand_seq(vector<float> {
		-0.211, 0.445, -0.13, 0.13, -0.331, 0.184, -0.288, -0.13, -0.276, 0.353,
		-0.154, -0.00805, -0.239, -0.126, -0.343, -0.37, 0.346, -0.496, -0.375,
		0.402, 0.336, 0.352, -0.333, 0.205, 0.297, -0.0315, 0.325, 0.231, -0.491, -0.315
	});
//	rand_prehis.gen_uniform_rand(30, -.5, .5);
//	rand_prehis.print_rand_seq();

	rand_input.set_rand_seq(vector<float> {
		-0.671, -0.0565, -0.0937, -0.862, -0.651, 0.057, 0.121, -0.141, 0.0475, -0.0606,
		-0.583, -0.252, -0.43, -0.782, 0.487, 0.372, -0.47, -0.515, -0.795, -0.732
	});
//	rand_input.gen_uniform_rand(20, -1, 1);
//	rand_input.print_rand_seq();

	rand_target.set_rand_seq(vector<float> {
		0.132, 0.666, 0.368, -0.936, -0.00213, 0.214, 0.0675, -0.89, -0.675, -0.546,
		-0.216, -0.861, -0.158, 0.96, -0.552, -0.827, 0.818, -0.961, -0.765,
		0.545, 0.405, 0.932, 0.313, 0.356, -0.792, -0.0062, -0.19, 0.915,
		-0.307, 0.509, 0.607, -0.244, -0.568, 0.501, 0.549, -0.136, -0.86, -0.128, 0.172, 0.346
	});
//	rand_target.gen_uniform_rand(40, -1, 1);
//	rand_target.print_rand_seq();

	const int HISTORY = 5;
	const int INPUT_DIM = 2;
	const int TARGET_DIM = 4;
	const int BATCH = 2;

	auto inLayer = Layer::make<ConstantLayer>(INPUT_DIM);
	auto lossLayer = Layer::make<SquareLossLayer>(TARGET_DIM);

	auto engine = EngineBase::make<VecmatEngine>();
	auto dataman = DataManagerBase::make<VecmatRandDataManager>(
			engine, INPUT_DIM, TARGET_DIM, BATCH);

	RecurrentNetwork net(engine, dataman, HISTORY);

	net.add_layer(inLayer);

	auto lstmComposite =
			Composite<RecurrentNetwork>::make<LstmComposite>(inLayer, 3);
	// or you can add the object directly:
	// auto lstmCompositeObject =
	//		Composite<RecurrentNetwork>::create<LstmComposite>(inLayer, 7);

	net.add_composite(lstmComposite);

	net.new_connection<FullConnection>(lstmComposite->out_layer(), lossLayer);

	net.add_layer(lossLayer);

	gradient_check<VecmatEngine, VecmatRandDataManager>(net, 1e-2f, 1.f);
}

/**
 * Agreement with hand-coded LstmDebugLayer
 */
TEST(VecmatLSTM, Agreement)
{
	rand_conn.set_rand_seq(vector<float> {
		1.01, -0.823, 1.36, 0.0479, 0.849, -0.0118, -0.828, -0.605, -1.12, -0.165,
		-0.706, -1.49, 0.0703, -0.398, -0.901, 0.0867, -1.5, 0.813, -1.34,
		-0.525, 1.03, -0.345, 0.14, -1.43, 1.16, 0.0451, 1.47, 0.493,
		-0.928, 1.41, 1.38, 0.116, -1.23, -0.709, 0.295, 1.05, -0.722,
		0.244, 0.921, -0.872, 0.809, 0.663, -0.523, 0.803, -1.43, 0.0162,
		-0.144, 0.736, -0.62, 1.32, 1.33, -0.516, 0.652, 0.0834, 0.838,
		1.32, -1.34, -1.1, -1.43, -1.45, 0.866, 0.47, -0.0979, 0.889,
		0.511, -0.959, 0.937, 0.19, -0.993, -0.374, -0.0723, -0.206, -0.00643,
		-0.0353, -1.48, 1.22, 1.1, -0.908, -0.531, 1.31, 0.536, 0.998,
		1.19, -0.67, 0.0238, 0.461, -1.14, 0.242, -0.912, -1.46
	});
//	rand_conn.gen_uniform_rand(90, -1.5, 1.5);
//	rand_conn.print_rand_seq();

	rand_prehis.set_rand_seq(vector<float> {
		-0.211, 0.445, -0.13, 0.13, -0.331, 0.184, -0.288, -0.13, -0.276, 0.353,
		-0.154, -0.00805, -0.239, -0.126, -0.343, -0.37, 0.346, -0.496, -0.375,
		0.402, 0.336, 0.352, -0.333, 0.205, 0.297, -0.0315, 0.325, 0.231, -0.491, -0.315
	});
//	rand_prehis.gen_uniform_rand(30, -.5, .5);
//	rand_prehis.print_rand_seq();

	rand_input.set_rand_seq(vector<float> {
		-0.671, -0.0565, -0.0937, -0.862, -0.651, 0.057, 0.121, -0.141, 0.0475, -0.0606,
		-0.583, -0.252, -0.43, -0.782, 0.487, 0.372, -0.47, -0.515, -0.795, -0.732
	});
//	rand_input.gen_uniform_rand(20, -1, 1);
//	rand_input.print_rand_seq();

	rand_target.set_rand_seq(vector<float> {
		0.132, 0.666, 0.368, -0.936, -0.00213, 0.214, 0.0675, -0.89, -0.675, -0.546,
		-0.216, -0.861, -0.158, 0.96, -0.552, -0.827, 0.818, -0.961, -0.765,
		0.545, 0.405, 0.932, 0.313, 0.356, -0.792, -0.0062, -0.19, 0.915,
		-0.307, 0.509, 0.607, -0.244, -0.568, 0.501, 0.549, -0.136, -0.86, -0.128, 0.172, 0.346
	});
//	rand_target.gen_uniform_rand(40, -1, 1);
//	rand_target.print_rand_seq();

	const int HISTORY = 5;
	const int INPUT_DIM = 2;
	const int LSTM_DIM = 3;
	const int TARGET_DIM = 4;
	const int BATCH = 2;

	auto inLayer = Layer::make<ConstantLayer>(INPUT_DIM);
	auto lossLayer = Layer::make<SquareLossLayer>(TARGET_DIM);

	auto engine = EngineBase::make<VecmatEngine>();
	auto dataman = DataManagerBase::make<VecmatRandDataManager>(
			engine, INPUT_DIM, TARGET_DIM, BATCH);

	RecurrentNetwork net(engine, dataman, HISTORY);

	net.add_layer(inLayer);

	auto lstmComposite =
			Composite<RecurrentNetwork>::make<LstmComposite>(inLayer, LSTM_DIM);

	net.add_composite(lstmComposite);
	net.new_connection<FullConnection>(lstmComposite->out_layer(), lossLayer);
	net.add_layer(lossLayer);

	net.execute("initialize");
	net.execute("load_input"); net.execute("load_target");
	net.execute("forward");

	/*********** hand-coded debug layer ***********/
	auto engine2 = EngineBase::make<VecmatEngine>();
	auto dataman2 = DataManagerBase::make<VecmatRandDataManager>(
			engine2, INPUT_DIM, TARGET_DIM, BATCH);

	dataman2->start_new_epoch();

	RecurrentNetwork lstmDebugNet(engine2, dataman2, HISTORY, 1);

	// suffix D for debugging
	auto inLayerD = Layer::make<ConstantLayer>(INPUT_DIM);
	auto lstmLayerD = Layer::make<LstmDebugLayer>(LSTM_DIM, INPUT_DIM, BATCH);
	auto lossLayerD = Layer::make<SquareLossLayer>(TARGET_DIM);

	lstmDebugNet.add_layer(inLayerD);
	lstmDebugNet.new_connection<ConstantConnection>(inLayerD, lstmLayerD);
	lstmDebugNet.add_layer(lstmLayerD);
	lstmDebugNet.new_connection<FullConnection>(lstmLayerD, lossLayerD);
	lstmDebugNet.add_layer(lossLayerD);

	// reset rand weights to fill the parameters exactly like LstmComposite network
	rand_conn.reset_seq();
	rand_prehis.reset_seq();
	lstmDebugNet.execute("initialize");
	lstmDebugNet.execute("load_input"); lstmDebugNet.execute("load_target");
	lstmDebugNet.execute("forward");

	/********* Output check against lstmDebugNet *********/
	vector<lmn::Vecmatf> netOutput;
	vector<lmn::Vecmatf> lstmDebugOutput;
	for (int t = 0; t < net.history_length(); ++t)
	{
		// lossLayer only propagates to inValue, outValue is left blank
		netOutput.push_back(*engine->read_memory(net.lossLayer->in_value(t)));
		lstmDebugOutput.push_back(*engine2->read_memory(lstmDebugNet.lossLayer->in_value(t)));
	}

	cout << "Net output: " << netOutput << "\n\n";
	cout << "LSTM debug output: " << lstmDebugOutput << endl;
//
	for (int t = 0; t < HISTORY; ++t)
		// Vecmat comparison by default 1e-6 tolerance
		ASSERT_EQ(netOutput[t], lstmDebugOutput[t]) <<
				"LSTM output doesn't agree with LstmDebugLayer:\n"
				<< netOutput[t] << "\n<->\n" << lstmDebugOutput[t];
}
