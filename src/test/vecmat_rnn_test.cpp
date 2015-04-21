/*
 * Eona Studio (c) 2015
 */

#include "test_utils.h"

FakeRand& rand_conn = FakeRand::instance_connection();
FakeRand& rand_prehis = FakeRand::instance_prehistory();
FakeRand& rand_input = FakeRand::instance_input();
FakeRand& rand_target = FakeRand::instance_target();

TEST(VecmatRNN, Simple)
{
	rand_conn.set_rand_seq(vector<float> {
		1.15, 0.549, -0.751, 0.575, 1.41, -0.39, 1.1, -1.18, -0.84, 1.5,
		0.0886, -1.22, 0.575, 1.42, 0.89, 0.546, -0.857, -0.734, 0.972,
		0.294, -1.1, 0.739, 1.33, -0.638, -0.969, 0.454, 1.23, 0.228,
		-0.68, 1.04, -0.745, -0.17, 0.877, -0.778, -0.0865, 0.664, 0.513,
		0.105, 0.0317, 1.38, 0.741, -0.894, 0.595, 0.414, 1.03, 0.961,
		-1.13, 1.41, 0.752, -0.0889
	});
//	rand_conn.gen_uniform_rand(50, -1.5, 1.5);
//	rand_conn.print_rand_seq();

	rand_prehis.set_rand_seq(vector<float> {
		-0.0998, -0.0139, 0.109, 0.402, 0.417, -0.352, -0.069, -0.347, 0.329, -0.35,
		0.294, 0.22, -0.479, -0.165, 0.425, 0.206, 0.3, 0.0826, -0.13,
		-0.486, 0.339, -0.497, 0.317, 0.327, -0.229, -0.187, -0.00966, -0.338
	});
//	rand_prehis.gen_uniform_rand(30, -.5, .5);
//	rand_prehis.print_rand_seq();

	rand_input.set_rand_seq(vector<float> {
		0.397, 0.572, 0.99, -0.131, -0.45, -0.322, -0.115, 0.448, 0.705, -0.799,
		-0.583, -0.527, 0.206, 0.963, 0.0171, -0.661, -0.367, 0.967, -0.0181,
		0.144, -0.0365, 0.942, -0.0321, 0.106, -0.541, -0.701, 0.0964, 0.951, 0.213, -0.00518
	});
//	rand_input.gen_uniform_rand(30, -1, 1);
//	rand_input.print_rand_seq();

	rand_target.set_rand_seq(vector<float> {
		-0.998, -0.559, -0.00601, 0.953, 0.703, -0.456, 0.221, 0.488, 0.131, 0.48,
		0.701, 0.0961, -0.596, 0.701, 0.633, -0.707, 0.984, 0.33, 0.582, -0.411
	});
//	rand_target.gen_uniform_rand(20, -1, 1);
//	rand_target.print_rand_seq();

	const int HISTORY = 5;
	const int INPUT_DIM = 3;
	const int TARGET_DIM = 2;
	const int BATCH = 2;

	auto l1 = Layer::make<ConstantLayer>(INPUT_DIM);
	auto l2 = Layer::make<SigmoidLayer>(2);
	auto l3 = Layer::make<SigmoidLayer>(3);
	auto l4 = Layer::make<SquareLossLayer>(TARGET_DIM);

	// Naming: c<in><out>_<skip>
	auto c12 = Connection::make<FullConnection>(l1, l2);
	auto c23 = Connection::make<FullConnection>(l2, l3);
	auto c34 = Connection::make<FullConnection>(l3, l4);

	auto c22_1 = Connection::make<FullConnection>(l2, l2);
	auto c23_1 = Connection::make<FullConnection>(l2, l3);
	auto c33_1 = Connection::make<FullConnection>(l3, l3);

	auto engine = EngineBase::make<VecmatEngine>();
	auto dataman = DataManagerBase::make<VecmatDataManager>(
			engine, INPUT_DIM, TARGET_DIM, BATCH);

	RecurrentNetwork net(engine, dataman, HISTORY);

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

	gradient_check<VecmatEngine, VecmatDataManager, float>(net, 1e-2, 1);
}

TEST(VecmatRNN, TemporalSkip)
{
	rand_conn.set_rand_seq(vector<float> {
		-0.0246, 0.134, 0.396, 0.424, 0.441, 1.01, 0.381, 0.926, -0.391, -0.0287,
		1.39, 1.47, 1.4, -0.257, 0.294, 0.159, -1.18, 1.49, 0.342,
		-0.201, -0.759, 0.0732, 0.529, -0.133, 0.0745, 1.12, 1.01, 1.13,
		-1.36, 0.374, 0.745, -0.817, -0.46, 0.0725, 0.834, 0.507, 0.304,
		1.33, -1.43, 1.37, 0.781, -0.233, 0.0268, 0.27, 1.02, -1,
		1.02, -0.606, 0.931, 1.25, -1.34, -0.702, -0.583, -0.404, 0.414,
		-0.943, -0.0178, 0.796, 0.222, 0.879, -0.206, 1.25, 0.628, 1.19,
		-1.2, -0.246, -0.928, 0.705, 0.783, 0.836, 1.37, 0.882, 1.41,
		-1.11, -0.0715, -1.19, 1.49, -1.41, 1.18, 0.866, -0.929, 0.00423,
		-0.859, 0.351, 0.577, -1.25, -1.09, -1.39, 0.579, 1.39
	});
//	rand_conn.gen_uniform_rand(90, -1.5, 1.5);
//	rand_conn.print_rand_seq();

	rand_prehis.set_rand_seq(vector<float> {
		-0.467, 0.436, -0.0718, -0.247, 0.141, -0.0445, -0.255, 0.336, -0.256, -0.277,
		0.414, 0.418, -0.421, 0.196, 0.382, -0.00522, 0.203, -0.257, 0.0887,
		0.338, 0.31, -0.197, 0.433, 0.405, 0.157, -0.296, -0.304, -0.32, 0.328, -0.259
	});
//	rand_prehis.gen_uniform_rand(30, -.5, .5);
//	rand_prehis.print_rand_seq();

	rand_input.set_rand_seq(vector<float> {
		-0.557, -0.218, 0.567, -0.806, 0.86, -0.942, -0.678, 0.0253, -0.35, -0.749,
		-0.0705, -0.179, 0.674, 0.974, -0.0647, 0.717, -0.167, 0.585, -0.888, 0.874
	});
//	rand_input.gen_uniform_rand(20, -1, 1);
//	rand_input.print_rand_seq();

	rand_target.set_rand_seq(vector<float> {
		-0.323, 0.236, 0.143, 0.999, 0.469, -0.939, -0.232, -0.635, -0.105, 0.83,
		-0.892, 0.293, -0.786, 0.542, 0.224, 0.634, 0.515, 0.73, -0.293,
		-0.811, -0.891, -0.0717, -0.881, -0.24, -0.359, -0.401, 0.0343, -0.262,
		-0.963, -0.13, -0.282, -0.133, -0.728, 0.42, -0.046, -0.34, 0.536, 0.988, -0.282, 0.893
	});
//	rand_target.gen_uniform_rand(40, -1, 1);
//	rand_target.print_rand_seq();

	const int HISTORY = 5;
	const int INPUT_DIM = 2;
	const int TARGET_DIM = 4;
	const int BATCH = 2;

	auto l1 = Layer::make<ConstantLayer>(INPUT_DIM);
	auto l2 = Layer::make<SigmoidLayer>(3);
	auto l3 = Layer::make<CosineLayer>(2);
	auto l4 = Layer::make<SquareLossLayer>(TARGET_DIM);

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

	auto engine = EngineBase::make<VecmatEngine>();
	auto dataman = DataManagerBase::make<VecmatDataManager>(
			engine, INPUT_DIM, TARGET_DIM, BATCH);

	RecurrentNetwork net(engine, dataman, HISTORY);

	net.init_max_temporal_skip(3); // or Layer::UNLIMITED_TEMPORAL_SKIP

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

	gradient_check<VecmatEngine, VecmatDataManager, float>(net, 1e-2, 1);
}


TEST(VecmatRNN, GatedTanhConnection)
{
	rand_conn.set_rand_seq(vector<float> {
		1.09, -0.949, -1.2, -0.479, 0.0218, 1.01, 0.636, 1.31, -1.46, -1.08,
		-0.55, 0.479, -1.24, 0.0951, -0.414, 0.345, 0.49, -0.788, 0.767,
		0.762, -1.02, -0.282, 0.466, 0.408, -1.3, -0.859, -0.372, 0.46,
		-0.15, -1.21, 0.866, -0.948, 0.488, 0.476, -0.238, 0.165, -0.104,
		0.0809, 0.855, 1.29, -1.5, 0.883, 1.05, 1.32, 1.27, -0.0554,
		-0.315, 0.24, 0.273, 0.0537, -0.0614, 0.366, -1.16, -0.463, 1.46
	});
//	rand_conn.gen_uniform_rand(50, -1.5, 1.5);
//	rand_conn.print_rand_seq();

	rand_prehis.set_rand_seq(vector<float> {
		-0.132, -0.0704, 0.335, -0.272, -0.224, -0.325, -0.185, 0.456, 0.314, -0.263,
		0.477, -0.28, 0.194, -0.0456, 0.439, -0.499, -0.347, 0.162, 0.353
	});
//	rand_prehis.gen_uniform_rand(20, -.5, .5);
//	rand_prehis.print_rand_seq();

	rand_input.set_rand_seq(vector<float> {
		0.55, -0.489, -0.678, -0.946, 0.952, 0.84, 0.704, 0.596, -0.698, 0.987,
		-0.071, 0.509, -0.14, 0.201, 0.143, 0.981, 0.297, 0.196, -0.0144,
		0.0804, 0.871, -0.951, -0.468, -0.235, -0.718, 0.536, 0.392, -0.476,
		0.249, 0.794, -0.812, 0.634, 0.127, 0.113, -0.335, 0.786, 0.621,
		0.316, -0.509, -0.906, 0.632, 0.996, -0.827, 0.173, 0.445, 0.252, 0.545, 0.688, -0.171, -0.809
	});
//	rand_input.gen_uniform_rand(50, -1, 1);
//	rand_input.print_rand_seq();

	rand_target.set_rand_seq(vector<float> {
		-0.927, -0.659, -0.866, 0.793, 0.0821, 0.496, 0.25, -0.175, -0.642, -0.0911,
		-0.569, 0.0953, 0.0641, 0.858, 0.204, -0.699, 0.31, 0.266, -0.284,
		-0.329, -0.25, 0.357, 0.214, 0.454, 0.614, -0.214, -0.377, 0.925,
		-0.146, 0.899, 0.592, 0.694, 0.213, -0.301, -0.259, -0.389, 0.959,
		-0.104, 0.986, -0.32, -0.605, -0.261, 0.989, -0.231, -0.447, 0.07,
		-0.152, -0.339, 0.855, -0.128, 0.622, -0.736, 0.255, -0.424, 0.335, -0.634, -0.579, 0.606, 0.998, -0.361
	});
//	rand_target.gen_uniform_rand(60, -1, 1);
//	rand_target.print_rand_seq();

	const int HISTORY = 5;
	const int INPUT_DIM = 3;
	const int TARGET_DIM = 4;
	const int BATCH = 3;

	auto l1 = Layer::make<ConstantLayer>(INPUT_DIM);

	// NOTE layers engaged in a gate must have the same dims
	auto l2 = Layer::make<ScalorLayer>(TARGET_DIM, 1.3f);
	auto l3 = Layer::make<CosineLayer>(TARGET_DIM); // gate

	auto l4 = Layer::make<SquareLossLayer>(TARGET_DIM);

	auto g234 = Connection::make<GatedTanhConnection>(l2, l3, l4);
	auto g234_1 = Connection::make<GatedTanhConnection>(l2, l3, l4);
	auto g234_2 = Connection::make<GatedTanhConnection>(l2, l3, l4);

	auto engine = EngineBase::make<VecmatEngine>();
	auto dataman = DataManagerBase::make<VecmatDataManager>(
			engine, INPUT_DIM, TARGET_DIM, BATCH);

	RecurrentNetwork net(engine, dataman, HISTORY, 2);

	net.add_layer(l1);
	net.new_connection<FullConnection>(l1, l2);
	net.add_layer(l2);
	net.new_connection<FullConnection>(l1, l3);
	net.add_layer(l3);
	net.add_connection(g234);
	net.add_recurrent_connection(g234_1);
	net.add_recurrent_connection(g234_2, 2);
	net.add_layer(l4);

	gradient_check<VecmatEngine, VecmatDataManager>(net, 1e-2f, 1.3f);
}
