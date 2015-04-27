/*
 * Eona Studio (c) 2015
 */

#include "test_utils.h"
#include "../backend/vecmat/vecmat_engine.h"
#include "../backend/vecmat/vecmat_rand_dataman.h"
#include "../backend/vecmat/vecmat_func_dataman.h"

FakeRand& rand_conn = FakeRand::instance_connection();
FakeRand& rand_input = FakeRand::instance_input();
FakeRand& rand_target = FakeRand::instance_target();

TEST(VecmatForward, Diamond)
{
	const int INPUT_DIM = 3;
	const int TARGET_DIM = 2;
	const int BATCH_SIZE = 2;

	rand_conn.set_rand_seq(vector<float> {
		0.869, -0.764, -0.255, 0.771, -0.913, 0.294, -0.957, 0.958, -0.388, -0.184,
		0.922, 0.434, 0.217, 0.655, 0.707, 0.655, 0.368, -0.383, -0.838,
		0.638, -0.706, 0.429, -0.72, -0.439, 0.429, -0.977, 0.858, -0.937,
		0.381, -0.973, 0.764, -0.776, 0.907, 0.483, -0.573, -0.728, 0.587,
		0.102, -0.763, 0.939, 0.876, 0.195, 0.423, 0.0761, -0.364, 0.0478,
		0.558, 0.0241, -0.13, 0.591, -0.294, -0.762, 0.741, 0.0955, 0.784,
		0.398, 0.475, -0.199, -0.533, -0.483, -0.939, -0.344
	});
//	rand_conn.gen_uniform_rand(62, -1, 1); 	rand_conn.print_rand_seq();

	rand_input.set_rand_seq(vector<float> {
		0.276, 2.54, 2.27, 2.81, -0.0979, 0.205
	});
//	rand_input.gen_uniform_rand(INPUT_DIM * BATCH_SIZE, -1, 3); rand_input.print_rand_seq();

	rand_target.set_rand_seq(vector<float> {
		0.457, -0.516, -0.312, 0.126
	});
//	rand_target.gen_uniform_rand(TARGET_DIM * BATCH_SIZE, -1, 3); rand_target.print_rand_seq();

	auto engine = EngineBase::make<VecmatEngine>();
	auto dataman = DataManagerBase::make<VecmatRandDataManager>(
					engine, INPUT_DIM, TARGET_DIM, BATCH_SIZE);

	auto l1 = Layer::make<ConstantLayer>(INPUT_DIM);

	auto l2_1 = Layer::make<ScalarLayer>(1, 1.7f);
	auto l2_2 = Layer::make<CosineLayer>(3);
	auto l3_1 = Layer::make<SigmoidLayer>(2);
	auto l3_2 = Layer::make<ScalarLayer>(2, -2.3f);

	auto l4 = Layer::make<SquareLossLayer>(TARGET_DIM);

	ForwardNetwork net(engine, dataman);

	net.add_layer(l1);
	net.add_connection(Connection::make<FullConnection>(l1, l2_1));
	net.add_connection(Connection::make<FullConnection>(l1, l2_2));
	// same as add_connection(make_connection<>)
	net.new_connection<FullConnection>(l1, l3_1);
	net.new_connection<FullConnection>(l1, l3_2);
	net.new_connection<FullConnection>(l1, l4);
	net.add_layer(l2_1);
	net.add_layer(l2_2);
	net.new_connection<FullConnection>(l2_1, l3_1);
	net.new_connection<FullConnection>(l2_1, l3_2);
	net.new_connection<FullConnection>(l2_1, l4);
	net.new_connection<FullConnection>(l2_2, l3_2);
	net.new_connection<FullConnection>(l2_2, l3_1);
	net.new_connection<FullConnection>(l2_2, l4);
	net.add_layer(l3_1);
	net.add_layer(l3_2);
	net.new_connection<FullConnection>(l3_1, l4);
	net.new_connection<FullConnection>(l3_2, l4);
	net.add_layer(l4);

	gradient_check<VecmatEngine, VecmatRandDataManager>(net, 1e-2f, 0.8f);
}


TEST(VecmatForward, Bias)
{
	rand_conn.set_rand_seq(vector<float> {
		-0.565, -0.00362, -0.901, 0.516, -0.902, 0.299, 0.192, 0.441, 0.683, 0.893,
		0.718, 0.336, 0.706, 0.564, 0.257, -0.875, -0.163, -0.693, 0.531,
		0.787, -0.767, 0.0672, 0.997, 0.0552, -0.666, -0.601, -0.12, 0.998,
		-0.13, -0.441, 0.0686, -0.276, 0.857, -0.942, 0.641, 0.263, -0.839,
		0.888, -0.669, -0.261, -0.675, -0.271, 0.0758, 0.442, -0.951, 0.417,
		-0.895, 0.91, -0.241, -0.865, 0.302, -0.697, -0.886, 0.125, 0.14,
		-0.606, 0.982, 0.0487, 0.0931, 0.638, 0.524, 0.442, -0.803, 0.726,
		-0.943, -0.58, -0.982, 0.751, 0.235, -0.39, 0.974, -0.29, -0.738,
		-0.985, 0.00643, 0.0759, 0.0186, -0.718, -0.222, -0.0668, 0.853, -0.345,
		-0.315, -0.0346, -0.0573, -0.961, 0.123, 0.15, 0.292, 0.571, -0.501,
		0.088, -0.184, -0.445, -0.973, 0.571, -0.2, 0.0235, 0.434, -0.637,
		0.456, -0.155, 0.696, 0.567, -0.74, -0.857, 0.456, -0.333, -0.895, -0.831
	});
//	rand_conn.gen_uniform_rand(110, -1, 1); rand_conn.print_rand_seq();


	const int INPUT_DIM = 3;
	const int TARGET_DIM = 5;
	const int BATCH_SIZE = 4;

	rand_input.set_rand_seq(vector<float> {
		-0.327, 2.04, -0.884, -0.0188, 2.5, 1.25, 2.81, 1.55, -0.187, 2.67, 2.04, 1.1
	});
//	rand_input.gen_uniform_rand(INPUT_DIM * BATCH_SIZE, -1, 3); rand_input.print_rand_seq();

	rand_target.set_rand_seq(vector<float> {
		0.074, 1.62, -0.0258, -0.14, -0.604, 2.71, 2.37, -0.87, -0.514, 0.511,
		0.139, 2.75, 1.17, -0.224, 1.26, 0.0545, 1.8, 2.53, 1.34, 1.41
	});
//	rand_target.gen_uniform_rand(TARGET_DIM * BATCH_SIZE, -1, 3); rand_target.print_rand_seq();

	auto engine = EngineBase::make<VecmatEngine>();
	auto dataman = DataManagerBase::make<VecmatRandDataManager>(
					engine, INPUT_DIM, TARGET_DIM, BATCH_SIZE);

	auto l1 = Layer::make<ConstantLayer>(INPUT_DIM);

	auto l2_1 = Layer::make<ScalarLayer>(1, 1.3f);
	auto l2_2 = Layer::make<CosineLayer>(3);
	auto l3_1 = Layer::make<SigmoidLayer>(2);
	auto l3_2 = Layer::make<TanhLayer>(2);

	auto l4 = Layer::make<SquareLossLayer>(TARGET_DIM);

	ForwardNetwork net(engine, dataman);

	net.add_layer(l1);
	net.add_connection(Connection::make<FullConnection>(l1, l2_1));
	net.add_connection(Connection::make<FullConnection>(l1, l2_2));
	// same as add_connection(make_connection<>)
	net.new_connection<FullConnection>(l1, l3_1);
	net.new_connection<FullConnection>(l1, l3_2);
	net.new_connection<FullConnection>(l1, l4);
	net.new_bias_layer(l2_1);
	net.add_layer(l2_1);
	net.new_bias_layer(l2_2);
	net.add_layer(l2_2);
	net.new_connection<FullConnection>(l2_1, l3_1);
	net.new_connection<FullConnection>(l2_1, l3_2);
	net.new_connection<FullConnection>(l2_1, l4);
	net.new_connection<FullConnection>(l2_2, l3_2);
	net.new_connection<FullConnection>(l2_2, l3_1);
	net.new_connection<FullConnection>(l2_2, l4);
	net.new_bias_layer(l3_1);
	net.add_layer(l3_1);
	net.new_bias_layer(l3_2);
	net.add_layer(l3_2);
	net.new_connection<FullConnection>(l3_1, l4);
	net.new_connection<FullConnection>(l3_2, l4);
	net.add_layer(l4);

	gradient_check<VecmatEngine, VecmatRandDataManager>(net, 1e-2f, 1.5f);
}


TEST(VecmatForward, SoftmaxEntropyLoss)
{
	rand_conn.set_rand_seq(vector<float> {
		-0.603, -0.333, -0.411, -0.0651, -0.309, -0.534, -0.936, 0.798, 0.00378, -0.53,
		-0.816, -0.718, 0.103, 0.26, 0.284, -0.852, 0.304, 0.176, -0.61,
		-0.751, -0.373, 0.146, 0.756, 0.303, 0.425, 0.2, 0.05, 0.354,
		0.0952, -0.236, 0.846, -0.819, -0.785, 0.247, -0.95, 0.502, -0.506,
		0.728, -0.326, -0.968, -0.698, 0.405, -0.606, 0.884, 0.0578, 0.688,
		0.926, 0.805, 0.306, -0.526, -0.101, -0.444, -0.405, 0.0493, -0.641,
		0.758, -0.637, -0.528, -0.496, -0.401, -0.406, 0.465, -0.523, -0.564,
		-0.0582, -0.6, 0.266, 0.125, -0.928, 0.325, 0.86, -0.416, -0.686,
		0.416, -0.832, 0.506, 0.374, -0.732, -0.461, -0.167, -0.965, 0.77,
		-0.466, -0.794, -0.513, -0.4, -0.467, -0.934, 0.745, -0.59, -0.285,
		0.44, 0.152, -0.421, -0.078, 0.57, -0.787, 0.556, -0.922, -0.396,
		0.911, -0.484, -0.637, 0.14, -0.728, 0.134, 0.406, -0.293, 0.452, 0.886
	});
//	rand_conn.gen_uniform_rand(110, -1, 1); rand_conn.print_rand_seq();

	const int INPUT_DIM = 5;
	const int TARGET_DIM = 7;
	const int BATCH_SIZE = 4;

	// NOTE out is a label!!!
	auto learnableFunc = [](const lmn::Vecmatf& in, lmn::Vecmatf& out) {
		// Each column is a batch
		for (int c = 0; c < in.col(); ++c)
		{
			out(0, c) = rand() % TARGET_DIM;
		}
	};

	auto engine = EngineBase::make<VecmatEngine>();
	auto dataman = DataManagerBase::make<VecmatFuncDataManager>(
						engine, INPUT_DIM, 1, BATCH_SIZE,
						learnableFunc,
						10, 0, 0,
						-1.f, 1.f);

	auto l1 = Layer::make<ConstantLayer>(INPUT_DIM);

	auto l2 = Layer::make<SigmoidLayer>(5);
	auto l3 = Layer::make<TanhLayer>(5);

	auto l4 = Layer::make<LabelSoftmaxEntropyLayer>(TARGET_DIM);

	ForwardNetwork net(engine, dataman);

	net.add_layer(l1);
	net.new_connection<FullConnection>(l1, l2);
	net.new_bias_layer(l2);
	net.add_layer(l2);
	net.new_connection<FullConnection>(l2, l3);
	net.new_bias_layer(l3);
	net.add_layer(l3);
	net.new_connection<FullConnection>(l3, l4);
	net.add_layer(l4);

	gradient_check<VecmatEngine, VecmatFuncDataManager>(net, 1e-2f, 1.2f);
}

