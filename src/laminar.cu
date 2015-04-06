/*
 * Eona Studio (c)2015
 */
#include "global_utils.h"
#include "timer.h"
#include "connection.h"
#include "transfer_layer.h"
#include "loss_layer.h"
#include "parameter.h"
#include "lstm_layer.h"
#include "network.h"
#include "gradient_check.h"

int main(int argc, char **argv)
{
	FakeRand::instance_connection().set_rand_seq(vector<float> {
		0.163, 1.96, 1.09, 0.516, -0.585, 0.776, 1, -0.301, -0.167, 0.732
	});

	FakeRand::instance_connection().use_fake_seq();
	FakeRand::instance_connection().use_uniform_rand(-1, 2);
	FakeRand::instance_connection().set_rand_display(true);

	FakeRand::instance_prehistory().set_rand_seq(vector<float> {
		.3
	});

	vector<float> input { 1.2, -0.9, 0.57, -1.47, -3.08 }; //, 1.2, .31, -2.33, -0.89 };
	vector<float> target { 1.39, 0.75, -0.45, -0.11, 1.55}; //, -.44, 2.39, 1.72, -3.06 };

	auto l1 = Layer::make<LinearLayer>();

	auto forgetGate = Layer::make<SigmoidLayer>();


	auto l2 = Layer::make<SigmoidLayer>();
	auto l3 = Layer::make<TanhLayer>(); // gate

	auto l4 = Layer::make<SquareLossLayer>();

	// NOTE IMPORTANT RULE
	// For recurrent gated connection conn[layer(alpha), layer(gate) => layer(beta)]
	// where alpha is t-1 (or more) and gate/beta are the current t
	// Must be added after you add gate & alpha, and before beta.

	// Naming: c<in><out>_<skip>
	// g<in><gate><out>_<skip>
	auto c12 = Connection::make<LinearConnection>(l1, l2);
	auto c13 = Connection::make<LinearConnection>(l1, l3);

	auto g234_1 = Connection::make<GatedConnection>(l2, l3, l4);
	auto g234_2 = Connection::make<GatedConnection>(l2, l3, l4);
//
//	auto c22_1 = Connection::make<LinearConnection>(l2, l2);
//	auto c22_3 = Connection::make<LinearConnection>(l2, l2);
//	auto c23_1 = Connection::make<LinearConnection>(l2, l3);
//	auto c23_2 = Connection::make<LinearConnection>(l2, l3);
//	auto c32_3 = Connection::make<LinearConnection>(l3, l2);
//	auto c33_1 = Connection::make<LinearConnection>(l3, l3);
//	auto c33_2 = Connection::make<LinearConnection>(l3, l3);

	RecurrentNetwork net;
	net.set_input(input);
	net.set_target(target);
//	net.set_max_temporal_skip(3);

	net.add_layer(l1);

	net.add_connection(c13);
	net.add_layer(l3);

	net.add_connection(c12);
	net.add_layer(l2);

	net.new_connection<LinearConnection>(l2, l4);
	net.new_connection<LinearConnection>(l3, l4);

//	net.add_recurrent_connection(g234_1);
//	net.add_recurrent_connection(g234_2, 2);
	net.add_layer(l4);

	gradient_check(net, 1e-2, 1);
/*
	net.reset();
	for (int i = 0; i < net.input.size(); ++i)
	{
		net.forward_prop();
		DEBUG_MSG(net);
	}
	DEBUG_MSG("BACKWARD");
	for (int i = 0; i < net.input.size(); ++i)
	{
		net.backward_prop();
		DEBUG_MSG(net);
	}

	for (ConnectionPtr c : { c12, c13 })
		cout << std::setprecision(4) << Connection::cast<LinearConnection>(c)->gradient << "  ";
	cout << endl;

	for (LayerPtr l : { l2, l3 })
	{
		if (key_exists(net.prehistoryLayerMap, l))
		cout << std::setprecision(4) << static_cast<ParamContainerPtr>(net.prehistoryLayerMap[l])->paramGradients << "  ";
	}
	cout << endl;*/
}
