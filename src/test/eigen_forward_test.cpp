/*
 * Eona Studio (c) 2015
 */

#include "eigen_helper.h"

TEST(EigenForward, Simple)
{
	const int INPUT_DIM = 3;
	const int TARGET_DIM = 4;
	const int BATCH_SIZE = 2;

	auto engine = EngineBase::make<EigenEngine>();

	auto dataman = DataManagerBase::make<EigenRandDataManager>(
					engine, INPUT_DIM, TARGET_DIM, BATCH_SIZE);

	auto l1 = Layer::make<ConstantLayer>(INPUT_DIM);
	auto l2 = Layer::make<SigmoidLayer>(7);
	auto l3 = Layer::make<TanhLayer>(4);
	auto l4 = Layer::make<SquareLossLayer>(TARGET_DIM);

	ForwardNetwork net(engine, dataman);

	net.add_layer(l1);
	net.new_bias_layer(l2);
	net.new_connection<FullConnection>(l1, l2);
	net.add_layer(l2);
	net.new_bias_layer(l3);
	net.new_connection<FullConnection>(l2, l3);
	net.add_layer(l3);
	net.new_bias_layer(l4);
	net.new_connection<FullConnection>(l3, l4);
	net.add_layer(l4);

	gradient_check<EigenEngine, EigenRandDataManager>(net, 1e-2f, 0.8f);
}

TEST(EigenForward, Softmax)
{
	const int INPUT_DIM = 3;
	const int TARGET_DIM = 4;
	const int BATCH_SIZE = 2;

	auto engine = EngineBase::make<EigenEngine>();

	auto dataman = DataManagerBase::make<EigenRandDataManager>(
					engine, INPUT_DIM, 1, BATCH_SIZE,
					// number of target label classes (classification task)
					TARGET_DIM);

	auto l1 = Layer::make<ConstantLayer>(INPUT_DIM);

	auto l2 = Layer::make<SigmoidLayer>(4);

	auto l3 = Layer::make<TanhLayer>(5);

	auto l4 = Layer::make<LabelSoftmaxEntropyLayer>(TARGET_DIM);

	ForwardNetwork net(engine, dataman);

	net.add_layer(l1);
	net.new_connection<FullConnection>(l1, l2);
	net.add_layer(l2);
	net.new_connection<FullConnection>(l2, l3);
	net.add_layer(l3);
	net.new_connection<FullConnection>(l3, l4);
	net.add_layer(l4);

	gradient_check<EigenEngine, EigenRandDataManager>(net, 1e-2f, 0.8f);
}
