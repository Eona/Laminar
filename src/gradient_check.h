/*
 * Eona Studio (c) 2015
 */

#ifndef GRADIENT_CHECK_H_
#define GRADIENT_CHECK_H_

#include "network.h"
#include "engine/dummy_engine.h"

/**
 * % difference between analytic (backprop)
 * and numeric (finite-difference) gradients
 */
inline void gradient_check(Network& net,
		float perturb = 1e-2f, float percentTol = 1.0f)
{
	auto engine = net.get_engine<DummyEngine>();

	net.upload("initialize");
	net.upload("forward");
	net.upload("backward");
	net.compile();

	net.execute("initialize");
	net.execute("forward");
	net.execute("backward");

	vector<Tensor> analyticGrads;
	for (ParamContainer::Ptr param : net.paramContainers)
	{
		auto gradients = param->paramGradients;
		for (auto gradPtr : gradients)
			analyticGrads.push_back(*gradPtr);
	}
	engine->flush_execute();
	for (auto& tensor : analyticGrads)
		DEBUG_MSG("gradient check: " << engine->read_memory(tensor));

	/****** perturb parameters matrices stored in connections ******/
	int agpt = 0; // point to analyticGrads
	for (ParamContainer::Ptr param : net.paramContainers)
	{
		for (int p = 0; p < param->size(); ++p)
		{
			net.zero_clear(); // refresh network
			net.input[0]->upload(Instruction("debug_fill", {}, net.input[0]->addr));
			net.target[0]->upload(Instruction("debug_fill", {}, net.target[0]->addr));
			param->gradient_check_perturb(p, -perturb);
			engine->flush_execute();

			net.execute("forward");

			float lossMinus = engine->read_memory(net.lossLayer->total_loss());

			param->gradient_check_restore();

			net.zero_clear(); // refresh network
			net.input[0]->upload(Instruction("debug_fill", {}, net.input[0]->addr));
			net.target[0]->upload(Instruction("debug_fill", {}, net.target[0]->addr));

			param->gradient_check_perturb(p, +perturb);
			engine->flush_execute();

			net.execute("forward");

			float lossPlus = engine->read_memory(net.lossLayer->total_loss());

			param->gradient_check_restore();
			engine->flush_execute();

			float numericGrad = (lossPlus - lossMinus) / (2.0 * perturb);

			float analyticGrad = engine->read_memory(analyticGrads[agpt++]);

			assert_float_percent_eq(analyticGrad, numericGrad, percentTol,
					"param analytic != numeric", "param gradcheck pass");
		}
	}

	/****** perturb the input ******/
	// TODO
	// Save the full gradient history for debugging ONLY
	/*try {
		RecurrentNetwork& _net = dynamic_cast<RecurrentNetwork&>(net);
		_net.set_max_temporal_skip(Layer::UNLIMITED_TEMPORAL_SKIP);
	}
	catch (std::bad_cast& err) { }

	vector<float> oldInput = net.input; // for restoration

	net.reset();
	for (int i = 0; i < timeLength; ++i)
		net.forward_prop();
	for (int i = 0; i < timeLength; ++i)
		net.backward_prop();

	vector<float> analyticGrad = net.layers[0]->inGradients;
	vector<float> numericGrad(net.input.size());

	// perturb each input in sequence
	for (int inp = 0; inp < timeLength; ++inp)
	{
		float restoreInputVal = oldInput[inp];

		oldInput[inp] = restoreInputVal - perturb;
		net.set_input(oldInput);
		net.reset();
		for (int i = 0; i < timeLength; ++i)
			net.forward_prop();
		float lossMinus = net.lossLayer->totalLoss;

		oldInput[inp] = restoreInputVal + perturb;
		net.set_input(oldInput);
		net.reset();
		for (int i = 0; i < timeLength; ++i)
			net.forward_prop();
		float lossPlus = net.lossLayer->totalLoss;

		float numericGrad = (lossPlus - lossMinus) / (2.0 * perturb);

		assert_float_percent_eq(analyticGrad[inp], numericGrad, percentTol,
				"input analytic != numeric", "input gradcheck pass");

		oldInput[inp] = restoreInputVal; // restore
	}
	net.set_input(oldInput);*/
}

#endif /* GRADIENT_CHECK_H_ */
