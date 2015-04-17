/*
 * Eona Studio (c) 2015
 */

#ifndef GRADIENT_CHECK_H_
#define GRADIENT_CHECK_H_

#include "network.h"
#include "engine/dummy_engine.h"
#include "engine/dummy_data.h"

/**
 * % difference between analytic (backprop)
 * and numeric (finite-difference) gradients
 */
inline void gradient_check(Network& net,
		float perturb = 1e-2f, float percentTol = 1.0f)
{
	auto engine = net.get_engine<DummyEngine>();
	auto dataman = net.get_data_manager<DummyDataManager>();

	net.upload("initialize");
	net.upload("forward");
	net.upload("backward");
	net.compile();

	net.execute("initialize");
	net.execute("forward");
	net.execute("backward");

	// helper
	auto reset_net = [&]()
	{
		net.zero_clear();
		dataman->start_new_epoch();
		// forward loads input, backward loads target,
		// but backward isn't called here, so we manually load_target
		net.load_target();
	};

	vector<Tensor> analyticGrads;
	for (ParamContainer::Ptr param : net.paramContainers)
	{
		auto gradients = param->paramGradients;
		for (auto gradPtr : gradients)
			analyticGrads.push_back(*gradPtr);
	}
	engine->flush_execute();
//	for (auto& tensor : analyticGrads)
//		DEBUG_MSG("gradient check: " << engine->read_memory(tensor));

	/****** perturb parameters matrices stored in connections ******/
	int agpt = 0; // point to analyticGrads
	for (ParamContainer::Ptr param : net.paramContainers)
	{
		for (int p = 0; p < param->size(); ++p)
		{
			reset_net();
			param->gradient_check_perturb(p, -perturb);
			engine->flush_execute();

			net.execute("forward");

			float lossMinus = engine->read_memory(net.lossLayer->total_loss());

			param->gradient_check_restore();
			reset_net();

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
	// Save the full gradient history for debugging ONLY
//	try {
//		RecurrentNetwork& _net = dynamic_cast<RecurrentNetwork&>(net);
//		_net.set_max_temporal_skip(Layer::UNLIMITED_TEMPORAL_SKIP);
//	}
//	catch (std::bad_cast& err) { }

	reset_net();
	engine->flush_execute();
	net.execute("forward");
	net.execute("backward");

	analyticGrads.clear();
	for (auto grad : net.layers[0]->inGradients)
		analyticGrads.push_back(*grad);
	engine->flush_execute();

	// perturb each input in sequence
	// FIXME sequence
	for (int inp = 0; inp < 1; ++inp)
	{
		reset_net();
		dataman->perturb_input(inp, -perturb);
		engine->flush_execute();

		net.execute("forward");

		float lossMinus = engine->read_memory(net.lossLayer->total_loss());

		dataman->restore_last_input();
		reset_net();

		dataman->perturb_input(inp, +perturb);
		engine->flush_execute();

		net.execute("forward");

		float lossPlus = engine->read_memory(net.lossLayer->total_loss());

		dataman->restore_last_input();

		float numericGrad = (lossPlus - lossMinus) / (2.0 * perturb);

		float analyticGrad = engine->read_memory(analyticGrads[inp]);

		assert_float_percent_eq(analyticGrad, numericGrad, percentTol,
				"input analytic != numeric", "input gradcheck pass");

	}
	reset_net();
}

#endif /* GRADIENT_CHECK_H_ */
